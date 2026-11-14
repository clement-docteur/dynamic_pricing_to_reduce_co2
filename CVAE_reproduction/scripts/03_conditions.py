"""Step 03 - rebuild the CVAE conditioning variables from raw data and check them
against the authors' (Bregere & Bessa 2020, IEEE Access 8:131949).

Run (from the project or the CVAE_reproduction directory):
    python -P scripts/03_conditions.py [--upstream PATH_TO_UPSTREAM_REPO]

Inputs (read only):
    PROJECT/temp UK 2013-14.csv          ERA5 t2m, hourly, 0.25 deg, Kelvin
    PROJECT/Tariffs.xlsx                 dToU schedule (Sheet1)
    UPSTREAM/data/power_consump_nmf.feather, train_nmf.feather   (authors' data)
Outputs:
    data/conditions_ours.parquet         365 rows, authors' schema + tariff versions
    outputs/conditions_checks.json       every check and number quoted in the report
    outputs/figures/temperature_check.png

Slot-offset convention for temperatures: our slot i (start 1 Jan 00:00 UTC + i*30 min)
takes ERA5 linearly interpolated at t_i + offset*30 min (positive = later ERA5 time).
Slot-offset convention for tariffs: see lib/tariffs.py (result[g] = xlsx[g - offset]).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPRO = Path(__file__).resolve().parents[1]
PROJECT = REPRO.parent
sys.path.insert(0, str(REPRO))

from lib import conditions as C  # noqa: E402
from lib import tariffs as TF  # noqa: E402

DEFAULT_UPSTREAM = REPRO / "data" / "authors"  # copy of the authors' repo data/*.feather (byte-identical; read as data only)
ERA5_CSV = PROJECT / "temp UK 2013-14.csv"
TARIFF_XLSX = PROJECT / "Tariffs.xlsx"
OUT_PARQUET = REPRO / "data" / "conditions_ours.parquet"
OUT_JSON = REPRO / "outputs" / "conditions_checks.json"
OUT_FIG = REPRO / "outputs" / "figures" / "temperature_check.png"
OFFSETS = list(range(-2, 3))

# Candidate London series.  The first five are the pre-specified candidates (the
# selection is made among them); the others are extra sensitivity rows only.
PRESCRIBED = [
    "central_nearest",
    "central_bilinear",
    "heathrow_nearest",
    "heathrow_bilinear",
    "london_box2x2",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


def n_bit_exact(a, b) -> int:
    return int(np.sum(np.asarray(a) == np.asarray(b)))


# ============================================================================ step 1
def temperature_candidates(times, lats, lons, cube):
    cands, info = {}, {}
    for name, (lat, lon), method in [
        ("central_nearest", C.LONDON_CENTRAL, "nearest"),
        ("central_bilinear", C.LONDON_CENTRAL, "bilinear"),
        ("heathrow_nearest", C.HEATHROW, "nearest"),
        ("heathrow_bilinear", C.HEATHROW, "bilinear"),
        ("london_box2x2", C.LONDON_CENTRAL, "box2x2"),
    ]:
        v, i = C.point_series(lats, lons, cube, lat, lon, method)
        cands[name], info[name] = v, dict(point=[lat, lon], **i)
    for name, lb, ob in [
        ("extra_box2x2_south", (51.25, 51.5), (-0.25, 0.0)),
        ("extra_box3x3", (51.25, 51.75), (-0.5, 0.0)),
        ("extra_grid_51.5N_0.0E", (51.5, 51.5), (0.0, 0.0)),
    ]:
        v, i = C.box_mean(lats, lons, cube, lb, ob)
        cands[name], info[name] = v, i
    return cands, info


def step_temperature(A: pd.DataFrame):
    log("step 1: ERA5 temperature candidates")
    T_auth = A[C.TEMP_COLS].to_numpy(float)
    era = C.load_era5_box(str(ERA5_CSV))
    times, lats, lons, cube = C.era5_cube(era)
    log(f"  ERA5 box: {cube.shape} (time, lat, lon); {times[0]} .. {times[-1]}")
    cands, info = temperature_candidates(times, lats, lons, cube)

    # --- structure of the authors' temperatures
    flat = T_auth.ravel()
    even, odd = flat[0::2], flat[1::2]
    mid = 0.5 * (flat[0:-2:2] + flat[2::2])
    on_grid = np.isclose(np.round(even * 10), even * 10, atol=1e-6)
    auth_struct = dict(
        odd_slot_minus_midpoint_maxabs=float(np.max(np.abs(odd[:-1] - mid))),
        last_slot_equals_previous=bool(flat[-1] == flat[-2]),
        even_slots_on_0p1_grid_share=float(on_grid.mean()),
        even_slots_off_0p1_grid=int((~on_grid).sum()),
        note="even slots = hourly values (0.1 degC precision except gap-filled hours), "
        "odd slots = exact mid-points (linear interpolation), last slot of the year "
        "clamped; off-grid even values (thirds, halves) are hours filled by linear "
        "interpolation across missing observations.",
        min=float(flat.min()),
        max=float(flat.max()),
        mean=float(flat.mean()),
    )

    rows, halfhourly = [], {}
    for name, v in cands.items():
        for off in OFFSETS:
            M, clamped = C.to_halfhourly(v, times, slot_offset=off)
            halfhourly[(name, off)] = M
            m = ~clamped
            r = C.compare_series(M, T_auth, m)
            Ma = M - M.mean(axis=1, keepdims=True)
            Aa = T_auth - T_auth.mean(axis=1, keepdims=True)
            r["corr_intraday_anomaly"] = float(np.corrcoef(Ma[m], Aa[m])[0, 1])
            r["n_clamped"] = int(clamped.sum())
            rows.append(dict(candidate=name, prescribed=name in PRESCRIBED, offset=off, **r))
    tab = pd.DataFrame(rows)
    elig = tab[tab.prescribed]
    best = elig.loc[elig.rmse.idxmin()]
    best_name, best_off = str(best["candidate"]), int(best["offset"])
    log(f"  chosen: {best_name} offset {best_off}: RMSE {best['rmse']:.3f}, bias {best['bias']:+.3f}, r {best['corr']:.4f}")
    overall = tab.loc[tab.rmse.idxmin()]

    # --- diagnostics of the chosen series
    v = cands[best_name]
    T_ours, clamped = C.to_halfhourly(v, times, slot_offset=best_off)
    dates = pd.DatetimeIndex(A["date"])
    month = np.asarray(dates.month)
    bst_day = np.asarray((dates >= "2013-03-31") & (dates <= "2013-10-26"))
    diag = {}
    for label, sel_days in [("GMT_days", ~bst_day), ("BST_days", bst_day), ("all_days", np.ones(365, bool))]:
        frac = {}
        for off in np.round(np.arange(-2.0, 2.0001, 0.05), 2):
            M, cl = C.to_halfhourly(v, times, slot_offset=off)
            m = ~cl & sel_days[:, None]
            frac[float(off)] = C.compare_series(M, T_auth, m)["rmse_centred"]
        ints = {o: frac[float(o)] for o in OFFSETS}
        diag[label] = dict(
            best_integer_offset=int(min(ints, key=ints.get)),
            rmse_centred_by_integer_offset=ints,
            best_fractional_offset=float(min(frac, key=frac.get)),
            rmse_centred_at_best_fractional=float(min(frac.values())),
        )
    per_month = {}
    for mo in range(1, 13):
        r = {}
        for off in OFFSETS:
            M, cl = C.to_halfhourly(v, times, slot_offset=off)
            r[off] = C.compare_series(M, T_auth, ~cl & (month == mo)[:, None])["rmse_centred"]
        per_month[mo] = dict(best_integer_offset=int(min(r, key=r.get)), rmse_centred=r)
    err = T_ours - T_auth
    diag.update(
        per_month_best_integer_offset={m: per_month[m]["best_integer_offset"] for m in per_month},
        per_month_rmse_centred=per_month,
        bias_by_month={mo: float(err[month == mo].mean()) for mo in range(1, 13)},
        bias_by_halfhour_utc=err.mean(axis=0).tolist(),
        bias_by_halfhour_gmt_days=err[~bst_day].mean(axis=0).tolist(),
        bias_by_halfhour_bst_days=err[bst_day].mean(axis=0).tolist(),
        daily_mean=C.compare_series(T_ours.mean(1), T_auth.mean(1)),
        daily_amplitude_mean_ours=float((T_ours.max(1) - T_ours.min(1)).mean()),
        daily_amplitude_mean_authors=float((T_auth.max(1) - T_auth.min(1)).mean()),
        clamped_slots_in_chosen=int(clamped.sum()),
        last_slot_note="the authors clamp 31 Dec 23:30 to the 23:00 value; ours interpolates "
        "with ERA5 2014-01-01 00:00 (available in the file)",
    )
    out = dict(
        era5=dict(
            file=str(ERA5_CSV),
            box_lat=[float(lats[0]), float(lats[-1])],
            box_lon=[float(lons[0]), float(lons[-1])],
            grid_step_deg=0.25,
            time_first=str(times[0]),
            time_last=str(times[-1]),
            n_hours=int(len(times)),
            units="t2m Kelvin -> degC (minus 273.15)",
            time_base="valid_time, UTC",
        ),
        authors_structure=auth_struct,
        candidates=info,
        offset_convention="our slot i = ERA5 interpolated at t_i + offset*30min (t_i in UTC)",
        selection_rule="minimum RMSE among the 5 prescribed candidates x integer offsets -2..+2; "
        "'extra_*' rows are sensitivity only",
        table=tab.to_dict(orient="records"),
        chosen=dict(candidate=best_name, offset=best_off, **{k: best[k] for k in ("rmse", "bias", "rmse_centred", "mae", "corr", "corr_intraday_anomaly")}),
        best_overall_including_extras=dict(
            candidate=str(overall["candidate"]),
            offset=int(overall["offset"]),
            rmse=float(overall["rmse"]),
            bias=float(overall["bias"]),
            corr=float(overall["corr"]),
        ),
        chosen_diagnostics=diag,
    )
    return out, T_ours, tab, halfhourly, (times, cands)


# ============================================================================ step 2
def step_smoothing_pca(A: pd.DataFrame, T_ours: np.ndarray, train_dates, halfhourly: dict):
    log("step 2: smoothing, normalisation and PCA")
    T = A[C.TEMP_COLS].to_numpy(float)
    ts_a = A["temp_smooth"].to_numpy(float)
    tsn_a = A["temp_smooth_n"].to_numpy(float)
    P = A[C.PCA_COLS].to_numpy(float)

    # --- smoothing formula on the authors' own temperatures
    ts = C.smoothed_daily(T)
    sm = dict(
        formula="s_1 = tau_1 (2013-01-01 00:00 slot), s_k = (1-a) tau_k + a s_{k-1}, a = 0.998; daily = mean of 48",
        bit_exact_days=n_bit_exact(ts, ts_a),
        maxabs=float(np.max(np.abs(ts - ts_a))),
        daily_mean_with_numpy_mean_maxabs=float(np.max(np.abs(C.smooth_series(T.ravel()).reshape(365, 48).mean(1) - ts_a))),
        variants={},
    )
    for name, kw in [
        ("init_s0=mean(day1)", dict(init=float(T[0].mean()))),
        ("init_s0=mean(year)", dict(init=float(T.mean()))),
        ("init_s0=0", dict(init=0.0)),
        ("a=0.99", dict(a=0.99)),
        ("a=0.995", dict(a=0.995)),
        ("a=0.997", dict(a=0.997)),
        ("a=0.999", dict(a=0.999)),
    ]:
        d = C.smoothed_daily(T, **kw) - ts_a
        sm["variants"][name] = dict(maxabs=float(np.max(np.abs(d))), rmse=float(np.sqrt(np.mean(d**2))))
    tsn = C.minmax(ts)
    norm = dict(
        recipe="temp_smooth_n = (temp_smooth - min) / (max - min) over the 365 days",
        bit_exact_days=n_bit_exact(tsn, tsn_a),
        maxabs=float(np.max(np.abs(tsn - tsn_a))),
        temp_smooth_min=float(ts_a.min()),
        temp_smooth_max=float(ts_a.max()),
    )

    # --- PCA recipe search on the authors' temperatures
    train_mask = A["date"].isin(pd.DatetimeIndex(train_dates)).to_numpy()
    variants = []
    mats = {"T48": T, "T48+temp_smooth": np.c_[T, ts_a]}
    for (mname, X), std, fit_on in itertools.product(mats.items(), [False, True], ["all", "train"]):
        f = C.pca_fit(X, 3, standardize=std, fit_rows=None if fit_on == "all" else np.flatnonzero(train_mask))
        S = f["scores"]
        best = {}
        for scl_name, scl in [("global", C.scale_global), ("per_component", lambda s: C.minmax(s, axis=0))]:
            best[scl_name] = float(min(np.max(np.abs(scl(S * np.array(sg)) - P)) for sg in itertools.product([1, -1], repeat=3)))
        variants.append(
            dict(
                matrix=mname,
                standardized=std,
                fitted_on=fit_on,
                evr3=float(f["evr"].sum()),
                evr=f["evr"].tolist(),
                r2_per_component=C.r2_per_component(P, S),
                r2_rotation_free=C.r2_rotation_free(P, S),
                maxabs_best_signs_global_scaling=best["global"],
                maxabs_best_signs_per_component_scaling=best["per_component"],
            )
        )
    rec = C.temperature_conditions(T)
    pca_auth_fit = rec["fit"]
    sign_check = {}
    for sg in itertools.product([1, -1], repeat=3):
        sign_check[str(sg)] = float(np.max(np.abs(C.scale_global(pca_auth_fit["scores"] * np.array(sg)) - P)))
    U = pca_auth_fit["scores"]
    pca_auth = dict(
        recipe="PCA of the 365x48 half-hourly temperatures only (centred, not standardised, "
        "fitted on all 365 days, scikit-learn<1.5 u-based sign convention), then ONE global "
        "min-max over the 365x3 score matrix",
        maxabs=float(np.max(np.abs(rec["pca"] - P))),
        maxabs_by_sign_choice=sign_check,
        authors_ranges=dict(min=P.min(0).tolist(), max=P.max(0).tolist()),
        global_extremes=dict(
            argmin_day=str(A["date"].iloc[int(np.unravel_index(U.argmin(), U.shape)[0])].date()),
            argmax_day=str(A["date"].iloc[int(np.unravel_index(U.argmax(), U.shape)[0])].date()),
            component_of_extremes=[int(np.unravel_index(U.argmin(), U.shape)[1]), int(np.unravel_index(U.argmax(), U.shape)[1])],
        ),
        evr_authors_temps_T48=pca_auth_fit["evr_all"][:6].tolist(),
        evr3_authors_temps_T48=float(pca_auth_fit["evr"].sum()),
        evr3_authors_temps_T48_plus_smooth=float(C.pca_fit(np.c_[T, ts_a], 3)["evr"].sum()),
        paper_claim="3 components explain 98% of the variance (PCA on 48 temps + smoothed temp)",
        variants=variants,
    )

    # --- our versions from the chosen ERA5 series
    ours = C.temperature_conditions(T_ours, ref_components=pca_auth_fit["components"])
    ours_cmp = dict(
        temp_smooth=C.compare_series(ours["temp_smooth"], ts_a),
        temp_smooth_n=C.compare_series(ours["temp_smooth_n"], tsn_a),
        pca_temp=[C.compare_series(ours["pca"][:, k], P[:, k]) for k in range(3)],
        pca_r2_per_component=C.r2_per_component(P, ours["fit"]["scores"]),
        pca_r2_rotation_free=C.r2_rotation_free(P, ours["fit"]["scores"]),
        loading_cosine_with_authors=[
            float(np.dot(ours["fit"]["components"][k], pca_auth_fit["components"][k])) for k in range(3)
        ],
        signs_flipped_vs_u_based_convention=ours["fit"]["flipped"],
        evr_ours_T48=ours["fit"]["evr_all"][:6].tolist(),
        evr3_ours_T48=float(ours["fit"]["evr"].sum()),
        evr3_ours_T48_plus_smooth=float(C.pca_fit(np.c_[T_ours, ours["temp_smooth"]], 3)["evr"].sum()),
        evr3_ours_T48_standardized=float(C.pca_fit(T_ours, 3, standardize=True)["evr"].sum()),
        pca_ranges=dict(min=ours["pca"].min(0).tolist(), max=ours["pca"].max(0).tolist()),
        note="our PCA is fitted on our own 365x48 ERA5 matrix (authors' recipe); component "
        "signs are aligned to the authors' loadings (positive dot product) before the global "
        "min-max, so that the scaled values are comparable",
    )

    # --- sensitivity of the derived conditions to the temperature candidate / offset
    sens = []
    for (name, off), M in sorted(halfhourly.items()):
        if off not in (-1, 0, 1):
            continue
        r = C.temperature_conditions(M, ref_components=pca_auth_fit["components"])
        sens.append(
            dict(
                candidate=name,
                offset=off,
                temp_smooth_n_rmse=C.compare_series(r["temp_smooth_n"], tsn_a)["rmse"],
                temp_smooth_n_corr=C.compare_series(r["temp_smooth_n"], tsn_a)["corr"],
                pca_corr=[C.compare_series(r["pca"][:, k], P[:, k])["corr"] for k in range(3)],
                pca_rmse=[C.compare_series(r["pca"][:, k], P[:, k])["rmse"] for k in range(3)],
                evr3=float(r["fit"]["evr"].sum()),
            )
        )
    ours_cmp["candidate_sensitivity"] = sens
    return dict(smoothing=sm, normalization=norm, pca_authors=pca_auth, ours_vs_authors=ours_cmp), ours, pca_auth_fit


# ============================================================================ step 3
def step_calendar(A: pd.DataFrame):
    log("step 3: calendar variables")
    pos_a = A["pos_y"].to_numpy(float)
    wd_a = A["work_day"].to_numpy(float)
    pos, wd = C.calendar_vars(A["date"])
    variants = {}
    for name, x in [
        ("i/17520 (spec wording)", np.arange(17520) / 17520),
        ("i/17519 = linspace(0,1,17520)", np.arange(17520) / 17519),
        ("(i+1)/17520", (np.arange(17520) + 1) / 17520),
    ]:
        for agg_name, agg in [("pandas row mean", C.daily_mean), ("numpy mean", lambda M: M.mean(axis=1))]:
            v = agg(x.reshape(365, 48))
            variants[f"{name} | {agg_name}"] = dict(bit_exact_days=n_bit_exact(v, pos_a), maxabs=float(np.max(np.abs(v - pos_a))))
    for name, v in [("day_index/364", np.arange(365) / 364), ("day_index/365", np.arange(365) / 365)]:
        variants[name] = dict(bit_exact_days=n_bit_exact(v, pos_a), maxabs=float(np.max(np.abs(v - pos_a))))
    bank_holidays = pd.to_datetime(
        ["2013-01-01", "2013-03-29", "2013-04-01", "2013-05-06", "2013-05-27", "2013-08-26", "2013-12-25", "2013-12-26"]
    )
    wd_bh = A.set_index("date").loc[bank_holidays, "work_day"].to_numpy()
    return dict(
        pos_y=dict(
            recipe="daily pandas row mean of i/17519, i = 0..17519 (0 at 1 Jan 00:00, 1 at 31 Dec 23:30)",
            bit_exact_days=n_bit_exact(pos, pos_a),
            maxabs=float(np.max(np.abs(pos - pos_a))),
            first=float(pos_a[0]),
            last=float(pos_a[-1]),
            variants=variants,
        ),
        work_day=dict(
            recipe="1 Monday-Friday, 0 Saturday-Sunday, no bank-holiday adjustment",
            bit_exact_days=n_bit_exact(wd, wd_a),
            n_work_days=int(wd_a.sum()),
            england_bank_holidays_2013=[str(d.date()) for d in bank_holidays],
            work_day_on_bank_holidays=wd_bh.tolist(),
        ),
    ), pos, wd


# ============================================================================ step 4
def step_tariffs(feather: Path):
    log("step 4: tariff dummies")
    tab = TF.alignment_table(str(TARIFF_XLSX), str(feather), OFFSETS)
    dates, la, ha, pa = TF.authors_dummies(str(feather))
    # version choices (made by agreement with the authors only; NOT a physical judgement)
    g = tab[(tab["mode"] == "gmt") & (tab.period == "GMT_period")].set_index("offset")["event_jaccard"]
    l_ = tab[(tab["mode"] == "local") & (tab.period == "year")].set_index("offset")["event_jaccard"]
    off_gmt, off_loc = int(g.idxmax()), int(l_.idxmax())
    log(f"  gmt offset best in the GMT period: {off_gmt:+d} (Jaccard {g.max():.3f}); "
        f"local offset best over the year: {off_loc:+d} (Jaccard {l_.max():.3f})")
    versions = {"auth": (la, ha, pa)}
    _, lg, hg, pg = TF.tariff_dummies(str(TARIFF_XLSX), "gmt", off_gmt)
    _, ll, hl, pl = TF.tariff_dummies(str(TARIFF_XLSX), "local", off_loc)
    versions["gmt"] = (lg, hg, pg)
    versions["loc"] = (ll, hl, pl)
    _, l0, h0, _ = TF.tariff_dummies(str(TARIFF_XLSX), "gmt", 0)

    months = [f"{m:02d}" for m in range(1, 13)]
    piv = {}
    for metric in ("event_jaccard", "slot_agreement", "event_days_identical", "event_days_union"):
        p = tab.pivot_table(index=["mode", "offset"], columns="period", values=metric)
        p = p[months + ["GMT_Jan1-Mar30", "GMT_period", "BST_period", "Jan-Oct", "year"]]
        piv[metric] = {f"{m}{o:+d}": row.to_dict() for (m, o), row in p.iterrows()}
    dd_loc = TF.differing_days(la, ha, ll, hl, dates)
    dd_gmt_jan_mar = TF.differing_days(la, ha, lg, hg, dates)
    dd_gmt_jan_mar = dd_gmt_jan_mar[pd.to_datetime(dd_gmt_jan_mar["date"]) < "2013-03-31"] if len(dd_gmt_jan_mar) else dd_gmt_jan_mar
    A_lab = TF.dummies_to_labels(la, ha).reshape(365, 48)
    copies = {}
    for ds in ["2013-03-29", "2013-05-03"]:
        d = int((pd.Timestamp(ds) - dates[0]).days)
        copies[ds] = dict(
            authors_equals_authors_previous_day=bool(np.array_equal(A_lab[d], A_lab[d - 1])),
            ours_loc_is_all_low=bool((ll[d] == 1).all()),
        )

    def counts(lo, hi):
        ev = (lo + hi).sum(axis=1) > 0
        return dict(n_low_slots=int(lo.sum()), n_high_slots=int(hi.sum()), n_event_days=int(ev.sum()))

    out = dict(
        offset_convention="result[g] = mapped_xlsx[g - offset]: positive offset = events later; "
        "mode 'gmt' = xlsx timestamps taken as GMT; 'local' = xlsx timestamps are Europe/London "
        "clock time converted to GMT",
        metric_definitions=dict(
            event_jaccard="Jaccard index of the (slot, label) event pairs, Low and High pooled",
            slot_agreement="share of half-hour slots with the same label (Normal included)",
            event_days_identical="days with an event in either set whose 48-slot profile is identical",
        ),
        counts=dict(
            authors=counts(la, ha),
            xlsx=counts(l0, h0),
            ours_gmt=counts(lg, hg),
            ours_loc=counts(ll, hl),
        ),
        by_mode_offset_period=piv,
        chosen_versions=dict(
            gmt=dict(mode="gmt", offset=off_gmt, rule="best event_jaccard vs authors over the GMT period (Jan 1-Mar 30, Oct 27-Dec 31)"),
            loc=dict(mode="local", offset=off_loc, rule="best event_jaccard vs authors over the whole year"),
            auth=dict(source="authors' power_consump_nmf.feather low/high/price columns"),
            warning="chosen by agreement with the authors' file only; which alignment is physically "
            "right is decided elsewhere (consumption event study)",
        ),
        differing_days_loc_vs_authors=dd_loc.to_dict(orient="records"),
        differing_days_gmt_vs_authors_before_BST=dd_gmt_jan_mar.to_dict(orient="records"),
        authors_copy_of_previous_day=copies,
        price_values_gbp_per_kwh=TF.PRICES_GBP,
    )
    return out, dates, versions, tab


# ============================================================================ figure
def make_figure(A, T_ours, temp_out, tab, ours, pca_auth_fit, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    seq = LinearSegmentedColormap.from_list(
        "blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    )
    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 9,
            "axes.edgecolor": BASE,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.labelcolor": INK2,
            "figure.facecolor": SURF,
            "axes.facecolor": SURF,
            "savefig.facecolor": SURF,
        }
    )
    T_auth = A[C.TEMP_COLS].to_numpy(float)
    dates = pd.DatetimeIndex(A["date"])
    ch = temp_out["chosen"]
    lab_ours = f"ERA5 ({ch['candidate'].replace('_', ' ')}, offset {ch['offset']:+d})"
    lab_auth = "Authors (NOAA obs.)"

    fig, axs = plt.subplots(2, 4, figsize=(19, 8.6))
    fig.suptitle(
        "London temperature conditions: authors' NOAA-based values vs our ERA5 rebuild (2013)",
        x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK,
    )

    # (a) daily mean
    ax = axs[0, 0]
    ax.plot(dates, T_auth.mean(1), color=BLUE, lw=1.3, label=lab_auth)
    ax.plot(dates, T_ours.mean(1), color=ORANGE, lw=1.3, label=lab_ours)
    dm = temp_out["chosen_diagnostics"]["daily_mean"]
    ax.set_title(f"a  Daily mean (r = {dm['corr']:.4f}, bias {dm['bias']:+.2f} \u00b0C)", loc="left")
    ax.set_ylabel("\u00b0C")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(loc="upper left", fontsize=8)

    # (b) zoom on a BST week, half-hourly
    ax = axs[0, 1]
    d0 = int((pd.Timestamp("2013-07-15") - dates[0]).days)
    tt = pd.date_range("2013-07-15", periods=7 * 48, freq="30min")
    ax.plot(tt, T_auth[d0 : d0 + 7].ravel(), color=BLUE, lw=1.3, label=lab_auth)
    ax.plot(tt, T_ours[d0 : d0 + 7].ravel(), color=ORANGE, lw=1.3, label=lab_ours)
    ax.set_title("b  Half-hourly, 15-21 Jul 2013 (BST period; UTC axis)", loc="left")
    ax.set_ylabel("\u00b0C")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    wk = np.r_[T_auth[d0 : d0 + 7].ravel(), T_ours[d0 : d0 + 7].ravel()]
    ax.set_ylim(wk.min() - 0.5, wk.max() + 0.3 * (wk.max() - wk.min()))
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    # (c) scatter
    ax = axs[0, 2]
    ax.scatter(T_auth.ravel(), T_ours.ravel(), s=2, color=ORANGE, alpha=0.12, linewidths=0, rasterized=True)
    lim = [min(T_auth.min(), T_ours.min()) - 1, max(T_auth.max(), T_ours.max()) + 1]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("authors, \u00b0C")
    ax.set_ylabel("ERA5 (ours), \u00b0C")
    ax.set_title(f"c  All 17,520 half-hours: RMSE {ch['rmse']:.2f}, bias {ch['bias']:+.2f}, r {ch['corr']:.4f}", loc="left")
    ax.text(lim[0] + 1, lim[1] - 2.5, "dashed: 1:1", color=MUTED, fontsize=8)

    # (d) RMSE heatmap
    ax = axs[0, 3]
    order = PRESCRIBED + [c for c in tab.candidate.unique() if c not in PRESCRIBED]
    H = tab.pivot(index="candidate", columns="offset", values="rmse").loc[order, OFFSETS]
    im = ax.imshow(H.to_numpy(), cmap=seq, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(OFFSETS)), [f"{o:+d}" for o in OFFSETS])
    ax.set_yticks(range(len(order)), [c.replace("extra_", "* ").replace("_", " ") for c in order], fontsize=8)
    ax.set_xlabel("slot offset (half-hours; + = later ERA5 time)")
    vmid = np.nanmean(H.to_numpy())
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            val = H.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7.5, color="#ffffff" if val > vmid else INK)
    bi = order.index(ch["candidate"])
    bj = OFFSETS.index(ch["offset"])
    ax.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec=INK, lw=1.8))
    ax.axhline(len(PRESCRIBED) - 0.5, color=SURF, lw=3)
    ax.set_title("d  RMSE vs authors (\u00b0C); box = chosen, * = extra", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)

    # (e) diurnal error, GMT vs BST
    ax = axs[1, 0]
    dg = temp_out["chosen_diagnostics"]
    hrs = np.arange(48) / 2
    ax.axhline(0, color=BASE, lw=1)
    ax.plot(hrs, dg["bias_by_halfhour_gmt_days"], color=BLUE, lw=1.5, label="GMT-period days")
    ax.plot(hrs, dg["bias_by_halfhour_bst_days"], color=ORANGE, lw=1.5, label="BST-period days")
    ax.set_xlim(0, 23.5)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("time of day (UTC, slot start)")
    ax.set_ylabel("ERA5 \u2212 authors, \u00b0C")
    ax.set_title(
        f"e  Mean error by half-hour (best shift GMT {dg['GMT_days']['best_fractional_offset']:+.2f}, "
        f"BST {dg['BST_days']['best_fractional_offset']:+.2f} slot)",
        loc="left",
    )
    ax.text(
        23.3, -0.06, "(a 1-h DST shift in the authors'\ntemps would give about -2 in BST)",
        color=INK2, fontsize=8, ha="right", va="top",
    )
    ax.legend(loc="upper left", fontsize=8)

    # (f) temp_smooth_n
    ax = axs[1, 1]
    tsn = A["temp_smooth_n"].to_numpy(float)
    ax.plot(dates, tsn, color=BLUE, lw=1.5, label="authors")
    ax.plot(dates, ours["temp_smooth_n"], color=ORANGE, lw=1.5, label="ours (ERA5)")
    r = C.compare_series(ours["temp_smooth_n"], tsn)
    ax.set_title(f"f  temp_smooth_n (model input): RMSE {r['rmse']:.3f}, r {r['corr']:.4f}", loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="upper left", fontsize=8)

    # (g) PCA components
    ax = axs[1, 2]
    P = A[C.PCA_COLS].to_numpy(float)
    for k, (col, mk) in enumerate(zip([BLUE, ORANGE, AQUA], ["o", "s", "^"])):
        rr = np.corrcoef(P[:, k], ours["pca"][:, k])[0, 1]
        ax.scatter(P[:, k], ours["pca"][:, k], s=12, marker=mk, color=col, alpha=0.75, linewidths=0, label=f"pca_temp_{k} (r = {rr:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("authors (scaled)")
    ax.set_ylabel("ours (scaled)")
    ax.set_title("g  Temperature PCA scores, global min-max scaling", loc="left")
    ax.legend(loc="upper left", fontsize=8, markerscale=1.2)

    # (h) cumulative explained variance
    ax = axs[1, 3]
    k = np.arange(1, 7)
    ca = np.cumsum(pca_auth_fit["evr_all"][:6])
    co = np.cumsum(ours["fit"]["evr_all"][:6])
    ax.axhline(0.98, color=MUTED, lw=1, ls="--")
    ax.text(6, 0.9815, "paper: 98% with 3 components", color=INK2, fontsize=8, ha="right", va="bottom")
    ax.plot(k, ca, color=BLUE, lw=1.5, marker="o", ms=5, label=f"authors' temps (3 comp.: {100 * ca[2]:.2f}%)")
    ax.plot(k, co, color=ORANGE, lw=1.5, marker="s", ms=5, label=f"ERA5 temps (3 comp.: {100 * co[2]:.2f}%)")
    ax.set_xticks(k)
    ax.set_ylim(0.9, 1.003)
    ax.set_xlabel("number of components (PCA on the 48 half-hourly temperatures)")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("h  Explained variance of the temperature PCA", loc="left")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upstream", default=str(DEFAULT_UPSTREAM), help="folder holding the authors' data/*.feather files (read as data only)")
    args = ap.parse_args(argv)
    up = Path(args.upstream)
    feather = up / "power_consump_nmf.feather"
    t0 = time.time()

    A = C.authors_conditions(str(feather))
    assert len(A) == 365 and A["date"].is_monotonic_increasing
    train_dates = pd.to_datetime(pd.read_feather(up / "train_nmf.feather")["date"]).unique()

    temp_out, T_ours, tab, halfhourly, _ = step_temperature(A)
    sp_out, ours, pca_auth_fit = step_smoothing_pca(A, T_ours, train_dates, halfhourly)
    cal_out, pos_y, work_day = step_calendar(A)
    tar_out, tdates, versions, _ = step_tariffs(feather)
    assert np.array_equal(tdates.to_numpy(), pd.DatetimeIndex(A["date"]).to_numpy())

    # ------------------------------------------------------------------ parquet
    log("step 5: writing outputs")
    data = {"date": pd.DatetimeIndex(A["date"]).astype("datetime64[ns]")}
    for h, c in enumerate(C.TEMP_COLS):
        data[c] = T_ours[:, h]
    data["temp_smooth"] = ours["temp_smooth"]
    data["temp_smooth_n"] = ours["temp_smooth_n"]
    for k, c in enumerate(C.PCA_COLS):
        data[c] = ours["pca"][:, k]
    data["pos_y"] = pos_y
    data["work_day"] = work_day
    for v in TF.VERSIONS:
        lo, hi, pr = versions[v]
        for h in range(48):
            data[f"low_{v}_{h:02d}"] = lo[:, h]
        for h in range(48):
            data[f"high_{v}_{h:02d}"] = hi[:, h]
        for h in range(48):
            data[f"price_{v}_{h:02d}"] = pr[:, h]
    df = pd.DataFrame(data)
    assert df.shape == (365, 1 + 48 + 2 + 3 + 2 + 3 * 3 * 48), df.shape
    assert np.isfinite(df.drop(columns="date").to_numpy(float)).all()
    df.attrs = {
        "temperature": f"ERA5 {temp_out['chosen']['candidate']} offset {temp_out['chosen']['offset']:+d}",
        "tariff_gmt": f"mode=gmt offset={tar_out['chosen_versions']['gmt']['offset']:+d}",
        "tariff_loc": f"mode=local offset={tar_out['chosen_versions']['loc']['offset']:+d}",
        "tariff_auth": "authors' dummies",
    }
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    back = pd.read_parquet(OUT_PARQUET)
    assert back.shape == df.shape and np.allclose(back.drop(columns="date").to_numpy(float), df.drop(columns="date").to_numpy(float))

    # ------------------------------------------------------------------ json
    checks = dict(
        script="scripts/03_conditions.py",
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        inputs=dict(era5=str(ERA5_CSV), tariffs=str(TARIFF_XLSX), authors=str(feather)),
        authors_conditions_identical_across_clusters=True,
        temperature=temp_out,
        smoothing_normalization_pca=sp_out,
        calendar=cal_out,
        tariffs=tar_out,
        parquet=dict(
            path=str(OUT_PARQUET),
            shape=list(df.shape),
            columns_note="date; temp_00..47, temp_smooth, temp_smooth_n, pca_temp_0..2, pos_y, work_day "
            "(ours, authors' recipes); low_/high_/price_{auth,gmt,loc}_00..47. Unsuffixed low_/high_/"
            "price_ columns are intentionally absent: pick a version with lib.tariffs.select_tariff_version.",
            attrs=df.attrs,
        ),
        runtime_s=round(time.time() - t0, 1),
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(jsonable(checks), indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ figure
    make_figure(A, T_ours, temp_out, tab, ours, pca_auth_fit, OUT_FIG)

    # ------------------------------------------------------------------ summary
    s = sp_out
    print("\n================ SUMMARY ================")
    print(f"temperature: chosen {temp_out['chosen']['candidate']} offset {temp_out['chosen']['offset']:+d}: "
          f"RMSE {temp_out['chosen']['rmse']:.3f} bias {temp_out['chosen']['bias']:+.3f} r {temp_out['chosen']['corr']:.4f}")
    print(f"  best integer offset GMT days {temp_out['chosen_diagnostics']['GMT_days']['best_integer_offset']:+d}, "
          f"BST days {temp_out['chosen_diagnostics']['BST_days']['best_integer_offset']:+d}; fractional (all) "
          f"{temp_out['chosen_diagnostics']['all_days']['best_fractional_offset']:+.2f}")
    print(f"smoothing on authors' temps: bit-exact days {s['smoothing']['bit_exact_days']}/365, maxabs {s['smoothing']['maxabs']:.2e}")
    print(f"temp_smooth_n: bit-exact days {s['normalization']['bit_exact_days']}/365")
    print(f"PCA recipe maxabs {s['pca_authors']['maxabs']:.2e}; EVR3 authors' temps {s['pca_authors']['evr3_authors_temps_T48']:.4f} "
          f"(with smooth {s['pca_authors']['evr3_authors_temps_T48_plus_smooth']:.4f}); ours {s['ours_vs_authors']['evr3_ours_T48']:.4f} "
          f"(with smooth {s['ours_vs_authors']['evr3_ours_T48_plus_smooth']:.4f})")
    o = s["ours_vs_authors"]
    print(f"ours vs authors: temp_smooth RMSE {o['temp_smooth']['rmse']:.3f} bias {o['temp_smooth']['bias']:+.3f}; temp_smooth_n RMSE "
          f"{o['temp_smooth_n']['rmse']:.4f} r {o['temp_smooth_n']['corr']:.4f}; pca r = "
          + ", ".join(f"{x['corr']:.3f}" for x in o["pca_temp"]))
    print(f"calendar: pos_y bit-exact {cal_out['pos_y']['bit_exact_days']}/365, work_day bit-exact {cal_out['work_day']['bit_exact_days']}/365")
    cv = tar_out["chosen_versions"]
    print(f"tariffs: gmt version offset {cv['gmt']['offset']:+d}, loc version offset {cv['loc']['offset']:+d}; "
          f"days differing (loc vs authors): {len(tar_out['differing_days_loc_vs_authors'])}")
    print(f"wrote {OUT_PARQUET}\n      {OUT_JSON}\n      {OUT_FIG}\nruntime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
