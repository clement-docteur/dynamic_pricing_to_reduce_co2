"""Data / alignment checks on the rebuilt 2013 LCL data (data task, steps 3-5).

  A. household counts (paper: 1,007 ToU households with > 95 % of 2013 data); which of our
     ToU households form the authors' 1,007 (exact NNLS identification), and tests of
     simple explanations for the other 25
  B. slot mapping: does the authors' ``consumption_hh`` of date d equal our all-ToU mean at
     LCL timestamp d + (hh + s) * 30 min?  shifts s = -2..+2, for our 1,032 and for the
     identified 1,007; reverse-engineering of the authors' gap-filling rule
  C. tariff alignment event study on d(t) = mean_ToU(t) - mean_Std(t) (filled data, in the
     authors' frame): candidates A (authors' dummies), B (Tariffs.xlsx read as GMT),
     C (Tariffs.xlsx read as Europe/London clock time, converted to GMT); offsets -3..+3;
     GMT months vs BST months; START / END of every High and Low block
  D. side effects / rebound under the right alignment; schedule content A vs C
     (Nov-Dec and the two Jan-Oct days that differ), scored against the consumption response

Inputs : data/lcl2013_* and data/selection.parquet (scripts/01_build_lcl2013.py),
         the authors' feather files (read as data only), Tariffs.xlsx, Small LCL Data/*.csv
Outputs: outputs/data_checks.json, outputs/figures/event_study_high.png, event_study_low.png
         (cache: data/lcl2013_smallfile_spans.parquet)

Run:  python -P scripts/02_lcl_checks.py          (about 1-2 min)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import lcl_data as L  # noqa: E402

N, H, ND = L.N_SLOTS, L.H, L.N_DAYS
LAGS = np.arange(-6, 7)
OFFSETS = np.arange(-3, 4)
SHIFTS = np.arange(-2, 3)
KIND_CODE = {"Low": 1, "High": 2}
CODE_NAME = {0: "Normal", 1: "Low", 2: "High"}
KIND_DIR = {"Low": +1, "High": -1}             # expected sign of the ToU-Std response inside the block
EDGE_SIGN = {"start": +1, "end": -1}           # step at START has the block's sign, at END the opposite
SEASONS = ("GMT", "BST")
CANDS = ("A", "B", "C")
CAND_LABEL = {"A": "A: authors' low_/high_ dummies",
              "B": "B: Tariffs.xlsx read as GMT",
              "C": "C: Tariffs.xlsx read as London clock time"}
OUT_JSON = L.OUT / "data_checks.json"
SPANS_CACHE = L.DATA / "lcl2013_smallfile_spans.parquet"
EXACT_TOL = 1e-5                               # kWh; float32 storage error is ~1e-7


def slot_time(h: int) -> str:
    return f"{(h * 30) // 60:02d}:{(h * 30) % 60:02d}"


# =========================================================================== A + B
def compare(ours, auth):
    m = np.isfinite(ours) & np.isfinite(auth)
    d = ours[m] - auth[m]
    return {"n_values": int(m.sum()), "corr": float(np.corrcoef(ours[m], auth[m])[0, 1]),
            "mean_abs_diff_kwh": float(np.abs(d).mean()), "max_abs_diff_kwh": float(np.abs(d).max()),
            "rmse_kwh": float(np.sqrt((d ** 2).mean())), "mean_diff_kwh": float(d.mean()),
            "ratio_of_means": float(ours[m].mean() / auth[m].mean()),
            "share_abs_diff_lt_1e-3": float((np.abs(d) < 1e-3).mean()),
            "share_abs_diff_lt_1e-6": float((np.abs(d) < 1e-6).mean())}


def frame_mean(M, E, shift):
    """mean over households of an LCL-grid matrix, in the authors' frame -> (365, 48)."""
    return L.to_authors_frame(M.mean(axis=0, keepdims=True), E.mean(axis=0, keepdims=True), shift)[0]


def per_shift(F, E, auth, label):
    out = {}
    for s in SHIFTS:
        out[f"{int(s):+d}"] = compare(frame_mean(F, E, int(s)), auth)
        c = out[f"{int(s):+d}"]
        print(f"  [{label}] shift {int(s):+d}: corr {c['corr']:.6f}  mean|diff| {c['mean_abs_diff_kwh']:.2e}  "
              f"max|diff| {c['max_abs_diff_kwh']:.2e}", flush=True)
    return out


def residual_by_missing(diff, n_missing):
    out = {}
    for lo, hi in [(0, 0), (1, 1), (2, 5), (6, 20), (21, 10 ** 6)]:
        m = (n_missing >= lo) & (n_missing <= hi) & np.isfinite(diff)
        if m.any():
            out[f"{lo}-{hi}"] = {"n_values": int(m.sum()), "mean_abs_diff_kwh": float(np.abs(diff[m]).mean()),
                                 "max_abs_diff_kwh": float(np.abs(diff[m]).max())}
    return out


def implied_fill_analysis(Xk, auth_flat, n_auth, fills: dict):
    """Where exactly one of the authors' households is missing, their fill value is implied:
    n_auth * auth(t) - sum of the observed others.  Compare it with candidate rules."""
    T = Xk.shape[1]
    miss = np.isnan(Xk)
    nm = miss.sum(axis=0)
    one = np.flatnonzero(nm == 1)
    hid = miss[:, one].argmax(axis=0)
    implied = n_auth * auth_flat[one] - np.nansum(Xk[:, one], axis=0)
    idx = np.arange(T)
    valid = ~miss
    prev = np.maximum.accumulate(np.where(valid, idx, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(valid, idx, T)[:, ::-1], axis=1)[:, ::-1]
    p = prev[hid, one]
    q = nxt[hid, one]
    gap_len = q - p - 1
    x = Xk
    cand = {}
    both = (p >= 0) & (q < T)
    lin_t = np.full(len(one), np.nan)
    lin_t[both] = x[hid[both], p[both]] + (x[hid[both], q[both]] - x[hid[both], p[both]]) * \
        (one[both] - p[both]) / (q[both] - p[both])
    cand["linear_in_time"] = lin_t
    pdv = np.where(one >= 48, x[hid, np.maximum(one - 48, 0)], np.nan)
    ndv = np.where(one + 48 < T, x[hid, np.minimum(one + 48, T - 1)], np.nan)
    cand["mean_prev_next_day"] = np.where(np.isfinite(pdv) & np.isfinite(ndv), (pdv + ndv) / 2, np.nan)
    for k, F in fills.items():
        cand[k] = F[hid, one]
    df = pd.DataFrame({"gap_len": gap_len, "implied": implied, **cand})
    bins = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 48), (49, 10 ** 6)]
    table = {}
    for lo, hi in bins:
        m = (df["gap_len"] >= lo) & (df["gap_len"] <= hi)
        row = {"n_slots": int(m.sum())}
        for c in cand:
            e = (df.loc[m, "implied"] - df.loc[m, c]).abs()
            row[f"share_exact_{c}"] = float((e < EXACT_TOL).mean()) if m.any() else None
        table[f"gap_len_{lo}-{hi if hi < 10 ** 6 else 'inf'}"] = row
    overall = {c: float(((df["implied"] - df[c]).abs() < EXACT_TOL).mean()) for c in cand}
    return {"n_single_missing_slots": int(len(one)), "n_households_involved": int(len(np.unique(hid))),
            "exact_tolerance_kwh": EXACT_TOL, "share_exact_overall": overall, "by_gap_length": table}


def smallfile_spans():
    """(file, household) spans of the 168 'Small LCL Data' chunk files (cached)."""
    if SPANS_CACHE.exists():
        return pd.read_parquet(SPANS_CACHE)
    con = L.duckdb_connect(4)
    glob = (L.PROJECT / "Small LCL Data" / "LCL-June2015v2_*.csv").as_posix()
    src = (f"read_csv('{glob}', header=true, delim=',', quote='\"', auto_detect=false, filename=true, "
           "columns={'LCLid':'VARCHAR','stdorToU':'VARCHAR','DateTime':'VARCHAR','kwh':'VARCHAR'})")
    spans = con.execute(f"""SELECT filename, LCLid, any_value(stdorToU) AS grp, count(*) AS n,
                            min(DateTime) AS first_dt, max(DateTime) AS last_dt,
                            count(*) FILTER (WHERE DateTime >= '2013-01-01' AND DateTime < '2014-01-01') AS n2013
                            FROM {src} GROUP BY 1, 2""").df()
    con.close()
    spans["file"] = spans["filename"].astype(str).str.extract(r"_(\d+)\.csv$")[0].astype(int)
    spans = spans.drop(columns=["filename"])
    spans.to_parquet(SPANS_CACHE, index=False)
    return spans


def longest_run(mask):
    m = np.concatenate([[0], np.asarray(mask, np.int8), [0]])
    d = np.diff(m)
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int((e - s).max()) if len(s) else 0


def excluded_household_tests(sel_tou, X_tou, hh):
    """Can a simple data property explain the 25 households the authors did not keep?"""
    ids = sel_tou["LCLid"].to_numpy(dtype=object)
    excl = ~sel_tou["in_authors_tou_1007"].to_numpy(bool)
    feat = pd.DataFrame({
        "completeness": sel_tou["completeness"].to_numpy(),
        "mean_kwh": np.nanmean(X_tou, axis=1),
        "max_kwh": np.nanmax(X_tou, axis=1),
        "zero_share": (X_tou == 0).sum(axis=1) / (~np.isnan(X_tou)).sum(axis=1),
        "longest_zero_run": [longest_run(x == 0) for x in X_tou],
        "longest_constant_run": [longest_run(np.r_[False, np.diff(x) == 0]) for x in X_tou],
        "longest_gap": [longest_run(np.isnan(x)) for x in X_tou],
    }, index=ids)
    h = hh.set_index("LCLid").loc[ids]
    feat["first_ts_all_days_after_2011-11-23"] = (h["first_ts_all"] - pd.Timestamp("2011-11-23")).dt.days.to_numpy()
    feat["last_ts_all_days_before_2014-02-28"] = (pd.Timestamp("2014-02-28") - h["last_ts_all"]).dt.days.to_numpy()
    feat["n_rows_all"] = h["n_rows_all"].to_numpy()
    out = {}
    for c in feat.columns:
        a, b = feat.loc[~excl, c], feat.loc[excl, c]
        out[c] = {"kept_min": float(a.min()), "kept_median": float(a.median()), "kept_max": float(a.max()),
                  "excluded_min": float(b.min()), "excluded_median": float(b.median()), "excluded_max": float(b.max()),
                  "separates": bool(b.min() > a.max() or b.max() < a.min())}
    spans = smallfile_spans()
    sp = spans[spans["LCLid"].isin(set(ids))]
    nfiles = sp.groupby("LCLid")["file"].nunique().reindex(ids).fillna(1).to_numpy()
    out["split_across_chunk_files"] = {
        "excluded_split": int(((nfiles > 1) & excl).sum()), "excluded_not_split": int(((nfiles == 1) & excl).sum()),
        "kept_split": int(((nfiles > 1) & ~excl).sum()), "kept_not_split": int(((nfiles == 1) & ~excl).sum()),
        "all_households_split_across_files": int((spans.groupby("LCLid")["file"].nunique() > 1).sum())}
    return out, feat.loc[excl].reset_index().rename(columns={"index": "LCLid"})


def slot_mapping_section(F, EF, n_tou, X_tou, E_tou, sel_tou, hh, fill_stats):
    dates, auth, _ = L.load_authors_tou_aggregate()
    cons, sizes = L.load_authors_clusters()
    n_auth = int(sizes.sum())
    res: dict = {"convention": "authors' consumption_hh of date d  <->  LCL DateTime d + (hh + shift) * 30 min",
                 "authors_households": n_auth, "authors_cluster_sizes": sizes.tolist()}
    wmean = np.tensordot(sizes / sizes.sum(), cons, axes=1)
    res["authors_ToU_equals_size_weighted_cluster_mean_max_abs_diff"] = float(np.abs(wmean - auth).max())
    Ft, Et = F[:n_tou], EF[:n_tou]
    keep = sel_tou["in_authors_tou_1007"].to_numpy(bool)
    res["per_shift_our_1032"] = per_shift(Ft, Et, auth, f"our {n_tou}")
    res["per_shift_identified_1007"] = per_shift(Ft[keep], Et[keep], auth, f"identified {int(keep.sum())}")
    best = int(min(SHIFTS, key=lambda s: res["per_shift_our_1032"][f"{int(s):+d}"]["mean_abs_diff_kwh"]))
    best2 = int(min(SHIFTS, key=lambda s: res["per_shift_identified_1007"][f"{int(s):+d}"]["mean_abs_diff_kwh"]))
    assert best == best2
    res["chosen_shift"] = best
    res["meaning"] = (f"authors' consumption_hh on date d = mean LCL reading labelled d + (hh {best:+d}) * 30 min"
                      if best else "authors' consumption_hh on date d = mean LCL reading labelled d + hh * 30 min "
                                   "(same date, same clock label; LCL labels are GMT)")
    # ---- why our 1,032 do not match within 1e-3: household set, not filling
    raw = L.to_authors_frame(X_tou, E_tou, best)
    n_missing = np.isnan(raw).sum(axis=0)
    diff = frame_mean(Ft, Et, best) - auth
    res["our_1032_residual_by_number_of_missing_households"] = residual_by_missing(diff, n_missing)
    tot = L.to_authors_frame(Ft.sum(axis=0, keepdims=True), Et.sum(axis=0, keepdims=True), best)[0]
    ratio = tot / auth
    res["our_1032_sum_over_authors_value"] = {"median": float(np.nanmedian(ratio)), "p05": float(np.nanpercentile(ratio, 5)),
                                              "p95": float(np.nanpercentile(ratio, 95)),
                                              "note": "constant 1007 would mean the same households; it varies, so "
                                                      "the sets differ"}
    ident = fill_stats.get("authors_subset_identification", {})
    res["authors_subset_identification"] = ident
    tests, excl_tab = excluded_household_tests(sel_tou, X_tou, hh)
    res["excluded_25_tests"] = tests
    res["excluded_25_households"] = excl_tab.to_dict(orient="records")
    # ---- identified 1,007: residual + gap-filling rule
    keep_raw = raw[keep]
    nm_keep = np.isnan(keep_raw).sum(axis=0)
    diff_k = frame_mean(Ft[keep], Et[keep], best) - auth
    res["identified_1007_residual_by_number_of_missing_households"] = residual_by_missing(diff_k, nm_keep)
    variants = {}
    Xk = X_tou[keep]
    Ek = E_tou[keep]
    with np.errstate(invalid="ignore"):
        variants["raw_nanmean_no_fill"] = compare(np.nanmean(keep_raw, axis=0), auth)
    fills = {}
    for name, sm, rule in [("authors_rule_short1_linear_days (main)", 1, "linear_days"),
                           ("spec_rule_short4_mean_nearest", 4, "mean_nearest"),
                           ("short1_mean_nearest", 1, "mean_nearest"),
                           ("short4_linear_days", 4, "linear_days"),
                           ("all_linear_in_time", 10 ** 6, "linear_days")]:
        Fv, _ = L.fill_gaps(Xk, short_max=sm, long_rule=rule)
        Ev = L.fill_edges(Ek, Fv)
        variants[name] = compare(frame_mean(Fv.astype(np.float64), Ev.astype(np.float64), best), auth)
        fills[name] = L.to_authors_frame(Fv.astype(np.float64), Ev.astype(np.float64), best).reshape(len(Fv), -1)
        print(f"  [identified 1007] {name:40s} mean|diff| {variants[name]['mean_abs_diff_kwh']:.2e} "
              f"max|diff| {variants[name]['max_abs_diff_kwh']:.2e}", flush=True)
    res["identified_1007_aggregate_by_fill_rule"] = variants
    res["gap_fill_rule_reverse_engineering"] = implied_fill_analysis(
        keep_raw.reshape(len(Xk), -1), auth.ravel(), n_auth,
        {"nearest_days_mean (spec long rule)": fills["spec_rule_short4_mean_nearest"],
         "authors_rule (main)": fills["authors_rule_short1_linear_days (main)"],
         "spec_rule (<=4 linear, then mean nearest days)": fills["spec_rule_short4_mean_nearest"]})
    res["authors_value_definition"] = (
        f"consumption_hh = arithmetic mean (kWh per household per half-hour) over the {n_auth} ToU households; "
        "the identified subset reproduces it to ~1e-9 kWh on fully observed slots; the cluster files are the "
        "per-cluster means (size-weighted mean of the 4 clusters = all-ToU mean)")
    return res, best, dates, auth


# =========================================================================== C
def tariff_candidates():
    ts, xcodes = L.load_tariffs_xlsx()
    A, same = L.load_authors_tariff_codes(0)
    B = L.tariff_candidate_gmt(xcodes)
    C = L.tariff_candidate_local(ts, xcodes)
    return {"A": A, "B": B, "C": C}, same


def baseline_days(cands):
    """Days that are all-Normal in every candidate, including the 3 boundary slots of the
    neighbouring days (robust to the +-3 slot offsets)."""
    ev = np.zeros(N, bool)
    for c in cands.values():
        ev |= np.asarray(c) != 0
    pad = ev.copy()
    for o in range(1, 4):
        pad[o:] |= ev[:-o]
        pad[:-o] |= ev[o:]
    return ~pad.reshape(ND, H).any(axis=1)


class EventStudy:
    def __init__(self, d_flat, base_mask, dates):
        self.d = d_flat
        self.day_season = L.season_of_day(np.arange(ND))
        self.day_work = (dates.dt.dayofweek.to_numpy() < 5).astype(int)
        self.base = base_mask
        dd = d_flat.reshape(ND, H)
        # residual: d minus its same-slot median over all-Normal days of the same season
        self.median = {s: np.median(dd[base_mask & (self.day_season == s)], axis=0) for s in SEASONS}
        r = np.empty_like(dd)
        for s in SEASONS:
            m = self.day_season == s
            r[m] = dd[m] - self.median[s][None, :]
        self.r = r.ravel()
        # placebo templates: mean residual on all-Normal days by (season, work day), positions -6..53
        self.P = np.arange(-6, H + 6)
        rows = []
        for b in np.flatnonzero(base_mask):
            pos = 48 * b + self.P
            ok = (pos >= 0) & (pos < N)
            v = np.full(len(self.P), np.nan)
            v[ok] = self.r[pos[ok]]
            rows.append(v)
        self.base_rows = np.array(rows)                                    # (n_base, 60)
        self.base_idx = np.flatnonzero(base_mask)
        self.T = {}
        for s in SEASONS:
            for w in (0, 1):
                m = (self.day_season[self.base_idx] == s) & (self.day_work[self.base_idx] == w)
                self.T[(s, w)] = np.nanmean(self.base_rows[m], axis=0)

    def tmpl(self, day, q):
        T = self.T[(self.day_season[day], self.day_work[day])]
        return T[q] if 0 <= q < len(self.P) else np.nan

    def excess(self, idx, lags):
        """r[idx + lag] minus the matched placebo template (season, work day, slot of day)."""
        idx = np.asarray(idx)
        day = np.minimum(idx // H, ND - 1)
        hod = idx - 48 * day
        out = np.full((len(idx), len(lags)), np.nan)
        for j, lag in enumerate(lags):
            pos = idx + lag
            ok = (pos >= 0) & (pos < N)
            tm = np.array([self.tmpl(d_, q_) for d_, q_ in zip(day, hod + lag + 6)])
            v = np.full(len(idx), np.nan)
            v[ok] = self.r[pos[ok]]
            out[:, j] = v - tm
        return out

    def null_sd(self, day, q):
        """SD over all-Normal days (same season and work-day type) of the mean residual over the
        template positions q (minus the template): noise level of a block-mean statistic."""
        m = (self.day_season[self.base_idx] == self.day_season[day]) & \
            (self.day_work[self.base_idx] == self.day_work[day])
        T = self.T[(self.day_season[day], self.day_work[day])]
        v = np.nanmean(self.base_rows[m][:, q] - T[q], axis=1)
        return float(np.nanstd(v, ddof=1))

    def events(self, S, kind, edge, season, clean=False):
        st, en = L.runs(S, KIND_CODE[kind])
        idx = st if edge == "start" else en
        idx = idx[(idx >= 0) & (idx < N)]
        if clean and len(idx):
            keep = [bool(((S[max(i - 6, 0):i] if edge == "start" else S[i:i + 6]) == 0).all()) for i in idx]
            idx = idx[np.array(keep, bool)]
        seas = self.day_season[np.minimum(idx // H, ND - 1)]
        return idx[seas == season]


def mean_se(a, axis=0):
    n = np.sum(np.isfinite(a), axis=axis)
    m = np.nanmean(a, axis=axis)
    sd = np.nanstd(a, axis=axis, ddof=1) if a.shape[axis] > 1 else np.full_like(m, np.nan)
    return m, sd / np.sqrt(np.maximum(n, 1))


def profile_stats(es: EventStudy, idx, kind, edge):
    """Event-study profile (placebo-corrected excess, lags -6..6) + step/contrast metrics."""
    if len(idx) == 0:
        return {"n_events": 0}
    ex = es.excess(idx, LAGS)
    prof, se = mean_se(ex)
    ex2 = es.excess(idx, np.arange(-7, 7))
    steps_e = ex2[:, 1:] - ex2[:, :-1]                       # step at lag L = x(L) - x(L-1), L = -6..6
    st_m, st_se = mean_se(steps_e)
    sgn = KIND_DIR[kind] * EDGE_SIGN[edge]                   # expected sign of the step at the edge
    j0 = int(np.flatnonzero(LAGS == 0)[0])
    exp_steps = sgn * st_m
    inside = slice(j0, j0 + 3) if edge == "start" else slice(j0 - 3, j0)
    outside = slice(j0 - 3, j0) if edge == "start" else slice(j0, j0 + 3)
    contrast_e = KIND_DIR[kind] * (np.nanmean(ex[:, inside], axis=1) - np.nanmean(ex[:, outside], axis=1))
    c_m, c_se = mean_se(contrast_e[:, None])
    near = np.abs(st_m[[j0 - 1, j0 + 1]])
    return {
        "n_events": int(len(idx)),
        "profile_excess_kwh": prof.tolist(), "profile_se_kwh": se.tolist(),
        "step_by_lag_kwh": {int(L_): float(st_m[j]) for j, L_ in enumerate(LAGS)},
        "step_se_by_lag_kwh": {int(L_): float(st_se[j]) for j, L_ in enumerate(LAGS)},
        "expected_step_sign": int(sgn),
        "step_lag0_expected_dir_kwh": float(exp_steps[j0]), "step_lag0_se_kwh": float(st_se[j0]),
        "step_lagm1_expected_dir_kwh": float(exp_steps[j0 - 1]), "step_lagp1_expected_dir_kwh": float(exp_steps[j0 + 1]),
        "lag_of_largest_expected_step_in_-3..3": int(LAGS[j0 - 3:j0 + 4][np.argmax(exp_steps[j0 - 3:j0 + 4])]),
        "sharpness_lag0_over_max_abs_neighbour": float(exp_steps[j0] / max(near.max(), 1e-12)),
        "contrast_inside_minus_outside_3slots_expected_dir_kwh": float(c_m[0]),
        "contrast_se_kwh": float(c_se[0]),
    }


def run_event_study(es: EventStudy, cands, cleans=(False, True)):
    out = {}
    for cn, codes in cands.items():
        for o in OFFSETS:
            S = L.shift_codes(codes, int(o))
            for season in SEASONS:
                for kind in ("High", "Low"):
                    for edge in ("start", "end"):
                        for clean in cleans:
                            idx = es.events(S, kind, edge, season, clean=clean)
                            key = f"{cn}|{int(o):+d}|{season}|{kind}|{edge}|{'clean' if clean else 'all'}"
                            out[key] = profile_stats(es, idx, kind, edge)
    return out


def alignment_scores(prof, which="all"):
    """Per (candidate, offset, season): sum over the 4 edge types of the lag-0 step in the
    expected direction, and of the inside-outside contrast."""
    rows = []
    for cn in CANDS:
        for o in OFFSETS:
            for season in SEASONS:
                tot = con = 0.0
                n_best = 0
                parts = {}
                for kind in ("High", "Low"):
                    for edge in ("start", "end"):
                        p = prof[f"{cn}|{int(o):+d}|{season}|{kind}|{edge}|{which}"]
                        tot += p["step_lag0_expected_dir_kwh"]
                        con += p["contrast_inside_minus_outside_3slots_expected_dir_kwh"]
                        n_best += int(p["lag_of_largest_expected_step_in_-3..3"] == 0)
                        parts[f"{kind}_{edge}"] = p["step_lag0_expected_dir_kwh"]
                        parts[f"{kind}_{edge}_n"] = p["n_events"]
                rows.append({"candidate": cn, "offset": int(o), "season": season,
                             "sum_step_lag0_expected_dir_kwh": tot, "sum_contrast_kwh": con,
                             "n_edge_types_with_largest_step_at_lag0": n_best, **parts})
    return pd.DataFrame(rows)


def best_offsets(scores):
    best = {}
    for cn in CANDS:
        for season in SEASONS:
            sc = scores[(scores["candidate"] == cn) & (scores["season"] == season)]
            r = sc.loc[sc["sum_step_lag0_expected_dir_kwh"].idxmax()]
            srt = sc["sum_step_lag0_expected_dir_kwh"].sort_values(ascending=False).to_numpy()
            best[f"{cn}|{season}"] = {"best_offset": int(r["offset"]),
                                      "sum_step_lag0_expected_dir_kwh": float(r["sum_step_lag0_expected_dir_kwh"]),
                                      "runner_up_kwh": float(srt[1]),
                                      "at_offset_0_kwh": float(sc.loc[sc["offset"] == 0,
                                                                      "sum_step_lag0_expected_dir_kwh"].iloc[0]),
                                      "n_edge_types_with_largest_step_at_lag0_at_best":
                                          int(r["n_edge_types_with_largest_step_at_lag0"])}
    return best


# =========================================================================== D
def side_effects(es: EventStudy, S):
    """Pre-/post-window behaviour of the placebo-corrected ToU-Std difference under the
    right alignment, clean events only (6 Normal slots on the outside of the edge)."""
    out = {}
    j0 = int(np.flatnonzero(LAGS == 0)[0])
    for season in SEASONS + ("all",):
        for kind in ("High", "Low"):
            for edge in ("start", "end"):
                seas = SEASONS if season == "all" else (season,)
                idx = np.concatenate([es.events(S, kind, edge, s, clean=True) for s in seas])
                if len(idx) < 3:
                    continue
                ex = es.excess(idx, LAGS)
                if edge == "start":
                    outside = {"lag-1": ex[:, j0 - 1], "lags-3..-1": np.nanmean(ex[:, j0 - 3:j0], axis=1),
                               "lags-6..-4": np.nanmean(ex[:, j0 - 6:j0 - 3], axis=1)}
                    inside = np.nanmean(ex[:, j0:j0 + 3], axis=1)
                else:
                    outside = {"lag0": ex[:, j0], "lags0..2": np.nanmean(ex[:, j0:j0 + 3], axis=1),
                               "lags3..5": np.nanmean(ex[:, j0 + 3:j0 + 6], axis=1)}
                    inside = np.nanmean(ex[:, j0 - 3:j0], axis=1)
                d = {"n_events": int(len(idx))}
                im, ise = mean_se(inside[:, None])
                d["inside_3slots_kwh_mean_se"] = [float(im[0]), float(ise[0])]
                for k_, v in outside.items():
                    m, se = mean_se(v[:, None])
                    d[f"outside_{k_}_kwh_mean_se"] = [float(m[0]), float(se[0])]
                    d[f"outside_{k_}_t"] = float(m[0] / se[0]) if se[0] > 0 else None
                    d[f"outside_{k_}_over_inside"] = float(m[0] / im[0]) if im[0] != 0 else None
                out[f"{season}|{kind}|{edge}"] = d
    return out


def day_rebound(es: EventStudy, S):
    """Rebound outside the window on single-block event days (block inside one day): mean
    excess over the day's Normal slots more than 3 slots before / after the block.
    Also the paper's two targeted statements: afternoon (12:00 to 3 slots before the block)
    before an evening High (block start in slots 33..41), and afternoon/evening (3 slots after
    the block to 24:00) after a morning Low (block end in slots 12..26).  Slot times are in
    the authors' frame (GMT label of the half-hour end)."""
    variants = {
        "High|before": ("High", "before", lambda s0, e0: True, 0),
        "High|after": ("High", "after", lambda s0, e0: True, 0),
        "Low|before": ("Low", "before", lambda s0, e0: True, 0),
        "Low|after": ("Low", "after", lambda s0, e0: True, 0),
        "eveningHigh|afternoon_before": ("High", "before", lambda s0, e0: 33 <= s0 <= 41, 24),
        "morningLow|afternoon_evening_after": ("Low", "after", lambda s0, e0: 12 <= e0 <= 26, 0),
    }
    out = {}
    Sd = S.reshape(ND, H)
    for key, (kind, part, ok_block, first_slot) in variants.items():
        lst = []
        for d in range(ND):
            ev = Sd[d] != 0
            if not ev.any() or set(Sd[d][ev].tolist()) != {KIND_CODE[kind]}:
                continue
            st, en = L.runs(Sd[d], KIND_CODE[kind])
            if len(st) != 1 or (d > 0 and Sd[d - 1][-3:].any()) or (d < ND - 1 and Sd[d + 1][:3].any()):
                continue
            s0, e0 = int(st[0]), int(en[0])
            if not ok_block(s0, e0):
                continue
            hs = np.arange(first_slot, max(s0 - 3, first_slot)) if part == "before" else np.arange(min(e0 + 3, H), H)
            if len(hs) < 4:
                continue
            T = es.T[(es.day_season[d], es.day_work[d])]
            ex = float(np.mean(es.r[48 * d + hs] - T[hs + 6]))
            lst.append((ex, es.null_sd(d, hs + 6), s0, e0, str(es.day_season[d])))
        if len(lst) < 2:
            out[key] = {"n_days": len(lst)}
            continue
        v = np.array([x[0] for x in lst])
        z = v / np.array([x[1] for x in lst])
        out[key] = {"n_days": int(len(v)), "mean_excess_kwh": float(v.mean()),
                    "se_kwh": float(v.std(ddof=1) / np.sqrt(len(v))),
                    "t": float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))),
                    "mean_z_vs_normal_days": float(z.mean()),
                    "n_GMT_BST": [sum(x[4] == "GMT" for x in lst), sum(x[4] == "BST" for x in lst)],
                    "median_block": f"{slot_time(int(np.median([x[2] for x in lst])))}-"
                                    f"{slot_time(int(np.median([x[3] for x in lst])) % 48)}"}
    return out


def describe_blocks(day_codes):
    parts = []
    for code in (2, 1):
        st, en = L.runs(day_codes, code)
        for a, b in zip(st, en):
            parts.append((int(a), f"{CODE_NAME[code]} {slot_time(int(a))}-{slot_time(int(b)) if b < 48 else '24:00'}"))
    return ", ".join(p for _, p in sorted(parts)) or "none"


def typical_response(es: EventStudy, A_c, C, o):
    """Calibration: mean placebo-corrected excess per (season, kind, slot of day) on event
    slots where the two schedules agree (C frame, response read o slots later)."""
    agree = (A_c == C) & (C != 0)
    E = {}
    for season in SEASONS:
        for code in (1, 2):
            vals = [[] for _ in range(H)]
            for g in np.flatnonzero(agree & (C == code)):
                d, h = divmod(int(g), H)
                if es.day_season[d] != season or not (0 <= g + o < N):
                    continue
                vals[h].append(es.r[g + o] - es.tmpl(d, h + o + 6))
            allv = [x for v in vals for x in v]
            overall = float(np.mean(allv)) if allv else 0.0
            E[(season, code)] = np.array([np.mean(v) if len(v) >= 3 else overall for v in vals])
    return E


def content_comparison(es: EventStudy, A, C, o, dates, months):
    """Compare the authors' dummies (brought to the C frame: A shifted -1 slot) with C day by
    day; score the disputed slots against the consumption response read at offset ``o``:
    Gaussian log-likelihood ratio of 'ours' vs 'authors' using calibrated typical responses
    (agreed event slots) and the Normal-day noise of the block mean.  LLR > 0 favours ours."""
    A_c = L.shift_codes(A, -1)
    E = typical_response(es, A_c, C, o)
    Ad, Cd = A_c.reshape(ND, H), C.reshape(ND, H)
    rows = []
    m_all = dates.dt.month.to_numpy()
    for d in range(ND):
        if m_all[d] not in months or np.array_equal(Ad[d], Cd[d]):
            continue
        season = es.day_season[d]
        rec = {"date": str(dates.iloc[d].date()), "weekday": dates.iloc[d].day_name()[:3], "season": season,
               "ours_Tariffs.xlsx_local_clock": describe_blocks(Cd[d]),
               "authors_local_clock_(A shifted back 1 slot)": describe_blocks(Ad[d]),
               "event_day_in": ("both" if Ad[d].any() and Cd[d].any() else "ours only" if Cd[d].any() else
                                "authors only")}
        tests = {}
        llr_day = 0.0
        for co in (0, 1, 2):
            for ca in (0, 1, 2):
                hs = np.flatnonzero((Cd[d] == co) & (Ad[d] == ca)) if co != ca else np.array([], int)
                if len(hs) == 0:
                    continue
                q = hs + o + 6
                pos = 48 * d + hs + o
                ok = (pos >= 0) & (pos < N) & (q >= 0) & (q < len(es.P))
                hs, q, pos = hs[ok], q[ok], pos[ok]
                obs = float(np.mean([es.r[p_] - es.tmpl(d, q_) for p_, q_ in zip(pos, q)]))
                sd = es.null_sd(d, q)
                e_o = float(np.mean(E[(season, co)][hs])) if co else 0.0
                e_a = float(np.mean(E[(season, ca)][hs])) if ca else 0.0
                llr = ((obs - e_a) ** 2 - (obs - e_o) ** 2) / (2 * sd ** 2)
                llr_day += llr
                tests[f"ours {CODE_NAME[co]} / authors {CODE_NAME[ca]}"] = {
                    "n_slots": int(len(hs)), "slots_local": f"{slot_time(int(hs[0]))}-{slot_time(int(hs[-1]) + 1)}",
                    "observed_excess_kwh": obs, "expected_if_ours_kwh": e_o, "expected_if_authors_kwh": e_a,
                    "noise_sd_kwh": sd, "z_vs_normal_days": obs / sd, "llr_ours_minus_authors": float(llr),
                    "favours": "ours" if llr > 2 else "authors" if llr < -2 else "inconclusive"}
        rec["disputed_slots"] = tests
        rec["llr_day_ours_minus_authors"] = float(llr_day)
        rec["day_verdict"] = "ours" if llr_day > 2 else "authors" if llr_day < -2 else "inconclusive"
        rows.append(rec)
    return rows


# =========================================================================== figures
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
MARKERS = {"A": "o", "B": "s", "C": "^"}


def make_figure(prof, scores, kind, best_txt, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#b9b8b3", "axes.linewidth": 0.6,
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "text.color": "#0b0b0b", "axes.titlesize": 9})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    fig.patch.set_facecolor("#fcfcfb")
    handles = {}
    for i, season in enumerate(SEASONS):
        for j, edge in enumerate(("start", "end")):
            ax = axes[i, j]
            ax.set_facecolor("#fcfcfb")
            ax.axhline(0, color="#b9b8b3", lw=0.6)
            ax.axvspan(-0.5, 6.5, color="#efeee9", lw=0) if edge == "start" else ax.axvspan(-6.5, -0.5, color="#efeee9", lw=0)
            ax.axvline(-0.5, color="#8b8a85", lw=0.6)
            ns = []
            for cn in CANDS:
                p = prof[f"{cn}|+0|{season}|{kind}|{edge}|clean"]
                if p.get("n_events", 0) == 0:
                    continue
                y = np.array(p["profile_excess_kwh"]) * 1000
                se = np.array(p["profile_se_kwh"]) * 1000
                ax.fill_between(LAGS, y - se, y + se, color=COLORS[cn], alpha=0.10, lw=0)
                handles[cn], = ax.plot(LAGS, y, color=COLORS[cn], lw=1.1, marker=MARKERS[cn], ms=3.2)
                ns.append(f"{cn} {p['n_events']}")
            ax.text(0.99, 0.02 if (kind == "Low") == (edge == "start") else 0.97, "events: " + ", ".join(ns),
                    transform=ax.transAxes, ha="right", va="bottom" if (kind == "Low") == (edge == "start") else "top",
                    fontsize=7, color="#52514e")
            ax.set_xlim(-6.5, 6.5)
            ax.set_xticks(LAGS)
            ax.grid(axis="y", color="#e4e3de", lw=0.5)
            ax.set_axisbelow(True)
            ax.set_title(f"{season} months: {kind} block {edge.upper()} (schedules at offset 0)")
            ax.set_xlabel(f"lag from block {edge} (half-hour slots); shaded = inside block")
            if j == 0:
                ax.set_ylabel("ToU - Std excess (Wh per household)")
        ax = axes[i, 2]
        ax.set_facecolor("#fcfcfb")
        ax.axhline(0, color="#b9b8b3", lw=0.6)
        for cn in CANDS:
            sc = scores[(scores["candidate"] == cn) & (scores["season"] == season)].sort_values("offset")
            y = (sc[f"{kind}_start"] + sc[f"{kind}_end"]).to_numpy() * 1000
            ax.plot(sc["offset"], y, color=COLORS[cn], lw=1.1, marker=MARKERS[cn], ms=3.5)
        ax.set_xticks(OFFSETS)
        ax.grid(axis="y", color="#e4e3de", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(f"{season} months: lag-0 step vs schedule offset")
        ax.set_xlabel("offset applied to the schedule (slots later)")
        ax.set_ylabel("START + END step at lag 0 (Wh, expected sign)")
    fig.legend([handles[c] for c in CANDS if c in handles], [CAND_LABEL[c] for c in CANDS if c in handles],
               loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"{kind}-price blocks: ToU - Std consumption around block edges in the authors' slot frame. "
                 f"Profiles: clean events (6 Normal slots outside the edge), mean +- 1 s.e., minus the matched "
                 f"Normal-day template;\noffset scan: all events. B and C coincide in GMT months. "
                 f"Sharpest step: {best_txt}", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# =========================================================================== main
def event_study_for(F, EF, n_a, cands, s_best, dates, base):
    a_mean, b_mean = F[:n_a].mean(axis=0), F[n_a:].mean(axis=0)
    ea, eb = EF[:n_a].mean(axis=0), EF[n_a:].mean(axis=0)
    d_frame = L.to_authors_frame(a_mean[None], ea[None], s_best)[0] - L.to_authors_frame(b_mean[None], eb[None], s_best)[0]
    assert np.isfinite(d_frame).all()
    return EventStudy(d_frame.ravel(), base, dates)


def main():
    el = L.timer()
    build = json.loads(L.BUILD_STATS_JSON.read_text(encoding="utf-8"))
    fillst = json.loads(L.FILL_STATS_JSON.read_text(encoding="utf-8"))
    hh = pd.read_parquet(L.HOUSEHOLDS_PARQUET)
    sel = pd.read_parquet(L.SELECTION_PARQUET)
    F = np.load(L.FILLED_NPY).astype(np.float64)
    EF = np.load(L.EDGES_FILLED_NPY).astype(np.float64)
    n_tou = int((sel["group"] == "ToU").sum())
    assert (sel["group"].to_numpy(dtype=object)[:n_tou] == "ToU").all()
    Xall = np.load(L.KWH_NPY, mmap_mode="r")
    Eall = np.load(L.EDGES_NPY)
    rows_tou = sel["row_in_dense"].to_numpy()[:n_tou]
    X_tou = np.asarray(Xall[rows_tou], dtype=np.float64)
    E_tou = Eall[rows_tou].astype(np.float64)
    sel_tou = sel.iloc[:n_tou].reset_index(drop=True)
    print(f"[{el()}] loaded: {len(sel)} selected ({n_tou} ToU)", flush=True)

    # ------------------------------------------------------------- A household counts
    counts = build["counts"]
    tou_gt = counts["completeness_gt_0.95 (n_valid/17520 > 0.95; main)"].get("ToU", 0)
    households = {
        "full_file_rows": build["full_file_rows"], "full_file_households": build["full_file_households"],
        "full_file_households_per_label": build["full_file_households_per_label"],
        "households_with_both_labels_full_file": build["households_with_both_labels_full_file"],
        "households_with_both_labels_2013_window": build["window_households_with_both_labels"],
        "counts": counts, "paper_ToU_count": 1007, "our_ToU_count_completeness_gt_0.95": int(tou_gt),
        "difference_vs_paper": int(tou_gt) - 1007,
        "off_grid_rows_in_window": build["window_off_grid_rows"],
        "off_grid_rows_in_window_with_numeric_value": build["window_off_grid_rows_numeric"],
        "null_string_rows_on_grid_2013": build["window_on_grid_null_string_rows"],
        "duplicates": {k: build[k] for k in build if k.startswith("dup_") and k != "dup_detail_by_timestamp"},
        "dst_check_rows_per_timestamp": build["rows_per_ts_around_dst"],
        "rows_per_year_full_file": build["rows_per_year"],
        "missing_share_2013_all_households": build["missing_share_2013_all_households"],
    }
    print(f"[{el()}] ToU households > 0.95: {tou_gt} (paper 1,007)", flush=True)

    # ------------------------------------------------------------- B slot mapping
    sm, s_best, dates, auth = slot_mapping_section(F, EF, n_tou, X_tou, E_tou, sel_tou, hh, fillst)
    print(f"[{el()}] chosen shift {s_best:+d}: {sm['meaning']}", flush=True)
    gfr = sm["gap_fill_rule_reverse_engineering"]
    for k, v in gfr["by_gap_length"].items():
        print(f"   {k:16s} n={v['n_slots']:5d} " + " ".join(f"{c.split(' ')[0]}={x:.3f}" for c, x in v.items()
                                                          if c.startswith("share_exact_") and x is not None), flush=True)

    # ------------------------------------------------------------- C event study
    cands, same = tariff_candidates()
    base = baseline_days(cands)
    es = event_study_for(F, EF, n_tou, cands, s_best, dates, base)
    print(f"[{el()}] baseline (all-Normal) days: {int(base.sum())} "
          f"(GMT {int((base & (es.day_season == 'GMT')).sum())}, BST {int((base & (es.day_season == 'BST')).sum())})",
          flush=True)
    prof = run_event_study(es, cands)
    scores = alignment_scores(prof, "all")
    scores_clean = alignment_scores(prof, "clean")
    best = best_offsets(scores)
    best_clean = best_offsets(scores_clean)
    print(scores.pivot_table(index=["candidate", "offset"], columns="season",
                             values="sum_step_lag0_expected_dir_kwh").round(4).to_string(), flush=True)
    o_C = {s: best[f"C|{s}"]["best_offset"] for s in SEASONS}
    o_B = {s: best[f"B|{s}"]["best_offset"] for s in SEASONS}
    o_A = {s: best[f"A|{s}"]["best_offset"] for s in SEASONS}
    consistent = {"A": o_A["GMT"] == o_A["BST"], "B": o_B["GMT"] == o_B["BST"], "C": o_C["GMT"] == o_C["BST"]}
    if consistent["C"] and not consistent["B"]:
        right = ("C", o_C["GMT"])
    elif consistent["B"] and not consistent["C"]:
        right = ("B", o_B["GMT"])
    else:
        right = ("undetermined", None)
    assert right[1] is not None, "no season-consistent alignment found"
    S_right = L.shift_codes(cands[right[0]], right[1])
    lcl_offset = right[1] + s_best
    authors_late = {s: -o_A[s] for s in SEASONS}
    jo = (dates.dt.month.to_numpy() <= 10).repeat(H)
    A_vs_right_janoct = {
        "slot_agreement_share": float((cands["A"][jo] == S_right[jo]).mean()),
        "differing_days": sorted({str(dates.iloc[g // H].date()) for g in np.flatnonzero(jo & (cands["A"] != S_right))}),
        "note": "timing identical; the only differences are schedule content on the listed days"}
    print(f"[{el()}] right alignment {right}; LCL offset {lcl_offset:+d}; authors' dummies late by "
          f"{authors_late} slots; A vs right alignment Jan-Oct: {A_vs_right_janoct}", flush=True)

    # robustness: the paper-sized design (identified 1,007 ToU + 1,007 Std)
    pair = sel["in_1007_pair"].to_numpy(bool)
    n_pair_tou = int((pair & (sel["group"] == "ToU").to_numpy()).sum())
    order = np.r_[np.flatnonzero(pair & (sel["group"] == "ToU").to_numpy()),
                  np.flatnonzero(pair & (sel["group"] == "Std").to_numpy())]
    es_pair = event_study_for(F[order], EF[order], n_pair_tou, cands, s_best, dates, base)
    prof_pair = run_event_study(es_pair, cands, cleans=(False,))
    best_pair = best_offsets(alignment_scores(prof_pair, "all"))

    # ------------------------------------------------------------- D side effects + content
    side = side_effects(es, S_right)
    rebound = day_rebound(es, S_right)
    o_resp = right[1] if right[0] == "C" else right[1] - 2      # response offset in the C frame (GMT season)
    novdec = content_comparison(es, cands["A"], cands["C"], o_resp, dates, months={11, 12})
    janoct = content_comparison(es, cands["A"], cands["C"], o_resp, dates, months=set(range(1, 11)))
    A_c = L.shift_codes(cands["A"], -1).reshape(ND, H)
    C_d = cands["C"].reshape(ND, H)
    ev_A, ev_C = A_c.any(axis=1), C_d.any(axis=1)
    nd_mask = np.isin(dates.dt.month.to_numpy(), [11, 12])
    summary_novdec = {
        "event_days_authors_year": int(ev_A.sum()), "event_days_ours_year": int(ev_C.sum()),
        "low_slots_authors_year": int((cands["A"] == 1).sum()), "low_slots_ours_year": int((cands["C"] == 1).sum()),
        "high_slots_authors_year": int((cands["A"] == 2).sum()), "high_slots_ours_year": int((cands["C"] == 2).sum()),
        "event_days_authors_novdec": int(ev_A[nd_mask].sum()), "event_days_ours_novdec": int(ev_C[nd_mask].sum()),
        "days_ours_only_novdec": [str(dates.iloc[d].date()) for d in np.flatnonzero(nd_mask & ev_C & ~ev_A)],
        "days_authors_only_novdec": [str(dates.iloc[d].date()) for d in np.flatnonzero(nd_mask & ev_A & ~ev_C)],
        "days_both_but_different_novdec": [r["date"] for r in novdec if r["event_day_in"] == "both"],
        "day_verdict_counts_novdec": {v: sum(r["day_verdict"] == v for r in novdec)
                                      for v in ("ours", "authors", "inconclusive")},
        "block_verdict_counts_novdec": {v: sum(t["favours"] == v for r in novdec for t in r["disputed_slots"].values())
                                        for v in ("ours", "authors", "inconclusive")},
        "sum_llr_novdec_ours_minus_authors": float(sum(r["llr_day_ours_minus_authors"] for r in novdec)),
        "day_verdicts_janoct": {r["date"]: r["day_verdict"] for r in janoct},
    }
    print(f"[{el()}] Nov-Dec: {json.dumps(summary_novdec)[:900]}", flush=True)

    # ------------------------------------------------------------- figures
    best_txt = (f"C at offset {right[1]:+d} = A at offset {right[1] - 1:+d}" if right[0] == "C" else
                f"{right[0]} at offset {right[1]:+d}")
    make_figure(prof, scores, "High", best_txt, L.FIG / "event_study_high.png")
    make_figure(prof, scores, "Low", best_txt, L.FIG / "event_study_low.png")

    # ------------------------------------------------------------- json
    keep_prof = {k: v for k, v in prof.items() if k.split("|")[1] == "+0" or k.startswith(f"{right[0]}|{right[1]:+d}|")}
    out = {
        "description": "Data and alignment checks for the 2013 LCL rebuild (scripts/02_lcl_checks.py). "
                       "Energies in kWh per household per half-hour unless stated.",
        "inputs": {"raw_csv": str(L.RAW_CSV), "tariffs_xlsx": str(L.TARIFFS_XLSX), "upstream_data": str(L.UP_DATA)},
        "households": households,
        "selection": {"n_tou": n_tou, "n_std": int(len(sel) - n_tou), "seed": fillst["seed"],
                      "threshold": fillst["threshold"], "n_in_authors_tou_1007": int(sel["in_authors_tou_1007"].sum()),
                      "n_in_1007_pair": int(pair.sum())},
        "gap_filling": fillst,
        "slot_mapping": sm,
        "tariff_alignment": {
            "frame": f"authors' slot frame (consumption mapped with shift {s_best:+d}); d = mean ToU - mean Std "
                     f"(filled, {n_tou} vs {len(sel) - n_tou} households)",
            "residual": "d minus its same-slot median over all-Normal days of the same season; event profiles are "
                        "additionally corrected by the matched (season, work-day, slot) mean residual of all-Normal days",
            "offset_convention": "offset o moves every event of the candidate o slots later",
            "score": "sum over High/Low x START/END of the mean lag-0 step in the expected direction (kWh)",
            "candidates": CAND_LABEL, "authors_dummies_identical_across_clusters": bool(same),
            "baseline_days": {"n": int(base.sum()), "GMT": int((base & (es.day_season == 'GMT')).sum()),
                              "BST": int((base & (es.day_season == 'BST')).sum())},
            "scores_all_events": scores.to_dict(orient="records"),
            "best_offset_per_candidate_season_all_events": best,
            "best_offset_per_candidate_season_clean_events": best_clean,
            "best_offset_robustness_1007_pair": best_pair,
            "season_consistent_candidates": consistent,
            "right_alignment": {"candidate": right[0], "offset": right[1], "lcl_offset_slots": lcl_offset,
                                "meaning": f"a price period starting at London clock time t (converted to GMT) "
                                           f"first shows in the LCL reading labelled t {lcl_offset:+d}*30 min, i.e. "
                                           "LCL timestamps label the END of the half-hour"
                                           if lcl_offset == 1 else f"LCL offset {lcl_offset:+d}",
                                "equivalent_lib_tariffs_call": (f"lib.tariffs.tariff_dummies(xlsx, mode='local', "
                                                                f"slot_offset={right[1]:+d})" if right[0] == "C" else
                                                                f"lib.tariffs.tariff_dummies(xlsx, mode='gmt', "
                                                                f"slot_offset={right[1]:+d})"),
                                "authors_slot_time_meaning": "authors' slot h of date d = LCL reading labelled "
                                                             "d + h*30 min GMT = consumption of the half-hour ENDING "
                                                             "then (slot 0 = 23:30-24:00 GMT of the previous day); "
                                                             "in BST months local clock = GMT + 1 h"},
            "authors_dummies_late_by_slots": authors_late,
            "authors_dummies_vs_right_alignment_jan_oct": A_vs_right_janoct,
            "profiles": keep_prof,
        },
        "side_effects_under_right_alignment": side,
        "day_level_rebound_under_right_alignment": rebound,
        "schedule_content": {"summary": summary_novdec, "novdec_differing_days": novdec,
                             "janoct_differing_days": janoct},
    }
    L.save_json(out, OUT_JSON)
    print(f"[{el()}] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
