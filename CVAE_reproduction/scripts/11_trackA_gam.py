"""Track A, semi-parametric generator: causality-model validation + GAM generator on the
authors' prepared data (Bregere & Bessa 2020, Sec. IV-A/App. B, Sec. VI/App. E, Sec. VII).

Steps
  1. Causality model (lib/causality.py) fitted on each cluster's aggregate series of
     power_consump_nmf.feather; compared to agreg_profiles_nmf.feather (variants: all 365 days /
     train days, LAML / GCV; cluster-label mapping detected from the data)
     -> outputs/trackA/causality_check.json
  2. GAM generator (lib/gam_generator.py) per cluster on train_nmf.feather
     -> outputs/trackA/models/gam_params_c<c>.pkl
  3. 200 samples for the 92 test days (ascending dates) -> outputs/trackA/gam_samples_c<c>.npy
     tariff scenarios                                -> outputs/trackA/gam_scen_<S>_c<c>.npy
        (Normal = all dummies 0; Low = low slots 9-19 = 04:00-09:30 GMT; High = high slots 39-44 = 19:00-22:00 GMT;
         windows from lib.effects via lib.gam_generator.SCENARIOS, REPRO_SPEC section 7)
     deterministic test means f                      -> outputs/trackA/gam_f_test_c<c>.npy
  4. Diagnostics -> outputs/trackA/gam_diagnostics.json, gam_smoothing.csv, gam_sigma.csv
     Figures     -> outputs/figures/trackA_gam_fig5_cluster3.png, trackA_gam_fig5_cluster2.png (the paper's Fig. 5
                    shows cluster 2), trackA_gam_fig6_cluster3.png, trackA_gam_fig7_allclusters.png,
                    trackA_gam_fig8_allclusters.png

Run (from the project or REPRO dir):  python -P scripts/11_trackA_gam.py [--stage all|scenarios]
  all        (default) steps 1-4: fits the causality model and the GAMs, samples, diagnostics, figures
  scenarios  no fitting: loads models/gam_params_c<c>.pkl, checks that the stored samples regenerate bitwise,
             regenerates the three scenario arrays with the stored seeds (identical arrays are not rewritten),
             updates the 'scenario_effects' entries and the scenario description of gam_diagnostics.json, and
             redraws the figures
The upstream repo is only read (feather files by absolute path); nothing in it is executed.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")  # spec: at most 4 threads

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))

from lib import effects as ef  # noqa: E402  (scenario windows: single source of truth)
from lib.causality import TARIFFS, fit_causality  # noqa: E402
from lib.gam_generator import (H, SCENARIOS, fit_gam_generator, generate,  # noqa: E402
                               predict_mean, set_sigma, tariff_scenario)

UPSTREAM = REPRO / "data" / "authors"  # copy of the authors' repo data/*.feather (byte-identical; read as data only)
OUT = REPRO / "outputs" / "trackA"
MODELS = OUT / "models"
FIGS = REPRO / "outputs" / "figures"
BASE_SEED = 20200727  # fixed; per-cluster streams via SeedSequence
N_SAMPLES = 200
CLUSTERS = (0, 1, 2, 3)
PAPER_FIG4 = {  # approximate medians read off Fig. 4 (spec section 3), clusters 0..3
    "rmse_gam": [0.015, 0.016, 0.015, 0.018], "rmse_cvae": [0.020, 0.022, 0.018, 0.025],
    "es_gam": [0.155, 0.150, 0.175, 0.135], "es_cvae": [0.150, 0.145, 0.170, 0.130],
    "vg_gam": [4, 4, 5, 5], "vg_cvae": [6, 7, 7, 10],
}


def cols(prefix):
    return [f"{prefix}_{h:02d}" for h in range(H)]


def load_feather(name):
    df = pd.read_feather(UPSTREAM / f"{name}.feather")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "cluster" in df.columns:
        df["cluster"] = df["cluster"].astype(int)
    return df


def log(msg):
    print(msg, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else round(v, 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ======================================================================================
# 1. causality model vs the authors' agreg_profiles_nmf
# ======================================================================================

def authors_profiles(ap, mapping):
    """(4, 48, 3) arrays (power_consump cluster order) of the authors' mean / variance."""
    out = {}
    for typ in ("mean", "variance"):
        arr = np.full((4, H, 3), np.nan)
        for c in CLUSTERS:
            for p, pr in enumerate(("low", "normal", "high")):
                g = ap[(ap["type"] == typ) & (ap["cluster"] == mapping[c]) & (ap["price"] == pr)]
                g = g.sort_values("instant")
                assert len(g) == H, (typ, c, pr, len(g))
                arr[c, :, p] = g["kWh"].to_numpy()
        out[typ] = arr
    return out


def compare_block(mine, ref):
    a, b = mine.ravel(), ref.ravel()
    return dict(corr=float(np.corrcoef(a, b)[0, 1]), max_abs_diff=float(np.abs(a - b).max()),
                mean_abs_diff=float(np.abs(a - b).mean()), mean_signed_diff=float((a - b).mean()),
                median_ratio=float(np.median(a / b)))


def compare_to_authors(res, A):
    out = {}
    for p, name in enumerate(TARIFFS):
        out[f"mean/{name}"] = compare_block(res["mu"][:, :, p], A["mean"][:, :, p])
        out[f"variance_vs_sigma/{name}"] = compare_block(res["sigma"][:, :, p], A["variance"][:, :, p])
        out[f"variance_vs_sigma2/{name}"] = compare_block(res["sigma"][:, :, p] ** 2,
                                                          A["variance"][:, :, p])
    for p, name in ((0, "Low"), (2, "High")):
        mine = res["mu"][:, :, p] - res["mu"][:, :, 1]
        ref = A["mean"][:, :, p] - A["mean"][:, :, 1]
        blk = compare_block(mine, ref)
        blk.pop("median_ratio")
        blk["per_cluster_corr"] = [float(np.corrcoef(mine[c], ref[c])[0, 1]) for c in CLUSTERS]
        out[f"effect_{name}_minus_Normal"] = blk
    return out


def causality_check():
    log("== 1. causality model vs agreg_profiles_nmf")
    pc = load_feather("power_consump_nmf").sort_values(["cluster", "date"])
    tr = load_feather("train_nmf")
    ap = pd.read_feather(UPSTREAM / "agreg_profiles_nmf.feather")
    train_dates = set(tr["date"])
    Y = np.stack([pc[pc["cluster"] == c][cols("consumption")].to_numpy(float) for c in CLUSTERS])
    d0 = pc[pc["cluster"] == 0]
    for c in CLUSTERS[1:]:  # tariffs / temperatures identical across clusters
        dc = pc[pc["cluster"] == c]
        assert np.array_equal(dc[cols("temp")].to_numpy(), d0[cols("temp")].to_numpy())
        assert np.array_equal(dc[cols("low")].to_numpy(), d0[cols("low")].to_numpy())
    temp = d0[cols("temp")].to_numpy(float)
    low, high = d0[cols("low")].to_numpy(float), d0[cols("high")].to_numpy(float)
    is_tr = d0["date"].isin(train_dates).to_numpy()
    assert is_tr.sum() == 273 and len(is_tr) == 365

    # ---- cluster-label mapping between power_consump_nmf and agreg_profiles_nmf
    raw_mean = Y.mean(axis=1)  # (4, 48) mean daily profile of each power_consump cluster
    corr = np.zeros((4, 4))
    for c in CLUSTERS:
        for c2 in CLUSTERS:
            ref = ap[(ap["type"] == "mean") & (ap["price"] == "normal") & (ap["cluster"] == c2)]
            corr[c, c2] = np.corrcoef(raw_mean[c], ref.sort_values("instant")["kWh"].to_numpy())[0, 1]
    from itertools import permutations
    best = max(permutations(range(4)), key=lambda pm: sum(corr[c, pm[c]] for c in CLUSTERS))
    mapping = {c: int(best[c]) for c in CLUSTERS}
    log(f"   corr(raw mean profile of power cluster c, authors' Normal mean of agreg cluster c'):\n"
        f"{np.round(corr, 4)}\n   best mapping power->agreg: {mapping}")
    identity = {c: c for c in CLUSTERS}
    A_map, A_id = authors_profiles(ap, mapping), authors_profiles(ap, identity)

    variants = {
        "all365_avg_all": dict(sl=slice(None), avg=None),
        "train273_avg_train": dict(sl=is_tr, avg=None),
        "train273_avg_all365": dict(sl=is_tr, avg=temp),
    }
    results = {}
    fits = {}
    for method in ("laml", "gcv"):
        for vname, v in variants.items():
            t0 = time.time()
            res = fit_causality(Y[:, v["sl"]], temp[v["sl"]], low[v["sl"]], high[v["sl"]],
                                avg_temp=v["avg"], method=method)
            key = f"{vname}/{method}"
            fits[key] = res
            results[key] = dict(
                with_detected_mapping=compare_to_authors(res, A_map),
                with_identity_labels={k: v2 for k, v2 in compare_to_authors(res, A_id).items()
                                      if k.startswith("mean/")},
                fit_seconds=round(time.time() - t0, 2),
                edf_median=float(np.median(res["edf"])),
                frac_lambda_on_grid_edge=float(res["on_grid_edge"].mean()),
                frac_sigma_at_floor=float(np.isclose(res["sigma"], 0.01).mean()),
                all_converged=bool(res["converged"].all()),
            )
            m = results[key]["with_detected_mapping"]
            log(f"   {key:32s} mean r L/N/H = " + "/".join(f"{m[f'mean/{t}']['corr']:.4f}" for t in TARIFFS)
                + "  mean|d| = " + "/".join(f"{m[f'mean/{t}']['mean_abs_diff']:.4f}" for t in TARIFFS)
                + "  | var~sigma r = " + "/".join(f"{m[f'variance_vs_sigma/{t}']['corr']:.3f}" for t in TARIFFS)
                + "  var~sigma2 r = " + "/".join(f"{m[f'variance_vs_sigma2/{t}']['corr']:.3f}" for t in TARIFFS))
    # best variant by mean-profile agreement
    score = {k: np.mean([r["with_detected_mapping"][f"mean/{t}"]["mean_abs_diff"] for t in TARIFFS])
             for k, r in results.items()}
    best_variant = min(score, key=score.get)
    bv = results[best_variant]["with_detected_mapping"]
    sig_better = {t: ("sigma" if bv[f"variance_vs_sigma/{t}"]["mean_abs_diff"]
                      < bv[f"variance_vs_sigma2/{t}"]["mean_abs_diff"] else "sigma^2") for t in TARIFFS}
    a_var = A_map["variance"]
    night = slice(2, 10)
    res_best = fits[best_variant]
    evidence = dict(
        authors_variance_min=float(a_var.min()),
        authors_variance_median=float(np.median(a_var)),
        authors_variance_night_median_normal=float(np.median(a_var[:, night, 1])),
        my_sigma_night_median_normal=float(np.median(res_best["sigma"][:, night, 1])),
        raw_sd_night_median_normal=float(np.median([Y[c, low[:, h] + high[:, h] == 0, h].std()
                                                    for c in CLUSTERS for h in range(2, 10)])),
        my_sigma_median=float(np.median(res_best["sigma"])),
        my_sigma2_median=float(np.median(res_best["sigma"] ** 2)),
        note=("mgcv gaulss(b=0.01) has sd = 0.01 + exp(eta) >= 0.01, yet 'variance' goes down to "
              f"{a_var.min():.4f}; as sigma^2 of the aggregate it would be ~{np.median(a_var)/np.median(res_best['sigma']**2):.0f}x "
              "too large. Its size (night ~0.004-0.006, evening peak ~0.06) is what household-level "
              "sigma^2 would give (sd ~0.07-0.25 kWh), i.e. the authors' profiles were probably "
              "aggregated from per-household causality fits (mean of mu_i, and sigma_i^2), which "
              "cannot be checked with the cluster-level data available in Track A."),
    )
    check = dict(
        description=("Causality model (lib/causality.py) fitted on each cluster aggregate of the "
                     "authors' power_consump_nmf.feather and compared with agreg_profiles_nmf.feather."),
        cluster_label_mapping_power_to_agreg=mapping,
        mapping_evidence_corr_matrix_rows_power_cols_agreg=corr,
        mapping_note=("agreg_profiles_nmf uses different cluster labels than power_consump_nmf / "
                      "sum_up_nmf (whose sizes 325/295/229/158 follow the power_consump labels); "
                      "under the mapping the raw mean profiles correlate ~1.000 with the authors' "
                      "Normal mean profiles and the levels agree."),
        agreg_cluster_sizes_under_mapping={int(mapping[c]): n for c, n in zip(CLUSTERS, (325, 295, 229, 158))},
        best_variant_by_mean_profiles=best_variant,
        variance_column_interpretation=sig_better,
        variance_evidence=evidence,
        variants=results,
        settings=fits[best_variant]["settings"],
    )
    with open(OUT / "causality_check.json", "w") as fh:
        json.dump(jsonable(check), fh, indent=1)
    log(f"   best variant: {best_variant}; 'variance' closer to: {sig_better}")
    return check, mapping, A_map


# ======================================================================================
# 2-4. GAM generator
# ======================================================================================

def energy_score(S, y):
    """eq. 14 per day; S (N, D, 48), y (D, 48) -> (D,)"""
    n2 = S.shape[0] // 2
    d1 = np.linalg.norm(S[:n2] - y[None], axis=2)
    d2 = np.linalg.norm(S[:n2] - S[n2:2 * n2], axis=2)
    return d1.mean(axis=0) - 0.5 * d2.mean(axis=0)


def variogram_score(S, y, p=0.5):
    """eq. 16 per day; S (N, D, 48), y (D, 48) -> (D,)"""
    out = np.empty(y.shape[0])
    for d in range(y.shape[0]):
        obs = np.abs(y[d][:, None] - y[d][None, :]) ** p
        pred = (np.abs(S[:, d, :, None] - S[:, d, None, :]) ** p).mean(axis=0)
        out[d] = ((obs - pred) ** 2).sum()
    return out


def day_rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean(axis=-1))


SCENARIO_TEXT = (f"Normal = all dummies 0; Low = low {ef.WINDOW_TEXT['Low']}; High = high {ef.WINDOW_TEXT['High']} "
                 f"(lib.effects windows, REPRO_SPEC section 7; phases 1-2 used 9-18 / 39-43)")


def scenario_effects_summary(scen, info):
    """In-window effect and confinement checks of the Low / High scenarios (common random numbers with Normal)."""
    win = {k: np.array(v) for k, v in ef.WINDOWS.items()}
    scen_eff = {}
    for name in ("Low", "High"):
        diff = scen[name].mean(axis=0) - scen["Normal"].mean(axis=0)  # (92, 48)
        outside = np.setdiff1d(np.arange(H), win[name])
        xi = info["xi_low"] if name == "Low" else info["xi_high"]
        scen_eff[name] = dict(
            window_slots=win[name].tolist(),
            window_clock_gmt=ef.WINDOW_CLOCK[name],
            mean_effect_in_window_kWh=float(diff[:, win[name]].mean()),
            mean_effect_in_window_pct=float(100 * diff[:, win[name]].mean()
                                            / scen["Normal"].mean(axis=0)[:, win[name]].mean()),
            xi_in_window=xi[win[name]],
            n_window_slots_with_expected_sign=int((xi[win[name]] > 0).sum() if name == "Low"
                                                  else (xi[win[name]] < 0).sum()),
            max_abs_mean_diff_outside_window_kWh=float(np.abs(diff[:, outside]).max()),
            max_abs_sample_diff_outside_window_kWh=float(
                np.abs(scen[name][:, :, outside] - scen["Normal"][:, :, outside]).max()),
            sample_sd_ratio_in_window=float(scen[name][:, :, win[name]].std(axis=0).mean()
                                            / scen["Normal"][:, :, win[name]].std(axis=0).mean()),
        )
    return scen_eff


def save_array(path, arr):
    """np.save, unless the file already holds a byte-identical array (then it is left untouched)."""
    arr = np.ascontiguousarray(arr)
    if Path(path).exists():
        old = np.load(path)
        if old.dtype == arr.dtype and old.shape == arr.shape and old.tobytes() == arr.tobytes():
            return "unchanged"
        np.save(path, arr)
        return "changed"
    np.save(path, arr)
    return "new"


def run_gam(mapping, A_map):
    log("== 2. GAM generator per cluster (train_nmf) / 3. samples and scenarios")
    tr, te = load_feather("train_nmf"), load_feather("test_nmf")
    diag = dict(clusters={}, settings={}, paper_fig4_approx_medians=PAPER_FIG4)
    smooth_rows, sigma_rows = [], []
    store = {}
    for c in CLUSTERS:
        trc = tr[tr["cluster"] == c].sort_values("date").reset_index(drop=True)
        tec = te[te["cluster"] == c].sort_values("date").reset_index(drop=True)
        assert len(trc) == 273 and len(tec) == 92 and tec["date"].is_monotonic_increasing
        t0 = time.time()
        P = fit_gam_generator(trc, method="gcv")
        t_fit = time.time() - t0
        with open(MODELS / f"gam_params_c{c}.pkl", "wb") as fh:
            pickle.dump(P, fh)
        y_tr = trc[cols("consumption")].to_numpy(float)
        y_te = tec[cols("consumption")].to_numpy(float)
        f_te = predict_mean(P, tec)
        np.save(OUT / f"gam_f_test_c{c}.npy", f_te)
        ss = np.random.SeedSequence([BASE_SEED, c])
        seed_samples, seed_scen = ss.spawn(2)
        S = generate(P, tec, N_SAMPLES, np.random.default_rng(seed_samples))
        assert S.shape == (N_SAMPLES, 92, H)
        np.save(OUT / f"gam_samples_c{c}.npy", S)
        scen = {}
        for name in SCENARIOS:  # common random numbers across the three scenarios
            cond = tariff_scenario(tec, name)
            scen[name] = generate(P, cond, N_SAMPLES, np.random.default_rng(seed_scen))
            np.save(OUT / f"gam_scen_{name}_c{c}.npy", scen[name])
        # ---- sensitivity: REML smoothing (fit quality only)
        P_reml = fit_gam_generator(trc, method="reml")
        f_te_reml = predict_mean(P_reml, tec)

        # ---- diagnostics
        rm_tr, rm_te = day_rmse(P["f_train"], y_tr), day_rmse(f_te, y_te)
        smean = S.mean(axis=0)
        rm_smean = day_rmse(smean, y_te)
        es, vg = energy_score(S, y_te), variogram_score(S, y_te)
        resid_te = y_te - f_te
        info = P["info"]
        # sigma variants for the scores (same random stream as the main samples)
        sig_variants = {}
        pc = load_feather("power_consump_nmf")
        pcc = pc[pc["cluster"] == c].sort_values("date")
        for vname, table in (
                ("authors_variance_as_sigma", A_map["variance"][c]),
                ("sqrt_authors_variance_as_sigma", np.sqrt(A_map["variance"][c])),
                ("causality_all365_days", "all"),
                ("causality_train_no_sd_floor", "nofloor")):
            Pv = dict(P)
            if isinstance(table, str) and table == "all":
                table = fit_causality(pcc[cols("consumption")].to_numpy(float),
                                      pcc[cols("temp")].to_numpy(float),
                                      pcc[cols("low")].to_numpy(float),
                                      pcc[cols("high")].to_numpy(float))["sigma"][0]
            elif isinstance(table, str) and table == "nofloor":
                table = fit_causality(y_tr, trc[cols("temp")].to_numpy(float),
                                      trc[cols("low")].to_numpy(float),
                                      trc[cols("high")].to_numpy(float), sd_floor=1e-9)["sigma"][0]
            Pv = set_sigma(Pv, table, source=vname)
            Sv = generate(Pv, tec, N_SAMPLES, np.random.default_rng(seed_samples))
            sig_variants[vname] = dict(
                es_median=float(np.median(energy_score(Sv, y_te))),
                vg_median=float(np.median(variogram_score(Sv, y_te))),
                sample_sd_mean=float(Sv.std(axis=0, ddof=1).mean()),
                sigma_median=float(np.median(Pv["sigma"])))
        scen_eff = scenario_effects_summary(scen, info)
        rough = lambda Z: np.abs(np.diff(Z, axis=-1)).mean()  # noqa: E731
        lag1 = lambda Z: np.mean([np.corrcoef(Z.reshape(-1, H)[:, h], Z.reshape(-1, H)[:, h + 1])[0, 1]  # noqa: E731
                                  for h in range(H - 1)])
        dc = dict(
            fit_seconds=round(t_fit, 2),
            train_rmse_f=dict(mean=rm_tr.mean(), median=np.median(rm_tr)),
            test_rmse_f=dict(mean=rm_te.mean(), median=np.median(rm_te),
                             euclidean_median=np.median(rm_te) * np.sqrt(H)),
            test_rmse_f_reml=dict(mean=day_rmse(f_te_reml, y_te).mean(),
                                  median=np.median(day_rmse(f_te_reml, y_te))),
            train_rmse_f_reml_mean=day_rmse(P_reml["f_train"], y_tr).mean(),
            test_rmse_sample_mean=dict(mean=rm_smean.mean(), median=np.median(rm_smean),
                                       euclidean_median=np.median(np.linalg.norm(smean - y_te, axis=1))),
            energy_score=dict(
                mean=es.mean(), median=np.median(es),
                median_term_mean_dist_to_obs=np.median(
                    np.linalg.norm(S[:N_SAMPLES // 2] - y_te[None], axis=2).mean(axis=0)),
                median_term_mean_dist_between_samples=np.median(
                    np.linalg.norm(S[:N_SAMPLES // 2] - S[N_SAMPLES // 2:], axis=2).mean(axis=0))),
            variogram_score_p05=dict(mean=vg.mean(), median=np.median(vg)),
            sigma_variants_scores=sig_variants,
            dispersion=dict(
                mean_sample_sd=float(S.std(axis=0, ddof=1).mean()),
                rms_test_residual_y_minus_f=float(np.sqrt((resid_te ** 2).mean())),
                ratio_sample_sd_to_rms_residual=float(S.std(axis=0, ddof=1).mean()
                                                      / np.sqrt((resid_te ** 2).mean())),
                train_std_resid_sd_mean=float(P["std_resid_sd"].mean()),
                train_std_resid_sd_range=[float(P["std_resid_sd"].min()), float(P["std_resid_sd"].max())],
            ),
            roughness_mean_abs_first_diff=dict(samples=rough(S), observed_test=rough(y_te),
                                               f_test=rough(f_te)),
            lag1_corr_across_halfhours=dict(samples_minus_f=lag1(S - f_te[None]),
                                            observed_residuals=lag1(resid_te[None])),
            frac_negative_samples=float((S < 0).mean()),
            sigma=dict(median_LNH=np.median(P["sigma"], axis=0),
                       n_fallback=int(P["sigma_fallback"].sum()),
                       frac_at_floor_0p01=float(np.isclose(P["sigma"], 0.01).mean()),
                       ratio_to_authors_variance_median_LNH=np.median(P["sigma"] / A_map["variance"][c], axis=0)),
            xi_missing=dict(low=int(info["xi_low_missing"].sum()), high=int(info["xi_high_missing"].sum())),
            smoothing_median=dict(g_tau_tbar_kappa=np.median(info["g"], axis=0),
                                  edf_tau_tbar_kappa=np.median(info["edf_smooth"], axis=0),
                                  edf_total=np.median(info["edf"])),
            frac_smoothing_at_bounds=float(np.mean(np.abs(info["g"]) >= 7.999)),
            frac_g_on_coarse_grid=float(np.mean(np.isclose(info["g"], np.round(info["g"] / 2) * 2,
                                                           atol=1e-3))),
            Sigma_offdiag_lag1_mean=float(np.mean(np.diag(P["Sigma"], 1))),
            jitter_used=P["jitter_used"],
            scenario_effects=scen_eff,
        )
        diag["clusters"][c] = dc
        for h in range(H):
            smooth_rows.append(dict(cluster=c, h=h, g_tau=info["g"][h, 0], g_tbar=info["g"][h, 1],
                                    g_kappa=info["g"][h, 2], lam_tau=info["lam"][h, 0],
                                    lam_tbar=info["lam"][h, 1], lam_kappa=info["lam"][h, 2],
                                    edf_tau=info["edf_smooth"][h, 0], edf_tbar=info["edf_smooth"][h, 1],
                                    edf_kappa=info["edf_smooth"][h, 2], edf_total=info["edf"][h],
                                    gcv=info["crit"][h], b0=info["b0"][h], alpha_workday=info["alpha"][h],
                                    xi_low=info["xi_low"][h], xi_high=info["xi_high"][h],
                                    xi_low_missing=info["xi_low_missing"][h],
                                    xi_high_missing=info["xi_high_missing"][h]))
            cz = P["causality"]
            sigma_rows.append(dict(cluster=c, h=h, sigma_low=P["sigma"][h, 0], sigma_normal=P["sigma"][h, 1],
                                   sigma_high=P["sigma"][h, 2], fallback_low=P["sigma_fallback"][h, 0],
                                   fallback_high=P["sigma_fallback"][h, 2], n_low_train=cz["counts"][h, 0],
                                   n_normal_train=cz["counts"][h, 1], n_high_train=cz["counts"][h, 2],
                                   causality_edf=cz["edf"][h], std_resid_sd=P["std_resid_sd"][h],
                                   authors_variance_low=A_map["variance"][c, h, 0],
                                   authors_variance_normal=A_map["variance"][c, h, 1],
                                   authors_variance_high=A_map["variance"][c, h, 2]))
        log(f"   cluster {c}: fit {t_fit:.1f}s | RMSE f train {rm_tr.mean():.4f} test {rm_te.mean():.4f} "
            f"(median {np.median(rm_te):.4f}; REML {np.median(day_rmse(f_te_reml, y_te)):.4f}) | "
            f"sample-mean RMSE median {np.median(rm_smean):.4f} (paper ~{PAPER_FIG4['rmse_gam'][c]}) | "
            f"ES median {np.median(es):.4f} (paper ~{PAPER_FIG4['es_gam'][c]}) | VG median {np.median(vg):.2f} "
            f"(paper ~{PAPER_FIG4['vg_gam'][c]}) | std-resid sd mean {P['std_resid_sd'].mean():.3f}")
        log(f"      scenario Low: in-window mean effect {scen_eff['Low']['mean_effect_in_window_kWh']:+.4f} kWh, "
            f"max |diff| outside {scen_eff['Low']['max_abs_sample_diff_outside_window_kWh']:.2e}; "
            f"High: {scen_eff['High']['mean_effect_in_window_kWh']:+.4f} kWh, outside "
            f"{scen_eff['High']['max_abs_sample_diff_outside_window_kWh']:.2e}")
        store[c] = dict(P=P, S=S, scen=scen, y_te=y_te, f_te=f_te, dates=tec["date"].to_numpy())
    diag["settings"] = dict(
        gam=dict(k=10, basis="natural cubic regression spline (mgcv cr analogue), centred",
                 smoothing="GCV (mgcv default GCV.Cp), grid -6..6 step 2 per smooth + L-BFGS-B",
                 terms="b0 + s(temp_hh) + s(temp_smooth) + s(pos_y) + work_day + low_hh + high_hh",
                 fit_days="train_nmf (273 days)"),
        sigma="causality model (LAML, per-series lambda, sd floor 0.01) on the cluster's 273 training days; "
              "NaN -> sigma(Normal)",
        Sigma="np.corrcoef of standardized training residuals (eqs. 37-40); Cholesky",
        seeds=dict(base=BASE_SEED, scheme="SeedSequence([base, cluster]).spawn(2): [samples, scenarios]; "
                   "the three scenarios share one stream (common random numbers)"),
        n_samples=N_SAMPLES, test_days="92, ascending dates",
        scenarios=SCENARIO_TEXT,
        sigma_variant_scores=("ES/VG medians with sigma replaced by the authors' 'variance' column (as sigma, "
                              "and its square root) and by the causality fit on all 365 days; same random stream"),
    )
    with open(OUT / "gam_diagnostics.json", "w") as fh:
        json.dump(jsonable(diag), fh, indent=1)
    pd.DataFrame(smooth_rows).to_csv(OUT / "gam_smoothing.csv", index=False)
    pd.DataFrame(sigma_rows).to_csv(OUT / "gam_sigma.csv", index=False)
    return diag, store


def regen_scenarios():
    """Stage 'scenarios': no fitting. The stored GAMs (models/gam_params_c<c>.pkl) regenerate the samples (checked
    bitwise against the stored arrays) and the three tariff scenarios with run_gam's seeds (SeedSequence([BASE_SEED,
    c]).spawn(2): samples from stream 0, the scenarios from stream 1, common random numbers); the scenario entries of
    gam_diagnostics.json are recomputed. Returns the figure store."""
    log(f"== GAM scenarios from the stored models (no refit): {SCENARIO_TEXT}")
    te = load_feather("test_nmf")
    diag_path = OUT / "gam_diagnostics.json"
    with open(diag_path) as fh:
        diag = json.load(fh)
    store = {}
    for c in CLUSTERS:
        tec = te[te["cluster"] == c].sort_values("date").reset_index(drop=True)
        assert len(tec) == 92 and tec["date"].is_monotonic_increasing
        with open(MODELS / f"gam_params_c{c}.pkl", "rb") as fh:
            P = pickle.load(fh)
        y_te = tec[cols("consumption")].to_numpy(float)
        f_te = predict_mean(P, tec)
        if not np.array_equal(f_te, np.load(OUT / f"gam_f_test_c{c}.npy")):
            raise RuntimeError(f"cluster {c}: the stored GAM does not reproduce gam_f_test_c{c}.npy")
        ss = np.random.SeedSequence([BASE_SEED, c])
        seed_samples, seed_scen = ss.spawn(2)
        S = generate(P, tec, N_SAMPLES, np.random.default_rng(seed_samples))
        st = {"samples": save_array(OUT / f"gam_samples_c{c}.npy", S)}
        if st["samples"] != "unchanged":
            raise RuntimeError(f"cluster {c}: regenerated GAM samples differ from the stored ones ({st['samples']})")
        scen = {}
        for name in SCENARIOS:  # common random numbers across the three scenarios
            scen[name] = generate(P, tariff_scenario(tec, name), N_SAMPLES, np.random.default_rng(seed_scen))
            st[name] = save_array(OUT / f"gam_scen_{name}_c{c}.npy", scen[name])
        scen_eff = scenario_effects_summary(scen, P["info"])
        diag["clusters"][str(c)]["scenario_effects"] = jsonable(scen_eff)
        store[c] = dict(P=P, S=S, scen=scen, y_te=y_te, f_te=f_te, dates=tec["date"].to_numpy())
        log(f"   cluster {c}: arrays vs stored {st}; Low in-window {scen_eff['Low']['mean_effect_in_window_kWh']:+.4f} "
            f"kWh ({scen_eff['Low']['mean_effect_in_window_pct']:+.1f}%), xi>0 in "
            f"{scen_eff['Low']['n_window_slots_with_expected_sign']}/{len(ef.LOW_WINDOW)} slots, max |diff| outside "
            f"{scen_eff['Low']['max_abs_sample_diff_outside_window_kWh']:.2e}; High "
            f"{scen_eff['High']['mean_effect_in_window_kWh']:+.4f} kWh ({scen_eff['High']['mean_effect_in_window_pct']:+.1f}%), "
            f"xi<0 in {scen_eff['High']['n_window_slots_with_expected_sign']}/{len(ef.HIGH_WINDOW)}, outside "
            f"{scen_eff['High']['max_abs_sample_diff_outside_window_kWh']:.2e}")
    diag["settings"]["scenarios"] = SCENARIO_TEXT
    diag["settings"]["scenario_entries_regenerated"] = (
        f"{time.strftime('%Y-%m-%d %H:%M')}: scenario arrays and 'scenario_effects' regenerated from the stored "
        "models (11_trackA_gam.py --stage scenarios, no refit); every other entry is from the fit run")
    with open(diag_path, "w") as fh:
        json.dump(jsonable(diag), fh, indent=1)
    return store


# ======================================================================================
# 5. figures
# ======================================================================================
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_NORMAL, C_HIGH, C_LOW = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated)
C_SAMPLE = "#9a9893"


def style_axes(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, width=0.6)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


# Time axes: slot h is drawn at h/2 h, the END of its half-hour (end-labelled slots, GMT; REPRO_SPEC sections 6-7),
# as in the CVAE figures; the tariff windows are shaded over their physical intervals (lib.effects.WINDOW_HOURS).
TIME_AXIS_LABEL = "time of day, GMT (the point at time t is the half-hour ending at t)"


def plot_three_days(ax, st, n_days=3, n_lines=20, title=None, legend=True):
    S, y = st["S"][:, :n_days], st["y_te"][:n_days]
    x = np.arange(n_days * H) / 2.0  # end of each half-hour, hours since 00:00 GMT of test day 1
    for i in range(n_lines):
        ax.plot(x, S[i].ravel(), color=C_SAMPLE, lw=0.5, alpha=0.55,
                label="20 GAM samples" if i == 0 else None)
    ax.plot(x, S.mean(axis=0).ravel(), color=C_NORMAL, lw=2.0, label="mean of 200 samples")
    ax.plot(x, y.ravel(), color=INK, lw=1.2, label="observed")
    for d in range(1, n_days):
        ax.axvline(24 * d, color=AXIS, lw=0.8)
    ticks = np.arange(0, 24 * n_days + 1, 6)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t % 24):02d}h" for t in ticks])
    dates = pd.to_datetime(st["dates"][:n_days])
    for d, dt in enumerate(dates):
        ax.text(24 * d + 12, 1.0, dt.strftime("%a %d %b %Y"), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xlim(0, 24 * n_days)
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("cluster-mean consumption (kWh per half-hour)")
    style_axes(ax)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left", pad=16)
    if legend:
        ax.legend(fontsize=8, frameon=False, loc="upper left")


# Scenario curves are drawn only from one slot before to one slot after their tariff window (slice a:b):
# with common random numbers they coincide exactly with the Normal scenario everywhere else.
SCEN_SEGMENTS = tuple((name, col, ef.WINDOWS[name][0] - 1, ef.WINDOWS[name][-1] + 2, f"{name} {ef.WINDOW_CLOCK[name]} GMT")
                      for name, col in (("Low", C_LOW), ("High", C_HIGH)))


def shade_windows(ax, offsets=(0.0,)):
    for off in offsets:
        for name, col in (("Low", C_LOW), ("High", C_HIGH)):
            a, b = ef.WINDOW_HOURS[name]
            ax.axvspan(off + a, off + b, color=col, alpha=0.10, lw=0)


def plot_scenarios_day(ax, st, day=0, n_lines=20, legend=True, title=None):
    x = np.arange(H) / 2.0  # end of each half-hour (GMT)
    shade_windows(ax)
    Zn = st["scen"]["Normal"][:, day]
    for i in range(n_lines):
        ax.plot(x, Zn[i], color=C_NORMAL, lw=0.6, alpha=0.45,
                label="Normal all day: 20 samples" if i == 0 else None)
    ax.plot(x, Zn.mean(axis=0), color=C_NORMAL, lw=2.0)
    for name, col, a, b, lab in SCEN_SEGMENTS:
        Z = st["scen"][name][:, day, a:b]
        for i in range(n_lines):
            ax.plot(x[a:b], Z[i], color=col, lw=0.6, alpha=0.8,
                    label=f"{lab}: 20 samples" if i == 0 else None)
        ax.plot(x[a:b], Z.mean(axis=0), color=col, lw=2.0)
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_, hi_ + 0.2 * (hi_ - lo_))      # room for the legend and window labels above the curves
    top = ax.get_ylim()[1]
    for name in ("Low", "High"):
        a, b = ef.WINDOW_HOURS[name]
        ax.text((a + b) / 2, top, f"{name} tariff window", ha="center", va="top", fontsize=8, color=INK2)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xticklabels([f"{t:02d}h" for t in range(0, 25, 3)])
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("cluster-mean consumption (kWh per half-hour)")
    style_axes(ax)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    if legend:  # between the two windows, where the profile is lowest relative to the top
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.41, 1.0))


def plot_scenario_means(ax, st, n_days=3, title=None, legend=True):
    x = np.arange(n_days * H) / 2.0  # end of each half-hour (GMT)
    shade_windows(ax, offsets=[24.0 * d for d in range(n_days)])
    ax.plot(x, st["scen"]["Normal"][:, :n_days].mean(axis=0).ravel(), color=C_NORMAL, lw=1.2,
            label="Normal all day")
    for name, col, a, b, lab in SCEN_SEGMENTS:
        m = st["scen"][name][:, :n_days].mean(axis=0)  # (n_days, 48)
        for d in range(n_days):
            ax.plot(x[d * H + a:d * H + b], m[d, a:b], color=col, lw=1.2,
                    label=lab if d == 0 else None)
    for d in range(1, n_days):
        ax.axvline(24 * d, color=AXIS, lw=0.8)
    ticks = np.arange(0, 24 * n_days + 1, 6)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t % 24):02d}h" for t in ticks])
    ax.set_xlim(0, 24 * n_days)
    ax.set_xlabel(TIME_AXIS_LABEL.replace("time of day", "time of day over 3 test days"))
    ax.set_ylabel("mean of 200 samples (kWh)")
    style_axes(ax)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    if legend:
        ax.legend(fontsize=8, frameon=False, loc="upper left")


def make_figures(store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "figure.facecolor": "white"})
    sizes = {0: 325, 1: 295, 2: 229, 3: 158}
    fig5_note = {2: "the paper's Fig. 5 shows this cluster",
                 3: "the cluster named in the paper's caption; the figure itself shows cluster 2"}

    for c5 in (3, 2):
        fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150)
        plot_three_days(ax, store[c5], title=f"Track A, GAM generator (paper Fig. 5 right): cluster {c5} "
                                             f"({sizes[c5]} households), first three test days\n({fig5_note[c5]})")
        fig.tight_layout()
        fig.savefig(FIGS / f"trackA_gam_fig5_cluster{c5}.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    d0 = pd.to_datetime(store[3]["dates"][0]).strftime("%d %b %Y")
    plot_scenarios_day(ax, store[3], day=0,
                       title=f"Track A, GAM generator (paper Fig. 6 right): cluster 3, test day 1 ({d0})\n"
                             "20 samples per tariff scenario, bold = mean of 200. Outside the shaded windows "
                             "the scenarios coincide\n(same random draws), so only Normal is drawn there.")
    fig.tight_layout()
    fig.savefig(FIGS / "trackA_gam_fig6_cluster3.png")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), dpi=130)
    for c, ax in zip(CLUSTERS, axes):
        plot_three_days(ax, store[c], title=f"cluster {c} ({sizes[c]} households)", legend=(c == 0))
    fig.suptitle("Track A, GAM generator (paper Fig. 7 right): 20 samples, mean of 200, observed",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "trackA_gam_fig7_allclusters.png")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), dpi=130)
    for c, ax in zip(CLUSTERS, axes):
        plot_scenario_means(ax, store[c], title=f"cluster {c} ({sizes[c]} households)", legend=(c == 0))
    fig.suptitle("Track A, GAM generator (paper Fig. 8 right): mean of 200 samples under three tariff "
                 "scenarios\n(Low/High curves drawn around their windows; elsewhere they equal Normal exactly)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "trackA_gam_fig8_allclusters.png")
    plt.close(fig)
    log("   figures written")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "scenarios"],
                    help="all = fit + sample + diagnostics + figures; scenarios = regenerate the tariff scenarios "
                         "and figures from the stored GAMs (no fitting)")
    args = ap.parse_args()
    for d in (OUT, MODELS, FIGS):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.stage == "scenarios":
        store = regen_scenarios()
    else:
        check, mapping, A_map = causality_check()
        diag, store = run_gam(mapping, A_map)
    make_figures(store)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
