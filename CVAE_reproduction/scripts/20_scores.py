"""Authoritative scoring of the generators (paper Sec. VII-A/B, Fig. 4) for Track A or Track B.

Usage (from PROJECT or REPRO):  python -P scripts/20_scores.py --track A|B [--no-figures]

Inputs
  observations  track A: UPSTREAM/data/test_nmf.feather (consumption_00..47 per cluster)
                track B: REPRO/data/test_ours.parquet (same schema); if it does not exist yet the script says so and
                exits with status 0 without writing anything
                test days ascending; checked against outputs/<track>/test_dates.csv when that file exists
  samples       outputs/<track>/{cvae,gam}_samples_c<c>.npy, (200, 92, 48) kWh (missing arrays are skipped, logged)
  min-max scale cluster train min/max over the 48 columns (track A train_nmf, track B train_ours; the CVAE scaling),
                only for the energy-score variant v5
Outputs
  outputs/<track>/scores_long.csv       one row per (model, cluster, date): all scores and ES variants
  outputs/<track>/scores_summary.json   medians/means per model x cluster next to the paper's Fig. 4 values, the
                                        "13% RMSE difference" analysis, the energy-score variant probe, the ES
                                        ranking claim and the distribution diagnostics, with verdicts
  outputs/figures/<track>_fig4_scores.png        Fig. 4 replica (RMSE / energy / variogram boxplots)
  outputs/figures/<track>_fig4_es_variants.png   energy-score variants vs the paper's medians
  outputs/figures/<track>_fig4_calibration.png   PIT histograms, spread-skill, coverage, roughness
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
if REPRO not in sys.path:
    sys.path.insert(0, REPRO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from lib import scores as sc  # noqa: E402

AUTHORS = REPRO + "/data/authors"  # copy of the authors' repo data/*.feather (byte-identical; read as data only)
FIGS = REPRO + "/outputs/figures"
MODELS = ("cvae", "gam")
H = 48
COLS = [f"consumption_{h:02d}" for h in range(H)]

# Paper Fig. 4, approximate medians read off the boxplots (spec section 3), clusters 0/1/2/3
PAPER_FIG4 = {
    "rmse": {"cvae": [0.020, 0.022, 0.018, 0.025], "gam": [0.015, 0.016, 0.015, 0.018]},
    "es": {"cvae": [0.150, 0.145, 0.170, 0.130], "gam": [0.155, 0.150, 0.175, 0.135]},
    "vg": {"cvae": [6.0, 7.0, 7.0, 10.0], "gam": [4.0, 4.0, 5.0, 5.0]},
}
# my own independent reading of the 4x-upscaled page image (pixel interpolation on the printed axes)
PAPER_FIG4_INDEPENDENT_READING = {
    "rmse": {"cvae": [0.019, 0.023, 0.018, 0.024], "gam": [0.015, 0.017, 0.015, 0.019]},
    "es": {"cvae": [0.151, 0.145, 0.170, 0.134], "gam": [0.155, 0.154, 0.169, 0.134]},
    "vg": {"cvae": [5.8, 7.0, 7.1, 10.6], "gam": [4.3, 4.6, 5.3, 6.0]},
}

# chart tokens (dataviz reference palette, light surface)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
MODEL_COLOR = {"cvae": "#2a78d6", "gam": "#eb6834"}
MODEL_LABEL = {"cvae": "CVAE", "gam": "GAM"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


# ------------------------------------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def track_paths(track: str) -> dict:
    t = track.upper()
    if t == "A":
        return {"obs": AUTHORS + "/test_nmf.feather", "train": AUTHORS + "/train_nmf.feather",
                "out": REPRO + "/outputs/trackA", "tag": "trackA"}
    if t == "B":
        return {"obs": REPRO + "/data/test_ours.parquet", "train": REPRO + "/data/train_ours.parquet",
                "out": REPRO + "/outputs/trackB", "tag": "trackB"}
    raise ValueError(track)


def read_table(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"])
    df["cluster"] = df["cluster"].astype(int)
    return df


def load_observations(path: str) -> tuple[dict, pd.Series]:
    te = read_table(path)
    obs = {}
    dates0 = None
    for c in sorted(te["cluster"].unique()):
        b = te.loc[te["cluster"] == c].sort_values("date", kind="stable")
        if not (b["date"].is_monotonic_increasing and b["date"].is_unique):
            raise ValueError(f"cluster {c}: test dates not unique/ascending")
        if dates0 is None:
            dates0 = b["date"].reset_index(drop=True)
        elif not b["date"].reset_index(drop=True).equals(dates0):
            raise ValueError(f"cluster {c}: test dates differ from cluster 0")
        Y = b[COLS].to_numpy(dtype=np.float64)
        if not np.isfinite(Y).all():
            raise ValueError(f"cluster {c}: non-finite observations")
        obs[int(c)] = Y
    return obs, dates0


def train_minmax(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    tr = read_table(path)
    out = {}
    for c in sorted(tr["cluster"].unique()):
        Y = tr.loc[tr["cluster"] == c, COLS].to_numpy(dtype=np.float64)
        out[int(c)] = (float(Y.min()), float(Y.max()))
    return out


def qstats(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {"median": float(np.median(x)), "mean": float(x.mean()), "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75)), "min": float(x.min()), "max": float(x.max()), "n": int(x.size)}


# ------------------------------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------------------------------
def score_block(S: np.ndarray, Y: np.ndarray, scale) -> dict:
    """All per-day scores for one model x cluster. S (N, D, 48), Y (D, 48)."""
    out = {"rmse_rootmean": sc.rmse_rootmean(S, Y), "rmse_euclid": sc.rmse_euclid(S, Y),
           "es_split": sc.energy_score_split(S, Y), "es_u": sc.energy_score_u(S, Y),
           "es_v": sc.energy_score_v(S, Y),
           "vg_p05_ordered": sc.variogram_score(S, Y, 0.5, "ordered"),
           "vg_p05_unordered": sc.variogram_score(S, Y, 0.5, "unordered")}
    terms = sc.energy_score_terms(S, Y)
    out["es_term_mean_dist_obs"] = terms["mean_dist_to_obs_all"]
    out["es_term_mean_pair_dist"] = terms["mean_pair_dist_u"]
    out["sample_sd_mean"] = S.std(0, ddof=1).mean(-1)
    out["obs_mean_kwh"] = Y.mean(-1)
    for k, v in sc.energy_score_variants(S, Y, scale=scale).items():
        out["esvar_" + k] = v
    return out


# ------------------------------------------------------------------------------------------------------------------
# analyses
# ------------------------------------------------------------------------------------------------------------------
def rmse_13pct(long: pd.DataFrame, clusters: list) -> dict:
    """Several readings of the abstract's '13% difference in RMSE' between the two generators."""
    piv = long.pivot_table(index=["cluster", "date"], columns="model", values="rmse_rootmean")
    piv = piv.dropna()
    c_, g_ = piv["cvae"].to_numpy(), piv["gam"].to_numpy()
    cands = {}

    def add(name, cv, gv, note=""):
        cands[name] = {"cvae": float(cv), "gam": float(gv), "diff_rel_to_cvae_pct": float(100 * (cv - gv) / cv),
                       "diff_rel_to_gam_pct": float(100 * (cv - gv) / gv), "note": note}
    add("mean_over_all_cluster_days", c_.mean(), g_.mean())
    add("median_over_all_cluster_days", np.median(c_), np.median(g_))
    med_c = [float(np.median(piv.loc[c, "cvae"])) for c in clusters]
    med_g = [float(np.median(piv.loc[c, "gam"])) for c in clusters]
    add("mean_of_per_cluster_medians", np.mean(med_c), np.mean(med_g))
    add("mean_of_per_cluster_means", np.mean([piv.loc[c, "cvae"].mean() for c in clusters]),
        np.mean([piv.loc[c, "gam"].mean() for c in clusters]))
    add("pooled_rms_over_cluster_days", math.sqrt(np.mean(c_ ** 2)), math.sqrt(np.mean(g_ ** 2)),
        "sqrt(mean over cluster-days of RMSE^2) = RMSE pooled over all half-hours")
    pc, pg = PAPER_FIG4["rmse"]["cvae"], PAPER_FIG4["rmse"]["gam"]
    add("paper_fig4_medians_mean_over_clusters", np.mean(pc), np.mean(pg), "the paper's own approximate medians")
    paired = {"mean_of_paired_rel_diff_rel_to_cvae_pct": float(100 * np.mean((c_ - g_) / c_)),
              "mean_of_paired_rel_diff_rel_to_gam_pct": float(100 * np.mean((c_ - g_) / g_)),
              "median_of_paired_rel_diff_rel_to_cvae_pct": float(100 * np.median((c_ - g_) / c_)),
              "median_of_paired_rel_diff_rel_to_gam_pct": float(100 * np.median((c_ - g_) / g_)),
              "frac_cluster_days_gam_better": float(np.mean(g_ < c_))}
    per_cluster = {str(c): {"median_cvae": med_c[i], "median_gam": med_g[i],
                            "diff_rel_to_cvae_pct": 100 * (med_c[i] - med_g[i]) / med_c[i],
                            "diff_rel_to_gam_pct": 100 * (med_c[i] - med_g[i]) / med_g[i]}
                   for i, c in enumerate(clusters)}
    allvals = [(k + ":rel_to_cvae", v["diff_rel_to_cvae_pct"]) for k, v in cands.items()] + \
              [(k + ":rel_to_gam", v["diff_rel_to_gam_pct"]) for k, v in cands.items()] + list(paired.items())[:4]
    near = [(k, round(v, 2)) for k, v in allvals if abs(v - 13.0) <= 2.0]
    closest = min(allvals, key=lambda kv: abs(kv[1] - 13.0))
    # alternative readings that are NOT a between-generator difference: RMSE relative to the consumption level
    lvl = long.drop_duplicates(["cluster", "date"]).set_index(["cluster", "date"])["obs_mean_kwh"]
    lvl = lvl.reindex(piv.index).to_numpy()
    alt = {}
    for m, arr in (("cvae", c_), ("gam", g_)):
        alt[m] = {"median_rmse_over_mean_consumption_pct": float(100 * np.median(arr / lvl)),
                  "mean_rmse_over_mean_consumption_pct": float(100 * arr.mean() / lvl.mean()),
                  "pooled_rmse_over_mean_consumption_pct": float(100 * math.sqrt(np.mean(arr ** 2)) / lvl.mean())}
    alt["difference_pct_points_median"] = (alt["cvae"]["median_rmse_over_mean_consumption_pct"]
                                           - alt["gam"]["median_rmse_over_mean_consumption_pct"])
    return {"definition": "RMSE = root-mean over the 48 half-hours of (mean of the 200 samples - observation), "
                          "per cluster-day; relative differences (CVAE - GAM)/CVAE and (CVAE - GAM)/GAM. The Euclidean "
                          "RMSE is sqrt(48) times larger per day, so every ratio is identical for it.",
            "candidates": cands, "paired": paired, "per_cluster_medians": per_cluster,
            "within_2pp_of_13pct": near, "closest_to_13pct": {"name": closest[0], "value_pct": closest[1]},
            "alternative_readings_rmse_relative_to_consumption_level": alt}


def es_variant_probe(long: pd.DataFrame, clusters: list) -> dict:
    paper = PAPER_FIG4["es"]
    rows = {}
    for v in sc.ES_VARIANTS:
        col = "esvar_" + v
        med = {m: [float(np.nanmedian(long.loc[(long.model == m) & (long.cluster == c), col]))
                   if ((long.model == m) & (long.cluster == c)).any() else float("nan") for c in clusters]
               for m in MODELS}
        cells = []
        for m in MODELS:
            for i, c in enumerate(clusters):
                if i < 4 and np.isfinite(med[m][i]) and med[m][i] > 0:
                    cells.append((m, c, med[m][i], paper[m][i]))
        r = np.array([math.log(o / p) for _, _, o, p in cells]) if cells else np.array([np.nan])
        rec = {"description": sc.ES_VARIANTS[v], "median": med,
               "ratio_to_paper": {m: [float(med[m][i] / paper[m][i]) if i < 4 else None
                                      for i in range(len(clusters))] for m in MODELS},
               "n_cells": len(cells), "mean_abs_log_ratio": float(np.mean(np.abs(r))),
               "max_abs_log_ratio": float(np.max(np.abs(r))),
               "n_cells_within_10pct": int(np.sum(np.abs(r) <= math.log(1.10))),
               "n_clusters_cvae_lower": int(sum(med["cvae"][i] < med["gam"][i] for i in range(len(clusters))
                                                if np.isfinite(med["cvae"][i]) and np.isfinite(med["gam"][i]))),
               "spearman_cluster_order_vs_paper": {
                   m: (float(stats.spearmanr(med[m][:4], paper[m]).statistic)
                       if len(cells) == 8 else None) for m in MODELS},
               # what a pure normalization (one common factor) could achieve
               "best_common_factor": float(math.exp(-np.mean(r))) if len(r) else None,
               "mean_abs_log_ratio_after_common_factor": float(np.mean(np.abs(r - np.mean(r))))}
        rows[v] = rec
    finite = {k: v for k, v in rows.items() if np.isfinite(v["mean_abs_log_ratio"]) and v["n_cells"] == 8}
    ranked = sorted(finite, key=lambda k: finite[k]["mean_abs_log_ratio"])
    reproduces = [k for k in ranked if finite[k]["n_cells_within_10pct"] == 8]
    return {"paper_es_medians": paper, "variants": rows, "ranked_by_mean_abs_log_ratio": ranked,
            "variants_with_all_8_cells_within_10pct": reproduces}


def es_level_analysis(long: pd.DataFrame, clusters: list) -> dict:
    """The paper's ES medians vs the clusters' mean test consumption and vs its own RMSE panel."""
    mean_y = [float(long.loc[long.cluster == c, "obs_mean_kwh"].mean()) for c in clusters]
    out = {"cluster_mean_test_consumption_kwh": mean_y}
    for m in MODELS:
        p = PAPER_FIG4["es"][m]
        ours = [float(np.median(long.loc[(long.model == m) & (long.cluster == c), "es_split"])) for c in clusters]
        out[m] = {"paper_es_over_mean_consumption": [p[i] / mean_y[i] for i in range(4)],
                  "our_correct_es_over_mean_consumption": [ours[i] / mean_y[i] for i in range(4)],
                  "paper_es_rank_order": list(np.argsort(p)[::-1].astype(int)),
                  "paper_rmse_rank_order": list(np.argsort(PAPER_FIG4["rmse"][m])[::-1].astype(int)),
                  "our_correct_es_rank_order": list(np.argsort(ours)[::-1].astype(int))}
    p_all = np.array(PAPER_FIG4["es"]["cvae"] + PAPER_FIG4["es"]["gam"]) / np.array(mean_y * 2)
    out["paper_es_over_mean_consumption_cv"] = float(p_all.std() / p_all.mean())
    eu = {m: [math.sqrt(H) * x for x in PAPER_FIG4["rmse"][m]] for m in MODELS}
    out["paper_cvae_es_minus_paper_euclidean_rmse"] = [PAPER_FIG4["es"]["cvae"][i] - eu["cvae"][i] for i in range(4)]
    out["note"] = ("For a generator with (almost) no spread every energy-score estimator equals ||mean - y|| = "
                   "sqrt(48) x root-mean RMSE. The paper's CVAE samples have too little variance by its own account, "
                   "so its ES should follow its RMSE panel (cluster order 3 > 1 > 0 > 2); the paper's ES panel has the "
                   "order 2 > 0 > 1 > 3, which is the order of the clusters' mean consumption.")
    return out


def es_claim(long: pd.DataFrame, clusters: list) -> dict:
    out = {}
    for c in clusters:
        a = long.loc[(long.model == "cvae") & (long.cluster == c)].sort_values("date")
        b = long.loc[(long.model == "gam") & (long.cluster == c)].sort_values("date")
        if a.empty or b.empty:
            continue
        x, y = a["es_split"].to_numpy(), b["es_split"].to_numpy()
        w = stats.wilcoxon(x, y)
        out[str(c)] = {"median_cvae": float(np.median(x)), "median_gam": float(np.median(y)),
                       "median_ratio_cvae_over_gam": float(np.median(x) / np.median(y)),
                       "frac_days_cvae_lower": float(np.mean(x < y)),
                       "wilcoxon_p": float(w.pvalue),
                       "u_stat_median_cvae": float(np.median(a["es_u"])), "u_stat_median_gam": float(np.median(b["es_u"]))}
    holds = [v["median_cvae"] < v["median_gam"] for v in out.values()]
    return {"per_cluster": out, "n_clusters_cvae_lower_correct_formula": int(sum(holds)),
            "claim_holds_correct_formula": bool(all(holds)) if holds else None}


def diagnostics(samples: dict, obs: dict, clusters: list) -> dict:
    out = {}
    for m in MODELS:
        for c in clusters:
            if (m, c) not in samples:
                continue
            out[f"{m}_c{c}"] = sc.distribution_diagnostics(samples[(m, c)], obs[c])
    # verdicts
    def get(m, key):
        return [out[f"{m}_c{c}"] for c in clusters if f"{m}_c{c}" in out]
    v = {}
    if all(f"{m}_c{c}" in out for m in MODELS for c in clusters):
        cv, gm = get("cvae", None), get("gam", None)
        ss_c = [d["spread_skill"]["ratio_calibrated"] for d in cv]
        ss_g = [d["spread_skill"]["ratio_calibrated"] for d in gm]
        cov90_c = [d["coverage_central90"] for d in cv]
        cov90_g = [d["coverage_central90"] for d in gm]
        cov50_g = [d["coverage_central50"] for d in gm]
        rough_c = [d["roughness"]["ratio_samples_to_observed"] for d in cv]
        rough_g = [d["roughness"]["ratio_samples_to_observed"] for d in gm]
        lag_cn = [d["lag1"]["across_samples_mean_over_day_slot"] for d in cv]
        lag_gn = [d["lag1"]["across_samples_mean_over_day_slot"] for d in gm]
        lag_cr = [d["lag1"]["residual_pooled"] for d in cv]
        lag_gr = [d["lag1"]["residual_pooled"] for d in gm]
        v["cvae_too_little_variance"] = {
            "holds": bool(all(x < 0.8 for x in ss_c) and all(x < 0.8 for x in cov90_c)),
            "spread_skill_ratio": ss_c, "coverage90": cov90_c}
        v["gam_too_much_variance"] = {
            "holds": bool(all(x > 1.1 for x in ss_g) and all(x > 0.9 for x in cov90_g)),
            "spread_skill_ratio": ss_g, "coverage90": cov90_g, "coverage50": cov50_g}
        v["cvae_consecutive_halfhours_very_correlated"] = {
            "holds": bool(all(a > max(b, r) for a, b, r in zip(lag_cn, lag_cr, lag_gr))),
            "cvae_across_sample_lag1": lag_cn,
            "observed_residual_lag1_vs_cvae_mean": lag_cr,
            "observed_residual_lag1_vs_gam_mean": lag_gr,
            "cvae_roughness_ratio_samples_to_obs": rough_c,
            "note": "holds = the CVAE's generated variation is more strongly lag-1 correlated than the actual "
                    "deviations of the observations from either generator's mean, in every cluster. The residual "
                    "w.r.t. the CVAE mean contains the CVAE's own smooth bias, hence its higher lag-1 value."}
        v["gam_erratic"] = {"holds": bool(all(x > 1.1 for x in rough_g)),
                            "gam_roughness_ratio_samples_to_obs": rough_g,
                            "gam_across_sample_lag1": lag_gn, "gam_residual_lag1": lag_gr,
                            "note": "GAM noise correlation = empirical residual correlation by construction; its "
                                    "erratic look comes from the noise amplitude, not from weaker correlation"}
        prof_c = [d["lag1"]["profile_demeaned_samples"] for d in cv]
        prof_g = [d["lag1"]["profile_demeaned_samples"] for d in gm]
        prof_o = [d["lag1"]["profile_demeaned_observed"] for d in cv]
        v["profile_lag1_autocorrelation"] = {"cvae_samples": prof_c, "gam_samples": prof_g, "observed": prof_o}
        v["real_behaviour_in_between"] = {
            "variance_holds": bool(all(a < 1 < b for a, b in zip(ss_c, ss_g))),
            "roughness_holds": bool(all(a < 1 < b for a, b in zip(rough_c, rough_g))),
            "profile_lag1_holds": bool(all(g < o < c for c, g, o in zip(prof_c, prof_g, prof_o))),
            "lag1_strictly_between_holds": bool(all(min(a, b) < r < max(a, b)
                                                    for a, b, r in zip(lag_cn, lag_gn, lag_gr))),
            "lag1_values": {"cvae_generated": lag_cn, "gam_generated": lag_gn,
                            "observed_residual_vs_gam_mean": lag_gr},
            "note": "variance: spread/skill CVAE < 1 < GAM; roughness: samples/observed CVAE < 1 < GAM; lag-1: "
                    "observed residual correlation strictly between the two generators' across-sample values"}
    return {"per_model_cluster": out, "verdicts": v,
            "definitions": {
                "spread_skill.ratio_calibrated": "sqrt(mean per-slot sample variance) / RMSE of the sample mean x "
                                                 "sqrt((N+1)/N); 1 = calibrated, <1 under-, >1 over-dispersed",
                "coverage_centralXX": "share of (day, half-hour) observations inside the central XX% sample interval",
                "pit": "rank PIT (#{S_i < y} + 0.5 + 0.5 ties)/(N+1); calibrated: mean 0.5, var 1/12, 10% in the "
                       "two 5% tails; U-shape (var > 1/12) = under-dispersion, hump (var < 1/12) = over-dispersion",
                "roughness": "mean |x_{h+1} - x_h| over the 47 within-day steps",
                "lag1.across_samples_mean_over_day_slot": "mean over (day, h) of corr over samples of (S_h, S_h+1)",
                "lag1.noise_pooled": "pooled lag-1 corr of S_i - mean_i S_i (generated deviations)",
                "lag1.residual_pooled": "pooled lag-1 corr of y - mean_i S_i (actual deviations from the "
                                        "generator's mean); the target for the generated deviations",
                "lag1.profile_demeaned_*": "pooled lag-1 autocorrelation of whole profiles after removing each "
                                           "profile's daily mean (samples vs observations)"}}


def conclusions(res: dict) -> dict:
    """Plain-language statements generated from the numbers (so that a Track B run states its own results)."""
    f4 = res["fig4_comparison"]
    out = {}
    try:
        rc = res["rmse_definition_check"]
        out["rmse"] = (f"Fig. 4 RMSE matches the root-mean over the 48 half-hours (mean ratio ours/paper CVAE "
                       f"{rc['cvae']['rootmean_mean_ratio']:.2f}, GAM {rc['gam']['rootmean_mean_ratio']:.2f}); "
                       f"eq. 12's Euclidean norm would be {rc['cvae']['euclid_mean_ratio']:.1f}x / "
                       f"{rc['gam']['euclid_mean_ratio']:.1f}x the paper's values.")
        vg = res["variogram_pairs_check"]
        out["variogram"] = (f"Variogram (p = 0.5) over all ordered pairs: mean ratio ours/paper CVAE "
                            f"{vg['cvae']['ordered_mean_ratio']:.2f}, GAM {vg['gam']['ordered_mean_ratio']:.2f}; "
                            f"unordered pairs would halve it ({vg['cvae']['unordered_mean_ratio']:.2f} / "
                            f"{vg['gam']['unordered_mean_ratio']:.2f}).")
        es = {m: f4["es"][m]["ratio_es_split_to_paper"] for m in MODELS}
        out["energy_score"] = (f"Energy score (eq. 14, correct): ratio ours/paper CVAE "
                               f"{min(es['cvae']):.2f}-{max(es['cvae']):.2f}, GAM {min(es['gam']):.2f}-"
                               f"{max(es['gam']):.2f}. Variants with all 8 cells within 10% of the paper: "
                               f"{res['es_variant_probe']['variants_with_all_8_cells_within_10pct'] or 'none'}; "
                               f"closest: {res['es_variant_probe']['ranked_by_mean_abs_log_ratio'][:2]}.")
        ec = res["es_claim_cvae_lower"]
        out["es_claim"] = (f"CVAE energy score lower than GAM's (correct formula) in "
                           f"{ec['n_clusters_cvae_lower_correct_formula']} of {len(ec['per_cluster'])} clusters "
                           f"(median ratio CVAE/GAM "
                           f"{min(v['median_ratio_cvae_over_gam'] for v in ec['per_cluster'].values()):.2f}-"
                           f"{max(v['median_ratio_cvae_over_gam'] for v in ec['per_cluster'].values()):.2f}).")
        r13 = res["rmse_13pct"]
        out["rmse_13pct"] = (f"RMSE difference CVAE vs GAM: median-based {r13['candidates']['median_over_all_cluster_days']['diff_rel_to_cvae_pct']:.1f}% "
                             f"(rel. to CVAE) / {r13['candidates']['median_over_all_cluster_days']['diff_rel_to_gam_pct']:.1f}% "
                             f"(rel. to GAM); readings within 2 pp of 13%: {r13['within_2pp_of_13pct'] or 'none'}.")
    except (KeyError, ValueError, TypeError) as exc:      # partial runs (missing model arrays)
        out["note"] = f"conclusions incomplete: {exc!r}"
    return out


# ------------------------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "font.size": 9, "axes.titlesize": 10, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def fig_scores(long, clusters, tag, path):
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    panels = [("rmse_rootmean", "rmse", "RMSE of the sample mean (kWh)\nroot-mean over 48 half-hours"),
              ("es_split", "es", "Energy score (kWh), eq. 14"),
              ("vg_p05_ordered", "vg", "Variogram score, p = 0.5\n(all ordered pairs)")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))
    for ax, (col, key, ylab) in zip(axes, panels):
        # y-limit: 1.3 x the highest upper whisker (or paper value); points above are counted, not drawn off-scale
        tops = [max(PAPER_FIG4[key][m]) for m in MODELS]
        for m in MODELS:
            for c in clusters:
                d = long.loc[(long.model == m) & (long.cluster == c), col].to_numpy()
                if len(d):
                    q1, q3 = np.quantile(d, [0.25, 0.75])
                    tops.append(d[d <= q3 + 1.5 * (q3 - q1)].max())
        ytop = 1.3 * max(tops)
        for j, m in enumerate(MODELS):
            data = [long.loc[(long.model == m) & (long.cluster == c), col].to_numpy() for c in clusters]
            pos = np.arange(len(clusters)) + (-0.19 if m == "cvae" else 0.19)
            if not any(len(d) for d in data):
                continue
            for p_, d in zip(pos, data):
                k_above = int((d > ytop).sum())
                if k_above:
                    ax.annotate(f"+{k_above}", (p_, ytop), xytext=(0, -9), textcoords="offset points",
                                ha="center", fontsize=7, color=INK2)
            bp = ax.boxplot([d if len(d) else [np.nan] for d in data], positions=pos, widths=0.32,
                            patch_artist=True, showfliers=True, whis=1.5,
                            medianprops={"color": INK, "linewidth": 1.2},
                            whiskerprops={"color": MODEL_COLOR[m], "linewidth": 0.8},
                            capprops={"color": MODEL_COLOR[m], "linewidth": 0.8},
                            flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": MODEL_COLOR[m],
                                        "markeredgecolor": "none", "alpha": 0.6})
            for b in bp["boxes"]:
                b.set_facecolor(MODEL_COLOR[m])
                b.set_alpha(0.45)
                b.set_edgecolor(MODEL_COLOR[m])
                b.set_linewidth(0.8)
            pv = PAPER_FIG4[key][m]
            ax.scatter(pos[:4], pv[: len(pos)], marker="D", s=22, color=INK, zorder=5, linewidths=0)
        ax.set_ylim(0, ytop)
        ax.set_xticks(np.arange(len(clusters)))
        ax.set_xticklabels([str(c) for c in clusters])
        ax.set_xlabel("Cluster")
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    handles = [Patch(facecolor=MODEL_COLOR[m], alpha=0.45, edgecolor=MODEL_COLOR[m], label=MODEL_LABEL[m] + " (ours)")
               for m in MODELS] + [Line2D([], [], marker="D", color=INK, linestyle="none", markersize=4.5,
                                          label="paper Fig. 4 median (read off the plot)")]
    fig.suptitle(f"{tag}: scores per test day (92 days per cluster, 200 samples per day); replica of paper Fig. 4. "
                 "'+k' = k days above the axis", y=0.985, fontsize=10, color=INK)
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_es_variants(probe, level, clusters, tag, path):
    plt = _plt()
    mean_y = np.array(level["cluster_mean_test_consumption_kwh"])
    k_fit = float(np.mean([PAPER_FIG4["es"][m][i] / mean_y[i] for m in MODELS for i in range(4)]))
    show = ["v1_correct_split", "v2_sq_spread", "v3_quarter_spread", "v4_first_term", "v5_minmax_scaled",
            "v6_eq14_typo_literal"]
    labels = {"v1_correct_split": "v1 correct (eq. 14)", "v2_sq_spread": "v2 squared-norm spread",
              "v3_quarter_spread": "v3 spread weight 1/4", "v4_first_term": "v4 first term only",
              "v5_minmax_scaled": "v5 on min-max scaled data", "v6_eq14_typo_literal": "v6 eq. 14 typo literal"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    x = np.arange(len(clusters))
    for ax, m in zip(axes, MODELS):
        ax.plot(x[:4], PAPER_FIG4["es"][m], color=INK, linewidth=2.0, marker="D", markersize=5,
                label="paper Fig. 4 median", zorder=5)
        for k, v in enumerate(show):
            med = probe["variants"][v]["median"][m]
            ax.plot(x, med, color=SERIES[k], linewidth=1.2, marker="o", markersize=4, label=labels[v])
        ax.plot(x, k_fit * mean_y, color=MUTED, linewidth=1.2, linestyle="--",
                label=f"{k_fit:.2f} x cluster mean consumption")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in clusters])
        ax.set_xlabel("Cluster")
        ax.set_title(f"{MODEL_LABEL[m]}: median energy score per variant", loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Median over test days (kWh; v5 unitless)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.suptitle(f"{tag}: which energy-score implementation could give the paper's Fig. 4 values?", y=0.99,
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_calibration(diag, clusters, tag, path):
    plt = _plt()
    d = diag["per_model_cluster"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    # (a) PIT histogram pooled over clusters
    ax = axes[0]
    edges = np.linspace(0, 1, 11)
    for m in MODELS:
        hs = [d[f"{m}_c{c}"]["pit"]["hist10"] for c in clusters if f"{m}_c{c}" in d]
        if not hs:
            continue
        h = np.mean(hs, axis=0)
        ax.stairs(h, edges, color=MODEL_COLOR[m], linewidth=1.6, label=MODEL_LABEL[m])
    ax.axhline(0.1, color=MUTED, linewidth=0.8, linestyle="--", label="calibrated (0.1)")
    ax.set_xlabel("PIT of the observation (10 bins)")
    ax.set_ylabel("Share of (day, half-hour)")
    ax.set_title("(a) PIT histogram, all clusters", loc="left")
    ax.legend(fontsize=8)
    # (b) spread-skill
    x = np.arange(len(clusters))
    for ax, key, title, ref, ylab in [
            (axes[1], ("spread_skill", "ratio_calibrated"), "(b) spread / skill", 1.0,
             "sample SD / RMSE of the sample mean"),
            (axes[2], ("coverage_central90",), "(c) coverage of the central 90% interval", 0.9, "share covered"),
            (axes[3], ("roughness", "ratio_samples_to_observed"), "(d) roughness, samples / observed", 1.0,
             "mean |consecutive difference| ratio")]:
        for k, m in enumerate(MODELS):
            vals = []
            for c in clusters:
                v = d.get(f"{m}_c{c}")
                if v is None:
                    vals.append(np.nan)
                    continue
                for kk in key:
                    v = v[kk]
                vals.append(v)
            ax.plot(x + (-0.08 if m == "cvae" else 0.08), vals, linestyle="none", marker="o", markersize=6,
                    color=MODEL_COLOR[m], label=MODEL_LABEL[m])
        ax.axhline(ref, color=MUTED, linewidth=0.8, linestyle="--", label="ideal")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in clusters])
        ax.set_xlabel("Cluster")
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{tag}: calibration and smoothness of the generated samples (test days)", y=1.0, fontsize=10,
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["A", "B", "a", "b"])
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    P = track_paths(args.track)
    t0 = time.perf_counter()
    if not os.path.exists(P["obs"]):
        log(f"[{P['tag']}] observation file {P['obs']} does not exist yet: nothing scored, nothing written.")
        return 0
    obs, dates = load_observations(P["obs"])
    clusters = sorted(obs)
    D = len(dates)
    date_check = "test_dates.csv not found"
    td_path = P["out"] + "/test_dates.csv"
    if os.path.exists(td_path):
        td = pd.read_csv(td_path)
        ok = td["date"].tolist() == dates.dt.strftime("%Y-%m-%d").tolist()
        if not ok:
            raise ValueError(f"test dates in {P['obs']} do not match {td_path}")
        date_check = f"matches {td_path} ({len(td)} days)"
    log(f"[{P['tag']}] {len(clusters)} clusters x {D} test days ({dates.iloc[0].date()} .. {dates.iloc[-1].date()}); "
        f"dates {date_check}")
    mm = train_minmax(P["train"])
    samples, missing = {}, []
    for m in MODELS:
        for c in clusters:
            p = P["out"] + f"/{m}_samples_c{c}.npy"
            if not os.path.exists(p):
                missing.append(p)
                continue
            S = np.load(p)
            if S.ndim != 3 or S.shape[1:] != (D, H) or not np.isfinite(S).all():
                raise ValueError(f"{p}: bad shape {S.shape} or non-finite values")
            samples[(m, c)] = S
    for p in missing:
        log(f"[{P['tag']}] WARNING missing sample array {p} (skipped)")
    if not samples:
        log(f"[{P['tag']}] no sample arrays found: nothing scored.")
        return 0
    rows = []
    for (m, c), S in samples.items():
        tb = time.perf_counter()
        res = score_block(S, obs[c], mm.get(c))
        for d in range(D):
            rows.append({"model": m, "cluster": c, "date": dates.iloc[d].strftime("%Y-%m-%d"),
                         **{k: float(v[d]) for k, v in res.items()}})
        log(f"[{P['tag']}] {m} c{c}: N={S.shape[0]} medians RMSE {np.median(res['rmse_rootmean']):.4f} kWh "
            f"(Euclid {np.median(res['rmse_euclid']):.4f}), ES {np.median(res['es_split']):.4f} "
            f"(U {np.median(res['es_u']):.4f}), VG {np.median(res['vg_p05_ordered']):.2f} "
            f"[{time.perf_counter() - tb:.1f}s]")
    long = pd.DataFrame(rows).sort_values(["model", "cluster", "date"]).reset_index(drop=True)
    long.to_csv(P["out"] + "/scores_long.csv", index=False)

    # independent check of the scores on 3 random cells with a direct (loop) implementation
    rng = np.random.default_rng(1)
    checks = []
    for (m, c) in list(samples)[:4]:
        d = int(rng.integers(0, D))
        S, y = samples[(m, c)][:, d], obs[c][d]
        N = S.shape[0]
        es_direct = (np.mean([np.linalg.norm(S[i] - y) for i in range(N // 2)])
                     - 0.5 * np.mean([np.linalg.norm(S[i] - S[N // 2 + i]) for i in range(N // 2)]))
        vg_direct = sum((abs(y[h] - y[k]) ** 0.5 - np.mean(np.abs(S[:, h] - S[:, k]) ** 0.5)) ** 2
                        for h in range(H) for k in range(H))
        row = long.loc[(long.model == m) & (long.cluster == c)].iloc[d]
        checks.append({"model": m, "cluster": c, "day": d, "es_direct": es_direct, "es_table": row["es_split"],
                       "vg_direct": vg_direct, "vg_table": row["vg_p05_ordered"]})
        assert abs(es_direct - row["es_split"]) < 1e-12 and abs(vg_direct - row["vg_p05_ordered"]) < 1e-9

    # summary
    summ = {}
    score_cols = ["rmse_rootmean", "rmse_euclid", "es_split", "es_u", "es_v", "vg_p05_ordered", "vg_p05_unordered",
                  "es_term_mean_dist_obs", "es_term_mean_pair_dist", "sample_sd_mean"]
    for m in MODELS:
        for c in clusters:
            b = long.loc[(long.model == m) & (long.cluster == c)]
            if b.empty:
                continue
            summ[f"{m}_c{c}"] = {k: qstats(b[k]) for k in score_cols}
    fig4 = {}
    for key, col, alt in [("rmse", "rmse_rootmean", "rmse_euclid"), ("es", "es_split", "es_u"),
                          ("vg", "vg_p05_ordered", "vg_p05_unordered")]:
        fig4[key] = {}
        for m in MODELS:
            ours = [summ.get(f"{m}_c{c}", {}).get(col, {}).get("median", float("nan")) for c in clusters]
            ours_alt = [summ.get(f"{m}_c{c}", {}).get(alt, {}).get("median", float("nan")) for c in clusters]
            pv = PAPER_FIG4[key][m]
            fig4[key][m] = {"paper_approx_median": pv, "paper_independent_reading":
                            PAPER_FIG4_INDEPENDENT_READING[key][m],
                            f"ours_median_{col}": ours, f"ours_median_{alt}": ours_alt,
                            f"ratio_{col}_to_paper": [o / p for o, p in zip(ours, pv)],
                            f"ratio_{alt}_to_paper": [o / p for o, p in zip(ours_alt, pv)]}
    rmse_def = {}
    for m in MODELS:
        r1 = np.array(fig4["rmse"][m]["ratio_rmse_rootmean_to_paper"])
        r2 = np.array(fig4["rmse"][m]["ratio_rmse_euclid_to_paper"])
        rmse_def[m] = {"rootmean_mean_ratio": float(np.mean(r1)), "euclid_mean_ratio": float(np.mean(r2))}
    rmse_def["matches"] = "root-mean" if all(abs(rmse_def[m]["rootmean_mean_ratio"] - 1) < 0.15 for m in MODELS) \
        else "unclear"
    vg_def = {m: {"ordered_mean_ratio": float(np.mean(fig4["vg"][m]["ratio_vg_p05_ordered_to_paper"])),
                  "unordered_mean_ratio": float(np.mean(fig4["vg"][m]["ratio_vg_p05_unordered_to_paper"]))}
              for m in MODELS}

    res = {"track": P["tag"], "n_samples": int(next(iter(samples.values())).shape[0]), "n_test_days": D,
           "clusters": clusters, "test_dates_check": date_check, "missing_sample_arrays": missing,
           "minmax_scale_for_v5": {str(k): v for k, v in mm.items()},
           "definitions": {
               "rmse_rootmean": "sqrt(mean_h (mean_i S_ih - y_h)^2)",
               "rmse_euclid": "||mean_i S_i - y||_2 (eq. 12 as printed) = sqrt(48) x rmse_rootmean",
               "es_split": "eq. 14 as written: mean_{i<=N/2} ||S_i - y|| - 0.5 mean_{i<=N/2} ||S_i - S_{N/2+i}||",
               "es_u": "U-statistic: mean_i ||S_i - y|| - 1/(2N(N-1)) sum_{i!=j} ||S_i - S_j||",
               "es_v": "V-statistic (scoringRules::es_sample)",
               "vg_p05_ordered": "eq. 16, p = 0.5, sum over all ORDERED pairs (h, h'); h = h' terms are 0",
               "vg_p05_unordered": "same over h < h' only = exactly half of the ordered value",
               "esvar_*": "energy-score variants, see lib/scores.py ES_VARIANTS"},
           "paper_fig4_source": "spec section 3 (approximate medians read off the boxplots); "
                                "'paper_independent_reading' = my own reading of the upscaled page image",
           "summary": summ, "fig4_comparison": fig4,
           "rmse_definition_check": rmse_def, "variogram_pairs_check": vg_def,
           "direct_recomputation_checks": checks}
    res["rmse_13pct"] = rmse_13pct(long, clusters)
    res["es_variant_probe"] = es_variant_probe(long, clusters)
    res["es_level_analysis"] = es_level_analysis(long, clusters)
    res["es_claim_cvae_lower"] = es_claim(long, clusters)
    diag = diagnostics(samples, obs, clusters)
    res["distribution_diagnostics"] = diag
    res["conclusions"] = conclusions(res)
    res["runtime_s"] = time.perf_counter() - t0
    with open(P["out"] + "/scores_summary.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"[{P['tag']}] wrote scores_long.csv ({len(long)} rows) and scores_summary.json")

    r13 = res["rmse_13pct"]
    log(f"[{P['tag']}] 13% check: closest = {r13['closest_to_13pct']}; within 2 pp: {r13['within_2pp_of_13pct']}")
    pr = res["es_variant_probe"]
    log(f"[{P['tag']}] ES variants ranked: {pr['ranked_by_mean_abs_log_ratio'][:5]}; all 8 cells within 10%: "
        f"{pr['variants_with_all_8_cells_within_10pct']}")
    for k, v in diag["verdicts"].items():
        log(f"[{P['tag']}] verdict {k}: {v.get('holds', v)}")

    if not args.no_figures:
        os.makedirs(FIGS, exist_ok=True)
        fig_scores(long, clusters, P["tag"], FIGS + f"/{P['tag']}_fig4_scores.png")
        fig_es_variants(res["es_variant_probe"], res["es_level_analysis"], clusters, P["tag"],
                        FIGS + f"/{P['tag']}_fig4_es_variants.png")
        fig_calibration(diag, clusters, P["tag"], FIGS + f"/{P['tag']}_fig4_calibration.png")
        log(f"[{P['tag']}] figures written to {FIGS}")
    log(f"[{P['tag']}] done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
