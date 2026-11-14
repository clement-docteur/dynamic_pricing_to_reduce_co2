"""Build outputs/comparison_AB.json (paper vs Track A vs Track B for every Fig. 4 score, distribution diagnostic and
tariff-effect metric; claim verdicts for Track B; attribution of A/B differences to the data or to training
randomness) and the figures outputs/figures/trackB_fig4_AB_comparison.png, trackB_effects_AB_comparison.png.

Inputs (all produced before): outputs/track{A,B}/scores_summary.json, effects_summary.json, mechanism_summary.json,
effects_all_restarts_regions.csv; outputs/trackB/scores_independent_check.json, scores_restarts_AB.csv,
mechanism_restarts_AB.csv, mechanism_restarts_AB_summary.json, mechanism_data_check_AB.json.
Run: python -P build_comparison.py [--sections all|effects]  (from PROJECT or REPRO)
  all      (default) rebuild the whole file and both figures
  effects  rebuild only what depends on the tariff scenarios: 'tariff_effects', the tariff claims (direction, side
           effect, rebound, heterogeneity), the headline, the tariff checks and trackB_effects_AB_comparison.png;
           'fig4_scores', 'distribution_diagnostics' and the score claims are kept from the existing file
Scenario windows: lib.effects (Low 9-19, High 39-44; REPRO_SPEC section 7). The phase-2 build used 9-18 / 39-43:
outputs/archive_windows_9-18_39-43/comparison_AB.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
sys.path.insert(0, REPRO)
import lib.effects as LE  # noqa: E402  (scenario windows and region names only)

KEY_LREB, KEY_HPOST = LE.KEY_LOW_REBOUND, LE.KEY_HIGH_POST_REBOUND        # 'rebound_20_31', 'rebound_post_45_47'
LREB_SPAN = KEY_LREB.replace("rebound_", "")                               # '20_31'
LSPAN = f"{LE.LOW_WINDOW[0]}_{LE.LOW_WINDOW[-1]}"                          # '9_19'
HSPAN = f"{LE.HIGH_WINDOW[0]}_{LE.HIGH_WINDOW[-1]}"                        # '39_44'
H_POST = LE.scenario_regions("High")["post4"]                              # 45-47 (3 slots)
TARIFF_CLAIMS = ("C_tariff_direction_B", "C_cvae_side_effect_B", "C_cvae_rebound_B", "C_cluster_heterogeneity_B")
A, B = REPRO + "/outputs/trackA", REPRO + "/outputs/trackB"
FIGS = REPRO + "/outputs/figures"
OUTJ = REPRO + "/outputs/comparison_AB.json"
CL = [0, 1, 2, 3]
MODELS = ("cvae", "gam")
SIZES = {"A": [325, 295, 229, 158], "B": [285, 299, 299, 149]}


def J(p):
    with open(p) as f:
        return json.load(f)


sA, sB = J(A + "/scores_summary.json"), J(B + "/scores_summary.json")
eA, eB = J(A + "/effects_summary.json"), J(B + "/effects_summary.json")
mA, mB = J(A + "/mechanism_summary.json"), J(B + "/mechanism_summary.json")
IC = J(B + "/scores_independent_check.json")
MR = J(B + "/mechanism_restarts_AB_summary.json")
MRDF = pd.read_csv(B + "/mechanism_restarts_AB.csv")
DC = J(B + "/mechanism_data_check_AB.json")
RA, RB = pd.read_csv(A + "/effects_all_restarts_regions.csv"), pd.read_csv(B + "/effects_all_restarts_regions.csv")
RS = pd.read_csv(B + "/scores_restarts_AB.csv")
S = {"A": sA, "B": sB}
PAT = {"A": J(A + "/mechanism_cvae_patterns.json"), "B": J(B + "/mechanism_cvae_patterns.json")}
E = {"A": eA, "B": eB}
M = {"A": mA, "B": mB}
RR = {"A": RA, "B": RB}


def clean(o):
    """JSON-safe: numpy -> python, NaN/inf -> None, rounded floats."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        if not math.isfinite(x):
            return None
        return float(f"{x:.6g}")
    return o


# ---------------------------------------------------------------------------------------------------------------
# attribution rules
# ---------------------------------------------------------------------------------------------------------------
def attribute_cvae(selA, selB, ra, rb) -> dict:
    """Selected-model difference split into a shift of the whole 50-restart distribution (data) and the rest
    (which restart was selected = training randomness)."""
    ra, rb = np.asarray(ra, float), np.asarray(rb, float)
    p = float(stats.mannwhitneyu(ra, rb).pvalue)
    sd = math.sqrt((ra.var(ddof=1) + rb.var(ddof=1)) / 2)
    d_sel, d_med = selB - selA, float(np.median(rb) - np.median(ra))
    z_sel = d_sel / sd if sd > 0 else float("nan")
    resid = (d_sel - d_med) / sd if sd > 0 else float("nan")
    if p >= 0.05:
        label = "no material difference" if abs(z_sel) < 1 else "training randomness (restart selection)"
    else:
        label = "data" if abs(resid) < 1 else "data + training randomness (restart selection)"
    return {"label": label, "selected_B_minus_A": d_sel, "restart_median_B_minus_A": d_med,
            "pooled_restart_sd": sd, "selected_diff_in_restart_sd": z_sel,
            "selection_component_in_restart_sd": resid, "mannwhitney_p_restarts_A_vs_B": p,
            "B_selected_percentile_in_B_restarts": float(100 * np.mean(rb < selB)),
            "A_selected_percentile_in_A_restarts": float(100 * np.mean(ra < selA))}


def attribute_ci(diff_ci) -> str:
    lo, hi = diff_ci
    return "data (B-A CI excludes 0)" if (lo > 0 or hi < 0) else "no significant difference (B-A CI includes 0)"


def ci_overlap(ca, cb) -> bool:
    return not (ca[1] < cb[0] or cb[1] < ca[0])


# ---------------------------------------------------------------------------------------------------------------
# 1. Fig. 4 scores
# ---------------------------------------------------------------------------------------------------------------
FIG4 = [("rmse_rootmean", "rmse", "rmse", "rmse_median", "RMSE of the sample mean, root-mean over 48 half-hours (kWh)"),
        ("es_split", "es", "es_split", "es_split_median", "energy score eq. 14 (kWh), correct split estimator"),
        ("vg_p05_ordered", "vg", "vg_ord", "vg_ord_median", "variogram score p = 0.5, all ordered pairs")]
ALT = [("rmse_euclid", "rmse", "eq. 12 Euclidean norm (= sqrt(48) x root-mean)"),
       ("es_u", "es", "energy score, U-statistic over all pairs"),
       ("vg_p05_unordered", "vg", "variogram score over h < h' only (= half of ordered)")]


def fig4_section() -> dict:
    out = {}
    boot = IC["stages"]["paired_bootstrap_days"]
    comp = IC["stages"]["restarts"]["comparison"]
    gmc = IC["stages"]["gam_mc"]
    for col, key, mk, rk, desc in FIG4:
        sec = {"definition": desc}
        for m in MODELS:
            pv = sA["fig4_comparison"][key][m]["paper_approx_median"]
            pi = sA["fig4_comparison"][key][m]["paper_independent_reading"]
            cells = {}
            for c in CL:
                a, b = (S[t]["summary"][f"{m}_c{c}"][col] for t in ("A", "B"))
                bt = boot[f"{m}_c{c}_{mk}"]
                cell = {"paper_approx_median": pv[c], "paper_independent_reading": pi[c],
                        "trackA_median": a["median"], "trackA_q25_q75": [a["q25"], a["q75"]],
                        "trackA_median_ci95_day_bootstrap": median_ci("A", m, c, col),
                        "trackB_median": b["median"], "trackB_q25_q75": [b["q25"], b["q75"]],
                        "trackB_median_ci95_day_bootstrap": median_ci("B", m, c, col),
                        "paper_inside_trackB_median_ci": bool(median_ci("B", m, c, col)[0] <= pv[c] <=
                                                              median_ci("B", m, c, col)[1]),
                        "ratio_A_to_paper": a["median"] / pv[c], "ratio_B_to_paper": b["median"] / pv[c],
                        "B_minus_A": b["median"] - a["median"], "B_minus_A_ci95_paired_day_bootstrap": bt["diff_ci95"],
                        "ratio_B_over_A": bt["ratio_B_over_A"], "ratio_B_over_A_ci95": bt["ratio_ci95"]}
                if m == "cvae":
                    rc = comp[f"c{c}_{rk}"]
                    ra = RS.loc[(RS.track == "A") & (RS.cluster == c), rk]
                    rb = RS.loc[(RS.track == "B") & (RS.cluster == c), rk]
                    cell["restarts_50"] = {"A_median": rc["A_restarts_median"], "A_q10_q90": rc["A_restarts_q10_q90"],
                                           "B_median": rc["B_restarts_median"], "B_q10_q90": rc["B_restarts_q10_q90"],
                                           "A_selected": rc["A_selected"], "B_selected": rc["B_selected"]}
                    cell["attribution"] = attribute_cvae(rc["A_selected"], rc["B_selected"], ra, rb)
                else:
                    cell["gam_monte_carlo_sd_of_median"] = {t: gmc[f"{t}_c{c}"][f"{mk}_mc_sd"] for t in ("A", "B")}
                    cell["attribution"] = {"label": attribute_ci(bt["diff_ci95"]).split(" (")[0],
                                           "rule": "GAM fit is deterministic: a B-A difference whose paired day-"
                                                   "bootstrap CI excludes 0 comes from the data; Monte-Carlo SD "
                                                   "of the median listed for scale",
                                           "detail": attribute_ci(bt["diff_ci95"])}
                cells[str(c)] = cell
            sec[m] = cells
            sec[m + "_summary"] = {
                "trackA_ratio_to_paper_range": [min(cells[str(c)]["ratio_A_to_paper"] for c in CL),
                                                max(cells[str(c)]["ratio_A_to_paper"] for c in CL)],
                "trackB_ratio_to_paper_range": [min(cells[str(c)]["ratio_B_to_paper"] for c in CL),
                                                max(cells[str(c)]["ratio_B_to_paper"] for c in CL)],
                "trackA_mean_ratio": float(np.mean([cells[str(c)]["ratio_A_to_paper"] for c in CL])),
                "trackB_mean_ratio": float(np.mean([cells[str(c)]["ratio_B_to_paper"] for c in CL])),
                "spearman_cluster_order_B_vs_paper": float(stats.spearmanr(
                    [cells[str(c)]["trackB_median"] for c in CL], pv).statistic),
                "spearman_cluster_order_A_vs_paper": float(stats.spearmanr(
                    [cells[str(c)]["trackA_median"] for c in CL], pv).statistic)}
        sec["gam_below_cvae_all_clusters"] = {t: all(S[t]["summary"][f"gam_c{c}"][col]["median"] <
                                                     S[t]["summary"][f"cvae_c{c}"][col]["median"] for c in CL)
                                              for t in ("A", "B")}
        sec["paper_gam_below_cvae_all_clusters"] = all(
            sA["fig4_comparison"][key]["gam"]["paper_approx_median"][c] <
            sA["fig4_comparison"][key]["cvae"]["paper_approx_median"][c] for c in CL)
        gap = {}
        for c in CL:
            g = boot[f"gap_c{c}_{mk}"]
            gap[str(c)] = {"cvae_over_gam_A": g["cvae_over_gam_A"], "ci95_A": g["ci95_A"],
                           "cvae_over_gam_B": g["cvae_over_gam_B"], "ci95_B": g["ci95_B"],
                           "B_minus_A_ci95": g["diff_B_minus_A_ci95"],
                           "paper_cvae_over_gam": sA["fig4_comparison"][key]["cvae"]["paper_approx_median"][c] /
                           sA["fig4_comparison"][key]["gam"]["paper_approx_median"][c]}
        sec["cvae_over_gam_median_ratio"] = gap
        out[key] = sec
    # alternative definitions
    alt = {}
    for col, key, desc in ALT:
        alt[col] = {"definition": desc}
        for m in MODELS:
            pv = sA["fig4_comparison"][key][m]["paper_approx_median"]
            alt[col][m] = {str(c): {"paper": pv[c], "trackA_median": S["A"]["summary"][f"{m}_c{c}"][col]["median"],
                                    "trackB_median": S["B"]["summary"][f"{m}_c{c}"][col]["median"],
                                    "ratio_A_to_paper": S["A"]["summary"][f"{m}_c{c}"][col]["median"] / pv[c],
                                    "ratio_B_to_paper": S["B"]["summary"][f"{m}_c{c}"][col]["median"] / pv[c]}
                           for c in CL}
    out["alternative_definitions"] = alt
    # how much of the GAM score change could cluster size alone explain (aggregate noise ~ 1/sqrt(n households))?
    out["cluster_size_noise_scaling"] = {
        str(c): {"n_households_A": SIZES["A"][c], "n_households_B": SIZES["B"][c],
                 "sqrt_nA_over_nB": math.sqrt(SIZES["A"][c] / SIZES["B"][c]),
                 "gam_rmse_ratio_B_over_A": out["rmse"]["gam"][str(c)]["ratio_B_over_A"],
                 "cvae_rmse_ratio_B_over_A": out["rmse"]["cvae"][str(c)]["ratio_B_over_A"],
                 "observed_roughness_A": sA["distribution_diagnostics"]["per_model_cluster"][f"gam_c{c}"]["roughness"]
                 ["observed"],
                 "observed_roughness_B": sB["distribution_diagnostics"]["per_model_cluster"][f"gam_c{c}"]["roughness"]
                 ["observed"]} for c in CL}
    out["cluster_size_noise_scaling"]["note"] = (
        "Track B cluster 2 (299 households) mixes the authors' clusters 2 and 1 (213 + 70 of the shared 1,007); "
        "sqrt(229/299) = 0.875 matches its GAM RMSE ratio 0.87, but cluster size alone does not explain clusters 0-1")
    out["definition_checks"] = {t: {"rmse": S[t]["rmse_definition_check"], "variogram": S[t]["variogram_pairs_check"]}
                                for t in ("A", "B")}
    # energy-score variant probe
    probe = {}
    for t in ("A", "B"):
        pr = S[t]["es_variant_probe"]
        probe[t] = {"ranked_by_mean_abs_log_ratio": pr["ranked_by_mean_abs_log_ratio"][:6],
                    "variants_with_all_8_cells_within_10pct": pr["variants_with_all_8_cells_within_10pct"],
                    "per_variant": {v: {"mean_abs_log_ratio": r["mean_abs_log_ratio"],
                                        "mean_abs_pct_error": 100 * (math.exp(r["mean_abs_log_ratio"]) - 1),
                                        "n_cells_within_10pct": r["n_cells_within_10pct"],
                                        "n_clusters_cvae_lower": r["n_clusters_cvae_lower"],
                                        "spearman_cluster_order_vs_paper": r["spearman_cluster_order_vs_paper"],
                                        "median": r["median"]}
                                    for v, r in pr["variants"].items()}}
    out["energy_score_variant_probe"] = probe
    lvl = {}
    for t in ("A", "B"):
        la = S[t]["es_level_analysis"]
        lvl[t] = {"cluster_mean_test_consumption_kwh": la["cluster_mean_test_consumption_kwh"],
                  "paper_es_over_this_tracks_mean_consumption_cvae": la["cvae"]["paper_es_over_mean_consumption"],
                  "paper_es_over_this_tracks_mean_consumption_gam": la["gam"]["paper_es_over_mean_consumption"],
                  "cv_of_paper_es_over_mean_consumption": la["paper_es_over_mean_consumption_cv"],
                  "our_correct_es_rank_order_cvae": la["cvae"]["our_correct_es_rank_order"],
                  "our_correct_es_rank_order_gam": la["gam"]["our_correct_es_rank_order"],
                  "paper_es_rank_order": la["cvae"]["paper_es_rank_order"],
                  "paper_rmse_rank_order_cvae": la["cvae"]["paper_rmse_rank_order"]}
    lvl["note"] = ("The paper's ES medians are proportional to the authors' cluster mean consumption (CV 2.4% with "
                   "Track A's means) and less so to ours (CV with Track B's means listed); the correct ES follows "
                   "the RMSE order for a nearly deterministic CVAE.")
    out["energy_score_level_analysis"] = lvl
    # 13% and ES claim
    r13 = {}
    for t in ("A", "B"):
        x = S[t]["rmse_13pct"]
        r13[t] = {"candidates_rel_to_cvae_pct": {k: v["diff_rel_to_cvae_pct"] for k, v in x["candidates"].items()},
                  "candidates_rel_to_gam_pct": {k: v["diff_rel_to_gam_pct"] for k, v in x["candidates"].items()},
                  "paired": x["paired"], "per_cluster_median_diff_rel_to_cvae_pct":
                      {k: v["diff_rel_to_cvae_pct"] for k, v in x["per_cluster_medians"].items()},
                  "within_2pp_of_13pct": x["within_2pp_of_13pct"], "closest_to_13pct": x["closest_to_13pct"]}
        vals = list(r13[t]["candidates_rel_to_cvae_pct"].values()) + list(r13[t]["candidates_rel_to_gam_pct"].values())
        vals = [v for k, v in list(r13[t]["candidates_rel_to_cvae_pct"].items()) +
                list(r13[t]["candidates_rel_to_gam_pct"].items()) if not k.startswith("paper")]
        vals += [x["paired"][k] for k in list(x["paired"])[:4]]
        r13[t]["range_of_all_our_readings_pct"] = [min(vals), max(vals)]
    r13["paper"] = {"claim_pct": 13.0, "paper_fig4_medians_imply_rel_to_cvae_pct":
                    sA["rmse_13pct"]["candidates"]["paper_fig4_medians_mean_over_clusters"]["diff_rel_to_cvae_pct"],
                    "paper_fig4_medians_imply_rel_to_gam_pct":
                    sA["rmse_13pct"]["candidates"]["paper_fig4_medians_mean_over_clusters"]["diff_rel_to_gam_pct"]}
    r13["independent_recomputation"] = IC["rmse_13pct_recomputed"]
    out["rmse_13pct"] = r13
    esc = {}
    for t in ("A", "B"):
        x = S[t]["es_claim_cvae_lower"]
        esc[t] = {"n_clusters_cvae_lower": x["n_clusters_cvae_lower_correct_formula"],
                  "per_cluster": {k: {"median_cvae": v["median_cvae"], "median_gam": v["median_gam"],
                                      "ratio_cvae_over_gam": v["median_ratio_cvae_over_gam"],
                                      "frac_days_cvae_lower": v["frac_days_cvae_lower"], "wilcoxon_p": v["wilcoxon_p"]}
                                  for k, v in x["per_cluster"].items()}}
    esc["paper"] = {"claim": "energy score slightly lower for the CVAE than for the GAM",
                    "paper_fig4_cvae_over_gam": [a / b for a, b in zip(sA["fig4_comparison"]["es"]["cvae"]
                                                                       ["paper_approx_median"],
                                                                       sA["fig4_comparison"]["es"]["gam"]
                                                                       ["paper_approx_median"])]}
    out["es_cvae_lower_claim"] = esc
    return out


# ---------------------------------------------------------------------------------------------------------------
# 2. distribution diagnostics
# ---------------------------------------------------------------------------------------------------------------
DIAG = [("spread_skill_ratio_calibrated", ("spread_skill", "ratio_calibrated"), "spread_skill",
         "sample SD / RMSE of the sample mean x sqrt((N+1)/N); 1 = calibrated"),
        ("coverage_central90", ("coverage_central90",), "coverage90", "share of observations in the central 90% interval"),
        ("coverage_central50", ("coverage_central50",), "coverage50", "share of observations in the central 50% interval"),
        ("pit_variance", ("pit", "var"), "pit_var", "variance of the rank PIT (uniform = 0.0833)"),
        ("pit_frac_outside_sample_range", ("pit", "frac_outside_sample_range"), "frac_outside_range",
         "share of observations outside the range of the 200 samples"),
        ("roughness_ratio_samples_to_observed", ("roughness", "ratio_samples_to_observed"), "roughness_ratio",
         "mean |x_h+1 - x_h| of the samples / of the observations"),
        ("lag1_across_samples", ("lag1", "across_samples_mean_over_day_slot"), "lag1_across_samples",
         "mean over (day, h) of corr over samples of (S_h, S_h+1)"),
        ("lag1_observed_residual_vs_own_mean", ("lag1", "residual_pooled"), "lag1_resid_vs_own_mean",
         "lag-1 corr of y - this generator's sample mean"),
        ("profile_lag1_samples", ("lag1", "profile_demeaned_samples"), "profile_lag1_samples",
         "lag-1 autocorrelation of whole sample profiles (daily mean removed)"),
        ("profile_lag1_observed", ("lag1", "profile_demeaned_observed"), "profile_lag1_observed",
         "same for the observed profiles"),
        ("roughness_observed_kwh", ("roughness", "observed"), None, "mean |y_h+1 - y_h| of the observed test profiles"),
        ("sample_sd_kwh_mean", None, "sample_sd_kwh_mean", "mean per-slot sample SD (kWh)")]
PAPER_DIAG = {"spread_skill_ratio_calibrated": "CVAE too little variance (< 1), GAM too much (> 1); real in between",
              "coverage_central90": "implied: CVAE under-covers, GAM over-covers",
              "roughness_ratio_samples_to_observed": "CVAE consecutive half-hours very correlated (smooth), GAM "
                                                      "erratic; real in between",
              "lag1_across_samples": "CVAE very correlated from one half-hour to the next; GAM less (erratic)",
              "profile_lag1_samples": "real correlations lie in between"}


def diag_section() -> dict:
    out = {}
    myd = IC["my_diagnostics"]
    comp = IC["stages"]["restarts"]["comparison"]
    for name, path, rk, desc in DIAG:
        sec = {"definition": desc, "paper_claim": PAPER_DIAG.get(name)}
        for m in MODELS:
            cells = {}
            for c in CL:
                vals = {}
                for t in ("A", "B"):
                    if path is None:
                        v = myd[f"{t}_{m}_c{c}"]["sample_sd_kwh_mean"]
                    else:
                        v = S[t]["distribution_diagnostics"]["per_model_cluster"][f"{m}_c{c}"]
                        for k in path:
                            v = v[k]
                    vals[t] = v
                cell = {"trackA": vals["A"], "trackB": vals["B"], "B_minus_A": vals["B"] - vals["A"]}
                rel = abs(vals["B"] - vals["A"]) / max(abs(vals["A"]), 1e-12)
                if name in ("profile_lag1_observed", "roughness_observed_kwh"):
                    cell["attribution"] = {"label": "data (property of the observations)" if rel >= 0.03 else
                                           "no material difference", "relative_difference": rel}
                elif m == "cvae" and rk is not None and f"c{c}_{rk}" in comp:
                    rc = comp[f"c{c}_{rk}"]
                    ra = RS.loc[(RS.track == "A") & (RS.cluster == c), rk]
                    rb = RS.loc[(RS.track == "B") & (RS.cluster == c), rk]
                    cell["restarts_50"] = {"A_median": rc["A_restarts_median"], "A_q10_q90": rc["A_restarts_q10_q90"],
                                           "B_median": rc["B_restarts_median"], "B_q10_q90": rc["B_restarts_q10_q90"]}
                    cell["attribution"] = attribute_cvae(rc["A_selected"], rc["B_selected"], ra, rb)
                elif m == "cvae" and rk is not None:
                    ra = RS.loc[(RS.track == "A") & (RS.cluster == c), rk].to_numpy()
                    rb = RS.loc[(RS.track == "B") & (RS.cluster == c), rk].to_numpy()
                    sa = float(RS.loc[(RS.track == "A") & (RS.cluster == c) & RS.selected, rk].iloc[0])
                    sb = float(RS.loc[(RS.track == "B") & (RS.cluster == c) & RS.selected, rk].iloc[0])
                    cell["restarts_50"] = {"A_median": float(np.median(ra)), "A_q10_q90": [float(np.quantile(ra, .1)),
                                                                                          float(np.quantile(ra, .9))],
                                           "B_median": float(np.median(rb)), "B_q10_q90": [float(np.quantile(rb, .1)),
                                                                                          float(np.quantile(rb, .9))]}
                    cell["attribution"] = attribute_cvae(sa, sb, ra, rb)
                elif m == "gam":
                    cell["attribution"] = {"label": "data" if rel >= 0.03 else "no material difference",
                                           "relative_difference": rel,
                                           "rule": "GAM fit deterministic, no CI for this diagnostic: 'data' if the "
                                                   "tracks differ by >= 3% (relative), else no material difference"}
                cells[str(c)] = cell
            sec[m] = cells
        out[name] = sec
    out["residual_lag1_vs_gam_mean"] = {t: [myd[f"{t}_cvae_c{c}"]["lag1_resid_vs_gam_mean"] for c in CL]
                                        for t in ("A", "B")}
    out["verdicts_by_track"] = {t: {k: (v.get("holds") if isinstance(v, dict) and "holds" in v else v)
                                    for k, v in S[t]["distribution_diagnostics"]["verdicts"].items()
                                    if k != "profile_lag1_autocorrelation"} for t in ("A", "B")}
    return out


# ---------------------------------------------------------------------------------------------------------------
# 3. tariff effects
# ---------------------------------------------------------------------------------------------------------------
REGIONS = {"Low": ["in_window", "pre2", "pre4", "post2", "post4", KEY_LREB, "rebound_32_47", "afternoon_24_31",
                   "daily_energy"],
           "High": ["in_window", "pre2", "pre4", "post2", "post4", "rebound_pre_24_38", "afternoon_24_34",
                    KEY_HPOST, "daily_energy"]}
CLAIM_SIGN = {k: v[0] for k, v in LE.CLAIMS.items()}


def effects_section() -> dict:
    out = {"definitions": eB["definitions"]}
    sc = {}
    for m in MODELS:
        sc[m] = {}
        for scen in ("Low", "High"):
            sc[m][scen] = {}
            for r in REGIONS[scen]:
                sign = CLAIM_SIGN.get((scen, r))
                cells = {}
                for c in CL:
                    st = {t: E[t]["per_model_cluster"][f"{m}_c{c}"][scen]["regions"][r] for t in ("A", "B")}
                    cell = {"paper_claimed_sign": sign}
                    for t in ("A", "B"):
                        cell[f"track{t}"] = {"pct": st[t]["pct"], "pct_ci95": st[t]["pct_ci95"],
                                             "kwh": st[t]["kwh"], "status": st[t].get("claim_status")}
                    cell["ci_overlap_A_B"] = ci_overlap(st["A"]["pct_ci95"], st["B"]["pct_ci95"])
                    col = f"{scen}:{r}:pct"
                    if m == "cvae" and col in RA.columns:
                        ra = RA.loc[RA.cluster == c, col].to_numpy(float)
                        rb = RB.loc[RB.cluster == c, col].to_numpy(float)
                        rest = {"A_median_pct": float(np.median(ra)), "B_median_pct": float(np.median(rb)),
                                "A_q10_q90": [float(np.quantile(ra, .1)), float(np.quantile(ra, .9))],
                                "B_q10_q90": [float(np.quantile(rb, .1)), float(np.quantile(rb, .9))]}
                        if sign is not None:
                            sa_ = RA.loc[RA.cluster == c, f"{scen}:{r}:status"]
                            sb_ = RB.loc[RB.cluster == c, f"{scen}:{r}:status"]
                            rest.update({"A_n_claimed_sign": int((np.sign(ra) == sign).sum()),
                                         "B_n_claimed_sign": int((np.sign(rb) == sign).sum()),
                                         "A_n_appears_ci_excl_0": int((sa_ == "appears").sum()),
                                         "B_n_appears_ci_excl_0": int((sb_ == "appears").sum()),
                                         "A_n_contradicted": int((sa_ == "contradicted").sum()),
                                         "B_n_contradicted": int((sb_ == "contradicted").sum())})
                        cell["restarts_50"] = rest
                        cell["attribution"] = attribute_cvae(st["A"]["pct"], st["B"]["pct"], ra, rb)
                    elif m == "gam":
                        cell["attribution"] = {"label": "data" if not cell["ci_overlap_A_B"] else
                                               "no clear difference (day-bootstrap CIs overlap)",
                                               "rule": "GAM fit deterministic; tracks' own day-bootstrap CIs"}
                    cells[str(c)] = cell
                sc[m][scen][r] = cells
    out["paper_scenarios"] = sc
    # joint appearance of all claims across restarts
    jt = {}
    for t in ("A", "B"):
        cnt = E[t]["restarts"]["recomputed"]["counts"]
        jt[t] = {str(c): {"all_10_claims_jointly_claimed_sign": cnt[str(c)]["all_claims_jointly"],
                          "n_restarts": cnt[str(c)]["n_restarts"]} for c in CL}
        jt[t]["phase1_table_6_claims_jointly"] = {k: v["all_claims_jointly"] for k, v in
                                                  E[t]["restarts"]["phase1_table"]["counts"].items()}
    out["restarts_joint_claims"] = jt
    # per-restart behaviour classes (mechanism_restarts_AB)
    out["restarts_classes"] = {c: {t: {k: MR["clusters"][c][t][k] for k in
                                       ("n_direction_both_appear", "n_side_effect_in_output",
                                        "n_low_rebound_both_appear", "n_high_rebound_24_34_appears",
                                        "selected_side_effect_in_output", "median_high_in_pct")}
                                   for t in ("A", "B")} for c in MR["clusters"]}
    # data-based reference (event study, lib.effects) and independent data checks
    dat = {}
    for kind in ("High", "Low"):
        dat[kind] = {}
        for r in ("in_block", "pre2", "pre4", "post2", "post4", "far_pre_5_12", "far_post_5_12"):
            cells = {}
            for c in CL:
                x = {t: M[t]["clusters"][str(c)]["data_all_events"][kind].get(r) for t in ("A", "B")}
                if x["A"] is None or x["B"] is None:
                    continue
                cells[str(c)] = {"trackA": {"pct": x["A"]["pct"], "pct_ci95": x["A"]["pct_ci95"]},
                                 "trackB": {"pct": x["B"]["pct"], "pct_ci95": x["B"]["pct_ci95"]},
                                 "ci_overlap_A_B": ci_overlap(x["A"]["pct_ci95"], x["B"]["pct_ci95"]),
                                 "significance_B": ("positive" if x["B"]["pct_ci95"][0] > 0 else "negative"
                                                    if x["B"]["pct_ci95"][1] < 0 else "not significant")}
            dat[kind][r] = cells
        dat[kind]["n_events"] = {t: M[t]["clusters"]["0"]["data_all_events"][kind]["n_events"] for t in ("A", "B")}
    dat["note"] = ("event day minus matched all-Normal days (same day type, +-14 days); real Low/High runs of the "
                   "continuous series; 95% CI = bootstrap over event days. Track A: authors' aggregates and tariff "
                   "dummies; Track B: our cluster aggregates (data/power_consump_ours.parquet) with Tariffs.xlsx "
                   "('loc' alignment). Differences between tracks are data only (no model).")
    dat["independent_check"] = {
        t: {"n_morning_low_only_days": DC[t]["n_morning_low_only_days"],
            "n_evening_high_days": DC[t]["n_evening_high_days"],
            "n_evening_high_days_clean_24_34": DC[t].get("n_evening_high_days_clean_24_34"),
            "clusters": {c: {bk: {k: DC[t]["clusters"][c][bk][k] for k in
                                  (f"morning_low_only_{LREB_SPAN}", "morning_low_only_32_47", "evening_high_pre4",
                                   "evening_high_post4", "evening_high_afternoon_24_34")} |
                              {"in_block_High": DC[t]["clusters"][c][bk]["in_block"]["High"],
                               "in_block_Low": DC[t]["clusters"][c][bk]["in_block"]["Low"]}
                              for bk in ("pm14", "week")} for c in DC[t]["clusters"]}}
        for t in ("A", "B", "B_authsched")}
    sa = {}
    for c in CL:
        rec = {}
        for kind in ("High", "Low"):
            v = {t: DC[t]["clusters"][str(c)]["pm14"]["in_block"][kind] for t in ("A", "B", "B_authsched")}
            gap_total = v["B"]["pct"] - v["A"]["pct"]
            gap_sched = v["B"]["pct"] - v["B_authsched"]["pct"]
            rec[kind] = {"trackA_authors_aggregates_authors_schedule": v["A"],
                         "trackB_our_aggregates_xlsx_schedule": v["B"],
                         "our_aggregates_authors_schedule": v["B_authsched"],
                         "B_minus_A_pct_points": gap_total,
                         "due_to_schedule_pct_points": gap_sched,
                         "due_to_households_pct_points": v["B_authsched"]["pct"] - v["A"]["pct"]}
        sa[str(c)] = rec
    sa["note"] = ("Tariffs.xlsx ('loc') and the authors' dummies differ on 17 days (6 extra Low 11-34 + High 35-46 days "
                  "and several extra morning-Low days in Nov-Dec, two Low-all-day days in Mar/May); phase 1 found no "
                  "consumption response on the extra xlsx event days. Their inclusion dilutes Track B's measured "
                  "responses; with the authors' schedule our cluster-3 aggregate gives the same High effect as "
                  "Track A.")
    dat["schedule_vs_households_attribution"] = sa
    n_clean = {t: DC[t].get("n_evening_high_days_clean_24_34") for t in ("A", "B", "B_authsched")}
    dat["independent_check"]["note"] = ("own re-implementation (no lib.effects functions): in-block effects reproduce "
                                        "the lib values; morning-Low-only days = every Low run ends by slot 10 and no "
                                        f"High; Low-rebound regions {LREB_SPAN.replace('_', '-')} and 32-47; "
                                        f"evening-High days = a High run covering {HSPAN.replace('_', '-')} (afternoon "
                                        f"test only on such days whose slots 24-34 are Normal: {n_clean}); 'week' = "
                                        "baseline from d-7/d+7 when all-Normal")
    out["data_reference"] = dat
    # CVAE implied effect on the same real event days
    ce = {}
    for kind in ("High", "Low"):
        ce[kind] = {}
        for r in ("in_block", "pre4", "post4", "far_pre_5_12", "far_post_5_12"):
            ce[kind][r] = {str(c): {t: {"pct": M[t]["clusters"][str(c)]["cvae_same_event_days"][kind][r]["pct"],
                                        "pct_ci95": M[t]["clusters"][str(c)]["cvae_same_event_days"][kind][r]["pct_ci95"]}
                                    for t in ("A", "B")} for c in CL
                           if M["A"]["clusters"][str(c)]["cvae_same_event_days"][kind].get(r)
                           and M["B"]["clusters"][str(c)]["cvae_same_event_days"][kind].get(r)}
    out["cvae_selected_on_real_event_days"] = ce
    # mechanism
    mech = {"training_blocks": {t: {k: v for k, v in M[t]["training_context"].items()
                                    if k.startswith(("train_high_runs", "n_train_high_runs", "train_low_runs",
                                                     "n_train_low_runs"))}
                                   | {"train_days_covering_paper_windows": M[t]["blocks_train_days_covering_paper_windows"]}
                                for t in ("A", "B")},
            "selected_decoder": {t: {str(c): M[t]["verdicts"]["clusters"][str(c)].get("cooccurrence") for c in CL}
                                 for t in ("A", "B")},
            "all_restarts": MR["clusters"],
            "verdict_overview_selected": {t: M[t]["verdicts"]["overall"] for t in ("A", "B")}}
    out["mechanism"] = mech
    # heterogeneity
    het = {}
    for t in ("A", "B"):
        dd = M[t]["verdicts"]["heterogeneity"]
        rows = {"data_in_block": {str(c): dd["data_in_block_pct_low_high"][str(c)] for c in CL},
                "gam_paper_scenarios": dd["gam"]["in_window_pct_low_high"],
                "cvae_selected_paper_scenarios": dd["cvae"]["in_window_pct_low_high"]}
        rr = RR[t]
        rows["cvae_restart_median_paper_scenarios"] = {
            str(c): [float(rr.loc[rr.cluster == c, "Low:in_window:pct"].median()),
                     float(rr.loc[rr.cluster == c, "High:in_window:pct"].median())] for c in CL}
        ratio = {}
        for k, v in rows.items():
            lo3, hi3 = v["3"]
            ratio[k] = {"low_c3_over_others": [lo3 / v[str(c)][0] if v[str(c)][0] else None for c in (0, 1, 2)],
                        "high_c3_over_others": [hi3 / v[str(c)][1] if v[str(c)][1] else None for c in (0, 1, 2)],
                        "c3_largest_low": bool(all(abs(lo3) > abs(v[str(c)][0]) for c in (0, 1, 2))),
                        "c3_largest_high": bool(all(abs(hi3) > abs(v[str(c)][1]) for c in (0, 1, 2)))}
        het[t] = {"low_high_pct": rows, "cluster3_ratios": ratio}
    out["heterogeneity"] = het
    return out


# ---------------------------------------------------------------------------------------------------------------
# 4. claims
# ---------------------------------------------------------------------------------------------------------------
def fmt(x, n=3):
    return "n/a" if x is None else f"{x:.{n}g}"


def claims(f4, dg, ef) -> dict:
    cl = {}
    # --- Fig. 4 RMSE
    r = f4["rmse"]
    rngB = (min(r["cvae_summary"]["trackB_ratio_to_paper_range"][0], r["gam_summary"]["trackB_ratio_to_paper_range"][0]),
            max(r["cvae_summary"]["trackB_ratio_to_paper_range"][1], r["gam_summary"]["trackB_ratio_to_paper_range"][1]))
    ok = 0.8 <= rngB[0] and rngB[1] <= 1.25 and r["gam_below_cvae_all_clusters"]["B"]
    cl["C_fig4_rmse_B"] = {
        "verdict": "reproduced" if ok else "partially reproduced",
        "claim": "Fig. 4 RMSE medians (CVAE ~0.020/0.022/0.018/0.025, GAM ~0.015/0.016/0.015/0.018 kWh); GAM better",
        "trackB": {m: [r[m][str(c)]["trackB_median"] for c in CL] for m in MODELS},
        "trackA": {m: [r[m][str(c)]["trackA_median"] for c in CL] for m in MODELS},
        "ratio_B_to_paper_range": rngB, "gam_below_cvae_all_clusters_B": r["gam_below_cvae_all_clusters"]["B"],
        "definition": "matches only the root-mean over 48 half-hours (Euclidean: ~7x)",
        "attribution": {m: {str(c): r[m][str(c)]["attribution"]["label"] for c in CL} for m in MODELS}}
    # --- Fig. 4 ES
    e = f4["es"]
    pr = f4["energy_score_variant_probe"]["B"]
    cl["C_fig4_es_B"] = {
        "verdict": "not reproduced",
        "claim": "Fig. 4 energy-score medians (CVAE ~0.150/0.145/0.170/0.130, GAM ~0.155/0.150/0.175/0.135 kWh)",
        "trackB": {m: [e[m][str(c)]["trackB_median"] for c in CL] for m in MODELS},
        "trackA": {m: [e[m][str(c)]["trackA_median"] for c in CL] for m in MODELS},
        "ratio_B_to_paper": {m: [e[m][str(c)]["ratio_B_to_paper"] for c in CL] for m in MODELS},
        "variants_all_8_cells_within_10pct_B": pr["variants_with_all_8_cells_within_10pct"],
        "closest_variants_B": pr["ranked_by_mean_abs_log_ratio"][:2],
        "closest_variant_mean_abs_pct_error_B": pr["per_variant"][pr["ranked_by_mean_abs_log_ratio"][0]]
        ["mean_abs_pct_error"],
        "attribution": {m: {str(c): e[m][str(c)]["attribution"]["label"] for c in CL} for m in MODELS}}
    # --- Fig. 4 VG
    v = f4["vg"]
    ratios = [v[m][str(c)]["ratio_B_to_paper"] for m in MODELS for c in CL]
    n_in = sum(0.8 <= x <= 1.25 for x in ratios)
    cl["C_fig4_vg_B"] = {
        "verdict": "reproduced" if n_in == 8 else ("partially reproduced" if n_in >= 5 else "not reproduced"),
        "claim": "Fig. 4 variogram medians (CVAE ~6/7/7/10, GAM ~4/4/5/5), GAM lower",
        "trackB": {m: [v[m][str(c)]["trackB_median"] for c in CL] for m in MODELS},
        "trackA": {m: [v[m][str(c)]["trackA_median"] for c in CL] for m in MODELS},
        "ratio_B_to_paper": {m: [v[m][str(c)]["ratio_B_to_paper"] for c in CL] for m in MODELS},
        "n_cells_within_0.8_1.25": n_in, "gam_below_cvae_all_clusters_B": v["gam_below_cvae_all_clusters"]["B"],
        "attribution": {m: {str(c): v[m][str(c)]["attribution"]["label"] for c in CL} for m in MODELS}}
    # --- 13%
    x = f4["rmse_13pct"]["B"]
    cl["C_rmse_13pct_B"] = {
        "verdict": "not reproduced" if not x["within_2pp_of_13pct"] else "partially reproduced",
        "claim": "13% RMSE difference between CVAE and GAM (abstract)",
        "median_based_rel_to_cvae_pct": x["candidates_rel_to_cvae_pct"]["median_over_all_cluster_days"],
        "median_based_rel_to_gam_pct": x["candidates_rel_to_gam_pct"]["median_over_all_cluster_days"],
        "range_of_readings_pct": x["range_of_all_our_readings_pct"],
        "trackA_median_based_rel_to_cvae_pct": f4["rmse_13pct"]["A"]["candidates_rel_to_cvae_pct"]
        ["median_over_all_cluster_days"],
        "paper_own_fig4_implies_pct": [f4["rmse_13pct"]["paper"]["paper_fig4_medians_imply_rel_to_cvae_pct"],
                                       f4["rmse_13pct"]["paper"]["paper_fig4_medians_imply_rel_to_gam_pct"]]}
    # --- ES claim
    x = f4["es_cvae_lower_claim"]["B"]
    cl["C_es_cvae_better_B"] = {
        "verdict": "not reproduced" if x["n_clusters_cvae_lower"] == 0 else
        ("reproduced" if x["n_clusters_cvae_lower"] == 4 else "partially reproduced"),
        "claim": "energy score slightly lower for the CVAE than for the GAM",
        "n_clusters_cvae_lower_B": x["n_clusters_cvae_lower"],
        "ratio_cvae_over_gam_B": [x["per_cluster"][str(c)]["ratio_cvae_over_gam"] for c in CL],
        "frac_days_cvae_lower_B": [x["per_cluster"][str(c)]["frac_days_cvae_lower"] for c in CL],
        "ratio_cvae_over_gam_A": [f4["es_cvae_lower_claim"]["A"]["per_cluster"][str(c)]["ratio_cvae_over_gam"]
                                  for c in CL],
        "gap_change_B_minus_A_ci95": {str(c): f4["es"]["cvae_over_gam_median_ratio"][str(c)]["B_minus_A_ci95"]
                                      for c in CL}}
    # --- variance
    vb = sB["distribution_diagnostics"]["verdicts"]
    cl["C_cvae_low_variance_B"] = {
        "verdict": "reproduced" if vb["cvae_too_little_variance"]["holds"] else "not reproduced",
        "claim": "CVAE samples have too little variance",
        "spread_skill_B": vb["cvae_too_little_variance"]["spread_skill_ratio"],
        "coverage90_B": vb["cvae_too_little_variance"]["coverage90"],
        "spread_skill_A": sA["distribution_diagnostics"]["verdicts"]["cvae_too_little_variance"]["spread_skill_ratio"],
        "coverage90_A": sA["distribution_diagnostics"]["verdicts"]["cvae_too_little_variance"]["coverage90"],
        "restarts_spread_skill_B_q10_q90": {str(c): dg["spread_skill_ratio_calibrated"]["cvae"][str(c)]["restarts_50"]
                                            ["B_q10_q90"] for c in CL}}
    cl["C_gam_high_variance_B"] = {
        "verdict": "reproduced" if vb["gam_too_much_variance"]["holds"] else "not reproduced",
        "claim": "GAM samples have too much variance",
        "spread_skill_B": vb["gam_too_much_variance"]["spread_skill_ratio"],
        "coverage90_B": vb["gam_too_much_variance"]["coverage90"],
        "coverage50_B": vb["gam_too_much_variance"]["coverage50"],
        "spread_skill_A": sA["distribution_diagnostics"]["verdicts"]["gam_too_much_variance"]["spread_skill_ratio"],
        "coverage90_A": sA["distribution_diagnostics"]["verdicts"]["gam_too_much_variance"]["coverage90"]}
    sm = vb["cvae_consecutive_halfhours_very_correlated"]["holds"] and vb["gam_erratic"]["holds"]
    rib = vb["real_behaviour_in_between"]
    cl["C_smoothness_B"] = {
        "verdict": "reproduced" if (sm and rib["variance_holds"] and rib["roughness_holds"]) else
        "partially reproduced",
        "claim": "CVAE consecutive half-hours very correlated, GAM erratic; real behaviour in between",
        "cvae_lag1_across_samples_B": vb["cvae_consecutive_halfhours_very_correlated"]["cvae_across_sample_lag1"],
        "observed_residual_lag1_vs_gam_mean_B": vb["cvae_consecutive_halfhours_very_correlated"]
        ["observed_residual_lag1_vs_gam_mean"],
        "gam_lag1_across_samples_B": vb["gam_erratic"]["gam_across_sample_lag1"],
        "roughness_ratio_cvae_B": vb["cvae_consecutive_halfhours_very_correlated"]["cvae_roughness_ratio_samples_to_obs"],
        "roughness_ratio_gam_B": vb["gam_erratic"]["gam_roughness_ratio_samples_to_obs"],
        "real_in_between_B": {k: rib[k] for k in ("variance_holds", "roughness_holds", "profile_lag1_holds",
                                                  "lag1_strictly_between_holds")}}
    # --- tariff direction
    ps = ef["paper_scenarios"]
    st = {m: {sc_: [ps[m][sc_]["in_window"][str(c)]["trackB"]["status"] for c in CL] for sc_ in ("Low", "High")}
          for m in MODELS}
    gam_all = all(s_ == "appears" for sc_ in ("Low", "High") for s_ in st["gam"][sc_])
    cvae_n = sum(st["cvae"]["Low"][c] == "appears" and st["cvae"]["High"][c] == "appears" for c in CL)
    rc = ef["restarts_classes"]
    cl["C_tariff_direction_B"] = {
        "verdict": "reproduced" if (gam_all and cvae_n == 4) else ("partially reproduced" if gam_all or cvae_n
                                                                 else "not reproduced"),
        "claim": "both generators raise consumption under Low and lower it under High",
        "status_B": st,
        "pct_B": {m: {sc_: [ps[m][sc_]["in_window"][str(c)]["trackB"]["pct"] for c in CL] for sc_ in ("Low", "High")}
                  for m in MODELS},
        "pct_A": {m: {sc_: [ps[m][sc_]["in_window"][str(c)]["trackA"]["pct"] for c in CL] for sc_ in ("Low", "High")}
                  for m in MODELS},
        "cvae_selected_clusters_both_directions_B": cvae_n,
        "cvae_restarts_both_directions_appear": {t: [rc[str(c)][t]["n_direction_both_appear"] for c in CL]
                                                 for t in ("A", "B")},
        "attribution_cvae": {sc_: {str(c): ps["cvae"][sc_]["in_window"][str(c)]["attribution"]["label"] for c in CL}
                             for sc_ in ("Low", "High")}}
    # --- side effect
    ov = {t: M[t]["verdicts"]["overall"] for t in ("A", "B")}
    n_sel = ov["B"]["n_clusters_cvae_side_effect_in_output"]
    n_maj = sum(rc[str(c)]["B"]["n_side_effect_in_output"] >= 25 for c in CL)
    n_dat = ov["B"]["n_clusters_data_side_effect"]
    cl["C_cvae_side_effect_B"] = {
        "verdict": ("reproduced" if (n_sel >= 3 and n_dat >= 3) else
                    "partially reproduced" if (n_sel >= 1 or n_maj >= 1) else "not reproduced"),
        "verdict_rule": "reproduced = side effect in the selected CVAE in >= 3 clusters AND in the data; partially = "
                        "present in the CVAE output (selected model or a majority of restarts) in some clusters",
        "n_clusters_majority_of_restarts_B": n_maj,
        "claim": "CVAE: the High drop starts before and lasts after the window (side effect), a real consumer "
                 "behaviour the GAM cannot model",
        "selected_clusters_with_side_effect_in_output": {t: ov[t]["n_clusters_cvae_side_effect_in_output"]
                                                         for t in ("A", "B")},
        "restarts_with_side_effect_in_output": {t: [rc[str(c)][t]["n_side_effect_in_output"] for c in CL]
                                                for t in ("A", "B")},
        "data_clusters_with_side_effect": {t: ov[t]["n_clusters_data_side_effect"] for t in ("A", "B")},
        "cvae_selected_side_effect_on_real_event_days": {t: ov[t]["n_clusters_cvae_side_effect_on_real_event_days"]
                                                         for t in ("A", "B")},
        "regions": {"pre2": "37-38", "pre4": "35-38", "in_window": LE.WINDOW_SPAN["High"], "post2": "45-46",
                    "post4": f"{LE.span(H_POST)} (only {len(H_POST)} slots remain in the day)"},
        "pct_B": {r_: [ps["cvae"]["High"][r_][str(c)]["trackB"]["pct"] for c in CL]
                  for r_ in ("pre2", "pre4", "in_window", "post2", "post4")},
        "pct_A": {r_: [ps["cvae"]["High"][r_][str(c)]["trackA"]["pct"] for c in CL]
                  for r_ in ("pre2", "pre4", "in_window", "post2", "post4")},
        "data_pre4_post4_B": {r_: [ef["data_reference"]["High"][r_][str(c)]["trackB"] for c in CL]
                              for r_ in ("pre4", "post4")}}
    # --- rebound
    nl, nh = ov["B"]["n_clusters_cvae_low_rebound"], ov["B"]["n_clusters_cvae_high_rebound"]
    cl["C_cvae_rebound_B"] = {
        "verdict": ("reproduced" if (nl >= 3 and nh >= 3) else
                    "partially reproduced" if (nl >= 1 or nh >= 1) else "not reproduced"),
        "verdict_rule": f"selected CVAE: Low rebound = drop in both {LREB_SPAN.replace('_', '-')} and 32-47 (CI excludes "
                        "0); High rebound = rise in 24-34 (CI excludes 0); reproduced = both in >= 3 clusters",
        "claim": "CVAE: after a morning Low, afternoon/evening consumption drops; with an evening High, afternoon "
                 "consumption rises a little",
        "selected_clusters_low_rebound": {t: ov[t]["n_clusters_cvae_low_rebound"] for t in ("A", "B")},
        "selected_clusters_high_rebound": {t: ov[t]["n_clusters_cvae_high_rebound"] for t in ("A", "B")},
        "restarts_low_rebound_both_regions": {t: [rc[str(c)][t]["n_low_rebound_both_appear"] for c in CL]
                                              for t in ("A", "B")},
        "restarts_high_rebound_24_34": {t: [rc[str(c)][t]["n_high_rebound_24_34_appears"] for c in CL]
                                        for t in ("A", "B")},
        "data_clusters_low_rebound": {t: ov[t]["n_clusters_data_low_rebound"] for t in ("A", "B")},
        "data_clusters_high_rebound": {t: ov[t]["n_clusters_data_high_rebound"] for t in ("A", "B")},
        "pct_B": {f"Low_{KEY_LREB}": [ps["cvae"]["Low"][KEY_LREB][str(c)]["trackB"]["pct"] for c in CL],
                  "Low_rebound_32_47": [ps["cvae"]["Low"]["rebound_32_47"][str(c)]["trackB"]["pct"] for c in CL],
                  "High_afternoon_24_34": [ps["cvae"]["High"]["afternoon_24_34"][str(c)]["trackB"]["pct"] for c in CL]},
        "pct_A": {f"Low_{KEY_LREB}": [ps["cvae"]["Low"][KEY_LREB][str(c)]["trackA"]["pct"] for c in CL],
                  "Low_rebound_32_47": [ps["cvae"]["Low"]["rebound_32_47"][str(c)]["trackA"]["pct"] for c in CL],
                  "High_afternoon_24_34": [ps["cvae"]["High"]["afternoon_24_34"][str(c)]["trackA"]["pct"] for c in CL]}}
    # --- heterogeneity
    h = ef["heterogeneity"]["B"]
    r3 = h["cluster3_ratios"]
    het_ok = (r3["data_in_block"]["c3_largest_low"] and r3["data_in_block"]["c3_largest_high"]
              and r3["gam_paper_scenarios"]["c3_largest_low"] and r3["gam_paper_scenarios"]["c3_largest_high"])
    cl["C_cluster_heterogeneity_B"] = {
        "verdict": "reproduced" if het_ok else "partially reproduced",
        "verdict_rule": "cluster 3 responds most to both Low and High in the data and in the GAM; the CVAE is reported "
                        "separately (selected model and restart medians)",
        "claim": "tariff responsiveness differs between clusters",
        "low_high_pct_B": h["low_high_pct"], "cluster3_ratios_B": h["cluster3_ratios"],
        "low_high_pct_A_data": ef["heterogeneity"]["A"]["low_high_pct"]["data_in_block"]}
    return cl


# ---------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
MCOL = {"cvae": "#2a78d6", "gam": "#eb6834", "data": INK}
TINT = {"cvae": "#86b6ef", "gam": "#f2a585"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "font.size": 8.5, "axes.titlesize": 9.5, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def median_ci(t, m, c, col, B=2000, seed=7):
    """95% bootstrap CI over test days of the median score (selected CVAE / GAM) from scores_long.csv."""
    key = (t, m, c, col)
    if key not in _CI_CACHE:
        L = LONG[t]
        x = L.loc[(L.model == m) & (L.cluster == c), col].to_numpy(float)
        idx = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
        bm = np.median(x[idx], 1)
        _CI_CACHE[key] = (float(np.quantile(bm, .025)), float(np.quantile(bm, .975)))
    return _CI_CACHE[key]


LONG = {"A": pd.read_csv(A + "/scores_long.csv"), "B": pd.read_csv(B + "/scores_long.csv")}
_CI_CACHE: dict = {}


def fig_fig4(f4, path):
    plt = _plt()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    rng = np.random.default_rng(0)
    for j, (col, key, mk, rk, desc) in enumerate(FIG4):
        for i, m in enumerate(MODELS):
            ax = axes[i, j]
            sec = f4[key][m]
            for c in CL:
                cell = sec[str(c)]
                ax.scatter([c], [cell["paper_approx_median"]], marker="D", s=30, color=INK, zorder=6, linewidths=0)
                for t, dx, face in (("A", -0.22, SURFACE), ("B", 0.22, MCOL[m])):
                    if m == "cvae":
                        rv = RS.loc[(RS.track == t) & (RS.cluster == c), rk].to_numpy()
                        ax.scatter(c + dx + rng.uniform(-0.07, 0.07, len(rv)), rv, s=7, color=TINT[m], alpha=0.7,
                                   linewidths=0, zorder=2)
                    med = cell[f"track{t}_median"]
                    lo, hi = median_ci(t, m, c, col)
                    ax.plot([c + dx, c + dx], [lo, hi], color=MCOL[m], linewidth=0.9, zorder=4)
                    ax.scatter([c + dx], [med], s=34, facecolors=face, edgecolors=MCOL[m], linewidths=1.3, zorder=5)
            ax.set_xticks(CL)
            ax.set_xticklabels([f"{c}" for c in CL])
            ax.set_xlim(-0.6, 3.6)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            if i == 1:
                ax.set_xlabel("Cluster")
            ax.set_title(f"{'CVAE' if m == 'cvae' else 'GAM'}: {desc.split(',')[0].split(' (')[0]}", loc="left")
            if j == 0:
                ax.set_ylabel("Median over the 92 test days")
    handles = [Line2D([], [], marker="D", color=INK, linestyle="none", markersize=5, label="paper Fig. 4 (read off the plot)"),
               Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=INK2, linestyle="none", markersize=6,
                      label="Track A (authors' data), selected CVAE / GAM"),
               Line2D([], [], marker="o", markerfacecolor=INK2, markeredgecolor=INK2, linestyle="none", markersize=6,
                      label="Track B (our rebuild), selected CVAE / GAM"),
               Line2D([], [], color=INK2, linewidth=0.9, label="95% CI of the median (bootstrap over test days)"),
               Line2D([], [], marker="o", color=TINT["cvae"], linestyle="none", markersize=3.5,
                      label="CVAE: median of each of the 50 restarts (left A, right B)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965), fontsize=8)
    fig.suptitle("Fig. 4 medians: paper vs Track A vs Track B (CVAE blue, GAM orange; RMSE = root-mean over 48 "
                 "half-hours, ES = eq. 14, VG = p 0.5 ordered pairs)", y=0.995, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    fig.savefig(path, dpi=160)
    plt.close(fig)


EFF_ROWS = [("Low", "in_window", f"Low window {LE.WINDOW_SPAN['Low']}", +1, ("Low", "in_block")),
            ("High", "in_window", f"High window {LE.WINDOW_SPAN['High']}", -1, ("High", "in_block")),
            ("High", "pre4", "4 slots before High (35-38)", -1, ("High", "pre4")),
            ("High", "post4", f"{len(H_POST)} slots after High ({LE.span(H_POST)})", -1, ("High", "post4")),
            ("Low", KEY_LREB, f"after Low: {LREB_SPAN.replace('_', '-')}", -1, ("morning_low", LREB_SPAN)),
            ("Low", "rebound_32_47", "after Low: 32-47", -1, ("morning_low", "32_47")),
            ("High", "afternoon_24_34", "before High: 24-34", +1, ("High", "far_pre_5_12"))]


def fig_effects(ef, path):
    plt = _plt()
    from matplotlib.lines import Line2D
    nr = len(EFF_ROWS)
    fig, axes = plt.subplots(nr, 4, figsize=(13.5, 2.05 * nr + 1.2))
    rng = np.random.default_rng(1)
    groups = [("data", 0.0), ("gam", 1.0), ("cvae", 2.0), ("cvae_r", 3.0)]
    for i, (scen, reg, title, sign, dref) in enumerate(EFF_ROWS):
        for j, c in enumerate(CL):
            ax = axes[i, j]
            ax.axhline(0, color=MUTED, linewidth=0.7, zorder=1)
            for t, dx, filled in (("A", -0.17, False), ("B", 0.17, True)):
                # data reference
                if dref[0] == "morning_low":
                    x = DC[t]["clusters"][str(c)]["pm14"][f"morning_low_only_{dref[1]}"]
                    dv, dci = x["pct"], x["pct_ci95"]
                else:
                    x = M[t]["clusters"][str(c)]["data_all_events"][dref[0]][dref[1]]
                    dv, dci = x["pct"], x["pct_ci95"]
                for gname, gx in groups:
                    xx = gx + dx
                    if gname == "data":
                        val, ci, colr = dv, dci, INK
                    elif gname in ("gam", "cvae"):
                        cell = ef["paper_scenarios"][gname][scen][reg][str(c)][f"track{t}"]
                        val, ci, colr = cell["pct"], cell["pct_ci95"], MCOL[gname]
                    else:
                        rv = RR[t].loc[RR[t].cluster == c, f"{scen}:{reg}:pct"].to_numpy(float)
                        ax.scatter(xx + rng.uniform(-0.08, 0.08, len(rv)), rv, s=5, color=TINT["cvae"], alpha=0.75,
                                   linewidths=0, zorder=2)
                        ax.plot([xx - 0.12, xx + 0.12], [np.median(rv)] * 2, color=MCOL["cvae"], linewidth=1.4, zorder=4)
                        continue
                    ax.plot([xx, xx], ci, color=colr, linewidth=0.9, zorder=3)
                    ax.scatter([xx], [val], s=26, facecolors=colr if filled else SURFACE, edgecolors=colr,
                               linewidths=1.2, zorder=5)
            ax.set_xticks([g for _, g in groups])
            ax.set_xticklabels(["data", "GAM", "CVAE\nselected", "CVAE\n50 restarts"] if i == nr - 1 else [""] * 4,
                               fontsize=7.5)
            ax.set_xlim(-0.5, 3.5)
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(f"Cluster {c} (A {SIZES['A'][c]} / B {SIZES['B'][c]} hh)\n{title}", loc="left", fontsize=8.5)
            else:
                ax.set_title(title, loc="left", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"% of baseline\n(paper: {'rise' if sign > 0 else 'drop'})", fontsize=8)
    handles = [Line2D([], [], marker="o", markerfacecolor=SURFACE, markeredgecolor=INK2, linestyle="none",
                      markersize=6, label="Track A (authors' data)"),
               Line2D([], [], marker="o", markerfacecolor=INK2, markeredgecolor=INK2, linestyle="none", markersize=6,
                      label="Track B (our rebuild)"),
               Line2D([], [], color=INK, marker="o", markersize=4, linewidth=0.9,
                      label="data: real tariff blocks vs matched Normal days, 95% CI over event days"),
               Line2D([], [], color=MCOL["gam"], marker="o", markersize=4, linewidth=0.9,
                      label="GAM, paper scenario, 95% CI over test days"),
               Line2D([], [], color=MCOL["cvae"], marker="o", markersize=4, linewidth=0.9,
                      label="CVAE (selected restart), paper scenario, 95% CI"),
               Line2D([], [], color=TINT["cvae"], marker="o", linestyle="none", markersize=3,
                      label="CVAE: each of 50 restarts (bar = median)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.985), fontsize=8)
    fig.suptitle(" ", y=1.0)
    fig.text(0.5, 0.951, f"Tariff effects, paper windows Low {LE.WINDOW_SPAN['Low']} ({LE.WINDOW_CLOCK['Low']} GMT), High "
                         f"{LE.WINDOW_SPAN['High']} ({LE.WINDOW_CLOCK['High']} GMT): Track A vs Track B.\nData rows use "
                         "the closest real-block analogue (in-block; 4 slots before / after High runs; morning-Low-only "
                         f"days for {LREB_SPAN.replace('_', '-')} / 32-47; 5-12 slots before High runs for 24-34)",
             ha="center", va="top", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def r2(x):
    return [round(v, 2) for v in x]


def verdicts_same_rule(t: str, f4: dict, ef: dict) -> dict:
    """The verdict rules of claims() applied to one track (to compare Track A and Track B like for like)."""
    v = {}
    rat = lambda key: [f4[key][m][str(c)][f"ratio_{t}_to_paper"] for m in MODELS for c in CL]  # noqa: E731
    r = rat("rmse")
    v["C_fig4_rmse"] = ("reproduced" if (min(r) >= 0.8 and max(r) <= 1.25 and f4["rmse"]["gam_below_cvae_all_clusters"][t])
                        else "partially reproduced")
    pr = f4["energy_score_variant_probe"][t]
    e = rat("es")
    v["C_fig4_es"] = ("reproduced" if (min(e) >= 0.8 and max(e) <= 1.25) else
                      "partially reproduced" if pr["variants_with_all_8_cells_within_10pct"] else "not reproduced")
    g = rat("vg")
    n_in = sum(0.8 <= x <= 1.25 for x in g)
    v["C_fig4_vg"] = "reproduced" if n_in == 8 else ("partially reproduced" if n_in >= 5 else "not reproduced")
    v["C_rmse_13pct"] = "not reproduced" if not f4["rmse_13pct"][t]["within_2pp_of_13pct"] else "partially reproduced"
    n = f4["es_cvae_lower_claim"][t]["n_clusters_cvae_lower"]
    v["C_es_cvae_better"] = "not reproduced" if n == 0 else ("reproduced" if n == 4 else "partially reproduced")
    vb = S[t]["distribution_diagnostics"]["verdicts"]
    v["C_cvae_low_variance"] = "reproduced" if vb["cvae_too_little_variance"]["holds"] else "not reproduced"
    v["C_gam_high_variance"] = "reproduced" if vb["gam_too_much_variance"]["holds"] else "not reproduced"
    rib = vb["real_behaviour_in_between"]
    v["C_smoothness"] = ("reproduced" if (vb["cvae_consecutive_halfhours_very_correlated"]["holds"] and
                                          vb["gam_erratic"]["holds"] and rib["variance_holds"] and
                                          rib["roughness_holds"]) else "partially reproduced")
    ps = ef["paper_scenarios"]
    st = {m: {s_: [ps[m][s_]["in_window"][str(c)][f"track{t}"]["status"] for c in CL] for s_ in ("Low", "High")}
          for m in MODELS}
    gam_all = all(x == "appears" for s_ in ("Low", "High") for x in st["gam"][s_])
    cvae_n = sum(st["cvae"]["Low"][c] == "appears" and st["cvae"]["High"][c] == "appears" for c in CL)
    v["C_tariff_direction"] = ("reproduced" if (gam_all and cvae_n == 4) else
                               "partially reproduced" if (gam_all or cvae_n) else "not reproduced")
    ov = M[t]["verdicts"]["overall"]
    n_sel, n_dat = ov["n_clusters_cvae_side_effect_in_output"], ov["n_clusters_data_side_effect"]
    n_maj = sum(MR["clusters"][str(c)][t]["n_side_effect_in_output"] >= 25 for c in CL)
    v["C_cvae_side_effect"] = ("reproduced" if (n_sel >= 3 and n_dat >= 3) else
                               "partially reproduced" if (n_sel >= 1 or n_maj >= 1) else "not reproduced")
    nl, nh = ov["n_clusters_cvae_low_rebound"], ov["n_clusters_cvae_high_rebound"]
    v["C_cvae_rebound"] = ("reproduced" if (nl >= 3 and nh >= 3) else
                           "partially reproduced" if (nl >= 1 or nh >= 1) else "not reproduced")
    r3 = ef["heterogeneity"][t]["cluster3_ratios"]
    v["C_cluster_heterogeneity"] = ("reproduced" if (r3["data_in_block"]["c3_largest_low"] and
                                                     r3["data_in_block"]["c3_largest_high"] and
                                                     r3["gam_paper_scenarios"]["c3_largest_low"] and
                                                     r3["gam_paper_scenarios"]["c3_largest_high"])
                                    else "partially reproduced")
    return v


def headline(f4, dg, ef, cl) -> dict:
    """Plain-language summary generated from the numbers."""
    mr = MR["clusters"]
    sv = ef["data_reference"]["schedule_vs_households_attribution"]
    rmse = cl["C_fig4_rmse_B"]
    es = cl["C_fig4_es_B"]
    vg = cl["C_fig4_vg_B"]
    out = {
        "claims_that_hold_on_track_B": [
            f"Fig. 4 RMSE level (root-mean over 48 half-hours): ratio to the paper "
            f"{rmse['ratio_B_to_paper_range'][0]:.2f}-{rmse['ratio_B_to_paper_range'][1]:.2f}, GAM better in all 4 clusters",
            f"Fig. 4 variogram (ordered pairs) in 6 of 8 cells (ratio {min(min(vg['ratio_B_to_paper'][m]) for m in MODELS):.2f}"
            f"-{max(max(vg['ratio_B_to_paper'][m]) for m in MODELS):.2f}; cluster 2 is 24-28% low)",
            "CVAE too little variance (spread/skill " + "/".join(f"{x:.2f}" for x in cl["C_cvae_low_variance_B"]
                                                                 ["spread_skill_B"]) + "), GAM too much ("
            + "/".join(f"{x:.2f}" for x in cl["C_gam_high_variance_B"]["spread_skill_B"]) + "); CVAE smooth, GAM "
            "erratic, real profiles in between",
        ],
        "claims_that_do_not_hold_on_track_B": [
            f"Fig. 4 energy-score values (correct ES: CVAE ratio {min(es['ratio_B_to_paper']['cvae']):.2f}-"
            f"{max(es['ratio_B_to_paper']['cvae']):.2f}, GAM {min(es['ratio_B_to_paper']['gam']):.2f}-"
            f"{max(es['ratio_B_to_paper']['gam']):.2f}; no variant matches all 8 cells)",
            f"'13% RMSE difference': median-based {cl['C_rmse_13pct_B']['median_based_rel_to_cvae_pct']:.1f}% "
            f"(rel. to CVAE) / {cl['C_rmse_13pct_B']['median_based_rel_to_gam_pct']:.1f}% (rel. to GAM)",
            "CVAE energy score lower than GAM's: CVAE/GAM = " + "/".join(
                f"{x:.2f}" for x in cl["C_es_cvae_better_B"]["ratio_cvae_over_gam_B"]),
        ],
        "trackB_vs_trackA": {
            "fig4_scores": "CVAE: cluster 0 no material difference; clusters 1 and 3 slightly harder in Track B (the "
                           "whole 50-restart distribution shifts up 2-7%, significant except cluster-1 VG: data); "
                           "cluster 2 easier (restart medians RMSE -4%, ES -5%, VG -18%: data, a different and larger "
                           "cluster, 299 vs 229 households). GAM: cluster 2 lower (RMSE x0.87, ES x0.89, VG x0.76, CIs "
                           "exclude 1); small significant increases for cluster 0 VG (x1.08) and cluster 1 ES (x1.06); "
                           "the rest n.s. CVAE/GAM ratios unchanged (all B-A CIs include 0).",
            "diagnostics": "same verdicts in both tracks; numeric differences small (spread/skill, coverage) or "
                           "data-driven (roughness ratios follow the observed series)",
            "tariff_effects_cvae": None,
            "tariff_schedule": None,
            "conclusion": None}}
    # ---- tariff statements, generated from the numbers (scenario windows from lib.effects)
    ps = ef["paper_scenarios"]
    hold, nothold = out["claims_that_hold_on_track_B"], out["claims_that_do_not_hold_on_track_B"]

    def pcts(m, scen, reg="in_window", t="B"):
        return "/".join(f"{ps[m][scen][reg][str(c)][f'track{t}']['pct']:+.1f}" for c in CL)

    gst = {s: [ps["gam"][s]["in_window"][str(c)]["trackB"]["status"] for c in CL] for s in ("Low", "High")}
    gmax = max(E["B"]["per_model_cluster"][f"gam_c{c}"][s]["regions"]["max_abs_outside_window_kwh"]
               for c in CL for s in ("Low", "High"))
    g_ok = all(x == "appears" for s in gst for x in gst[s])
    txt = (f"GAM (windows Low {LE.WINDOW_SPAN['Low']}, High {LE.WINDOW_SPAN['High']}): Low "
           f"{'raises' if g_ok else 'changes'} ({pcts('gam', 'Low')}% in clusters 0-3) and High "
           f"{'lowers' if g_ok else 'changes'} ({pcts('gam', 'High')}%) consumption"
           f"{' in every cluster' if g_ok else ' (claimed sign with CI excluding 0 not in every cluster)'}, effects "
           f"confined to the window (max |change| outside {gmax:.1e} kWh)")
    (hold if (g_ok and gmax == 0) else nothold).append(txt)
    r3 = ef["heterogeneity"]["B"]["cluster3_ratios"]

    def rr(xs):
        xs = [x for x in xs if x is not None]
        return f"{min(xs):.1f}-{max(xs):.1f}x"
    het = cl["C_cluster_heterogeneity_B"]["verdict"]
    txt = (f"cluster heterogeneity ({het}): cluster 3's effect relative to clusters 0-2 is Low "
           f"{rr(r3['data_in_block']['low_c3_over_others'])}, High {rr(r3['data_in_block']['high_c3_over_others'])} in "
           f"the data (real blocks) and Low {rr(r3['gam_paper_scenarios']['low_c3_over_others'])}, High "
           f"{rr(r3['gam_paper_scenarios']['high_c3_over_others'])} in the GAM (paper scenarios)")
    (hold if het == "reproduced" else nothold).append(txt)
    td = cl["C_tariff_direction_B"]
    cst = td["status_B"]["cvae"]
    both = [c for c in CL if cst["Low"][c] == "appears" and cst["High"][c] == "appears"]
    txt = (f"CVAE tariff direction: selected decoders right in both directions (CI excludes 0) in "
           f"{'cluster(s) ' + ', '.join(map(str, both)) if both else 'no cluster'} (in-window Low {pcts('cvae', 'Low')}%, "
           f"High {pcts('cvae', 'High')}%); across restarts both directions appear in "
           + "/".join(str(x) for x in td["cvae_restarts_both_directions_appear"]["B"]) + " of 50")
    (hold if len(both) == len(CL) else nothold).append(txt)
    se = cl["C_cvae_side_effect_B"]
    nd = se["data_clusters_with_side_effect"]["B"]
    c33 = "/".join(f"{mr[str(c)]['B']['median_corr_paperHigh_real_33_44']:.2f}" for c in CL)
    c35 = "/".join(f"{mr[str(c)]['B']['median_corr_paperHigh_real_35_46']:.2f}" for c in CL)
    txt = (f"side effect ({se['verdict']}): "
           + ("no significant drop before/after real High blocks in any cluster of the data" if nd == 0 else
              f"a significant drop before/after real High blocks in {nd} cluster(s) of the data")
           + f"; the selected CVAE's High scenario shows it (pre and post CI exclude 0) in "
           f"{se['selected_clusters_with_side_effect_in_output']['B']} of 4 clusters (pre4 {pcts('cvae', 'High', 'pre4')}%, "
           f"post 45-47 {pcts('cvae', 'High', 'post4')}%), restarts "
           + "/".join(str(x) for x in se["restarts_with_side_effect_in_output"]["B"])
           + f" of 50; median correlation of the restarts' paper-High response with real training block 33-44 / "
           f"35-46 inserted alone {c33} / {c35} (co-occurrence with the longer training blocks)")
    (hold if se["verdict"] == "reproduced" else nothold).append(txt)
    rb = cl["C_cvae_rebound_B"]
    vcl = M["B"]["verdicts"]["clusters"]
    dl = [c for c in CL if vcl[str(c)].get("data_low_rebound")]
    dh = [c for c in CL if vcl[str(c)].get("data_high_rebound")]
    txt = (f"rebound ({rb['verdict']}): selected CVAE Low rebound (drop in {LREB_SPAN.replace('_', '-')} and 32-47) in "
           f"{rb['selected_clusters_low_rebound']['B']} and High rebound (rise in 24-34) in "
           f"{rb['selected_clusters_high_rebound']['B']} of 4 clusters (Low {LREB_SPAN.replace('_', '-')}: "
           f"{pcts('cvae', 'Low', KEY_LREB)}%, 32-47: {pcts('cvae', 'Low', 'rebound_32_47')}%; High 24-34: "
           f"{pcts('cvae', 'High', 'afternoon_24_34')}%); restarts Low "
           + "/".join(str(x) for x in rb["restarts_low_rebound_both_regions"]["B"]) + ", High "
           + "/".join(str(x) for x in rb["restarts_high_rebound_24_34"]["B"]) + " of 50; data: significant post-Low "
           f"drop in {'cluster(s) ' + ', '.join(map(str, dl)) if dl else 'no cluster'}, pre-High rise in "
           f"{'cluster(s) ' + ', '.join(map(str, dh)) if dh else 'no cluster'}")
    (hold if rb["verdict"] == "reproduced" else nothold).append(txt)
    # ---- Track B vs Track A, CVAE tariff effects
    keys = ("high_in_pct", "high_pre4_pct", "high_post4_pct", "low_in_pct")
    indist = [c for c in CL if all(mr[str(c)]["A_vs_B_mannwhitney_p"][k] > 0.05 for k in keys)]
    sel = []
    for c in CL:
        for scen in ("Low", "High"):
            cell = ps["cvae"][scen]["in_window"][str(c)]
            sel.append(f"c{c} {scen} A {cell['trackA']['pct']:+.1f}% vs B {cell['trackB']['pct']:+.1f}% "
                       f"({cell['attribution']['label']})")
    p3 = mr["3"]["A_vs_B_mannwhitney_p"]["high_in_pct"]
    out["trackB_vs_trackA"]["tariff_effects_cvae"] = (
        "restart distributions of the in-window, pre4 and post effects (Mann-Whitney A vs B, all four p > 0.05) "
        f"indistinguishable in {'cluster(s) ' + ', '.join(map(str, indist)) if indist else 'no cluster'}; selected "
        "decoders, in-window effect and attribution: " + "; ".join(sel) + f"; cluster 3 restart median High effect A "
        f"{mr['3']['A']['median_high_in_pct']:.1f}% vs B {mr['3']['B']['median_high_in_pct']:.1f}% (p = {p3:.3f})")
    g3 = ps["gam"]["High"]["in_window"]["3"]
    out["trackB_vs_trackA"]["tariff_schedule"] = (
        "Track B's tariff file has 6 extra High 35-46 training days (Nov-Dec) and other event days without "
        f"consumption response. They dilute the measured cluster-3 High effect (data, real blocks): "
        f"{sv['3']['High']['trackB_our_aggregates_xlsx_schedule']['pct']:.1f}% with the xlsx schedule vs "
        f"{sv['3']['High']['our_aggregates_authors_schedule']['pct']:.1f}% for the same households with the authors' "
        f"schedule (Track A {sv['3']['High']['trackA_authors_aggregates_authors_schedule']['pct']:.1f}%), and plausibly "
        f"the GAM's High {LE.WINDOW_SPAN['High']} effect ({g3['trackB']['pct']:.1f}% in B vs {g3['trackA']['pct']:.1f}% "
        "in A) and the CVAE restarts' High effects")
    return out


def tariff_checks() -> dict:
    return {"mechanism_restarts_high_in_pct_vs_effects_csv_max_abs_diff": MR["max_abs_check_high_in_pct_vs_effects_csv"],
            "scenario_arrays_regenerated_by_21_tariff_effects": {
                t: {str(c): PAT[t]["clusters"][str(c)].get("max_abs_diff_vs_saved_scenarios") for c in CL}
                for t in ("A", "B")}}


def finish_claims(cl: dict, f4: dict, ef: dict, only=None) -> tuple[dict, dict]:
    """Same-rule Track A verdicts for the claims in `only` (all if None), the headline and its conclusion."""
    va, vb_ = verdicts_same_rule("A", f4, ef), verdicts_same_rule("B", f4, ef)
    for k, rec in cl.items():
        if only is not None and k not in only:
            continue
        base = k[:-2]
        assert rec["verdict"] == vb_[base], (k, rec["verdict"], vb_[base])
        rec["verdict_trackA_same_rule"] = va[base]
        rec["same_verdict_as_trackA"] = va[base] == vb_[base]
    hl = headline(f4, None, ef, cl)
    diff = [k for k, rec in cl.items() if not rec["same_verdict_as_trackA"]]
    hl["trackB_vs_trackA"]["verdicts_differing_from_trackA_same_rule"] = {
        k: {"trackA": cl[k]["verdict_trackA_same_rule"], "trackB": cl[k]["verdict"]} for k in diff}
    labels = [ef["paper_scenarios"]["cvae"][s]["in_window"][str(c)]["attribution"]["label"]
              for s in ("Low", "High") for c in CL]
    lab_counts = {lb: labels.count(lb) for lb in dict.fromkeys(labels)}
    hl["trackB_vs_trackA"]["conclusion"] = (
        "Under the same criteria Track B gives the same verdict as Track A on "
        f"{len(cl) - len(diff)} of {len(cl)} claims" + (f"; it differs on {', '.join(diff)}" if diff else "") +
        ". Fig. 4 differences come from the data (cluster composition) and are small except for cluster 2 (variogram "
        "24-28% below the paper in Track B). CVAE in-window tariff effects, Track B vs Track A (8 cells = 4 clusters x "
        f"Low/High, paper windows Low {LE.WINDOW_SPAN['Low']}, High {LE.WINDOW_SPAN['High']}): attribution "
        + ", ".join(f"{k} {v}" for k, v in lab_counts.items()) + "; the tariff schedule also dilutes Track B's measured "
        "cluster-3 High response in the data.")
    return cl, hl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", default="all", choices=["all", "effects"])
    args = ap.parse_args()
    windows = {k: {"slots": v, "clock_gmt": LE.WINDOW_CLOCK[k]} for k, v in LE.WINDOWS.items()}
    if args.sections == "effects":
        res = J(OUTJ)
        f4, dg = res["fig4_scores"], res["distribution_diagnostics"]
        ef = effects_section()
        new = claims(f4, dg, ef)
        cl = {k: (new[k] if k in TARIFF_CLAIMS else res["claims"][k]) for k in res["claims"]}
        cl, hl = finish_claims(cl, f4, ef, only=TARIFF_CLAIMS)
        res["headline"] = hl
        res["tariff_effects"] = ef
        res["claims"] = cl
        res["checks"].update(tariff_checks())
        res["meta"]["scenario_windows"] = windows
        res["meta"]["rebuild_scenario_windows"] = {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "rebuilt": ["tariff_effects", "claims: " + ", ".join(TARIFF_CLAIMS), "headline",
                        "checks: mechanism_restarts_high_in_pct_vs_effects_csv_max_abs_diff, "
                        "scenario_arrays_regenerated_by_21_tariff_effects", "figures/trackB_effects_AB_comparison.png"],
            "unchanged": ["fig4_scores", "distribution_diagnostics", "the other claims", "the other checks",
                          "figures/trackB_fig4_AB_comparison.png"],
            "why": "tariff-scenario windows corrected from Low 9-18 / High 39-43 to Low 9-19 / High 39-44 (REPRO_SPEC "
                   "section 7). Scores and sample diagnostics do not depend on the scenario windows (the sample arrays "
                   "are bitwise unchanged), so those sections are kept from the phase-2 build; the previous file is "
                   "outputs/archive_windows_9-18_39-43/comparison_AB.json",
            "command": "python -P outputs/trackB/aux_scripts/build_comparison.py --sections effects"}
        with open(OUTJ, "w") as f:
            json.dump(clean(res), f, indent=1)
        print("wrote", OUTJ, "(tariff sections only)")
        os.makedirs(FIGS, exist_ok=True)
        fig_effects(ef, FIGS + "/trackB_effects_AB_comparison.png")
        print("figure written")
        for k in TARIFF_CLAIMS:
            print(k, cl[k]["verdict"], "| Track A same rule:", cl[k]["verdict_trackA_same_rule"])
        return 0
    f4 = fig4_section()
    dg = diag_section()
    ef = effects_section()
    cl, hl = finish_claims(claims(f4, dg, ef), f4, ef)
    res = {"meta": {
        "scenario_windows": windows,
        "task": "phase 2 scores-trackB: Track B scored with the Track A scripts (20_scores.py, 21_tariff_effects.py), "
                "compared with Track A and the paper",
        "tracks": {"A": "authors' prepared data (UPSTREAM feathers): 1,007 ToU households in clusters 325/295/229/158, "
                        "NOAA temperatures, authors' tariff dummies",
                   "B": "our rebuild: 1,032 ToU households, our NMF run-(a) clusters relabelled to the authors' 0-3 "
                        "(285/299/299/149; matched-cluster series r = 0.986-0.994, ARI 0.58 on the shared 1,007), ERA5 "
                        "temperatures, Tariffs.xlsx 'loc' alignment; same model code, seeds and selection rule"},
        "paper_values": "Fig. 4 medians read off the boxplots (spec section 3) plus an independent reading; tariff "
                        "claims are qualitative (Sec. VII-C), encoded as claimed signs",
        "attribution_rules": {
            "cvae": "selected-model difference B - A split into (i) the shift of the whole 50-restart distribution "
                    "(Mann-Whitney A vs B) = data and (ii) the remainder = restart selection (training randomness). "
                    "label 'data' if the restart distributions differ (p < 0.05) and the remainder < 1 pooled restart "
                    "SD; 'data + training randomness' if they differ and the remainder >= 1 SD; 'training randomness "
                    "(restart selection)' if they do not differ but the selected models differ by >= 1 SD; otherwise "
                    "'no material difference'",
            "gam": "deterministic fit: 'data' if the paired day-bootstrap CI of the B - A median difference excludes "
                   "0 (scores) or the tracks' 95% CIs do not overlap (effects); otherwise 'no significant difference'",
            "data_reference": "no model involved: any A/B difference is data (cluster membership, tariff schedule)"},
        "sources": {"trackA": [A + "/scores_summary.json", A + "/effects_summary.json", A + "/mechanism_summary.json"],
                    "trackB": [B + "/scores_summary.json", B + "/effects_summary.json", B + "/mechanism_summary.json"],
                    "comparison_inputs": [B + "/scores_independent_check.json", B + "/scores_restarts_AB.csv",
                                          B + "/mechanism_restarts_AB.csv", B + "/mechanism_restarts_AB_summary.json",
                                          B + "/mechanism_data_check_AB.json"]},
        "cluster_sizes": SIZES},
        "checks": {"independent_scores_max_rel_diff_vs_scores_long": IC["stages"]["perday_check"]
                   ["max_rel_diff_vs_scores_long"],
                   "independent_diagnostics_max_abs_diff": IC["stages"]["perday_check"]["max_abs_diff_diagnostics"],
                   "selected_cvae_regenerated_from_restart_models_max_abs_diff": IC["stages"]["restarts"]
                   ["selected_regenerated_max_abs_diff_vs_saved"],
                   "gam_regenerated_phase1_seed_max_abs_diff": {k: v["regen_phase1_seed_max_abs_diff"]
                                                                for k, v in IC["stages"]["gam_mc"].items()}}
        | tariff_checks(),
        "headline": hl,
        "fig4_scores": f4, "distribution_diagnostics": dg, "tariff_effects": ef, "claims": cl}
    with open(OUTJ, "w") as f:
        json.dump(clean(res), f, indent=1)
    print("wrote", OUTJ)
    os.makedirs(FIGS, exist_ok=True)
    fig_fig4(f4, FIGS + "/trackB_fig4_AB_comparison.png")
    fig_effects(ef, FIGS + "/trackB_effects_AB_comparison.png")
    print("figures written")
    for k, v in cl.items():
        print(k, v.get("verdict"))


if __name__ == "__main__":
    sys.exit(main())
