"""v-numbers: compare my recomputed numbers with scores_summary.json, effects_summary.json, comparison_AB.json and
phase-2 key numbers; write outputs/verify/v-numbers/recomputed.json."""
import json
import numpy as np

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
SP = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad"
MY = SP + "/verify/v-numbers"
FINAL = REPRO + "/outputs/verify/v-numbers/recomputed.json"

mine = json.load(open(MY + "/scores_recomputed.json"))
kl = json.load(open(MY + "/kl_recomputed.json"))
al = json.load(open(MY + "/alignment_checks.json"))
dead = json.load(open(MY + "/encoder_dead_units.json"))
cmpab = json.load(open(REPRO + "/outputs/comparison_AB.json"))
T = {"trackA": "A", "trackB": "B"}
rows = []


def add(src, name, theirs, ours, kind="rel", tol=None):
    if theirs is None or ours is None:
        rows.append({"source": src, "name": name, "theirs": theirs, "ours": ours, "flag": "missing"})
        return
    if kind == "rel":
        diff = abs(ours - theirs) / max(abs(theirs), 1e-300)
        flag = diff > (tol if tol is not None else 0.01)
    else:  # percentage points
        diff = abs(ours - theirs)
        flag = diff > (tol if tol is not None else 0.1)
    rows.append({"source": src, "name": name, "theirs": theirs, "ours": ours, "kind": kind, "diff": diff, "flag": bool(flag)})


for track in ("trackA", "trackB"):
    ss = json.load(open(REPRO + f"/outputs/{track}/scores_summary.json"))
    ef = json.load(open(REPRO + f"/outputs/{track}/effects_summary.json"))
    M = mine[track]
    src = f"{track}/scores_summary.json"
    for gen in ("cvae", "gam"):
        for c in range(4):
            k = f"{gen}_c{c}"
            S, my = ss["summary"][k], M["scores"][k]
            add(src, f"{k} rmse_rootmean median", S["rmse_rootmean"]["median"], my["rmse_rootmean"]["median"])
            add(src, f"{k} rmse_euclid median", S["rmse_euclid"]["median"], my["rmse_euclid"]["median"])
            add(src, f"{k} es_split(eq14) median", S["es_split"]["median"], my["es_eq14"]["median"])
            add(src, f"{k} es_u median", S["es_u"]["median"], my["es_ustat"]["median"])
            add(src, f"{k} vg ordered median", S["vg_p05_ordered"]["median"], my["vg_p05_ordered"]["median"])
            D = ss["distribution_diagnostics"]["per_model_cluster"][k]
            add(src, f"{k} spread_skill ratio_calibrated (pooled)", D["spread_skill"]["ratio_calibrated"], my["spread_skill_pooled_calibrated"])
            add(src, f"{k} mean_sample_sd", D["spread_skill"]["mean_sample_sd"], my["sample_sd_mean_kwh"])
            add(src, f"{k} rmse_of_sample_mean (pooled)", D["spread_skill"]["rmse_of_sample_mean"], my["rmse_pooled_kwh"])
            add(src, f"{k} coverage_central90 (pooled)", D["coverage_central90"], my["coverage90_pooled"])
            add(src, f"{k} coverage_central50 (pooled)", D["coverage_central50"], my["coverage50_pooled"])
            add(src, f"{k} pit var", D["pit"]["var"], my["pit_var"])
            add(src, f"{k} frac_outside_sample_range", D["pit"]["frac_outside_sample_range"], my["frac_outside_sample_range"])
    R, mr = ss["rmse_13pct"], M["rmse_difference"]
    add(src, "rmse13 median_over_all_cluster_days rel cvae %", R["candidates"]["median_over_all_cluster_days"]["diff_rel_to_cvae_pct"], mr["median_over_cluster_days_of_rmse"]["rel_cvae_pct"], "pp")
    add(src, "rmse13 median_over_all_cluster_days rel gam %", R["candidates"]["median_over_all_cluster_days"]["diff_rel_to_gam_pct"], mr["median_over_cluster_days_of_rmse"]["rel_gam_pct"], "pp")
    add(src, "rmse13 mean_of_per_cluster_medians rel cvae %", R["candidates"]["mean_of_per_cluster_medians"]["diff_rel_to_cvae_pct"], mr["mean_of_per_cluster_medians_ratio_of_means"]["rel_cvae_pct"], "pp")
    add(src, "rmse13 mean_of_per_cluster_medians rel gam %", R["candidates"]["mean_of_per_cluster_medians"]["diff_rel_to_gam_pct"], mr["mean_of_per_cluster_medians_ratio_of_means"]["rel_gam_pct"], "pp")
    add(src, "rmse13 mean_over_all_cluster_days rel cvae %", R["candidates"]["mean_over_all_cluster_days"]["diff_rel_to_cvae_pct"], mr["mean_over_cluster_days_of_rmse"]["rel_cvae_pct"], "pp")
    add(src, "rmse13 pooled_rms rel cvae %", R["candidates"]["pooled_rms_over_cluster_days"]["diff_rel_to_cvae_pct"], mr["pooled_rms"]["rel_cvae_pct"], "pp")
    add(src, "rmse13 paired median rel cvae %", R["paired"]["median_of_paired_rel_diff_rel_to_cvae_pct"], mr["median_over_cluster_days_of_paired_rel_diff"]["rel_cvae_pct"], "pp")
    add(src, "rmse13 paired median rel gam %", R["paired"]["median_of_paired_rel_diff_rel_to_gam_pct"], mr["median_over_cluster_days_of_paired_rel_diff"]["rel_gam_pct"], "pp")
    add(src, "rmse13 paired mean rel cvae %", R["paired"]["mean_of_paired_rel_diff_rel_to_cvae_pct"], mr["mean_over_cluster_days_of_paired_rel_diff"]["rel_cvae_pct"], "pp")
    add(src, "rmse13 paired mean rel gam %", R["paired"]["mean_of_paired_rel_diff_rel_to_gam_pct"], mr["mean_over_cluster_days_of_paired_rel_diff"]["rel_gam_pct"], "pp")
    add(src, "rmse13 frac_cluster_days_gam_better", R["paired"]["frac_cluster_days_gam_better"], mr["frac_cluster_days_gam_better"])
    for c in range(4):
        add(src, f"rmse13 per-cluster median diff rel cvae % c{c}", R["per_cluster_medians"][str(c)]["diff_rel_to_cvae_pct"], mr["per_cluster_median_rel_cvae_pct"][c], "pp")
        E = ss["es_claim_cvae_lower"]["per_cluster"][str(c)]
        me = M["es_cvae_vs_gam"][str(c)]
        for kk_t, kk_m in (("ratio_cvae_over_gam", "median_ratio_cvae_over_gam"), ("frac_days_cvae_lower", "frac_days_cvae_lower")):
            if kk_t in E:
                add(src, f"es_claim c{c} {kk_t}", E[kk_t], me[kk_m])
    src = f"{track}/effects_summary.json"
    for gen in ("cvae", "gam"):
        for c in range(4):
            P = ef["per_model_cluster"][f"{gen}_c{c}"]
            me = M["effects"][f"{gen}_c{c}"]
            for scen, reg, myk in (("Low", "in_window", "Low_in_9_18"), ("High", "in_window", "High_in_39_43"),
                                   ("High", "pre4", "High_pre4_35_38"), ("High", "post4", "High_post4_44_47"),
                                   ("High", "pre2", "High_pre2_37_38"), ("High", "post2", "High_post2_44_45")):
                rr = P[scen]["regions"][reg]
                add(src, f"{gen}_c{c} {scen} {reg} pct", rr["pct"], me[myk]["pct"], "pp")
                # CI comparison (different bootstrap draws -> loose tolerance 0.5 pp) and CI-excludes-0 agreement
                add(src, f"{gen}_c{c} {scen} {reg} ci_lo", rr["pct_ci95"][0], me[myk]["ci95"][0], "pp", 0.5)
                add(src, f"{gen}_c{c} {scen} {reg} ci_hi", rr["pct_ci95"][1], me[myk]["ci95"][1], "pp", 0.5)
                theirs_excl = (rr["pct_ci95"][0] > 0) or (rr["pct_ci95"][1] < 0)
                rows.append({"source": src, "name": f"{gen}_c{c} {scen} {reg} ci_excludes_0", "theirs": theirs_excl,
                             "ours": me[myk]["ci_excludes_0"], "flag": bool(theirs_excl != me[myk]["ci_excludes_0"])})

# comparison_AB.json
src = "comparison_AB.json"
F = cmpab["fig4_scores"]
for metric, myk in (("rmse", "rmse_rootmean"), ("es", "es_eq14"), ("vg", "vg_p05_ordered")):
    for gen in ("cvae", "gam"):
        for c in range(4):
            for track in ("trackA", "trackB"):
                add(src, f"fig4 {metric} {gen} c{c} {track}", F[metric][gen][str(c)][f"{track}_median"], mine[track]["scores"][f"{gen}_c{c}"][myk]["median"])
for track in ("trackA", "trackB"):
    t = T[track]
    R = F["rmse_13pct"][t]
    mr = mine[track]["rmse_difference"]
    add(src, f"rmse13 {t} median_over_all rel cvae", R["candidates_rel_to_cvae_pct"]["median_over_all_cluster_days"], mr["median_over_cluster_days_of_rmse"]["rel_cvae_pct"], "pp")
    add(src, f"rmse13 {t} median_over_all rel gam", R["candidates_rel_to_gam_pct"]["median_over_all_cluster_days"], mr["median_over_cluster_days_of_rmse"]["rel_gam_pct"], "pp")
    add(src, f"rmse13 {t} mean_of_per_cluster_medians rel cvae", R["candidates_rel_to_cvae_pct"]["mean_of_per_cluster_medians"], mr["mean_of_per_cluster_medians_ratio_of_means"]["rel_cvae_pct"], "pp")
    add(src, f"rmse13 {t} mean_of_per_cluster_medians rel gam", R["candidates_rel_to_gam_pct"]["mean_of_per_cluster_medians"], mr["mean_of_per_cluster_medians_ratio_of_means"]["rel_gam_pct"], "pp")
    add(src, f"rmse13 {t} paired median rel cvae", R["paired"]["median_of_paired_rel_diff_rel_to_cvae_pct"], mr["median_over_cluster_days_of_paired_rel_diff"]["rel_cvae_pct"], "pp")
    rng_t = R.get("range_of_all_our_readings_pct")
    if rng_t:
        allr = [mr[k]["rel_cvae_pct"] for k in ("median_over_cluster_days_of_rmse", "mean_over_cluster_days_of_rmse", "pooled_rms", "mean_of_per_cluster_medians_ratio_of_means", "median_over_cluster_days_of_paired_rel_diff", "mean_over_cluster_days_of_paired_rel_diff")] + \
               [mr[k]["rel_gam_pct"] for k in ("median_over_cluster_days_of_rmse", "mean_over_cluster_days_of_rmse", "pooled_rms", "mean_of_per_cluster_medians_ratio_of_means", "median_over_cluster_days_of_paired_rel_diff", "mean_over_cluster_days_of_paired_rel_diff")]
        add(src, f"rmse13 {t} range min", rng_t[0], min(allr), "pp")
        add(src, f"rmse13 {t} range max", rng_t[1], max(allr), "pp")
    E = F["es_cvae_lower_claim"][t]["per_cluster"]
    for c in range(4):
        me = mine[track]["es_cvae_vs_gam"][str(c)]
        add(src, f"es ratio {t} c{c}", E[str(c)]["ratio_cvae_over_gam"], me["median_ratio_cvae_over_gam"])
        add(src, f"es frac days cvae lower {t} c{c}", E[str(c)]["frac_days_cvae_lower"], me["frac_days_cvae_lower"], "pp", 0.001)
DD = cmpab["distribution_diagnostics"]
for gen in ("cvae", "gam"):
    for c in range(4):
        for track in ("trackA", "trackB"):
            add(src, f"spread_skill {gen} c{c} {track}", DD["spread_skill_ratio_calibrated"][gen][str(c)][track], mine[track]["scores"][f"{gen}_c{c}"]["spread_skill_pooled_calibrated"])
            add(src, f"coverage90 {gen} c{c} {track}", DD["coverage_central90"][gen][str(c)][track], mine[track]["scores"][f"{gen}_c{c}"]["coverage90_pooled"])
PS = cmpab["tariff_effects"]["paper_scenarios"]
for gen in ("cvae", "gam"):
    for scen, reg, myk in (("Low", "in_window", "Low_in_9_18"), ("High", "in_window", "High_in_39_43"), ("High", "pre4", "High_pre4_35_38"),
                           ("High", "post4", "High_post4_44_47"), ("High", "pre2", "High_pre2_37_38"), ("High", "post2", "High_post2_44_45")):
        for c in range(4):
            for track in ("trackA", "trackB"):
                add(src, f"effect {gen} {scen} {reg} c{c} {track}", PS[gen][scen][reg][str(c)][track]["pct"], mine[track]["effects"][f"{gen}_c{c}"][myk]["pct"], "pp")

# phase-2 key numbers as printed (rounded) -> tolerance = half a unit of the last printed digit or 1%
P2 = [
    ("A rmse cvae", [0.0191, 0.0223, 0.0181, 0.0235], lambda: [mine["trackA"]["scores"][f"cvae_c{c}"]["rmse_rootmean"]["median"] for c in range(4)], 0.00005),
    ("A rmse gam", [0.0142, 0.0162, 0.0148, 0.0187], lambda: [mine["trackA"]["scores"][f"gam_c{c}"]["rmse_rootmean"]["median"] for c in range(4)], 0.00005),
    ("B rmse cvae", [0.0195, 0.0246, 0.0173, 0.0244], lambda: [mine["trackB"]["scores"][f"cvae_c{c}"]["rmse_rootmean"]["median"] for c in range(4)], 0.00005),
    ("B rmse gam", [0.0144, 0.0171, 0.0129, 0.0188], lambda: [mine["trackB"]["scores"][f"gam_c{c}"]["rmse_rootmean"]["median"] for c in range(4)], 0.00005),
    ("A es cvae", [0.125, 0.147, 0.113, 0.153], lambda: [mine["trackA"]["scores"][f"cvae_c{c}"]["es_eq14"]["median"] for c in range(4)], 0.0005),
    ("A es gam", [0.074, 0.084, 0.076, 0.094], lambda: [mine["trackA"]["scores"][f"gam_c{c}"]["es_eq14"]["median"] for c in range(4)], 0.0005),
    ("B es cvae", [0.123, 0.162, 0.107, 0.162], lambda: [mine["trackB"]["scores"][f"cvae_c{c}"]["es_eq14"]["median"] for c in range(4)], 0.0005),
    ("B es gam", [0.075, 0.089, 0.068, 0.094], lambda: [mine["trackB"]["scores"][f"gam_c{c}"]["es_eq14"]["median"] for c in range(4)], 0.0005),
    ("A vg cvae", [5.63, 6.61, 6.62, 9.02], lambda: [mine["trackA"]["scores"][f"cvae_c{c}"]["vg_p05_ordered"]["median"] for c in range(4)], 0.005),
    ("A vg gam", [3.76, 3.97, 4.79, 5.81], lambda: [mine["trackA"]["scores"][f"gam_c{c}"]["vg_p05_ordered"]["median"] for c in range(4)], 0.005),
    ("B vg cvae", [5.82, 6.97, 5.31, 9.77], lambda: [mine["trackB"]["scores"][f"cvae_c{c}"]["vg_p05_ordered"]["median"] for c in range(4)], 0.005),
    ("B vg gam", [4.04, 4.11, 3.62, 6.05], lambda: [mine["trackB"]["scores"][f"gam_c{c}"]["vg_p05_ordered"]["median"] for c in range(4)], 0.005),
    ("A es ratio cvae/gam", [1.70, 1.74, 1.48, 1.62], lambda: [mine["trackA"]["es_cvae_vs_gam"][str(c)]["median_ratio_cvae_over_gam"] for c in range(4)], 0.005),
    ("B es ratio cvae/gam", [1.65, 1.83, 1.58, 1.72], lambda: [mine["trackB"]["es_cvae_vs_gam"][str(c)]["median_ratio_cvae_over_gam"] for c in range(4)], 0.005),
    ("A spread/skill cvae", [0.079, 0.091, 0.134, 0.094], lambda: [mine["trackA"]["scores"][f"cvae_c{c}"]["spread_skill_pooled_calibrated"] for c in range(4)], 0.0005),
    ("A spread/skill gam", [1.45, 1.54, 1.37, 1.39], lambda: [mine["trackA"]["scores"][f"gam_c{c}"]["spread_skill_pooled_calibrated"] for c in range(4)], 0.005),
    ("B spread/skill cvae", [0.10, 0.12, 0.14, 0.08], lambda: [mine["trackB"]["scores"][f"cvae_c{c}"]["spread_skill_pooled_calibrated"] for c in range(4)], 0.005),
    ("B spread/skill gam", [1.44, 1.55, 1.49, 1.30], lambda: [mine["trackB"]["scores"][f"gam_c{c}"]["spread_skill_pooled_calibrated"] for c in range(4)], 0.005),
    ("A cov90 gam", [0.97, 0.97, 0.96, 0.96], lambda: [mine["trackA"]["scores"][f"gam_c{c}"]["coverage90_pooled"] for c in range(4)], 0.005),
    ("B cov90 cvae %", [11.3, 8.7, 11.4, 7.2], lambda: [100 * mine["trackB"]["scores"][f"cvae_c{c}"]["coverage90_pooled"] for c in range(4)], 0.05),
    ("B cov90 gam %", [97.2, 97.7, 96.8, 94.9], lambda: [100 * mine["trackB"]["scores"][f"gam_c{c}"]["coverage90_pooled"] for c in range(4)], 0.05),
    ("A cvae Low %", [-2.4, 4.9, -5.7, 50.3], lambda: [mine["trackA"]["effects"][f"cvae_c{c}"]["Low_in_9_18"]["pct"] for c in range(4)], 0.05),
    ("A cvae High %", [3.8, -9.3, -4.3, -25.1], lambda: [mine["trackA"]["effects"][f"cvae_c{c}"]["High_in_39_43"]["pct"] for c in range(4)], 0.05),
    ("A gam Low %", [3.7, 4.7, 2.2, 34.0], lambda: [mine["trackA"]["effects"][f"gam_c{c}"]["Low_in_9_18"]["pct"] for c in range(4)], 0.05),
    ("A gam High %", [-3.5, -5.4, -3.6, -25.5], lambda: [mine["trackA"]["effects"][f"gam_c{c}"]["High_in_39_43"]["pct"] for c in range(4)], 0.05),
    ("B cvae Low %", [-5.9, 0.5, -3.3, 38.2], lambda: [mine["trackB"]["effects"][f"cvae_c{c}"]["Low_in_9_18"]["pct"] for c in range(4)], 0.05),
    ("B cvae High %", [-0.2, 1.4, -4.7, -2.5], lambda: [mine["trackB"]["effects"][f"cvae_c{c}"]["High_in_39_43"]["pct"] for c in range(4)], 0.05),
    ("B gam Low %", [3.2, 2.9, 2.4, 30.1], lambda: [mine["trackB"]["effects"][f"gam_c{c}"]["Low_in_9_18"]["pct"] for c in range(4)], 0.05),
    ("B gam High %", [-3.0, -3.4, -3.6, -19.2], lambda: [mine["trackB"]["effects"][f"gam_c{c}"]["High_in_39_43"]["pct"] for c in range(4)], 0.05),
    ("A cvae pre4 % c1-c3 (c0 +5.2)", [5.2, -10.5, -3.8, -25.3], lambda: [mine["trackA"]["effects"][f"cvae_c{c}"]["High_pre4_35_38"]["pct"] for c in range(4)], 0.05),
    ("A cvae post4 % (c0 +6.3)", [6.3, -12.6, -4.0, -21.9], lambda: [mine["trackA"]["effects"][f"cvae_c{c}"]["High_post4_44_47"]["pct"] for c in range(4)], 0.05),
    ("B cvae c3 pre/post %", [-0.2, -1.7], lambda: [mine["trackB"]["effects"]["cvae_c3"]["High_pre4_35_38"]["pct"], mine["trackB"]["effects"]["cvae_c3"]["High_post4_44_47"]["pct"]], 0.05),
    ("B cvae c2 pre2/post2/in %", [-5.2, -5.3, -4.7], lambda: [mine["trackB"]["effects"]["cvae_c2"]["High_pre2_37_38"]["pct"], mine["trackB"]["effects"]["cvae_c2"]["High_post2_44_45"]["pct"], mine["trackB"]["effects"]["cvae_c2"]["High_in_39_43"]["pct"]], 0.05),
    ("A rmse13 median rel cvae/gam %", [22.5, 29.1], lambda: [mine["trackA"]["rmse_difference"]["median_over_cluster_days_of_rmse"]["rel_cvae_pct"], mine["trackA"]["rmse_difference"]["median_over_cluster_days_of_rmse"]["rel_gam_pct"]], 0.05),
    ("B rmse13 median rel cvae/gam %", [23.1, 30.0], lambda: [mine["trackB"]["rmse_difference"]["median_over_cluster_days_of_rmse"]["rel_cvae_pct"], mine["trackB"]["rmse_difference"]["median_over_cluster_days_of_rmse"]["rel_gam_pct"]], 0.05),
    ("A rmse13 mean of per-cluster medians % (rel cvae/gam)", [23.0, 29.8], lambda: [mine["trackA"]["rmse_difference"]["mean_of_per_cluster_medians_ratio_of_means"]["rel_cvae_pct"], mine["trackA"]["rmse_difference"]["mean_of_per_cluster_medians_ratio_of_means"]["rel_gam_pct"]], 0.05),
    ("B per-cluster GAM lower %, mean of medians", [25.9, 30.6, 25.4, 22.7, 26.2], lambda: mine["trackB"]["rmse_difference"]["per_cluster_median_rel_cvae_pct"] + [mine["trackB"]["rmse_difference"]["mean_of_per_cluster_medians_ratio_of_means"]["rel_cvae_pct"]], 0.05),
    ("A KL selected (phase 1 'about 1e-6')", [1e-6] * 4, lambda: [kl["trackA"][str(c)]["train_kl_per_day_mean_nats"] for c in range(4)], None),
    ("B KL selected 2.4e-7..9.7e-7", [9.7e-7, 2.4e-7, 3.0e-7, 4.5e-7], lambda: [kl["trackB"][str(c)]["train_kl_per_day_mean_nats"] for c in range(4)], 0.05e-7),
]
p2rows = []
for name, theirs, f, tol in P2:
    ours = f()
    if tol is None:
        within = [bool(0.1 * t <= o <= 10 * t) for t, o in zip(theirs, ours)]
        p2rows.append({"name": name, "theirs": theirs, "ours": ours, "rule": "same order of magnitude", "flags": [not w for w in within]})
    else:
        fl = [bool(abs(o - t) > max(tol, 0.01 * abs(t))) for t, o in zip(theirs, ours)]
        p2rows.append({"name": name, "theirs": theirs, "ours": ours, "rule": f"|diff| <= max({tol}, 1% rel)", "flags": fl})

flagged = [r for r in rows if r.get("flag")]
flagged_p2 = [r for r in p2rows if any(r["flags"])]
print("n comparisons json:", len(rows), " flagged:", len(flagged))
for r in flagged:
    print("  FLAG", r)
print("n phase-2 key number groups:", len(p2rows), " flagged groups:", len(flagged_p2))
for r in flagged_p2:
    print("  FLAG P2", r["name"], "theirs", r["theirs"], "ours", np.round(r["ours"], 6).tolist(), r["flags"])
maxrel = max((r["diff"] for r in rows if r.get("kind") == "rel"), default=None)
maxpp = max((r["diff"] for r in rows if r.get("kind") == "pp" and "ci_" not in r["name"]), default=None)
print("max rel diff (point values):", maxrel, " max pp diff (point values, excl. CIs):", maxpp)


# ---------------- compact recomputed.json for the notebook ----------------
def r4(x):
    return [round(float(v), 6) for v in x]


out = {"meta": {"verifier": "v-numbers", "method": "independent numpy/scipy recomputation from saved arrays and observed test data; no lib/scores.py, lib/effects.py or scorer scripts; KL from own numpy forward pass of the saved state_dicts (torch only to read the files)",
                "scripts": [MY + "/recompute_scores.py", MY + "/alignment_check.py", MY + "/kl_check.py", MY + "/compare.py"],
                "full_outputs": [MY + "/scores_recomputed.json", MY + "/alignment_checks.json", MY + "/kl_recomputed.json", MY + "/encoder_dead_units.json"],
                "definitions": {"rmse": "root-mean over 48 half-hours of (mean of 200 samples - obs), median over 92 test days; euclid = sqrt(48) x that",
                                "es": "eq. 14: (2/N) sum_{i<=100} ||X_i-y|| - (1/N) sum_{i<=100} ||X_i-X_{100+i}||, Euclidean, kWh",
                                "vg": "p = 0.5, all ordered pairs h != h'",
                                "spread_skill": "pooled: sqrt(mean per-cell sample variance, ddof=1)/sqrt(mean squared error of sample mean) x sqrt(201/200); per-day median also given",
                                "coverage90": "share of (day, slot) obs inside [q05, q95] of the 200 samples (numpy linear quantiles), pooled; per-day median also given",
                                "effects": "100 x mean over samples and days of (scenario - Normal) / mean of Normal, same slots; CI = percentile bootstrap over days, B = 2000",
                                "kl": "per training day, sum over the 4 latent dims of KL(N(mu, sd^2) || N(0,1)), nats"}},
       "comparison": {"n_point_comparisons": len(rows), "n_flagged": len(flagged), "flagged": flagged,
                      "max_rel_diff_point_values": maxrel, "max_pp_diff_effects": maxpp,
                      "phase2_key_numbers": p2rows}}
for track in ("trackA", "trackB"):
    M = mine[track]
    o = {"day_axis": {"test_dates_csv_equals_sorted_obs_dates_all_clusters": all(v["sorted_dates_equal_test_dates_csv"] for v in M["day_order_checks"].values()),
                      "obs_file_order_ascending_all_clusters": all(v["file_order_ascending"] for v in M["day_order_checks"].values()),
                      "cvae_samples_equal_scenNormal_exactly_on_all_normal_days_only": all(al[track][str(c)]["cvae_zero_diff_days_equal_all_normal_days"] for c in range(4)),
                      "cvae_fingerprint_match_frac_aligned_vs_shift1": [1.0, min(al[track][str(c)]["cvae_match_frac_shift_prev"] for c in range(4))],
                      "gam_ratio_mean_over_normal_at_real_low_high_slots_aligned": [[round(al[track][str(c)]["gam_ratio_aligned"]["low_slots"], 4), round(al[track][str(c)]["gam_ratio_aligned"]["high_slots"], 4)] for c in range(4)],
                      "gam_ratio_mean_over_normal_at_real_low_high_slots_shift_prev": [[round(al[track][str(c)]["gam_ratio_shift_prev"]["low_slots"], 4), round(al[track][str(c)]["gam_ratio_shift_prev"]["high_slots"], 4)] for c in range(4)],
                      "corr_daily_mean_changes_gam_aligned_prev_next": [[round(al[track][str(c)][f"gam_corr_daily_mean_changes_{k}"], 3) for k in ("aligned", "shift_prev", "shift_next")] for c in range(4)],
                      "corr_daily_mean_changes_cvae_aligned_prev_next": [[round(al[track][str(c)][f"cvae_corr_daily_mean_changes_{k}"], 3) for k in ("aligned", "shift_prev", "shift_next")] for c in range(4)],
                      "median_rmse_ratio_obs_shift_prev_next_over_aligned": {k: [round(M["alignment"][f"{k}_c{c}"]["ratio_prev_over_aligned"], 3) for c in range(4)] + [round(M["alignment"][f"{k}_c{c}"]["ratio_next_over_aligned"], 3) for c in range(4)] for k in ("cvae", "gam")},
                      "cluster_cross_rmse_diagonal_is_min": {k: M["alignment"][f"{k}_cluster_cross_median_rmse"]["diagonal_is_row_min"] for k in ("cvae", "gam")},
                      "cvae_arrays_regenerated_from_selected_decoders_max_abs_kwh": max(kl[track][str(c)]["regenerated_with_documented_z_max_abs_diff_kwh"] for c in range(4))},
         "fig4_medians": {m: {g: r4(M["fig4"][m][g]["median"]) for g in ("cvae", "gam")} for m in ("rmse", "es", "vg")},
         "fig4_ratio_to_paper": {m: {g: r4(M["fig4"][m][g]["ratio_to_paper"]) for g in ("cvae", "gam")} for m in ("rmse", "es", "vg")},
         "rmse_euclid_medians": {g: r4([M["scores"][f"{g}_c{c}"]["rmse_euclid"]["median"] for c in range(4)]) for g in ("cvae", "gam")},
         "es_ustat_medians": {g: r4([M["scores"][f"{g}_c{c}"]["es_ustat"]["median"] for c in range(4)]) for g in ("cvae", "gam")},
         "es_cvae_over_gam_median_ratio": r4([M["es_cvae_vs_gam"][str(c)]["median_ratio_cvae_over_gam"] for c in range(4)]),
         "es_frac_days_cvae_lower": r4([M["es_cvae_vs_gam"][str(c)]["frac_days_cvae_lower"] for c in range(4)]),
         "es_wilcoxon_p": [float(f"{M['es_cvae_vs_gam'][str(c)]['wilcoxon_p']:.3g}") for c in range(4)],
         "rmse_difference_pct": M["rmse_difference"],
         "spread_skill_pooled": {g: r4([M["scores"][f"{g}_c{c}"]["spread_skill_pooled_calibrated"] for c in range(4)]) for g in ("cvae", "gam")},
         "spread_skill_perday_median": {g: r4([M["scores"][f"{g}_c{c}"]["spread_skill_perday_median"] for c in range(4)]) for g in ("cvae", "gam")},
         "coverage90_pooled": {g: r4([M["scores"][f"{g}_c{c}"]["coverage90_pooled"] for c in range(4)]) for g in ("cvae", "gam")},
         "coverage90_perday_median": {g: r4([M["scores"][f"{g}_c{c}"]["coverage90_perday_median"] for c in range(4)]) for g in ("cvae", "gam")},
         "coverage50_pooled": {g: r4([M["scores"][f"{g}_c{c}"]["coverage50_pooled"] for c in range(4)]) for g in ("cvae", "gam")},
         "pit_var": {g: r4([M["scores"][f"{g}_c{c}"]["pit_var"] for c in range(4)]) for g in ("cvae", "gam")},
         "frac_outside_sample_range": {g: r4([M["scores"][f"{g}_c{c}"]["frac_outside_sample_range"] for c in range(4)]) for g in ("cvae", "gam")},
         "mean_sample_sd_kwh": {g: r4([M["scores"][f"{g}_c{c}"]["sample_sd_mean_kwh"] for c in range(4)]) for g in ("cvae", "gam")},
         "effects_pct": {g: {reg: r4([M["effects"][f"{g}_c{c}"][reg]["pct"] for c in range(4)]) for reg in ("Low_in_9_18", "High_in_39_43", "High_pre4_35_38", "High_post4_44_47", "High_pre2_37_38", "High_post2_44_45")} for g in ("cvae", "gam")},
         "effects_ci95": {g: {reg: [[round(x, 3) for x in M["effects"][f"{g}_c{c}"][reg]["ci95"]] for c in range(4)] for reg in ("Low_in_9_18", "High_in_39_43", "High_pre4_35_38", "High_post4_44_47")} for g in ("cvae", "gam")},
         "gam_effect_outside_window_max_abs_kwh": max(max(M["effects"][f"gam_c{c}"]["max_abs_diff_outside_window_Low"], M["effects"][f"gam_c{c}"]["max_abs_diff_outside_window_High"]) for c in range(4)),
         "posterior_collapse_selected": {"train_kl_nats_per_day": [kl[track][str(c)]["train_kl_per_day_mean_nats"] for c in range(4)],
                                         "posterior_sd_range": [min(kl[track][str(c)]["posterior_sd_min"] for c in range(4)), max(kl[track][str(c)]["posterior_sd_max"] for c in range(4))],
                                         "posterior_mean_abs_max": max(kl[track][str(c)]["posterior_mean_abs_max"] for c in range(4)),
                                         "encoder_hidden_units_active_on_any_train_day_of_15": [dead[f"{track}_c{c}"]["enc_hidden_units_ever_active"] for c in range(4)],
                                         "test_kl_nats_per_day": [kl[track][str(c)]["test_kl_per_day_mean_nats"] for c in range(4)],
                                         "latent_unit_change_mean_abs_output_wh": [[round(1000 * v, 3) for v in kl[track][str(c)]["latent_unit_change_mean_abs_output_kwh"]] for c in range(4)],
                                         "scaling_equals_train_minmax": all(kl[track][str(c)]["scaling_meta_equals_train_minmax"] for c in range(4))}}
    out[track] = o
json.dump(out, open(FINAL, "w"), indent=1)
print("wrote", FINAL)
