"""Post-analysis of the final eta grid (reads the deliverable CSV; writes nothing outside scratch).
(a) determinism: final runs vs the previous attempt's runs (weights + every JSON field except runtimes);
(b) selection-aware view: per (track, cluster, eta) the seed with the lowest authors' selection metric
    (test_mse_post) among seeds 0-4, vs the eta = 10 selected seed;
(c) per-eta pooled summaries over clusters (paired changes vs eta = 10)."""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
SCR = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-collapse"
NEW = REPRO + "/outputs/verify/v-collapse/runs"
OLD = SCR + "/prev_attempt/runs"
df = pd.read_csv(REPRO + "/outputs/verify/v-collapse/eta_sensitivity.csv")
out = {}

# (a) determinism vs previous attempt
skip = {"runtime_train_s", "runtime_total_s"}
nfiles, wmax, field_mism = 0, 0.0, {}
for p in sorted(glob.glob(NEW + "/*.json")):
    b = os.path.basename(p)
    q = OLD + "/" + b
    if not os.path.exists(q):
        continue
    a, o = json.load(open(p)), json.load(open(q))
    for k in set(a) | set(o):
        if k in skip:
            continue
        if a.get(k) != o.get(k):
            field_mism[k] = field_mism.get(k, 0) + 1
    sa = torch.load(p[:-5] + ".pt", weights_only=True)["state_dict"]
    so = torch.load(q[:-5] + ".pt", weights_only=True)["state_dict"]
    wmax = max(wmax, max(float((sa[n] - so[n]).abs().max()) for n in sa))
    nfiles += 1
out["determinism_vs_previous_attempt"] = {"n_models_compared": nfiles, "max_abs_weight_diff": wmax,
                                          "json_fields_with_mismatch": field_mism}
print(out["determinism_vs_previous_attempt"])

# (b) selection-aware: lowest test_mse_post among the 5 seeds
sel = df.loc[df.groupby(["track", "cluster", "eta"])["test_mse_post"].idxmin()].copy()
base = sel.loc[sel["eta"] == 10.0].set_index(["track", "cluster"])
for col in ("rmse_median_kwh", "spread_skill", "coverage90", "es_median", "vg_median"):
    sel[f"{col}_vs_eta10sel_pct"] = [100 * (r[col] / base.loc[(r["track"], r["cluster"]), col] - 1) for _, r in sel.iterrows()]
cols = ["track", "cluster", "eta", "seed", "test_mse_post", "test_mse_prior", "active_units", "spread_skill", "coverage90",
        "rmse_median_kwh", "rmse_median_kwh_vs_eta10sel_pct", "es_median_vs_eta10sel_pct", "vg_median_vs_eta10sel_pct"]
print(sel[cols].round(4).to_string(index=False))
out["selected_by_test_mse_post"] = json.loads(sel[cols].to_json(orient="records"))

# (c) per track x eta: ranges over clusters of seed-medians, and pooled-over-everything medians
g = df.groupby(["track", "cluster", "eta"]).median(numeric_only=True).reset_index()
agg = g.groupby(["track", "eta"]).agg(
    active_min=("active_units", "min"), active_max=("active_units", "max"),
    kl_min=("train_kl", "min"), kl_max=("train_kl", "max"),
    ss_min=("spread_skill", "min"), ss_max=("spread_skill", "max"),
    cov90_min=("coverage90", "min"), cov90_max=("coverage90", "max"),
    rmse_chg_min=("rmse_median_kwh_rel_change_vs_eta10_pct", "min"),
    rmse_chg_max=("rmse_median_kwh_rel_change_vs_eta10_pct", "max"),
    es_chg_min=("es_median_rel_change_vs_eta10_pct", "min"), es_chg_max=("es_median_rel_change_vs_eta10_pct", "max"),
    vg_chg_min=("vg_median_rel_change_vs_eta10_pct", "min"), vg_chg_max=("vg_median_rel_change_vs_eta10_pct", "max"),
    tmp_post_min=("test_mse_post", "min"), tmp_post_max=("test_mse_post", "max"),
    tmp_prior_min=("test_mse_prior", "min"), tmp_prior_max=("test_mse_prior", "max"),
    aggv_min=("agg_post_var_min", "min"))
print(agg.round(4).to_string())
out["by_track_eta"] = json.loads(agg.reset_index().to_json(orient="records"))
# share of individual models with >= 1 active unit per eta
share = df.groupby(["eta"]).apply(lambda x: float((x["active_units"] > 0).mean()), include_groups=False)
print("share of models with an active unit by eta:", share.to_dict())
out["share_models_active_by_eta"] = {str(k): v for k, v in share.items()}
# max spread/skill and cov90 over ALL models
print("max spread/skill over all 280 models:", df["spread_skill"].max(), "max cov90:", df["coverage90"].max())
out["max_spread_skill_any_model"] = float(df["spread_skill"].max())
out["max_coverage90_any_model"] = float(df["coverage90"].max())
# models with spread/skill >= 0.5 : RMSE change distribution
m = df.loc[df["spread_skill"] >= 0.5]
print("models with spread/skill >= 0.5:", len(m), "RMSE change vs eta10 (paired) min/median/max:",
      m["rmse_median_kwh_rel_change_vs_eta10_pct"].min(), m["rmse_median_kwh_rel_change_vs_eta10_pct"].median(),
      m["rmse_median_kwh_rel_change_vs_eta10_pct"].max())
out["models_ss_ge_0p5_rmse_change_pct"] = [float(m["rmse_median_kwh_rel_change_vs_eta10_pct"].min()),
                                           float(m["rmse_median_kwh_rel_change_vs_eta10_pct"].median()),
                                           float(m["rmse_median_kwh_rel_change_vs_eta10_pct"].max())]
json.dump(out, open(SCR + "/post_final.json", "w"), indent=1)
print("wrote", SCR + "/post_final.json")
