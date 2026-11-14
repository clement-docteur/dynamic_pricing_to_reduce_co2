"""v-numbers: compare my per-restart scenario effects with outputs/<track>/effects_all_restarts_regions.csv and the
phase-2 restart counts; append the summary to recomputed.json."""
import json
import numpy as np
import pandas as pd

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
MY = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-numbers"
FINAL = REPRO + "/outputs/verify/v-numbers/recomputed.json"
R = json.load(open(MY + "/restart_effects.json"))
cmpab = json.load(open(REPRO + "/outputs/comparison_AB.json"))
MAP = {"Low_in": "Low:in_window", "High_in": "High:in_window", "High_pre4": "High:pre4", "High_post4": "High:post4",
       "High_pre2": "High:pre2", "High_post2": "High:post2"}
out = {}
for track, t in (("trackA", "A"), ("trackB", "B")):
    st = pd.read_csv(REPRO + f"/outputs/{track}/effects_all_restarts_regions.csv")
    out[track] = {}
    for c in range(4):
        rows = R[track][str(c)]["restarts"]
        maxd = {}
        status_mismatch = 0
        for r in rows:
            s = st.loc[(st.cluster == c) & (st.seed == r["seed"])].iloc[0]
            for mk, sk in MAP.items():
                maxd[mk] = max(maxd.get(mk, 0.0), abs(r[mk] - s[f"{sk}:pct"]))
                if (s[f"{sk}:status"] == "appears") != r[mk + "_appears"]:
                    status_mismatch += 1
        ap = lambda k: np.array([r[k + "_appears"] for r in rows])
        side = (ap("High_pre2") | ap("High_pre4")) & (ap("High_post2") | ap("High_post4"))
        both = ap("Low_in") & ap("High_in")
        mine = {"n_direction_both_appear": int(both.sum()), "n_side_effect_in_output": int(side.sum()),
                "pct_Low_in_positive": float(100 * np.mean([r["Low_in"] > 0 for r in rows])),
                "pct_High_in_negative": float(100 * np.mean([r["High_in"] < 0 for r in rows])),
                "pct_High_pre4_negative": float(100 * np.mean([r["High_pre4"] < 0 for r in rows])),
                "median_High_in_pct": float(np.median([r["High_in"] for r in rows])),
                "median_Low_in_pct": float(np.median([r["Low_in"] for r in rows]))}
        theirs = cmpab["tariff_effects"]["restarts_classes"][str(c)][t]
        out[track][c] = {"mine": mine, "theirs_comparison_AB": {k: theirs[k] for k in ("n_direction_both_appear", "n_side_effect_in_output", "median_high_in_pct")},
                         "max_abs_pct_diff_vs_effects_all_restarts_csv": maxd, "n_status_mismatches_of_300": status_mismatch,
                         "selected_regen": R[track][str(c)]["summary"]["selected_model_regenerates_saved_scen_arrays_max_abs_kwh"],
                         "selected_seed": R[track][str(c)]["summary"]["selected_seed_by_weights"]}
        print(track, c, json.dumps(mine), "| theirs", json.dumps(out[track][c]["theirs_comparison_AB"]),
              "| max pct diff", {k: float(f"{v:.2e}") for k, v in maxd.items()}, "status mismatches", status_mismatch,
              "| sel", out[track][c]["selected_seed"], out[track][c]["selected_regen"])
fin = json.load(open(FINAL))
fin["restarts_50"] = {"method": "own numpy decoder over the 50 stored restart models per cluster; scenario conditions built from the test rows; z = default_rng([20200801, c]); CI = percentile bootstrap over days (B = 2000); side effect = (pre2 or pre4 appears) and (post2 or post4 appears), the phase-2 rule",
                      "results": out}
json.dump(fin, open(FINAL, "w"), indent=1)
print("appended to", FINAL)
