"""Fig. 2 light check: weekly profiles of the authors' 4 ToU clusters (directly from power_consump_nmf),
levels and standardized-shape differences, vs random 4-cluster labellings of the 1,007 households."""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, pandas as pd
REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/"
U = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo/data/"
SCR = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-clustering/"
OUT = REPRO + "outputs/verify/v-clustering/"
pc = pd.read_feather(U + "power_consump_nmf.feather"); pc["d"] = pd.to_datetime(pc.date.astype(str))
cc = [f"consumption_{h:02d}" for h in range(48)]
dates = pd.to_datetime(sorted(pc.d.unique())); dow = dates.dayofweek.to_numpy()      # Mon=0
def week(Y365x48):
    return np.stack([Y365x48[dow == d].mean(0) for d in range(7)]).reshape(-1)       # Mon..Sun, 336
A = {c: pc[pc.cluster == c].sort_values("d")[cc].to_numpy(float) for c in range(4)}
sizes = np.array([325, 295, 229, 158.])
def summarize(series):  # dict c -> 365x48
    wk = {c: week(v) for c, v in series.items()}
    lvl = {c: float(v.mean()) for c, v in series.items()}
    std = {c: wk[c] / wk[c].mean() for c in wk}
    pop = np.mean([std[c] for c in std], axis=0)
    rms_dev = {c: float(np.sqrt(((std[c] - pop) ** 2).mean())) for c in std}
    cor = np.corrcoef(np.stack([std[c] for c in range(4)]))
    daily = {c: series[c].mean(0) for c in series}
    return dict(level_kwh=lvl, level_ratio=float(max(lvl.values()) / min(lvl.values())),
                rms_shape_dev=rms_dev, mean_rms_shape_dev=float(np.mean(list(rms_dev.values()))),
                min_offdiag_corr_std_week=float(cor[np.triu_indices(4, 1)].min()),
                evening_peak_kwh={c: float(daily[c][34:44].max()) for c in daily},
                evening_peak_std={c: float((daily[c] / daily[c].mean())[34:44].max()) for c in daily},
                night_base_std={c: float((daily[c] / daily[c].mean())[4:12].mean()) for c in daily},
                night_base_kwh={c: float(daily[c][4:12].mean()) for c in daily},
                weekend_over_weekday={c: float(series[c][dow >= 5].mean() / series[c][dow < 5].mean()) for c in series})
res = {"authors_clusters": summarize(A)}
z = np.load(SCR + "t3_inputs.npz"); X = z["X"]
rnd = []
for s in range(20):
    l = np.random.default_rng(500 + s).integers(0, 4, X.shape[0])
    rnd.append(summarize({c: X[l == c].mean(0).reshape(365, 48) for c in range(4)}))
res["random_20"] = dict(level_ratio_mean=float(np.mean([r["level_ratio"] for r in rnd])),
                        mean_rms_shape_dev_mean=float(np.mean([r["mean_rms_shape_dev"] for r in rnd])),
                        min_offdiag_corr_mean=float(np.mean([r["min_offdiag_corr_std_week"] for r in rnd])))
json.dump(res, open(OUT + "c4_fig2.json", "w"), indent=1)
print(json.dumps(res, indent=1))
