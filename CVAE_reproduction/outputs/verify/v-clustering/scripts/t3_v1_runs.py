"""Table 3 rebuilt under the identified definition V1 for our runs a/b/c (ToU and Std): NMF, NMF-PAM, Features, Rd."""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np, pandas as pd
REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/"
U = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo/data/"
SCR = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-clustering/"
OUT = REPRO + "outputs/verify/v-clustering/"
z = np.load(SCR + "t3_inputs.npz"); sp_auth = z["special"]
co = pd.read_parquet(REPRO + "data/conditions_ours.parquet").sort_values("date")
sp_loc = ((co[[f"low_loc_{h:02d}" for h in range(48)]].to_numpy() + co[[f"high_loc_{h:02d}" for h in range(48)]].to_numpy()) > 0).reshape(-1)
F = np.load(REPRO + "data/lcl2013_kwh_filled.npy", mmap_mode="r")
def V1(Y, l, k=4):
    n = len(l); c = Y.mean(0); nl = np.bincount(l, minlength=k).astype(float)
    C = np.stack([Y[l == j].mean(0) for j in range(k)])
    b1 = np.sqrt(((C - c) ** 2).sum(1))
    w1 = np.array([np.sqrt(((Y[l == j] - C[j]) ** 2).sum(1)).sum() for j in range(k)])
    return float((n - k) / (k - 1) * (nl * b1).sum() / (w1 / nl).sum())
paper = {"ToU": {"Rd": [4627, 4950, 5151], "Features": [40764, 6870, 7033], "NMF": [13432, 14088, 16070]},
         "Std": {"Rd": [5014, 4834, 4904], "Features": [5088, 4741, 4694], "NMF": [12971, 14934, 15460]}}
res = {"paper": paper}
for run in "abc":
    df = pd.read_parquet(REPRO + f"data/clusters_run_{run}.parquet")
    res[run] = {}
    for g in ["ToU", "Std"]:
        d = df[df.group.to_numpy(dtype=object) == g]
        X = np.asarray(F[d.row_in_filled.to_numpy()], dtype=np.float64)
        Xs = X / X.mean(1, keepdims=True)
        sp = sp_auth if run == "c" else sp_loc
        S = {"raw": X, "std": Xs, "special": Xs[:, sp]}
        r = {}
        for col in ["nmf", "nmf_pam", "features"]:
            r[col] = [V1(S[k], d[col].to_numpy()) for k in ["raw", "std", "special"]]
        rd = np.array([[V1(S[k], d[f"random_{s:02d}"].to_numpy()) for k in ["raw", "std", "special"]] for s in range(20)])
        r["Rd_mean"] = rd.mean(0).tolist(); r["Rd_sd"] = rd.std(0).tolist()
        r["NMF_vs_Features_special_pct"] = 100 * (r["nmf"][2] / r["features"][2] - 1)
        r["NMF_special_over_std"] = r["nmf"][2] / r["nmf"][1]
        r["orderings"] = {"Features>NMF raw": r["features"][0] > r["nmf"][0], "NMF>Features std": r["nmf"][1] > r["features"][1],
                          "NMF special>std": r["nmf"][2] > r["nmf"][1], "NMF>Rd all": all(r["nmf"][j] > r["Rd_mean"][j] for j in range(3))}
        r["special_mask_slots"] = int(sp.sum()); r["n"] = int(len(d))
        res[run][g] = r
        print(run, g, "n", len(d), {k: np.round(v, 0).tolist() if isinstance(v, list) else v for k, v in r.items() if k not in ("orderings",)})
        print("    ", r["orderings"])
json.dump(res, open(OUT + "t3_v1_runs.json", "w"), indent=1)
