"""Table 3: D1 (sklearn CH), D2 (eqs. 3-5) and new non-squared-norm variants, own implementation."""
import os, json, sys
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
from sklearn.metrics import calinski_harabasz_score
SCR = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-clustering/"
OUT = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/outputs/verify/v-clustering/"
z = np.load(SCR + "t3_inputs.npz")
X, lab, sp = z["X"], z["lab"], z["special"]
XS = z["XS"]
def series(X):
    Xs = X / X.mean(axis=1, keepdims=True)
    return {"raw": X, "std": Xs, "special": Xs[:, sp]}

def stats(Y, l, k=4):
    n = len(l); c = Y.mean(0)
    nl = np.array([(l == j).sum() for j in range(k)], float)
    C = np.stack([Y[l == j].mean(0) for j in range(k)])
    dC = C - c                                                         # k x d
    b2 = (dC ** 2).sum(1)                                              # ||c_l - c||^2
    b1 = np.sqrt(b2)
    w2 = np.array([((Y[l == j] - C[j]) ** 2).sum() for j in range(k)])  # sum_i ||x_i - c_l||^2
    w1 = np.array([np.sqrt(((Y[l == j] - C[j]) ** 2).sum(1)).sum() for j in range(k)])  # sum_i ||x_i - c_l||
    bL1 = np.abs(dC).sum(1)
    wL1 = np.array([np.abs(Y[l == j] - C[j]).sum() for j in range(k)])
    f = (n - k) / (k - 1)
    return {
        "D1_sklearn": float(calinski_harabasz_score(Y, l)),
        "D2_eq345": float(f * b2.sum() / (w2 / nl).sum()),
        "V1_norm_wnum_meanden": float(f * (nl * b1).sum() / (w1 / nl).sum()),   # sum n_l||.|| / sum mean||.||
        "V2_norm_eq345": float(f * b1.sum() / (w1 / nl).sum()),
        "V3_norm_sklearn": float(f * (nl * b1).sum() / w1.sum()),
        "V4_L1_wnum_meanden": float(f * (nl * bL1).sum() / (wL1 / nl).sum()),
        "V5_sq_wnum_meanden": float(f * (nl * b2).sum() / (w2 / nl).sum()),
    }

def table(Xg, labs, seeds=range(20)):
    S = series(Xg); out = {}
    for name, l in labs.items():
        out[name] = {r: stats(Y, l) for r, Y in S.items()}
    rnd = {r: [] for r in S}
    for s in seeds:
        l = np.random.default_rng(1000 + s).integers(0, 4, Xg.shape[0])
        for r, Y in S.items():
            rnd[r].append(stats(Y, l))
    out["Rd"] = {r: {m: [float(np.mean([d[m] for d in rnd[r]])), float(np.std([d[m] for d in rnd[r]]))]
                     for m in rnd[r][0]} for r in rnd}
    return out

res = {"ToU": table(X, {"NMF_authors": lab, "Features_runc": z["feat_c"], "NMF_runc": z["nmf_c"]}),
       "Std": table(XS, {"NMF_runc": z["nmfS"], "Features_runc": z["featS"]})}
paper = {"ToU": {"Rd": [4627, 4950, 5151], "Features": [40764, 6870, 7033], "NMF": [13432, 14088, 16070]},
         "Std": {"Rd": [5014, 4834, 4904], "Features": [5088, 4741, 4694], "NMF": [12971, 14934, 15460]}}
res["paper"] = paper
json.dump(res, open(OUT + "t3_core.json", "w"), indent=1)
rows = ["raw", "std", "special"]
for g in ["ToU", "Std"]:
    print("=====", g)
    metrics = list(res[g]["Rd"]["raw"].keys())
    for m in metrics:
        line = f"{m:22s}"
        for name in res[g]:
            if name == "Rd":
                line += " | Rd " + "/".join(f"{res[g]['Rd'][r][m][0]:.4g}±{res[g]['Rd'][r][m][1]:.2g}" for r in rows)
            else:
                line += f" | {name} " + "/".join(f"{res[g][name][r][m]:.4g}" for r in rows)
        print(line)
    print("paper", paper[g])
