"""Table 3 detail for the matching variant V1 (non-squared norms, size-weighted between, mean within):
exact values, close algebraic variants, special-mask sensitivity."""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np, pandas as pd
SCR = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-clustering/"
REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/"
OUT = REPRO + "outputs/verify/v-clustering/"
z = np.load(SCR + "t3_inputs.npz")
X, lab, sp = z["X"], z["lab"], z["special"]
Xs = X / X.mean(1, keepdims=True)
paper = np.array([13432, 14088, 16070])

def parts(Y, l, k=4):
    n = len(l); c = Y.mean(0)
    nl = np.bincount(l, minlength=k).astype(float)
    C = np.stack([Y[l == j].mean(0) for j in range(k)])
    b1 = np.sqrt(((C - c) ** 2).sum(1))
    w1 = np.array([np.sqrt(((Y[l == j] - C[j]) ** 2).sum(1)).sum() for j in range(k)])
    return n, nl, b1, w1

def variants(Y, l, k=4):
    n, nl, b1, w1 = parts(Y, l, k)
    f = (n - k) / (k - 1)
    return {
        "V1 f*sum(n_l*b)/sum(w/n_l)": f * (nl * b1).sum() / (w1 / nl).sum(),
        "V1b f*sum(b)/sum(w/n_l^2)": f * b1.sum() / (w1 / nl ** 2).sum(),
        "V1c f*(n/k)*sum(b)/sum(w/n_l)": f * (n / k) * b1.sum() / (w1 / nl).sum(),
        "V1d f*sum(n_l*b)/(k*sum(w)/n)": f * (nl * b1).sum() / (k * w1.sum() / n),
        "V1e f*sum(n_l*b)/sum(mean_l of ||.||) [same as V1]": f * (nl * b1).sum() / sum(w1[j] / nl[j] for j in range(k)),
    }

out = {}
S = {"raw": X, "std": Xs, "special_auth2160": Xs[:, sp]}
# alternative special masks from conditions_ours (xlsx 'loc' and 'gmt'), same date order
co = pd.read_parquet(REPRO + "data/conditions_ours.parquet").sort_values("date")
for v in ["loc", "gmt"]:
    lo = co[[f"low_{v}_{h:02d}" for h in range(48)]].to_numpy().reshape(-1)
    hi = co[[f"high_{v}_{h:02d}" for h in range(48)]].to_numpy().reshape(-1)
    m = (lo + hi) > 0
    S[f"special_{v}{int(m.sum())}"] = Xs[:, m]
# special with the mean computed over special records only (re-normalised)
S["special_auth_renorm"] = X[:, sp] / X[:, sp].mean(1, keepdims=True)
S["special_raw_auth"] = X[:, sp]
for name, Y in S.items():
    out[name] = {kk: float(v) for kk, v in variants(Y, lab).items()}
for name in out:
    print(f"{name:22s}", {kk[:4]: round(v, 1) for kk, v in out[name].items()})
v1 = np.array([out[r]["V1 f*sum(n_l*b)/sum(w/n_l)"] for r in ["raw", "std", "special_auth2160"]])
print("V1 vs paper NMF ToU:", v1.round(1), paper, "rel diff %:", ((v1 / paper - 1) * 100).round(3))
out["V1_vs_paper_rel_diff_pct"] = ((v1 / paper - 1) * 100).tolist()
json.dump(out, open(OUT + "t3_v1_detail.json", "w"), indent=1)
