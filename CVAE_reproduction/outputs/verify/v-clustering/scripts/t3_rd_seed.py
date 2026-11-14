"""Search numpy seeds whose random 4-cluster labelling of 1,007 households has the paper's Fig. 2 random sizes
248/264/242/253 (ToU), then score V1 for those labellings; also std-normalisation sensitivity of V1."""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np, pandas as pd
SCR = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-clustering/"
REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/"
OUT = REPRO + "outputs/verify/v-clustering/"
z = np.load(SCR + "t3_inputs.npz"); X, lab, sp = z["X"], z["lab"], z["special"]
rec = pd.read_parquet(REPRO + "data/authors_clusters_recovered.parquet")
ids = rec.LCLid.to_numpy(dtype=object)
print("rec sorted by LCLid:", bool((np.array(sorted(ids)) == ids).all()))
target = (248, 264, 242, 253)
def V1(Y, l, k=4):
    n = len(l); c = Y.mean(0); nl = np.bincount(l, minlength=k).astype(float)
    C = np.stack([Y[l == j].mean(0) for j in range(k)])
    b1 = np.sqrt(((C - c) ** 2).sum(1))
    w1 = np.array([np.sqrt(((Y[l == j] - C[j]) ** 2).sum(1)).sum() for j in range(k)])
    return (n - k) / (k - 1) * (nl * b1).sum() / (w1 / nl).sum()
gens = {
    "legacy_randint(0,4,n)": lambda s: np.random.RandomState(s).randint(0, 4, 1007),
    "legacy_choice(4,n)": lambda s: np.random.RandomState(s).choice(4, 1007),
    "default_rng.integers(0,4,n)": lambda s: np.random.default_rng(s).integers(0, 4, 1007),
}
hits = []
for g, fn in gens.items():
    for s in range(200000):
        l = fn(s)
        if tuple(np.bincount(l, minlength=4)) == target:
            hits.append((g, s))
            if len([h for h in hits if h[0] == g]) >= 5:
                break
print("seed hits:", hits)
Xs = X / X.mean(1, keepdims=True)
res = {"hits": []}
for g, s in hits:
    l = gens[g](s)
    v = [V1(X, l), V1(Xs, l), V1(Xs[:, sp], l)]
    res["hits"].append({"gen": g, "seed": s, "V1_raw_std_special": v})
    print(g, s, np.round(v, 1), "paper 4627/4950/5151")
# distribution of V1 for random labellings with exactly the target sizes (random permutations)
rng = np.random.default_rng(7)
base = np.repeat(np.arange(4), target)
vals = []
for _ in range(200):
    l = rng.permutation(base)
    vals.append([V1(X, l), V1(Xs, l), V1(Xs[:, sp], l)])
vals = np.array(vals)
res["perm_target_sizes_mean_sd"] = [vals.mean(0).tolist(), vals.std(0).tolist()]
res["perm_pct_within_paper_band"] = float(((vals[:, 0] > 4400) & (vals[:, 0] < 4850)).mean())
print("random with target sizes: mean", vals.mean(0).round(0), "sd", vals.std(0).round(0))
print("percentile of paper values:", [float((vals[:, j] < p).mean()) for j, p in enumerate([4627, 4950, 5151])])
res["percentile_of_paper_rd"] = [float((vals[:, j] < p).mean()) for j, p in enumerate([4627, 4950, 5151])]
# std normalisation sensitivity: mean over observed (unfilled) records
sel = pd.read_parquet(REPRO + "data/selection.parquet")
rd = dict(zip(sel.LCLid.to_numpy(dtype=object), sel.row_in_dense.to_numpy()))
R = np.load(REPRO + "data/lcl2013_kwh.npy", mmap_mode="r")
Rr = np.asarray(R[[rd[i] for i in ids]], dtype=np.float64)
mobs = np.nanmean(Rr, axis=1)
Xs2 = X / mobs[:, None]
res["std_nanmean_unfilled"] = [V1(Xs2, lab), V1(Xs2[:, sp], lab)]
X32 = X.astype(np.float32); Xs32 = X32 / X32.mean(1, keepdims=True)
res["float32"] = [float(V1(X32.astype(np.float64), lab)), float(V1(Xs32.astype(np.float64), lab)), float(V1(Xs32[:, sp].astype(np.float64), lab))]
print("std with nanmean of unfilled data:", np.round(res["std_nanmean_unfilled"], 1), " float32 path:", np.round(res["float32"], 1))
json.dump(res, open(OUT + "t3_rd_seed.json", "w"), indent=1)
