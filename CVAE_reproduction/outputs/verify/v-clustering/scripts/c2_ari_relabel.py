"""v-clustering check 2: ARI of runs a/b/c vs the recovered authors' partition; Track B relabel mapping.
Own implementation; does NOT import REPRO/lib."""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from itertools import permutations

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/"
U = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo/data/"
OUT = REPRO + "outputs/verify/v-clustering/"
res = {}
rec = pd.read_parquet(REPRO + "data/authors_clusters_recovered.parquet")
auth = dict(zip(rec.LCLid.to_numpy(dtype=object), rec.cluster.to_numpy()))

def matched_fraction(a, b):
    ct = pd.crosstab(a, b).to_numpy()
    r, c = linear_sum_assignment(-ct)
    return float(ct[r, c].sum() / ct.sum()), ct

res["ari"] = {}
for run in "abc":
    df = pd.read_parquet(REPRO + f"data/clusters_run_{run}.parquet")
    ids = df.LCLid.to_numpy(dtype=object); grp = df.group.to_numpy(dtype=object)
    tou = grp == "ToU"
    shared = np.array([(g == "ToU") and (i in auth) for i, g in zip(ids, grp)])
    a = np.array([auth[i] for i in ids[shared]])
    out = dict(n_ToU=int(tou.sum()), n_shared=int(shared.sum()))
    # consistency of the 'authors_cluster' column stored in the run file
    out["authors_cluster_col_consistent"] = bool((df.authors_cluster.to_numpy()[shared] == a).all())
    for col in ["nmf", "nmf_pam", "features", "random_00"]:
        b = df[col].to_numpy()[shared]
        mf, ct = matched_fraction(a, b)
        out[col] = dict(ari=float(adjusted_rand_score(a, b)), nmi=float(normalized_mutual_info_score(a, b)),
                        matched_fraction=mf)
        if col == "nmf":
            out[col]["contingency_authors_x_ours"] = ct.tolist()
            out[col]["sizes_ours_ToU_all"] = df.loc[tou, "nmf"].value_counts().sort_index().tolist()
    rnd = [adjusted_rand_score(a, df[f"random_{s:02d}"].to_numpy()[shared]) for s in range(20)]
    out["random_ari_mean_sd"] = [float(np.mean(rnd)), float(np.std(rnd))]
    res["ari"][run] = out

# ---- relabel mapping: run (a) cluster-mean series vs authors' series ---------------------------
sel = pd.read_parquet(REPRO + "data/selection.parquet")
F = np.load(REPRO + "data/lcl2013_kwh_filled.npy", mmap_mode="r")
dfa = pd.read_parquet(REPRO + "data/clusters_run_a.parquet")
dfa = dfa[dfa.group.to_numpy(dtype=object) == "ToU"]
Xa = np.asarray(F[dfa.row_in_filled.to_numpy()], dtype=np.float64)
la = dfa.nmf.to_numpy()
ours = np.stack([Xa[la == c].mean(0) for c in range(4)], 1)            # 17520 x 4 (our run a, 1,032 hh)
pc = pd.read_feather(U + "power_consump_nmf.feather"); pc["d"] = pd.to_datetime(pc.date.astype(str))
cc = [f"consumption_{h:02d}" for h in range(48)]
A = np.stack([pc[pc.cluster == c].sort_values("d")[cc].to_numpy(float).reshape(-1) for c in range(4)], 1)
sizesA = np.array([325, 295, 229, 158.])
def corrmat(P, Q):
    P = (P - P.mean(0)) / P.std(0); Q = (Q - Q.mean(0)) / Q.std(0)
    return P.T @ Q / P.shape[0]
C_raw = corrmat(ours, A)                                                # rows ours, cols authors
# deviation from each side's all-household mean
ours_all = Xa.mean(0); A_all = A @ sizesA / sizesA.sum()
C_dev = corrmat(ours - ours_all[:, None], A - A_all[:, None])
# deviation using a common baseline (the authors' all-ToU mean) for both
C_dev_common = corrmat(ours - A_all[:, None], A - A_all[:, None])
def best_perm(C):
    best = max(permutations(range(4)), key=lambda p: sum(C[i, p[i]] for i in range(4)))
    return {int(i): int(best[i]) for i in range(4)}
r_raw, c_raw = linear_sum_assignment(-C_raw)
map_raw = {int(i): int(j) for i, j in zip(r_raw, c_raw)}
res["relabel"] = dict(
    corr_raw=C_raw.round(4).tolist(), corr_dev=C_dev.round(4).tolist(), corr_dev_commonbaseline=C_dev_common.round(4).tolist(),
    map_hungarian_raw=map_raw, map_bruteforce_raw=best_perm(C_raw),
    map_hungarian_dev=best_perm(C_dev), map_hungarian_dev_common=best_perm(C_dev_common),
    matched_raw_corr=[float(C_raw[i, map_raw[i]]) for i in range(4)],
    matched_dev_corr=[float(C_dev[i, map_raw[i]]) for i in range(4)],
    max_offdiag_dev_corr=float(max(C_dev[i, j] for i in range(4) for j in range(4) if j != map_raw[i])),
    max_offdiag_raw_corr=float(max(C_raw[i, j] for i in range(4) for j in range(4) if j != map_raw[i])),
    mutual_best_raw=bool(all(np.argmax(C_raw[i]) == map_raw[i] and np.argmax(C_raw[:, map_raw[i]]) == i for i in range(4))),
    mutual_best_dev=bool(all(np.argmax(C_dev[i]) == map_raw[i] and np.argmax(C_dev[:, map_raw[i]]) == i for i in range(4))),
)
# household overlap matching (shared 1,007)
ids = dfa.LCLid.to_numpy(dtype=object); sh = np.array([i in auth for i in ids])
ct = pd.crosstab(la[sh], np.array([auth[i] for i in ids[sh]])).to_numpy()
r, c = linear_sum_assignment(-ct)
res["relabel"]["map_overlap"] = {int(i): int(j) for i, j in zip(r, c)}
res["relabel"]["overlap_contingency_ours_x_authors"] = ct.tolist()
res["relabel"]["sizes_ours_raw_labels"] = [int((la == c).sum()) for c in range(4)]
inv = {v: k for k, v in map_raw.items()}
res["relabel"]["sizes_in_authors_order"] = [int((la == inv[c]).sum()) for c in range(4)]
res["relabel"]["level_kwh_ours_in_authors_order"] = [float(ours[:, inv[c]].mean()) for c in range(4)]
res["relabel"]["level_kwh_authors"] = [float(A[:, c].mean()) for c in range(4)]
# does power_consump_ours follow this mapping?
po = pd.read_parquet(REPRO + "data/power_consump_ours.parquet")
po["d"] = pd.to_datetime(po.date)
chk = {}
for c in range(4):
    s = po[po.cluster == c].sort_values("d")
    v = s[cc].to_numpy(float).reshape(-1)
    chk[c] = dict(max_abs_vs_our_member_mean=float(np.abs(v - ours[:, inv[c]]).max()),
                  n_households=(int(s.n_households.iloc[0]) if "n_households" in s else None))
res["relabel"]["power_consump_ours_check"] = chk
json.dump(res, open(OUT + "c2_ari_relabel.json", "w"), indent=1)
print(json.dumps(res, indent=1))
