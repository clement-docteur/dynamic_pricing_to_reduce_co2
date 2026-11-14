"""v-clustering check 1: exact partition recovery (independent re-check).

Reads (read-only): REPRO/data/{authors_clusters_recovered,selection}.parquet,
REPRO/data/lcl2013_kwh_filled.npy, REPRO/data/lcl2013_kwh.npy (unfilled),
UPSTREAM power_consump_nmf.feather / sum_up_nmf.feather (by absolute path, data only).
Does NOT import REPRO/lib.
"""
import os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/"
U = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo/data/"
OUT = REPRO + "outputs/verify/v-clustering/"
SCR = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-clustering/"
res = {}
t0 = time.time()

sel = pd.read_parquet(REPRO + "data/selection.parquet")
rec = pd.read_parquet(REPRO + "data/authors_clusters_recovered.parquet")
sel_id = sel.LCLid.to_numpy(dtype=object); rec_id = rec.LCLid.to_numpy(dtype=object)

# --- consistency of the row mapping and the set -----------------------------------------
m = dict(zip(sel_id, sel.row_in_filled.to_numpy()))
grp = dict(zip(sel_id, sel.group.to_numpy(dtype=object)))
flag = dict(zip(sel_id, sel.in_authors_tou_1007.to_numpy()))
res["n_rows_recovered"] = int(len(rec))
res["unique_LCLid"] = int(len(set(rec_id)))
res["row_mapping_consistent"] = bool(all(m[i] == r for i, r in zip(rec_id, rec.row_in_filled.to_numpy())))
res["all_ToU"] = bool(all(grp[i] == "ToU" for i in rec_id))
res["set_equals_in_authors_tou_1007"] = bool(set(rec_id) == set(sel_id[sel.in_authors_tou_1007.to_numpy()]))
res["sizes"] = {int(c): int(n) for c, n in rec.cluster.value_counts().sort_index().items()}
su = pd.read_feather(U + "sum_up_nmf.feather")
res["sum_up_sizes"] = {int(c): int(n) for c, n in zip(su.cluster, su.number_clients)}

# --- authors' cluster-mean series -------------------------------------------------------
pc = pd.read_feather(U + "power_consump_nmf.feather")
pc["date_s"] = pd.to_datetime(pc["date"].astype(str))
ccols = [f"consumption_{h:02d}" for h in range(48)]
A = {}
for c in range(4):
    d = pc[pc.cluster == c].sort_values("date_s")
    assert len(d) == 365 and d.date_s.is_unique and d.date_s.iloc[0] == pd.Timestamp("2013-01-01")
    A[c] = d[ccols].to_numpy(dtype=np.float64).reshape(-1)          # 17,520
Y = np.stack([A[c] for c in range(4)], axis=1)                       # 17520 x 4

# --- household matrix (1,007 x 17,520), float64 --------------------------------------
F = np.load(REPRO + "data/lcl2013_kwh_filled.npy", mmap_mode="r")
rows = rec.row_in_filled.to_numpy()
X = np.asarray(F[rows], dtype=np.float64)                            # 1007 x 17520
lab = rec.cluster.to_numpy()
assert np.isfinite(X).all()

# --- member means vs authors ---------------------------------------------------------
err = {}
for c in range(4):
    mu = X[lab == c].mean(axis=0)
    e = mu - A[c]
    err[c] = dict(n=int((lab == c).sum()), max_abs=float(np.abs(e).max()), mean_abs=float(np.abs(e).mean()),
                  rms=float(np.sqrt((e**2).mean())), l2=float(np.linalg.norm(e)),
                  n_slots_abs_gt_1e6=int((np.abs(e) > 1e-6).sum()))
res["member_mean_vs_authors"] = err

# where do the residuals sit? unfilled data (NaN at gaps) via row_in_dense
raw = np.load(REPRO + "data/lcl2013_kwh.npy", mmap_mode="r")
rd = dict(zip(sel_id, sel.row_in_dense.to_numpy()))
R = np.asarray(raw[[rd[i] for i in rec_id]], dtype=np.float64)
gap = ~np.isfinite(R)                                                 # True where originally missing
res["n_missing_cells_1007"] = int(gap.sum())
res["n_slots_with_any_gap_1007"] = int(gap.any(axis=0).sum())
# check filled equals raw where observed
obs = ~gap
res["filled_equals_raw_where_observed_maxabs"] = float(np.abs(X[obs] - R[obs]).max())
loc = {}
for c in range(4):
    e = X[lab == c].mean(axis=0) - A[c]
    g_c = gap[lab == c].any(axis=0)                                   # slots where a member had a gap
    big = np.abs(e) > 1e-6
    loc[c] = dict(n_slots_err_gt_1e6=int(big.sum()), of_which_at_member_gap_slots=int((big & g_c).sum()),
                  max_abs_at_fully_observed_slots=float(np.abs(e[~g_c]).max()),
                  mean_abs_at_fully_observed_slots=float(np.abs(e[~g_c]).mean()),
                  n_fully_observed_slots=int((~g_c).sum()))
res["residual_location"] = loc

# --- independent recovery: unconstrained least squares via SVD -----------------------
XT = X.T                                                              # 17520 x 1007
U_, s, Vt = np.linalg.svd(XT, full_matrices=False)
tol = s.max() * max(XT.shape) * np.finfo(float).eps
res["svd"] = dict(rank=int((s > tol).sum()), s_max=float(s[0]), s_min=float(s[-1]), cond=float(s[0] / s[-1]),
                  tol=float(tol), s_smallest5=[float(v) for v in s[-5:]])
Wls = Vt.T @ ((U_.T @ Y) / s[:, None])                                # 1007 x 4
nW = Wls * np.array([res["sum_up_sizes"][c] for c in range(4)])[None, :]
lab_ls = np.argmax(nW, axis=1)
member = nW[np.arange(len(lab_ls)), lab_ls]
nonmem = nW.copy(); nonmem[np.arange(len(lab_ls)), lab_ls] = 0
res["lstsq"] = dict(agrees_with_recovered=bool((lab_ls == lab).all()),
                    member_ncw_min=float(member.min()), member_ncw_max=float(member.max()),
                    nonmember_abs_ncw_max=float(np.abs(nonmem).max()),
                    each_household_exactly_one_weight_gt_half=bool(((nW > 0.5).sum(axis=1) == 1).all()),
                    residual_l2=[float(np.linalg.norm(XT @ Wls[:, c] - Y[:, c])) for c in range(4)])
# hold-out version: fit on a random half of the slots, check the other half
rng = np.random.default_rng(12345)
idx = rng.permutation(XT.shape[0]); tr, te = idx[: XT.shape[0] // 2], idx[XT.shape[0] // 2:]
Wtr, *_ = np.linalg.lstsq(XT[tr], Y[tr], rcond=None)
lab_tr = np.argmax(Wtr * np.array([res["sum_up_sizes"][c] for c in range(4)])[None, :], axis=1)
res["lstsq_halfslots"] = dict(agrees=bool((lab_tr == lab).all()),
                              heldout_max_abs=float(np.abs(XT[te] @ Wtr - Y[te]).max()),
                              heldout_mean_abs=float(np.abs(XT[te] @ Wtr - Y[te]).mean()))
# guaranteed lower bound on residual of ANY other weight vector differing by dw: ||X dw|| >= s_min ||dw||
nsz = np.array([res["sum_up_sizes"][c] for c in range(4)], float)
res["swap_lower_bound_from_smin"] = {int(c): float(s[-1] * np.sqrt(2) / nsz[c]) for c in range(4)}

# --- single-swap detectability: exact error each swap would produce -------------------
# swapping i (in a) with j (in b): mean_a changes by (x_j - x_i)/n_a, mean_b by (x_i - x_j)/n_b
t1 = time.time()
L1 = cdist(X, X, metric="cityblock")          # sum_t |x_i - x_j|
LI = cdist(X, X, metric="chebyshev")          # max_t |x_i - x_j|
L2 = cdist(X, X, metric="euclidean")
T = X.shape[1]
cross = lab[:, None] != lab[None, :]
na = nsz[lab][:, None]                          # size of i's cluster (a)
nb = nsz[lab][None, :]
# error on cluster a (the larger effect is on the smaller cluster; report the max over the two clusters)
mean_abs_swap = (L1 / T) / np.minimum(na, nb)
max_abs_swap = LI / np.minimum(na, nb)
rms_swap = (L2 / np.sqrt(T)) / np.minimum(na, nb)
for M_ in (mean_abs_swap, max_abs_swap, rms_swap):
    M_[~cross] = np.inf
k_mean = np.unravel_index(np.argmin(mean_abs_swap), mean_abs_swap.shape)
k_max = np.unravel_index(np.argmin(max_abs_swap), max_abs_swap.shape)
k_rms = np.unravel_index(np.argmin(rms_swap), rms_swap.shape)
res["swap"] = dict(
    n_cross_pairs=int(cross.sum() // 2),
    min_mean_abs_err_kwh=float(mean_abs_swap[k_mean]), pair_min_mean=[str(rec_id[k_mean[0]]), str(rec_id[k_mean[1]])],
    min_max_abs_err_kwh=float(max_abs_swap[k_max]), pair_min_max=[str(rec_id[k_max[0]]), str(rec_id[k_max[1]])],
    min_rms_err_kwh=float(rms_swap[k_rms]),
    median_mean_abs_err_kwh=float(np.median(mean_abs_swap[cross])),
    median_max_abs_err_kwh=float(np.median(max_abs_swap[cross])),
    observed_mean_abs_err_max_over_clusters=float(max(err[c]["mean_abs"] for c in range(4))),
    observed_rms_err_max_over_clusters=float(max(err[c]["rms"] for c in range(4))),
    observed_max_abs_err_max_over_clusters=float(max(err[c]["max_abs"] for c in range(4))),
    seconds=time.time() - t1)
res["swap"]["ratio_min_swap_mean_abs_to_observed"] = res["swap"]["min_mean_abs_err_kwh"] / res["swap"]["observed_mean_abs_err_max_over_clusters"]
res["swap"]["ratio_min_swap_rms_to_observed"] = res["swap"]["min_rms_err_kwh"] / res["swap"]["observed_rms_err_max_over_clusters"]
# single move (i from a to b; sizes change) -- error on cluster a = (mean_a - x_i)/(n_a-1)
mv = []
for c in range(4):
    mu = X[lab == c].mean(axis=0)
    d = X[lab == c] - mu[None, :]
    mv.append(float((np.abs(d).mean(axis=1) / (nsz[c] - 1)).min()))
res["move_min_mean_abs_err_kwh_by_cluster"] = mv
res["seconds_total"] = time.time() - t0
json.dump(res, open(OUT + "c1_partition.json", "w"), indent=1)
print(json.dumps(res, indent=1))
