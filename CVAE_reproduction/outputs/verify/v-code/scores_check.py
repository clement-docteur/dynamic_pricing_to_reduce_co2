"""v-code: scores re-implemented from the paper text (eqs. 12, 14, 16), independent of lib.scores.
Part 1: explicit-loop versions on 10 days of cluster 0 (cvae and gam), compared with lib.scores and scores_long.csv.
Part 2: my own vectorized versions on all 92 days x 4 clusters x 2 models -> medians; compared with scores_long.csv."""
import os, sys, json, math
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
import numpy as np, pandas as pd

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
UP = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo"
OUTD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPRO)
from lib import scores as LS   # only to compare against

H = 48
cd = [f"consumption_{h:02d}" for h in range(H)]
te = pd.read_feather(UP + "/data/test_nmf.feather")
te["date"] = pd.to_datetime(te["date"]).dt.strftime("%Y-%m-%d")
dates = pd.read_csv(REPRO + "/outputs/trackA/test_dates.csv")["date"].tolist()

def obs(c):
    t = te[te["cluster"] == c].set_index("date").loc[dates]
    return t[cd].to_numpy(float)

def norm(v):
    return math.sqrt(sum(x * x for x in v))

# ---------------- Part 1: explicit loops (pure Python on lists) ----------------
def loop_scores(S, y):
    N = S.shape[0]; Sl = S.tolist(); yl = y.tolist()
    m = [sum(Sl[i][h] for i in range(N)) / N for h in range(H)]
    rmse_e = norm([m[h] - yl[h] for h in range(H)])                       # eq. 12 literally
    rmse_rm = math.sqrt(sum((m[h] - yl[h]) ** 2 for h in range(H)) / H)  # root-mean variant
    n2 = N // 2
    t1 = (2.0 / N) * sum(norm([Sl[i][h] - yl[h] for h in range(H)]) for i in range(n2))
    t2 = (1.0 / N) * sum(norm([Sl[i][h] - Sl[n2 + i][h] for h in range(H)]) for i in range(n2))
    es_split = t1 - t2                                                    # eq. 14 as written
    d_obs = sum(norm([Sl[i][h] - yl[h] for h in range(H)]) for i in range(N)) / N
    pair = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            pair += norm([Sl[i][h] - Sl[j][h] for h in range(H)])
    es_u = d_obs - pair / (N * (N - 1))                                   # 1/(2N(N-1)) * sum_{i!=j} = sum_{i<j}/(N(N-1))
    vg = 0.0
    for h in range(H):
        for k in range(H):                                                 # ordered pairs (h, h'), p = 0.5
            e = sum(abs(Sl[i][h] - Sl[i][k]) ** 0.5 for i in range(N)) / N
            vg += (abs(yl[h] - yl[k]) ** 0.5 - e) ** 2
    return dict(rmse_euclid=rmse_e, rmse_rootmean=rmse_rm, es_split=es_split, es_u=es_u, vg_ordered=vg)

sl = pd.read_csv(REPRO + "/outputs/trackA/scores_long.csv")
y0 = obs(0)
days = [0, 1, 2, 10, 25, 40, 55, 70, 85, 91]
part1 = {}
for model in ("cvae", "gam"):
    S = np.load(REPRO + f"/outputs/trackA/{model}_samples_c0.npy")
    assert S.shape == (200, 92, 48)
    rows = []
    for d in days:
        mine = loop_scores(S[:, d, :], y0[d])
        lib = dict(rmse_euclid=LS.rmse_euclid(S[:, d, :], y0[d]), rmse_rootmean=LS.rmse_rootmean(S[:, d, :], y0[d]),
                   es_split=LS.energy_score_split(S[:, d, :], y0[d]), es_u=LS.energy_score_u(S[:, d, :], y0[d]),
                   vg_ordered=LS.variogram_score(S[:, d, :], y0[d], p=0.5, pairs="ordered"))
        st = sl[(sl["model"] == model) & (sl["cluster"] == 0) & (sl["date"] == dates[d])].iloc[0]
        stored = dict(rmse_euclid=st["rmse_euclid"], rmse_rootmean=st["rmse_rootmean"], es_split=st["es_split"],
                      es_u=st["es_u"], vg_ordered=st["vg_p05_ordered"])
        rows.append({k: (mine[k], float(lib[k]), float(stored[k])) for k in mine})
    maxdiff_lib = {k: max(abs(r[k][0] - r[k][1]) for r in rows) for k in rows[0]}
    maxdiff_stored = {k: max(abs(r[k][0] - r[k][2]) for r in rows) for k in rows[0]}
    part1[model] = {"max_abs_diff_vs_lib": maxdiff_lib, "max_abs_diff_vs_scores_long": maxdiff_stored,
                    "example_day0_mine": {k: rows[0][k][0] for k in rows[0]}}
print(json.dumps(part1, indent=1))

# ---------------- Part 2: my vectorized versions on all days ----------------
def vec_scores(S, Y):
    N = S.shape[0]; n2 = N // 2
    m = S.mean(0)
    rmse_e = np.sqrt(((m - Y) ** 2).sum(1)); rmse_rm = rmse_e / np.sqrt(H)
    dobs = np.sqrt(((S - Y[None]) ** 2).sum(2))                      # (N, D)
    es_split = (2.0 / N) * dobs[:n2].sum(0) - (1.0 / N) * np.sqrt(((S[:n2] - S[n2:2 * n2]) ** 2).sum(2)).sum(0)
    es_u = np.empty(S.shape[1])
    for d in range(S.shape[1]):
        X = S[:, d, :]
        G = X @ X.T; q = np.diag(G)
        D2 = np.clip(q[:, None] + q[None, :] - 2 * G, 0, None)
        es_u[d] = dobs[:, d].mean() - np.sqrt(D2).sum() / (2 * N * (N - 1))
    vg = np.empty(S.shape[1])
    for d in range(S.shape[1]):
        X = S[:, d, :]
        E = (np.abs(X[:, :, None] - X[:, None, :]) ** 0.5).mean(0)
        O = np.abs(Y[d][:, None] - Y[d][None, :]) ** 0.5
        vg[d] = ((O - E) ** 2).sum()
    return dict(rmse_euclid=rmse_e, rmse_rootmean=rmse_rm, es_split=es_split, es_u=es_u, vg_ordered=vg)

part2 = {}
key_map = dict(rmse_euclid="rmse_euclid", rmse_rootmean="rmse_rootmean", es_split="es_split", es_u="es_u",
               vg_ordered="vg_p05_ordered")
for model in ("cvae", "gam"):
    for c in range(4):
        S = np.load(REPRO + f"/outputs/trackA/{model}_samples_c{c}.npy"); Y = obs(c)
        v = vec_scores(S, Y)
        st = sl[(sl["model"] == model) & (sl["cluster"] == c)].set_index("date").loc[dates]
        part2[f"{model}_c{c}"] = {k: {"median_mine": float(np.median(v[k])),
                                      "max_abs_diff_vs_scores_long": float(np.abs(v[k] - st[key_map[k]].to_numpy()).max())}
                                  for k in v}
        part2[f"{model}_c{c}"]["obs_mean_kwh"] = float(Y.mean())
        part2[f"{model}_c{c}"]["frac_days_gam_better_rmse"] = None
print(json.dumps({k: {kk: (vv["median_mine"] if isinstance(vv, dict) else vv) for kk, vv in v.items()} for k, v in part2.items()}, indent=1))
print("max diffs vs scores_long:", max(vv["max_abs_diff_vs_scores_long"] for v in part2.values() for vv in v.values() if isinstance(vv, dict)))
json.dump({"part1": part1, "part2": part2}, open(OUTD + "/scores_check.json", "w"), indent=1)
