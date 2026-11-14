"""v-clustering check 2b: re-derive run (c) and run (a) ToU clusterings from the cached causality
profiles with my own M / sklearn NMF / own k-medoids (sklearn_extra 0.2 'alternate' semantics),
and test the authors'-medoid forensic numbers.  Does NOT import REPRO/lib."""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np, pandas as pd, warnings
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cdist

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/"
OUT = REPRO + "outputs/verify/v-clustering/"
rec = pd.read_parquet(REPRO + "data/authors_clusters_recovered.parquet")
auth = dict(zip(rec.LCLid.to_numpy(dtype=object), rec.cluster.to_numpy()))

def kmedoids_alt(D, k=4, medoids=None, max_iter=300):
    if medoids is None:
        medoids = np.argpartition(D.sum(axis=1), k - 1)[:k].copy()
    medoids = np.array(medoids)
    for it in range(max_iter):
        old = medoids.copy()
        labels = np.argmin(D[medoids, :], axis=0)
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            costs = D[np.ix_(idx, idx)].sum(axis=1)
            j = np.argmin(costs)
            cur = costs[np.argmax(idx == medoids[c])]
            if costs[j] < cur:
                medoids[c] = idx[j]
        if np.all(old == medoids):
            break
    labels = np.argmin(D[medoids, :], axis=0)
    return labels, medoids, float(D[medoids].min(axis=0).sum()), it + 1

res = {}
for run, npz in [("c", "causality_profiles_auth.npz"), ("a", "causality_profiles_era5_loc.npz")]:
    z = np.load(REPRO + "data/" + npz)
    tou = z["group"] == "ToU"
    ids = z["LCLid"][tou]; mu = z["mu"][tou]
    mubar = mu[:, :, 1].mean(axis=1)
    M = np.concatenate([mu[:, :, 0], mu[:, :, 1], mu[:, :, 2]], axis=1) / mubar[:, None]
    nneg = int((M < 0).sum()); M = np.maximum(M, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nmf = NMF(n_components=5, init="nndsvd", solver="cd", beta_loss="frobenius", tol=1e-4, max_iter=2000,
                  random_state=0, shuffle=False)
        W = nmf.fit_transform(M)
    D = cdist(W, W)
    lab, med, cost, nit = kmedoids_alt(D)
    df = pd.read_parquet(REPRO + f"data/clusters_run_{run}.parquet")
    df = df[df.group.to_numpy(dtype=object) == "ToU"].set_index("LCLid")
    stored = df.loc[list(ids), "nmf"].to_numpy()
    Wst = df.loc[list(ids), ["W0", "W1", "W2", "W3", "W4"]].to_numpy()
    sh = np.array([i in auth for i in ids]); A = np.array([auth[i] for i in ids[sh]])
    out = dict(n=int(len(ids)), n_negative_clipped=nneg, nmf_n_iter=int(nmf.n_iter_),
               W_maxabs_vs_stored=float(np.abs(W - Wst).max()),
               ari_mine_vs_stored=float(adjusted_rand_score(stored, lab)),
               ari_mine_vs_authors=float(adjusted_rand_score(A, lab[sh])),
               my_cost=cost, my_iters=nit, my_sizes=np.bincount(lab).tolist())
    if run == "c":
        # authors' medoids in our W: per authors' cluster, point minimising within-cluster sum of distances
        Ds = D[np.ix_(sh, sh)]
        amed = []
        for c in range(4):
            idx = np.where(A == c)[0]
            amed.append(idx[np.argmin(Ds[np.ix_(idx, idx)].sum(axis=1))])
        amed = np.array(amed)
        out["authors_medoid_cost"] = float(Ds[amed].min(axis=0).sum())
        out["my_cost_on_same_1007"] = float(D[med].min(axis=0).sum())  # run c has exactly the 1,007
        near = np.argmin(Ds[amed, :], axis=0)
        out["pct_nearest_own_authors_medoid"] = float((near == A).mean() * 100)
        l2, m2, c2, it2 = kmedoids_alt(Ds, medoids=amed.copy())
        out["alt_from_authors_medoids"] = dict(ari_vs_authors=float(adjusted_rand_score(A, l2)), cost=c2, iters=it2,
                                               sizes=np.bincount(l2).tolist())
        # within-cluster sum of distances objective of the authors' partition with their own medoids
    res[run] = out
json.dump(res, open(OUT + "c2b_rerun_kmedoids.json", "w"), indent=1)
print(json.dumps(res, indent=1))
