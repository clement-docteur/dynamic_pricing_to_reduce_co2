"""Household clustering of Bregere & Bessa (2020), Sec. IV and Appendices B-C, the forensic
recovery of the authors' ToU partition, and the Calinski-Harabasz variants of Table 3.

Pipeline (paper Sec. IV-B, App. C), per household set I (ToU or Std, clustered separately):

1. causality profiles (``lib.causality``, Gaussian location-scale model fitted per household
   and per half-hour on all 365 days; tau = temp_hh; LAML smoothing):
       mu_i^h(p) = (1/T) sum_t muhat_i(x_t^h, p)          (eqs. 1 / 25)
   Std households get the same tariff schedule as the ToU ones (placebo "tariff").
2. M = [mu(Low) | mu(Normal) | mu(High)] / mubar_i,  mubar_i = mean_h mu_i^h(Normal)   (eq. 27);
   negative entries (a household whose fitted mean is < 0 somewhere) are clipped to 0 because
   NMF needs M >= 0 (counts reported).
3. NMF, r = 5 (sklearn, coordinate descent, init 'nndsvd' = the sklearn-0.22 default for
   r < n_features, Frobenius loss, no regularisation): W (|I| x 5) = the household features.
4. k-medoids, k = 4, Euclidean, as sklearn_extra 0.1/0.2 ``KMedoids`` (method 'alternate',
   init 'heuristic' = the k points with the smallest total distance, max_iter 300);
   a PAM (BUILD + SWAP) variant is provided for robustness.

Baselines (Sec. IV-C): random labels (an integer in 0..k-1 per household) and "classical
features" (min / max / mean consumption in winter and in summer, peak and off half-hour =
average over days of the argmax / argmin half-hour; each feature min-max rescaled to [0, 1];
k-medoids k = 4).  ``classical_features`` documents the exact definitions and variants.

Forensics: ``recover_partition`` solves, for each authors' cluster c, the non-negative least
squares problem  min_w || A' w - ybar_c ||  (A = the identified households' 17,520-vectors,
ybar_c = the authors' cluster-mean series); members get w = 1/n_c.

Calinski-Harabasz (Table 3): ``ch_scores`` evaluates several candidate definitions (sklearn on
household vectors, the paper's eqs. 3-5 literally, pooled 1-D records, household-days, ...)
for the three record series of the paper (raw, divided by the household mean, and only the
normalized Low/High records).

Nothing here imports or runs code from the upstream repository; the authors' ``.feather``
files are read as data only (through ``lib.lcl_data``).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import lcl_data as L
from .causality import fit_causality
from .tariffs import cols, select_tariff_version

K = 4
R_NMF = 5
HOT_MONTHS = (4, 5, 6, 7, 8, 9)  # summer = April-September (spec section 3)
DOW_NAMES = ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
CONDITIONS_PARQUET = L.DATA / "conditions_ours.parquet"

__all__ = [
    "K", "R_NMF", "household_sets", "load_records", "authors_cluster_means", "authors_temps",
    "authors_dummies_frame", "era5_conditions", "recover_partition", "recover_partition_gram",
    "causality_profiles", "profile_matrix", "nmf_features", "pairwise_euclidean",
    "kmedoids_alternate", "kmedoids_pam", "kmedoids_cost", "random_labels",
    "classical_features", "rescale_features", "cluster_means", "relabel_hungarian",
    "relabel_by_size", "contingency", "ari", "record_series", "ch_scores", "ch_sklearn",
    "ch_paper", "ch_grouped", "weekly_profile",
]


# ----------------------------------------------------------------------------- data access
def household_sets(sel: pd.DataFrame | None = None) -> dict:
    """Household sets used in the runs (each a DataFrame sorted by ``row_in_filled``):
    ``tou1032`` / ``std1032`` (our full selection), ``tou1007`` (the identified authors' ToU
    set) and ``std1007`` (the first 1,007 Std drawn, i.e. the Std half of ``in_1007_pair``)."""
    if sel is None:
        sel = pd.read_parquet(L.SELECTION_PARQUET)
    sel = sel.sort_values("row_in_filled").reset_index(drop=True)
    out = {
        "tou1032": sel[sel["group"] == "ToU"],
        "std1032": sel[sel["group"] == "Std"],
        "tou1007": sel[sel["in_authors_tou_1007"]],
        "std1007": sel[(sel["group"] == "Std") & sel["in_1007_pair"]],
    }
    for name, df in out.items():
        out[name] = df.reset_index(drop=True)
    assert len(out["tou1007"]) == 1007 and len(out["std1007"]) == 1007
    assert (out["tou1007"]["group"] == "ToU").all()
    return out


def load_records(rows, as_days: bool = True, filled=None) -> np.ndarray:
    """Filled 2013 records (kWh, authors' slot convention) of the given ``row_in_filled`` rows
    as float64, shape (n, 365, 48) (or (n, 17520) if ``as_days`` is False)."""
    F = np.load(L.FILLED_NPY, mmap_mode="r") if filled is None else filled
    X = np.asarray(F[np.asarray(rows, dtype=np.int64)], dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError("non-finite values in the filled records")
    return X.reshape(len(X), L.N_DAYS, L.H) if as_days else X


def authors_cluster_means():
    """Authors' cluster-mean consumption (4, 365, 48) (power_consump_nmf, sorted by date) and
    the cluster sizes (sum_up_nmf)."""
    cons, sizes = L.load_authors_clusters()
    return cons, np.asarray(sizes, dtype=np.int64)


def _authors_power():
    df = pd.read_feather(L.UP_DATA / "power_consump_nmf.feather")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["cluster", "date"]).reset_index(drop=True)


def authors_temps() -> np.ndarray:
    """The authors' half-hourly temperatures temp_00..47 (365, 48), checked identical across
    their 4 clusters."""
    df = _authors_power()
    ref = None
    for _, g in df.groupby("cluster"):
        a = g.sort_values("date")[cols("temp")].to_numpy(np.float64)
        if ref is None:
            ref = a
        elif not np.array_equal(ref, a):
            raise ValueError("authors' temperatures differ between clusters")
    return ref


def authors_dummies_frame() -> pd.DataFrame:
    """One cluster's rows of power_consump_nmf (365 days, sorted by date): the authors'
    conditions and tariff dummies (identical across clusters, checked in lib.tariffs)."""
    df = _authors_power()
    return df[df["cluster"] == 0].sort_values("date").reset_index(drop=True)


def era5_conditions(version: str = "loc") -> pd.DataFrame:
    """conditions_ours (365 rows sorted by date) with unsuffixed low/high/price aliased from
    ``version`` ('loc' = Tariffs.xlsx with the physically right alignment)."""
    c = pd.read_parquet(CONDITIONS_PARQUET).sort_values("date").reset_index(drop=True)
    return select_tariff_version(c, version)


def condition_arrays(df: pd.DataFrame):
    """(temp, low, high) arrays (365, 48) from a frame in the authors' schema."""
    return (df[cols("temp")].to_numpy(np.float64), df[cols("low")].to_numpy(np.float64),
            df[cols("high")].to_numpy(np.float64))


# ------------------------------------------------------------------ forensic recovery (NNLS)
def _partition_diag(A, Ymeans, sizes, W, threshold):
    k = len(sizes)
    nw = W * np.asarray(sizes, dtype=float)[:, None]
    member = nw > threshold
    n_in = member.sum(axis=0)
    labels = np.where(n_in == 1, np.argmax(member, axis=0), -1)
    diag = {
        "threshold_n_times_w": threshold,
        "sizes_recovered": [int((labels == c).sum()) for c in range(k)],
        "sizes_expected": [int(s) for s in sizes],
        "sizes_match": bool(all((labels == c).sum() == sizes[c] for c in range(k))),
        "disjoint": bool((n_in <= 1).all()),
        "union_is_all": bool((n_in == 1).all()),
        "n_households": int(A.shape[0]),
        "per_cluster": [],
    }
    for c in range(k):
        m = A[labels == c].mean(axis=0) if (labels == c).any() else np.full(A.shape[1], np.nan)
        d = np.abs(m - Ymeans[c])
        diag["per_cluster"].append({
            "cluster": c,
            "member_n_times_w_min": float(nw[c][member[c]].min()) if member[c].any() else None,
            "member_n_times_w_max": float(nw[c][member[c]].max()) if member[c].any() else None,
            "nonmember_n_times_w_max": float(nw[c][~member[c]].max()) if (~member[c]).any() else None,
            "n_nonmember_w_gt_1e-9": int((W[c][~member[c]] > 1e-9).sum()),
            "member_mean_vs_authors_max_abs_kwh": float(np.nanmax(d)),
            "member_mean_vs_authors_mean_abs_kwh": float(np.nanmean(d)),
            "member_mean_vs_authors_corr": float(np.corrcoef(m, Ymeans[c])[0, 1]),
        })
    return labels, diag


def recover_partition(A, Ymeans, sizes, *, threshold: float = 0.5, maxiter_factor: int = 50):
    """NNLS recovery of a partition from cluster means.

    A : (n, R) household records; Ymeans : (k, R) cluster-mean series; sizes : (k,).
    For each c, w_c = argmin_{w >= 0} ||A' w - Ymeans[c]||; household i is a member of c when
    n_c * w_c[i] > ``threshold``.  Returns (labels (n,), W (k, n), diagnostics dict)."""
    from scipy.optimize import nnls

    A = np.asarray(A, dtype=np.float64)
    Ymeans = np.asarray(Ymeans, dtype=np.float64)
    W, rnorm = [], []
    for c in range(len(Ymeans)):
        w, rn = nnls(A.T, Ymeans[c], maxiter=maxiter_factor * A.shape[0])
        W.append(w)
        rnorm.append(float(rn))
    W = np.asarray(W)
    labels, diag = _partition_diag(A, Ymeans, sizes, W, threshold)
    diag["solver"] = "scipy.optimize.nnls on the full (17520 x n) system"
    for c, rn in enumerate(rnorm):
        diag["per_cluster"][c]["nnls_residual_norm_kwh"] = rn
    return labels, W, diag


def recover_partition_gram(A, Ymeans, sizes, *, fit_slots=None, threshold: float = 0.5):
    """Same problem solved through the Cholesky factor of the Gram matrix (A_f A_f' = R'R) on
    a subset of slots ``fit_slots`` (bool mask; default all), so that the remaining slots can
    serve as a held-out check.  Returns (labels, W, diagnostics incl. held-out errors)."""
    from scipy.optimize import nnls

    A = np.asarray(A, dtype=np.float64)
    Ymeans = np.asarray(Ymeans, dtype=np.float64)
    fit = np.ones(A.shape[1], bool) if fit_slots is None else np.asarray(fit_slots, bool)
    Af = A[:, fit]
    Rf = np.linalg.cholesky(Af @ Af.T).T
    W = []
    for c in range(len(Ymeans)):
        q = np.linalg.solve(Rf.T, Af @ Ymeans[c, fit])
        W.append(nnls(Rf, q, maxiter=50 * A.shape[0])[0])
    W = np.asarray(W)
    labels, diag = _partition_diag(A[:, fit], Ymeans[:, fit], sizes, W, threshold)
    diag["solver"] = "NNLS on the Cholesky factor of the Gram matrix"
    diag["n_fit_slots"] = int(fit.sum())
    if (~fit).any():
        ho = []
        for c in range(len(Ymeans)):
            m = A[labels == c][:, ~fit].mean(axis=0)
            d = np.abs(m - Ymeans[c, ~fit])
            ho.append({"cluster": c, "heldout_max_abs_kwh": float(d.max()),
                       "heldout_mean_abs_kwh": float(d.mean())})
        diag["heldout"] = ho
    return labels, W, diag


# ------------------------------------------------------------------- causality profiles
def causality_profiles(Y, temp, low, high, *, chunk: int = 256, method: str = "laml",
                       log=None, **kwargs) -> dict:
    """Household-level causality model (lib.causality, one fit per household and half-hour on
    the given days; eq. (1) averages over the same days).  Y: (S, T, 48).

    The default ``lambda_mode='per_series'`` makes each household's fit independent of the
    others, so chunking changes nothing.  Returns a dict of (S, 48, 3) arrays ``mu``,
    ``sigma``, ``xi`` (tariff axis = Low, Normal, High) and (S, 48) diagnostics."""
    Y = np.asarray(Y, dtype=np.float64)
    S = Y.shape[0]
    parts = []
    for s0 in range(0, S, chunk):
        r = fit_causality(Y[s0:s0 + chunk], temp, low, high, method=method, **kwargs)
        parts.append(r)
        if log is not None:
            log(f"  causality fits {min(s0 + chunk, S)}/{S}")
    out = {key: np.concatenate([p[key] for p in parts], axis=0)
           for key in ("mu", "sigma", "xi", "g", "edf", "on_grid_edge", "converged", "n_iter")}
    out["counts"] = parts[0]["counts"]
    out["settings"] = parts[0]["settings"]
    return out


def profile_matrix(mu, clip: bool = True):
    """M (S, 144) = [mu(Low) | mu(Normal) | mu(High)] / mubar_i, mubar_i = mean_h mu_i^h(Normal)
    (eq. 27).  Negative entries are clipped to 0 if ``clip`` (NMF needs M >= 0).
    Returns (M, info dict)."""
    mu = np.asarray(mu, dtype=np.float64)
    if np.isnan(mu).any():
        raise ValueError("NaN in mu: a tariff never occurs at some slot")
    mubar = mu[:, :, 1].mean(axis=1)
    if (mubar <= 0).any():
        raise ValueError(f"{int((mubar <= 0).sum())} households have mubar <= 0")
    M = np.concatenate([mu[:, :, 0], mu[:, :, 1], mu[:, :, 2]], axis=1) / mubar[:, None]
    neg = M < 0
    info = {
        "n_entries": int(M.size),
        "n_negative_entries": int(neg.sum()),
        "n_households_with_negative": int(neg.any(axis=1).sum()),
        "min_entry_before_clip": float(M.min()),
        "negative_entries_by_tariff": {p: int(neg[:, j * 48:(j + 1) * 48].sum())
                                       for j, p in enumerate(("Low", "Normal", "High"))},
        "mubar_min_kwh": float(mubar.min()),
        "mubar_median_kwh": float(np.median(mubar)),
        "mubar_max_kwh": float(mubar.max()),
        "clipped": bool(clip),
    }
    if clip:
        M = np.maximum(M, 0.0)
    return M, info


# ----------------------------------------------------------------------------------- NMF
def nmf_features(M, r: int = R_NMF, *, init: str = "nndsvd", random_state: int = 0,
                 max_iter: int = 2000, tol: float = 1e-4, solver: str = "cd"):
    """sklearn NMF (Frobenius, no regularisation, ``shuffle=False``) -> (W, H, info)."""
    from sklearn.decomposition import NMF
    from sklearn.exceptions import ConvergenceWarning

    model = NMF(n_components=r, init=init, solver=solver, beta_loss="frobenius", tol=tol,
                max_iter=max_iter, random_state=random_state, alpha_W=0.0, alpha_H="same",
                l1_ratio=0.0, shuffle=False)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always", ConvergenceWarning)
        W = model.fit_transform(M)
    Hm = model.components_
    info = {
        "init": init, "random_state": random_state, "solver": solver, "max_iter": max_iter,
        "tol": tol, "n_iter": int(model.n_iter_),
        "reconstruction_err": float(model.reconstruction_err_),
        "relative_err": float(np.linalg.norm(M - W @ Hm) / np.linalg.norm(M)),
        "converged": not any(issubclass(w.category, ConvergenceWarning) for w in wlist),
        "n_zero_rows_W": int((W.sum(axis=1) == 0).sum()),
    }
    return W, Hm, info


# ------------------------------------------------------------------------------ k-medoids
def pairwise_euclidean(X) -> np.ndarray:
    from scipy.spatial.distance import cdist

    X = np.asarray(X, dtype=np.float64)
    return cdist(X, X, metric="euclidean")


def kmedoids_cost(D, medoids) -> float:
    return float(D[np.asarray(medoids)].min(axis=0).sum())


def kmedoids_alternate(X=None, k: int = K, *, D=None, init: str = "heuristic",
                       max_iter: int = 300, random_state=None):
    """Port of sklearn_extra 0.1/0.2 ``KMedoids(method='alternate')`` (Voronoi iteration).

    init: 'heuristic' (the k points with the smallest sum of distances, via np.argpartition
    as upstream), 'random' (k distinct random points), 'k-medoids++', or an index array.
    Labels are re-assigned to the final medoids (sklearn_extra 0.2; identical to 0.1 at
    convergence).  Returns dict(labels, medoids, cost (sum of distances), n_iter, converged)."""
    if D is None:
        D = pairwise_euclidean(X)
    n = D.shape[0]
    rng = np.random.default_rng(random_state) if not isinstance(random_state, np.random.Generator) \
        else random_state
    if isinstance(init, str):
        if init == "heuristic":
            medoids = np.argpartition(D.sum(axis=1), k - 1)[:k].copy()
        elif init == "random":
            medoids = rng.choice(n, size=k, replace=False)
        elif init == "k-medoids++":
            medoids = _kpp_init(D, k, rng)
        else:
            raise ValueError(f"unknown init {init!r}")
    else:
        medoids = np.asarray(init, dtype=np.int64).copy()
    converged = False
    n_iter = 0
    for n_iter in range(max_iter):
        old = medoids.copy()
        labels = np.argmin(D[medoids, :], axis=0)
        for c in range(k):
            idx = np.flatnonzero(labels == c)
            if idx.size == 0:
                warnings.warn(f"k-medoids: cluster {c} is empty")
                continue
            costs = D[np.ix_(idx, idx)].sum(axis=1)
            j = int(np.argmin(costs))
            cur = costs[int(np.argmax(idx == medoids[c]))]
            if costs[j] < cur:
                medoids[c] = idx[j]
        if np.array_equal(old, medoids):
            converged = True
            break
    labels = np.argmin(D[medoids, :], axis=0)
    return {"labels": labels, "medoids": medoids, "cost": kmedoids_cost(D, medoids),
            "n_iter": int(n_iter + 1), "converged": converged}


def _kpp_init(D, k, rng, n_local_trials=None):
    """k-medoids++ (as sklearn_extra: greedy k-means++ on distances)."""
    n = D.shape[0]
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(k))
    centers = [int(rng.integers(n))]
    closest = D[centers[0]] ** 2
    pot = closest.sum()
    for _ in range(1, k):
        rand_vals = rng.random(n_local_trials) * pot
        cand = np.searchsorted(np.cumsum(closest), rand_vals)
        cand = np.minimum(cand, n - 1)
        dist_cand = D[cand] ** 2
        best, best_pot, best_dist = None, None, None
        for t in range(n_local_trials):
            new_dist = np.minimum(closest, dist_cand[t])
            p = new_dist.sum()
            if best is None or p < best_pot:
                best, best_pot, best_dist = int(cand[t]), p, new_dist
        centers.append(best)
        closest, pot = best_dist, best_pot
    return np.asarray(centers, dtype=np.int64)


def kmedoids_pam(X=None, k: int = K, *, D=None, max_iter: int = 300):
    """PAM: greedy BUILD then SWAP (best improving medoid/non-medoid exchange per iteration,
    until no exchange lowers the sum of distances).  Returns the same dict as
    ``kmedoids_alternate``."""
    if D is None:
        D = pairwise_euclidean(X)
    n = D.shape[0]
    medoids = [int(np.argmin(D.sum(axis=1)))]
    dnear = D[medoids[0]].copy()
    for _ in range(1, k):
        gain = np.maximum(dnear[None, :] - D, 0.0).sum(axis=1)
        gain[medoids] = -np.inf
        j = int(np.argmax(gain))
        medoids.append(j)
        dnear = np.minimum(dnear, D[j])
    medoids = np.asarray(medoids, dtype=np.int64)
    converged = False
    it = 0
    for it in range(max_iter):
        Dm = D[medoids]                       # (k, n)
        order = np.argsort(Dm, axis=0)
        nearest = order[0]
        d1 = Dm[order[0], np.arange(n)]
        d2 = Dm[order[1], np.arange(n)] if k > 1 else np.full(n, np.inf)
        cost0 = d1.sum()
        best = (0.0, None, None)
        for mpos in range(k):
            base = np.where(nearest == mpos, d2, d1)          # cost without medoid mpos
            new = np.minimum(base[None, :], D)                # (n_candidates, n) add candidate o
            delta = new.sum(axis=1) - cost0
            delta[medoids] = np.inf
            o = int(np.argmin(delta))
            if delta[o] < best[0] - 1e-12:
                best = (float(delta[o]), mpos, o)
        if best[1] is None:
            converged = True
            break
        medoids[best[1]] = best[2]
    labels = np.argmin(D[medoids, :], axis=0)
    return {"labels": labels, "medoids": medoids, "cost": kmedoids_cost(D, medoids),
            "n_iter": int(it + 1), "converged": converged}


# ------------------------------------------------------------------------------ baselines
def random_labels(n: int, k: int = K, seed: int = 0) -> np.ndarray:
    """Random clustering: an integer in 0..k-1 drawn independently for each household."""
    return np.random.default_rng(seed).integers(0, k, size=n)


def classical_features(Y, dates, variant: str = "records") -> pd.DataFrame:
    """'Classical features' of Sec. IV-C for households Y (S, 365, 48) (kWh).

    Seasons: summer = April-September (HOT_MONTHS), winter = the other months.
    variant 'records' (primary): min / max / mean of the half-hourly records of the season;
    variant 'profile': min / max / mean of the season's average daily profile (48 values);
    variant 'daily':   min / max / mean of the season's daily totals.
    peak_hh / off_hh (all variants) = average over the 365 days of the half-hour index (0..47)
    of the day's maximum / minimum record (first index on ties)."""
    Y = np.asarray(Y, dtype=np.float64)
    months = pd.DatetimeIndex(dates).month.to_numpy()
    hot = np.isin(months, HOT_MONTHS)
    feats = {}
    for season, mask in (("winter", ~hot), ("summer", hot)):
        Z = Y[:, mask, :]
        if variant == "records":
            v = Z.reshape(Z.shape[0], -1)
        elif variant == "profile":
            v = Z.mean(axis=1)
        elif variant == "daily":
            v = Z.sum(axis=2)
        else:
            raise ValueError(f"unknown variant {variant!r}")
        feats[f"min_{season}"] = v.min(axis=1)
        feats[f"max_{season}"] = v.max(axis=1)
        feats[f"mean_{season}"] = v.mean(axis=1)
    feats["peak_hh"] = Y.argmax(axis=2).mean(axis=1)
    feats["off_hh"] = Y.argmin(axis=2).mean(axis=1)
    return pd.DataFrame(feats)


def rescale_features(F, how: str = "minmax") -> np.ndarray:
    """'minmax' (each feature to [0, 1]) or 'standard' (z-scores)."""
    F = np.asarray(F, dtype=np.float64)
    if how == "minmax":
        rng = F.max(axis=0) - F.min(axis=0)
        return (F - F.min(axis=0)) / np.where(rng > 0, rng, 1.0)
    if how == "standard":
        sd = F.std(axis=0)
        return (F - F.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    raise ValueError(how)


# ---------------------------------------------------------------- labels / comparisons
def cluster_means(X, labels, k: int = K) -> np.ndarray:
    """Mean of the rows of X (n, ...) per cluster -> (k, ...)."""
    labels = np.asarray(labels)
    return np.stack([np.asarray(X)[labels == c].mean(axis=0) for c in range(k)])


def relabel_hungarian(our_means, their_means):
    """Match our clusters to theirs by maximising the summed Pearson correlation of the
    flattened cluster-mean series (scipy linear_sum_assignment).
    Returns (mapping dict ours -> theirs, correlation matrix (ours x theirs))."""
    from scipy.optimize import linear_sum_assignment

    a = np.asarray(our_means, dtype=np.float64).reshape(len(our_means), -1)
    b = np.asarray(their_means, dtype=np.float64).reshape(len(their_means), -1)
    C = np.corrcoef(np.vstack([a, b]))[: len(a), len(a):]
    r, c = linear_sum_assignment(-C)
    return {int(i): int(j) for i, j in zip(r, c)}, C


def relabel_by_size(labels, k: int = K) -> tuple[np.ndarray, dict]:
    """Relabel clusters 0..k-1 by size, descending."""
    labels = np.asarray(labels)
    sizes = np.bincount(labels, minlength=k)
    order = np.argsort(-sizes, kind="stable")
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    return np.vectorize(mapping.get)(labels), mapping


def contingency(a, b, k: int = K) -> np.ndarray:
    """k x k contingency table: rows = labels a, columns = labels b."""
    T = np.zeros((k, k), dtype=np.int64)
    np.add.at(T, (np.asarray(a), np.asarray(b)), 1)
    return T


def ari(a, b) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(np.asarray(a), np.asarray(b)))


# ------------------------------------------------------------- Calinski-Harabasz (Table 3)
def record_series(X, low, high) -> dict:
    """The three record series of Sec. IV-C for households X (S, 17520):
    'raw'; 'std' = records / household mean over the year; 'special' = the 'std' records at
    the Low-or-High slots only (special-tariff mask from ``low``/``high`` (365, 48))."""
    X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
    m = X.mean(axis=1, keepdims=True)
    Z = X / m
    mask = ((np.asarray(low) > 0.5) | (np.asarray(high) > 0.5)).ravel()
    return {"raw": X, "std": Z, "special": Z[:, mask], "special_mask": mask}


def ch_sklearn(X, labels) -> float:
    """sklearn.metrics.calinski_harabasz_score (observations = rows of X)."""
    from sklearn.metrics import calinski_harabasz_score

    return float(calinski_harabasz_score(X, labels))


def ch_paper(X, labels, k: int | None = None) -> float:
    """Eqs. (3)-(5) literally: (|I|-k) sum_l ||Y(C_l)-Y(I)||^2 /
    ((k-1) sum_l (1/|C_l|) sum_{i in C_l} ||Y(i)-Y(C_l)||^2), Y(I) = mean over households."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    ks = np.unique(labels)
    k = len(ks) if k is None else k
    YI = X.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in ks:
        Xc = X[labels == c]
        mc = Xc.mean(axis=0)
        between += float(((mc - YI) ** 2).sum())
        within += float(((Xc - mc) ** 2).sum()) / len(Xc)
    return (len(X) - k) * between / ((k - 1) * within)


def ch_grouped(X3, labels) -> float:
    """CH with observations = all (household, j) rows of X3 (S, n_obs, d), each labelled by its
    household's cluster (sklearn's definition, computed from sufficient statistics).
    d = 1 gives the pooled 1-D records (each household x half-hour record one observation)."""
    X3 = np.asarray(X3, dtype=np.float64)
    if X3.ndim == 2:
        X3 = X3[:, :, None]
    labels = np.asarray(labels)
    S, n_obs, d = X3.shape
    hsum = X3.sum(axis=1)                         # (S, d)
    hss = (X3 ** 2).sum(axis=(1, 2))              # (S,)
    N = S * n_obs
    mean = hsum.sum(axis=0) / N
    ks = np.unique(labels)
    B = 0.0
    Wd = 0.0
    for c in ks:
        sel = labels == c
        nc = sel.sum() * n_obs
        mc = hsum[sel].sum(axis=0) / nc
        B += nc * float(((mc - mean) ** 2).sum())
        Wd += float(hss[sel].sum()) - nc * float((mc ** 2).sum())
    kk = len(ks)
    return (B / (kk - 1)) / (Wd / (N - kk))


def ch_scores(series: dict, labels, definitions=("D1", "D2", "D3", "D4")) -> dict:
    """CH of one clustering for the record series of ``record_series`` under several
    definitions:
      D1  sklearn calinski_harabasz_score, observations = households (vectors Y(i));
      D2  the paper's eqs. (3)-(5) literally (unweighted between-cluster sum);
      D3  pooled 1-D records (each household x record one observation);
      D4  observations = household-days, 48-dim daily profiles (raw / std only; for
          'special' the non-special records of each day are set to 0)."""
    out = {}
    for name in ("raw", "std", "special"):
        X = series[name]
        res = {}
        if "D1" in definitions:
            res["D1"] = ch_sklearn(X, labels)
        if "D2" in definitions:
            res["D2"] = ch_paper(X, labels)
        if "D3" in definitions:
            res["D3"] = ch_grouped(X[:, :, None], labels)
        if "D4" in definitions:
            if name == "special":
                Z = np.where(series["special_mask"][None, :], series["std"], 0.0)
                res["D4"] = ch_grouped(Z.reshape(len(Z), L.N_DAYS, L.H), labels)
            else:
                res["D4"] = ch_grouped(X.reshape(len(X), L.N_DAYS, L.H), labels)
        out[name] = res
    return out


# ------------------------------------------------------------------------------- Fig. 2
def weekly_profile(series_days, dates) -> np.ndarray:
    """Average week, Sunday..Saturday: (..., 365, 48) -> (..., 7, 48) (mean over all days of
    each weekday)."""
    a = np.asarray(series_days, dtype=np.float64)
    dow = (pd.DatetimeIndex(dates).dayofweek.to_numpy() + 1) % 7   # Sunday = 0
    return np.stack([a[..., dow == j, :].mean(axis=-2) for j in range(7)], axis=-2)
