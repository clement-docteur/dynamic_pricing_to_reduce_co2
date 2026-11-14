"""Scores and distribution diagnostics for generated daily profiles.

Bregere & Bessa (2020), IEEE Access 8:131949, Sec. VII-A (eqs. 12-16).  Track-agnostic and vectorized.

Array conventions
  S : samples, shape (N, D, H) = (generated samples, days, half-hours)  -- or (N, H) for a single day
  y : observations, shape (D, H)                                        -- or (H,) for a single day
Every score returns one value per day: shape (D,) (a float for single-day input).

Scores
  rmse_rootmean     sqrt(mean_h (mean_i S_ih - y_h)^2)            (what the paper's Fig. 4 values match, ~0.02 kWh)
  rmse_euclid       || mean_i S_i - y ||_2                        (eq. 12 as printed; = sqrt(H) * rmse_rootmean)
  energy_score_split  eq. 14 as written (N even):
                      (2/N) sum_{i<=N/2} ||S_i - y|| - (1/N) sum_{i<=N/2} ||S_i - S_{N/2+i}||
                    i.e. mean distance of the first half to y minus 0.5 x the mean distance of the N/2 disjoint
                    pairs (i, N/2+i).  For odd N the last sample is ignored (n2 = N // 2).
  energy_score_u    unbiased U-statistic: mean_i ||S_i - y|| - 1/(2 N (N-1)) sum_{i != j} ||S_i - S_j||
  energy_score_v    V-statistic (scoringRules::es_sample): mean_i ||S_i - y|| - 1/(2 N^2) sum_{i,j} ||S_i - S_j||
  variogram_score   eq. 16, p = 0.5, sum over ALL ORDERED pairs (h, h') (h = h' terms are 0):
                      sum_{h,h'} (|y_h - y_h'|^p - (1/N) sum_i |S_ih - S_ih'|^p)^2
                    pairs='unordered' sums over h < h' only and is exactly half of the ordered value.

Energy-score variants (`energy_score_variants`) used to probe the paper's Fig. 4 ES values:
  v1_correct_split, v1_correct_u   correct estimators (eq. 14 split, U-statistic)
  v2_sq_spread                     spread term with the squared norm: (2/N) sum||S_i-y|| - (1/N) sum||S_i-S_{N/2+i}||^2
  v3_quarter_spread                spread term weighted 1/4 instead of 1/2
  v4_first_term                    first term only, (2/N) sum_{i<=N/2} ||S_i - y||
  v5_minmax_scaled                 correct split ES on min-max scaled data ((x - lo) / (hi - lo); needs `scale`)
  v6_eq14_typo_literal             eq. 14 with its index typo implemented literally: outer sum over n = 1..N/2,
                                   summand indexed by the leftover loop variable i = N/2 (the last value of the
                                   first loop in R / Python), i.e. (2/N) sum||S_i - y|| - 0.5 ||S_{N/2} - S_N||
  v7_first_term_all_2overN         first sum taken over all N samples with the 2/N factor (2 x E||S-y||) - correct spread
  v8_first_term_over_N             first sum over N/2 samples but divided by N (0.5 x E||S-y||) - correct spread
  v9_euclid_rmse                   ||mean_i S_i - y|| (what any ES collapses to for a deterministic generator)
  v10_sorted_pairs                 split estimator after sorting the samples independently per half-hour
  v11_l1_norm, v12_rootmean_norm   correct split ES with the L1 norm / the root-mean (norm / sqrt(H))
  v13_obs_next_day, v14_obs_prev_day  correct split ES against the observation of the next / previous day in the
                                   given day order (misaligned observations; only meaningful for the day axis)

Diagnostics: PIT / rank histogram, central-interval coverage, spread-skill, roughness, lag-1 correlations.

Run `python -P lib/scores.py --selftest` for the brute-force unit tests.
"""
from __future__ import annotations

import math
import sys

import numpy as np

__all__ = [
    "rmse_rootmean", "rmse_euclid", "energy_score_split", "energy_score_u", "energy_score_v",
    "energy_score_terms", "pairwise_distance_stats", "variogram_score", "energy_score_variants",
    "ES_VARIANTS", "pit_values", "rank_histogram", "interval_coverage", "spread_skill", "roughness",
    "lag1_corr_pooled", "lag1_corr_across_samples", "distribution_diagnostics", "selftest",
]


# ----------------------------------------------------------------------------------------------------------------
# input handling
# ----------------------------------------------------------------------------------------------------------------
def _prep(S, y):
    S = np.asarray(S, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    single = S.ndim == 2
    if single:
        if y.ndim != 1:
            raise ValueError(f"single-day samples (N, H) need y of shape (H,), got {y.shape}")
        S = S[:, None, :]
        y = y[None, :]
    if S.ndim != 3 or y.ndim != 2 or S.shape[1:] != y.shape:
        raise ValueError(f"samples must be (N, D, H) and y (D, H); got {S.shape} and {y.shape}")
    if S.shape[0] < 2:
        raise ValueError("need at least 2 samples")
    return S, y, single


def _out(x, single):
    x = np.asarray(x)
    return float(x[0]) if single else x


# ----------------------------------------------------------------------------------------------------------------
# RMSE (eq. 12)
# ----------------------------------------------------------------------------------------------------------------
def rmse_rootmean(S, y):
    S, y, single = _prep(S, y)
    return _out(np.sqrt(((S.mean(0) - y) ** 2).mean(-1)), single)


def rmse_euclid(S, y):
    S, y, single = _prep(S, y)
    return _out(np.sqrt(((S.mean(0) - y) ** 2).sum(-1)), single)


# ----------------------------------------------------------------------------------------------------------------
# energy score (eqs. 13-14)
# ----------------------------------------------------------------------------------------------------------------
def _dist_to_obs(S, y, norm: str = "l2"):
    """(N, D) distances ||S_i - y|| per sample and day."""
    d = S - y[None]
    if norm == "l2":
        return np.sqrt((d * d).sum(-1))
    if norm == "l1":
        return np.abs(d).sum(-1)
    if norm == "rootmean":
        return np.sqrt((d * d).mean(-1))
    raise ValueError(norm)


def _split_pair_dist(S, norm: str = "l2", squared: bool = False):
    """(n2, D) distances between the disjoint pairs (i, n2 + i), n2 = N // 2."""
    n2 = S.shape[0] // 2
    d = S[:n2] - S[n2: 2 * n2]
    if norm == "l2":
        q = (d * d).sum(-1)
        return q if squared else np.sqrt(q)
    if norm == "l1":
        return np.abs(d).sum(-1)
    if norm == "rootmean":
        return np.sqrt((d * d).mean(-1))
    raise ValueError(norm)


def energy_score_split(S, y, norm: str = "l2"):
    """Eq. (14) as written: (2/N) sum_{i<=N/2} ||S_i - y|| - (1/N) sum_{i<=N/2} ||S_i - S_{N/2+i}||."""
    S, y, single = _prep(S, y)
    n2 = S.shape[0] // 2
    first = _dist_to_obs(S[:n2], y, norm).mean(0)
    spread = _split_pair_dist(S, norm).mean(0)
    return _out(first - 0.5 * spread, single)


def pairwise_distance_stats(S, chunk_days: int = 16):
    """Per day: mean over i != j of ||S_i - S_j|| (U), and mean over all (i, j) incl. i = j (V).
    Exact pairwise Euclidean distances via the Gram matrix of the per-day CENTRED samples (no cancellation issue:
    distances are computed from deviations around the day's sample mean), chunked over days."""
    S = np.asarray(S, dtype=np.float64)
    N, D, _ = S.shape
    u = np.empty(D)
    for d0 in range(0, D, chunk_days):
        X = S[:, d0: d0 + chunk_days]
        X = X - X.mean(0, keepdims=True)
        G = np.einsum("idh,jdh->dij", X, X, optimize=True)            # (dc, N, N)
        sq = np.einsum("dii->di", G)                                     # (dc, N)
        D2 = sq[:, :, None] + sq[:, None, :] - 2.0 * G
        np.maximum(D2, 0.0, out=D2)
        idx = np.arange(N)
        D2[:, idx, idx] = 0.0
        u[d0: d0 + chunk_days] = np.sqrt(D2).sum((1, 2)) / (N * (N - 1))
    v = u * (N - 1) / N
    return u, v


def energy_score_u(S, y):
    S, y, single = _prep(S, y)
    u, _ = pairwise_distance_stats(S)
    return _out(_dist_to_obs(S, y).mean(0) - 0.5 * u, single)


def energy_score_v(S, y):
    S, y, single = _prep(S, y)
    _, v = pairwise_distance_stats(S)
    return _out(_dist_to_obs(S, y).mean(0) - 0.5 * v, single)


def energy_score_terms(S, y):
    """Per day: E||S - y|| (all N), its first-half version, U mean pair distance, split mean pair distance."""
    S, y, single = _prep(S, y)
    n2 = S.shape[0] // 2
    d_obs = _dist_to_obs(S, y)
    u, _ = pairwise_distance_stats(S)
    out = {"mean_dist_to_obs_all": d_obs.mean(0), "mean_dist_to_obs_first_half": d_obs[:n2].mean(0),
           "mean_pair_dist_u": u, "mean_pair_dist_split": _split_pair_dist(S).mean(0)}
    return {k: _out(v, single) for k, v in out.items()}


ES_VARIANTS = {
    "v1_correct_split": "eq. 14 as written (correct split estimator)",
    "v1_correct_u": "correct ES, full U-statistic over all pairs",
    "v2_sq_spread": "spread term with the squared Euclidean norm",
    "v3_quarter_spread": "spread term weighted 1/4 instead of 1/2",
    "v4_first_term": "first term only, (2/N) sum_{i<=N/2} ||S_i - y||",
    "v5_minmax_scaled": "correct split ES on min-max scaled data (cluster train min/max)",
    "v6_eq14_typo_literal": "eq. 14 typo literally: second sum over n with summand indexed by the leftover i = N/2",
    "v7_first_term_all_2overN": "first sum over all N samples with factor 2/N (= 2 E||S-y||), correct spread",
    "v8_first_term_over_N": "first sum over N/2 samples divided by N (= 0.5 E||S-y||), correct spread",
    "v9_euclid_rmse": "||mean_i S_i - y|| (limit of any ES for a deterministic generator)",
    "v10_sorted_pairs": "split estimator after sorting the samples independently per half-hour",
    "v11_l1_norm": "correct split ES with the L1 norm",
    "v12_rootmean_norm": "correct split ES with the root-mean norm (Euclidean / sqrt(H))",
    "v13_obs_next_day": "correct split ES against the NEXT day's observation (misaligned by +1 day)",
    "v14_obs_prev_day": "correct split ES against the PREVIOUS day's observation (misaligned by -1 day)",
}


def energy_score_variants(S, y, scale: tuple | None = None) -> dict:
    """All energy-score variants per day (dict name -> (D,) array; NaN where undefined).
    `scale` = (lo, hi) for v5 (min-max scaling of samples and observations)."""
    S, y, single = _prep(S, y)
    N, D, H = S.shape
    n2 = N // 2
    first = _dist_to_obs(S[:n2], y).mean(0)
    pair = _split_pair_dist(S)
    spread = pair.mean(0)
    u, _ = pairwise_distance_stats(S)
    out = {
        "v1_correct_split": first - 0.5 * spread,
        "v1_correct_u": _dist_to_obs(S, y).mean(0) - 0.5 * u,
        "v2_sq_spread": first - 0.5 * _split_pair_dist(S, squared=True).mean(0),
        "v3_quarter_spread": first - 0.25 * spread,
        "v4_first_term": first,
        "v6_eq14_typo_literal": first - 0.5 * pair[n2 - 1],
        "v7_first_term_all_2overN": 2.0 * _dist_to_obs(S, y).mean(0) - 0.5 * spread,
        "v8_first_term_over_N": 0.5 * first - 0.5 * spread,
        "v9_euclid_rmse": np.sqrt(((S.mean(0) - y) ** 2).sum(-1)),
    }
    if scale is not None:
        lo, hi = float(scale[0]), float(scale[1])
        Ss, ys = (S - lo) / (hi - lo), (y - lo) / (hi - lo)
        out["v5_minmax_scaled"] = _dist_to_obs(Ss[:n2], ys).mean(0) - 0.5 * _split_pair_dist(Ss).mean(0)
    else:
        out["v5_minmax_scaled"] = np.full(D, np.nan)
    So = np.sort(S, axis=0)
    out["v10_sorted_pairs"] = _dist_to_obs(So[:n2], y).mean(0) - 0.5 * _split_pair_dist(So).mean(0)
    out["v11_l1_norm"] = _dist_to_obs(S[:n2], y, "l1").mean(0) - 0.5 * _split_pair_dist(S, "l1").mean(0)
    out["v12_rootmean_norm"] = (_dist_to_obs(S[:n2], y, "rootmean").mean(0)
                                - 0.5 * _split_pair_dist(S, "rootmean").mean(0))
    nxt = np.full(D, np.nan)
    prv = np.full(D, np.nan)
    if D > 1:
        nxt[:-1] = _dist_to_obs(S[:n2, :-1], y[1:]).mean(0) - 0.5 * spread[:-1]
        prv[1:] = _dist_to_obs(S[:n2, 1:], y[:-1]).mean(0) - 0.5 * spread[1:]
    out["v13_obs_next_day"] = nxt
    out["v14_obs_prev_day"] = prv
    assert set(out) == set(ES_VARIANTS)
    return {k: _out(v, single) for k, v in out.items()}


# ----------------------------------------------------------------------------------------------------------------
# variogram score (eqs. 15-16)
# ----------------------------------------------------------------------------------------------------------------
def variogram_score(S, y, p: float = 0.5, pairs: str = "ordered", chunk_days: int = 16):
    """Eq. (16). pairs='ordered': all (h, h') incl. both orders (the h = h' terms are 0); 'unordered': h < h'
    only (exactly half of 'ordered')."""
    S, y, single = _prep(S, y)
    N, D, H = S.shape
    iu, ju = np.triu_indices(H, 1)
    out = np.empty(D)
    for d0 in range(0, D, chunk_days):
        Sd = S[:, d0: d0 + chunk_days]
        Em = (np.abs(Sd[..., iu] - Sd[..., ju]) ** p).mean(0)          # (dc, H(H-1)/2)
        yd = y[d0: d0 + chunk_days]
        Ay = np.abs(yd[:, iu] - yd[:, ju]) ** p
        out[d0: d0 + chunk_days] = ((Ay - Em) ** 2).sum(-1)
    if pairs == "ordered":
        out = 2.0 * out
    elif pairs != "unordered":
        raise ValueError(pairs)
    return _out(out, single)


# ----------------------------------------------------------------------------------------------------------------
# distribution diagnostics
# ----------------------------------------------------------------------------------------------------------------
def pit_values(S, y):
    """Rank-based PIT per (day, half-hour): (#{S_i < y} + 0.5 #{S_i == y} + 0.5) / (N + 1), in (0, 1).
    Under calibration (y exchangeable with the samples) the rank #{S_i < y} is uniform on {0..N}."""
    S, y, single = _prep(S, y)
    N = S.shape[0]
    below = (S < y[None]).sum(0)
    ties = (S == y[None]).sum(0)
    out = (below + 0.5 * ties + 0.5) / (N + 1)
    return out[0] if single else out


def rank_histogram(pit, bins: int = 10) -> list:
    h, _ = np.histogram(np.asarray(pit).ravel(), bins=bins, range=(0.0, 1.0))
    return (h / h.sum()).tolist()


def interval_coverage(S, y, level: float):
    """Fraction of (day, half-hour) whose observation lies in the central `level` interval of the samples
    (empirical quantiles, linear interpolation)."""
    S, y, _ = _prep(S, y)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(S, [a, 1.0 - a], axis=0)
    return float(((y >= lo) & (y <= hi)).mean())


def spread_skill(S, y) -> dict:
    """Mean per-slot sample SD (ddof=1) vs RMS error of the sample mean, pooled over days and half-hours.
    For a calibrated N-member ensemble E[(mean - y)^2] = (N + 1)/N E[var], so
    ratio_calibrated = sqrt(mean var) / RMSE * sqrt((N + 1)/N) is ~1 (<1 under-, >1 over-dispersed)."""
    S, y, _ = _prep(S, y)
    N = S.shape[0]
    var = S.var(0, ddof=1)
    err2 = (S.mean(0) - y) ** 2
    spread_rms = float(np.sqrt(var.mean()))
    rmse = float(np.sqrt(err2.mean()))
    return {"mean_sample_sd": float(np.sqrt(var).mean()), "rms_sample_sd": spread_rms,
            "rmse_of_sample_mean": rmse, "ratio_mean_sd_to_rmse": float(np.sqrt(var).mean() / rmse),
            "ratio_calibrated": spread_rms / rmse * math.sqrt((N + 1) / N)}


def roughness(X) -> float:
    """Mean |X_{h+1} - X_h| over all leading axes and the 47 within-day differences (last axis = half-hour)."""
    X = np.asarray(X, dtype=np.float64)
    return float(np.abs(np.diff(X, axis=-1)).mean())


def lag1_corr_pooled(E) -> float:
    """Pooled lag-1 correlation along the last axis: corr(E[..., h], E[..., h+1]) over all other indices and h
    (no re-centring per slot: E is a deviation series with mean ~0)."""
    E = np.asarray(E, dtype=np.float64)
    a = E[..., :-1].ravel()
    b = E[..., 1:].ravel()
    return float((a * b).mean() / math.sqrt((a * a).mean() * (b * b).mean()))


def lag1_corr_across_samples(S) -> float:
    """Mean over (day, h) of the correlation ACROSS samples between half-hours h and h+1 (how strongly the
    generated variation at consecutive half-hours moves together)."""
    S = np.asarray(S, dtype=np.float64)
    X = S - S.mean(0, keepdims=True)
    a, b = X[..., :-1], X[..., 1:]
    num = (a * b).sum(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return float(np.nanmean(r))


def distribution_diagnostics(S, y, bins: int = 10) -> dict:
    """Spread-skill, coverage, PIT summary, roughness and lag-1 structure for one model x cluster.
    S (N, D, H), y (D, H)."""
    S, y, _ = _prep(S, y)
    m = S.mean(0)
    pit = pit_values(S, y)
    noise = S - m[None]                      # generated deviations around the generator's mean
    resid = y - m                            # actual deviations of the observation from the generator's mean
    out = {"spread_skill": spread_skill(S, y),
           "coverage_central50": interval_coverage(S, y, 0.50),
           "coverage_central90": interval_coverage(S, y, 0.90),
           "pit": {"mean": float(pit.mean()), "var": float(pit.var()), "var_uniform": 1.0 / 12.0,
                   "frac_below_0.05": float((pit < 0.05).mean()), "frac_above_0.95": float((pit > 0.95).mean()),
                   "frac_outside_sample_range": float(((y < S.min(0)) | (y > S.max(0))).mean()),
                   "hist10": rank_histogram(pit, bins)},
           "roughness": {"samples": roughness(S), "sample_mean_profile": roughness(m), "observed": roughness(y),
                         "ratio_samples_to_observed": roughness(S) / roughness(y),
                         "noise_samples_minus_mean": roughness(noise), "residual_obs_minus_mean": roughness(resid)},
           "lag1": {"noise_pooled": lag1_corr_pooled(noise), "residual_pooled": lag1_corr_pooled(resid),
                    "across_samples_mean_over_day_slot": lag1_corr_across_samples(S),
                    "profile_demeaned_samples": lag1_corr_pooled(S - S.mean(-1, keepdims=True)),
                    "profile_demeaned_observed": lag1_corr_pooled(y - y.mean(-1, keepdims=True))}}
    return out


# ----------------------------------------------------------------------------------------------------------------
# self test: brute-force loops on random data
# ----------------------------------------------------------------------------------------------------------------
def _bf_norm(v):
    return math.sqrt(sum(x * x for x in v))


def selftest(verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    rng = np.random.default_rng(0)
    res = {}
    for (N, D, H) in [(10, 3, 48), (7, 2, 5), (200, 2, 48)]:
        S = rng.gamma(2.0, 0.1, (N, D, H)) + rng.normal(0, 0.01, (N, D, 1))
        y = rng.gamma(2.0, 0.1, (D, H))
        n2 = N // 2
        bf = {k: np.empty(D) for k in ("rm", "eu", "es_split", "es_u", "es_v", "vg_o", "vg_u", "typo", "sq",
                                       "sorted")}
        for d in range(D):
            mean = [sum(S[i, d, h] for i in range(N)) / N for h in range(H)]
            dev = [mean[h] - y[d, h] for h in range(H)]
            bf["rm"][d] = math.sqrt(sum(x * x for x in dev) / H)
            bf["eu"][d] = math.sqrt(sum(x * x for x in dev))
            first = (2.0 / N if N % 2 == 0 else 1.0 / n2) * sum(
                _bf_norm([S[i, d, h] - y[d, h] for h in range(H)]) for i in range(n2))
            pairs = [_bf_norm([S[i, d, h] - S[n2 + i, d, h] for h in range(H)]) for i in range(n2)]
            second = (1.0 / N if N % 2 == 0 else 0.5 / n2) * sum(pairs)
            bf["es_split"][d] = first - second
            bf["typo"][d] = first - 0.5 * pairs[n2 - 1]
            bf["sq"][d] = first - 0.5 * sum(p * p for p in pairs) / n2
            allobs = sum(_bf_norm([S[i, d, h] - y[d, h] for h in range(H)]) for i in range(N)) / N
            tot = 0.0
            for i in range(N):
                for j in range(N):
                    if i != j:
                        tot += _bf_norm([S[i, d, h] - S[j, d, h] for h in range(H)])
            bf["es_u"][d] = allobs - 0.5 * tot / (N * (N - 1))
            bf["es_v"][d] = allobs - 0.5 * tot / (N * N)
            vg = 0.0
            vgu = 0.0
            for h in range(H):
                for k in range(H):
                    e = sum(abs(S[i, d, h] - S[i, d, k]) ** 0.5 for i in range(N)) / N
                    t = (abs(y[d, h] - y[d, k]) ** 0.5 - e) ** 2
                    vg += t
                    if h < k:
                        vgu += t
            bf["vg_o"][d], bf["vg_u"][d] = vg, vgu
            cols = [sorted(S[i, d, h] for i in range(N)) for h in range(H)]
            So = [[cols[h][i] for h in range(H)] for i in range(N)]
            bf["sorted"][d] = (sum(_bf_norm([So[i][h] - y[d, h] for h in range(H)]) for i in range(n2)) / n2
                               - 0.5 * sum(_bf_norm([So[i][h] - So[n2 + i][h] for h in range(H)])
                                           for i in range(n2)) / n2)
        V = energy_score_variants(S, y)
        got = {"rm": rmse_rootmean(S, y), "eu": rmse_euclid(S, y), "es_split": energy_score_split(S, y),
               "es_u": energy_score_u(S, y), "es_v": energy_score_v(S, y), "vg_o": variogram_score(S, y),
               "vg_u": variogram_score(S, y, pairs="unordered"), "typo": V["v6_eq14_typo_literal"],
               "sq": V["v2_sq_spread"], "sorted": V["v10_sorted_pairs"]}
        err = {k: float(np.max(np.abs(got[k] - bf[k]) / np.maximum(1e-12, np.abs(bf[k])))) for k in bf}
        assert max(err.values()) < 1e-10, err
        assert np.allclose(V["v1_correct_split"], bf["es_split"], rtol=1e-12)
        assert np.allclose(V["v1_correct_u"], bf["es_u"], rtol=1e-12)
        assert np.allclose(got["vg_o"], 2 * got["vg_u"], rtol=1e-13)
        # single-day API
        assert abs(energy_score_split(S[:, 0], y[0]) - bf["es_split"][0]) < 1e-12
        assert abs(variogram_score(S[:, 0], y[0]) - bf["vg_o"][0]) < 1e-9
        res[f"N{N}_D{D}_H{H}_max_rel_err"] = max(err.values())
        say(f"[scores selftest] N={N} D={D} H={H}: max rel. error vs brute force {max(err.values()):.2e} "
            f"({', '.join(f'{k} {v:.1e}' for k, v in err.items())})")
    # analytic checks
    # (a) deterministic generator: every ES variant built on the correct formula = Euclidean RMSE
    S1 = np.repeat(rng.normal(0.2, 0.05, (1, 4, 48)), 6, axis=0)
    y1 = rng.normal(0.2, 0.05, (4, 48))
    assert np.allclose(energy_score_split(S1, y1), rmse_euclid(S1, y1), rtol=1e-12)
    assert np.allclose(energy_score_u(S1, y1), rmse_euclid(S1, y1), rtol=1e-12)
    # (b) 1-D: ES = CRPS; for N(0,1) forecast and y = 0, CRPS = (sqrt(2) - 1)/sqrt(pi) = 0.2337
    z = np.random.default_rng(5).standard_normal((2000, 1, 1))
    crps = float(energy_score_u(z, np.zeros((1, 1)))[0])
    assert abs(crps - (math.sqrt(2) - 1) / math.sqrt(math.pi)) < 0.02, crps
    # (c) PIT of a calibrated ensemble is ~uniform; coverage ~ nominal
    Sc = rng.standard_normal((200, 300, 8))
    yc = rng.standard_normal((300, 8))
    pit = pit_values(Sc, yc)
    assert abs(pit.mean() - 0.5) < 0.02 and abs(pit.var() - 1 / 12) < 0.006
    assert abs(interval_coverage(Sc, yc, 0.9) - 0.9) < 0.03 and abs(interval_coverage(Sc, yc, 0.5) - 0.5) < 0.04
    ss = spread_skill(Sc, yc)
    assert abs(ss["ratio_calibrated"] - 1.0) < 0.05, ss
    # (d) lag-1 helpers on an AR(1) with phi = 0.8
    e = np.zeros((500, 48))
    w = rng.standard_normal((500, 48))
    for h in range(48):
        e[:, h] = (0.8 * e[:, h - 1] if h else 0.0) + w[:, h] * (1.0 if h else 1 / math.sqrt(1 - 0.64))
    assert abs(lag1_corr_pooled(e) - 0.8) < 0.03
    assert abs(lag1_corr_across_samples(e[:, None, :]) - 0.8) < 0.03
    res.update(crps_check=crps, pit_mean=float(pit.mean()), pit_var=float(pit.var()),
               ratio_calibrated=ss["ratio_calibrated"])
    say(f"[scores selftest] deterministic ES = Euclidean RMSE ok; 1-D CRPS {crps:.4f} vs exact 0.2337; calibrated "
        f"PIT mean {pit.mean():.3f} var {pit.var():.4f} (1/12 = 0.0833); spread-skill {ss['ratio_calibrated']:.3f}; "
        f"AR(1) lag-1 ok")
    res["status"] = "ok"
    return res


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
