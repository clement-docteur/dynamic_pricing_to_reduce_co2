"""Causality model of Bregere & Bessa (2020), Sec. IV-A and Appendix B (eqs. 1-2, 19-26).

For one consumption series i (a household or a cluster aggregate) and one half-hour h:

    Y_t^h ~ N( mu(x_t^h, p_t^h), sigma(p_t^h)^2 )
    mu    = s(tau_t^h) + xi_L 1{Low} + xi_N 1{Normal} + xi_H 1{High}     (eq. 21)
    sigma = gamma_L 1{Low} + gamma_N 1{Normal} + gamma_H 1{High}        (eq. 22)

and the reported profiles are (eqs. 1-2 / 25-26)

    mu^h(p)    = (1/T) sum_t muhat(x_t^h, p) = mean_t shat(tau_t^h) + xihat_p
    sigma^h(p) = (1/T) sum_t sigmahat(p, x_t^h) = gammahat_p.

The authors fitted this with R/mgcv (``gaulss`` Gaussian location-scale family).  mgcv is not
available, so this module re-implements the fit with numpy, vectorised over series.

Estimation (documented choices)
-------------------------------
* ``s`` is a centred penalized cubic regression spline (``lib.splines.CRSpline``, k = 10 knots
  at quantiles of the fit-day temperatures at slot h, sum-to-zero over the fit days, integrated
  squared second-derivative penalty).  There is no separate intercept: the three tariff dummies
  sum to one and play that role (eq. 21).
* The sd model only contains the three tariff dummies, i.e. it is *saturated* in the tariff
  classes.  The penalized ML fit therefore does not depend on the sd link function and reduces to
  iteratively re-weighted penalized least squares:  beta = argmin sum_t r_t^2/gamma_{p_t}^2 +
  lambda beta'S beta,  then gamma_p^2 = mean of r_t^2 over the tariff-p observations
  (ML denominator n_p), iterated to convergence (coordinate ascent of the penalized likelihood
  at fixed lambda, hence monotone).
* sd floor: gamma_p >= ``sd_floor`` (default 0.01, mirroring mgcv's ``gaulss(b = 0.01)``,
  whose sd link is sigma = b + exp(eta)); it only binds for (near-)constant series.
* Smoothing parameter: one lambda per (series, half-hour) by default (``lambda_mode =
  "per_series"``), chosen on a log10 grid.  The grid is data-scale free: at grid value g the
  unweighted-scale lambda is  lambda_u = 10**g * tr(B'B)/tr(S)  and the likelihood-scale lambda
  used in the weighted fit is  lambda_u / sigma_pool^2  (sigma_pool^2 = mean squared residual of
  the unweighted fit at that g), kept fixed during the re-weighting iterations.
  Criteria:
    - ``"laml"`` (default; mgcv's gaulss uses REML/LAML): Laplace approximate marginal
      likelihood of the location-scale model, up to constants,
        V = sum_p n_p log gamma_p + 1/2 sum_t r_t^2/gamma_{p_t}^2 + 1/2 lambda beta'S beta
            + 1/2 log|H_bb + lambda S - H_be H_ee^{-1} H_eb| - rank(S)/2 log lambda,
      where H is the Hessian of the negative log-likelihood in (beta, eta_p = log gamma_p)
      (the eta-block is diag(2 n_p) at the optimum, constant in lambda, and is profiled out
      through the Schur complement);
    - ``"gcv"``: GCV of the unweighted penalized least-squares fit,  T*RSS/(T - edf)^2  (mgcv's
      default criterion for Gaussian GAMs); the re-weighted fit then reuses the chosen lambda.
  ``lambda_mode = "shared"`` picks one grid value per half-hour for all series (minimising the
  sum of the series' LAML criteria, or the sum of log GCV) -- cheaper bookkeeping, same cost here.
* Tariffs absent at a slot: if tariff p never occurs at slot h in the fit data, its dummy is
  dropped and mu^h(p) = sigma^h(p) = NaN (no information; callers decide the fallback, e.g.
  ``lib.gam_generator`` uses sigma^h(Normal)).  If p occurs fewer than ``min_obs`` (default 2)
  times, mu^h(p) is still estimated (its dummy absorbs those observations exactly) but
  sigma^h(p) = NaN, because the ML variance of a single observation around its own dummy is 0.
* The average in eq. (1) runs over the fit days by default (then mean_t shat = 0 by centring and
  mu^h(p) = xi_p); pass ``avg_temp`` to average over other days (e.g. all 365 days while fitting
  on the training days) -- the spline is then evaluated there (linear extrapolation outside the
  boundary knots).

Scaling: the design matrix at a given h is shared by all series (same temperatures and the same
tariff schedule), so per-tariff Gram matrices G_p = X_p'X_p are computed once and every series
only needs its own 3 weights and cross-products: all solves are batched (S, P, P) numpy calls
with P = k - 1 + 3 = 12.  2,014 series x 48 half-hours x 365 days takes on the order of a minute.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .splines import CRSpline

__all__ = [
    "TARIFFS",
    "tariff_codes",
    "fit_causality",
    "fit_causality_frame",
    "profiles_long",
]

TARIFFS = ("Low", "Normal", "High")  # index 0, 1, 2 in every output array
DEFAULT_GRID = np.arange(-4.0, 6.01, 0.5)


def tariff_codes(low, high):
    """(T, H) 0/1 dummies -> int8 codes 0 = Low, 1 = Normal, 2 = High."""
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if low.shape != high.shape:
        raise ValueError("low/high shapes differ")
    if np.any((low > 0.5) & (high > 0.5)):
        raise ValueError("a slot is flagged both Low and High")
    codes = np.ones(low.shape, dtype=np.int8)
    codes[low > 0.5] = 0
    codes[high > 0.5] = 2
    return codes


def _batched_solve(A, b):
    """Solve A[s] x[s] = b[s] for (S, P, P) A and (S, P) b (numpy >= 2 broadcasting-safe)."""
    return np.linalg.solve(A, b[..., None])[..., 0]


def _class_rss(yy, bvec, G, beta):
    """Residual sum of squares per (series, class): yy - 2 b'beta + beta'G beta."""
    rss = yy - 2.0 * np.einsum("sqp,sp->sq", bvec, beta) + np.einsum("sp,qpr,sr->sq", beta, G, beta)
    return np.maximum(rss, 0.0)


def _fit_slot(Yh, x, code, x_avg, *, k, method, lambda_mode, grid, sd_floor, min_obs,
              max_iter, tol, keep_model):
    """Fit the causality model for all series at one half-hour.  Yh: (S, T)."""
    S, T = Yh.shape
    counts = np.bincount(code, minlength=3)[:3]
    present = [p for p in range(3) if counts[p] >= 1]
    q = len(present)
    sp = CRSpline(k).fit(x)
    B = sp.transform(x)
    m = B.shape[1]
    D = (code[:, None] == np.asarray(present)[None, :]).astype(float)
    X = np.hstack([B, D])
    P = m + q
    Sfull = np.zeros((P, P))
    Sfull[:m, :m] = sp.S
    rank = sp.rank

    masks = [code == p for p in present]
    G = np.stack([X[mk].T @ X[mk] for mk in masks])  # (q, P, P)
    bvec = np.stack([Yh[:, mk] @ X[mk] for mk in masks], axis=1)  # (S, q, P)
    yy = np.stack([(Yh[:, mk] ** 2).sum(axis=1) for mk in masks], axis=1)  # (S, q)
    n_p = counts[present].astype(float)  # (q,)
    est = n_p >= min_obs  # classes whose sd is estimated
    floor2 = float(sd_floor) ** 2
    XtX = G.sum(axis=0)
    rel = np.trace(XtX[:m, :m]) / np.trace(sp.S)
    b_all = bvec.sum(axis=1)  # (S, P)
    i_norm = present.index(1) if (1 in present and est[present.index(1)]) else None

    def fill_unestimated(sig2, pooled):
        if est.all():
            return sig2
        ref = sig2[:, i_norm] if i_norm is not None else pooled
        sig2 = sig2.copy()
        sig2[:, ~est] = ref[:, None]
        return sig2

    G_ng = len(grid)
    store_beta = np.empty((G_ng, S, P))
    store_sig2 = np.empty((G_ng, S, q))
    store_crit = np.empty((G_ng, S))
    store_edf = np.empty((G_ng, S))
    store_lam = np.empty((G_ng, S))
    store_iter = np.zeros((G_ng, S), dtype=int)
    store_conv = np.zeros((G_ng, S), dtype=bool)

    for gi, g in enumerate(grid):
        lam_u = rel * 10.0 ** g
        # ---- unweighted penalized LS (shared matrix): GCV and starting values
        A0 = XtX + lam_u * Sfull
        beta0 = np.linalg.solve(A0, b_all.T).T
        rss_p0 = _class_rss(yy, bvec, G, beta0)
        rss0 = rss_p0.sum(axis=1)
        edf0 = float(np.trace(np.linalg.solve(A0, XtX)))
        gcv0 = T * rss0 / max(T - edf0, 1e-8) ** 2
        pooled = np.maximum(rss0 / T, floor2)
        lam_i = lam_u / pooled  # likelihood-scale lambda, fixed during the iterations
        sig2 = fill_unestimated(np.maximum(rss_p0 / n_p, floor2), pooled)
        # ---- iteratively re-weighted penalized LS = penalized ML of the location-scale model
        active = np.ones(S, dtype=bool)
        n_iter = np.zeros(S, dtype=int)
        beta = beta0.copy()
        for it in range(max_iter):
            idx = np.flatnonzero(active)
            if idx.size == 0:
                break
            w = 1.0 / sig2[idx]
            A = np.einsum("sq,qab->sab", w, G) + lam_i[idx, None, None] * Sfull
            b = np.einsum("sq,sqa->sa", w, bvec[idx])
            beta_new = _batched_solve(A, b)
            rss_p = _class_rss(yy[idx], bvec[idx], G, beta_new)
            new = fill_unestimated(np.maximum(rss_p / n_p, floor2), pooled[idx])
            change = np.max(np.abs(new - sig2[idx]) / sig2[idx], axis=1)
            beta[idx] = beta_new
            sig2[idx] = new
            n_iter[idx] += 1
            done = change < tol
            active[idx[done]] = False
        # ---- final quantities at the converged point
        w = 1.0 / sig2
        Hbb = np.einsum("sq,qab->sab", w, G)
        A = Hbb + lam_i[:, None, None] * Sfull
        rss_p = _class_rss(yy, bvec, G, beta)
        edf = np.einsum("sii->s", np.linalg.solve(A, Hbb))
        if method == "laml":
            free = est[None, :] & (rss_p / n_p > floor2 * (1 + 1e-9))  # eta_p not at the floor
            resid_grad = bvec - np.einsum("qab,sb->sqa", G, beta)  # X_p' r_p, (S, q, P)
            U = 2.0 * w[:, :, None] * resid_grad  # d2(-l)/dbeta d eta_p
            U = U * free[:, :, None]
            schur = A - np.einsum("sqa,sqb->sab", U, U / (2.0 * n_p)[None, :, None])
            sign, logdet = np.linalg.slogdet(schur)
            bad = sign <= 0
            if bad.any():  # fall back to the known-scale Hessian
                logdet[bad] = np.linalg.slogdet(A[bad])[1]
            pen = np.einsum("sa,ab,sb->s", beta, Sfull, beta)
            ll = (n_p[None, :] * 0.5 * np.log(sig2) * est[None, :]).sum(axis=1) \
                + 0.5 * (w * rss_p * est[None, :]).sum(axis=1)
            crit = ll + 0.5 * lam_i * pen + 0.5 * logdet - 0.5 * rank * np.log(lam_i)
        elif method == "gcv":
            crit = gcv0
        else:
            raise ValueError(f"unknown method {method!r}")
        store_beta[gi] = beta
        store_sig2[gi] = sig2
        store_crit[gi] = crit
        store_edf[gi] = edf
        store_lam[gi] = lam_i
        store_iter[gi] = n_iter
        store_conv[gi] = ~active

    if lambda_mode == "per_series":
        best = np.argmin(store_crit, axis=0)
    elif lambda_mode == "shared":
        tot = store_crit.sum(axis=1) if method == "laml" else np.log(store_crit).sum(axis=1)
        best = np.full(S, int(np.argmin(tot)))
    else:
        raise ValueError(f"unknown lambda_mode {lambda_mode!r}")
    sidx = np.arange(S)
    beta = store_beta[best, sidx]
    sig2 = store_sig2[best, sidx]
    on_edge = (best == 0) | (best == G_ng - 1)

    mean_B = sp.transform(x_avg).mean(axis=0)
    mean_s = beta[:, :m] @ mean_B
    mu = np.full((S, 3), np.nan)
    sigma = np.full((S, 3), np.nan)
    xi = np.full((S, 3), np.nan)
    for j, p in enumerate(present):
        xi[:, p] = beta[:, m + j]
        mu[:, p] = mean_s + beta[:, m + j]
        if est[j]:
            sigma[:, p] = np.sqrt(sig2[:, j])
    out = dict(mu=mu, sigma=sigma, xi=xi, mean_s=mean_s, counts=counts,
               g=np.asarray(grid)[best], lam=store_lam[best, sidx], edf=store_edf[best, sidx],
               crit=store_crit[best, sidx], n_iter=store_iter[best, sidx],
               converged=store_conv[best, sidx], on_grid_edge=on_edge)
    if keep_model:
        out["model"] = dict(spline=sp, present=present, beta=beta)
    return out


def fit_causality(Y, temp, low, high, *, avg_temp=None, k=10, method="laml",
                  lambda_mode="per_series", grid=None, sd_floor=0.01, min_obs=2,
                  max_iter=200, tol=1e-9, keep_models=False):
    """Fit the causality model for S series sharing the same temperatures and tariffs.

    Parameters
    ----------
    Y : (S, T, H) or (T, H) consumption (kWh), no NaN.
    temp : (T, H) temperature tau_t^h at each slot (deg C).
    low, high : (T, H) 0/1 tariff dummies (Normal = neither).
    avg_temp : (T_avg, H) temperatures of the days over which eq. (1) averages; default the
        fit days (``temp``).
    k, method, lambda_mode, grid, sd_floor, min_obs : see the module docstring.

    Returns
    -------
    dict with
      ``mu``, ``sigma``, ``xi`` : (S, H, 3) arrays, last axis = TARIFFS (Low, Normal, High);
      ``mean_s`` (S, H): mean of shat over the averaging days;
      ``counts`` (H, 3): number of fit days with each tariff at each slot;
      ``g``, ``lam``, ``edf``, ``crit``, ``n_iter``, ``converged``, ``on_grid_edge`` (S, H);
      ``models`` (list of H dicts: spline, present classes, beta (S, P)) if ``keep_models``;
      ``settings``.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 2:
        Y = Y[None]
    S, T, H = Y.shape
    temp = np.asarray(temp, dtype=float)
    if temp.shape != (T, H):
        raise ValueError(f"temp shape {temp.shape} != {(T, H)}")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y contains non-finite values")
    codes = tariff_codes(low, high)
    if codes.shape != (T, H):
        raise ValueError("tariff dummies shape mismatch")
    avg_temp = temp if avg_temp is None else np.asarray(avg_temp, dtype=float)
    grid = DEFAULT_GRID if grid is None else np.asarray(grid, dtype=float)

    keys2 = ("g", "lam", "edf", "crit", "n_iter", "converged", "on_grid_edge", "mean_s")
    res = {key: np.empty((S, H, 3)) for key in ("mu", "sigma", "xi")}
    res.update({key: [None] * H for key in keys2})
    res["counts"] = np.zeros((H, 3), dtype=int)
    models = []
    for h in range(H):
        o = _fit_slot(Y[:, :, h], temp[:, h], codes[:, h], avg_temp[:, h], k=k, method=method,
                      lambda_mode=lambda_mode, grid=grid, sd_floor=sd_floor, min_obs=min_obs,
                      max_iter=max_iter, tol=tol, keep_model=keep_models)
        for key in ("mu", "sigma", "xi"):
            res[key][:, h, :] = o[key]
        for key in keys2:
            res[key][h] = o[key]
        res["counts"][h] = o["counts"]
        if keep_models:
            models.append(o["model"])
    for key in keys2:
        res[key] = np.stack(res[key], axis=1)
    if keep_models:
        res["models"] = models
    res["settings"] = dict(k=k, method=method, lambda_mode=lambda_mode,
                           grid=[float(v) for v in grid], sd_floor=sd_floor, min_obs=min_obs,
                           max_iter=max_iter, tol=tol, n_fit_days=T,
                           n_avg_days=int(avg_temp.shape[0]))
    return res


def _cols(prefix, H=48):
    return [f"{prefix}_{h:02d}" for h in range(H)]


def fit_causality_frame(df, *, avg_df=None, cons_prefix="consumption", **kwargs):
    """Convenience wrapper for one series stored in the authors' schema (one row per day:
    ``consumption_hh``, ``temp_hh``, ``low_hh``, ``high_hh``).  ``avg_df`` (same schema, only the
    temperatures are used) sets the averaging days of eq. (1)."""
    Y = df[_cols(cons_prefix)].to_numpy(dtype=float)
    temp = df[_cols("temp")].to_numpy(dtype=float)
    low = df[_cols("low")].to_numpy(dtype=float)
    high = df[_cols("high")].to_numpy(dtype=float)
    avg_temp = None if avg_df is None else avg_df[_cols("temp")].to_numpy(dtype=float)
    return fit_causality(Y, temp, low, high, avg_temp=avg_temp, **kwargs)


def profiles_long(res, series_labels=None):
    """Long table (series, instant, price, type in {mean, sd}, kWh) like the authors'
    ``agreg_profiles_nmf.feather`` (prices in lower case)."""
    S, H, _ = res["mu"].shape
    labels = list(range(S)) if series_labels is None else list(series_labels)
    rows = []
    for key, typ in (("mu", "mean"), ("sigma", "sd")):
        arr = res[key]
        for s in range(S):
            for p, name in enumerate(TARIFFS):
                for h in range(H):
                    rows.append((labels[s], h, name.lower(), typ, arr[s, h, p]))
    return pd.DataFrame(rows, columns=["series", "instant", "price", "type", "kWh"])
