"""Penalized cubic regression splines for additive models (an mgcv ``s(x, bs="cr")`` analogue).

Used by ``lib/causality.py`` (Bregere & Bessa 2020, Sec. IV-A / App. B) and
``lib/gam_generator.py`` (Sec. VI / App. E).  The authors used R/mgcv, which is not available
here, so the smooth terms are re-implemented with numpy/scipy.

Documented modelling choices
----------------------------
* **Basis.**  Natural cubic spline with ``k`` knots (default ``k = 10``, mgcv's default basis
  dimension) placed at quantiles of the *unique* covariate values (mgcv ``place.knots`` rule).
  It is represented with cubic B-splines (``scipy.interpolate.BSpline``) plus the two natural
  boundary constraints f''(first knot) = f''(last knot) = 0, which is exactly the function
  space of mgcv's ``bs="cr"`` basis.
* **Identifiability ("centred" smooth).**  Sum-to-zero constraint over the fitting data,
  sum_t f(x_t) = 0, absorbed by a QR null-space re-parameterisation (as mgcv does).  A term
  therefore has ``k - 1`` free coefficients and its fitted values average to zero over the
  data it was set up with.
* **Penalty.**  Exact integrated squared second derivative  int f''(x)^2 dx  over the knot range
  (Gauss-Legendre quadrature, exact because f'' is piecewise linear).  Its null space inside a
  centred term is the (centred) linear function, so ``rank(S) = k - 2``.
* **Extrapolation.**  Linear beyond the boundary knots (natural-spline continuation; value and
  slope continuous), so predictions for unseen temperatures / dates stay sane.
* **Smoothing parameters.**  One lambda per smooth term, parameterised on a data-scale-free
  log10 grid:  lambda_j = 10**g_j * tr(X_j' W X_j) / tr(S_j).  Criteria:
    - ``"gcv"``  : n * RSS_w / (n - edf)^2  (mgcv's default ``GCV.Cp`` for a Gaussian GAM);
    - ``"reml"`` : Laplace/REML marginal likelihood of the Gaussian additive model with the
      scale profiled out (``scale=None``) or known (``scale=1``, used when the weights already
      are 1/sigma^2, as in the location-scale causality model).
  Selection = coarse grid search followed by an L-BFGS-B refinement of the continuous
  log-lambdas inside bounds (on log GCV for "gcv", so that the stopping rule is scale-free).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline
from scipy.optimize import minimize

__all__ = [
    "CRSpline",
    "penalty_blocks_matrix",
    "penalized_fit",
    "select_smoothing",
]


class CRSpline:
    """Centred, penalized natural cubic regression spline term.

    Parameters
    ----------
    k : int
        Number of knots = dimension of the natural-cubic-spline space before centring
        (mgcv's ``k``).  The term has ``k - 1`` coefficients after centring.

    After :meth:`fit` the attributes ``knots``, ``Z`` (constraint null space), ``S`` (penalty
    for the ``k - 1`` coefficients), ``rank`` (penalty rank) and ``n_coef`` are available.
    The object only stores numpy arrays and floats, so it pickles cleanly.
    """

    def __init__(self, k: int = 10):
        if k < 3:
            raise ValueError("k must be >= 3 for a natural cubic spline")
        self.k = int(k)
        self.knots = None
        self.Z = None
        self.S = None
        self.rank = None
        self.n_coef = None
        self._bs_cache = None

    # ------------------------------------------------------------------ pickling helpers
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_bs_cache"] = None  # rebuilt lazily
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ------------------------------------------------------------------ raw B-spline basis
    def _full_knots(self):
        a, b = self.knots[0], self.knots[-1]
        return np.r_[[a] * 3, self.knots, [b] * 3]

    def _bspline(self):
        if self._bs_cache is None:
            t = self._full_knots()
            nb = len(t) - 4
            bs = BSpline(t, np.eye(nb), 3, extrapolate=False)
            a, b = self.knots[0], self.knots[-1]
            d1 = bs.derivative(1)(np.array([a, b]))
            self._bs_cache = (bs, d1[0].copy(), d1[1].copy())
        return self._bs_cache

    def _raw_basis(self, x):
        """Unconstrained cubic B-spline basis with linear extrapolation outside the knots."""
        x = np.asarray(x, dtype=float).ravel()
        bs, d1a, d1b = self._bspline()
        a, b = self.knots[0], self.knots[-1]
        B = bs(np.clip(x, a, b))
        lo, hi = x < a, x > b
        if lo.any():
            B[lo] += np.outer(x[lo] - a, d1a)
        if hi.any():
            B[hi] += np.outer(x[hi] - b, d1b)
        return B

    def _raw_penalty(self):
        """Omega_ij = int B_i''(x) B_j''(x) dx over [first knot, last knot] (exact)."""
        bs, _, _ = self._bspline()
        d2 = bs.derivative(2)
        nodes, weights = leggauss(3)
        uk = np.unique(self.knots)
        nb = bs.c.shape[0]
        om = np.zeros((nb, nb))
        for lo, hi in zip(uk[:-1], uk[1:]):
            # evaluate strictly inside the interval (B'' is discontinuous at knots)
            xq = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
            wq = 0.5 * (hi - lo) * weights
            D = d2(xq)
            om += (D * wq[:, None]).T @ D
        return om

    # ------------------------------------------------------------------ public API
    def fit(self, x):
        """Place knots on ``x`` and build the centred basis / penalty.  Returns ``self``."""
        x = np.asarray(x, dtype=float).ravel()
        if not np.all(np.isfinite(x)):
            raise ValueError("non-finite covariate values")
        ux = np.unique(x)
        k = min(self.k, ux.size)
        if k < 3:
            raise ValueError(f"need at least 3 distinct covariate values, got {ux.size}")
        self.k_used = k
        self.knots = np.quantile(ux, np.linspace(0.0, 1.0, k))
        self._bs_cache = None
        bs, _, _ = self._bspline()
        a, b = self.knots[0], self.knots[-1]
        B = self._raw_basis(x)
        nat = bs.derivative(2)(np.array([a, b]))  # natural boundary constraints
        C = np.vstack([nat, B.sum(axis=0, keepdims=True)])  # + sum-to-zero over the data
        Q, _ = np.linalg.qr(C.T, mode="complete")
        self.Z = Q[:, C.shape[0]:]
        S = self.Z.T @ self._raw_penalty() @ self.Z
        S = 0.5 * (S + S.T)
        S /= np.linalg.norm(S)  # cosmetic; lambdas are re-scaled relative to the data anyway
        ev = np.linalg.eigvalsh(S)
        self.rank = int((ev > ev.max() * 1e-9).sum())
        self.S = S
        self.n_coef = self.Z.shape[1]
        self.x_range = (float(x.min()), float(x.max()))
        return self

    def transform(self, x):
        """Centred basis matrix (n, k-1) evaluated at ``x`` (linear extrapolation outside)."""
        if self.Z is None:
            raise RuntimeError("CRSpline not fitted")
        return self._raw_basis(x) @ self.Z

    def fit_transform(self, x):
        return self.fit(x).transform(x)


# ======================================================================================
# Penalized least squares with several smoothing parameters
# ======================================================================================

def penalty_blocks_matrix(p, blocks, lam):
    """Assemble sum_j lam_j * S_j embedded at column offsets.  ``blocks`` = [(start, S_j)]."""
    out = np.zeros((p, p))
    for (start, S), l in zip(blocks, lam):
        m = S.shape[0]
        out[start:start + m, start:start + m] += l * S
    return out


def _crit_from_parts(method, n, wrss, pen, logdetA, lam, ranks, edf, Mp, scale):
    if method == "gcv":
        return n * wrss / np.maximum(n - edf, 1e-8) ** 2
    if method == "reml":
        lr = np.sum([r * np.log(l) for r, l in zip(ranks, lam)], axis=0)
        if scale is None:  # profiled unknown scale
            nm = n - Mp
            phi = np.maximum((wrss + pen) / nm, 1e-300)
            return 0.5 * nm * np.log(phi) + 0.5 * logdetA - 0.5 * lr
        return 0.5 * (wrss + pen) / scale + 0.5 * logdetA - 0.5 * lr
    raise ValueError(f"unknown method {method!r}")


def penalized_fit(XtWX, XtWy, yWy, n, blocks, lam, method="gcv", scale=None):
    """Solve (X'WX + S_lambda) beta = X'Wy for one lambda vector.

    Returns a dict with beta, edf (total), edf per coefficient (``edf_coef``; sum a block's
    entries for its edf), weighted RSS, penalty value beta'S_lambda beta, log|X'WX + S_lambda|
    and the inverse of the penalized Hessian (``Ainv``).  ``method``/``scale`` are accepted for
    signature compatibility; the criterion itself is computed by the caller.
    """
    p = XtWX.shape[0]
    lam = np.asarray(lam, dtype=float)
    A = XtWX + penalty_blocks_matrix(p, blocks, lam)
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        A = A + np.eye(p) * 1e-10 * np.trace(A) / p
        L = np.linalg.cholesky(A)
    Linv = np.linalg.inv(L)
    Ainv = Linv.T @ Linv
    beta = Ainv @ XtWy
    wrss = float(yWy - 2.0 * beta @ XtWy + beta @ XtWX @ beta)
    wrss = max(wrss, 0.0)
    F = Ainv @ XtWX  # influence-type matrix in coefficient space
    edf_coef = np.diag(F)
    edf = float(edf_coef.sum())
    pen = float(beta @ (A - XtWX) @ beta)
    logdetA = 2.0 * float(np.log(np.diag(L)).sum())
    return dict(beta=beta, edf=edf, edf_coef=edf_coef, wrss=wrss, pen=pen, logdetA=logdetA,
                Ainv=Ainv, lam=lam)


def select_smoothing(X, y, blocks, w=None, method="gcv", scale=None, grid=None,
                     refine=True, bounds=(-8.0, 8.0)):
    """Choose one smoothing parameter per penalty block and fit.

    Parameters
    ----------
    X : (n, p) design matrix; y : (n,) response; w : optional (n,) prior weights.
    blocks : list of (start, S_j) -- penalty S_j acts on columns start:start+m_j.
    method : "gcv" or "reml"; ``scale`` only used by "reml" (None = unknown, profiled).
    grid : 1-D array of log10 relative lambdas for the coarse search (common grid per block).

    Returns dict(beta, lam, g (log10 relative lambdas), edf, edf_blocks, criterion, wrss, ...).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if w is None:
        w = np.ones(n)
    Xw = X * w[:, None]
    XtWX = X.T @ Xw
    XtWy = Xw.T @ y
    yWy = float((w * y * y).sum())
    S_list = [S for _, S in blocks]
    ranks = []
    rel = []
    for start, S in blocks:
        m = S.shape[0]
        ev = np.linalg.eigvalsh(S)
        ranks.append(int((ev > ev.max() * 1e-9).sum()))
        rel.append(np.trace(XtWX[start:start + m, start:start + m]) / np.trace(S))
    ranks = np.array(ranks)
    rel = np.array(rel)
    Mp = p - int(ranks.sum())
    nb = len(blocks)
    if grid is None:
        grid = np.arange(-6.0, 6.01, 1.0) if nb == 1 else np.arange(-6.0, 6.01, 2.0)

    def crit(g):
        lam = rel * 10.0 ** np.asarray(g)
        r = penalized_fit(XtWX, XtWy, yWy, n, blocks, lam)
        return _crit_from_parts(method, n, r["wrss"], r["pen"], r["logdetA"], lam, ranks,
                                r["edf"], Mp, scale), r

    if nb == 0:
        r = penalized_fit(XtWX, XtWy, yWy, n, blocks, np.zeros(0))
        c = _crit_from_parts(method, n, r["wrss"], r["pen"], r["logdetA"], np.zeros(0), ranks,
                             r["edf"], Mp, scale)
        g_best = np.zeros(0)
    else:
        # coarse full grid (nb <= 3 in this project, so at most 7**3 = 343 points)
        mesh = np.array(np.meshgrid(*([grid] * nb), indexing="ij")).reshape(nb, -1).T
        vals = np.array([crit(g)[0] for g in mesh])
        g_best = mesh[int(np.argmin(vals))].astype(float)
        if refine:
            # GCV is in squared data units (~1e-4 for kWh data): optimise its log so the
            # gradient is scale-free (otherwise L-BFGS-B's default gtol stops it at the start).
            if method == "gcv":
                def obj(g):
                    return float(np.log(max(crit(g)[0], 1e-300)))
            else:
                def obj(g):
                    return float(crit(g)[0])
            res = minimize(obj, g_best, method="L-BFGS-B", bounds=[bounds] * nb,
                           options=dict(maxiter=500, ftol=1e-13, gtol=1e-9))
            if crit(res.x)[0] <= vals.min():
                g_best = np.asarray(res.x, dtype=float)
        c, r = crit(g_best)
    lam = rel * 10.0 ** g_best if nb else np.zeros(0)
    edf_blocks = []
    for start, S in blocks:
        m = S.shape[0]
        edf_blocks.append(float(r["edf_coef"][start:start + m].sum()))
    return dict(beta=r["beta"], lam=lam, g=g_best, edf=r["edf"], edf_blocks=edf_blocks,
                criterion=float(c), wrss=r["wrss"], method=method, n=n, p=p,
                penalty_ranks=ranks.tolist())
