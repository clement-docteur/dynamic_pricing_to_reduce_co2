"""Semi-parametric (GAM) daily-profile generator: Bregere & Bessa (2020), Sec. VI (eqs. 9-11)
and Appendix E (eqs. 37-40).  Data-agnostic: works on any frame in the authors' schema (one row
per day of ONE cluster: ``date``, ``consumption_hh``, ``temp_hh`` (deg C), ``low_hh``,
``high_hh``, ``temp_smooth``, ``pos_y``, ``work_day``; hh = 00..47).

Model, per half-hour h (fitted on the training days only):

    f^h(x_t, p_t^h) = b0 + s_tau(tau_t^h) + s_taubar(taubar_t) + s_kappa(kappa_t)
                      + alpha^h w_t + xi_Low^h 1{Low} + xi_High^h 1{High}                 (eq. 11)

with tau = ``temp_hh``, taubar = ``temp_smooth``, kappa = ``pos_y``, w = ``work_day``.
mgcv is unavailable, so each s is a centred penalized cubic regression spline
(``lib.splines.CRSpline``, k = 10 knots at quantiles of the training values, an mgcv
``bs = "cr"`` analogue) and the three smoothing parameters of each half-hour are chosen by GCV
(mgcv's default ``GCV.Cp`` for Gaussian GAMs; ``method = "reml"`` available) with a coarse grid
followed by L-BFGS-B refinement (``lib.splines.select_smoothing``).

Noise (eq. 10, App. E):  sigma^h(p) comes from the causality model (``lib.causality``) refitted
on the cluster's training-day aggregate series (``sigma_df``; default = the training frame).
Fallback: when tariff p is absent (or seen only once) at slot h in that data, sigma^h(p) is NaN
and is replaced by sigma^h(Normal); likewise a tariff dummy never active at slot h in the GAM
training data is dropped and its xi^h is set to 0.  Both fallbacks are recorded in the params.
Standardized training residuals e_t^h = (Y_t^h - fhat^h) / sigma^h(p_t^h)  (eq. 37) give
Sigma = their empirical correlation matrix (eqs. 38-40).  Generation:
Y = f + diag(sigma(p)) E, E ~ N(0, Sigma), via the Cholesky factor of Sigma (a tiny diagonal
jitter is added only if the factorisation fails).  Samples are not clipped at zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .causality import TARIFFS, fit_causality, tariff_codes
from .effects import HIGH_WINDOW, LOW_WINDOW
from .splines import CRSpline, select_smoothing

__all__ = [
    "H",
    "PRICES",
    "frame_arrays",
    "fit_gam_generator",
    "predict_mean",
    "sigma_matrix",
    "generate",
    "set_sigma",
    "tariff_scenario",
    "SCENARIOS",
]

H = 48
PRICES = {"Low": 0.0399, "Normal": 0.1176, "High": 0.672}  # GBP/kWh, as in the authors' price_hh
# The paper's tariff scenarios, windows from lib.effects (single source of truth; REPRO_SPEC section 7):
# Low = slots 9-19 (04:00-09:30 GMT), High = slots 39-44 (19:00-22:00 GMT), endpoints inclusive.
SCENARIOS = {"Normal": None, "Low": ("low", tuple(LOW_WINDOW)), "High": ("high", tuple(HIGH_WINDOW))}


def _cols(prefix):
    return [f"{prefix}_{h:02d}" for h in range(H)]


def frame_arrays(df, require_y=True):
    """Extract arrays from a frame in the authors' schema (rows sorted by date)."""
    if "cluster" in df.columns and df["cluster"].nunique() > 1:
        raise ValueError("frame contains several clusters; pass one cluster at a time")
    df = df.sort_values("date")
    out = dict(
        dates=pd.to_datetime(df["date"]).to_numpy(),
        temp=df[_cols("temp")].to_numpy(dtype=float),
        low=df[_cols("low")].to_numpy(dtype=float),
        high=df[_cols("high")].to_numpy(dtype=float),
        tbar=df["temp_smooth"].to_numpy(dtype=float),
        kappa=df["pos_y"].to_numpy(dtype=float),
        w=df["work_day"].to_numpy(dtype=float),
    )
    if require_y:
        out["Y"] = df[_cols("consumption")].to_numpy(dtype=float)
    for key, val in out.items():
        if key != "dates" and not np.all(np.isfinite(val)):
            raise ValueError(f"non-finite values in {key}")
    return out


def _design(slot, a, h):
    """Design matrix for half-hour h given fitted spline objects in ``slot``."""
    parts = [np.ones((a["temp"].shape[0], 1)),
             slot["sp_tau"].transform(a["temp"][:, h]),
             slot["sp_tbar"].transform(a["tbar"]),
             slot["sp_kappa"].transform(a["kappa"]),
             a["w"][:, None]]
    if slot["has_low"]:
        parts.append(a["low"][:, h][:, None])
    if slot["has_high"]:
        parts.append(a["high"][:, h][:, None])
    return np.hstack(parts)


def fit_gam_generator(train_df, *, k=10, method="gcv", sigma_df=None, causality_kwargs=None,
                      refine=True, grid=None, jitter=1e-10):
    """Fit the 48 half-hourly GAMs, sigma^h(p) and Sigma on the training frame of one cluster.

    Returns a picklable dict (numpy arrays, floats and CRSpline objects only).
    """
    a = frame_arrays(train_df)
    T0 = a["Y"].shape[0]
    sp_tbar = CRSpline(k).fit(a["tbar"])
    sp_kappa = CRSpline(k).fit(a["kappa"])
    slots = []
    f_train = np.empty((T0, H))
    info = dict(lam=np.full((H, 3), np.nan), g=np.full((H, 3), np.nan),
                edf_smooth=np.full((H, 3), np.nan), edf=np.full(H, np.nan),
                crit=np.full(H, np.nan), alpha=np.full(H, np.nan), b0=np.full(H, np.nan),
                xi_low=np.zeros(H), xi_high=np.zeros(H),
                xi_low_missing=np.zeros(H, dtype=bool), xi_high_missing=np.zeros(H, dtype=bool))
    for h in range(H):
        slot = dict(sp_tau=CRSpline(k).fit(a["temp"][:, h]), sp_tbar=sp_tbar, sp_kappa=sp_kappa,
                    has_low=bool(a["low"][:, h].sum() > 0), has_high=bool(a["high"][:, h].sum() > 0))
        X = _design(slot, a, h)
        m1, m2, m3 = slot["sp_tau"].n_coef, sp_tbar.n_coef, sp_kappa.n_coef
        blocks = [(1, slot["sp_tau"].S), (1 + m1, sp_tbar.S), (1 + m1 + m2, sp_kappa.S)]
        res = select_smoothing(X, a["Y"][:, h], blocks, method=method, refine=refine, grid=grid)
        beta = res["beta"]
        slot["beta"] = beta
        slots.append(slot)
        f_train[:, h] = X @ beta
        j = 1 + m1 + m2 + m3
        info["lam"][h], info["g"][h] = res["lam"], res["g"]
        info["edf_smooth"][h], info["edf"][h] = res["edf_blocks"], res["edf"]
        info["crit"][h], info["b0"][h], info["alpha"][h] = res["criterion"], beta[0], beta[j]
        j += 1
        if slot["has_low"]:
            info["xi_low"][h] = beta[j]
            j += 1
        else:
            info["xi_low_missing"][h] = True
        if slot["has_high"]:
            info["xi_high"][h] = beta[j]
        else:
            info["xi_high_missing"][h] = True

    # ---- sigma^h(p): causality model refitted on the cluster's (training-day) aggregate
    sdf = train_df if sigma_df is None else sigma_df
    sa = frame_arrays(sdf)
    ckw = dict(causality_kwargs or {})
    cres = fit_causality(sa["Y"], sa["temp"], sa["low"], sa["high"], **ckw)
    sigma_raw = cres["sigma"][0]  # (H, 3) Low, Normal, High
    params = dict(
        k=k, method=method, n_train_days=T0, slots=slots, info=info,
        train_dates=a["dates"], f_train=f_train, y_train=a["Y"],
        codes_train=tariff_codes(a["low"], a["high"]),
        causality=dict(sigma_raw=sigma_raw, mu=cres["mu"][0], counts=cres["counts"],
                       edf=cres["edf"][0], g=cres["g"][0], settings=cres["settings"],
                       n_days=int(sa["Y"].shape[0])),
        jitter=jitter, tariffs=TARIFFS,
    )
    return set_sigma(params, sigma_raw)


def set_sigma(params, sigma_table, source="causality"):
    """(Re)set sigma^h(p) (H x 3 table, columns Low/Normal/High; NaN -> sigma^h(Normal)) and
    recompute Sigma from the stored standardized training residuals.  Returns ``params``."""
    sig = np.array(sigma_table, dtype=float, copy=True)
    if sig.shape != (H, 3):
        raise ValueError(f"sigma table must be (48, 3), got {sig.shape}")
    fallback = np.isnan(sig)
    if np.isnan(sig[:, 1]).any():  # Normal itself missing (should not happen): global median
        sig[np.isnan(sig[:, 1]), 1] = np.nanmedian(sig[:, 1])
    for p in (0, 2):
        miss = np.isnan(sig[:, p])
        sig[miss, p] = sig[miss, 1]
    if np.any(sig <= 0):
        raise ValueError("non-positive sigma")
    codes = params["codes_train"]
    sd_train = sig[np.arange(H)[None, :], codes]  # (T0, H)
    e = (params["y_train"] - params["f_train"]) / sd_train  # eq. 37
    Sigma = np.corrcoef(e, rowvar=False)  # eqs. 38-40
    jit = 0.0
    while True:
        try:
            L = np.linalg.cholesky(Sigma + jit * np.eye(H))
            break
        except np.linalg.LinAlgError:
            jit = max(params["jitter"], 10 * jit)
    params.update(sigma=sig, sigma_fallback=fallback, sigma_source=source, Sigma=Sigma,
                  chol=L, jitter_used=jit, std_resid_sd=e.std(axis=0, ddof=1),
                  std_resid_mean=e.mean(axis=0))
    return params


def predict_mean(params, cond_df):
    """f^h(x_t, p_t^h) for every day of ``cond_df`` (sorted by date) -> (n_days, 48)."""
    a = frame_arrays(cond_df, require_y=False)
    out = np.empty((a["temp"].shape[0], H))
    for h, slot in enumerate(params["slots"]):
        out[:, h] = _design(slot, a, h) @ slot["beta"]
    return out


def sigma_matrix(params, cond_df):
    """sigma^h(p_t^h) for every day of ``cond_df`` (sorted by date) -> (n_days, 48)."""
    a = frame_arrays(cond_df, require_y=False)
    codes = tariff_codes(a["low"], a["high"])
    return params["sigma"][np.arange(H)[None, :], codes]


def generate(params, cond_df, n_samples, rng):
    """Draw ``n_samples`` daily profiles per day of ``cond_df`` (ascending dates):
    Y = f + diag(sigma(p)) E, E ~ N(0, Sigma).  Returns (n_samples, n_days, 48) in kWh."""
    f = predict_mean(params, cond_df)
    sd = sigma_matrix(params, cond_df)
    z = rng.standard_normal((n_samples, f.shape[0], H))
    E = z @ params["chol"].T
    return f[None] + sd[None] * E


def tariff_scenario(cond_df, scenario):
    """Copy of ``cond_df`` whose 96 tariff dummies follow one of the paper's scenarios (``SCENARIOS``):
    Normal = all zero; Low = low slots 9-19 set to 1; High = high slots 39-44 set to 1 (lib.effects windows).
    ``price_hh`` / ``normal_hh`` columns (if present) are made consistent; nothing else changes."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    df = cond_df.copy()
    low = np.zeros((len(df), H))
    high = np.zeros((len(df), H))
    spec = SCENARIOS[scenario]
    if spec is not None:
        kind, slots = spec
        (low if kind == "low" else high)[:, list(slots)] = 1.0
    df[_cols("low")] = low
    df[_cols("high")] = high
    if all(c in df.columns for c in _cols("normal")):
        df[_cols("normal")] = 1.0 - low - high
    if all(c in df.columns for c in _cols("price")):
        df[_cols("price")] = np.where(low > 0, PRICES["Low"],
                                      np.where(high > 0, PRICES["High"], PRICES["Normal"]))
    return df
