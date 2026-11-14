"""Step 2: demand-response simulators. Both expose the same method

    log_effect(cluster, low, high, days, timeline=False) -> log consumption multiplier of a schedule vs an all-Normal day

so that every simulator plugs into the same counterfactual (per cluster, day and slot):

    y_hat(z) = y_observed * exp(m(z) - m(z_historical))

Only the tariff response is modelled; the level, weather and noise of each day come from the observed data.

* ResponseModel (main): per cluster, log(ToU cluster mean) - log(Std-household mean), regressed on slot x day-type
  intercepts, temperature x time-of-day band, and tariff terms. The flat-tariff (Std) households are the control, so
  weather, holidays and seasonal swings cancel out (a difference-in-differences design). Specifications (SPECS):
  'main' = in-block High/Low effects (pooled + ridge-shrunk time-of-day band deviations) + the edge effects verified
  in the CVAE reproduction (2 h dip before Low, the half-hour after Low, 2 h before High); 'full' adds post-block and
  day-level terms; 'inblock' has no edge terms.
* CVAEEnsemble (sensitivity, the proposal's original simulator): the 20 CVAE restarts per cluster trained on
  Jan-Aug 2013 in CVAE_reproduction/scripts/50_chrono_split.py (track B, without position-in-year), run in numpy.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .common import H, REPRO, cols, load_torch_checkpoint, read_parquet, split_of

BANDS = {"night 00-06": range(1, 13), "morning 06-12": range(13, 25), "afternoon 12-16": range(25, 33),
         "peak 16-20": range(33, 41), "evening 20-24": list(range(41, 48)) + [0]}
BAND_OF = np.zeros(H, int)
for _i, _s in enumerate(BANDS.values()):
    BAND_OF[list(_s)] = _i
POOLED = ["in_High", "in_Low"]
BAND_TERMS = [f"in_High|{b}" for b in BANDS] + [f"in_Low|{b}" for b in BANDS]
EDGE_TERMS = ["pre2h_High", "post2h_High", "pre2h_Low", "post30m_Low", "post2h_Low", "day_High", "day_Low"]
SPECS = {"main": ["pre2h_Low", "post30m_Low", "pre2h_High"], "full": EDGE_TERMS, "inblock": []}


def pct_effect(log_multiplier):
    return 100 * (np.exp(log_multiplier) - 1)


def _any_shift(x, ks):
    """True where x is True k steps later (k > 0) or earlier (k < 0), along the last axis, zero-filled."""
    out = np.zeros_like(x)
    for k in ks:
        if k > 0:
            out[..., :-k] |= x[..., k:]
        else:
            out[..., -k:] |= x[..., :k]
    return out


def tariff_regressors(low, high, timeline=False) -> dict:
    """Tariff regressors for schedules given as (n_days, 48) 0/1 arrays. timeline=True treats the rows as one
    continuous series (blocks and their edges may cross midnight, as in the historical schedule); timeline=False
    treats each day on its own. Returns name -> (n_days, 48) float."""
    low, high = np.asarray(low, bool), np.asarray(high, bool)
    n = low.shape[0]
    if timeline:
        L, Hh = low.reshape(1, -1), high.reshape(1, -1)
        band = np.tile(BAND_OF, n)[None]
        anyH, anyL = np.repeat(high.any(1), H)[None], np.repeat(low.any(1), H)[None]
    else:
        L, Hh, band = low, high, BAND_OF[None]
        anyH, anyL = high.any(1, keepdims=True), low.any(1, keepdims=True)
    ev = L | Hh
    preH = ~Hh & _any_shift(Hh, range(1, 5))            # a High slot within the next 2 h
    postH = ~Hh & _any_shift(Hh, range(-4, 0))          # a High slot within the last 2 h
    preL = ~L & _any_shift(L, range(1, 5))
    postL1 = ~L & _any_shift(L, [-1])
    postL2 = ~L & ~postL1 & _any_shift(L, range(-4, -1))
    r = {"in_High": Hh, "in_Low": L, "pre2h_High": preH, "post2h_High": postH, "pre2h_Low": preL,
         "post30m_Low": postL1, "post2h_Low": postL2}
    for i, b in enumerate(BANDS):
        r[f"in_High|{b}"] = Hh & (band == i)
        r[f"in_Low|{b}"] = L & (band == i)
    r["day_High"] = anyH & ~Hh & ~preH & ~postH & ~ev
    r["day_Low"] = anyL & ~L & ~preL & ~postL1 & ~postL2 & ~ev
    return {k: np.broadcast_to(v, L.shape).reshape(n, H).astype(float) for k, v in r.items()}


class ResponseModel:
    """Per-cluster log-ratio regression: fit_cluster() on chosen days, then log_effect() for any schedule."""

    def __init__(self, spec="main", ridge_band=30.0, scale=1.0):
        self.spec, self.ridge_band, self.scale = spec, ridge_band, scale
        self.terms = POOLED + BAND_TERMS + SPECS[spec]
        self.p0 = 2 * H + len(BANDS)                       # columns before the tariff terms
        self.theta: dict[int, np.ndarray] = {}
        self.baseline_coef: dict[int, np.ndarray] = {}

    def design(self, low, high, temp, daytype):
        """Design matrix for the historical year: [slot x day-type intercepts | temperature x band | tariff terms]."""
        n = low.shape[0]
        R = tariff_regressors(low, high, timeline=True)
        slot = np.tile(np.arange(H), n)
        icpt = np.zeros((n * H, 2 * H))
        icpt[np.arange(n * H), slot + H * np.repeat(daytype, H)] = 1.0
        tb = np.zeros((n * H, len(BANDS)))
        tb[np.arange(n * H), BAND_OF[slot]] = (temp.ravel() - 10.0) / 10.0
        return np.hstack([icpt, tb, np.column_stack([R[k].ravel() for k in self.terms])])

    def penalty(self, n_cols):
        pen = np.zeros(n_cols)
        pen[self.p0 + len(POOLED): self.p0 + len(POOLED) + len(BAND_TERMS)] = self.ridge_band
        return pen

    def fit_cluster(self, c, logratio, X, rows):
        """Ridge fit (penalty on the band deviations only) of one cluster's log ratio on the day mask `rows`."""
        keep = np.repeat(rows, H)
        Xk, y = X[keep], logratio.ravel()[keep]
        beta = np.linalg.solve(Xk.T @ Xk + np.diag(self.penalty(X.shape[1])), Xk.T @ y)
        self.theta[c], self.baseline_coef[c] = beta[self.p0:], beta[: self.p0]
        return beta

    def combine(self, theta, R):
        return sum(theta[i] * R[k] for i, k in enumerate(self.terms))

    def log_effect(self, c, low, high, days=None, timeline=False):
        return self.combine(self.theta[c] * self.scale, tariff_regressors(low, high, timeline))

    def baseline_pred(self, c, X):
        """Fitted log ratio WITHOUT tariff terms, (n_days, 48) (used for the held-out validation)."""
        return (X[:, : self.p0] @ self.baseline_coef[c]).reshape(-1, H)

    def in_block_effects(self, c, theta=None):
        """Multiplicative effects, %: in-block by band = exp(pooled + band deviation) - 1; edges = exp(coef) - 1."""
        th = self.theta[c] if theta is None else theta
        ix = {k: i for i, k in enumerate(self.terms)}
        out = {f"{kind}|{b}": pct_effect(th[ix[f"in_{kind}"]] + th[ix[f"in_{kind}|{b}"]])
               for kind in ("High", "Low") for b in BANDS}
        out.update({k: pct_effect(th[ix[k]]) for k in SPECS[self.spec]})
        return out


# ------------------------------------------------------------------------------------------------ CVAE ensemble
def cvae_conditions() -> pd.DataFrame:
    """Leak-free CVAE conditions for 2013 (as scripts/50_chrono_split.leak_free_conditions, track B / ERA5)."""
    sys.path.insert(0, str(REPRO))
    from lib import conditions as Cnd  # CVAE_reproduction/lib/conditions.py (works because lib/ has no __init__.py)
    pc = read_parquet(REPRO / "data" / "power_consump_ours.parquet")
    g = pc[pc.cluster == 0].sort_values("date").reset_index(drop=True)
    tr = split_of(g["date"]) == "train"
    ts = g["temp_smooth"].to_numpy(float)
    out = pd.DataFrame({"temp_smooth_n": (ts - ts[tr].min()) / (ts[tr].max() - ts[tr].min()),
                        "pos_y": g["pos_y"].to_numpy(float), "work_day": g["work_day"].to_numpy(float)})
    s = Cnd.pca_fit(g[cols("temp")].to_numpy(float), 3, fit_rows=np.where(tr)[0])["scores"]
    s = (s - s[tr].min()) / (s[tr].max() - s[tr].min())
    for j in range(3):
        out[f"pca_temp_{j}"] = s[:, j]
    return out


class CVAEEnsemble:
    """Mean output of the 20 CVAE restarts per cluster x 20 latent draws (the latent space has collapsed, so the
    draws barely matter)."""

    def __init__(self, n_eps=20, seed=0):
        self.cond = cvae_conditions()
        self.models = {}
        for c in range(4):
            self.models[c] = []
            for s in range(20):
                ck = load_torch_checkpoint(REPRO / "outputs" / "chrono" / "models" / f"B_c{c}_no_pos_y_s{s:02d}.pt")
                sd, meta = ck["state_dict"], ck["meta"]
                self.models[c].append(dict(W1=sd["dec_hidden.weight"], b1=sd["dec_hidden.bias"],
                                           W2=sd["dec_out.weight"], b2=sd["dec_out.bias"], lo=meta["conso_min"],
                                           hi=meta["conso_max"], cols=tuple(meta["cond_cols"]), d=int(meta["latent_dim"])))
        self.eps = np.random.default_rng(seed).standard_normal((n_eps, 4))
        self._normal_level = {}

    def cond_matrix(self, cond_cols, low, high, days):
        base = self.cond.iloc[days]
        M = np.zeros((len(days), len(cond_cols)))
        for j, col in enumerate(cond_cols):
            if col.startswith("low_"):
                M[:, j] = low[:, int(col[4:])]
            elif col.startswith("high_"):
                M[:, j] = high[:, int(col[5:])]
            else:
                M[:, j] = base[col].to_numpy(float)
        return M

    @staticmethod
    def decode(m, Z, C):
        h = np.maximum(np.hstack([Z, C]) @ m["W1"].T + m["b1"], 0.0)
        return (h @ m["W2"].T + m["b2"]) * (m["hi"] - m["lo"]) + m["lo"]

    def level(self, c, low, high, days):
        """Ensemble-mean consumption (n_days, 48) in kWh for schedules low/high (n_days, 48) on days `days`."""
        low, high = np.asarray(low, float), np.asarray(high, float)
        conds = {m["cols"]: None for m in self.models[c]}
        conds = {k: self.cond_matrix(k, low, high, days) for k in conds}
        acc = np.zeros((len(days), H))
        for m in self.models[c]:
            for e in self.eps:
                acc += self.decode(m, np.repeat(e[None, : m["d"]], len(days), 0), conds[m["cols"]])
        return np.maximum(acc / (len(self.models[c]) * len(self.eps)), 1e-3)

    def log_effect(self, c, low, high, days=None, timeline=False):
        """Per-day effect (the CVAE has no notion of a continuous timeline, so `timeline` is ignored)."""
        days = np.arange(len(low)) if days is None else np.asarray(days)
        key = (c, days.tobytes())
        if key not in self._normal_level:
            zero = np.zeros((len(days), H))
            self._normal_level[key] = np.log(self.level(c, zero, zero, days))
        return np.log(self.level(c, low, high, days)) - self._normal_level[key]
