"""Verifier v-collapse: does eta (the KL weight of paper eq. 7) cause the CVAE's posterior collapse and its too-small
sample variance? A causal test: retrain the CVAE with eta in {10, 1, 0.1, 0.01} and change nothing else.

Mirrors scripts/10_trackA_cvae.py (Track A: the authors' train_nmf / test_nmf feather files, read by absolute path),
scripts/30_trackB_models.py (Track B: data/train_ours / test_ours parquet) and lib/cvae_torch.py. Those were read as
text. Everything is re-implemented here; the only thing taken from lib/ is the tariff-scenario window definition
(lib.effects.LOW_WINDOW / HIGH_WINDOW, the project's single source of truth), and nothing from the upstream repo is
imported:
  * model: encoder Dense(150 -> 15, relu) -> Dense(15 -> 8) = [mean | logscale]; decoder Dense(106 -> 15, relu) ->
    Dense(15 -> 48); Keras glorot_uniform kernels, zero biases, float64;
  * loss per sample: sum_h (x_h - xhat_h)^2 + eta * KL(N(mean, exp(logscale)^2) || N(0, I)) (upstream compute_loss, L2);
  * Keras Adam (lr 1e-3, eps 1e-7 added to the uncorrected sqrt(v)), batch 1, training rows in file order (= ascending
    date), 50 epochs, one reparameterisation draw per step;
  * seeds: restart seed s -> SeedSequence([s, 20200731]).spawn(2) -> (Glorot init stream, noise stream), as in lib.
    Seeds 0-4 at every eta (paired design: the same init and noise streams at every eta);
  * consumption min-max scaled with the cluster's train min/max over all 48 columns; 102 conditions in upstream order;
  * generation: z = default_rng([20200801, cluster]).standard_normal((200, 92, 4)), yhat = decoder(z, c) rescaled to
    kWh; the same z for every model of a (track, cluster) and for the three tariff scenarios (common random numbers).
Only eta changes. At eta = 10 the retrained weights are compared with the stored restarts
outputs/track{A,B}/models/cvae_c<c>_s<ss>.pt (seeds 0-4); bitwise equality proves the settings are identical.

Per trained model (one row of eta_sensitivity.csv):
  * latent use: train KL per sample (nats, analytic, training days) and per dim; active units (variance over the
    training days of the posterior mean > 0.01); posterior SD; aggregate-posterior mean and variance per dim (does
    N(0, I) still match the aggregate posterior?); test KL; the authors' selection metric (compute_mse: posterior z,
    RNG seed 0) and its prior-z version;
  * 200 samples per test day: medians over the 92 test days of the root-mean RMSE of the sample mean, the energy
    score (eq. 14: N/2 split, Euclidean, kWh) and the variogram score (p = 0.5, ordered pairs); pooled spread/skill
    (sqrt(mean sample variance) * sqrt((N+1)/N) / RMS error of the sample mean), central-50% / 90% coverage, share of
    observations outside the sample range;
  * the same RMSE / spread-skill / coverage with z drawn from the training days' aggregate posterior instead of
    N(0, I) (a diagnostic of prior mismatch, not the paper's generator);
  * cluster 3 only: tariff scenarios (Normal / Low = slots 9-19 = 04:00-09:30 GMT / High = slots 39-44 = 19:00-22:00
    GMT, the paper's windows as verified in REPRO_SPEC section 7, same z), giving Low and High in-window % effects and
    the High pre-window (35-38) and post-window (45-47; only 3 slots remain in the day, column name kept as
    eff_high_post4_pct) % effects. Phases 1-2 used 9-18 / 39-43 (post 44-47).

Stages (--stage all, the default, runs selftest, stored, train, rescore, collect, figure; every stage is re-runnable):
  selftest  score functions vs brute-force loops; calibrated toy ensemble gives spread/skill ~1, coverage ~nominal
  stored    my encoder on all 400 stored eta = 10 restarts (both tracks): KL, active units, posterior SD; and my scores on
            the stored selected-model sample arrays -> stored_checks.json
  train     the grid (2 tracks x 4 clusters x 4 eta x 5 seeds = 160 models) in a pool of 4 workers x 1 torch thread,
            then an extension beyond the brief, eta {0.03, 0.003, 0.001} (120 models, same seeds; skip it with
            --main-grid-only); per-model cache runs/<track>_c<c>_eta<eta>_s<seed>.{json,pt}; finished models are
            skipped unless --force
  rescore   cluster-3 tariff-scenario columns (eff_*) of the cached models, recomputed from runs/*_c3_*.pt (loads the
            weights, never trains): a cached value made with the current windows is left as is; one made with the
            superseded windows (lib.effects.SUPERSEDED_WINDOWS) must be reproduced exactly from the weights before it is
            replaced by the current-window value -> runs/*_c3_*.json, eta_rescore_windows.json
  collect   eta_sensitivity.csv, eta_summary.json (medians over seeds; eta = 10 vs the stored restarts and their scores)
  figure    outputs/figures/sensitivity_eta.png

Usage (from PROJECT or REPRO):  python -P scripts/40_eta_sensitivity.py
                                [--stage all|selftest|stored|train|rescore|collect|figure] [--workers 4] [--force]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
if REPRO not in sys.path:
    sys.path.insert(0, REPRO)
from lib.effects import HIGH_WINDOW, LOW_WINDOW, SUPERSEDED_WINDOWS  # noqa: E402  (scenario windows only)

UPSTREAM_DATA = REPRO + "/data/authors"  # copy of the authors' repo data/*.feather (byte-identical; read as data only)
PATHS = {"A": (UPSTREAM_DATA + "/train_nmf.feather", UPSTREAM_DATA + "/test_nmf.feather"),
         "B": (REPRO + "/data/train_ours.parquet", REPRO + "/data/test_ours.parquet")}
STORED_DIR = {"A": REPRO + "/outputs/trackA", "B": REPRO + "/outputs/trackB"}
OUT = REPRO + "/outputs/verify/v-collapse"
RUNS = OUT + "/runs"
CSV_PATH = OUT + "/eta_sensitivity.csv"
SUMMARY_PATH = OUT + "/eta_summary.json"
STORED_CHECKS_PATH = OUT + "/stored_checks.json"
FIG_PATH = REPRO + "/outputs/figures/sensitivity_eta.png"
LOG_PATH = OUT + "/eta_run_log.txt"

TRACKS = ("A", "B")
CLUSTERS = (0, 1, 2, 3)
ETAS = (10.0, 1.0, 0.1, 0.01)                 # the specified grid
EXTRA_ETAS = (0.03, 0.003, 0.001)             # extension (beyond the brief): locate the transition, push below 0.01
ALL_ETAS = tuple(sorted(ETAS + EXTRA_ETAS, reverse=True))
SEEDS = (0, 1, 2, 3, 4)
N_STORED_RESTARTS = 50
D_LAT, UNITS, H = 4, 15, 48
EPOCHS = 50
LR = 1e-3
SEED_SALT = 20200731          # lib/cvae_torch._seed_pair
GEN_SEED = 20200801           # Track A / B generation: default_rng([GEN_SEED, cluster])
AGG_SEED = 777                # aggregate-posterior diagnostic draws: default_rng([GEN_SEED, cluster, AGG_SEED])
N_SAMPLES = 200
ACTIVE_THRESHOLD = 0.01
LOW_SLOTS = list(LOW_WINDOW)          # slots 9-19 = 04:00-09:30 GMT (lib.effects; REPRO_SPEC section 7)
HIGH_SLOTS = list(HIGH_WINDOW)        # slots 39-44 = 19:00-22:00 GMT
EFF_COLS = ("eff_low_in_pct", "eff_high_in_pct", "eff_high_pre4_pct", "eff_high_post4_pct")
DT = torch.float64

COLS_DATA = [f"consumption_{h:02d}" for h in range(H)]
COLS_COND = (["temp_smooth_n", "pca_temp_0", "pca_temp_1", "pca_temp_2", "pos_y", "work_day"]
             + [n for h in range(H) for n in (f"low_{h:02d}", f"high_{h:02d}")])
assert len(COLS_COND) == 102
PARAM_NAMES = ["enc_hidden.weight", "enc_hidden.bias", "enc_out.weight", "enc_out.bias",
               "dec_hidden.weight", "dec_hidden.bias", "dec_out.weight", "dec_out.bias"]
LAYERS = ((UNITS, H + len(COLS_COND)), (2 * D_LAT, UNITS), (UNITS, D_LAT + len(COLS_COND)), (H, UNITS))  # (out, in)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def jdump(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def eta_tag(eta: float) -> str:
    return f"{eta:g}"


def run_base(track: str, c: int, eta: float, seed: int, runs_dir: str = RUNS) -> str:
    return f"{runs_dir}/{track}_c{c}_eta{eta_tag(eta)}_s{seed}"


# ------------------------------------------------------------------------------------------------------------------
# data (one cluster of one track), exactly as the track scripts prepare it
# ------------------------------------------------------------------------------------------------------------------
_TABLES: dict = {}
_DATA: dict = {}


def _read(path: str) -> pd.DataFrame:
    if path not in _TABLES:
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_feather(path)
        df["date"] = pd.to_datetime(df["date"])
        df["cluster"] = df["cluster"].astype(int)
        _TABLES[path] = df
    return _TABLES[path]


def track_data(track: str, c: int) -> dict:
    if (track, c) in _DATA:
        return _DATA[(track, c)]
    tr, te = (_read(p) for p in PATHS[track])
    trc = tr.loc[tr["cluster"] == c]                                   # file order (ascending date), as upstream
    tec = te.loc[te["cluster"] == c].sort_values("date", kind="stable")
    assert trc["date"].is_monotonic_increasing and len(trc) == 273 and len(tec) == 92
    Ytr = np.array(trc[COLS_DATA].to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    Ctr = np.array(trc[COLS_COND].to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    Yte = np.array(tec[COLS_DATA].to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    Cte = np.array(tec[COLS_COND].to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    assert all(np.isfinite(a).all() for a in (Ytr, Ctr, Yte, Cte))
    cmin, cmax = float(Ytr.min()), float(Ytr.max())
    d = {"Ytr": Ytr, "Ctr": Ctr, "Yte": Yte, "Cte": Cte, "cmin": cmin, "cmax": cmax,
         "Ys_tr": (Ytr - cmin) / (cmax - cmin), "Ys_te": (Yte - cmin) / (cmax - cmin),
         "test_dates": tec["date"].dt.strftime("%Y-%m-%d").tolist()}
    _DATA[(track, c)] = d
    return d


def scenario_cond(C: np.ndarray, name: str, low_slots=None, high_slots=None) -> np.ndarray:
    """Only the 96 tariff dummies change. Normal: all 0; Low: low_09..19 = 1; High: high_39..44 = 1 (the current
    windows, LOW_SLOTS / HIGH_SLOTS, unless other slots are passed)."""
    low_slots = LOW_SLOTS if low_slots is None else list(low_slots)
    high_slots = HIGH_SLOTS if high_slots is None else list(high_slots)
    C = np.array(C, dtype=np.float64, copy=True)
    il = [COLS_COND.index(f"low_{h:02d}") for h in range(H)]
    ih = [COLS_COND.index(f"high_{h:02d}") for h in range(H)]
    C[:, il] = 0.0
    C[:, ih] = 0.0
    if name == "Low":
        C[:, [il[h] for h in low_slots]] = 1.0
    elif name == "High":
        C[:, [ih[h] for h in high_slots]] = 1.0
    elif name != "Normal":
        raise ValueError(name)
    return C


def tariff_effects_c3(P, d: dict, Z: np.ndarray, low_slots=None, high_slots=None) -> dict:
    """Cluster-3 scenario effects (% of the Normal scenario, same z): Low and High in-window, High pre-window (the
    4 slots before) and post-window (the slots after, at most 4: 45-47 for the current High window)."""
    low_slots = LOW_SLOTS if low_slots is None else list(low_slots)
    high_slots = HIGH_SLOTS if high_slots is None else list(high_slots)
    pre4 = [h for h in range(high_slots[0] - 4, high_slots[0]) if h >= 0]
    post4 = [h for h in range(high_slots[-1] + 1, high_slots[-1] + 5) if h < H]
    M = {name: generate(P, scenario_cond(d["Cte"], name, low_slots, high_slots), Z, d["cmin"], d["cmax"]).mean(0)
         for name in ("Normal", "Low", "High")}

    def pct(name, slots):
        return float(100.0 * (M[name] - M["Normal"])[:, slots].mean() / M["Normal"][:, slots].mean())
    return {"eff_low_in_pct": pct("Low", low_slots), "eff_high_in_pct": pct("High", high_slots),
            "eff_high_pre4_pct": pct("High", pre4), "eff_high_post4_pct": pct("High", post4)}


# ------------------------------------------------------------------------------------------------------------------
# model, Keras Adam, training (independent re-implementation)
# ------------------------------------------------------------------------------------------------------------------
def seed_pair(seed: int) -> tuple[int, int]:
    ch = np.random.SeedSequence([int(seed), SEED_SALT]).spawn(2)
    return tuple(int(x.generate_state(1, dtype=np.uint64)[0] & np.uint64(0x7FFFFFFFFFFFFFFF)) for x in ch)


def init_params(gen: torch.Generator) -> list[torch.Tensor]:
    """Keras glorot_uniform kernels U(-l, l), l = sqrt(6 / (fan_in + fan_out)); zero biases. Layer order: encoder
    hidden, encoder out, decoder hidden, decoder out (weights stored torch-style as (out, in))."""
    P = []
    for fan_out, fan_in in LAYERS:
        lim = math.sqrt(6.0 / (fan_in + fan_out))
        W = torch.rand((fan_out, fan_in), generator=gen, dtype=DT) * (2.0 * lim) - lim
        P += [W.requires_grad_(True), torch.zeros(fan_out, dtype=DT).requires_grad_(True)]
    return P


def encode(P, X: torch.Tensor, C: torch.Tensor):
    W1, b1, W2, b2 = P[:4]
    o = F.linear(torch.relu(F.linear(torch.cat((X, C), 1), W1, b1)), W2, b2)
    return o[:, :D_LAT], o[:, D_LAT:]


def decode(P, Z: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    V1, a1, V2, a2 = P[4:]
    return F.linear(torch.relu(F.linear(torch.cat((Z, C), 1), V1, a1)), V2, a2)


def step_loss(P, xc: torch.Tensor, c: torch.Tensor, eps: torch.Tensor, eta: float) -> torch.Tensor:
    """One sample: sum_h (x - xhat)^2 + eta * KL. xc = [x, c] (data only)."""
    W1, b1, W2, b2, V1, a1, V2, a2 = P
    o = F.linear(torch.relu(F.linear(xc, W1, b1)), W2, b2)
    mean = o[:, :D_LAT]
    logscale = o[:, D_LAT:]
    z = mean + torch.exp(logscale) * eps
    xhat = F.linear(torch.relu(F.linear(torch.cat((z, c), 1), V1, a1)), V2, a2)
    x = xc[:, :H]
    rec = (x - xhat).square().sum()
    kl = 0.5 * (torch.exp(2.0 * logscale) + mean.square() - 1.0 - 2.0 * logscale).sum()
    return rec + eta * kl


class KerasAdam:
    """tf.keras Adam (OptimizerV2 / ResourceApplyAdam): lr_t = lr sqrt(1 - b2^t) / (1 - b1^t);
    m = b1 m + (1 - b1) g; v = b2 v + (1 - b2) g^2; p -= lr_t m / (sqrt(v) + eps)."""

    def __init__(self, params, lr=LR, b1=0.9, b2=0.999, eps=1e-7):
        self.p = list(params)
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [torch.zeros_like(q) for q in self.p]
        self.v = [torch.zeros_like(q) for q in self.p]
        self.t = 0

    @torch.no_grad()
    def step(self, grads) -> None:
        self.t += 1
        lr_t = self.lr * math.sqrt(1.0 - self.b2 ** self.t) / (1.0 - self.b1 ** self.t)
        g = list(grads)
        torch._foreach_mul_(self.m, self.b1)
        torch._foreach_add_(self.m, g, alpha=1.0 - self.b1)
        torch._foreach_mul_(self.v, self.b2)
        torch._foreach_addcmul_(self.v, g, g, value=1.0 - self.b2)
        den = torch._foreach_sqrt(self.v)
        torch._foreach_add_(den, self.eps)
        torch._foreach_addcdiv_(self.p, self.m, den, value=-lr_t)


def train_model(Ys: np.ndarray, C: np.ndarray, seed: int, eta: float, epochs: int = EPOCHS):
    init_s, noise_s = seed_pair(seed)
    g_init = torch.Generator().manual_seed(init_s)
    g_noise = torch.Generator().manual_seed(noise_s)
    P = init_params(g_init)
    opt = KerasAdam(P)
    X = torch.from_numpy(np.ascontiguousarray(Ys))
    Ct = torch.from_numpy(np.ascontiguousarray(C))
    XC = torch.cat((X, Ct), 1)
    n = X.shape[0]
    xcs = [XC[i: i + 1] for i in range(n)]
    cs = [Ct[i: i + 1] for i in range(n)]
    hist = []
    for _ in range(epochs):
        E = torch.randn((n, D_LAT), generator=g_noise, dtype=DT)
        tot = 0.0
        for i in range(n):
            loss = step_loss(P, xcs[i], cs[i], E[i: i + 1], eta)
            opt.step(torch.autograd.grad(loss, P))
            tot += loss.item()
        hist.append(tot / n)
    return P, hist


@torch.no_grad()
def generate(P, C: np.ndarray, Z: np.ndarray, cmin: float, cmax: float) -> np.ndarray:
    """Decoder output in kWh, shape (N, n_days, 48) for latent values Z (N, n_days, d) and conditions C (n_days, 102)."""
    N, nd, d = Z.shape
    Zt = torch.from_numpy(np.ascontiguousarray(Z.reshape(-1, d), dtype=np.float64))
    Cr = torch.from_numpy(np.ascontiguousarray(C, dtype=np.float64)).repeat(N, 1)   # sample-major rows
    out = decode(P, Zt, Cr).numpy().reshape(N, nd, H)
    return out * (cmax - cmin) + cmin


# ------------------------------------------------------------------------------------------------------------------
# scores (own code; lib.scores is not imported)
# ------------------------------------------------------------------------------------------------------------------
def day_scores(S: np.ndarray, y: np.ndarray) -> dict:
    """Per test day: root-mean RMSE of the sample mean, energy score eq. 14 (N/2 split, Euclidean), variogram p = 0.5
    over ordered pairs (h, h'). S (N, D, H) kWh, y (D, H) kWh."""
    N, D, _ = S.shape
    m = S.mean(0)
    rmse = np.sqrt(((m - y) ** 2).mean(1))
    h = N // 2
    d_obs = np.sqrt(((S[:h] - y[None]) ** 2).sum(2))
    d_pair = np.sqrt(((S[:h] - S[h: 2 * h]) ** 2).sum(2))
    es = (2.0 / N) * d_obs.sum(0) - (1.0 / N) * d_pair.sum(0)
    vg = np.empty(D)
    for t in range(D):
        vy = np.sqrt(np.abs(y[t][:, None] - y[t][None, :]))
        st = S[:, t, :]
        vs = np.sqrt(np.abs(st[:, :, None] - st[:, None, :])).mean(0)
        vg[t] = ((vy - vs) ** 2).sum()
    return {"rmse": rmse, "es": es, "vg": vg}


def pooled_stats(S: np.ndarray, y: np.ndarray) -> dict:
    """Pooled over all (day, half-hour) cells: spread/skill with the sqrt((N+1)/N) ensemble correction, central-50/90%
    coverage (empirical quantiles, linear interpolation, closed interval), share outside the sample range."""
    N = S.shape[0]
    m = S.mean(0)
    var = S.var(0, ddof=1)
    rmse_pooled = math.sqrt(float(((m - y) ** 2).mean()))
    ss = math.sqrt(float(var.mean())) * math.sqrt((N + 1) / N) / rmse_pooled
    q = np.quantile(S, [0.05, 0.25, 0.75, 0.95], axis=0)
    return {"spread_skill": ss,
            "coverage50": float(((y >= q[1]) & (y <= q[2])).mean()),
            "coverage90": float(((y >= q[0]) & (y <= q[3])).mean()),
            "frac_outside_range": float(((y < S.min(0)) | (y > S.max(0))).mean()),
            "sample_sd_kwh_mean": float(np.sqrt(var).mean()),
            "rmse_pooled_kwh": rmse_pooled}


def stage_selftest(args=None) -> dict:
    rng = np.random.default_rng(0)
    N, D, Hh = 6, 3, 5
    S = rng.normal(size=(N, D, Hh))
    y = rng.normal(size=(D, Hh))
    r = day_scores(S, y)
    worst = 0.0
    for t in range(D):
        first = 2.0 / N * sum(math.sqrt(sum((S[i, t, k] - y[t, k]) ** 2 for k in range(Hh))) for i in range(N // 2))
        spread = 1.0 / N * sum(math.sqrt(sum((S[i, t, k] - S[N // 2 + i, t, k]) ** 2 for k in range(Hh)))
                               for i in range(N // 2))
        vg = sum((abs(y[t, a] - y[t, b]) ** 0.5 - sum(abs(S[i, t, a] - S[i, t, b]) ** 0.5 for i in range(N)) / N) ** 2
                 for a in range(Hh) for b in range(Hh))
        rm = math.sqrt(sum((sum(S[i, t, k] for i in range(N)) / N - y[t, k]) ** 2 for k in range(Hh)) / Hh)
        worst = max(worst, abs(r["es"][t] - (first - spread)), abs(r["vg"][t] - vg), abs(r["rmse"][t] - rm))
    assert worst < 1e-12, worst
    # calibrated toy ensemble: obs and members are exchangeable draws -> spread/skill ~ 1, coverage ~ nominal
    Nn, Dd = 200, 3000
    mu = rng.normal(size=(Dd, 2))
    sd = rng.uniform(0.5, 2.0, size=(Dd, 2))
    Sc = mu[None] + sd[None] * rng.normal(size=(Nn, Dd, 2))
    yc = mu + sd * rng.normal(size=(Dd, 2))
    p = pooled_stats(Sc, yc)
    assert abs(p["spread_skill"] - 1) < 0.04 and abs(p["coverage90"] - 0.9) < 0.02 and abs(p["coverage50"] - 0.5) < 0.03, p
    # a collapsed toy ensemble (spread 10x too small) -> spread/skill ~ 0.1
    Sd = mu[None] + 0.1 * sd[None] * rng.normal(size=(Nn, Dd, 2))
    pd_ = pooled_stats(Sd, yc)
    assert abs(pd_["spread_skill"] - 0.1) < 0.01, pd_
    out = {"brute_force_max_abs_diff": worst, "calibrated_toy": p, "collapsed_toy_spread_skill": pd_["spread_skill"]}
    log(f"[selftest] scores vs brute-force loops max |diff| {worst:.1e}; calibrated toy spread/skill "
        f"{p['spread_skill']:.3f}, cov90 {p['coverage90']:.3f}, cov50 {p['coverage50']:.3f}; 10x-too-narrow toy "
        f"spread/skill {pd_['spread_skill']:.3f}")
    return out


# ------------------------------------------------------------------------------------------------------------------
# per-model evaluation
# ------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def latent_stats(P, Xs: np.ndarray, C: np.ndarray) -> dict:
    X = torch.from_numpy(np.ascontiguousarray(Xs))
    Ct = torch.from_numpy(np.ascontiguousarray(C))
    mu, ls = encode(P, X, Ct)
    sd = torch.exp(ls)
    kl_dim = 0.5 * (sd.square() + mu.square() - 1.0 - 2.0 * ls)
    var_mu = mu.var(0, unbiased=True)
    return {"kl": float(kl_dim.sum(1).mean()), "kl_dim": kl_dim.mean(0).numpy(),
            "var_mu": var_mu.numpy(), "active_units": int((var_mu > ACTIVE_THRESHOLD).sum()),
            "post_sd": sd.mean(0).numpy(), "post_sd_min": float(sd.min()), "post_sd_max": float(sd.max()),
            "agg_mean": mu.mean(0).numpy(),
            "agg_var": (mu.var(0, unbiased=False) + sd.square().mean(0)).numpy(),
            "mu": mu.numpy(), "sd": sd.numpy()}


@torch.no_grad()
def compute_mse(P, Xs: np.ndarray, C: np.ndarray, posterior: bool) -> float:
    """Authors' compute_mse (posterior z) / compute_mse_test (prior z): per day 100 sqrt(mean_h (x - xhat)^2) on
    scaled data, averaged over days; one draw per day from torch.Generator seed 0 (as lib, common to all models)."""
    X = torch.from_numpy(np.ascontiguousarray(Xs))
    Ct = torch.from_numpy(np.ascontiguousarray(C))
    eps = torch.randn((X.shape[0], D_LAT), generator=torch.Generator().manual_seed(0), dtype=DT)
    if posterior:
        mu, ls = encode(P, X, Ct)
        Z = mu + torch.exp(ls) * eps
    else:
        Z = eps
    return float((100.0 * torch.sqrt((X - decode(P, Z, Ct)).square().mean(1))).mean())


def evaluate(P, d: dict, c: int) -> dict:
    tr = latent_stats(P, d["Ys_tr"], d["Ctr"])
    te = latent_stats(P, d["Ys_te"], d["Cte"])
    with torch.no_grad():
        X = torch.from_numpy(d["Ys_tr"])
        Ct = torch.from_numpy(d["Ctr"])
        eps0 = torch.randn((X.shape[0], D_LAT), generator=torch.Generator().manual_seed(0), dtype=DT)
        xh = decode(P, torch.from_numpy(tr["mu"]) + torch.from_numpy(tr["sd"]) * eps0, Ct)
        train_rec = float((X - xh).square().sum(1).mean())
    rec = {"train_kl": tr["kl"], "active_units": tr["active_units"], "post_sd_min": tr["post_sd_min"],
           "post_sd_max": tr["post_sd_max"], "train_rec_sse_scaled": train_rec, "test_kl": te["kl"],
           "test_mse_post": compute_mse(P, d["Ys_te"], d["Cte"], True),
           "test_mse_prior": compute_mse(P, d["Ys_te"], d["Cte"], False),
           "agg_post_max_abs_mean": float(np.abs(tr["agg_mean"]).max()),
           "agg_post_var_min": float(tr["agg_var"].min()), "agg_post_var_max": float(tr["agg_var"].max())}
    for k in range(D_LAT):
        rec[f"kl_dim{k}"] = float(tr["kl_dim"][k])
        rec[f"var_mu_dim{k}"] = float(tr["var_mu"][k])
        rec[f"post_sd_dim{k}"] = float(tr["post_sd"][k])
        rec[f"agg_mean_dim{k}"] = float(tr["agg_mean"][k])
        rec[f"agg_var_dim{k}"] = float(tr["agg_var"][k])
    # the paper's generator: z ~ N(0, I), fixed draws shared by every model of this (track, cluster)
    nd = d["Cte"].shape[0]
    Z = np.random.default_rng([GEN_SEED, c]).standard_normal((N_SAMPLES, nd, D_LAT))
    S = generate(P, d["Cte"], Z, d["cmin"], d["cmax"])
    ds = day_scores(S, d["Yte"])
    rec.update({"rmse_median_kwh": float(np.median(ds["rmse"])), "rmse_mean_kwh": float(ds["rmse"].mean()),
                "es_median": float(np.median(ds["es"])), "vg_median": float(np.median(ds["vg"]))})
    rec.update(pooled_stats(S, d["Yte"]))
    # diagnostic: z from the aggregate posterior of the training days (mixture of the 273 posteriors)
    rng = np.random.default_rng([GEN_SEED, c, AGG_SEED])
    j = rng.integers(0, tr["mu"].shape[0], size=(N_SAMPLES, nd))
    Zagg = tr["mu"][j] + tr["sd"][j] * rng.standard_normal((N_SAMPLES, nd, D_LAT))
    Sa = generate(P, d["Cte"], Zagg, d["cmin"], d["cmax"])
    pa = pooled_stats(Sa, d["Yte"])
    rec.update({"aggz_rmse_median_kwh": float(np.median(np.sqrt(((Sa.mean(0) - d["Yte"]) ** 2).mean(1)))),
                "aggz_spread_skill": pa["spread_skill"], "aggz_coverage90": pa["coverage90"],
                "aggz_coverage50": pa["coverage50"]})
    if c == 3:
        rec.update(tariff_effects_c3(P, d, Z))
    return rec


def compare_stored(P, hist, rec: dict, track: str, c: int, seed: int) -> dict:
    base = f"{STORED_DIR[track]}/models/cvae_c{c}_s{seed:02d}"
    if not (os.path.exists(base + ".pt") and os.path.exists(base + ".json")):
        return {"stored_available": False}
    ck = torch.load(base + ".pt", map_location="cpu", weights_only=True)
    sdict = ck["state_dict"]
    wdiff = max(float((P[k].detach() - sdict[n]).abs().max()) for k, n in enumerate(PARAM_NAMES))
    with open(base + ".json") as f:
        js = json.load(f)
    return {"stored_available": True, "stored_max_abs_weight_diff": wdiff,
            "stored_epoch_loss_identical": bool(js["epoch_loss"] == hist),
            "stored_meta_eta": float(ck["meta"]["eta"]), "stored_meta_epochs": int(ck["meta"]["epochs"]),
            "stored_test_mse_post": float(js["test_mse_post"]), "stored_train_kl_final": float(js["train_kl_final"]),
            "stored_conso_min_max_equal": bool(js["conso_min"] == rec["conso_min"] and js["conso_max"] == rec["conso_max"])}


def run_job(job: dict) -> dict:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t0 = time.perf_counter()
    track, c, eta, seed = job["track"], int(job["cluster"]), float(job["eta"]), int(job["seed"])
    epochs = int(job.get("epochs", EPOCHS))
    d = track_data(track, c)
    P, hist = train_model(d["Ys_tr"], d["Ctr"], seed, eta, epochs)
    t_train = time.perf_counter() - t0
    rec = {"track": track, "cluster": c, "eta": eta, "seed": seed, "epochs": epochs, "conso_min": d["cmin"],
           "conso_max": d["cmax"], "epoch_loss_last": hist[-1],
           "epoch_loss_rel_change_last5": (hist[-6] - hist[-1]) / hist[-6] if len(hist) >= 6 else None}
    rec.update(evaluate(P, d, c))
    if eta == 10.0 and epochs == EPOCHS:
        rec.update(compare_stored(P, hist, rec, track, c, seed))
    rec["runtime_train_s"] = t_train
    rec["runtime_total_s"] = time.perf_counter() - t0
    rec["epoch_loss"] = hist
    base = run_base(track, c, eta, seed, job.get("runs_dir", RUNS))
    os.makedirs(os.path.dirname(base), exist_ok=True)
    torch.save({"state_dict": {n: p.detach().clone() for n, p in zip(PARAM_NAMES, P)},
                "meta": {k: rec[k] for k in ("track", "cluster", "eta", "seed", "epochs", "conso_min", "conso_max")}},
               base + ".pt")
    jdump(rec, base + ".json")
    return rec


# ------------------------------------------------------------------------------------------------------------------
# stages
# ------------------------------------------------------------------------------------------------------------------
def stage_stored(args) -> dict:
    """My encoder on every stored eta = 10 restart (both tracks, seeds 0-49) and my scores on the stored selected-model
    sample arrays: an independent recomputation of the collapse and low-variance claims."""
    out = {"definition": "stored restarts outputs/track{A,B}/models/cvae_c<c>_s<ss>.pt (eta = 10) re-evaluated with this "
                         "script's own encoder on the training days; active unit = variance over training days of the "
                         "posterior mean > 0.01; selected-model samples = outputs/track{A,B}/cvae_samples_c<c>.npy scored "
                         "with this script's own score code", "tracks": {}}
    for track in TRACKS:
        rows = []
        for c in CLUSTERS:
            d = track_data(track, c)
            for s in range(N_STORED_RESTARTS):
                p = f"{STORED_DIR[track]}/models/cvae_c{c}_s{s:02d}.pt"
                if not os.path.exists(p):
                    continue
                ck = torch.load(p, map_location="cpu", weights_only=True)
                P = [ck["state_dict"][n] for n in PARAM_NAMES]
                st = latent_stats(P, d["Ys_tr"], d["Ctr"])
                rows.append({"cluster": c, "seed": s, "eta_meta": float(ck["meta"]["eta"]), "kl": st["kl"],
                             "active_units": st["active_units"], "max_var_mu": float(st["var_mu"].max()),
                             "post_sd_min": st["post_sd_min"], "post_sd_max": st["post_sd_max"],
                             "scale_matches": bool(ck["meta"]["conso_min"] == d["cmin"] and ck["meta"]["conso_max"] == d["cmax"])})
        df = pd.DataFrame(rows)
        sel = {}
        diag_path = f"{STORED_DIR[track]}/scores_summary.json"
        ref = None
        if os.path.exists(diag_path):
            with open(diag_path) as f:
                ref = json.load(f)["distribution_diagnostics"]["per_model_cluster"]
        for c in CLUSTERS:
            d = track_data(track, c)
            S = np.load(f"{STORED_DIR[track]}/cvae_samples_c{c}.npy")
            ps = pooled_stats(S, d["Yte"])
            ds = day_scores(S, d["Yte"])
            ck = torch.load(f"{STORED_DIR[track]}/models/cvae_selected_c{c}.pt", map_location="cpu", weights_only=True)
            st = latent_stats([ck["state_dict"][n] for n in PARAM_NAMES], d["Ys_tr"], d["Ctr"])
            e = {"selected_seed": int(ck["meta"]["seed"]), "train_kl": st["kl"], "active_units": st["active_units"],
                 "max_var_mu": float(st["var_mu"].max()), **ps, "rmse_median_kwh": float(np.median(ds["rmse"])),
                 "es_median": float(np.median(ds["es"])), "vg_median": float(np.median(ds["vg"]))}
            if ref is not None:
                r = ref[f"cvae_c{c}"]
                e["phase2_spread_skill"] = r["spread_skill"]["ratio_calibrated"]
                e["phase2_coverage90"] = r["coverage_central90"]
                e["phase2_coverage50"] = r["coverage_central50"]
            sel[str(c)] = e
        out["tracks"][track] = {
            "n_restarts": int(len(df)), "all_meta_eta_10": bool((df["eta_meta"] == 10.0).all()),
            "all_scales_match": bool(df["scale_matches"].all()),
            "train_kl_max": float(df["kl"].max()), "train_kl_median": float(df["kl"].median()),
            "n_restarts_with_active_unit": int((df["active_units"] > 0).sum()),
            "max_var_mu_over_restarts": float(df["max_var_mu"].max()),
            "post_sd_min": float(df["post_sd_min"].min()), "post_sd_max": float(df["post_sd_max"].max()),
            "selected": sel}
        t = out["tracks"][track]
        log(f"[stored] track {track}: {t['n_restarts']} restarts, train KL max {t['train_kl_max']:.2e} (median "
            f"{t['train_kl_median']:.1e}), restarts with an active unit {t['n_restarts_with_active_unit']}, max var of "
            f"posterior mean {t['max_var_mu_over_restarts']:.1e}, posterior SD {t['post_sd_min']:.4f}-{t['post_sd_max']:.4f}")
        for c in CLUSTERS:
            e = sel[str(c)]
            log(f"[stored]   {track} c{c} selected s{e['selected_seed']}: spread/skill {e['spread_skill']:.3f} "
                f"(phase 2 {e.get('phase2_spread_skill', float('nan')):.3f}), cov90 {e['coverage90']:.3f} "
                f"(phase 2 {e.get('phase2_coverage90', float('nan')):.3f}), cov50 {e['coverage50']:.3f}, RMSE "
                f"{e['rmse_median_kwh']:.4f}, KL {e['train_kl']:.1e}, active {e['active_units']}")
    jdump(out, STORED_CHECKS_PATH)
    return out


def grid_jobs(force: bool, runs_dir: str = RUNS, etas_groups=(ETAS, EXTRA_ETAS)) -> list[dict]:
    jobs = []
    # the specified grid first, then the extension; within each, interleave so that every eta progresses together
    for etas in etas_groups:
        for seed in SEEDS:
            for eta in etas:
                for track in TRACKS:
                    for c in CLUSTERS:
                        b = run_base(track, c, eta, seed, runs_dir)
                        if not force and os.path.exists(b + ".json") and os.path.exists(b + ".pt"):
                            continue
                        jobs.append({"track": track, "cluster": c, "eta": eta, "seed": seed, "runs_dir": runs_dir})
    return jobs


def run_etas(args) -> tuple:
    return ETAS if args.main_grid_only else ALL_ETAS


def stage_train(args) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(RUNS, exist_ok=True)
    jobs = grid_jobs(args.force, etas_groups=(ETAS,) if args.main_grid_only else (ETAS, EXTRA_ETAS))
    log(f"[train] {len(jobs)} models to train with {args.workers} workers x 1 torch thread")
    if not jobs:
        return
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs)):
            r = fut.result()
            log(f"[train] {k + 1:3d}/{len(jobs)} {r['track']} c{r['cluster']} eta {eta_tag(r['eta']):>4} s{r['seed']}: "
                f"KL {r['train_kl']:.2e} active {r['active_units']} | spread/skill {r['spread_skill']:.3f} cov90 "
                f"{r['coverage90']:.3f} RMSE {r['rmse_median_kwh']:.4f} ES {r['es_median']:.4f} VG {r['vg_median']:.2f}"
                + (f" | stored |dW| {r['stored_max_abs_weight_diff']:.1e}" if r.get("stored_available") else "")
                + f" | {r['runtime_train_s']:.0f}s, wall {time.perf_counter() - t0:.0f}s")
    log(f"[train] done in {time.perf_counter() - t0:.0f}s wall")


RESCORE_PATH = OUT + "/eta_rescore_windows.json"


@torch.no_grad()
def stage_rescore(args) -> dict:
    """Recompute the cluster-3 scenario columns of every cached model from its saved weights (no training). A cached
    value equal to the current-window value is kept; a value equal to the superseded-window value (reproduced from
    the weights to 1e-9 %) is replaced; anything else is an error."""
    out = {"definition": "cluster-3 tariff-scenario effects of the cached models (runs/*_c3_*.pt), recomputed from the "
                         "saved weights with the same z draws default_rng([20200801, 3]) as at training time",
           "current_windows": {"Low": LOW_SLOTS, "High": HIGH_SLOTS, "High_pre4": list(range(HIGH_SLOTS[0] - 4, HIGH_SLOTS[0])),
                               "High_post": [h for h in range(HIGH_SLOTS[-1] + 1, HIGH_SLOTS[-1] + 5) if h < H]},
           "superseded_windows": {"Low": SUPERSEDED_WINDOWS["Low"], "High": SUPERSEDED_WINDOWS["High"]},
           "models": {}}
    n_upd = n_cur = 0
    max_old_diff = 0.0
    for track in TRACKS:
        d = track_data(track, 3)
        Z = np.random.default_rng([GEN_SEED, 3]).standard_normal((N_SAMPLES, d["Cte"].shape[0], D_LAT))
        for eta in ALL_ETAS:
            for seed in SEEDS:
                base = run_base(track, 3, eta, seed)
                if not (os.path.exists(base + ".json") and os.path.exists(base + ".pt")):
                    continue
                with open(base + ".json") as f:
                    rec = json.load(f)
                ck = torch.load(base + ".pt", map_location="cpu", weights_only=True)
                if not (rec["conso_min"] == d["cmin"] and rec["conso_max"] == d["cmax"]):
                    raise RuntimeError(f"{base}: scaling differs from the current data")
                P = [ck["state_dict"][n] for n in PARAM_NAMES]
                new = tariff_effects_c3(P, d, Z)
                stored = {k: rec.get(k) for k in EFF_COLS}
                if all(stored[k] is not None and abs(stored[k] - new[k]) <= 1e-9 for k in EFF_COLS):
                    status = "current"
                    n_cur += 1
                else:
                    old = tariff_effects_c3(P, d, Z, SUPERSEDED_WINDOWS["Low"], SUPERSEDED_WINDOWS["High"])
                    diff = max(abs(stored[k] - old[k]) for k in EFF_COLS)
                    if not diff <= 1e-9:
                        raise RuntimeError(f"{base}: cached eff_* match neither the current nor the superseded windows "
                                           f"(max |diff| vs superseded {diff:.3e})")
                    max_old_diff = max(max_old_diff, diff)
                    rec.update(new)
                    jdump(rec, base + ".json")
                    status = "updated"
                    n_upd += 1
                out["models"][os.path.basename(base)] = {"status": status, "previous": stored, "current": new}
    out["n_models"] = len(out["models"])
    out["n_updated_from_superseded_windows"] = n_upd
    out["n_already_current"] = n_cur
    out["max_abs_diff_cached_vs_superseded_window_recomputation_pct"] = max_old_diff
    out["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["archive_of_superseded_values"] = ("outputs/archive_windows_9-18_39-43/verify/v-collapse/runs/*_c3_*.json "
                                           "(the run records before the re-score)")
    if os.path.exists(RESCORE_PATH):          # keep the record of earlier re-scores (e.g. the one that updated)
        with open(RESCORE_PATH) as f:
            prev = json.load(f)
        hist = list(prev.get("history", []))
        prev_models = prev.get("models", {})
        hist.append({k: prev.get(k) for k in ("date", "n_models", "n_updated_from_superseded_windows",
                                              "n_already_current",
                                              "max_abs_diff_cached_vs_superseded_window_recomputation_pct")}
                    | {"superseded_values_replaced": {k: v["previous"] for k, v in prev_models.items()
                                                      if v.get("status") == "updated"}})
        out["history"] = hist
    jdump(out, RESCORE_PATH)
    log(f"[rescore] {out['n_models']} cluster-3 models: {n_upd} updated from the superseded windows (cached values "
        f"reproduced from the weights to {max_old_diff:.1e} %), {n_cur} already current")
    return out


SUMMARY_COLS = ["train_kl", "active_units", "post_sd_min", "post_sd_max", "train_rec_sse_scaled", "test_kl",
                "test_mse_post", "test_mse_prior", "agg_post_max_abs_mean", "agg_post_var_min", "agg_post_var_max",
                "rmse_median_kwh", "es_median", "vg_median", "spread_skill", "coverage50", "coverage90",
                "frac_outside_range", "sample_sd_kwh_mean", "rmse_pooled_kwh", "aggz_rmse_median_kwh",
                "aggz_spread_skill", "aggz_coverage90", "aggz_coverage50", "epoch_loss_last",
                "epoch_loss_rel_change_last5", "eff_low_in_pct", "eff_high_in_pct", "eff_high_pre4_pct",
                "eff_high_post4_pct", "runtime_train_s"]
RANGE_COLS = ["train_kl", "active_units", "spread_skill", "coverage90", "coverage50", "rmse_median_kwh", "es_median",
              "vg_median", "eff_low_in_pct", "eff_high_in_pct", "eff_high_pre4_pct", "eff_high_post4_pct"]


def stage_collect(args) -> dict:
    etas_run = run_etas(args)
    rows, missing = [], []
    for track in TRACKS:
        for c in CLUSTERS:
            for eta in etas_run:
                for seed in SEEDS:
                    p = run_base(track, c, eta, seed) + ".json"
                    if not os.path.exists(p):
                        missing.append((track, c, eta, seed))
                        continue
                    with open(p) as f:
                        r = json.load(f)
                    r.pop("epoch_loss", None)
                    rows.append(r)
    if missing:
        log(f"[collect] WARNING {len(missing)} models missing, e.g. {missing[:5]}")
    df = pd.DataFrame(rows)
    df.insert(4, "in_specified_grid", df["eta"].isin(ETAS))
    df["eta_order"] = df["eta"].map({e: i for i, e in enumerate(ALL_ETAS)})
    df = df.sort_values(["track", "cluster", "eta_order", "seed"]).drop(columns="eta_order").reset_index(drop=True)
    # paired (same seed) relative changes against eta = 10
    for col in ("rmse_median_kwh", "es_median", "vg_median", "spread_skill", "coverage90"):
        base = df.loc[df["eta"] == 10.0].set_index(["track", "cluster", "seed"])[col]
        key = pd.MultiIndex.from_frame(df[["track", "cluster", "seed"]])
        df[f"{col}_rel_change_vs_eta10_pct"] = 100.0 * (df[col].to_numpy() / base.reindex(key).to_numpy() - 1.0)
    # phase-2 per-restart scores of the stored eta = 10 restarts (same seeds) for a direct comparison
    sr_path = STORED_DIR["B"] + "/scores_restarts_AB.csv"
    if os.path.exists(sr_path):
        sr = pd.read_csv(sr_path)
        sr = sr.rename(columns={"spread_skill": "phase2_spread_skill", "coverage90": "phase2_coverage90",
                                "rmse_median": "phase2_rmse_median", "es_split_median": "phase2_es_median",
                                "vg_ord_median": "phase2_vg_median"})
        keep = ["track", "cluster", "seed", "phase2_spread_skill", "phase2_coverage90", "phase2_rmse_median",
                "phase2_es_median", "phase2_vg_median"]
        sr = sr[keep].assign(eta=10.0)
        df = df.merge(sr, on=["track", "cluster", "seed", "eta"], how="left")
    df.to_csv(CSV_PATH, index=False)
    log(f"[collect] {len(df)} rows -> {CSV_PATH}")

    summ = {"definition": {
        "grid": "tracks A (authors' data) and B (our data) x clusters 0-3 x eta {10, 1, 0.1, 0.01} x seeds 0-4; "
                "everything else as scripts/10_trackA_cvae.py and 30_trackB_models.py. Extension beyond the brief "
                "(column in_specified_grid = False): eta {0.03, 0.003, 0.001}, same seeds and settings",
        "train_kl": "mean over the 273 training days of the analytic KL(q(z|x,c) || N(0, I)), nats",
        "active_units": "latent dims whose posterior mean has variance > 0.01 across training days",
        "agg_post": "aggregate posterior over training days: mean of posterior means; variance = Var(mu) + mean(sd^2) "
                    "per dim (N(0, I) prior: 0 and 1)",
        "rmse_median_kwh / es_median / vg_median": "medians over the 92 test days; 200 samples per day with the fixed "
                                                    "z draws default_rng([20200801, cluster]); RMSE root-mean over 48 "
                                                    "half-hours; ES eq. 14 (N/2 split, Euclidean, kWh); VG p = 0.5, "
                                                    "ordered pairs",
        "spread_skill": "pooled over 92 x 48 cells: sqrt(mean sample variance, ddof 1) x sqrt((N+1)/N) / RMS error "
                        "of the sample mean (1 = calibrated)",
        "coverage50 / coverage90": "share of (day, half-hour) observations inside the central 50% / 90% sample interval",
        "aggz_*": "diagnostic: z drawn from the training days' aggregate posterior instead of N(0, I)",
        "eff_*_pct (cluster 3)": "100 x mean over test days and window slots of (scenario - Normal) / mean of Normal; "
                                 f"Low = low slots {LOW_SLOTS[0]}-{LOW_SLOTS[-1]} (04:00-09:30 GMT), High = high slots "
                                 f"{HIGH_SLOTS[0]}-{HIGH_SLOTS[-1]} (19:00-22:00 GMT) (lib.effects, REPRO_SPEC section "
                                 "7); pre4 = slots 35-38, post4 = slots 45-47 (only 3 slots remain in the day; column "
                                 "name kept); same z. Phases 1-2 used 9-18 / 39-43 / post 44-47 (stage rescore)",
        "*_rel_change_vs_eta10_pct": "paired by seed (same init and noise streams), 100 x (value / value at eta 10 - 1)",
        "medians": "median over the 5 seeds; ranges = [min, max] over seeds"},
        "n_models": int(len(df)), "missing": [list(m) for m in missing], "groups": {}}
    for (track, c, eta), g in df.groupby(["track", "cluster", "eta"], sort=False):
        e = {col: float(g[col].median()) for col in SUMMARY_COLS if col in g and g[col].notna().any()}
        for col in ("rmse_median_kwh", "es_median", "vg_median", "spread_skill", "coverage90"):
            k = f"{col}_rel_change_vs_eta10_pct"
            e[k] = float(g[k].median())
        e["ranges"] = {col: [float(g[col].min()), float(g[col].max())] for col in RANGE_COLS
                       if col in g and g[col].notna().any()}
        e["n_seeds"] = int(len(g))
        summ["groups"][f"{track}|c{c}|eta{eta_tag(eta)}"] = e
    # eta = 10 retrains vs stored restarts
    s10 = df.loc[df["eta"] == 10.0]
    chk = {"n": int(len(s10))}
    if "stored_max_abs_weight_diff" in s10:
        chk.update({"max_abs_weight_diff": float(s10["stored_max_abs_weight_diff"].max()),
                    "n_bitwise_identical": int((s10["stored_max_abs_weight_diff"] == 0).sum()),
                    "n_epoch_loss_identical": int(s10["stored_epoch_loss_identical"].sum()),
                    "max_abs_test_mse_post_diff": float((s10["test_mse_post"] - s10["stored_test_mse_post"]).abs().max())})
    if "phase2_spread_skill" in s10:
        for col, ref in (("spread_skill", "phase2_spread_skill"), ("coverage90", "phase2_coverage90"),
                         ("rmse_median_kwh", "phase2_rmse_median"), ("es_median", "phase2_es_median"),
                         ("vg_median", "phase2_vg_median")):
            chk[f"max_abs_diff_{col}_vs_phase2_restart_scores"] = float((s10[col] - s10[ref]).abs().max())
    summ["eta10_vs_stored_restarts"] = chk
    # compact verdict inputs per track and eta: ranges over clusters of the per-cluster medians
    vi = {}
    for track in TRACKS:
        for eta in etas_run:
            gs = [summ["groups"][f"{track}|c{c}|eta{eta_tag(eta)}"] for c in CLUSTERS
                  if f"{track}|c{c}|eta{eta_tag(eta)}" in summ["groups"]]
            if not gs:
                continue
            vi[f"{track}|eta{eta_tag(eta)}"] = {
                k: [min(x[k] for x in gs), max(x[k] for x in gs)]
                for k in ("train_kl", "active_units", "spread_skill", "coverage90", "coverage50", "rmse_median_kwh",
                          "rmse_median_kwh_rel_change_vs_eta10_pct", "es_median", "es_median_rel_change_vs_eta10_pct",
                          "vg_median", "vg_median_rel_change_vs_eta10_pct", "aggz_spread_skill", "aggz_coverage90",
                          "agg_post_var_min", "agg_post_var_max", "test_mse_post", "test_mse_prior")}
    summ["by_track_eta_range_over_clusters"] = vi
    c3 = {}
    for track in TRACKS:
        for eta in etas_run:
            k = f"{track}|c3|eta{eta_tag(eta)}"
            if k in summ["groups"]:
                g = summ["groups"][k]
                c3[f"{track}|eta{eta_tag(eta)}"] = {x: g.get(x) for x in ("eff_low_in_pct", "eff_high_in_pct",
                                                                          "eff_high_pre4_pct", "eff_high_post4_pct")}
                c3[f"{track}|eta{eta_tag(eta)}"]["ranges"] = {x: g["ranges"].get(x) for x in (
                    "eff_low_in_pct", "eff_high_in_pct", "eff_high_pre4_pct", "eff_high_post4_pct")}
    summ["cluster3_scenario_effects"] = c3
    if os.path.exists(STORED_CHECKS_PATH):
        with open(STORED_CHECKS_PATH) as f:
            summ["stored_checks"] = json.load(f)
    jdump(summ, SUMMARY_PATH)
    for k, v in vi.items():
        log(f"[collect] {k}: KL {v['train_kl'][0]:.1e}-{v['train_kl'][1]:.1e}, active {v['active_units'][0]:.0f}-"
            f"{v['active_units'][1]:.0f}, spread/skill {v['spread_skill'][0]:.3f}-{v['spread_skill'][1]:.3f}, cov90 "
            f"{v['coverage90'][0]:.3f}-{v['coverage90'][1]:.3f}, RMSE {v['rmse_median_kwh'][0]:.4f}-"
            f"{v['rmse_median_kwh'][1]:.4f} ({v['rmse_median_kwh_rel_change_vs_eta10_pct'][0]:+.1f} to "
            f"{v['rmse_median_kwh_rel_change_vs_eta10_pct'][1]:+.1f}% vs eta 10), ES change "
            f"{v['es_median_rel_change_vs_eta10_pct'][0]:+.1f} to {v['es_median_rel_change_vs_eta10_pct'][1]:+.1f}%, "
            f"agg-post var {v['agg_post_var_min'][0]:.2f}-{v['agg_post_var_max'][1]:.2f}")
    log(f"[collect] eta 10 vs stored restarts: {chk}")
    return summ


# ------------------------------------------------------------------------------------------------------------------
# figure
# ------------------------------------------------------------------------------------------------------------------
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Okabe-Ito blue / orange / bluish green / reddish purple: passes the dataviz validator on ALL pairs (light surface:
# worst CVD dE 7.6, normal-vision dE 18.7), with a distinct marker per cluster as the secondary encoding
CLUSTER_COLOR = {0: "#0072B2", 1: "#E69F00", 2: "#009E73", 3: "#CC79A7"}
CLUSTER_MARKER = {0: "o", 1: "s", 2: "^", 3: "D"}
TRACK_LS = {"A": "-", "B": (0, (4.5, 2.5))}
TRACK_LABEL = {"A": "Track A (authors' data): solid, filled", "B": "Track B (our data): dashed, hollow"}
TRACK_XOFF = {"A": 0.94, "B": 1.065}


def stage_figure(args) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.frameon": False, "legend.fontsize": 8.5,
        "axes.titlelocation": "left"})
    df = pd.read_csv(CSV_PATH)
    panels = (("spread_skill", "(a) Spread / skill of the 200 samples", "Spread / skill (1 = calibrated)", 1.0,
               "calibrated"),
              ("coverage90", "(b) Coverage of the central 90% sample interval", "Share of observations covered", 0.9,
               "nominal 90%"),
              ("rmse_median_kwh", "(c) RMSE of the sample mean", "kWh per household per half-hour\n(median over 92 test days)",
               None, None))
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5))
    for ax, (col, title, ylab, ref, reflab) in zip(axes, panels):
        for track in TRACKS:
            for c in CLUSTERS:
                g = df.loc[(df["track"] == track) & (df["cluster"] == c)].groupby("eta")[col]
                med, lo, hi = g.median(), g.min(), g.max()
                x = med.index.to_numpy() * TRACK_XOFF[track]
                ax.vlines(x, lo.to_numpy(), hi.to_numpy(), color=CLUSTER_COLOR[c], lw=1.0, alpha=0.5, zorder=2)
                ax.plot(x, med.to_numpy(), color=CLUSTER_COLOR[c], ls=TRACK_LS[track], lw=1.6, zorder=3,
                        marker=CLUSTER_MARKER[c], ms=6.5, mew=1.3, mec=CLUSTER_COLOR[c],
                        mfc=CLUSTER_COLOR[c] if track == "A" else SURFACE)
        if ref is not None:
            ax.axhline(ref, color=INK2, lw=1.0, ls=(0, (1.5, 2.0)), zorder=1)
            ax.text(0.985, ref, reflab, transform=ax.get_yaxis_transform(), ha="right", va="bottom", color=INK2,
                    fontsize=8)
        etas_present = sorted(float(e) for e in df["eta"].unique())
        ax.set_xscale("log")
        ax.set_xticks(etas_present, [eta_tag(e) for e in etas_present], fontsize=8)
        ax.minorticks_off()
        ax.set_xlim(min(etas_present) * 0.7, 14.5)
        ax.set_xlabel("KL weight η (log scale; paper and reproduction: η = 10)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        if col in ("spread_skill", "coverage90"):
            ax.set_ylim(0.0, max(1.08, float(np.nanmax(df[col])) * 1.06))
    handles = [Line2D([], [], color=CLUSTER_COLOR[c], marker=CLUSTER_MARKER[c], ms=6.5, lw=1.6, label=f"Cluster {c}")
               for c in CLUSTERS]
    handles += [Line2D([], [], color=INK2, ls=TRACK_LS[t], lw=1.6, marker="o", ms=6, mec=INK2,
                       mfc=INK2 if t == "A" else SURFACE, label=TRACK_LABEL[t]) for t in TRACKS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1.035), ncol=6, handlelength=2.6,
               columnspacing=1.4)
    extra = sorted(set(float(e) for e in df["eta"].unique()) - set(ETAS))
    sub = ("Lines = median over 5 restarts (seeds 0-4), whiskers = min-max across them; 200 samples per test day, same "
           "z draws for every model." + (f" η = {', '.join(eta_tag(e) for e in extra)} extend the specified grid "
                                         f"{{10, 1, 0.1, 0.01}}." if extra else ""))
    fig.text(0.01, 1.105, "CVAE sample spread and accuracy vs the KL weight η (everything else as in the reproduction)",
             ha="left", va="bottom", fontsize=10.5, color=INK)
    fig.text(0.01, 1.065, sub, ha="left", va="bottom", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log(f"[figure] wrote {FIG_PATH}")
    return FIG_PATH


# ------------------------------------------------------------------------------------------------------------------
def stage_smoke(args) -> None:
    """Quick end-to-end check into a scratch folder (not part of 'all'): one full eta = 10 model (compared with the
    stored restart) and a 3-epoch eta = 0.01 model."""
    sp = args.smoke_dir
    os.makedirs(sp, exist_ok=True)
    for job in ({"track": "A", "cluster": 3, "eta": 10.0, "seed": 0, "runs_dir": sp},
                {"track": "B", "cluster": 3, "eta": 0.01, "seed": 0, "runs_dir": sp, "epochs": 3}):
        r = run_job(job)
        log(f"[smoke] {job}: " + ", ".join(f"{k}={r[k]}" for k in (
            "train_kl", "active_units", "spread_skill", "coverage90", "rmse_median_kwh", "es_median", "vg_median",
            "test_mse_post", "stored_max_abs_weight_diff", "stored_epoch_loss_identical", "stored_test_mse_post",
            "eff_low_in_pct", "eff_high_in_pct", "runtime_train_s") if k in r))


STAGES = ("selftest", "stored", "train", "rescore", "collect", "figure")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", help="all, smoke, or a comma list of: " + ",".join(STAGES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="retrain models that already have results")
    ap.add_argument("--main-grid-only", action="store_true",
                    help="only the specified grid eta {10, 1, 0.1, 0.01} (skip the extension 0.03, 0.003, 0.001)")
    ap.add_argument("--smoke-dir", default=("C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/"
                                            "1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-collapse/smoke"))
    args = ap.parse_args()
    if args.workers > 4:
        raise SystemExit("max 4 workers (shared machine)")
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    stages = STAGES if args.stage == "all" else tuple(s.strip() for s in args.stage.split(","))
    log(f"=== 40_eta_sensitivity.py stages={','.join(stages)} workers={args.workers} force={args.force}")
    fns = {"selftest": stage_selftest, "stored": stage_stored, "train": stage_train, "rescore": stage_rescore,
           "collect": stage_collect, "figure": stage_figure, "smoke": stage_smoke}
    bad = [s for s in stages if s not in fns]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {tuple(fns)}")
    for s in stages:
        t0 = time.perf_counter()
        fns[s](args)
        log(f"--- stage {s} done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
