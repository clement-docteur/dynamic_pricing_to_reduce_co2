"""PyTorch (float64) port of the CVAE of Bregere & Bessa (2020), IEEE Access 8:131949.

Upstream reference (read as TEXT only, never imported or executed; commit 41157dd):
  codes/cvae.py                      model, loss, compute_mse, compute_mse_test, simulate_cvae
  codes/cvae_generate.ipynb          hyper-parameters, training loop (50 restarts per cluster)
  codes/cvae_estimate.ipynb          generation of 200 samples per test day
  codes/optimize_architecture.ipynb  evaluation of train/test MSE (selection metric)

What is ported faithfully
  * Encoder  Dense(48+102 -> 15, relu) -> Dense(15 -> 2d, linear); output split as [mean | logscale],
    scale = exp(logscale) is the standard deviation, z = mean + exp(logscale) * eps.
  * Decoder  Dense(d+102 -> 15, relu) -> Dense(15 -> 48, linear). Inputs are concatenated [x, c] / [z, c].
  * Loss per sample (loss='L2'): eta * KL(N(mean, diag(scale^2)) || N(0, I)) + sum_h (x_h - xhat_h)^2, eta = 10.
  * Glorot-uniform kernels (limit sqrt(6/(fan_in+fan_out))), zero biases (Keras Dense defaults), float64.
  * Adam exactly as tf.keras.optimizers.Adam (TF2 OptimizerV2): lr 1e-3, beta1 0.9, beta2 0.999, eps 1e-7,
    eps added to the *uncorrected* sqrt(v) (torch.optim.Adam adds it after bias correction).
  * Batch size 1, rows visited in the given (fixed) order, no shuffling, 50 epochs, one fresh reparameterisation
    draw per step.
  * Consumption min-max scaled with the train min/max over all 48 columns of the cluster.
  * compute_mse: per day 100*sqrt(mean_h (x-xhat)^2) on scaled data with z ~ encoder posterior, averaged over days.
    compute_mse_test: same with z ~ N(0, I).
  * Generation: z ~ N(0, I_d), yhat = decoder([z, c]) * (max - min) + min.

Deliberate deviations (see also the report)
  * Each restart gets its own seed (Glorot init + reparameterisation noise). Upstream reuses
    glorot_uniform(seed=0) for every restart and draws tf.random.normal without a seed.
  * Each restart gets a fresh Adam. Upstream shares ONE Adam instance across all restarts and clusters, so the
    Keras iteration counter keeps growing and later restarts start with (almost) no bias correction (first steps
    ~3x larger). `adam_iterations0` reproduces that quirk if wanted.
  * Evaluation metrics use a fixed RNG (common random numbers across restarts) so that selection is reproducible.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float64
INPUT_DIM = 48
LATENT_DIM = 4
UNITS = 15
KL_COEFF = 10.0
EPOCHS = 50
LR = 1e-3

COLS_DATA = [f"consumption_{i:02d}" for i in range(48)]
LOW_COLS = [f"low_{i:02d}" for i in range(48)]
HIGH_COLS = [f"high_{i:02d}" for i in range(48)]
COLS_PRICE = [name for i in range(48) for name in (f"low_{i:02d}", f"high_{i:02d}")]
# exact upstream order: cols_temp + ['pos_y', 'work_day'] + cols_price (low_00, high_00, low_01, ...)
COLS_COND = ["temp_smooth_n", "pca_temp_0", "pca_temp_1", "pca_temp_2", "pos_y", "work_day"] + COLS_PRICE
assert len(COLS_COND) == 102


# ----------------------------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------------------------
def _seed_pair(seed: int) -> tuple[int, int]:
    """Two independent 63-bit seeds (init, noise) derived from one restart seed."""
    children = np.random.SeedSequence([int(seed), 20200731]).spawn(2)
    return tuple(int(ch.generate_state(1, dtype=np.uint64)[0] & np.uint64(0x7FFFFFFFFFFFFFFF)) for ch in children)


def glorot_uniform_(w: torch.Tensor, gen: torch.Generator) -> None:
    """Keras glorot_uniform. torch weight is (out, in), Keras kernel (in, out): the limit is symmetric."""
    fan_out, fan_in = w.shape
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        w.copy_(torch.rand(tuple(w.shape), generator=gen, dtype=w.dtype) * (2.0 * limit) - limit)


def kl_diag_std_normal(mean: torch.Tensor, logscale: torch.Tensor) -> torch.Tensor:
    """KL(N(mean, diag(exp(logscale)^2)) || N(0, I)) per row (tfp MultivariateNormalDiag, scale_diag=exp(logscale))."""
    return 0.5 * (torch.exp(2.0 * logscale) + mean.square() - 1.0 - 2.0 * logscale).sum(1)


class CVAE(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, cond_dim: int = len(COLS_COND), latent_dim: int = LATENT_DIM,
                 units: int = UNITS):
        super().__init__()
        self.input_dim, self.cond_dim, self.latent_dim, self.units = input_dim, cond_dim, latent_dim, units
        self.enc_hidden = nn.Linear(input_dim + cond_dim, units, dtype=DTYPE)
        self.enc_out = nn.Linear(units, 2 * latent_dim, dtype=DTYPE)
        self.dec_hidden = nn.Linear(latent_dim + cond_dim, units, dtype=DTYPE)
        self.dec_out = nn.Linear(units, input_dim, dtype=DTYPE)
        self.meta: dict = {}

    def keras_init(self, gen: torch.Generator) -> None:
        for lin in (self.enc_hidden, self.enc_out, self.dec_hidden, self.dec_out):
            glorot_uniform_(lin.weight, gen)
            nn.init.zeros_(lin.bias)

    def params_list(self) -> list[torch.Tensor]:
        return [self.enc_hidden.weight, self.enc_hidden.bias, self.enc_out.weight, self.enc_out.bias,
                self.dec_hidden.weight, self.dec_hidden.bias, self.dec_out.weight, self.dec_out.bias]

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.enc_out(torch.relu(self.enc_hidden(torch.cat((x, c), 1))))
        return o[:, : self.latent_dim], o[:, self.latent_dim:]

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.dec_out(torch.relu(self.dec_hidden(torch.cat((z, c), 1))))

    def loss_terms(self, x, c, eps, eta: float = KL_COEFF):
        """Per-row (loss, reconstruction, kl) exactly as upstream compute_loss with loss='L2'."""
        mean, logscale = self.encode(x, c)
        z = mean + torch.exp(logscale) * eps
        xhat = self.decode(z, c)
        rec = (x - xhat).square().sum(1)
        kl = kl_diag_std_normal(mean, logscale)
        return eta * kl + rec, rec, kl

    # scaling helpers (metadata set by train_cvae / load_model)
    def scale(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float64) - self.meta["conso_min"]) / (self.meta["conso_max"] - self.meta["conso_min"])

    def unscale(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64) * (self.meta["conso_max"] - self.meta["conso_min"]) + self.meta["conso_min"]


def _step_loss(P, xc, c, eps, eta: float, d: int) -> torch.Tensor:
    """Functional single-sample loss used in the training loop (same maths as CVAE.loss_terms, fewer Python calls).
    xc = precomputed concat([x, c]) row (data only, so precomputing it does not change the gradient)."""
    W1, b1, W2, b2, V1, a1, V2, a2 = P
    o = F.linear(torch.relu(F.linear(xc, W1, b1)), W2, b2)
    mean = o[:, :d]
    logscale = o[:, d:]
    z = mean + torch.exp(logscale) * eps
    xhat = F.linear(torch.relu(F.linear(torch.cat((z, c), 1), V1, a1)), V2, a2)
    x = xc[:, : V2.shape[0]]
    rec = (x - xhat).square().sum()
    kl = 0.5 * (torch.exp(2.0 * logscale) + mean.square() - 1.0 - 2.0 * logscale).sum()
    return rec + eta * kl


class KerasAdam:
    """tf.keras.optimizers.Adam (TF 2.x OptimizerV2, amsgrad=False), i.e. the ResourceApplyAdam kernel:
        t = iterations + 1; lr_t = lr * sqrt(1 - b2^t) / (1 - b1^t)
        m = b1 m + (1 - b1) g;  v = b2 v + (1 - b2) g^2;  p -= lr_t * m / (sqrt(v) + eps)
    `iterations` can start > 0 to mimic the upstream quirk of one optimizer shared across restarts."""

    def __init__(self, params, lr: float = LR, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-7,
                 iterations: int = 0):
        self.params = list(params)
        self.lr, self.beta_1, self.beta_2, self.epsilon = lr, beta_1, beta_2, epsilon
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.iterations = int(iterations)

    @torch.no_grad()
    def step(self, grads) -> None:
        self.iterations += 1
        t = self.iterations
        lr_t = self.lr * math.sqrt(1.0 - self.beta_2 ** t) / (1.0 - self.beta_1 ** t)
        grads = list(grads)
        torch._foreach_mul_(self.m, self.beta_1)
        torch._foreach_add_(self.m, grads, alpha=1.0 - self.beta_1)
        torch._foreach_mul_(self.v, self.beta_2)
        torch._foreach_addcmul_(self.v, grads, grads, value=1.0 - self.beta_2)
        denom = torch._foreach_sqrt(self.v)
        torch._foreach_add_(denom, self.epsilon)
        torch._foreach_addcdiv_(self.params, self.m, denom, value=-lr_t)


# ----------------------------------------------------------------------------------------------------------------
# training / evaluation / generation (data-agnostic: any DataFrame with COLS_DATA + cond columns)
# ----------------------------------------------------------------------------------------------------------------
def _xy(df, cond_cols):
    Y = np.array(df[COLS_DATA].to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    C = np.array(df[list(cond_cols)].to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    if not (np.isfinite(Y).all() and np.isfinite(C).all()):
        raise ValueError("non-finite values in consumption or conditions")
    return Y, C


def train_cvae(train_df, seed: int, epochs: int = EPOCHS, eta: float = KL_COEFF, lr: float = LR,
               latent_dim: int = LATENT_DIM, units: int = UNITS, cond_cols=COLS_COND, adam_iterations0: int = 0,
               init_seed: int | None = None, log=None):
    """Train one CVAE restart on the rows of `train_df` (ONE cluster), visited in the given row order.

    `seed` drives the Glorot init and the reparameterisation noise. `init_seed` (optional) overrides the init part
    only: restarts sharing init_seed start from identical weights (upstream quirk emulation: a seeded TF2
    initializer is stateless, so glorot_uniform(seed=0) gives the same weights at every restart).
    Returns (model, info). model.meta holds conso_min/max (train min/max over all 48 columns), cond_cols, seed."""
    t0 = time.perf_counter()
    Y, C = _xy(train_df, cond_cols)
    cmin, cmax = float(Y.min()), float(Y.max())
    Ys = (Y - cmin) / (cmax - cmin)
    n = Ys.shape[0]
    init_s, noise_seed = _seed_pair(seed)
    if init_seed is not None:
        init_s = _seed_pair(init_seed)[0]
    g_init = torch.Generator().manual_seed(init_s)
    g_noise = torch.Generator().manual_seed(noise_seed)
    model = CVAE(INPUT_DIM, C.shape[1], latent_dim, units)
    model.keras_init(g_init)
    P = model.params_list()
    opt = KerasAdam(P, lr=lr, iterations=adam_iterations0)
    X = torch.from_numpy(Ys)
    Ct = torch.from_numpy(C)
    XC = torch.cat((X, Ct), 1)
    xcs = [XC[i: i + 1] for i in range(n)]
    cs = [Ct[i: i + 1] for i in range(n)]
    epoch_loss = []
    for ep in range(epochs):
        E = torch.randn((n, latent_dim), generator=g_noise, dtype=DTYPE)
        tot = 0.0
        for i in range(n):
            loss = _step_loss(P, xcs[i], cs[i], E[i: i + 1], eta, latent_dim)
            grads = torch.autograd.grad(loss, P)
            opt.step(grads)
            tot += loss.item()
        epoch_loss.append(tot / n)
        if log is not None:
            log(f"epoch {ep + 1}/{epochs} mean step loss {tot / n:.6f}")
    model.meta = {"conso_min": cmin, "conso_max": cmax, "cond_cols": list(cond_cols), "seed": int(seed),
                  "init_seed": int(seed if init_seed is None else init_seed),
                  "epochs": int(epochs), "eta": float(eta), "lr": float(lr), "latent_dim": int(latent_dim),
                  "units": int(units), "n_train": int(n), "adam_iterations0": int(adam_iterations0)}
    info = {"epoch_loss": epoch_loss, "train_loss_last_epoch": epoch_loss[-1] if epoch_loss else float("nan"),
            "runtime_train_s": time.perf_counter() - t0, "n_steps": n * epochs}
    return model, info


@torch.no_grad()
def per_day_errors(model: CVAE, df, z: str = "posterior", seed: int = 0, n_draws: int = 1) -> np.ndarray:
    """(n_draws, n_days) array of 100*sqrt(mean_h (x - xhat)^2) on scaled data (upstream compute_mse /
    compute_mse_test). z='posterior': z ~ N(mean(x,c), diag(scale^2)); z='prior': z ~ N(0, I)."""
    Y, C = _xy(df, model.meta["cond_cols"])
    X = torch.from_numpy(model.scale(Y))
    Ct = torch.from_numpy(C)
    gen = torch.Generator().manual_seed(int(seed))
    out = np.empty((n_draws, X.shape[0]))
    mean, logscale = model.encode(X, Ct)
    for k in range(n_draws):
        eps = torch.randn((X.shape[0], model.latent_dim), generator=gen, dtype=DTYPE)
        Z = mean + torch.exp(logscale) * eps if z == "posterior" else eps
        xhat = model.decode(Z, Ct)
        out[k] = (100.0 * torch.sqrt((X - xhat).square().mean(1))).numpy()
    return out


def test_mse(model: CVAE, df, z: str = "posterior", seed: int = 0, n_draws: int = 1) -> float:
    """Authors' selection metric (compute_mse, z='posterior') or compute_mse_test (z='prior'), averaged over days
    (and over n_draws independent draws if n_draws > 1). Fixed RNG -> deterministic."""
    return float(per_day_errors(model, df, z=z, seed=seed, n_draws=n_draws).mean())


@torch.no_grad()
def loss_on(model: CVAE, df, eta: float = KL_COEFF, seed: int = 0) -> dict:
    """Mean training objective on df with a fixed reparameterisation draw, plus its parts and KL per latent dim."""
    Y, C = _xy(df, model.meta["cond_cols"])
    X = torch.from_numpy(model.scale(Y))
    Ct = torch.from_numpy(C)
    gen = torch.Generator().manual_seed(int(seed))
    eps = torch.randn((X.shape[0], model.latent_dim), generator=gen, dtype=DTYPE)
    loss, rec, kl = model.loss_terms(X, Ct, eps, eta)
    mean, logscale = model.encode(X, Ct)
    kl_dim = 0.5 * (torch.exp(2.0 * logscale) + mean.square() - 1.0 - 2.0 * logscale)
    return {"loss": float(loss.mean()), "rec": float(rec.mean()), "kl": float(kl.mean()),
            "kl_per_dim": kl_dim.mean(0).tolist(),
            "posterior_std_mean": torch.exp(logscale).mean(0).tolist(),
            "posterior_mean_std": mean.std(0).tolist()}


@torch.no_grad()
def generate(model: CVAE, cond_matrix, n_samples: int, rng: np.random.Generator | None = None,
             eps: np.ndarray | None = None) -> np.ndarray:
    """Decoder samples in kWh, shape (n_samples, n_days, 48). z ~ N(0, I_d) drawn from `rng` unless `eps`
    (n_samples, n_days, d) is given (common random numbers across scenarios)."""
    C = np.asarray(cond_matrix, dtype=np.float64)
    if C.ndim != 2 or C.shape[1] != model.cond_dim:
        raise ValueError(f"cond_matrix must be (n_days, {model.cond_dim}), got {C.shape}")
    n_days = C.shape[0]
    if eps is None:
        eps = rng.standard_normal((n_samples, n_days, model.latent_dim))
    eps = np.asarray(eps, dtype=np.float64)
    assert eps.shape == (n_samples, n_days, model.latent_dim), eps.shape
    Z = torch.from_numpy(eps.reshape(-1, model.latent_dim))
    Cr = torch.from_numpy(C).repeat(n_samples, 1)  # rows: sample-major, matches Z reshape
    out = model.decode(Z, Cr).numpy().reshape(n_samples, n_days, model.input_dim)
    return model.unscale(out)


def save_model(model: CVAE, path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": model.meta,
                "arch": {"input_dim": model.input_dim, "cond_dim": model.cond_dim, "latent_dim": model.latent_dim,
                         "units": model.units}}, path)


def load_model(path: str) -> CVAE:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = CVAE(**ck["arch"])
    model.load_state_dict(ck["state_dict"])
    model.meta = dict(ck["meta"])
    model.eval()
    return model


# ----------------------------------------------------------------------------------------------------------------
# restart job (for process pools; lives here so that spawned workers can import it)
# ----------------------------------------------------------------------------------------------------------------
_TABLES: dict = {}


def _read_table(path: str):
    import pandas as pd
    if path not in _TABLES:
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_feather(path)
        df["date"] = pd.to_datetime(df["date"])
        df["cluster"] = df["cluster"].astype(int)
        _TABLES[path] = df
    return _TABLES[path]


def cluster_rows(df, cluster: int, sort_by_date: bool = False):
    sub = df.loc[df["cluster"] == int(cluster)]
    if sort_by_date:
        sub = sub.sort_values("date", kind="stable")
    return sub


def worker_init() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def evaluate_restart(model: CVAE, train_c, test_c, eta: float = KL_COEFF, n_avg: int = 20,
                     n_gen: int = 200, gen_seed: int = 12345) -> dict:
    """All per-restart diagnostics. Draw seeds are fixed and shared by all restarts (common random numbers)."""
    err_post0 = per_day_errors(model, test_c, "posterior", seed=0, n_draws=1)[0]
    err_post1 = per_day_errors(model, test_c, "posterior", seed=1, n_draws=1)[0]
    err_post_avg = per_day_errors(model, test_c, "posterior", seed=100, n_draws=n_avg).mean(0)
    err_prior0 = per_day_errors(model, test_c, "prior", seed=0, n_draws=1)[0]
    err_prior_avg = per_day_errors(model, test_c, "prior", seed=200, n_draws=n_avg).mean(0)
    tr_fixed = loss_on(model, train_c, eta=eta, seed=0)
    te_fixed = loss_on(model, test_c, eta=eta, seed=0)
    # paper eq. (8): mean over test days of ||Y - Yhat||^2 (scaled), z ~ posterior, same draw as err_post0
    eq8 = float(np.mean((err_post0 / 100.0) ** 2 * INPUT_DIM))
    # cheap preview of the paper's RMSE score: mean of n_gen prior samples vs observation, kWh
    Yte, Cte = _xy(test_c, model.meta["cond_cols"])
    S = generate(model, Cte, n_gen, rng=np.random.default_rng(gen_seed))
    dev = S.mean(0) - Yte
    rmse_rootmean = np.sqrt((dev ** 2).mean(1))
    return {
        "train_loss_final": tr_fixed["loss"], "train_rec_final": tr_fixed["rec"], "train_kl_final": tr_fixed["kl"],
        "train_kl_per_dim": tr_fixed["kl_per_dim"], "train_posterior_std_mean": tr_fixed["posterior_std_mean"],
        "train_posterior_mean_std": tr_fixed["posterior_mean_std"],
        "train_mse_post": float(per_day_errors(model, train_c, "posterior", seed=0)[0].mean()),
        "test_loss_final": te_fixed["loss"], "test_kl_final": te_fixed["kl"],
        "test_mse_post": float(err_post0.mean()),            # selection metric (authors' compute_mse)
        "test_mse_post_seed1": float(err_post1.mean()),      # same metric, fresh draws (metric noise)
        f"test_mse_post_avg{n_avg}": float(err_post_avg.mean()),
        "test_mse_prior": float(err_prior0.mean()),          # authors' compute_mse_test
        f"test_mse_prior_avg{n_avg}": float(err_prior_avg.mean()),
        "test_eq8_post": eq8,
        "test_rmse_mean200_kwh_median": float(np.median(rmse_rootmean)),
        "test_rmse_mean200_kwh_mean": float(rmse_rootmean.mean()),
        "test_sample_sd_kwh_mean": float(S.std(0, ddof=1).mean()),
        "per_day_post_seed0": err_post0.tolist(),
        f"per_day_post_avg{n_avg}": err_post_avg.tolist(),
        f"per_day_prior_avg{n_avg}": err_prior_avg.tolist(),
    }


def run_restart_job(job: dict) -> dict:
    """job keys: train_path, test_path, cluster, seed, model_path, json_path, [epochs, eta, adam_iterations0].
    Trains one restart, evaluates it, saves the model (.pt) and the metrics (.json); returns the metrics."""
    worker_init()
    t0 = time.perf_counter()
    tr = _read_table(job["train_path"])
    te = _read_table(job["test_path"])
    c = int(job["cluster"])
    train_c = cluster_rows(tr, c, sort_by_date=False)          # upstream: file order (= ascending date here)
    test_c = cluster_rows(te, c, sort_by_date=True)
    model, info = train_cvae(train_c, seed=int(job["seed"]), epochs=int(job.get("epochs", EPOCHS)),
                             eta=float(job.get("eta", KL_COEFF)),
                             adam_iterations0=int(job.get("adam_iterations0", 0)),
                             init_seed=job.get("init_seed"))
    metrics = evaluate_restart(model, train_c, test_c, eta=float(job.get("eta", KL_COEFF)))
    save_model(model, job["model_path"])
    rec = {"cluster": c, "seed": int(job["seed"]), "init_seed": model.meta["init_seed"],
           "epochs": model.meta["epochs"], "eta": model.meta["eta"],
           "adam_iterations0": model.meta["adam_iterations0"], "conso_min": model.meta["conso_min"],
           "conso_max": model.meta["conso_max"], "n_train": model.meta["n_train"], "n_test": int(len(test_c)),
           **info, **metrics, "runtime_total_s": time.perf_counter() - t0, "model_path": job["model_path"]}
    tmp = job["json_path"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, job["json_path"])
    return rec


# ----------------------------------------------------------------------------------------------------------------
# self test
# ----------------------------------------------------------------------------------------------------------------
def _hand_forward_loss(W, x, c, eps, eta):
    """Pure-Python loop implementation (no numpy/torch maths) of the loss for one sample."""
    (W1, b1, W2, b2, V1, a1, V2, a2) = W
    d = len(eps)
    inp = list(x) + list(c)
    h = [max(0.0, sum(W1[j][k] * inp[k] for k in range(len(inp))) + b1[j]) for j in range(len(b1))]
    o = [sum(W2[j][k] * h[k] for k in range(len(h))) + b2[j] for j in range(len(b2))]
    mean, logscale = o[:d], o[d:]
    z = [mean[i] + math.exp(logscale[i]) * eps[i] for i in range(d)]
    inp2 = z + list(c)
    h2 = [max(0.0, sum(V1[j][k] * inp2[k] for k in range(len(inp2))) + a1[j]) for j in range(len(a1))]
    xhat = [sum(V2[j][k] * h2[k] for k in range(len(h2))) + a2[j] for j in range(len(a2))]
    rec = sum((x[i] - xhat[i]) ** 2 for i in range(len(x)))
    kl = sum(0.5 * (math.exp(2 * logscale[i]) + mean[i] ** 2 - 1.0 - 2.0 * logscale[i]) for i in range(d))
    return eta * kl + rec, rec, kl, xhat, mean, logscale


def selftest(verbose: bool = True) -> dict:
    import pandas as pd
    torch.set_num_threads(1)
    res = {}
    say = print if verbose else (lambda *a, **k: None)

    # 1) toy model vs hand computation
    in_dim, cdim, d, u = 3, 2, 2, 3
    m = CVAE(in_dim, cdim, d, u)
    vals = {}
    with torch.no_grad():
        for name, p in m.named_parameters():
            idx = torch.arange(p.numel(), dtype=DTYPE).reshape(p.shape)
            p.copy_(0.3 * torch.sin(1.7 * idx + 0.4 * len(name)) + 0.05)
            vals[name] = p.tolist()
    W = (vals["enc_hidden.weight"], vals["enc_hidden.bias"], vals["enc_out.weight"], vals["enc_out.bias"],
         vals["dec_hidden.weight"], vals["dec_hidden.bias"], vals["dec_out.weight"], vals["dec_out.bias"])
    x, c, eps, eta = [0.2, 0.5, 0.9], [1.0, 0.0], [0.3, -1.2], 10.0
    hand_loss, hand_rec, hand_kl, hand_xhat, hand_mean, hand_ls = _hand_forward_loss(W, x, c, eps, eta)
    xt, ct, et = (torch.tensor([v], dtype=DTYPE) for v in (x, c, eps))
    with torch.no_grad():
        loss, rec, kl = m.loss_terms(xt, ct, et, eta)
        fun_loss = _step_loss(m.params_list(), torch.cat((xt, ct), 1), ct, et, eta, d)
    kl_torch = torch.distributions.kl_divergence(
        torch.distributions.Normal(torch.tensor(hand_mean, dtype=DTYPE), torch.exp(torch.tensor(hand_ls, dtype=DTYPE))),
        torch.distributions.Normal(torch.zeros(d, dtype=DTYPE), torch.ones(d, dtype=DTYPE))).sum()
    res["toy_loss_model"] = float(loss)
    res["toy_loss_hand"] = hand_loss
    res["toy_loss_functional"] = float(fun_loss)
    res["toy_kl_hand"], res["toy_kl_torchdist"] = hand_kl, float(kl_torch)
    assert abs(float(loss) - hand_loss) < 1e-12, (float(loss), hand_loss)
    assert abs(float(fun_loss) - hand_loss) < 1e-12
    assert abs(float(rec) - hand_rec) < 1e-12 and abs(float(kl) - hand_kl) < 1e-12
    assert abs(float(kl_torch) - hand_kl) < 1e-12
    say(f"[selftest] toy loss model={float(loss):.12f} hand={hand_loss:.12f} functional={float(fun_loss):.12f} "
        f"(rec {hand_rec:.6f}, kl {hand_kl:.6f}, torch.distributions kl {float(kl_torch):.6f})")

    # 2) gradient check (float64 finite differences)
    P = [p.detach().clone().requires_grad_(True) for p in m.params_list()]
    ok = torch.autograd.gradcheck(lambda *ps: _step_loss(list(ps), torch.cat((xt, ct), 1), ct, et, eta, d), tuple(P),
                                  eps=1e-6, atol=1e-7, rtol=1e-5)
    res["gradcheck"] = bool(ok)
    say(f"[selftest] autograd vs finite differences: {ok}")

    # 3) one Keras-Adam step vs hand formula (t = 1) and with a shared-iteration counter (quirk)
    p = torch.tensor([0.5, -0.2, 0.0], dtype=DTYPE)
    g = torch.tensor([0.3, -2.0, 1e-9], dtype=DTYPE)
    for it0 in (0, 10000):
        q = p.clone()
        opt = KerasAdam([q], lr=1e-3, iterations=it0)
        opt.step([g])
        t = it0 + 1
        lr_t = 1e-3 * math.sqrt(1 - 0.999 ** t) / (1 - 0.9 ** t)
        expect = p - lr_t * (0.1 * g) / (torch.sqrt(0.001 * g * g) + 1e-7)
        assert torch.allclose(q, expect, rtol=0, atol=1e-15), (q, expect)
        res[f"adam_first_step_iter0_{it0}"] = (q - p).tolist()
    say(f"[selftest] Keras Adam first step (fresh): {res['adam_first_step_iter0_0']} ; "
        f"(shared counter at 10000): {res['adam_first_step_iter0_10000']}")

    # 4) Glorot init bounds, zero biases, shapes, determinism
    full = CVAE()
    full.keras_init(torch.Generator().manual_seed(0))
    for lin in (full.enc_hidden, full.enc_out, full.dec_hidden, full.dec_out):
        fo, fi = lin.weight.shape
        lim = math.sqrt(6.0 / (fi + fo))
        assert float(lin.weight.detach().abs().max()) <= lim and float(lin.bias.detach().abs().max()) == 0.0
    assert tuple(full.enc_hidden.weight.shape) == (15, 150) and tuple(full.enc_out.weight.shape) == (8, 15)
    assert tuple(full.dec_hidden.weight.shape) == (15, 106) and tuple(full.dec_out.weight.shape) == (48, 15)
    nparam = sum(p.numel() for p in full.parameters())
    res["n_params"] = int(nparam)
    say(f"[selftest] shapes ok: enc 150->15->8, dec 106->15->48, {nparam} parameters, glorot bounds ok")

    # 5) synthetic data: one epoch reduces the loss; shapes of test_mse / generate; determinism; save/load
    rng = np.random.default_rng(0)
    n = 120
    cond = np.zeros((n, 102))
    cond[:, 0] = rng.uniform(0, 1, n)
    cond[:, 5] = rng.integers(0, 2, n)
    hh = np.arange(48)
    Y = (0.2 + 0.1 * np.sin(2 * np.pi * (hh[None, :] - 20) / 48) * (1 + cond[:, [0]])
         + 0.05 * cond[:, [5]] + 0.01 * rng.standard_normal((n, 48)))
    df = pd.DataFrame(np.hstack([Y, cond]), columns=COLS_DATA + COLS_COND)
    m0, _ = train_cvae(df, seed=3, epochs=0)
    before = loss_on(m0, df, seed=0)["loss"]
    m1, info1 = train_cvae(df, seed=3, epochs=1)
    after = loss_on(m1, df, seed=0)["loss"]
    res["synthetic_loss_before"], res["synthetic_loss_after_1epoch"] = before, after
    assert after < before, (before, after)
    say(f"[selftest] synthetic: mean loss (fixed eps) before {before:.4f} -> after 1 epoch {after:.4f}; "
        f"runtime {info1['runtime_train_s']:.2f}s for {info1['n_steps']} steps "
        f"({1e6 * info1['runtime_train_s'] / info1['n_steps']:.0f} us/step)")
    m1b, _ = train_cvae(df, seed=3, epochs=1)
    m2, _ = train_cvae(df, seed=4, epochs=1)
    same = all(torch.equal(a, b) for a, b in zip(m1.params_list(), m1b.params_list()))
    diff = not all(torch.equal(a, b) for a, b in zip(m1.params_list(), m2.params_list()))
    assert same and diff
    res["deterministic_same_seed"], res["different_seed_differs"] = same, diff
    # init_seed override (upstream-quirk emulation): identical initial weights, different noise stream
    a0, _ = train_cvae(df, seed=5, epochs=0, init_seed=3)
    a1, _ = train_cvae(df, seed=5, epochs=1, init_seed=3)
    same_init = all(torch.equal(a, b) for a, b in zip(m0.params_list(), a0.params_list()))
    noise_differs = not all(torch.equal(a, b) for a, b in zip(m1.params_list(), a1.params_list()))
    assert same_init and noise_differs and a1.meta["init_seed"] == 3 and m1.meta["init_seed"] == 3
    res["init_seed_override_same_init"], res["init_seed_override_noise_differs"] = same_init, noise_differs
    E = per_day_errors(m1, df, "posterior", seed=0, n_draws=3)
    assert E.shape == (3, n)
    # hand check of compute_mse for day 0 / draw 0
    gen = torch.Generator().manual_seed(0)
    eps0 = torch.randn((n, 4), generator=gen, dtype=DTYPE)[0:1]
    X0 = torch.from_numpy(m1.scale(Y[0:1]))
    C0 = torch.from_numpy(cond[0:1])
    with torch.no_grad():
        mu, ls = m1.encode(X0, C0)
        xh = m1.decode(mu + torch.exp(ls) * eps0, C0)
    hand = 100.0 * math.sqrt(float(((X0 - xh) ** 2).mean()))
    assert abs(hand - E[0, 0]) < 1e-10, (hand, E[0, 0])
    S = generate(m1, cond[:7], 5, rng=np.random.default_rng(1))
    assert S.shape == (5, 7, 48)
    # sample k, day j must equal decoding of eps[k, j] with cond j
    eps_g = np.random.default_rng(1).standard_normal((5, 7, 4))
    with torch.no_grad():
        one = m1.decode(torch.from_numpy(eps_g[3, 6:7]), torch.from_numpy(cond[6:7])).numpy()
    assert np.allclose(m1.unscale(one)[0], S[3, 6], atol=1e-13)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.pt")
        save_model(m1, path)
        m3 = load_model(path)
        S3 = generate(m3, cond[:7], 5, rng=np.random.default_rng(1))
    assert np.array_equal(S, S3) and m3.meta == m1.meta
    res["generate_shape"] = list(S.shape)
    say("[selftest] determinism, per-day compute_mse hand check, generate indexing, save/load round-trip: ok")
    res["status"] = "ok"
    return res


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        out = selftest()
        print(json.dumps({k: v for k, v in out.items()}, indent=1))
    else:
        print(__doc__)
