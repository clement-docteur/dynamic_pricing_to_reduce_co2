"""Demand-response simulators for the Low Carbon London 2013 dynamic time-of-use trial.

Everything this folder needs is in this file:

* `load_data()`        the 2013 data: ToU cluster-mean consumption, flat-tariff (Std) mean, tariff schedule, temperature
* `ResponseModel`      the demand-response baseline: a per-cluster regression with the Std households as control
* `CVAEEnsemble`       the conditional VAEs from the Brégère & Bessa (2020) reproduction, run in numpy
* `predictions_frame`, `write_predictions`, `read_predictions`   the shared prediction format (see README.md)

Time convention: slot h (0..47) of date d is the half-hour ENDING at d + h*30 min GMT, so slot 0 is 23:30-24:00 of the
previous day. The tariff is the trial's schedule as published by the paper's authors, which the consumption data
confirm is aligned to these slots (CVAE_reproduction/REPRO_SPEC.md, section 6).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import pickle  # noqa: E402
import zipfile  # noqa: E402
from collections import OrderedDict  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = HERE / "data"                      # filled by prepare_data.py from CVAE_reproduction/
CVAE_MODELS = DATA / "cvae_models"

H = 48
N_CLUSTERS = 4
SPLIT = {"train": ("2013-01-01", "2013-08-31"), "val": ("2013-09-01", "2013-10-31"),
         "test": ("2013-11-01", "2013-12-31")}
BANK_HOLIDAYS_2013 = pd.to_datetime(["2013-01-01", "2013-03-29", "2013-04-01", "2013-05-06", "2013-05-27",
                                     "2013-08-26", "2013-12-25", "2013-12-26"])


def cols(prefix):
    return [f"{prefix}_{h:02d}" for h in range(H)]


def read_parquet(path) -> pd.DataFrame:
    # DuckDB rather than pyarrow: pyarrow's parquet module is blocked on the machine this was developed on.
    return duckdb.sql(f"select * from read_parquet('{Path(path).as_posix()}')").df()


def split_of(dates) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    out = np.full(len(d), "", dtype=object)
    for name, (a, b) in SPLIT.items():
        out[(d >= a) & (d <= b)] = name
    return out


# ================================================================================================== data
@dataclass
class Data:
    dates: pd.DatetimeIndex     # 365 days of 2013
    low: np.ndarray             # (365, 48) 1 = Low-price slot
    high: np.ndarray            # (365, 48) 1 = High-price slot
    temp: np.ndarray            # (365, 48) ERA5 2 m temperature, central London, degC
    y_tou: np.ndarray           # (4, 365, 48) mean kWh per ToU household, by cluster
    y_std: np.ndarray           # (365, 48) mean kWh per flat-tariff (Std) household
    cluster_sizes: np.ndarray   # (4,) ToU households per cluster
    cvae_cond: pd.DataFrame     # 365 rows: the CVAE's non-tariff conditions

    @property
    def split(self):
        return split_of(self.dates)

    @property
    def day_type(self):
        """0 = working day, 1 = weekend or bank holiday."""
        return np.asarray((self.dates.dayofweek >= 5) | self.dates.isin(BANK_HOLIDAYS_2013), int)


def load_data() -> Data:
    """Reads only demand_response/data/ (see README.md for where each file comes from)."""
    cond = read_parquet(DATA / "conditions_ours.parquet").sort_values("date").reset_index(drop=True)
    pc = read_parquet(DATA / "power_consump_ours.parquet").sort_values(["cluster", "date"])
    y_tou = pc[cols("consumption")].to_numpy(float).reshape(N_CLUSTERS, 365, H)
    sizes = pc.groupby("cluster")["n_households"].first().to_numpy(int)
    y_std = pd.read_csv(DATA / "std_mean_kwh.csv").sort_values("date")[cols("kwh")].to_numpy(float)

    return Data(dates=pd.DatetimeIndex(cond["date"]), low=cond[cols("low_auth")].to_numpy(float),
                high=cond[cols("high_auth")].to_numpy(float), temp=cond[cols("temp")].to_numpy(float),
                y_tou=y_tou, y_std=y_std, cluster_sizes=sizes,
                cvae_cond=_cvae_conditions(pc[pc.cluster == 0].reset_index(drop=True)))


# ================================================================================================== baseline
BANDS = {"night 00-06": range(1, 13), "morning 06-12": range(13, 25), "afternoon 12-16": range(25, 33),
         "peak 16-20": range(33, 41), "evening 20-24": [*range(41, 48), 0]}
BAND_OF = np.zeros(H, int)
for _i, _slots in enumerate(BANDS.values()):
    BAND_OF[list(_slots)] = _i
BAND_TERMS = [f"in_{k}|{b}" for k in ("High", "Low") for b in BANDS]
EDGE_TERMS = ["pre2h_Low", "post30m_Low", "pre2h_High"]
TARIFF_TERMS = ["in_High", "in_Low", *BAND_TERMS, *EDGE_TERMS]


def _shift_any(x, ks):
    """True where x is True k slots later (k > 0) or earlier (k < 0) along the last axis."""
    out = np.zeros_like(x)
    for k in ks:
        if k > 0:
            out[..., :-k] |= x[..., k:]
        else:
            out[..., -k:] |= x[..., :k]
    return out


def tariff_terms(low, high, continuous=True) -> np.ndarray:
    """Tariff regressors (n_days, 48, len(TARIFF_TERMS)) for 0/1 schedules low, high of shape (n_days, 48).

    continuous=True treats the rows as consecutive days (a block's edges can cross midnight, as in the historical
    schedule); continuous=False treats each day on its own (for hypothetical one-day schedules)."""
    low, high = np.asarray(low, bool), np.asarray(high, bool)
    n = low.shape[0]
    L, Hi = (low.reshape(1, -1), high.reshape(1, -1)) if continuous else (low, high)
    band = np.tile(BAND_OF, n)[None] if continuous else BAND_OF[None]
    pre_low = ~L & _shift_any(L, range(1, 5))          # a Low slot starts within the next 2 h
    post_low = ~L & _shift_any(L, [-1])                # the half-hour right after a Low block
    pre_high = ~Hi & _shift_any(Hi, range(1, 5))       # a High slot starts within the next 2 h
    terms = {"in_High": Hi, "in_Low": L, "pre2h_Low": pre_low, "post30m_Low": post_low, "pre2h_High": pre_high}
    for i, b in enumerate(BANDS):
        terms[f"in_High|{b}"] = Hi & (band == i)
        terms[f"in_Low|{b}"] = L & (band == i)
    return np.stack([np.broadcast_to(terms[k], L.shape).reshape(n, H) for k in TARIFF_TERMS], -1).astype(float)


class ResponseModel:
    """Demand-response baseline, fitted per ToU cluster c:

        log(y_tou[c, d, h] / y_std[d, h]) = a_c[h, day type] + b_c[band(h)] * (temp[d, h] - 10) / 10
                                            + theta_c . tariff_terms[d, h]

    The flat-tariff (Std) households are the control: weather, holidays and seasonal swings hit both groups and
    cancel in the ratio, so theta_c is a difference-in-differences estimate of the tariff response. Tariff terms:
    in-block High / Low (pooled, plus ridge-shrunk deviations per time-of-day band) and three edge effects found in
    the data (dip in the 2 h before a Low block, carry-over in the half-hour after it, pre-loading in the 2 h before
    a High block). With tariff=False the same model is fitted without tariff terms (a no-response control).

    Predictions are y_std * exp(fitted log ratio), so they use the same day's observed Std consumption: this is a
    simulator of the tariff response, not a day-ahead forecast.
    """

    def __init__(self, tariff=True, ridge_band=30.0):
        self.tariff, self.ridge_band = tariff, ridge_band
        self.n_base = 2 * H + len(BANDS)
        self.beta: dict[int, np.ndarray] = {}

    def design(self, data: Data, low=None, high=None, continuous=True) -> np.ndarray:
        low = data.low if low is None else low
        high = data.high if high is None else high
        n = len(data.dates)
        X = np.zeros((n, H, self.n_base))
        X[np.arange(n)[:, None], np.arange(H)[None], np.arange(H)[None] + H * data.day_type[:, None]] = 1.0
        X[np.arange(n)[:, None], np.arange(H)[None], 2 * H + BAND_OF[None]] = (data.temp - 10.0) / 10.0
        if self.tariff:
            X = np.concatenate([X, tariff_terms(low, high, continuous)], -1)
        return X.reshape(n * H, -1)

    def fit(self, data: Data, days) -> "ResponseModel":
        """Fit every cluster on the boolean day mask `days`."""
        X = self.design(data)
        rows = np.repeat(np.asarray(days, bool), H)
        pen = np.zeros(X.shape[1])
        if self.tariff:
            pen[[self.n_base + TARIFF_TERMS.index(t) for t in BAND_TERMS]] = self.ridge_band
        A = X[rows].T @ X[rows] + np.diag(pen)
        for c in range(N_CLUSTERS):
            y = np.log(data.y_tou[c]).ravel() - np.log(data.y_std).ravel()
            self.beta[c] = np.linalg.solve(A, X[rows].T @ y[rows])
        return self

    def predict(self, data: Data, low=None, high=None, continuous=True) -> np.ndarray:
        """(4, 365, 48) kWh per household under schedule low/high (default: the historical one)."""
        X = self.design(data, low, high, continuous)
        return np.stack([data.y_std * np.exp((X @ self.beta[c]).reshape(-1, H)) for c in range(N_CLUSTERS)])

    def tariff_effects_pct(self, c) -> dict:
        """Multiplicative effects in %: in-block High/Low per time-of-day band, and the three edge effects."""
        th = dict(zip(TARIFF_TERMS, self.beta[c][self.n_base:]))
        out = {f"{k}|{b}": 100 * np.expm1(th[f"in_{k}"] + th[f"in_{k}|{b}"]) for k in ("High", "Low") for b in BANDS}
        out.update({t: 100 * np.expm1(th[t]) for t in EDGE_TERMS})
        return out


# ================================================================================================== CVAE
def _cvae_conditions(g: pd.DataFrame) -> pd.DataFrame:
    """Non-tariff CVAE conditions with no look-ahead: min-max scaling and the temperature PCA are fitted on the
    training days only (as in CVAE_reproduction/scripts/50_chrono_split.py)."""
    tr = split_of(g["date"]) == "train"
    ts = g["temp_smooth"].to_numpy(float)
    out = pd.DataFrame({"temp_smooth_n": (ts - ts[tr].min()) / (ts[tr].max() - ts[tr].min()),
                        "work_day": g["work_day"].to_numpy(float)})
    T = g[cols("temp")].to_numpy(float)
    mean = T[tr].mean(0)
    U, _, Vt = np.linalg.svd(T[tr] - mean, full_matrices=False)
    sign = np.sign(U[np.argmax(np.abs(U), 0), np.arange(U.shape[1])])   # scikit-learn < 1.5 sign convention
    sign[sign == 0] = 1.0
    s = (T - mean) @ (Vt[:3] * sign[:3, None]).T
    s = (s - s[tr].min()) / (s[tr].max() - s[tr].min())
    for j in range(3):
        out[f"pca_temp_{j}"] = s[:, j]
    return out


class CVAEEnsemble:
    """The reproduction's CVAEs trained on Jan-Aug 2013 (track B, without the position-in-year condition; 20 restarts
    per cluster; CVAE_reproduction/scripts/50_chrono_split.py). Prediction = mean decoder output over the 20 restarts
    and `n_eps` latent draws. The latent space has collapsed (KL ~ 1e-6), so the draws barely matter."""

    def __init__(self, data: Data, n_eps=20, seed=0):
        self.cond = data.cvae_cond
        self.eps = np.random.default_rng(seed).standard_normal((n_eps, 4))
        self.models = {c: [self._load(CVAE_MODELS / f"B_c{c}_no_pos_y_s{s:02d}.pt") for s in range(20)]
                       for c in range(N_CLUSTERS)}

    @staticmethod
    def _load(path):
        ck = load_torch_checkpoint(path)
        sd, meta = ck["state_dict"], ck["meta"]
        return dict(W1=sd["dec_hidden.weight"], b1=sd["dec_hidden.bias"], W2=sd["dec_out.weight"],
                    b2=sd["dec_out.bias"], lo=meta["conso_min"], hi=meta["conso_max"], cols=tuple(meta["cond_cols"]))

    def _cond_matrix(self, cond_cols, low, high):
        M = np.zeros((len(low), len(cond_cols)))
        for j, col in enumerate(cond_cols):
            kind, _, h = col.partition("_")
            if kind in ("low", "high") and h.isdigit():
                M[:, j] = (low if kind == "low" else high)[:, int(h)]
            else:
                M[:, j] = self.cond[col].to_numpy(float)
        return M

    @staticmethod
    def decode(m, z, C):
        hidden = np.maximum(np.hstack([z, C]) @ m["W1"].T + m["b1"], 0.0)
        return (hidden @ m["W2"].T + m["b2"]) * (m["hi"] - m["lo"]) + m["lo"]

    def predict(self, low, high) -> np.ndarray:
        """(4, 365, 48) kWh per household under schedule low/high (365, 48)."""
        out = np.zeros((N_CLUSTERS, len(low), H))
        for c in range(N_CLUSTERS):
            for m in self.models[c]:
                C = self._cond_matrix(m["cols"], np.asarray(low, float), np.asarray(high, float))
                for e in self.eps:
                    out[c] += self.decode(m, np.repeat(e[None], len(low), 0), C)
            out[c] /= len(self.models[c]) * len(self.eps)
        return np.maximum(out, 1e-3)


_TORCH_DTYPES = {"DoubleStorage": np.float64, "FloatStorage": np.float32, "LongStorage": np.int64,
                 "IntStorage": np.int32}


def load_torch_checkpoint(path) -> dict:
    """Read a torch.save() checkpoint of plain tensors and dicts into numpy arrays without importing torch."""
    zf = zipfile.ZipFile(path)
    root = zf.namelist()[0].split("/")[0]

    def rebuild_tensor(storage, offset, size, stride, *_):
        dtype, key = storage
        raw = np.frombuffer(zf.read(f"{root}/data/{key}"), dtype=dtype)
        if len(size) == 0:
            return raw[offset].copy()
        return np.lib.stride_tricks.as_strided(raw[offset:], shape=tuple(size),
                                               strides=tuple(s * raw.itemsize for s in stride)).copy()

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return rebuild_tensor
            if (module, name) == ("torch._utils", "_rebuild_parameter"):
                return lambda data, *a: data
            if module == "torch" and name in _TORCH_DTYPES:
                return name
            if (module, name) == ("collections", "OrderedDict"):
                return OrderedDict
            raise pickle.UnpicklingError(f"blocked {module}.{name}")

        def persistent_load(self, pid):
            _, storage_type, key, _location, _n = pid
            name = storage_type if isinstance(storage_type, str) else getattr(storage_type, "__name__", "")
            return _TORCH_DTYPES[name], key

    return Unpickler(zf.open(f"{root}/data.pkl")).load()


# ================================================================================================== shared format
PREDICTION_COLUMNS = ["model", "scenario", "cluster", "date", "slot", "interval_end_utc", "split", "tariff",
                      "y_true_kwh", "y_pred_kwh"]


def predictions_frame(model: str, scenario: str, data: Data, y_pred, low, high, y_true=None) -> pd.DataFrame:
    """Long table in the shared format (one row per model x scenario x cluster x day x slot; see README.md).
    y_pred and y_true are (4, 365, 48) kWh per household; y_true=None leaves it empty (counterfactual scenarios)."""
    n = len(data.dates)
    c, d, h = (a.ravel() for a in np.meshgrid(np.arange(N_CLUSTERS), np.arange(n), np.arange(H), indexing="ij"))
    tariff = np.where(np.asarray(high) > 0, "H", np.where(np.asarray(low) > 0, "L", "N"))
    return pd.DataFrame({
        "model": model, "scenario": scenario, "cluster": c, "date": data.dates.strftime("%Y-%m-%d").to_numpy()[d], "slot": h,
        "interval_end_utc": (data.dates[d] + pd.to_timedelta(30 * h, "min")).strftime("%Y-%m-%d %H:%M"),
        "split": data.split[d], "tariff": tariff[d, h],
        "y_true_kwh": np.nan if y_true is None else np.round(np.asarray(y_true).ravel(), 6),
        "y_pred_kwh": np.round(np.asarray(y_pred).ravel(), 6)})[PREDICTION_COLUMNS]


def write_predictions(df: pd.DataFrame, path):
    df[PREDICTION_COLUMNS].to_csv(path, index=False, compression="gzip" if str(path).endswith(".gz") else None)


def read_predictions(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date", "interval_end_utc"])
