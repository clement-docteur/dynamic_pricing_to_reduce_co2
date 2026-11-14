"""Shared paths, constants, loaders and time conventions for the carbon-aware tariff pipeline.

Time conventions (verified in CVAE_reproduction/, see REPRO_SPEC.md sections 6-7):
  * slot h (0..47) of date d = the LCL half-hour ENDING at d + h*30 min GMT, i.e. covering
    [d + (h-1)*30 min, d + h*30 min). Slot 0 = 23:30-24:00 of the previous day.
  * NESO DATETIME is UTC. We assume it labels the START of its half-hour (not documented; lag-1
    autocorrelation is 0.99, so a 30-min error barely matters). CI of slot h of d = NESO row d + (h-1)*30 min.
  * Historical tariff = the authors' schedule (`low_auth_hh` / `high_auth_hh` in conditions_ours), which is
    physically aligned to the slots and more accurate than Tariffs.xlsx in Nov-Dec 2013.

Environment: pyarrow's parquet/feather DLLs and torch are blocked by Windows Application Control on this
machine, so parquet is read through DuckDB and CVAE checkpoints through `load_torch_checkpoint` (pure Python).
Import this module before numpy: it caps the BLAS thread pool at 4 (project rule; 8 threads oversubscribe the CPU).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from collections import OrderedDict  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
PIPE = PROJECT / "carbon_tariff_pipeline"
REPRO = PROJECT / "CVAE_reproduction"
OUT = PIPE / "outputs"
FIG = PIPE / "figures"

H = 48
PRICE = {"L": 3.99, "N": 11.76, "H": 67.20}           # p/kWh
FLAT_PRICE = 14.228                                    # Std (flat) tariff, p/kWh
SPLIT = {"train": ("2013-01-01", "2013-08-31"), "val": ("2013-09-01", "2013-10-31"),
         "test": ("2013-11-01", "2013-12-31")}
DATES_2013 = pd.date_range("2013-01-01", "2013-12-31", freq="D")

# England & Wales bank holidays 2009-2014 (gov.uk)
BANK_HOLIDAYS = pd.to_datetime([
    "2009-01-01", "2009-04-10", "2009-04-13", "2009-05-04", "2009-05-25", "2009-08-31", "2009-12-25", "2009-12-28",
    "2010-01-01", "2010-04-02", "2010-04-05", "2010-05-03", "2010-05-31", "2010-08-30", "2010-12-27", "2010-12-28",
    "2011-01-03", "2011-04-22", "2011-04-25", "2011-04-29", "2011-05-02", "2011-05-30", "2011-08-29", "2011-12-26",
    "2011-12-27", "2012-01-02", "2012-04-06", "2012-04-09", "2012-05-07", "2012-06-04", "2012-06-05", "2012-08-27",
    "2012-12-25", "2012-12-26", "2013-01-01", "2013-03-29", "2013-04-01", "2013-05-06", "2013-05-27", "2013-08-26",
    "2013-12-25", "2013-12-26", "2014-01-01"])

# palette (dataviz reference palette, light mode)
BLUE, ORANGE, AQUA, VIOLET, GREEN, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#008300", "#e87ba4"
INK, INK2, MUTED, GRID, SHADE, EDGE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f0efec", "#c3c2b7"


def hours(a, b) -> slice:
    """Slots covering [a:00, b:00) GMT under the slot convention above (e.g. hours(16, 20) = slots 33-40)."""
    return slice(2 * a + 1, 2 * b + 1)


def make_log():
    t0 = time.time()
    return (lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)), t0


def mpl_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.edgecolor": EDGE, "axes.axisbelow": True, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10, "axes.titlelocation": "left",
                         "legend.frameon": False, "lines.linewidth": 2, "figure.facecolor": "white"})
    return plt


def read_parquet(path) -> pd.DataFrame:
    return duckdb.sql(f"select * from read_parquet('{Path(path).as_posix()}')").df()


def split_of(dates) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    out = np.full(len(d), "", dtype=object)
    for k, (a, b) in SPLIT.items():
        out[(d >= np.datetime64(a)) & (d <= np.datetime64(b))] = k
    return out


def day_type(dates) -> np.ndarray:
    """0 = working day, 1 = weekend or bank holiday."""
    dates = pd.DatetimeIndex(dates)
    return np.asarray((dates.dayofweek >= 5) | dates.isin(BANK_HOLIDAYS)).astype(int)


def save_json(obj, path):
    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=conv), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ NESO
def load_neso() -> pd.DataFrame:
    df = pd.read_csv(PROJECT / "df_fuel_ckan.csv", parse_dates=["DATETIME"])
    df = df.sort_values("DATETIME").set_index("DATETIME").loc[:"2014-01-02"]
    assert not df.index.duplicated().any()
    return df


def day_matrix(series: pd.Series, dates) -> np.ndarray:
    """(n_days, 48) matrix: value of slot h of date d = series at d + (h-1)*30 min (NESO start label)."""
    dates = pd.DatetimeIndex(dates)
    idx = (dates.values[:, None] + (np.arange(H)[None, :] - 1) * np.timedelta64(30, "m")).ravel()
    return series.reindex(idx).to_numpy(float).reshape(len(dates), H)


# ------------------------------------------------------------------------------------------------ LCL
def cols(prefix):
    return [f"{prefix}_{h:02d}" for h in range(H)]


@lru_cache(maxsize=1)
def _conditions() -> pd.DataFrame:
    return read_parquet(REPRO / "data" / "conditions_ours.parquet").sort_values("date").reset_index(drop=True)


def load_tariff_2013():
    """Historical schedule (authors' version) as (365, 48) arrays: low, high (0/1) and price (p/kWh)."""
    cond = _conditions()
    low, high = cond[cols("low_auth")].to_numpy(float), cond[cols("high_auth")].to_numpy(float)
    return low, high, schedule_price(low, high)


def load_temps_2013() -> np.ndarray:
    """ERA5 2 m temperature at central London, (365, 48) degC, slot convention."""
    return _conditions()[cols("temp")].to_numpy(float)


def schedule_price(low, high):
    return np.where(high > 0, PRICE["H"], np.where(low > 0, PRICE["L"], PRICE["N"]))


def price_paid(price, y, axis=None):
    """Consumption-weighted average price (p/kWh) = bill / kWh."""
    return (price * y).sum(axis) / y.sum(axis)


def load_households(std=True):
    """Household kWh (n, 365, 48) float32 for the 1,032 ToU households (+ cluster labels) and, if std, the 1,032
    flat-tariff (Std) ones.

    Clusters = Track B of the reproduction (our NMF + k-medoids run (a), relabelled to the paper's 0-3 with the
    Hungarian map {nmf 0->2, 1->0, 2->1, 3->3}; sizes 285/299/299/149). Cluster means reproduce
    power_consump_ours.parquet to < 4e-7 kWh.
    """
    K = np.load(REPRO / "data" / "lcl2013_kwh_filled.npy", mmap_mode="r")
    cl = read_parquet(REPRO / "data" / "clusters_run_a.parquet")
    tou = cl[cl.group == "ToU"].sort_values("row_in_filled")
    out = dict(cluster=tou["nmf"].map({0: 2, 1: 0, 2: 1, 3: 3}).to_numpy(int),
               Y_tou=np.asarray(K[tou.row_in_filled.to_numpy()]).reshape(-1, 365, H))
    if std:
        rows = cl[cl.group == "Std"].sort_values("row_in_filled").row_in_filled.to_numpy()
        out["Y_std"] = np.asarray(K[rows]).reshape(-1, 365, H)
    return out


def cluster_aggregates(hh):
    """sizes (4,), household weights w (4,), cluster-mean kWh Yc (4, 365, 48) and the household mean y_all (365, 48)."""
    cl = hh["cluster"]
    sizes = np.bincount(cl)
    w = sizes / sizes.sum()
    Yc = np.stack([hh["Y_tou"][cl == c].mean(0) for c in range(4)]).astype(float)
    return sizes, w, Yc, np.einsum("c,cdh->dh", w, Yc)


# ------------------------------------------------------------------------------------------------ torch-free checkpoint reader
_DT = {"DoubleStorage": np.float64, "FloatStorage": np.float32, "LongStorage": np.int64, "IntStorage": np.int32}


def load_torch_checkpoint(path) -> dict:
    """Read a torch.save() zip checkpoint of plain tensors/dicts into numpy without importing torch."""
    zf = zipfile.ZipFile(path)
    root = zf.namelist()[0].split("/")[0]

    def rebuild_tensor(storage, offset, size, stride, *args):
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
            if module == "torch" and name in _DT:
                return name
            if (module, name) == ("collections", "OrderedDict"):
                return OrderedDict
            raise pickle.UnpicklingError(f"blocked {module}.{name}")

        def persistent_load(self, pid):
            _, stype, key, _loc, _n = pid
            stype = stype if isinstance(stype, str) else getattr(stype, "__name__", str(stype))
            return _DT[stype], key

    return Unpickler(zf.open(f"{root}/data.pkl")).load()
