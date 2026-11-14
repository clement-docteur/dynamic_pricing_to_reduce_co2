"""Conditioning variables of the CVAE (Bregere & Bessa 2020) rebuilt from raw data.

Every recipe below was reverse-engineered from, and verified against, the authors'
``power_consump_nmf.feather`` by ``scripts/03_conditions.py`` (numbers in
``outputs/conditions_checks.json``).  The condition columns are identical for the
four clusters (checked), so one row per day is enough.

* ``temp_00..47``   half-hourly London temperature (degC).  Authors: hourly NOAA
  observations at the even slots (full hours, UTC: no DST shift is visible), the odd
  slots are exactly the mid-point of the two neighbouring hours (linear
  interpolation) and the last slot of the year is clamped to 23:00.  Ours: ERA5
  2 m temperature (hourly, UTC, 0.25 deg grid) at a London point, linearly
  interpolated in time the same way (``to_halfhourly``).
* ``temp_smooth``   paper App. A, eqs. (17)-(18): one continuous half-hourly series,
  s_1 = tau_1 (first slot of 2013), s_k = (1-a) tau_k + a s_{k-1}, a = 0.998, and the
  daily value is the mean of the day's 48 smoothed values.  Bit-exact on the
  authors' temperatures when the daily mean is a pandas row mean (``daily_mean``).
* ``temp_smooth_n`` min-max of ``temp_smooth`` over the 365 days (bit-exact).
* ``pca_temp_0..2`` PCA (centred, NOT standardised) of the 365 x 48 matrix of
  half-hourly temperatures ONLY - the smoothed temperature is not included,
  contrary to the paper's text - fitted on all 365 days, scikit-learn sign
  convention, then min-max scaled with ONE global min and max taken over the whole
  365 x 3 score matrix (so only component 0 spans [0, 1]).  Max abs error 3.3e-16.
* ``pos_y``         daily (pandas row) mean of i / 17519, i = 0..17519, i.e. a
  half-hourly ramp from 0 (1 Jan 00:00) to 1 (31 Dec 23:30); bit-exact.  The
  i / 17520 variant is off by up to 5.7e-5.
* ``work_day``      1 Monday-Friday, 0 Saturday/Sunday, no bank-holiday adjustment
  (bit-exact; 261 work days in 2013).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

YEAR = 2013
SLOTS_PER_DAY = 48
N_DAYS = 365
N_SLOTS = N_DAYS * SLOTS_PER_DAY  # 17520
A_SMOOTH = 0.998
LONDON_CENTRAL = (51.51, -0.13)  # (lat, lon), roughly Charing Cross
HEATHROW = (51.48, -0.45)

TEMP_COLS = [f"temp_{h:02d}" for h in range(SLOTS_PER_DAY)]
PCA_COLS = [f"pca_temp_{k}" for k in range(3)]
COND_COLS = TEMP_COLS + ["temp_smooth", "temp_smooth_n"] + PCA_COLS + ["pos_y", "work_day"]


# --------------------------------------------------------------------------- ERA5 I/O
def load_era5_box(
    csv_path: str,
    lat_range=(51.0, 52.0),
    lon_range=(-1.0, 0.25),
    threads: int = 4,
    memory_limit: str = "8GB",
) -> pd.DataFrame:
    """Read the ERA5 CSV (columns time,latitude,longitude,number,step,surface,
    valid_time,d2m,t2m; t2m in Kelvin) with DuckDB, keeping only the grid points inside
    the box.  Returns ``valid_time`` (naive UTC), ``latitude``, ``longitude``,
    ``t2m_k`` and ``t2m_c`` (degC), sorted by time, lat, lon."""
    import duckdb

    con = duckdb.connect()
    try:
        con.execute(f"SET threads={int(threads)}; SET memory_limit='{memory_limit}'")
        path = str(csv_path).replace("'", "''")
        q = f"""
            SELECT CAST(valid_time AS TIMESTAMP) AS valid_time, latitude, longitude,
                   t2m AS t2m_k, t2m - 273.15 AS t2m_c
            FROM read_csv('{path}', header=true, columns={{
                'time':'VARCHAR','latitude':'DOUBLE','longitude':'DOUBLE',
                'number':'INTEGER','step':'VARCHAR','surface':'DOUBLE',
                'valid_time':'VARCHAR','d2m':'DOUBLE','t2m':'DOUBLE'}})
            WHERE latitude BETWEEN {float(lat_range[0])} AND {float(lat_range[1])}
              AND longitude BETWEEN {float(lon_range[0])} AND {float(lon_range[1])}
            ORDER BY valid_time, latitude, longitude
        """
        df = con.execute(q).df()
    finally:
        con.close()
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    return df


def era5_cube(df: pd.DataFrame):
    """(times, lats ascending, lons ascending, cube[time, lat, lon] in degC) from the
    long format of ``load_era5_box``.  Raises if the cube has holes, duplicates or an
    irregular time step."""
    if df.duplicated(["valid_time", "latitude", "longitude"]).any():
        raise ValueError("duplicated ERA5 records")
    times = pd.DatetimeIndex(np.sort(df["valid_time"].unique()))
    lats = np.sort(df["latitude"].unique())
    lons = np.sort(df["longitude"].unique())
    cube = np.full((len(times), len(lats), len(lons)), np.nan)
    cube[
        times.get_indexer(df["valid_time"]),
        np.searchsorted(lats, df["latitude"].to_numpy()),
        np.searchsorted(lons, df["longitude"].to_numpy()),
    ] = df["t2m_c"].to_numpy()
    if np.isnan(cube).any():
        raise ValueError("ERA5 cube has holes")
    step = np.diff(times.values).astype("timedelta64[m]").astype(np.int64)
    if not np.all(step == 60):
        raise ValueError("ERA5 series is not regular hourly")
    return times, lats, lons, cube


def _grid_idx(grid: np.ndarray, x: float) -> int:
    i = np.flatnonzero(np.isclose(grid, x))
    if len(i) != 1:
        raise KeyError(f"{x} is not a grid value")
    return int(i[0])


def _bracket(grid: np.ndarray, x: float):
    j = int(np.searchsorted(grid, x)) - 1
    j = min(max(j, 0), len(grid) - 2)
    w = (x - grid[j]) / (grid[j + 1] - grid[j])
    return j, float(w)


def point_series(lats, lons, cube, lat: float, lon: float, method: str):
    """Hourly series at a point.  ``method``: 'nearest' (nearest grid point, longitude
    distance scaled by cos(lat)), 'bilinear', or 'box2x2' (plain mean of the 4 grid
    points enclosing the point).  Returns (values, info dict)."""
    if method == "nearest":
        dlat = lats[:, None] - lat
        dlon = (lons[None, :] - lon) * np.cos(np.deg2rad(lat))
        ia, io = np.unravel_index(np.argmin(dlat**2 + dlon**2), (len(lats), len(lons)))
        return cube[:, ia, io], dict(method=method, grid_points=[[float(lats[ia]), float(lons[io])]])
    ja, wa = _bracket(lats, lat)
    jo, wo = _bracket(lons, lon)
    c00, c01 = cube[:, ja, jo], cube[:, ja, jo + 1]
    c10, c11 = cube[:, ja + 1, jo], cube[:, ja + 1, jo + 1]
    pts = [[float(lats[a]), float(lons[o])] for a in (ja, ja + 1) for o in (jo, jo + 1)]
    if method == "bilinear":
        v = (1 - wa) * ((1 - wo) * c00 + wo * c01) + wa * ((1 - wo) * c10 + wo * c11)
        return v, dict(method=method, grid_points=pts, w_lat=round(wa, 4), w_lon=round(wo, 4))
    if method == "box2x2":
        return (c00 + c01 + c10 + c11) / 4.0, dict(method=method, grid_points=pts)
    raise ValueError(method)


def box_mean(lats, lons, cube, lat_bounds, lon_bounds):
    """Plain mean of all grid points with lat in [lat_bounds] and lon in [lon_bounds]."""
    a0, a1 = _grid_idx(lats, lat_bounds[0]), _grid_idx(lats, lat_bounds[1])
    o0, o1 = _grid_idx(lons, lon_bounds[0]), _grid_idx(lons, lon_bounds[1])
    sub = cube[:, a0 : a1 + 1, o0 : o1 + 1]
    pts = [[float(lats[a]), float(lons[o])] for a in range(a0, a1 + 1) for o in range(o0, o1 + 1)]
    return sub.mean(axis=(1, 2)), dict(method="box_mean", grid_points=pts)


def to_halfhourly(hourly_values, hourly_times, *, slot_offset: float = 0, year: int = YEAR):
    """(365, 48) half-hourly matrix.  Slot i (start = 1 Jan 00:00 UTC + i * 30 min) takes
    the hourly series linearly interpolated at t_i + slot_offset * 30 min, so at offset 0
    the even slots are the hourly values and the odd slots the mid-points (the authors'
    construction).  Times outside the hourly record are clamped to the first/last value;
    the boolean (365, 48) mask of clamped slots is returned as well."""
    t0 = pd.Timestamp(f"{year}-01-01 00:00")
    xh = np.asarray((pd.DatetimeIndex(hourly_times) - t0) / pd.Timedelta(hours=1), dtype=float)
    xq = (np.arange(N_SLOTS) + float(slot_offset)) / 2.0
    vals = np.interp(xq, xh, np.asarray(hourly_values, dtype=float))
    clamped = (xq < xh[0]) | (xq > xh[-1])
    return vals.reshape(N_DAYS, SLOTS_PER_DAY), clamped.reshape(N_DAYS, SLOTS_PER_DAY)


# ------------------------------------------------------------------- derived variables
def smooth_series(x, a: float = A_SMOOTH, init: float | None = None) -> np.ndarray:
    """Exponential smoothing of paper eq. (17): s_1 = x_1, s_k = (1-a) x_k + a s_{k-1}.
    If ``init`` is given it is used as s_0 instead (variant, for sensitivity checks)."""
    x = np.asarray(x, dtype=float).ravel()
    out = np.empty_like(x)
    if init is None:
        prev = x[0]
        out[0] = x[0]
        start = 1
    else:
        prev = float(init)
        start = 0
    b = 1.0 - a
    for k in range(start, len(x)):
        prev = b * x[k] + a * prev
        out[k] = prev
    return out


def daily_mean(M: np.ndarray) -> np.ndarray:
    """Row mean computed like pandas ``DataFrame.mean(axis=1)`` (sequential summation);
    this is what makes ``temp_smooth`` and ``pos_y`` bit-identical to the authors'."""
    return pd.DataFrame(np.asarray(M, dtype=float)).mean(axis=1).to_numpy()


def smoothed_daily(T: np.ndarray, a: float = A_SMOOTH, init: float | None = None) -> np.ndarray:
    """Daily smoothed temperature (paper eq. 18) from a (n_days, 48) matrix."""
    T = np.asarray(T, dtype=float)
    return daily_mean(smooth_series(T.ravel(), a, init).reshape(T.shape))


def minmax(x: np.ndarray, axis=None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.min(x, axis=axis, keepdims=axis is not None)
    hi = np.max(x, axis=axis, keepdims=axis is not None)
    return (x - lo) / (hi - lo)


def pca_fit(X: np.ndarray, n_components: int = 3, standardize: bool = False, fit_rows=None):
    """Centred PCA by SVD (optionally standardised; optionally fitted on a subset of rows
    and applied to all rows).  Returns dict(scores, components, mean, scale, evr,
    evr_all).  Sign convention of scikit-learn < 1.5 (``svd_flip`` u-based: the largest
    |score| of each component on the fitted rows is positive), which is what the
    authors had in 2020; for their data the >= 1.5 (v-based) convention gives the
    same signs."""
    X = np.asarray(X, dtype=float)
    Xf = X if fit_rows is None else X[np.asarray(fit_rows)]
    mean = Xf.mean(axis=0)
    scale = Xf.std(axis=0) if standardize else np.ones(X.shape[1])
    U, S, Vt = np.linalg.svd((Xf - mean) / scale, full_matrices=False)
    sgn = np.sign(U[np.argmax(np.abs(U), axis=0), np.arange(U.shape[1])])
    sgn[sgn == 0] = 1.0
    Vt = Vt * sgn[:, None]
    ev = S**2 / (Xf.shape[0] - 1)
    evr_all = ev / ev.sum()
    k = n_components
    return dict(
        scores=((X - mean) / scale) @ Vt[:k].T,
        components=Vt[:k],
        mean=mean,
        scale=scale,
        evr=evr_all[:k],
        evr_all=evr_all,
    )


def align_signs(fit: dict, ref_components: np.ndarray) -> dict:
    """Flip components so that each has a positive dot product with the reference
    loadings (same input columns).  Adds ``flipped`` (list of bool)."""
    s = np.sign(np.sum(fit["components"] * np.asarray(ref_components), axis=1))
    s[s == 0] = 1.0
    out = dict(fit)
    out["components"] = fit["components"] * s[:, None]
    out["scores"] = fit["scores"] * s
    out["flipped"] = (s < 0).tolist()
    return out


def scale_global(scores: np.ndarray) -> np.ndarray:
    """The authors' rescaling of the PCA scores: ONE global min and max over the whole
    (n_days, 3) matrix."""
    s = np.asarray(scores, dtype=float)
    return (s - s.min()) / (s.max() - s.min())


def temperature_conditions(T: np.ndarray, ref_components=None, a: float = A_SMOOTH) -> dict:
    """The authors' recipe end-to-end from a (365, 48) temperature matrix: returns
    dict(temp_smooth, temp_smooth_n, pca (365, 3) scaled, fit).  If ``ref_components``
    (3, 48) is given, component signs are aligned to it before the global scaling."""
    T = np.asarray(T, dtype=float)
    ts = smoothed_daily(T, a)
    fit = pca_fit(T, 3)
    if ref_components is not None:
        fit = align_signs(fit, ref_components)
    return dict(temp_smooth=ts, temp_smooth_n=minmax(ts), pca=scale_global(fit["scores"]), fit=fit)


def calendar_vars(dates) -> tuple[np.ndarray, np.ndarray]:
    """pos_y (daily row mean of the half-hourly ramp i / (N - 1), N = half-hours in the
    year) and work_day (1 Mon-Fri, no holiday adjustment), for any list of dates."""
    d = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    pos_y = np.empty(len(d))
    for y in np.unique(d.year):
        n_days = 366 if pd.Timestamp(f"{y}-12-31").dayofyear == 366 else 365
        n = n_days * SLOTS_PER_DAY
        daily = daily_mean((np.arange(n) / (n - 1)).reshape(n_days, SLOTS_PER_DAY))
        sel = np.asarray(d.year == y)
        pos_y[sel] = daily[np.asarray(d.dayofyear[sel]) - 1]
    work_day = (np.asarray(d.dayofweek) < 5).astype(np.float64)
    return pos_y, work_day


# --------------------------------------------------------------------- authors' values
def authors_conditions(feather_path: str) -> pd.DataFrame:
    """The authors' daily conditions (365 rows; raises if they differ between clusters):
    date, temp_00..47, temp_smooth, temp_smooth_n, pca_temp_0..2, pos_y, work_day."""
    df = pd.read_feather(feather_path)
    df["date"] = pd.to_datetime(df["date"])
    if "cluster" in df.columns:
        groups = [g.sort_values("date") for _, g in df.groupby("cluster")]
        ref = groups[0][COND_COLS].to_numpy(dtype=float)
        for g in groups[1:]:
            if not np.array_equal(g[COND_COLS].to_numpy(dtype=float), ref):
                raise ValueError("condition columns differ between clusters")
        df = groups[0]
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date"] + COND_COLS].copy()


# ------------------------------------------------------------------------- comparisons
def compare_series(ours, ref, mask=None) -> dict:
    """RMSE, bias (ours - ref), MAE, max |error|, Pearson r on the (optionally masked)
    values."""
    a = np.asarray(ours, dtype=float).ravel()
    b = np.asarray(ref, dtype=float).ravel()
    if mask is not None:
        m = np.asarray(mask, dtype=bool).ravel()
        a, b = a[m], b[m]
    d = a - b
    return dict(
        rmse=float(np.sqrt(np.mean(d**2))),
        bias=float(np.mean(d)),
        rmse_centred=float(np.sqrt(np.mean((d - d.mean()) ** 2))),
        mae=float(np.mean(np.abs(d))),
        max_abs=float(np.max(np.abs(d))),
        corr=float(np.corrcoef(a, b)[0, 1]),
        n=int(a.size),
    )


def r2_per_component(P: np.ndarray, S: np.ndarray) -> list[float]:
    """Squared Pearson correlation between matching columns (sign/affine-free)."""
    return [float(np.corrcoef(P[:, k], S[:, k])[0, 1] ** 2) for k in range(P.shape[1])]


def r2_rotation_free(P: np.ndarray, S: np.ndarray) -> list[float]:
    """R^2 of the OLS regression of each column of P on all columns of S plus an
    intercept (free of sign flips, rotations and affine rescaling)."""
    X = np.c_[np.ones(len(S)), S]
    out = []
    for k in range(P.shape[1]):
        beta, *_ = np.linalg.lstsq(X, P[:, k], rcond=None)
        r = P[:, k] - X @ beta
        out.append(float(1.0 - r.var() / P[:, k].var()))
    return out
