"""Rebuild of the 2013 Low Carbon London (LCL) household data set used by
Bregere & Bessa (2020), plus helpers to compare it with the authors' prepared
data (UPSTREAM/data/*.feather) and with Tariffs.xlsx.

Conventions
-----------
* "LCL grid": index i = 0..17519  <->  LCL ``DateTime`` 2013-01-01 00:00 + i*30 min
  (LCL timestamps are GMT: no DST gap or duplication; checked in the build stats).
  Two edge slots are kept separately: i = -1 (2012-12-31 23:30) and i = 17520
  (2014-01-01 00:00), so that one-slot label shifts can be evaluated on the first and
  last day ("edges" arrays, column 0 = i=-1, column 1 = i=17520).
* "authors' frame": (day d = 0..364, slot h = 0..47) = column ``consumption_hh`` /
  ``low_hh`` / ``high_hh`` of the authors' files.  With slot-mapping shift ``s``,
  authors' (d, h)  <->  LCL grid index 48*d + h + s.  The empirically determined
  ``s`` is recorded in outputs/data_checks.json (``slot_mapping``).
* Tariff codes: 0 = Normal, 1 = Low, 2 = High.

Nothing in this module imports or executes code from the upstream repository; the
authors' ``.feather`` files are only read as data by absolute path.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- paths
# Portable paths (MVP integration, 2026-11-14): repository-relative; raw data are looked up in
# $CARBON_RAW_DIR, else the folder that contains the repository (raw files are never copied or modified).
REPRO = Path(__file__).resolve().parents[1]
PROJECT = REPRO.parent                                   # repository root
RAW_DIR = Path(os.environ.get("CARBON_RAW_DIR", PROJECT.parent))
DATA = REPRO / "data"
OUT = REPRO / "outputs"
FIG = OUT / "figures"
UPSTREAM = Path(os.environ.get("CVAE_UPSTREAM", REPRO / "data" / "authors_upstream_not_present"))
UP_DATA = DATA / "authors"  # copy of the authors' repo data/*.feather (byte-identical; read as data only); the clone above may be gone


def _first_existing(*cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return Path(cands[0])


RAW_CSV = _first_existing(RAW_DIR / "LCL_Data" / "CC_LCL-FullData.csv", RAW_DIR / "CC_LCL-FullData.csv",
                          PROJECT / "CC_LCL-FullData.csv")
TARIFFS_XLSX = _first_existing(RAW_DIR / "Tariffs.xlsx", PROJECT / "Tariffs.xlsx")
_SCRATCH = DATA / "_scratch"                             # gitignored

# output files of the data build (scripts/01_build_lcl2013.py)
LONG_PARQUET = DATA / "lcl2013_long.parquet"        # LCLid, group, ts, kwh (deduplicated, on-grid)
KWH_NPY = DATA / "lcl2013_kwh.npy"                  # (n_households, 17520) float32, NaN = missing
EDGES_NPY = DATA / "lcl2013_edges.npy"              # (n_households, 2): 2012-12-31 23:30, 2014-01-01 00:00
HOUSEHOLDS_PARQUET = DATA / "lcl2013_households.parquet"
FILLED_NPY = DATA / "lcl2013_kwh_filled.npy"        # (n_selected, 17520) float32, ToU rows first
EDGES_FILLED_NPY = DATA / "lcl2013_edges_filled.npy"  # (n_selected, 2) float32
SELECTION_PARQUET = DATA / "selection.parquet"
BUILD_STATS_JSON = DATA / "lcl2013_build_stats.json"
FILL_STATS_JSON = DATA / "lcl2013_fill_stats.json"

# ------------------------------------------------------------------------ constants
YEAR_START = pd.Timestamp("2013-01-01 00:00:00")
N_DAYS = 365
H = 48
N_SLOTS = N_DAYS * H                                 # 17520
W_EXT = N_SLOTS + 2                                  # LCL grid -1 .. 17520
DATES_2013 = pd.date_range("2013-01-01", periods=N_DAYS, freq="D")
SLOT_TS = pd.date_range(YEAR_START, periods=N_SLOTS, freq="30min")
TARIFF_CODES = {"Normal": 0, "Low": 1, "High": 2}
BST_FIRST_DAY = pd.Timestamp("2013-03-31")           # clocks forward 2013-03-31 01:00 GMT
BST_LAST_DAY = pd.Timestamp("2013-10-26")            # clocks back 2013-10-27 01:00 GMT


def season_of_day(day_index):
    """'BST' for 2013-03-31 .. 2013-10-26, 'GMT' otherwise (day_index 0 = 2013-01-01).

    The two change-over days are assigned whole: 03-31 to BST, 10-27 to GMT."""
    day_index = np.asarray(day_index)
    d0 = (BST_FIRST_DAY - YEAR_START).days          # 89
    d1 = (BST_LAST_DAY - YEAR_START).days           # 298
    return np.where((day_index >= d0) & (day_index <= d1), "BST", "GMT")


def work_dir() -> Path:
    """Scratch directory for large intermediates: $LCL_WORK, else CVAE_reproduction/data/_scratch (gitignored)."""
    env = os.environ.get("LCL_WORK")
    if env:
        p = Path(env)
    else:
        p = _SCRATCH / "lcl_work"
    p.mkdir(parents=True, exist_ok=True)
    return p


def duckdb_connect(threads: int = 4, memory_limit: str = "8GB"):
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit='{memory_limit}'")
    td = work_dir() / "duckdb_tmp"
    td.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{td.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def _sql_path(p) -> str:
    return Path(p).as_posix().replace("'", "''")


def csv_source(csv_path=RAW_CSV) -> str:
    """read_csv() clause for the LCL csv.  The 4th header is 'KWH/hh (per half hour) '
    (trailing space); we rename it to ``kwh`` and keep everything as text."""
    return (f"read_csv('{_sql_path(csv_path)}', header=true, delim=',', quote='\"', "
            "auto_detect=false, columns={'LCLid':'VARCHAR','stdorToU':'VARCHAR',"
            "'DateTime':'VARCHAR','kwh':'VARCHAR'})")


# ------------------------------------------------------- step 1a: csv -> parquet
def csv_to_parquet(csv_path=RAW_CSV, out_parquet=None, threads=4, memory_limit="8GB") -> Path:
    """ONE streaming scan of the raw csv -> typed parquet (all years, all rows).

    Columns: LCLid, grp (stdorToU as written), ts (TIMESTAMP parsed from the first 19
    characters, NULL if unparseable), frac_ok (DateTime is 27 chars and ends '.0000000'),
    is_null_str (value == 'Null'), kwh (DOUBLE, NULL if not numeric)."""
    out = Path(out_parquet) if out_parquet else work_dir() / "lcl_full.parquet"
    tmp = out.with_suffix(".tmp.parquet")
    con = duckdb_connect(threads, memory_limit)
    con.execute(f"""COPY (
        SELECT LCLid, stdorToU AS grp,
               try_strptime(substr(DateTime, 1, 19), '%Y-%m-%d %H:%M:%S') AS ts,
               (length(DateTime) = 27 AND substr(DateTime, 20) = '.0000000') AS frac_ok,
               (trim(kwh) = 'Null') AS is_null_str,
               TRY_CAST(trim(kwh) AS DOUBLE) AS kwh
        FROM {csv_source(csv_path)}
    ) TO '{_sql_path(tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)""")
    con.close()
    os.replace(tmp, out)
    return out


ON_GRID_SQL = "(ts IS NOT NULL AND frac_ok AND second(ts) = 0 AND minute(ts) IN (0, 30))"
WINDOW_SQL = ("(ts >= TIMESTAMP '2012-12-31 23:30:00' AND ts <= TIMESTAMP '2014-01-01 00:00:00')")


# ------------------------------------------- step 1b: labels + window statistics
def full_file_stats(full_parquet, threads=4, memory_limit="8GB") -> tuple[dict, pd.DataFrame]:
    """Whole-file statistics and per (LCLid, label) summary."""
    con = duckdb_connect(threads, memory_limit)
    fp = _sql_path(full_parquet)
    q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
    st: dict = {}
    labels = con.execute(f"""
        SELECT LCLid, grp, count(*) AS n_rows_all, min(ts) AS first_ts_all, max(ts) AS last_ts_all
        FROM read_parquet('{fp}') GROUP BY ALL ORDER BY 1, 2""").df()
    st["full_file_rows"] = int(labels["n_rows_all"].sum())
    st["full_file_households"] = int(labels["LCLid"].nunique())
    st["full_file_households_per_label"] = {str(k): int(v) for k, v in
                                            labels.groupby("grp")["LCLid"].nunique().items()}
    nlab = labels.groupby("LCLid")["grp"].nunique()
    st["households_with_both_labels_full_file"] = sorted(nlab[nlab > 1].index.astype(str).tolist())
    st["lclid_bad_format"] = int((~labels["LCLid"].astype(str).str.fullmatch(r"MAC\d{6}")).sum())
    st["unparseable_timestamps"] = int(q(f"SELECT count(*) FROM read_parquet('{fp}') WHERE ts IS NULL"))
    st["rows_per_year"] = {str(int(r[0])): int(r[1]) for r in con.execute(
        f"SELECT year(ts), count(*) FROM read_parquet('{fp}') WHERE ts IS NOT NULL GROUP BY 1 ORDER BY 1").fetchall()}
    st["off_grid_rows_full_file"] = int(q(f"SELECT count(*) FROM read_parquet('{fp}') WHERE NOT {ON_GRID_SQL}"))
    st["off_grid_rows_full_file_numeric"] = int(q(
        f"SELECT count(*) FROM read_parquet('{fp}') WHERE NOT {ON_GRID_SQL} AND kwh IS NOT NULL"))
    con.close()
    return st, labels


def window_rows(full_parquet, threads=4, memory_limit="8GB"):
    """On-grid rows with LCL grid index -1..17520 -> (households DataFrame, hid, k, kwh)
    plus window statistics (off-grid rows, Null strings, label consistency, DST check).

    ``hid`` indexes the households DataFrame (sorted by LCLid); ``k`` is the LCL grid
    index (-1 .. 17520); ``kwh`` float64 with NaN for 'Null'/non-numeric."""
    con = duckdb_connect(threads, memory_limit)
    fp = _sql_path(full_parquet)
    q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
    st: dict = {}
    base = f"read_parquet('{fp}') WHERE {WINDOW_SQL}"
    st["window_rows_total"] = int(q(f"SELECT count(*) FROM {base}"))
    st["window_off_grid_rows"] = int(q(f"SELECT count(*) FROM {base} AND NOT {ON_GRID_SQL}"))
    st["window_off_grid_rows_numeric"] = int(q(f"SELECT count(*) FROM {base} AND NOT {ON_GRID_SQL} AND kwh IS NOT NULL"))
    st["window_off_grid_households"] = int(q(f"SELECT count(DISTINCT LCLid) FROM {base} AND NOT {ON_GRID_SQL}"))
    st["window_off_grid_examples"] = [str(r[0]) for r in con.execute(
        f"SELECT DISTINCT ts FROM {base} AND NOT {ON_GRID_SQL} ORDER BY ts LIMIT 10").fetchall()]
    st["window_on_grid_rows"] = int(q(f"SELECT count(*) FROM {base} AND {ON_GRID_SQL}"))
    st["window_on_grid_null_string_rows"] = int(q(f"SELECT count(*) FROM {base} AND {ON_GRID_SQL} AND is_null_str"))
    st["window_on_grid_unparseable_non_null"] = int(q(
        f"SELECT count(*) FROM {base} AND {ON_GRID_SQL} AND NOT is_null_str AND kwh IS NULL"))
    st["window_negative_values"] = int(q(f"SELECT count(*) FROM {base} AND {ON_GRID_SQL} AND kwh < 0"))
    st["window_max_value"] = float(q(f"SELECT max(kwh) FROM {base} AND {ON_GRID_SQL}"))
    st["window_mean_value"] = float(q(f"SELECT avg(kwh) FROM {base} AND {ON_GRID_SQL}"))
    # rows per timestamp around the DST changes (all households, raw on-grid rows incl.
    # duplicates): a local-time file would show a hole on 2013-03-31 01:00/01:30 and
    # doubled counts on 2013-10-27 01:00/01:30
    st["rows_per_ts_around_dst"] = {str(r[0]): int(r[1]) for r in con.execute(f"""
        SELECT ts, count(*) FROM {base} AND {ON_GRID_SQL}
          AND ((ts BETWEEN TIMESTAMP '2013-03-30 23:00' AND TIMESTAMP '2013-03-31 03:00')
            OR (ts BETWEEN TIMESTAMP '2013-10-26 22:00' AND TIMESTAMP '2013-10-27 03:00'))
        GROUP BY 1 ORDER BY 1""").fetchall()}
    hh = con.execute(f"""
        SELECT LCLid, string_agg(DISTINCT grp, '|' ORDER BY grp) AS "group",
               count(DISTINCT grp) AS n_labels_window,
               count(*) FILTER (WHERE NOT {ON_GRID_SQL}) AS n_offgrid_rows_window
        FROM {base} GROUP BY 1 ORDER BY 1""").df()
    hh = hh.sort_values("LCLid").reset_index(drop=True)
    st["window_households_with_both_labels"] = hh.loc[hh["n_labels_window"] > 1, "LCLid"].astype(str).tolist()
    con.register("hh_ids", pd.DataFrame({"LCLid": hh["LCLid"].to_numpy(dtype=object),
                                         "hid": np.arange(len(hh), dtype=np.int32)}))
    arr = con.execute(f"""
        SELECT h.hid AS hid,
               CAST(date_diff('minute', TIMESTAMP '2013-01-01 00:00:00', f.ts) // 30 AS INTEGER) AS k,
               coalesce(f.kwh, CAST('NaN' AS DOUBLE)) AS v
        FROM read_parquet('{fp}') f JOIN hh_ids h USING (LCLid)
        WHERE f.ts >= TIMESTAMP '2012-12-31 23:30:00' AND f.ts <= TIMESTAMP '2014-01-01 00:00:00'
          AND f.frac_ok AND second(f.ts) = 0 AND minute(f.ts) IN (0, 30)""").fetchnumpy()
    con.close()
    hid = np.asarray(arr["hid"], dtype=np.int64)
    k = np.asarray(arr["k"], dtype=np.int64)
    v = np.asarray(arr["v"], dtype=np.float64)
    assert len(hid) == st["window_on_grid_rows"], (len(hid), st["window_on_grid_rows"])
    assert k.min() >= -1 and k.max() <= N_SLOTS, (k.min(), k.max())
    return hh, hid, k, v, st


# ------------------------------------------------ step 1c: dedup + dense matrix
def dedup_dense(hid, k, v, n_households):
    """Average duplicated (household, slot) rows (mean of the non-null values; NaN if all
    are null) and build the dense extended grid (n, 17522) covering LCL index -1..17520.

    Returns (mean float64 (n, W_EXT), n_rows int32 (n, W_EXT), stats dict, dup_detail
    DataFrame of the duplicated (household, slot) groups)."""
    n = int(n_households)
    flat = hid * W_EXT + (k + 1)
    cnt = np.bincount(flat, minlength=n * W_EXT)
    valid = ~np.isnan(v)
    cntv = np.bincount(flat[valid], minlength=n * W_EXT)
    sumv = np.bincount(flat[valid], weights=v[valid], minlength=n * W_EXT)
    mean = np.full(n * W_EXT, np.nan)
    ok = cntv > 0
    mean[ok] = sumv[ok] / cntv[ok]
    del sumv
    st: dict = {}
    dupflat = np.flatnonzero(cnt > 1)
    st["dup_groups"] = int(len(dupflat))
    st["dup_extra_rows"] = int((cnt[dupflat] - 1).sum())
    st["dup_max_multiplicity"] = int(cnt.max()) if len(cnt) else 0
    sel = np.isin(flat, dupflat)
    d = pd.DataFrame({"flat": flat[sel], "v": v[sel]})
    g = d.groupby("flat")["v"]
    det = pd.DataFrame({"n_rows": g.size(), "n_valid": g.count(), "vmin": g.min(), "vmax": g.max(),
                        "vmean": g.mean()}).reset_index()
    det["hid"] = (det["flat"] // W_EXT).astype(np.int64)
    det["k"] = (det["flat"] % W_EXT - 1).astype(np.int64)
    det["ts"] = YEAR_START + pd.to_timedelta(det["k"] * 30, unit="min")
    st["dup_groups_with_conflicting_values"] = int((det["vmax"] > det["vmin"]).sum())
    st["dup_groups_mixed_null_and_value"] = int(((det["n_valid"] > 0) & (det["n_valid"] < det["n_rows"])).sum())
    st["dup_groups_all_null"] = int((det["n_valid"] == 0).sum())
    st["dup_max_abs_spread_kwh"] = float((det["vmax"] - det["vmin"]).max()) if len(det) else 0.0
    st["dup_households"] = int(det["hid"].nunique())
    st["dup_top_timestamps"] = {str(t): int(c) for t, c in det["ts"].value_counts().head(10).items()}
    st["dup_distinct_timestamps"] = int(det["ts"].nunique())
    return mean.reshape(n, W_EXT), cnt.reshape(n, W_EXT).astype(np.int32), st, det


def write_long_parquet(hh: pd.DataFrame, mean_ext, cnt_ext, out_parquet=LONG_PARQUET, chunk_households=500):
    """Write the deduplicated long table (one row per household x on-grid timestamp that
    had >= 1 raw row; kwh NULL where all rows were 'Null'), ordered by LCLid, ts."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    schema = pa.schema([("LCLid", pa.string()), ("group", pa.string()),
                        ("ts", pa.timestamp("us")), ("kwh", pa.float64())])
    ids = hh["LCLid"].to_numpy(dtype=object)
    grp = hh["group"].to_numpy(dtype=object)
    ts_ext = (YEAR_START + pd.to_timedelta((np.arange(W_EXT) - 1) * 30, unit="min")).to_numpy(
        dtype="datetime64[us]")
    n_written = 0
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        for a in range(0, len(hh), chunk_households):
            b = min(a + chunk_households, len(hh))
            c = cnt_ext[a:b]
            r, s = np.nonzero(c > 0)
            vals = mean_ext[a:b][r, s]
            tab = pa.table({
                "LCLid": pa.array(ids[a + r], type=pa.string()),
                "group": pa.array(grp[a + r], type=pa.string()),
                "ts": pa.array(ts_ext[s], type=pa.timestamp("us")),
                "kwh": pa.array(vals, mask=np.isnan(vals), type=pa.float64()),
            }, schema=schema)
            w.write_table(tab, row_group_size=1_000_000)
            n_written += len(r)
    os.replace(tmp, out)
    return n_written


def households_table(hh: pd.DataFrame, mean_ext, cnt_ext, labels_full: pd.DataFrame) -> pd.DataFrame:
    """Per household: row, LCLid, group, n_rows_2013, n_valid_2013, completeness,
    first/last valid ts (2013), duplicates, off-grid rows, edge validity, full-file span."""
    X = mean_ext[:, 1:N_SLOTS + 1]
    C = cnt_ext[:, 1:N_SLOTS + 1]
    valid = ~np.isnan(X)
    nv = valid.sum(axis=1)
    first = np.where(nv > 0, valid.argmax(axis=1), -1)
    last = np.where(nv > 0, N_SLOTS - 1 - valid[:, ::-1].argmax(axis=1), -1)
    out = pd.DataFrame({
        "row": np.arange(len(hh), dtype=np.int32),
        "LCLid": hh["LCLid"].to_numpy(dtype=object),
        "group": hh["group"].to_numpy(dtype=object),
        "n_rows_2013": (C > 0).sum(axis=1).astype(np.int32),
        "n_raw_rows_2013": C.sum(axis=1).astype(np.int32),
        "n_valid_2013": nv.astype(np.int32),
    })
    out["completeness"] = out["n_valid_2013"] / N_SLOTS
    out["completeness_rows"] = out["n_rows_2013"] / N_SLOTS
    out["first_valid_ts"] = pd.to_datetime(np.where(first >= 0, SLOT_TS.to_numpy()[np.maximum(first, 0)],
                                                    np.datetime64("NaT")))
    out["last_valid_ts"] = pd.to_datetime(np.where(last >= 0, SLOT_TS.to_numpy()[np.maximum(last, 0)],
                                                   np.datetime64("NaT")))
    out["n_dup_slots_2013"] = (C > 1).sum(axis=1).astype(np.int32)
    out["n_offgrid_rows_window"] = hh["n_offgrid_rows_window"].to_numpy().astype(np.int32)
    out["edge_prev_valid"] = ~np.isnan(mean_ext[:, 0])
    out["edge_next_valid"] = ~np.isnan(mean_ext[:, -1])
    span = labels_full.groupby("LCLid").agg(n_rows_all=("n_rows_all", "sum"), first_ts_all=("first_ts_all", "min"),
                                            last_ts_all=("last_ts_all", "max")).reset_index()
    span["LCLid"] = span["LCLid"].astype(str)
    out["LCLid"] = out["LCLid"].astype(str)
    out = out.merge(span, on="LCLid", how="left")
    out["group"] = out["group"].astype(str)
    assert (out["row"].to_numpy() == np.arange(len(out))).all()
    return out


# ------------------------------------------------------------- step 2: gap filling
def gap_runs(valid_row):
    """Maximal runs of missing slots in one boolean 'valid' row -> (start, length) arrays."""
    m = (~np.asarray(valid_row)).astype(np.int8)
    dm = np.diff(np.concatenate([[0], m, [0]]))
    s = np.flatnonzero(dm == 1)
    e = np.flatnonzero(dm == -1)
    return s, e - s


AUTHORS_SHORT_MAX = 1
AUTHORS_LONG_RULE = "linear_days"


def fill_gaps(X, short_max: int = AUTHORS_SHORT_MAX, long_rule: str = AUTHORS_LONG_RULE, chunk: int = 400):
    """Paper Sec. III gap filling on the LCL grid (n, 17520).

    * gaps (maximal runs of missing slots) of length <= ``short_max`` that have an
      observed value on both sides: linear interpolation in time between those two values;
    * every other missing slot (long gaps, or gaps touching the start/end of the year), at
      day d and half-hour h, from the nearest preceding day p and nearest following day q
      that have an OBSERVED value at half-hour h:
        - ``long_rule='linear_days'``: linear interpolation across days,
          x[p,h] + (x[q,h] - x[p,h]) * (d - p) / (q - p)   (= their mean when p, q = d -+ 1);
        - ``long_rule='mean_nearest'``: the mean of x[p,h] and x[q,h];
      only one side when the other does not exist (year edges).
    Defaults (short_max=1, 'linear_days') = the authors' rule, reverse-engineered from their
    all-ToU aggregate (scripts/02_lcl_checks.py, outputs/data_checks.json 'gap_fill_rule').
    The orchestration spec's variant is short_max=4, 'mean_nearest'.
    Returns (filled float32, stats dict)."""
    assert long_rule in ("linear_days", "mean_nearest"), long_rule
    X = np.asarray(X)
    n, T = X.shape
    assert T == N_SLOTS
    out = np.empty((n, T), dtype=np.float32)
    n_short = n_long = n_left = 0
    n_long_one_side = 0
    run_hist: dict[int, int] = {}
    idx = np.arange(T)
    days = np.arange(N_DAYS)
    for a in range(0, n, chunk):
        x = X[a:a + chunk].astype(np.float64)
        k = x.shape[0]
        valid = ~np.isnan(x)
        miss = ~valid
        prev = np.maximum.accumulate(np.where(valid, idx, -1), axis=1)
        nxt = np.minimum.accumulate(np.where(valid, idx, T)[:, ::-1], axis=1)[:, ::-1]
        gap_len = nxt - prev - 1
        # run-length histogram (each run counted once, at its first slot)
        first = miss & (prev == idx - 1)
        for L, c in zip(*np.unique(np.minimum(gap_len[first], 49), return_counts=True)):
            run_hist[int(L)] = run_hist.get(int(L), 0) + int(c)
        short = miss & (prev >= 0) & (nxt < T) & (gap_len <= short_max)
        y = x.copy()
        rr, cc = np.nonzero(short)
        p = prev[rr, cc]
        q = nxt[rr, cc]
        w = (cc - p) / (q - p)
        y[rr, cc] = x[rr, p] * (1.0 - w) + x[rr, q] * w
        n_short += len(rr)
        longm = miss & ~short
        if longm.any():
            xd = x.reshape(k, N_DAYS, H)
            vd = valid.reshape(k, N_DAYS, H)
            pdy = np.maximum.accumulate(np.where(vd, days[None, :, None], -1), axis=1)
            ndy = np.minimum.accumulate(np.where(vd, days[None, :, None], N_DAYS)[:, ::-1, :], axis=1)[:, ::-1, :]
            li, ld, lh = np.nonzero(longm.reshape(k, N_DAYS, H))
            p_d = pdy[li, ld, lh]
            n_d = ndy[li, ld, lh]
            vp = np.where(p_d >= 0, xd[li, np.clip(p_d, 0, N_DAYS - 1), lh], np.nan)
            vn = np.where(n_d < N_DAYS, xd[li, np.clip(n_d, 0, N_DAYS - 1), lh], np.nan)
            both = np.vstack([vp, vn])
            cnt_ = (~np.isnan(both)).sum(axis=0)
            fill = np.where(cnt_ > 0, np.nansum(both, axis=0) / np.maximum(cnt_, 1), np.nan)
            if long_rule == "linear_days":
                two = cnt_ == 2
                wd = (ld[two] - p_d[two]) / (n_d[two] - p_d[two])
                fill[two] = vp[two] + (vn[two] - vp[two]) * wd
            y.reshape(k, N_DAYS, H)[li, ld, lh] = fill
            n_long += len(li)
            n_long_one_side += int((cnt_ == 1).sum())
            n_left += int(np.isnan(fill).sum())
        out[a:a + k] = y.astype(np.float32)
    stats = {"short_max_slots": short_max, "long_rule": long_rule, "filled_short_linear": int(n_short),
             "filled_long_sameslot_days": int(n_long), "filled_long_one_side_only": int(n_long_one_side),
             "left_missing": int(n_left),
             "gap_runs_by_length_(49=>=49)": {str(k_): v for k_, v in sorted(run_hist.items())}}
    return out, stats


def fill_edges(E, filled):
    """Fill NaN edge values (2012-12-31 23:30, 2014-01-01 00:00) with the same half-hour of the
    adjacent 2013 day (2013-01-01 23:30, resp. 2013-12-31 00:00) of the filled matrix (the
    long-gap rule with one side at the year edges)."""
    E = np.array(E, dtype=np.float32, copy=True)
    m0 = np.isnan(E[:, 0])
    E[m0, 0] = filled[m0, 47]
    m1 = np.isnan(E[:, 1])
    E[m1, 1] = filled[m1, N_SLOTS - 48]
    return E


def select_households(hh: pd.DataFrame, threshold: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """All ToU households with completeness > threshold, plus a random sample (numpy
    default_rng(seed), without replacement, from the LCLid-sorted list) of as many Std
    households with completeness > threshold.  ToU rows first, each block sorted by LCLid."""
    tou = hh[(hh["group"] == "ToU") & (hh["completeness"] > threshold)].sort_values("LCLid")
    std = hh[(hh["group"] == "Std") & (hh["completeness"] > threshold)].sort_values("LCLid")
    rng = np.random.default_rng(seed)
    if len(std) < len(tou):           # only possible on small test extracts
        print(f"WARNING: Std pool ({len(std)}) smaller than ToU set ({len(tou)}); taking all Std")
    draw = rng.choice(len(std), size=min(len(tou), len(std)), replace=False)
    order = np.argsort(draw)
    pick = draw[order]                                   # sorted positions in the LCLid-sorted pool
    std_sel = std.iloc[pick].copy()
    std_sel["std_draw_order"] = order.astype(np.int32)   # i-th household drawn (exchangeable)
    tou = tou.copy()
    tou["std_draw_order"] = np.int32(-1)
    sel = pd.concat([tou, std_sel], ignore_index=True)
    sel = sel[["LCLid", "group", "completeness", "row", "std_draw_order"]].rename(columns={"row": "row_in_dense"})
    sel.insert(2, "row_in_filled", np.arange(len(sel), dtype=np.int32))
    return sel


def best_slot_shift(F_tou, E_tou, auth, shifts=range(-2, 3)):
    """Shift s minimising mean |our all-ToU mean (authors' frame, shift s) - authors' aggregate|."""
    m = F_tou.mean(axis=0, keepdims=True)
    e = E_tou.mean(axis=0, keepdims=True)
    res = {}
    for s in shifts:
        ours = to_authors_frame(m, e, int(s))[0]
        ok = np.isfinite(ours)
        res[int(s)] = float(np.abs(ours[ok] - auth[ok]).mean())
    return min(res, key=res.get), res


def identify_authors_subset(X_tou_frame, auth, n_auth: int):
    """Which of our ToU households make up the authors' all-ToU aggregate?

    X_tou_frame: (n, 365, 48) raw (NaN = missing) consumption of our ToU households in the
    authors' frame; auth: (365, 48) authors' aggregate (mean over ``n_auth`` households).
    On slots where all our households are observed, R(t) = sum_i y_i(t) - n_auth * auth(t)
    equals the summed consumption of the households we have and the authors do not (if
    their set is a subset of ours).  Non-negative least squares of R on the household
    series then returns 0/1 weights.  Returns (weights (n,), diagnostics dict)."""
    from scipy.optimize import nnls

    n = X_tou_frame.shape[0]
    Xf = X_tou_frame.reshape(n, -1)
    a = np.asarray(auth, dtype=np.float64).ravel()
    full = ~np.isnan(Xf).any(axis=0) & np.isfinite(a)
    Y = Xf[:, full].T.astype(np.float64)
    R = Y.sum(axis=1) - n_auth * a[full]
    w, rnorm = nnls(Y, R, maxiter=50 * n)
    excl = w > 0.5
    keep_mean = Y[:, ~excl].mean(axis=1)
    diag = {
        "n_ours": int(n), "n_authors": int(n_auth), "fully_observed_slots_used": int(full.sum()),
        "nnls_residual_norm": float(rnorm), "sum_weights": float(w.sum()),
        "n_weight_gt_0.5": int(excl.sum()), "n_weight_in_(0.01,0.99)": int(((w > 0.01) & (w < 0.99)).sum()),
        "max_abs_weight_minus_round": float(np.abs(w - np.round(w)).max()),
        "subset_mean_vs_authors_max_abs_diff_kwh_fully_observed": float(np.abs(keep_mean - a[full]).max()),
        "subset_size": int((~excl).sum()),
    }
    return w, diag


# ------------------------------------------------------- authors' frame + data
def extend_with_edges(X, E):
    """(n, 17520) + (n, 2) -> (n, 17522) covering LCL grid indices -1 .. 17520."""
    return np.concatenate([E[:, :1], X, E[:, 1:2]], axis=1)


def to_authors_frame(X, E=None, shift: int = 0):
    """LCL grid (n, 17520) -> authors' frame (n, 365, 48):
    out[:, d, h] = value at LCL timestamp 2013-01-01 + d days + (h + shift)*30 min.
    Needs the edges E for shift = -1 / +1; slots outside the available range are NaN."""
    X = np.asarray(X)
    n = X.shape[0]
    ext = extend_with_edges(X, E) if E is not None else np.concatenate(
        [np.full((n, 1), np.nan, X.dtype), X, np.full((n, 1), np.nan, X.dtype)], axis=1)
    k = np.arange(N_SLOTS) + shift + 1          # index into ext
    ok = (k >= 0) & (k < ext.shape[1])
    out = np.full((n, N_SLOTS), np.nan, dtype=X.dtype)
    out[:, ok] = ext[:, k[ok]]
    return out.reshape(n, N_DAYS, H)


def cols(prefix: str):
    return [f"{prefix}_{h:02d}" for h in range(H)]


def _read_feather(name: str) -> pd.DataFrame:
    df = pd.read_feather(UP_DATA / f"{name}.feather")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_authors_tou_aggregate():
    """train_ToU + test_ToU, sorted by date -> (dates, consumption (365, 48), test mask)."""
    tr = _read_feather("train_ToU")
    te = _read_feather("test_ToU")
    tr["is_test"] = False
    te["is_test"] = True
    df = pd.concat([tr, te], ignore_index=True).sort_values("date").reset_index(drop=True)
    assert len(df) == N_DAYS and df["date"].is_unique
    assert (df["date"].to_numpy() == DATES_2013.to_numpy()).all()
    return df["date"], df[cols("consumption")].to_numpy(np.float64), df["is_test"].to_numpy()


def load_authors_clusters():
    """power_consump_nmf -> consumption (4, 365, 48), cluster sizes (4,)."""
    df = _read_feather("power_consump_nmf").sort_values(["cluster", "date"])
    cons = np.stack([df[df["cluster"] == c][cols("consumption")].to_numpy(np.float64) for c in range(4)])
    sizes = _read_feather("sum_up_nmf").sort_values("cluster")["number_clients"].to_numpy()
    return cons, sizes


def load_authors_tariff_codes(cluster: int = 0):
    """Authors' low_/high_ dummies (rows of one cluster of power_consump_nmf) -> codes (17520,)
    in the authors' frame; also reports whether all clusters carry identical dummies."""
    df = _read_feather("power_consump_nmf").sort_values(["cluster", "date"])
    lows = [df[df["cluster"] == c][cols("low")].to_numpy(np.float64) for c in range(4)]
    highs = [df[df["cluster"] == c][cols("high")].to_numpy(np.float64) for c in range(4)]
    same = all(np.array_equal(lows[0], l_) and np.array_equal(highs[0], h_) for l_, h_ in zip(lows, highs))
    low, high = lows[cluster], highs[cluster]
    assert set(np.unique(low)) <= {0.0, 1.0} and set(np.unique(high)) <= {0.0, 1.0}
    assert not ((low > 0) & (high > 0)).any()
    codes = np.zeros((N_DAYS, H), dtype=np.int8)
    codes[low > 0.5] = 1
    codes[high > 0.5] = 2
    return codes.ravel(), same


def load_tariffs_xlsx(path=TARIFFS_XLSX):
    """Tariffs.xlsx Sheet1 -> (timestamps as written, codes int8)."""
    df = pd.read_excel(path, sheet_name="Sheet1")
    ts = pd.to_datetime(df["TariffDateTime"])
    lab = df["Tariff"].astype(str).str.strip().to_numpy(dtype=object)
    codes = np.array([TARIFF_CODES[x] for x in lab], dtype=np.int8)
    assert len(ts) == N_SLOTS and (ts.to_numpy() == SLOT_TS.to_numpy()).all()
    return ts, codes


def tariff_candidate_gmt(xlsx_codes):
    """Candidate B: Tariffs.xlsx clock labels taken as GMT; label d + h*30min -> slot (d, h)."""
    return np.asarray(xlsx_codes, dtype=np.int8).copy()


def tariff_candidate_local(xlsx_ts, xlsx_codes):
    """Candidate C: Tariffs.xlsx clock labels taken as Europe/London local time, converted to
    GMT, then label (GMT) d + h*30min -> slot (d, h).  Equivalent: slot g gets the xlsx label
    whose timestamp equals the local clock time of GMT g.  The two non-existent local labels
    (2013-03-31 01:00, 01:30) are unused; the two ambiguous local labels (2013-10-27 01:00,
    01:30) are used for both of their GMT occurrences."""
    xlsx_ts = pd.DatetimeIndex(xlsx_ts)
    local_naive = SLOT_TS.tz_localize("UTC").tz_convert("Europe/London").tz_localize(None)
    pos = pd.Index(xlsx_ts).get_indexer(local_naive)
    codes = np.zeros(N_SLOTS, dtype=np.int8)              # local labels after 2013 -> Normal
    ok = pos >= 0
    codes[ok] = np.asarray(xlsx_codes)[pos[ok]]
    return codes


def shift_codes(codes, offset: int):
    """Move a schedule later in time by ``offset`` slots: out[k] = codes[k - offset] (Normal fill)."""
    codes = np.asarray(codes)
    out = np.zeros_like(codes)
    if offset > 0:
        out[offset:] = codes[:-offset]
    elif offset < 0:
        out[:offset] = codes[-offset:]
    else:
        out[:] = codes
    return out


# ---------------------------------------------------------------- event study
def runs(codes, value):
    """Maximal runs of ``codes == value``: arrays (start, end_exclusive)."""
    m = (np.asarray(codes) == value).astype(np.int8)
    dm = np.diff(np.concatenate([[0], m, [0]]))
    return np.flatnonzero(dm == 1), np.flatnonzero(dm == -1)


def event_matrix(r, idx, lags):
    """r: series (17520,), idx: event slots, lags: array -> (n_events, n_lags), NaN outside."""
    lags = np.asarray(lags)
    pos = np.asarray(idx)[:, None] + lags[None, :]
    ok = (pos >= 0) & (pos < len(r))
    out = np.full(pos.shape, np.nan)
    out[ok] = r[pos[ok]]
    return out


# ---------------------------------------------------------------- json helpers
def to_jsonable(o):
    """Recursively convert numpy / pandas scalars and arrays; NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(o))
    return o


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def timer():
    t0 = time.time()
    return lambda: f"{time.time() - t0:7.1f}s"
