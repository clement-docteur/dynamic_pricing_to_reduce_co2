"""Tariff dummies for the LCL dynamic Time-of-Use trial (2013).

Two sources:
  * our ``Tariffs.xlsx`` (Sheet1: ``TariffDateTime``, ``Tariff`` in {Normal, Low, High};
    17,520 rows on a regular 30-min grid covering 2013, no DST gap/duplicate);
  * the authors' prepared data (``power_consump_nmf.feather``: ``low_00..47``,
    ``high_00..47``, ``price_00..47``), read by absolute path only.

Slot convention: slot ``g`` of the year (0..17519) is the half-hour starting at
2013-01-01 00:00 GMT + g * 30 min; day ``d = g // 48``, column ``h = g % 48``.

``slot_offset`` convention (used everywhere in this module):
    result[g] = mapped_xlsx[g - slot_offset]
i.e. a positive offset moves every tariff event LATER by ``slot_offset`` half-hours
(slots shifted in from outside the year are filled with Normal).

``mode``:
  * ``'gmt'``   - the xlsx timestamps are taken as GMT: mapped_xlsx[g] = xlsx row g.
  * ``'local'`` - the xlsx timestamps are Europe/London clock time (GMT in winter,
    BST = GMT+1 from 2013-03-31 01:00 GMT to 2013-10-27 01:00 GMT) converted to GMT:
    mapped_xlsx[g] = xlsx row whose (naive) timestamp equals the local clock time of
    GMT slot g.  The two non-existent local half-hours of 2013-03-31 (01:00, 01:30)
    are never used and the two ambiguous local half-hours of 2013-10-27 (01:00, 01:30)
    are used twice; ``read_tariff_xlsx`` checks that all four are 'Normal', so this
    choice has no effect.

Relation found by ``scripts/03_conditions.py`` (outputs/conditions_checks.json):
the authors' dummies equal ``mode='local', slot_offset=+1`` (equivalently 'gmt' +1 in
the GMT period and 'gmt' -1 in the BST period) on every day of Jan, Feb, Apr and
Jun-Oct; the remaining differences are content (2013-03-29, 2013-05-03, Nov, Dec).
Which alignment is physically right is NOT decided here (it needs consumption data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

YEAR = 2013
SLOTS_PER_DAY = 48
N_DAYS = 365
N_SLOTS = N_DAYS * SLOTS_PER_DAY  # 17520
# Prices in GBP/kWh (paper: Low 3.99, Normal 11.76, High 67.20 p/kWh); same values as
# the authors' price_00..47 columns.
PRICES_GBP = {"Low": 0.0399, "Normal": 0.1176, "High": 0.672}
LABELS = ("Low", "Normal", "High")
VERSIONS = ("auth", "gmt", "loc")


def cols(prefix: str) -> list[str]:
    return [f"{prefix}_{h:02d}" for h in range(SLOTS_PER_DAY)]


def year_slots(year: int = YEAR) -> pd.DatetimeIndex:
    """Naive GMT start time of every half-hour slot of ``year`` (365-day years)."""
    return pd.date_range(pd.Timestamp(f"{year}-01-01 00:00"), periods=N_SLOTS, freq="30min")


def year_dates(year: int = YEAR) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", periods=N_DAYS, freq="D")


def local_clock(gmt_slots: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Naive Europe/London clock time of naive GMT timestamps."""
    return gmt_slots.tz_localize("UTC").tz_convert("Europe/London").tz_localize(None)


def bst_mask(gmt_slots: pd.DatetimeIndex) -> np.ndarray:
    """True where the (naive GMT) slot start is inside British Summer Time; cross-checked
    against the EU rule (last Sunday of March / October, 01:00 GMT)."""
    mask = np.asarray(local_clock(gmt_slots) != gmt_slots)
    y = gmt_slots[0].year
    mar, oct_ = pd.Timestamp(f"{y}-03-31"), pd.Timestamp(f"{y}-10-31")
    start = mar - pd.Timedelta(days=(mar.dayofweek + 1) % 7) + pd.Timedelta(hours=1)
    end = oct_ - pd.Timedelta(days=(oct_.dayofweek + 1) % 7) + pd.Timedelta(hours=1)
    rule = np.asarray((gmt_slots >= start) & (gmt_slots < end))
    if not np.array_equal(mask, rule):
        raise RuntimeError("tz database and EU DST rule disagree")
    return mask


def read_tariff_xlsx(xlsx_path: str) -> pd.DataFrame:
    """Read Sheet1 of Tariffs.xlsx and validate it: a regular 30-min grid of one full
    year, labels in {Low, Normal, High}, and 'Normal' on the 4 DST-ambiguous rows."""
    t = pd.read_excel(xlsx_path, sheet_name="Sheet1")
    t = t[["TariffDateTime", "Tariff"]].copy()
    t["TariffDateTime"] = pd.to_datetime(t["TariffDateTime"])
    t["Tariff"] = t["Tariff"].astype(str).str.strip()
    year = int(t["TariffDateTime"].iloc[0].year)
    if len(t) != N_SLOTS or not np.array_equal(
        t["TariffDateTime"].to_numpy(), year_slots(year).to_numpy()
    ):
        raise ValueError("Tariffs.xlsx is not a regular 30-min grid of one full year")
    bad = set(t["Tariff"].unique()) - set(LABELS)
    if bad:
        raise ValueError(f"unexpected tariff labels {bad}")
    step = np.diff(bst_mask(year_slots(year)).astype(int))
    on = int(np.flatnonzero(step == 1)[0]) + 1  # row of 2013-03-31 01:00 (local gap)
    off = int(np.flatnonzero(step == -1)[0]) + 1  # row of 2013-10-27 01:00 (local repeat)
    dst_rows = [on, on + 1, off, off + 1]
    if not (t["Tariff"].iloc[dst_rows] == "Normal").all():
        raise ValueError("a DST-ambiguous xlsx row is not Normal: 'local' mode is ambiguous")
    return t.reset_index(drop=True)


def mapped_labels(xlsx_path: str, mode: str = "gmt") -> np.ndarray:
    """Tariff label of each GMT slot of the year (object array, len 17520), offset 0."""
    t = read_tariff_xlsx(xlsx_path)
    lab = t["Tariff"].to_numpy(dtype=object)
    if mode == "gmt":
        return lab.copy()
    if mode != "local":
        raise ValueError("mode must be 'gmt' or 'local'")
    gmt = year_slots(int(t["TariffDateTime"].iloc[0].year))
    pos = pd.Index(t["TariffDateTime"]).get_indexer(local_clock(gmt))
    if (pos < 0).any():
        raise RuntimeError("local clock time not found in Tariffs.xlsx")
    return lab[pos]


def shift_labels(lab: np.ndarray, slot_offset: int, fill: str = "Normal") -> np.ndarray:
    """result[g] = lab[g - slot_offset]; positions shifted in from outside get ``fill``."""
    o = int(slot_offset)
    out = np.full(len(lab), fill, dtype=object)
    if o > 0:
        out[o:] = lab[:-o]
    elif o < 0:
        out[:o] = lab[-o:]
    else:
        out[:] = lab
    return out


def labels_to_arrays(lab: np.ndarray):
    """(low, high, price) float arrays of shape (365, 48) from a 17520 label vector."""
    lab = np.asarray(lab, dtype=object)
    low = (lab == "Low").astype(np.float64).reshape(N_DAYS, SLOTS_PER_DAY)
    high = (lab == "High").astype(np.float64).reshape(N_DAYS, SLOTS_PER_DAY)
    price = np.array([PRICES_GBP[x] for x in lab], dtype=np.float64).reshape(N_DAYS, SLOTS_PER_DAY)
    return low, high, price


def tariff_dummies(xlsx_path: str, mode: str = "gmt", slot_offset: int = 0):
    """Tariff dummies from our Tariffs.xlsx.

    Returns ``(dates, low, high, price)``: ``dates`` a DatetimeIndex of the 365 days,
    ``low``/``high`` float arrays (365, 48) of 0/1, ``price`` (365, 48) in GBP/kWh
    (0.0399 / 0.1176 / 0.672).  See the module docstring for ``mode``/``slot_offset``.
    """
    lab = shift_labels(mapped_labels(xlsx_path, mode), slot_offset)
    low, high, price = labels_to_arrays(lab)
    return year_dates(), low, high, price


def authors_dummies(feather_path: str):
    """The authors' tariff dummies (raises if they differ between clusters, if Low and
    High overlap, or if price is inconsistent with the dummies).

    Returns ``(dates, low, high, price)`` with the same shapes as ``tariff_dummies``.
    """
    df = pd.read_feather(feather_path)
    df["date"] = pd.to_datetime(df["date"])
    lc, hc, pc = cols("low"), cols("high"), cols("price")
    if "cluster" in df.columns:
        ref = first = None
        for _, g in df.groupby("cluster"):
            g = g.sort_values("date")
            a = g[lc + hc + pc].to_numpy(dtype=np.float64)
            if ref is None:
                ref, first = a, g
            elif not np.array_equal(a, ref):
                raise ValueError("tariff columns differ between clusters")
        df = first
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) != N_DAYS:
        raise ValueError(f"expected {N_DAYS} days, got {len(df)}")
    low = df[lc].to_numpy(dtype=np.float64)
    high = df[hc].to_numpy(dtype=np.float64)
    price = df[pc].to_numpy(dtype=np.float64)
    if ((low > 0.5) & (high > 0.5)).any():
        raise ValueError("Low and High overlap in the authors' dummies")
    expect = np.where(low > 0.5, PRICES_GBP["Low"], np.where(high > 0.5, PRICES_GBP["High"], PRICES_GBP["Normal"]))
    if not np.allclose(price, expect, rtol=0, atol=1e-12):
        raise ValueError("authors' price columns inconsistent with their dummies")
    return pd.DatetimeIndex(df["date"]), low, high, price


def dummies_to_labels(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    lab = np.full(np.asarray(low).size, "Normal", dtype=object)
    lab[np.asarray(low).ravel() > 0.5] = "Low"
    lab[np.asarray(high).ravel() > 0.5] = "High"
    return lab


def agreement(low_a, high_a, low_b, high_b, slot_mask=None) -> dict:
    """Agreement of two dummy sets (365 x 48) on the slots in ``slot_mask`` (17520 bool;
    all slots if None): share of identical labels, Jaccard index of the (slot, label)
    event pairs (Low and High pooled), counts, and event days (any event in either)
    whose profile is identical on the selected slots of that day."""
    la, ha = np.asarray(low_a).ravel() > 0.5, np.asarray(high_a).ravel() > 0.5
    lb, hb = np.asarray(low_b).ravel() > 0.5, np.asarray(high_b).ravel() > 0.5
    sel = np.ones(N_SLOTS, bool) if slot_mask is None else np.asarray(slot_mask, bool).ravel()
    same = (la == lb) & (ha == hb)
    # Jaccard over (slot, label) pairs; a Low-vs-High conflict counts twice in the union.
    inter = int((la & lb)[sel].sum() + (ha & hb)[sel].sum())
    union = int((la | lb)[sel].sum() + (ha | hb)[sel].sum())
    ev_a = ((la | ha) & sel).reshape(N_DAYS, SLOTS_PER_DAY).any(1)
    ev_b = ((lb | hb) & sel).reshape(N_DAYS, SLOTS_PER_DAY).any(1)
    same_day = (same | ~sel).reshape(N_DAYS, SLOTS_PER_DAY).all(1)
    return dict(
        slot_agreement=float(same[sel].mean()),
        event_jaccard=float(inter / union) if union else float("nan"),
        n_low_a=int(la[sel].sum()),
        n_low_b=int(lb[sel].sum()),
        n_high_a=int(ha[sel].sum()),
        n_high_b=int(hb[sel].sum()),
        event_days_a=int(ev_a.sum()),
        event_days_b=int(ev_b.sum()),
        event_days_union=int((ev_a | ev_b).sum()),
        event_days_identical=int(((ev_a | ev_b) & same_day).sum()),
    )


def period_masks(year: int = YEAR) -> dict:
    """Slot masks (17520 bool) of the calendar months and of the GMT / BST periods."""
    g = year_slots(year)
    bst = bst_mask(g)
    m = {f"{k:02d}": np.asarray(g.month == k) for k in range(1, 13)}
    m["GMT_period"] = ~bst
    m["GMT_Jan1-Mar30"] = ~bst & np.asarray(g < pd.Timestamp(f"{year}-04-01"))
    m["BST_period"] = bst
    m["year"] = np.ones(N_SLOTS, bool)
    m["Jan-Oct"] = np.asarray(g.month <= 10)
    return m


def alignment_table(xlsx_path: str, feather_path: str, offsets=range(-2, 3)) -> pd.DataFrame:
    """Long table: one row per (mode, offset, period) with ``agreement`` metrics of
    our xlsx dummies (b) against the authors' (a)."""
    _, la, ha, _ = authors_dummies(feather_path)
    masks = period_masks()
    rows = []
    for mode in ("gmt", "local"):
        base = mapped_labels(xlsx_path, mode)
        for o in offsets:
            lb, hb, _ = labels_to_arrays(shift_labels(base, o))
            for name, msk in masks.items():
                rows.append(dict(mode=mode, offset=int(o), period=name, **agreement(la, ha, lb, hb, msk)))
    return pd.DataFrame(rows)


def differing_days(low_a, high_a, low_b, high_b, dates=None) -> pd.DataFrame:
    """Days whose 48-slot profiles differ, with the 48-character profiles
    ('L'/'H'/'.') and the counts of disagreeing slots by (a, b) label pair."""
    dates = year_dates() if dates is None else pd.DatetimeIndex(dates)
    A = dummies_to_labels(low_a, high_a).reshape(N_DAYS, SLOTS_PER_DAY)
    B = dummies_to_labels(low_b, high_b).reshape(N_DAYS, SLOTS_PER_DAY)
    code = {"Low": "L", "High": "H", "Normal": "."}
    rows = []
    for d in range(N_DAYS):
        if np.array_equal(A[d], B[d]):
            continue
        diff = A[d] != B[d]
        pairs = pd.Series([f"{a}->{b}" for a, b in zip(A[d][diff], B[d][diff])]).value_counts().to_dict()
        rows.append(
            dict(
                date=str(dates[d].date()),
                profile_a="".join(code[x] for x in A[d]),
                profile_b="".join(code[x] for x in B[d]),
                n_slots_differ=int(diff.sum()),
                label_pairs_a_to_b={k: int(v) for k, v in pairs.items()},
            )
        )
    return pd.DataFrame(rows)


def select_tariff_version(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """Return a copy of ``conditions_ours`` with unsuffixed ``low_hh/high_hh/price_hh``
    aliased from one stored version: ``'auth'`` (authors' dummies), ``'gmt'`` or
    ``'loc'`` (our xlsx under the two time bases; see conditions_checks.json)."""
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}")
    new = {
        f"{k}_{h:02d}": df[f"{k}_{version}_{h:02d}"].to_numpy()
        for k in ("low", "high", "price")
        for h in range(SLOTS_PER_DAY)
    }
    keep = df.drop(columns=[c for c in new if c in df.columns])
    return pd.concat([keep, pd.DataFrame(new, index=df.index)], axis=1)
