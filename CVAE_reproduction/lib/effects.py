"""Tariff-scenario effects of the generators and data-based tariff responses (paper Sec. VII-C, Figs. 6 and 8).

Track-agnostic. Slot h = column consumption_hh / low_hh / high_hh of the authors' schema (spec section 4). Clock
convention (verified, REPRO_SPEC sections 6-7): slot h is the half-hour ENDING at h x 30 min GMT (slot 0 = 23:30-24:00
of the previous day), so a block of slots s..e covers (s - 1)/2 h to e/2 h GMT.

THE SCENARIO WINDOWS ARE DEFINED HERE (single source of truth; lib.gam_generator.SCENARIOS and every script import
LOW_WINDOW / HIGH_WINDOW / WINDOWS from this module):
  Low  = slots 9-19  (11 half-hours, endpoints inclusive) = 04:00-09:30 GMT
  High = slots 39-44 (6 half-hours, endpoints inclusive)  = 19:00-22:00 GMT
verified from the paper's Figs. 6/8 (REPRO_SPEC section 7; outputs/verify/v-paper/). The paper's text says
"04:30-09:30" / "19:30-22:00" because it labels slot h by the time h x 30 min. Phases 1-2 used 9-18 / 39-43 (off by
one at the end); those are kept in SUPERSEDED_WINDOWS only to check cached results.

Scenario effects (generated arrays, (N, D, 48) kWh, common random numbers across scenarios)
  per day d and slot h:  Delta[d, h] = mean_i S_scen[i, d, h] - mean_i S_Normal[i, d, h]
  region statistic    :  mean over the region's slots, then over days; % = 100 x that / the Normal mean over the
                         same slots and days; 95% CI = percentile bootstrap over test days (days resampled with
                         replacement, numerator and denominator together).
  Regions (scenario_regions): in_window = the window; side effects pre2/pre4 = 2/4 slots before, post2/post4 = 2/4
  slots after, clipped at the end of the day: Low pre2 7-8, pre4 5-8, post2 20-21, post4 20-23; High pre2 37-38,
  pre4 35-38, post2 45-46, post4 = 45-47 (only 3 slots remain in the day; the key stays 'post4'); rebound windows:
  after Low 20-31 ('rebound_20_31') and 32-47; around High 24-38 (before, 'rebound_pre_24_38') and 45-47 (after,
  'rebound_post_45_47', = post4); afternoon_24_34 = the High 'before' rebound window without the 4 pre-window slots;
  afternoon_24_31 = the Low rebound window 20-31 without the 4 post-window slots; daily energy = sum over 48.
  phase1_regions(): the coarser region set of scripts 10 / 30 (cvae_scenario_effects*.json/csv), built from the same
  windows: Low in-window, 3 slots before (6-8) / after (20-22), afternoon-evening 24-47; High in-window, 3 slots
  before (36-38) / after (45-47), afternoon 24-35.

Data-based reference (cluster aggregates, all days of the year)
  baseline(d, h) = mean of y(d', h) over all-Normal days d' (no Low/High slot) of the same day type (work_day)
                   with 0 < |d' - d| <= 14;  Delta(d, h) = y(d, h) - baseline(d, h).
  Events = maximal runs of consecutive Low (or High) half-hours in the continuous 365 x 48 series (runs may cross
  midnight). Lag profiles around the run start (lags -12..-1 before, 0..5 inside) and end (-5..0 inside, +1..+12
  after); a lag outside the run counts only if that half-hour is Normal (else masked). 95% CI = percentile
  bootstrap over event days (events grouped by the day of their start).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

H = 48
# ---- the paper's scenario windows (Figs. 6 and 8; REPRO_SPEC section 7), endpoints inclusive: single source of truth
LOW_WINDOW = list(range(9, 20))      # slots 9-19, 11 half-hours = 04:00-09:30 GMT
HIGH_WINDOW = list(range(39, 45))    # slots 39-44, 6 half-hours = 19:00-22:00 GMT
WINDOWS = {"Low": LOW_WINDOW, "High": HIGH_WINDOW}
# superseded phase-1/2 windows (off by one at the end). Use ONLY to check that cached results were made with them.
SUPERSEDED_WINDOWS = {"Low": list(range(9, 19)), "High": list(range(39, 44))}
CODE = {"Normal": 0, "Low": 1, "High": 2}


def span(slots) -> str:
    """'9-19' for slots 9..19 (endpoints inclusive)."""
    slots = list(slots)
    return f"{slots[0]}-{slots[-1]}" if len(slots) > 1 else f"{slots[0]}"


def _clock(minutes: int) -> str:
    m = int(minutes) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def slots_hours(slots) -> tuple[float, float]:
    """Physical interval (hours since 00:00 GMT of the day) covered by a block of slots: slot h = the half-hour
    ending at h/2 h, so slots s..e cover [(s - 1)/2, e/2]."""
    slots = list(slots)
    return (slots[0] - 1) / 2.0, slots[-1] / 2.0


def slots_clock(slots) -> str:
    """GMT clock interval covered by a block of slots, e.g. '04:00-09:30' for slots 9-19."""
    slots = list(slots)
    return f"{_clock(30 * (slots[0] - 1))}-{_clock(30 * slots[-1])}"


WINDOW_SPAN = {k: span(v) for k, v in WINDOWS.items()}            # {'Low': '9-19', 'High': '39-44'}
WINDOW_CLOCK = {k: slots_clock(v) for k, v in WINDOWS.items()}    # {'Low': '04:00-09:30', 'High': '19:00-22:00'}
WINDOW_HOURS = {k: slots_hours(v) for k, v in WINDOWS.items()}    # {'Low': (4.0, 9.5), 'High': (19.0, 22.0)}
WINDOW_TEXT = {k: f"slots {WINDOW_SPAN[k]} = {WINDOW_CLOCK[k]} GMT" for k in WINDOWS}

# region keys that carry window-dependent slot numbers (scenario_regions)
KEY_LOW_REBOUND = f"rebound_{LOW_WINDOW[-1] + 1}_31"                 # 'rebound_20_31'
KEY_HIGH_POST_REBOUND = f"rebound_post_{HIGH_WINDOW[-1] + 1}_{H - 1}"  # 'rebound_post_45_47'

# paper's claims, Sec. VII-C: (sign, text); None = no claim
CLAIMS = {
    ("Low", "in_window"): (+1, "direction: Low raises consumption inside the window"),
    ("High", "in_window"): (-1, "direction: High lowers consumption inside the window"),
    ("High", "pre2"): (-1, "side effect: the fall starts a little before the High window"),
    ("High", "pre4"): (-1, "side effect: the fall starts a little before the High window"),
    ("High", "post2"): (-1, "side effect: the fall continues a little after the High window"),
    ("High", "post4"): (-1, "side effect: the fall continues a little after the High window (slots "
                            f"{span(range(HIGH_WINDOW[-1] + 1, H))}: only {H - 1 - HIGH_WINDOW[-1]} slots remain in the "
                            f"day; = {KEY_HIGH_POST_REBOUND})"),
    ("Low", KEY_LOW_REBOUND): (-1, "rebound: after a morning Low, consumption drops in the afternoon"),
    ("Low", "rebound_32_47"): (-1, "rebound: after a morning Low, consumption drops in the evening"),
    ("High", "rebound_pre_24_38"): (+1, "rebound: a little increase in the afternoon with an evening High "
                                        "(window as specified; overlaps pre4)"),
    ("High", "afternoon_24_34"): (+1, "rebound: afternoon increase, window without the 4 pre-window slots"),
}


def scenario_regions(name: str) -> dict:
    """Regions of the paper scenario `name` ('Low' / 'High'); side regions are clipped at the end of the day (High
    post4 = 45-47 has 3 slots)."""
    w = WINDOWS[name]
    s, e = w[0], w[-1]
    reg = {"in_window": list(w),
           "pre2": [h for h in range(s - 2, s) if h >= 0], "pre4": [h for h in range(s - 4, s) if h >= 0],
           "post2": [h for h in range(e + 1, e + 3) if h < H], "post4": [h for h in range(e + 1, e + 5) if h < H],
           "outside_window": [h for h in range(H) if h not in w]}
    if name == "Low":
        reg[KEY_LOW_REBOUND] = list(range(e + 1, 32))        # after the window to 15:30 GMT
        reg["rebound_32_47"] = list(range(32, H))
        reg["afternoon_24_31"] = list(range(e + 5, 32))      # descriptive: the rebound window without post4
    else:
        reg["rebound_pre_24_38"] = list(range(24, s))
        reg[KEY_HIGH_POST_REBOUND] = list(range(e + 1, H))   # = post4 (clipped)
        reg["afternoon_24_34"] = list(range(24, s - 4))
    for k, v in reg.items():
        assert v and all(0 <= h < H for h in v), (name, k, v)
        m = _SPAN_KEY_RE.match(k)
        if m:   # region names that carry slot numbers must match their slots
            assert v == list(range(int(m.group(1)), int(m.group(2)) + 1)), (name, k, v)
    return reg


_SPAN_KEY_RE = re.compile(r"^[a-z_]+?_(\d+)_(\d+)$")


# ---- the coarser phase-1 region set of scripts 10 / 30 (cvae_scenario_effects*.json/csv), from the same windows
def _key(prefix: str, slots) -> str:
    slots = list(slots)
    return f"{prefix}_{slots[0]:02d}_{slots[-1]:02d}"


_P1_LOW_BEFORE = list(range(LOW_WINDOW[0] - 3, LOW_WINDOW[0]))                            # 6-8
_P1_LOW_AFTER = [h for h in range(LOW_WINDOW[-1] + 1, LOW_WINDOW[-1] + 4) if h < H]        # 20-22
_P1_HIGH_BEFORE = list(range(HIGH_WINDOW[0] - 3, HIGH_WINDOW[0]))                          # 36-38
_P1_HIGH_AFTER = [h for h in range(HIGH_WINDOW[-1] + 1, HIGH_WINDOW[-1] + 4) if h < H]     # 45-47
_P1_HIGH_AFTERNOON = list(range(24, _P1_HIGH_BEFORE[0]))                                   # 24-35
PHASE1_KEY = {("Low", "in_window"): _key("in_window", LOW_WINDOW),            # 'in_window_09_19'
              ("Low", "before"): _key("before", _P1_LOW_BEFORE),              # 'before_06_08'
              ("Low", "after"): _key("after", _P1_LOW_AFTER),                 # 'after_20_22'
              ("Low", "afternoon_evening"): "afternoon_evening_24_47",
              ("High", "in_window"): _key("in_window", HIGH_WINDOW),          # 'in_window_39_44'
              ("High", "before"): _key("before", _P1_HIGH_BEFORE),            # 'before_36_38'
              ("High", "after"): _key("after", _P1_HIGH_AFTER),               # 'after_45_47'
              ("High", "afternoon"): _key("afternoon", _P1_HIGH_AFTERNOON)}   # 'afternoon_24_35'
# direction claimed by the paper (Sec. VII-C) for each phase-1 region; regions not listed carry no claim
PHASE1_CLAIMED_SIGN = {("Low", PHASE1_KEY["Low", "in_window"]): +1,
                       ("Low", PHASE1_KEY["Low", "afternoon_evening"]): -1,
                       ("High", PHASE1_KEY["High", "in_window"]): -1,
                       ("High", PHASE1_KEY["High", "before"]): -1,
                       ("High", PHASE1_KEY["High", "after"]): -1,
                       ("High", PHASE1_KEY["High", "afternoon"]): +1}


def phase1_regions() -> dict:
    K = PHASE1_KEY
    reg = {"Low": {K["Low", "in_window"]: list(LOW_WINDOW), K["Low", "before"]: list(_P1_LOW_BEFORE),
                   K["Low", "after"]: list(_P1_LOW_AFTER), K["Low", "afternoon_evening"]: list(range(24, H)),
                   "outside_window": [h for h in range(H) if h not in LOW_WINDOW]},
           "High": {K["High", "in_window"]: list(HIGH_WINDOW), K["High", "before"]: list(_P1_HIGH_BEFORE),
                    K["High", "after"]: list(_P1_HIGH_AFTER), K["High", "afternoon"]: list(_P1_HIGH_AFTERNOON),
                    "outside_window": [h for h in range(H) if h not in HIGH_WINDOW]}}
    for name, rr in reg.items():
        for k, v in rr.items():
            m = _SPAN_KEY_RE.match(k)
            assert v and (m is None or v == list(range(int(m.group(1)), int(m.group(2)) + 1))), (name, k, v)
    return reg


# ----------------------------------------------------------------------------------------------------------------
# bootstrap statistics over days
# ----------------------------------------------------------------------------------------------------------------
def boot_index(n: int, B: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (B, n))


def ratio_stats(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> dict:
    """num, den: per-unit values (n,); estimate = mean(num) (kWh) and 100 mean(num)/mean(den) (%)."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    est, pct = float(num.mean()), float(100.0 * num.mean() / den.mean())
    bn, bd = num[idx].mean(1), den[idx].mean(1)
    bp = 100.0 * bn / bd
    return {"kwh": est, "kwh_ci95": [float(np.quantile(bn, 0.025)), float(np.quantile(bn, 0.975))],
            "pct": pct, "pct_ci95": [float(np.quantile(bp, 0.025)), float(np.quantile(bp, 0.975))],
            "frac_units_positive": float((num > 0).mean()), "frac_units_negative": float((num < 0).mean()),
            "n_units": int(num.size)}


def claim_status(stat: dict, sign: int | None) -> str | None:
    """'appears' (right sign, CI excludes 0), 'weak' (right sign, CI includes 0), 'absent' (wrong sign, CI includes
    0), 'contradicted' (wrong sign, CI excludes 0)."""
    if sign is None:
        return None
    lo, hi = stat["kwh_ci95"]
    right = np.sign(stat["kwh"]) == sign
    excl = (lo > 0) or (hi < 0)
    if right:
        return "appears" if excl else "weak"
    return "contradicted" if excl else "absent"


def scenario_effects(normal: np.ndarray, scen: dict, B: int = 2000, seed: int = 0) -> dict:
    """normal: (N, D, 48); scen: {'Low': (N, D, 48), 'High': (N, D, 48)} sharing random numbers with `normal`.
    Returns region statistics, claim statuses, daily energy and per-slot profiles with bootstrap CIs."""
    Nm = normal.mean(0)
    D = Nm.shape[0]
    idx = boot_index(D, B, seed)
    out = {}
    for name, S in scen.items():
        Dd = S.mean(0) - Nm                                           # (D, 48)
        reg = {}
        for rname, slots in scenario_regions(name).items():
            st = ratio_stats(Dd[:, slots].mean(1), Nm[:, slots].mean(1), idx)
            sign = CLAIMS.get((name, rname), (None, None))[0]
            st["claimed_sign"] = sign
            st["claim_status"] = claim_status(st, sign)
            reg[rname] = st
        reg["daily_energy"] = ratio_stats(Dd.sum(1), Nm.sum(1), idx)
        bprof = Dd[idx].mean(1)                                        # (B, 48)
        prof = {"diff_kwh": Dd.mean(0).tolist(),
                "diff_kwh_ci95_lo": np.quantile(bprof, 0.025, axis=0).tolist(),
                "diff_kwh_ci95_hi": np.quantile(bprof, 0.975, axis=0).tolist(),
                "normal_kwh": Nm.mean(0).tolist(),
                "frac_days_negative": (Dd < 0).mean(0).tolist()}
        side = reg["pre4"]["kwh"], reg["post4"]["kwh"], reg["in_window"]["kwh"]
        reg["side_to_window_ratio"] = {"pre4": float(side[0] / side[2]) if side[2] else None,
                                       "post4": float(side[1] / side[2]) if side[2] else None}
        reg["max_abs_outside_window_kwh"] = float(np.abs(Dd[:, scenario_regions(name)["outside_window"]]).max())
        out[name] = {"regions": reg, "profile": prof}
    return out


# ----------------------------------------------------------------------------------------------------------------
# tariff blocks
# ----------------------------------------------------------------------------------------------------------------
def runs(x) -> list:
    """Maximal runs of non-zero entries of a 1-D array: [(start, end)] inclusive."""
    x = np.asarray(x).astype(bool)
    out, t, n = [], 0, len(x)
    while t < n:
        if x[t]:
            s = t
            while t < n and x[t]:
                t += 1
            out.append((s, t - 1))
        else:
            t += 1
    return out


def block_table(L: np.ndarray, Hh: np.ndarray) -> dict:
    """Within-day runs (as the CVAE sees each day) and continuous-series runs (crossing midnight allowed) of Low
    and High dummies; L, Hh: (n_days, 48) 0/1 in date order."""
    out = {}
    for name, M in (("Low", L), ("High", Hh)):
        wd = Counter(r for row in M for r in runs(row))
        cont = runs(M.ravel())
        out[name] = {
            "n_days_with": int((M.sum(1) > 0).sum()), "n_slots": int(M.sum()),
            "within_day_runs": [{"start": int(s), "end": int(e), "length": int(e - s + 1), "count": int(c)}
                                for (s, e), c in wd.most_common()],
            "n_within_day_runs": int(sum(wd.values())),
            "continuous_runs": {"n": len(cont), "n_crossing_midnight": int(sum(1 for s, e in cont if s // H != e // H)),
                                "length_counts": {str(k): int(v) for k, v in
                                                  sorted(Counter(e - s + 1 for s, e in cont).items())},
                                "start_slot_counts": {str(k): int(v) for k, v in
                                                      sorted(Counter(s % H for s, e in cont).items())}}}
    both = (L.sum(1) > 0) & (Hh.sum(1) > 0)
    out["n_days_with_both"] = int(both.sum())
    combos = Counter((tuple(runs(lr)), tuple(runs(hr))) for lr, hr in zip(L, Hh) if lr.sum() or hr.sum())
    out["day_patterns"] = [{"low_runs": [list(r) for r in k[0]], "high_runs": [list(r) for r in k[1]], "count": int(v)}
                           for k, v in combos.most_common(25)]
    return out


def days_covering(M: np.ndarray, slots) -> np.ndarray:
    return np.where(M[:, list(slots)].sum(1) == len(slots))[0]


# ----------------------------------------------------------------------------------------------------------------
# condition vectors for arbitrary tariff patterns (CVAE)
# ----------------------------------------------------------------------------------------------------------------
_TARIFF_RE = re.compile(r"^(low|high)(?:_[A-Za-z]+)?_(\d{2})$")


def tariff_col_index(cond_cols) -> tuple[list, list]:
    il, ih = [None] * H, [None] * H
    for j, name in enumerate(cond_cols):
        m = _TARIFF_RE.match(name)
        if m:
            h = int(m.group(2))
            (il if m.group(1) == "low" else ih)[h] = j
    if any(v is None for v in il + ih):
        raise ValueError("could not locate the 96 tariff dummies in cond_cols")
    return il, ih


def pattern_condition(C: np.ndarray, cond_cols, low_slots=(), high_slots=()) -> np.ndarray:
    """Copy of the (n_days, n_cond) condition matrix with all 96 dummies zeroed, then low_slots / high_slots = 1."""
    il, ih = tariff_col_index(cond_cols)
    C2 = np.array(C, dtype=np.float64, copy=True)
    C2[:, il] = 0.0
    C2[:, ih] = 0.0
    if len(low_slots):
        C2[:, [il[h] for h in low_slots]] = 1.0
    if len(high_slots):
        C2[:, [ih[h] for h in high_slots]] = 1.0
    return C2


def block_relative_stats(Dd: np.ndarray, Nm: np.ndarray, block: tuple, pattern_code: np.ndarray, idx: np.ndarray,
                         ks=(2, 4)) -> dict:
    """Effects of a pattern around one of its blocks (s, e) for per-day differences Dd (D, 48) and Normal means
    Nm (D, 48): in-block, k slots before / after (only slots that are Normal in the pattern), 'far' slots 5-12 away,
    and the rest of the day outside every block."""
    s, e = block
    normal = pattern_code == 0
    reg = {"in_block": list(range(s, e + 1))}
    for k in ks:
        reg[f"pre{k}"] = [h for h in range(s - k, s) if 0 <= h and normal[h]]
        reg[f"post{k}"] = [h for h in range(e + 1, e + k + 1) if h < H and normal[h]]
    reg["far_pre_5_12"] = [h for h in range(s - 12, s - 4) if 0 <= h and normal[h]]
    reg["far_post_5_12"] = [h for h in range(e + 5, e + 13) if h < H and normal[h]]
    reg["all_normal_slots"] = [h for h in range(H) if normal[h]]
    out = {}
    for name, slots in reg.items():
        if slots:
            st = ratio_stats(Dd[:, slots].mean(1), Nm[:, slots].mean(1), idx)
            st["slots"] = [int(slots[0]), int(slots[-1])] if len(slots) > 1 else [int(slots[0])]
            out[name] = st
        else:
            out[name] = None
    return out


# ----------------------------------------------------------------------------------------------------------------
# data-based reference: matched-day baseline and event study
# ----------------------------------------------------------------------------------------------------------------
def tariff_codes(L: np.ndarray, Hh: np.ndarray) -> np.ndarray:
    if np.any((L > 0) & (Hh > 0)):
        raise ValueError("a half-hour is both Low and High")
    return (L > 0).astype(int) * CODE["Low"] + (Hh > 0).astype(int) * CODE["High"]


def matched_baseline(Y: np.ndarray, work_day: np.ndarray, all_normal: np.ndarray, window: int = 14):
    """baseline(d, h) = mean of Y(d', h) over all-Normal days d' of the same day type, 0 < |d - d'| <= window."""
    n = Y.shape[0]
    B = np.full_like(Y, np.nan, dtype=np.float64)
    cnt = np.zeros(n, dtype=int)
    for d in range(n):
        lo, hi = max(0, d - window), min(n, d + window + 1)
        cand = [k for k in range(lo, hi) if k != d and all_normal[k] and work_day[k] == work_day[d]]
        if cand:
            B[d] = Y[cand].mean(0)
            cnt[d] = len(cand)
    return B, cnt


LAGS_START = list(range(-12, 6))     # relative to the first slot of the run
LAGS_END = list(range(-5, 13))       # relative to the last slot of the run


def event_lag_values(delta_f: np.ndarray, base_f: np.ndarray, code_f: np.ndarray, t0: int, t1: int,
                     same_day_only: bool = False):
    """Delta / baseline at the start- and end-aligned lags of one run [t0, t1] of the flattened series (NaN when
    the half-hour is outside the series, belongs to another non-Normal block, lies past the other end of the run,
    or - with same_day_only - falls on another day than the run)."""
    T = len(delta_f)
    day0 = t0 // H

    def val(t, inside):
        if t < 0 or t >= T:
            return np.nan, np.nan
        if same_day_only and t // H != day0:
            return np.nan, np.nan
        if not inside and code_f[t] != 0:
            return np.nan, np.nan
        return delta_f[t], base_f[t]

    ds, bs, de, be = [], [], [], []
    for k in LAGS_START:
        t = t0 + k
        inside = 0 <= k and t <= t1
        if k >= 0 and t > t1:
            a, b = np.nan, np.nan
        else:
            a, b = val(t, inside)
        ds.append(a)
        bs.append(b)
    for k in LAGS_END:
        t = t1 + k
        inside = k <= 0 and t >= t0
        if k <= 0 and t < t0:
            a, b = np.nan, np.nan
        else:
            a, b = val(t, inside)
        de.append(a)
        be.append(b)
    return np.array(ds), np.array(bs), np.array(de), np.array(be)


def _agg_masks():
    s = np.array(LAGS_START)
    e = np.array(LAGS_END)
    return {"pre2": ("start", (s >= -2) & (s <= -1)), "pre4": ("start", (s >= -4) & (s <= -1)),
            "far_pre_5_12": ("start", (s >= -12) & (s <= -5)), "first4_in_block": ("start", (s >= 0) & (s <= 3)),
            "last4_in_block": ("end", (e >= -3) & (e <= 0)), "post2": ("end", (e >= 1) & (e <= 2)),
            "post4": ("end", (e >= 1) & (e <= 4)), "far_post_5_12": ("end", (e >= 5) & (e <= 12))}


def event_study(delta_f, base_f, code_f, kind: str, B: int = 2000, seed: int = 0, same_day_only: bool = False,
                min_len: int = 1, select=None) -> dict:
    """Lag profiles (mean Delta in kWh and in % of the baseline) around all runs of `kind` ('Low'/'High') in the
    flattened series, with percentile-bootstrap CIs over event days. `select(t0, t1) -> bool` filters runs."""
    T = len(code_f)
    rr = [(s, e) for s, e in runs(code_f == CODE[kind]) if e - s + 1 >= min_len]
    if select is not None:
        rr = [(s, e) for s, e in rr if select(s, e)]
    if not rr:
        return {"n_events": 0}
    vals = [event_lag_values(delta_f, base_f, code_f, s, e, same_day_only) for s, e in rr]
    DS = np.array([v[0] for v in vals])
    BS = np.array([v[1] for v in vals])
    DE = np.array([v[2] for v in vals])
    BE = np.array([v[3] for v in vals])
    # in-block mean per event
    inb = np.array([np.nanmean(delta_f[s:e + 1]) for s, e in rr])
    inb_b = np.array([np.nanmean(base_f[s:e + 1]) for s, e in rr])
    day = np.array([s // H for s, e in rr])
    udays = np.unique(day)
    rng = np.random.default_rng(seed)
    groups = [np.where(day == d)[0] for d in udays]
    boot_sets = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]

    def lag_profile(Dm, Bm, lags):
        mean = np.nanmean(Dm, 0)
        n = np.sum(np.isfinite(Dm), 0)
        pct = 100.0 * np.nansum(Dm, 0) / np.where(n > 0, np.nansum(np.where(np.isfinite(Dm), Bm, np.nan), 0), np.nan)
        bm = np.array([np.nanmean(Dm[b], 0) for b in boot_sets])
        return {"lags": list(lags), "mean_kwh": mean.tolist(), "pct": pct.tolist(), "n_events": n.tolist(),
                "ci95_lo_kwh": np.nanquantile(bm, 0.025, axis=0).tolist(),
                "ci95_hi_kwh": np.nanquantile(bm, 0.975, axis=0).tolist()}

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof_s = lag_profile(DS, BS, LAGS_START)
        prof_e = lag_profile(DE, BE, LAGS_END)
        aggs = {}
        for name, (side, mask) in _agg_masks().items():
            Dm, Bm = (DS, BS) if side == "start" else (DE, BE)
            per_ev = np.nanmean(Dm[:, mask], 1)
            per_b = np.nanmean(np.where(np.isfinite(Dm[:, mask]), Bm[:, mask], np.nan), 1)
            ok = np.isfinite(per_ev) & np.isfinite(per_b)
            aggs[name] = _event_agg(per_ev, per_b, ok, boot_sets)
        ok = np.isfinite(inb) & np.isfinite(inb_b)
        aggs["in_block"] = _event_agg(inb, inb_b, ok, boot_sets)
    return {"kind": kind, "n_events": len(rr), "n_event_days": int(len(udays)), "same_day_only": same_day_only,
            "events": [[int(s), int(e)] for s, e in rr],
            "length_counts": {str(k): int(v) for k, v in sorted(Counter(e - s + 1 for s, e in rr).items())},
            "start_aligned": prof_s, "end_aligned": prof_e, "aggregates": aggs}


def _event_agg(per_ev, per_b, ok, boot_sets) -> dict:
    if ok.sum() == 0:
        return {"n_events": 0}
    est = float(per_ev[ok].mean())
    pct = float(100.0 * per_ev[ok].sum() / per_b[ok].sum())
    bs, bp = [], []
    for b in boot_sets:
        bb = b[ok[b]]
        if len(bb):
            bs.append(per_ev[bb].mean())
            bp.append(100.0 * per_ev[bb].sum() / per_b[bb].sum())
    return {"kwh": est, "kwh_ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "pct": pct, "pct_ci95": [float(np.quantile(bp, 0.025)), float(np.quantile(bp, 0.975))],
            "n_events": int(ok.sum())}


def selftest(verbose: bool = True) -> dict:
    """Synthetic checks of the scenario windows and regions, runs, baselines, the event study (known injected
    effects) and scenario statistics."""
    say = print if verbose else (lambda *a, **k: None)
    # --- windows (REPRO_SPEC section 7) and their derived regions / labels
    assert LOW_WINDOW == list(range(9, 20)) and len(LOW_WINDOW) == 11
    assert HIGH_WINDOW == list(range(39, 45)) and len(HIGH_WINDOW) == 6
    assert WINDOW_CLOCK == {"Low": "04:00-09:30", "High": "19:00-22:00"}, WINDOW_CLOCK
    assert WINDOW_HOURS == {"Low": (4.0, 9.5), "High": (19.0, 22.0)} and WINDOW_SPAN == {"Low": "9-19", "High": "39-44"}
    assert slots_clock([0]) == "23:30-00:00" and slots_clock([36, 37, 38]) == "17:30-19:00"
    rl, rh = scenario_regions("Low"), scenario_regions("High")
    assert rh["pre2"] == [37, 38] and rh["pre4"] == [35, 36, 37, 38] and rh["post2"] == [45, 46]
    assert rh["post4"] == [45, 46, 47] == rh[KEY_HIGH_POST_REBOUND] and KEY_HIGH_POST_REBOUND == "rebound_post_45_47"
    assert rh["afternoon_24_34"] == list(range(24, 35)) and rh["rebound_pre_24_38"] == list(range(24, 39))
    assert rl["pre2"] == [7, 8] and rl["pre4"] == [5, 6, 7, 8] and rl["post2"] == [20, 21]
    assert rl["post4"] == [20, 21, 22, 23] and KEY_LOW_REBOUND == "rebound_20_31"
    assert rl[KEY_LOW_REBOUND] == list(range(20, 32)) and rl["rebound_32_47"] == list(range(32, 48))
    assert rl["afternoon_24_31"] == list(range(24, 32))
    assert set(rl["outside_window"]) | set(LOW_WINDOW) == set(range(H)) and len(rh["outside_window"]) == H - 6
    assert all(r in scenario_regions(s) for s, r in CLAIMS), "every claimed region exists"
    p1 = phase1_regions()
    assert list(p1["Low"]) == ["in_window_09_19", "before_06_08", "after_20_22", "afternoon_evening_24_47",
                               "outside_window"], list(p1["Low"])
    assert list(p1["High"]) == ["in_window_39_44", "before_36_38", "after_45_47", "afternoon_24_35",
                                "outside_window"], list(p1["High"])
    assert all(r in p1[s] for s, r in PHASE1_CLAIMED_SIGN)
    assert SUPERSEDED_WINDOWS == {"Low": list(range(9, 19)), "High": list(range(39, 44))}
    assert runs([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)] and runs([1, 1, 1]) == [(0, 2)] and runs([0, 0]) == []
    rng = np.random.default_rng(3)
    n = 200
    wd = (np.arange(n) % 7 < 5).astype(int)
    base = 0.2 + 0.05 * np.sin(2 * np.pi * np.arange(H) / H)
    Y = base[None] + 0.01 * rng.standard_normal((n, H))
    L = np.zeros((n, H), int)
    Hh = np.zeros((n, H), int)
    ev_days = np.arange(5, n - 5, 9)
    for d in ev_days:                               # High 35-46 with a -0.03 effect, +0.01 one slot before
        Hh[d, 35:47] = 1
        Y[d, 35:47] -= 0.03
        Y[d, 34] += 0.01
    code = tariff_codes(L, Hh)
    alln = (code.sum(1) == 0)
    Bm, cnt = matched_baseline(Y, wd, alln, 14)
    assert cnt[ev_days].min() > 0
    es = event_study((Y - Bm).ravel(), Bm.ravel(), code.ravel(), "High", B=300, seed=1)
    a = es["aggregates"]
    assert abs(a["in_block"]["kwh"] + 0.03) < 0.004, a["in_block"]
    lag_m1 = es["start_aligned"]["mean_kwh"][LAGS_START.index(-1)]
    assert abs(lag_m1 - 0.01) < 0.005, lag_m1
    assert abs(a["post4"]["kwh"]) < 0.004 and abs(a["far_pre_5_12"]["kwh"]) < 0.004
    # scenario statistics: injected window effect only
    N, D = 50, 30
    nm = 0.2 + 0.01 * rng.standard_normal((N, D, H))
    lo = nm.copy()
    lo[:, :, LOW_WINDOW] += 0.02
    hi = nm.copy()
    hi[:, :, HIGH_WINDOW] -= 0.01
    se = scenario_effects(nm, {"Low": lo, "High": hi}, B=200)
    assert abs(se["Low"]["regions"]["in_window"]["kwh"] - 0.02) < 1e-12
    assert se["Low"]["regions"]["in_window"]["claim_status"] == "appears"
    assert abs(se["Low"]["regions"]["post2"]["kwh"]) < 1e-12 and abs(se["Low"]["regions"][KEY_LOW_REBOUND]["kwh"]) < 1e-12
    assert abs(se["High"]["regions"]["pre4"]["kwh"]) < 1e-12 and se["High"]["regions"]["max_abs_outside_window_kwh"] < 1e-12
    assert abs(se["High"]["regions"]["post4"]["kwh"]) < 1e-12 and se["High"]["regions"]["post4"]["n_units"] == D
    assert abs(se["High"]["regions"]["daily_energy"]["kwh"] + 0.01 * len(HIGH_WINDOW)) < 1e-12
    assert abs(se["Low"]["regions"]["daily_energy"]["kwh"] - 0.02 * len(LOW_WINDOW)) < 1e-12
    C = np.zeros((3, 102))
    cols = ["temp_smooth_n", "pca_temp_0", "pca_temp_1", "pca_temp_2", "pos_y", "work_day"] + \
        [nm_ for i in range(48) for nm_ in (f"low_{i:02d}", f"high_{i:02d}")]
    C2 = pattern_condition(C, cols, low_slots=LOW_WINDOW, high_slots=[39])
    assert C2[:, cols.index("low_09")].all() and C2[:, cols.index("low_19")].all() and not C2[:, cols.index("low_20")].any()
    assert C2[:, cols.index("high_39")].all() and C2.sum() == 3 * (len(LOW_WINDOW) + 1)
    C3 = pattern_condition(C2, cols, high_slots=HIGH_WINDOW)
    assert C3[:, cols.index("high_44")].all() and not C3[:, cols.index("high_45")].any() and C3.sum() == 3 * 6
    say(f"[effects selftest] windows Low {WINDOW_TEXT['Low']}, High {WINDOW_TEXT['High']} and their regions (High post4 "
        f"= {span(rh['post4'])}, {len(rh['post4'])} slots), runs, matched baseline, event study (injected -0.03 in-block, "
        "+0.01 at lag -1, 0 elsewhere recovered), scenario statistics and pattern conditions: ok")
    return {"status": "ok", "in_block": a["in_block"]["kwh"], "lag_minus1": lag_m1}


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
