"""Independent re-implementation (no lib.effects functions; only its scenario-window constants) of the data-based
tariff reference, both tracks:
  matched baseline = mean of all-Normal days of the same day type within +-14 days (excluding the day itself)
  (1) in-block effect of every maximal High / Low run of the continuous 365 x 48 series (pct = 100 sum delta /
      sum baseline over runs) -> cross-check of mechanism_summary.json
  (2) morning-Low-only days (every Low run ends at slot <= 10, no High that day): effect in slots 20-31 and 32-47
      (the paper's Low-rebound regions after the Low window 9-19) with a day bootstrap; the CVAE's paper Low scenario
      predicts a drop there
  (3) evening-High days (a High run covering the paper's High window 39-44): effect in slots 24-34 restricted to days
      with no Low/High in 24-34 (paper's High-rebound region) and in the 4 Normal slots before / after the High run
  (4) baseline sensitivity: one-week-lag baseline (same weekday, d-7 and d+7 when all-Normal)
  (5) variant 'B_authsched': our cluster aggregates with the authors' tariff dummies (conditions_ours *_auth), to
      separate the effect of the tariff schedule (17 days differ, mostly Nov-Dec) from the cluster composition
Windows: lib.effects (REPRO_SPEC section 7); the phase-2 version used 9-18 / 39-43 (regions 19-31, runs covering
39-43): outputs/archive_windows_9-18_39-43/trackB/mechanism_data_check_AB.json.
Writes outputs/trackB/mechanism_data_check_AB.json (merged into outputs/comparison_AB.json).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
sys.path.insert(0, REPRO)
from lib.effects import HIGH_WINDOW, KEY_LOW_REBOUND, LOW_WINDOW  # noqa: E402  (window constants only)

HW = list(HIGH_WINDOW)                                   # 39-44
LREB = list(range(LOW_WINDOW[-1] + 1, 32))               # 20-31: after the paper's Low window
LREB_KEY = KEY_LOW_REBOUND.replace("rebound_", "")       # '20_31'
AUTHORS = REPRO + "/data/authors"  # copy of the authors' repo data/*.feather (byte-identical; read as data only)
OUTJ = REPRO + "/outputs/trackB/mechanism_data_check_AB.json"
POWER = {"A": AUTHORS + "/power_consump_nmf.feather", "B": REPRO + "/data/power_consump_ours.parquet"}
H = 48
B = 4000


def read(p):
    df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_feather(p)
    df["date"] = pd.to_datetime(df["date"])
    df["cluster"] = df["cluster"].astype(int)
    return df


def read_cond():
    co = pd.read_parquet(REPRO + "/data/conditions_ours.parquet")
    co["date"] = pd.to_datetime(co["date"])
    return co.sort_values("date").reset_index(drop=True)


def runs1d(x):
    out, t = [], 0
    x = np.asarray(x).astype(bool)
    while t < len(x):
        if x[t]:
            s = t
            while t < len(x) and x[t]:
                t += 1
            out.append((s, t - 1))
        else:
            t += 1
    return out


def baseline(Y, wd, alln, kind="pm14"):
    n = len(Y)
    Bm = np.full(Y.shape, np.nan)
    for d in range(n):
        if kind == "pm14":
            cand = [k for k in range(max(0, d - 14), min(n, d + 15)) if k != d and alln[k] and wd[k] == wd[d]]
        else:                                   # one-week lag: d-7 and d+7 if all-Normal
            cand = [k for k in (d - 7, d + 7) if 0 <= k < n and alln[k]]
        if cand:
            Bm[d] = Y[cand].mean(0)
    return Bm


def boot_ratio(num, den, seed):
    num, den = np.asarray(num), np.asarray(den)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (B, len(num)))
    bp = 100 * num[idx].sum(1) / den[idx].sum(1)
    return {"pct": float(100 * num.sum() / den.sum()), "pct_ci95": [float(np.quantile(bp, .025)),
                                                                   float(np.quantile(bp, .975))],
            "n": int(len(num))}


def main():
    res = {"windows": {"High_window": HW, "Low_window": list(LOW_WINDOW), "Low_rebound_region": [LREB[0], LREB[-1]],
                       "evening_high_days": f"a High run covering slots {HW[0]}-{HW[-1]}",
                       "note": "phase-2 version (9-18 / 39-43): outputs/archive_windows_9-18_39-43/trackB/"}}
    for t in ("A", "B", "B_authsched"):
        pw = read(POWER["B" if t == "B_authsched" else t])
        b0 = pw.loc[pw.cluster == 0].sort_values("date")
        L = b0[[f"low_{h:02d}" for h in range(H)]].to_numpy(float) > 0
        Hh = b0[[f"high_{h:02d}" for h in range(H)]].to_numpy(float) > 0
        for c in range(1, 4):
            bc = pw.loc[pw.cluster == c].sort_values("date")
            assert np.array_equal(bc[[f"low_{h:02d}" for h in range(H)]].to_numpy(float) > 0, L)
        if t == "B_authsched":      # our cluster aggregates with the AUTHORS' tariff dummies (conditions_ours *_auth)
            co = read_cond()
            assert np.array_equal(co["date"].to_numpy(), b0["date"].to_numpy())
            L = co[[f"low_auth_{h:02d}" for h in range(H)]].to_numpy(float) > 0
            Hh = co[[f"high_auth_{h:02d}" for h in range(H)]].to_numpy(float) > 0
        code = L.astype(int) + 2 * Hh.astype(int)
        alln = code.sum(1) == 0
        wd = b0["work_day"].to_numpy(int)
        res[t] = {"n_event_days": int((~alln).sum()), "clusters": {}}
        # day classes
        morning_low = [d for d in range(len(L)) if L[d].any() and not Hh[d].any()
                       and max(e for s, e in runs1d(L[d])) <= 10]
        eve_high = [d for d in range(len(L)) if Hh[d, HW].all()]
        eve_high_clean_aft = [d for d in eve_high if code[d, 24:35].sum() == 0]
        res[t]["n_morning_low_only_days"] = len(morning_low)
        res[t]["n_evening_high_days"] = len(eve_high)
        res[t]["n_evening_high_days_clean_24_34"] = len(eve_high_clean_aft)
        for c in range(4):
            bc = pw.loc[pw.cluster == c].sort_values("date")
            Y = bc[[f"consumption_{h:02d}" for h in range(H)]].to_numpy(float)
            rc = {}
            for bk in ("pm14", "week"):
                Bm = baseline(Y, wd, alln, bk)
                Dl = Y - Bm
                yf, bf, cf = Dl.ravel(), Bm.ravel(), code.ravel()
                inb = {}
                for kind, k in (("High", 2), ("Low", 1)):
                    rr = runs1d(cf == k)
                    num = [np.nanmean(yf[s:e + 1]) for s, e in rr]
                    den = [np.nanmean(bf[s:e + 1]) for s, e in rr]
                    inb[kind] = boot_ratio(num, den, 10 + c)
                    inb[kind]["n_runs"] = len(rr)
                r = {"in_block": inb}
                # (2) morning Low only: afternoon / evening
                for reg, sl in ((LREB_KEY, slice(LREB[0], LREB[-1] + 1)), ("32_47", slice(32, 48))):
                    r[f"morning_low_only_{reg}"] = boot_ratio(Dl[morning_low, sl].mean(1), Bm[morning_low, sl].mean(1),
                                                              20 + c)
                # (3) evening High: afternoon 24-34 (clean days), 4 Normal slots before/after the High run on the day
                r["evening_high_afternoon_24_34"] = boot_ratio(Dl[eve_high_clean_aft, 24:35].mean(1),
                                                               Bm[eve_high_clean_aft, 24:35].mean(1), 30 + c)
                pre, post, preb, postb = [], [], [], []
                for d in eve_high:
                    s, e = [(s, e) for s, e in runs1d(Hh[d]) if s <= HW[0] and e >= HW[-1]][0]
                    ps = [h for h in range(s - 4, s) if h >= 0 and code[d, h] == 0]
                    qs = [h for h in range(e + 1, e + 5) if h < H and code[d, h] == 0]
                    pre.append(Dl[d, ps].mean() if ps else np.nan)
                    preb.append(Bm[d, ps].mean() if ps else np.nan)
                    post.append(Dl[d, qs].mean() if qs else np.nan)
                    postb.append(Bm[d, qs].mean() if qs else np.nan)
                r["evening_high_pre4"] = boot_ratio(pre, preb, 40 + c)
                r["evening_high_post4"] = boot_ratio(post, postb, 50 + c)
                rc[bk] = r
            res[t]["clusters"][str(c)] = rc
            print(t, c, "in-block High", rc["pm14"]["in_block"]["High"]["pct"], "Low", rc["pm14"]["in_block"]["Low"]["pct"],
                  f"| morning-low {LREB_KEY}", rc["pm14"][f"morning_low_only_{LREB_KEY}"], "32-47",
                  rc["pm14"]["morning_low_only_32_47"]["pct"], "| eve-high aft", rc["pm14"]["evening_high_afternoon_24_34"],
                  "| pre4", rc["pm14"]["evening_high_pre4"]["pct"], "post4", rc["pm14"]["evening_high_post4"]["pct"],
                  flush=True)
        print(t, "days: morning-low-only", len(morning_low), "evening High", len(eve_high), "clean aft",
              len(eve_high_clean_aft))
    with open(OUTJ, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
