"""v-numbers: sharper checks that the arrays' day axis = test_dates.csv order = observed rows sorted by date."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"
import json
import numpy as np
import pandas as pd

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
SP = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad"
UPDATA = SP + "/upstream/repo/data/"
OUTDIR = SP + "/verify/v-numbers"
CONS = [f"consumption_{h:02d}" for h in range(48)]
LOWC = [f"low_{h:02d}" for h in range(48)]
HIGHC = [f"high_{h:02d}" for h in range(48)]


def load(track):
    df = pd.read_feather(UPDATA + "test_nmf.feather") if track == "trackA" else pd.read_parquet(REPRO + "/data/test_ours.parquet")
    df = df.assign(cluster=df["cluster"].to_numpy(dtype=np.float64).astype(int), date=pd.to_datetime(df["date"]))
    out = {}
    for c in range(4):
        s = df.loc[df.cluster == c].sort_values("date", kind="stable")
        out[c] = (s[CONS].to_numpy(np.float64), s[LOWC].to_numpy(np.float64), s[HIGHC].to_numpy(np.float64))
    return out


res = {}
for track in ("trackA", "trackB"):
    data = load(track)
    res[track] = {}
    for c in range(4):
        Y, low, high = data[c]
        ev = (low.sum(1) + high.sum(1)) > 0
        r = {}
        # 1. CVAE: samples == scen_Normal exactly iff the day's real tariff is all Normal
        S = np.load(REPRO + f"/outputs/{track}/cvae_samples_c{c}.npy")
        Nn = np.load(REPRO + f"/outputs/{track}/cvae_scen_Normal_c{c}.npy")
        dmax = np.abs(S - Nn).max(axis=(0, 2))
        zero_day = dmax == 0
        r["cvae_zero_diff_days_equal_all_normal_days"] = bool((zero_day == ~ev).all())
        r["cvae_n_event_days"] = int(ev.sum())
        # same test under shifted calendars
        r["cvae_match_frac_aligned"] = float((zero_day == ~ev).mean())
        r["cvae_match_frac_shift_prev"] = float((zero_day == ~np.roll(ev, 1)).mean())
        r["cvae_match_frac_shift_next"] = float((zero_day == ~np.roll(ev, -1)).mean())
        # 2. GAM: ratio of sample mean to Normal-scenario mean at real Low / High / other slots
        G = np.load(REPRO + f"/outputs/{track}/gam_samples_c{c}.npy").mean(0)
        GN = np.load(REPRO + f"/outputs/{track}/gam_scen_Normal_c{c}.npy").mean(0)
        ratio = G / GN
        for lab, sh in (("aligned", 0), ("shift_prev", 1), ("shift_next", -1)):
            lo = np.roll(low, sh, axis=0) > 0
            hi = np.roll(high, sh, axis=0) > 0
            ot = ~(lo | hi)
            r[f"gam_ratio_{lab}"] = {"low_slots": float(ratio[lo].mean()), "high_slots": float(ratio[hi].mean()),
                                     "other_slots": float(ratio[ot].mean())}
        # 3. generic: correlation of day-to-day changes of the daily mean, generated mean vs observed
        for gen, M in (("cvae", S.mean(0)), ("gam", G)):
            dm = np.diff(M.mean(1))
            for lab, sh in (("aligned", 0), ("shift_prev", 1), ("shift_next", -1)):
                dy = np.diff(np.roll(Y, sh, axis=0).mean(1))
                r[f"{gen}_corr_daily_mean_changes_{lab}"] = float(np.corrcoef(dm, dy)[0, 1])
            e0 = np.sqrt(((M - Y) ** 2).mean(1))
            e1 = np.sqrt(((M - np.roll(Y, 1, axis=0)) ** 2).mean(1))
            e2 = np.sqrt(((M - np.roll(Y, -1, axis=0)) ** 2).mean(1))
            r[f"{gen}_frac_days_aligned_better_than_both_shifts"] = float(((e0 < e1) & (e0 < e2)).mean())
            # profile shape: correlation of within-day de-meaned profiles, aligned vs shifted
            def shape_corr(A, B):
                a = A - A.mean(1, keepdims=True)
                b = B - B.mean(1, keepdims=True)
                return float(np.mean([np.corrcoef(a[d], b[d])[0, 1] for d in range(len(a))]))
            r[f"{gen}_mean_within_day_shape_corr_aligned"] = shape_corr(M, Y)
            r[f"{gen}_mean_within_day_shape_corr_shift_prev"] = shape_corr(M, np.roll(Y, 1, axis=0))
        res[track][c] = r
        print(track, c, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)}))
        print("   gam ratio", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in r.items() if isinstance(v, dict)})
json.dump(res, open(OUTDIR + "/alignment_checks.json", "w"), indent=1)
