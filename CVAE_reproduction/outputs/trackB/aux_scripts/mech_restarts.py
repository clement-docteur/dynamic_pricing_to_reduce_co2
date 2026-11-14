"""Mechanism test across ALL 50 CVAE restarts per cluster, Track A and Track B (the phase-2 mechanism stage of
scripts/21_tariff_effects.py only looks at the selected decoder).

For every restart: CRN means (z = default_rng([20200801, cluster]), as in phase 1) for Normal, the paper's windows
(High 39-44, Low 9-19: lib.effects, REPRO_SPEC section 7) and the most frequent real training blocks that contain them
(High 33-44, 35-46, 21-44; Low 9-32, 9-20) inserted alone into every test day's conditions. Per restart:
  corr(paper response, real-block response) of the 48-slot mean difference profiles
  share of the |outside-window| response of the paper High scenario that lies in slots 33-38 + 45-46
  (uniform spreading would give 8/42 = 19%)
  drop at 33-34 (only in block 33-44) vs 45-46 (only in block 35-46) for the paper High scenario
  in-window / pre4 (35-38) / post4 (45-47, 3 slots left in the day) effects (% of Normal), joined with the
  per-restart claim statuses of outputs/track<T>/effects_all_restarts_regions.csv
Writes outputs/trackB/mechanism_restarts_AB.csv and outputs/trackB/mechanism_restarts_AB_summary.json
(phase-2 version with the superseded windows 9-18 / 39-43: outputs/archive_windows_9-18_39-43/trackB/)
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
sys.path.insert(0, REPRO)
AUTHORS = REPRO + "/data/authors"  # copy of the authors' repo data/*.feather (byte-identical; read as data only)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from lib import cvae_torch as ct  # noqa: E402
from lib import effects as ef  # noqa: E402  (only to cross-check my own pattern builder)

torch.set_num_threads(1)
H = 48
GEN_SEED = 20200801
N = 200
TEST = {"A": AUTHORS + "/test_nmf.feather", "B": REPRO + "/data/test_ours.parquet"}
OUT = {"A": REPRO + "/outputs/trackA", "B": REPRO + "/outputs/trackB"}
HW = list(ef.HIGH_WINDOW)                                       # 39-44
LW = list(ef.LOW_WINDOW)                                        # 9-19
P_HIGH, P_LOW = f"paper_High_{HW[0]}_{HW[-1]}", f"paper_Low_{LW[0]}_{LW[-1]}"
PATTERNS = {P_HIGH: ((), HW), "real_High_33_44": ((), range(33, 45)),
            "real_High_35_46": ((), range(35, 47)), "real_High_21_44": ((), range(21, 45)),
            P_LOW: (LW, ()), "real_Low_9_32": (range(9, 33), ()),
            "real_Low_9_20": (range(9, 21), ())}
assert all(set(HW) <= set(hi) for k, (lo, hi) in PATTERNS.items() if k.startswith("real_High"))
assert all(set(LW) <= set(lo) for k, (lo, hi) in PATTERNS.items() if k.startswith("real_Low"))
OUTW = [h for h in range(H) if h not in HW]
NEAR = [h for h in OUTW if 33 <= h <= 46]                       # 33-38 and 45-46
PRE4 = list(range(HW[0] - 4, HW[0]))                            # 35-38
POST4 = [h for h in range(HW[-1] + 1, HW[-1] + 5) if h < H]     # 45-47
KEY_LREB = ef.KEY_LOW_REBOUND                                   # 'rebound_20_31'


def read(path):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"])
    df["cluster"] = df["cluster"].astype(int)
    return df


def pattern(C, cols, lo=(), hi=()):
    il = [cols.index(f"low_{h:02d}") for h in range(H)]
    ih = [cols.index(f"high_{h:02d}") for h in range(H)]
    C2 = np.array(C, dtype=np.float64, copy=True)
    C2[:, il] = 0.0
    C2[:, ih] = 0.0
    for h in lo:
        C2[:, il[h]] = 1.0
    for h in hi:
        C2[:, ih[h]] = 1.0
    return C2


def main():
    t0 = time.perf_counter()
    rows = []
    checked = False
    for t in ("A", "B"):
        te = read(TEST[t])
        st = pd.read_csv(OUT[t] + "/effects_all_restarts_regions.csv")
        for c in range(4):
            blk = te.loc[te.cluster == c].sort_values("date").reset_index(drop=True)
            for s in range(50):
                m = ct.load_model(OUT[t] + f"/models/cvae_c{c}_s{s:02d}.pt")
                cols = list(m.meta["cond_cols"])
                C = blk[cols].to_numpy(dtype=np.float64)
                if not checked:
                    assert np.array_equal(pattern(C, cols, LW, [39]), ef.pattern_condition(C, cols, LW, [39]))
                    assert np.array_equal(pattern(C, cols, (), HW), ef.pattern_condition(C, cols, (), HW))
                    checked = True
                eps = np.random.default_rng([GEN_SEED, c]).standard_normal((N, len(C), m.latent_dim))
                Nm = ct.generate(m, pattern(C, cols), N, eps=eps).mean(0)
                prof = {}
                for name, (lo, hi) in PATTERNS.items():
                    prof[name] = (ct.generate(m, pattern(C, cols, lo, hi), N, eps=eps).mean(0) - Nm).mean(0)
                nprof = Nm.mean(0)
                a = prof[P_HIGH]
                b = prof[P_LOW]
                row = {"track": t, "cluster": c, "seed": s,
                       "high_in_pct": 100 * a[HW].sum() / nprof[HW].sum(),
                       "high_pre4_pct": 100 * a[PRE4].sum() / nprof[PRE4].sum(),
                       "high_post4_pct": 100 * a[POST4].sum() / nprof[POST4].sum(),
                       "low_in_pct": 100 * b[LW].sum() / nprof[LW].sum(),
                       "high_share_abs_outside_in_33_46": float(np.abs(a[NEAR]).sum() / np.abs(a[OUTW]).sum()),
                       "high_share_signed_outside_in_33_46": float(a[NEAR].sum() / a[OUTW].sum()),
                       "high_drop_33_34_pct": 100 * a[33:35].sum() / nprof[33:35].sum(),
                       "high_drop_45_46_pct": 100 * a[45:47].sum() / nprof[45:47].sum()}
                for k in ("real_High_33_44", "real_High_35_46", "real_High_21_44"):
                    row[f"corr_paperHigh_{k}"] = float(np.corrcoef(a, prof[k])[0, 1])
                for k in ("real_Low_9_32", "real_Low_9_20"):
                    row[f"corr_paperLow_{k}"] = float(np.corrcoef(b, prof[k])[0, 1])
                # outside-window shape similarity only (excluding the paper window itself)
                row["corr_outside_paperHigh_real_High_33_44"] = float(np.corrcoef(a[OUTW], prof["real_High_33_44"][OUTW])[0, 1])
                row["corr_outside_paperHigh_real_High_35_46"] = float(np.corrcoef(a[OUTW], prof["real_High_35_46"][OUTW])[0, 1])
                r = st.loc[(st.cluster == c) & (st.seed == s)].iloc[0]
                row["selected"] = bool(r["selected"])
                row["status_pre2"], row["status_pre4"] = r["High:pre2:status"], r["High:pre4:status"]
                row["status_post2"], row["status_post4"] = r["High:post2:status"], r["High:post4:status"]
                row["status_high_in"], row["status_low_in"] = r["High:in_window:status"], r["Low:in_window:status"]
                row["side_effect_in_output"] = bool(("appears" in (r["High:pre2:status"], r["High:pre4:status"]))
                                                    and ("appears" in (r["High:post2:status"], r["High:post4:status"])))
                row["check_high_in_pct_vs_csv"] = float(row["high_in_pct"] - r["High:in_window:pct"])
                row["low_rebound_both"] = bool(r[f"Low:{KEY_LREB}:status"] == "appears"
                                               and r["Low:rebound_32_47:status"] == "appears")
                row["high_rebound_24_34"] = bool(r["High:afternoon_24_34:status"] == "appears")
                row["direction_both"] = bool(r["Low:in_window:status"] == "appears"
                                             and r["High:in_window:status"] == "appears")
                rows.append(row)
            print(f"[{time.strftime('%H:%M:%S')}] track {t} c{c} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT["B"] + "/mechanism_restarts_AB.csv", index=False)
    summ = {"scenario_windows": {"Low": LW, "High": HW, "High_pre4": PRE4, "High_post4": POST4,
                                 "slots_33_46_outside_window": NEAR,
                                 "uniform_share_of_outside_slots": len(NEAR) / len(OUTW)},
            "n_restarts_per_cluster": 50, "max_abs_check_high_in_pct_vs_effects_csv":
            float(df["check_high_in_pct_vs_csv"].abs().max()), "clusters": {}}
    for c in range(4):
        rec = {}
        for t in ("A", "B"):
            d = df.loc[(df.track == t) & (df.cluster == c)]
            sel = d.loc[d.selected].iloc[0]
            resp = d.loc[d.high_in_pct <= -5]          # restarts with a real (>= 5%) High response
            rec[t] = {
                "n_side_effect_in_output": int(d.side_effect_in_output.sum()),
                "n_direction_both_appear": int(d.direction_both.sum()),
                "n_low_rebound_both_appear": int(d.low_rebound_both.sum()),
                "n_high_rebound_24_34_appears": int(d.high_rebound_24_34.sum()),
                "selected_seed": int(sel.seed), "selected_side_effect_in_output": bool(sel.side_effect_in_output),
                "median_high_in_pct": float(d.high_in_pct.median()),
                "median_corr_paperHigh_real_33_44": float(d.corr_paperHigh_real_High_33_44.median()),
                "median_corr_paperHigh_real_35_46": float(d.corr_paperHigh_real_High_35_46.median()),
                "median_corr_outside_paperHigh_real_33_44": float(d.corr_outside_paperHigh_real_High_33_44.median()),
                "median_corr_outside_paperHigh_real_35_46": float(d.corr_outside_paperHigh_real_High_35_46.median()),
                "median_share_abs_outside_in_33_46": float(d.high_share_abs_outside_in_33_46.median()),
                "n_restarts_high_response_ge_5pct": int(len(resp)),
                "among_responsive_median_corr_33_44": float(resp.corr_paperHigh_real_High_33_44.median()) if len(resp) else None,
                "among_responsive_median_corr_35_46": float(resp.corr_paperHigh_real_High_35_46.median()) if len(resp) else None,
                "among_responsive_median_share_abs_outside_in_33_46":
                    float(resp.high_share_abs_outside_in_33_46.median()) if len(resp) else None,
                "among_responsive_median_drop_33_34_pct": float(resp.high_drop_33_34_pct.median()) if len(resp) else None,
                "among_responsive_median_drop_45_46_pct": float(resp.high_drop_45_46_pct.median()) if len(resp) else None,
                "median_corr_paperLow_real_9_32": float(d.corr_paperLow_real_Low_9_32.median()),
                "median_corr_paperLow_real_9_20": float(d.corr_paperLow_real_Low_9_20.median())}
        a = df.loc[(df.track == "A") & (df.cluster == c)]
        b = df.loc[(df.track == "B") & (df.cluster == c)]
        ra, rb = a.loc[a.high_in_pct <= -5], b.loc[b.high_in_pct <= -5]
        tests = {}
        for k in ("high_in_pct", "high_pre4_pct", "high_post4_pct", "low_in_pct", "corr_paperHigh_real_High_33_44",
                  "high_share_abs_outside_in_33_46"):
            tests[k] = float(stats.mannwhitneyu(a[k], b[k]).pvalue)
        # co-occurrence prediction: Track B trains on 35-46 twice as often as on 33-44 (12 vs 6; Track A 6 vs 7)
        if len(ra) >= 5 and len(rb) >= 5:
            xa = ra.high_drop_45_46_pct - ra.high_drop_33_34_pct
            xb = rb.high_drop_45_46_pct - rb.high_drop_33_34_pct
            tests["responsive_drop45_46_minus_drop33_34_A_median"] = float(xa.median())
            tests["responsive_drop45_46_minus_drop33_34_B_median"] = float(xb.median())
            tests["responsive_drop45_46_minus_drop33_34_mannwhitney_p"] = float(stats.mannwhitneyu(xa, xb).pvalue)
        rec["A_vs_B_mannwhitney_p"] = tests
        summ["clusters"][str(c)] = rec
    summ["runtime_s"] = time.perf_counter() - t0
    with open(OUT["B"] + "/mechanism_restarts_AB_summary.json", "w") as f:
        json.dump(summ, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(summ, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
