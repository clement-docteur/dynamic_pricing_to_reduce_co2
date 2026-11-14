"""v-numbers: independent recomputation of Fig. 4 scores, calibration, RMSE difference and scenario effects.

Plain numpy/scipy only. Does NOT import lib/scores.py, lib/effects.py or any scorer script.
Inputs: observed test data (upstream test_nmf.feather read by absolute path as data; data/test_ours.parquet),
saved sample / scenario arrays (outputs/<track>/*.npy), test_dates.csv.
Output: <scratch>/scores_recomputed.json
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
SP = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad"
UPDATA = SP + "/upstream/repo/data/"
OUTDIR = SP + "/verify/v-numbers"
N = 200
H = 48
CONS = [f"consumption_{h:02d}" for h in range(H)]
LOWC = [f"low_{h:02d}" for h in range(H)]
HIGHC = [f"high_{h:02d}" for h in range(H)]
TRACKS = ["trackA", "trackB"]
GENS = ["cvae", "gam"]
PAPER = {  # spec section 3, approximate Fig. 4 medians
    "rmse": {"cvae": [0.020, 0.022, 0.018, 0.025], "gam": [0.015, 0.016, 0.015, 0.018]},
    "es": {"cvae": [0.150, 0.145, 0.170, 0.130], "gam": [0.155, 0.150, 0.175, 0.135]},
    "vg": {"cvae": [6, 7, 7, 10], "gam": [4, 4, 5, 5]},
}
t0 = time.time()


def load_obs(track):
    if track == "trackA":
        df = pd.read_feather(UPDATA + "test_nmf.feather")
    else:
        df = pd.read_parquet(REPRO + "/data/test_ours.parquet")
    cl = df["cluster"].to_numpy(dtype=np.float64)
    assert np.all(cl == np.round(cl)), "non-integer cluster labels"
    df = df.assign(cluster=cl.astype(int), date=pd.to_datetime(df["date"]))
    td = pd.read_csv(REPRO + f"/outputs/{track}/test_dates.csv")
    td_dates = pd.to_datetime(td["date"]).to_numpy()
    assert (td["day_index"].to_numpy() == np.arange(len(td))).all()
    obs, tar, chk = {}, {}, {}
    for c in range(4):
        sub = df.loc[df["cluster"] == c]
        d_file = sub["date"].to_numpy()
        sub_s = sub.sort_values("date", kind="stable")
        d_sorted = sub_s["date"].to_numpy()
        chk[c] = {
            "n_rows": int(len(sub)),
            "n_unique_dates": int(len(np.unique(d_sorted))),
            "file_order_ascending": bool((np.diff(d_file.astype("datetime64[D]").astype(np.int64)) > 0).all()),
            "sorted_dates_equal_test_dates_csv": bool(len(d_sorted) == len(td_dates) and (d_sorted.astype("datetime64[D]") == td_dates.astype("datetime64[D]")).all()),
        }
        obs[c] = sub_s[CONS].to_numpy(dtype=np.float64)
        tar[c] = (sub_s[LOWC].to_numpy(dtype=np.float64), sub_s[HIGHC].to_numpy(dtype=np.float64))
    return obs, tar, chk, td_dates


def es_eq14(S, Y):
    """eq. 14: (2/N) sum_{i<=N/2} ||X_i - y|| - (1/N) sum_{i<=N/2} ||X_i - X_{N/2+i}||, per day."""
    h = N // 2
    d_obs = np.sqrt(((S[:h] - Y[None]) ** 2).sum(2))  # (N/2, days)
    d_pair = np.sqrt(((S[:h] - S[h:]) ** 2).sum(2))
    return (2.0 / N) * d_obs.sum(0) - (1.0 / N) * d_pair.sum(0)


def es_ustat(S, Y):
    out = np.empty(S.shape[1])
    for d in range(S.shape[1]):
        X = S[:, d, :]
        t1 = np.sqrt(((X - Y[d]) ** 2).sum(1)).mean()
        D = cdist(X, X)
        out[d] = t1 - D.sum() / (2.0 * N * (N - 1))
    return out


def vg_ordered(S, Y, p=0.5):
    out = np.empty(S.shape[1])
    for d in range(S.shape[1]):
        X = S[:, d, :]
        A = (np.abs(X[:, :, None] - X[:, None, :]) ** p).mean(0)  # (48,48)
        B = np.abs(Y[d][:, None] - Y[d][None, :]) ** p
        out[d] = ((B - A) ** 2).sum()  # all ordered pairs; diagonal terms are 0
    return out


def q(v):
    v = np.asarray(v, dtype=np.float64)
    return {"median": float(np.median(v)), "mean": float(v.mean()), "q25": float(np.quantile(v, 0.25)),
            "q75": float(np.quantile(v, 0.75)), "n": int(v.size)}


obs_all, tar_all, chk_all, dates_all, res = {}, {}, {}, {}, {}
perday = {}  # (track, gen, c) -> dict of per-day arrays
for track in TRACKS:
    obs, tar, chk, tdd = load_obs(track)
    obs_all[track], tar_all[track], chk_all[track], dates_all[track] = obs, tar, chk, tdd
    res[track] = {"day_order_checks": chk, "scores": {}, "alignment": {}}
    for gen in GENS:
        for c in range(4):
            S = np.load(REPRO + f"/outputs/{track}/{gen}_samples_c{c}.npy")
            assert S.shape == (N, 92, H), S.shape
            assert np.isfinite(S).all()
            Y = obs[c]
            m = S.mean(0)
            err = m - Y
            rmse_rm = np.sqrt((err ** 2).mean(1))
            rmse_eu = np.sqrt((err ** 2).sum(1))
            es = es_eq14(S, Y)
            esu = es_ustat(S, Y)
            vg = vg_ordered(S, Y)
            var1 = S.var(0, ddof=1)
            var0 = S.var(0, ddof=0)
            corr = np.sqrt((N + 1) / N)
            ss_pooled = np.sqrt(var1.mean()) / np.sqrt((err ** 2).mean()) * corr
            ss_pooled_ddof0_nocorr = np.sqrt(var0.mean()) / np.sqrt((err ** 2).mean())
            ss_day = np.sqrt(var1.mean(1)) / rmse_rm * corr
            q05, q95 = np.quantile(S, [0.05, 0.95], axis=0)
            q25, q75 = np.quantile(S, [0.25, 0.75], axis=0)
            in90 = (Y >= q05) & (Y <= q95)
            in50 = (Y >= q25) & (Y <= q75)
            # order-statistic version of the central 90% interval (10th and 191st of 200 sorted)
            Ss = np.sort(S, axis=0)
            in90_os = (Y >= Ss[9]) & (Y <= Ss[190])
            below = (S < Y[None]).sum(0)
            ties = (S == Y[None]).sum(0)
            pit = (below + 0.5 * ties + 0.5) / (N + 1)
            outside = (Y < S.min(0)) | (Y > S.max(0))
            # alignment check: shift observations by +-1 day (along the 92-day test axis)
            def rm(Yx):
                return np.sqrt(((m - Yx) ** 2).mean(1))
            r_prev = rm(np.roll(Y, 1, axis=0))
            r_next = rm(np.roll(Y, -1, axis=0))
            Dall = np.sqrt(((m[:, None, :] - Y[None, :, :]) ** 2).mean(2))  # (gen day, obs day)
            argmin_ok = float((Dall.argmin(1) == np.arange(92)).mean())
            key = f"{gen}_c{c}"
            res[track]["scores"][key] = {
                "rmse_rootmean": q(rmse_rm), "rmse_euclid": q(rmse_eu), "es_eq14": q(es), "es_ustat": q(esu),
                "es_eq14_minus_ustat_median_abs": float(np.median(np.abs(es - esu))),
                "vg_p05_ordered": q(vg),
                "spread_skill_pooled_calibrated": float(ss_pooled),
                "spread_skill_pooled_plain_ddof0": float(ss_pooled_ddof0_nocorr),
                "spread_skill_perday_median": float(np.median(ss_day)),
                "coverage90_pooled": float(in90.mean()),
                "coverage90_pooled_orderstat": float(in90_os.mean()),
                "coverage90_perday_median": float(np.median(in90.mean(1))),
                "coverage50_pooled": float(in50.mean()),
                "coverage50_perday_median": float(np.median(in50.mean(1))),
                "pit_mean": float(pit.mean()), "pit_var": float(pit.var()),
                "pit_frac_below_005": float((pit < 0.05).mean()), "pit_frac_above_095": float((pit > 0.95).mean()),
                "frac_outside_sample_range": float(outside.mean()),
                "n_ties": int(ties.sum()),
                "sample_sd_mean_kwh": float(np.sqrt(var1).mean()),
                "rms_sample_sd_kwh": float(np.sqrt(var1.mean())),
                "rmse_pooled_kwh": float(np.sqrt((err ** 2).mean())),
                "mean_obs_kwh": float(Y.mean()),
            }
            res[track]["alignment"][key] = {
                "median_rmse_aligned": float(np.median(rmse_rm)),
                "median_rmse_obs_shift_prev_day": float(np.median(r_prev)),
                "median_rmse_obs_shift_next_day": float(np.median(r_next)),
                "ratio_prev_over_aligned": float(np.median(r_prev) / np.median(rmse_rm)),
                "ratio_next_over_aligned": float(np.median(r_next) / np.median(rmse_rm)),
                "frac_days_aligned_obs_is_nearest_of_92": argmin_ok,
            }
            perday[(track, gen, c)] = {"rmse": rmse_rm, "es": es, "vg": vg}
            print(f"{time.time()-t0:6.1f}s {track} {key}: rmse {np.median(rmse_rm):.5f} es {np.median(es):.4f} "
                  f"vg {np.median(vg):.3f} ss {ss_pooled:.3f} cov90 {in90.mean():.3f} shift x{np.median(r_prev)/np.median(rmse_rm):.2f}",
                  flush=True)
        # cluster cross-check: generator cluster c vs observed cluster c'
    for gen in GENS:
        M = np.zeros((4, 4))
        for c in range(4):
            m = np.load(REPRO + f"/outputs/{track}/{gen}_samples_c{c}.npy").mean(0)
            for c2 in range(4):
                M[c, c2] = np.median(np.sqrt(((m - obs[c2]) ** 2).mean(1)))
        res[track]["alignment"][f"{gen}_cluster_cross_median_rmse"] = {
            "matrix_gen_row_obs_col": M.round(5).tolist(),
            "diagonal_is_row_min": bool((M.argmin(1) == np.arange(4)).all())}

# ---------------- Fig. 4 comparison to the paper, and the claims ----------------
for track in TRACKS:
    f4 = {}
    for metric, k in (("rmse", "rmse_rootmean"), ("es", "es_eq14"), ("vg", "vg_p05_ordered")):
        f4[metric] = {}
        for gen in GENS:
            med = [res[track]["scores"][f"{gen}_c{c}"][k]["median"] for c in range(4)]
            f4[metric][gen] = {"median": med, "paper": PAPER[metric][gen],
                               "ratio_to_paper": [med[c] / PAPER[metric][gen][c] for c in range(4)]}
        f4[metric]["gam_below_cvae_all_clusters"] = bool(all(f4[metric]["gam"]["median"][c] < f4[metric]["cvae"]["median"][c] for c in range(4)))
    f4["rmse_euclid_over_paper"] = {gen: [res[track]["scores"][f"{gen}_c{c}"]["rmse_euclid"]["median"] / PAPER["rmse"][gen][c] for c in range(4)] for gen in GENS}
    res[track]["fig4"] = f4
    # ES CVAE vs GAM
    esc = {}
    for c in range(4):
        a = perday[(track, "cvae", c)]["es"]
        b = perday[(track, "gam", c)]["es"]
        w = stats.wilcoxon(a, b)
        esc[c] = {"median_ratio_cvae_over_gam": float(np.median(a) / np.median(b)),
                  "frac_days_cvae_lower": float((a < b).mean()), "wilcoxon_p": float(w.pvalue)}
    res[track]["es_cvae_vs_gam"] = esc
    # RMSE difference
    rc = np.stack([perday[(track, "cvae", c)]["rmse"] for c in range(4)])
    rg = np.stack([perday[(track, "gam", c)]["rmse"] for c in range(4)])
    mc, mg = np.median(rc), np.median(rg)
    pcm_c = np.median(rc, axis=1)
    pcm_g = np.median(rg, axis=1)
    rel_c = (rc - rg) / rc
    rel_g = (rc - rg) / rg
    res[track]["rmse_difference"] = {
        "median_over_cluster_days_of_rmse": {"cvae": float(mc), "gam": float(mg),
                                             "rel_cvae_pct": float(100 * (mc - mg) / mc), "rel_gam_pct": float(100 * (mc - mg) / mg)},
        "median_over_cluster_days_of_paired_rel_diff": {"rel_cvae_pct": float(100 * np.median(rel_c)), "rel_gam_pct": float(100 * np.median(rel_g))},
        "mean_over_cluster_days_of_paired_rel_diff": {"rel_cvae_pct": float(100 * rel_c.mean()), "rel_gam_pct": float(100 * rel_g.mean())},
        "mean_over_cluster_days_of_rmse": {"rel_cvae_pct": float(100 * (rc.mean() - rg.mean()) / rc.mean()), "rel_gam_pct": float(100 * (rc.mean() - rg.mean()) / rg.mean())},
        "pooled_rms": {"rel_cvae_pct": float(100 * (np.sqrt((rc ** 2).mean()) - np.sqrt((rg ** 2).mean())) / np.sqrt((rc ** 2).mean())),
                       "rel_gam_pct": float(100 * (np.sqrt((rc ** 2).mean()) - np.sqrt((rg ** 2).mean())) / np.sqrt((rg ** 2).mean()))},
        "per_cluster_median_rel_cvae_pct": (100 * (pcm_c - pcm_g) / pcm_c).tolist(),
        "per_cluster_median_rel_gam_pct": (100 * (pcm_c - pcm_g) / pcm_g).tolist(),
        "mean_of_per_cluster_medians_ratio_of_means": {"rel_cvae_pct": float(100 * (pcm_c.mean() - pcm_g.mean()) / pcm_c.mean()),
                                                       "rel_gam_pct": float(100 * (pcm_c.mean() - pcm_g.mean()) / pcm_g.mean())},
        "mean_of_per_cluster_pct_differences": {"rel_cvae_pct": float(np.mean(100 * (pcm_c - pcm_g) / pcm_c)),
                                                "rel_gam_pct": float(np.mean(100 * (pcm_c - pcm_g) / pcm_g))},
        "frac_cluster_days_gam_better": float((rg < rc).mean()),
        "frac_days_gam_better_per_cluster": (rg < rc).mean(1).tolist(),
        "paper_fig4_medians_implied": {"rel_cvae_pct": float(100 * (np.mean(PAPER['rmse']['cvae']) - np.mean(PAPER['rmse']['gam'])) / np.mean(PAPER['rmse']['cvae'])),
                                       "rel_gam_pct": float(100 * (np.mean(PAPER['rmse']['cvae']) - np.mean(PAPER['rmse']['gam'])) / np.mean(PAPER['rmse']['gam']))},
    }

# ---------------- scenario effects ----------------
REG = {"Low_in_9_18": ("Low", list(range(9, 19))), "High_in_39_43": ("High", list(range(39, 44))),
       "High_pre4_35_38": ("High", list(range(35, 39))), "High_post4_44_47": ("High", list(range(44, 48))),
       "High_pre2_37_38": ("High", [37, 38]), "High_post2_44_45": ("High", [44, 45]),
       "Low_19_31": ("Low", list(range(19, 32))), "Low_32_47": ("Low", list(range(32, 48))),
       "High_24_38": ("High", list(range(24, 39))), "High_24_34": ("High", list(range(24, 35)))}
rng = np.random.default_rng(12345)
BOOT = rng.integers(0, 92, size=(2000, 92))
for track in TRACKS:
    eff = {}
    lo_t, hi_t = None, None
    for gen in GENS:
        for c in range(4):
            sc = {s: np.load(REPRO + f"/outputs/{track}/{gen}_scen_{s}_c{c}.npy") for s in ("Normal", "Low", "High")}
            for s in sc:
                assert sc[s].shape == (N, 92, H)
            mN = sc["Normal"].mean(0)
            e = {}
            for name, (s, slots) in REG.items():
                mS = sc[s].mean(0)
                num_d = (mS[:, slots] - mN[:, slots]).mean(1)
                den_d = mN[:, slots].mean(1)
                pct = 100 * num_d.mean() / den_d.mean()
                bp = 100 * num_d[BOOT].mean(1) / den_d[BOOT].mean(1)
                lo, hi = np.percentile(bp, [2.5, 97.5])
                e[name] = {"pct": float(pct), "ci95": [float(lo), float(hi)], "kwh": float(num_d.mean()),
                           "frac_days_negative": float((num_d < 0).mean()),
                           "ci_excludes_0": bool(lo > 0 or hi < 0)}
            # where do the scenarios differ from Normal at all? (sample-level, any sample/day)
            dl = np.abs(sc["Low"] - sc["Normal"]).max(axis=(0, 1))
            dh = np.abs(sc["High"] - sc["Normal"]).max(axis=(0, 1))
            e["slots_where_Low_differs"] = [int(h) for h in np.nonzero(dl > 0)[0]]
            e["slots_where_High_differs"] = [int(h) for h in np.nonzero(dh > 0)[0]]
            e["max_abs_diff_outside_window_Low"] = float(np.delete(dl, range(9, 19)).max())
            e["max_abs_diff_outside_window_High"] = float(np.delete(dh, range(39, 44)).max())
            # CRN check vs the real-condition samples on test days whose real tariff is all Normal
            lowd, highd = tar_all[track][c]
            alln = (lowd.sum(1) == 0) & (highd.sum(1) == 0)
            S = np.load(REPRO + f"/outputs/{track}/{gen}_samples_c{c}.npy")
            e["n_test_days_all_normal"] = int(alln.sum())
            e["max_abs_samples_minus_scenNormal_on_all_normal_days"] = float(np.abs(S[:, alln] - sc["Normal"][:, alln]).max()) if alln.any() else None
            e["max_abs_samples_minus_scenNormal_on_event_days"] = float(np.abs(S[:, ~alln] - sc["Normal"][:, ~alln]).max()) if (~alln).any() else None
            eff[f"{gen}_c{c}"] = e
            print(f"{time.time()-t0:6.1f}s {track} {gen}_c{c} effects: Low {e['Low_in_9_18']['pct']:+.2f} High {e['High_in_39_43']['pct']:+.2f} "
                  f"pre4 {e['High_pre4_35_38']['pct']:+.2f} post4 {e['High_post4_44_47']['pct']:+.2f}", flush=True)
    res[track]["effects"] = eff

res["meta"] = {"script": OUTDIR + "/recompute_scores.py", "runtime_s": time.time() - t0,
               "bootstrap": "percentile over the 92 test days, B = 2000, numpy default_rng(12345)",
               "notes": "no import of lib/scores.py, lib/effects.py or scorer scripts; plain numpy/scipy"}
with open(OUTDIR + "/scores_recomputed.json", "w") as f:
    json.dump(res, f, indent=1)
print("done", time.time() - t0)
