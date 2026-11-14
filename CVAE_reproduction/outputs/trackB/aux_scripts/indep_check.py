"""Independent checks of the Track B (and Track A) scores + restart-level / Monte-Carlo comparison of the tracks.

Does NOT use lib/scores.py or lib/effects.py (own implementations with scipy), only lib.cvae_torch.generate /
load_model and lib.gam_generator.generate to regenerate samples.

Stages (all run by default):
  perday    my own per-day scores (RMSE root-mean, ES split eq. 14, ES U-statistic, VG p=0.5 ordered) and
            distribution diagnostics for both tracks, compared with outputs/track{A,B}/scores_long.csv and
            scores_summary.json
  boot      paired bootstrap over the 92 common test days of the Track B - Track A difference of the medians
  restarts  all 50 CVAE restarts per cluster and track regenerated with the phase-1 z draws -> per-restart medians
            of every score and diagnostic (outputs/trackB/scores_restarts_AB.csv); selected restart regenerated
            and compared with the saved cvae_samples arrays
  gammc     GAM Monte-Carlo noise: the saved GAM regenerated with the phase-1 seed (check) and 5 other seeds
Writes: outputs/trackB/scores_restarts_AB.csv, outputs/trackB/scores_independent_check.json
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
sys.path.insert(0, REPRO)
AUTHORS = REPRO + "/data/authors"  # copy of the authors' repo data/*.feather (byte-identical; read as data only)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import cdist, pdist  # noqa: E402
from scipy import stats  # noqa: E402

H = 48
N_SAMPLES = 200
GEN_SEED = 20200801
GAM_BASE_SEED = 20200727
COLS = [f"consumption_{h:02d}" for h in range(H)]
TEST = {"A": AUTHORS + "/test_nmf.feather", "B": REPRO + "/data/test_ours.parquet"}
OUT = {"A": REPRO + "/outputs/trackA", "B": REPRO + "/outputs/trackB"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def read(path):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"])
    df["cluster"] = df["cluster"].astype(int)
    return df


def block(df, c):
    return df.loc[df["cluster"] == c].sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------
# my own score implementations (per day loops, scipy distances)
# ---------------------------------------------------------------------------------------------------------------
def my_scores(S, Y, with_u=True):
    N, D, _ = S.shape
    n2 = N // 2
    out = {k: np.empty(D) for k in ("rmse", "es_split", "es_u", "vg_ord")}
    for d in range(D):
        X, y = S[:, d, :], Y[d]
        out["rmse"][d] = np.linalg.norm(X.mean(0) - y) / math.sqrt(H)
        dobs = cdist(X, y[None, :]).ravel()
        pairs = np.linalg.norm(X[:n2] - X[n2:2 * n2], axis=1)
        out["es_split"][d] = dobs[:n2].mean() - 0.5 * pairs.mean()
        out["es_u"][d] = dobs.mean() - 0.5 * pdist(X).mean() if with_u else np.nan
        Ay = np.sqrt(np.abs(y[:, None] - y[None, :]))
        Es = np.sqrt(np.abs(X[:, :, None] - X[:, None, :])).mean(0)
        out["vg_ord"][d] = ((Ay - Es) ** 2).sum()
    return out


def uncentred_lag1(E):
    a, b = E[..., :-1].ravel(), E[..., 1:].ravel()
    return float((a * b).mean() / math.sqrt((a * a).mean() * (b * b).mean()))


def my_diag(S, Y, gam_mean=None):
    N = S.shape[0]
    m = S.mean(0)
    var = S.var(0, ddof=1)
    err2 = (m - Y) ** 2
    q05, q25, q75, q95 = np.percentile(S, [5, 25, 75, 95], axis=0)
    rank = (S < Y[None]).sum(0) + 0.5 * (S == Y[None]).sum(0)
    pit = (rank + 0.5) / (N + 1)
    X = S - m[None]
    num = (X[..., :-1] * X[..., 1:]).sum(0)
    den = np.sqrt((X[..., :-1] ** 2).sum(0) * (X[..., 1:] ** 2).sum(0))
    lag_across = float(np.nanmean(num / den))
    out = {"spread_skill": float(math.sqrt(var.mean()) / math.sqrt(err2.mean()) * math.sqrt((N + 1) / N)),
           "mean_sd_over_rmse": float(np.sqrt(var).mean() / math.sqrt(err2.mean())),
           "coverage90": float(((Y >= q05) & (Y <= q95)).mean()),
           "coverage50": float(((Y >= q25) & (Y <= q75)).mean()),
           "pit_var": float(pit.var()), "pit_tails_5pct": float((pit < 0.05).mean() + (pit > 0.95).mean()),
           "frac_outside_range": float(((Y < S.min(0)) | (Y > S.max(0))).mean()),
           "roughness_ratio": float(np.abs(np.diff(S, axis=-1)).mean() / np.abs(np.diff(Y, axis=-1)).mean()),
           "lag1_across_samples": lag_across,
           "lag1_resid_vs_own_mean": uncentred_lag1(Y - m),
           "profile_lag1_samples": uncentred_lag1(S - S.mean(-1, keepdims=True)),
           "profile_lag1_observed": uncentred_lag1(Y - Y.mean(-1, keepdims=True)),
           "sample_sd_kwh_mean": float(np.sqrt(var).mean())}
    if gam_mean is not None:
        out["lag1_resid_vs_gam_mean"] = uncentred_lag1(Y - gam_mean)
    return out


def main():
    t0 = time.perf_counter()
    res = {"stages": {}}
    obs = {}
    for t in ("A", "B"):
        te = read(TEST[t])
        obs[t] = {c: block(te, c)[COLS].to_numpy(np.float64) for c in range(4)}
        dates = block(te, 0)["date"].dt.strftime("%Y-%m-%d").tolist()
        obs[t]["dates"] = dates
    assert obs["A"]["dates"] == obs["B"]["dates"], "test dates differ between tracks"
    # ------------------------------------------------------------------ perday
    per = {}
    chk = {}
    diag = {}
    for t in ("A", "B"):
        long = pd.read_csv(OUT[t] + "/scores_long.csv")
        summ = json.load(open(OUT[t] + "/scores_summary.json"))
        gam_means = {c: np.load(OUT[t] + f"/gam_samples_c{c}.npy").mean(0) for c in range(4)}
        for m in ("cvae", "gam"):
            for c in range(4):
                S = np.load(OUT[t] + f"/{m}_samples_c{c}.npy")
                Y = obs[t][c]
                sc = my_scores(S, Y)
                per[(t, m, c)] = sc
                L = long.loc[(long.model == m) & (long.cluster == c)].sort_values("date")
                assert L["date"].tolist() == obs[t]["dates"]
                d = {}
                for mine, theirs in (("rmse", "rmse_rootmean"), ("es_split", "es_split"), ("es_u", "es_u"),
                                     ("vg_ord", "vg_p05_ordered")):
                    d[mine] = float(np.max(np.abs(sc[mine] - L[theirs].to_numpy()) /
                                           np.maximum(1e-12, np.abs(L[theirs].to_numpy()))))
                dg = my_diag(S, Y, gam_mean=gam_means[c])
                diag[(t, m, c)] = dg
                lib = summ["distribution_diagnostics"]["per_model_cluster"][f"{m}_c{c}"]
                d["diag_vs_lib"] = {
                    "spread_skill": abs(dg["spread_skill"] - lib["spread_skill"]["ratio_calibrated"]),
                    "coverage90": abs(dg["coverage90"] - lib["coverage_central90"]),
                    "coverage50": abs(dg["coverage50"] - lib["coverage_central50"]),
                    "roughness_ratio": abs(dg["roughness_ratio"] - lib["roughness"]["ratio_samples_to_observed"]),
                    "lag1_across": abs(dg["lag1_across_samples"] - lib["lag1"]["across_samples_mean_over_day_slot"]),
                    "lag1_resid_own": abs(dg["lag1_resid_vs_own_mean"] - lib["lag1"]["residual_pooled"]),
                    "pit_var": abs(dg["pit_var"] - lib["pit"]["var"])}
                chk[f"{t}_{m}_c{c}"] = d
        log(f"perday track {t} done")
    maxrel = {k: max(v[k] for v in chk.values()) for k in ("rmse", "es_split", "es_u", "vg_ord")}
    maxdiag = {k: max(v["diag_vs_lib"][k] for v in chk.values()) for k in next(iter(chk.values()))["diag_vs_lib"]}
    log(f"max rel diff my scores vs scores_long.csv: {maxrel}; diagnostics max abs diff {maxdiag}")
    res["stages"]["perday_check"] = {"max_rel_diff_vs_scores_long": maxrel, "max_abs_diff_diagnostics": maxdiag,
                                     "per_block": chk}
    meds = {f"{t}_{m}_c{c}": {k: float(np.median(v)) for k, v in sc.items()} for (t, m, c), sc in per.items()}
    res["my_medians"] = meds
    res["my_diagnostics"] = {f"{t}_{m}_c{c}": v for (t, m, c), v in diag.items()}
    # 13% readings, recomputed
    r13 = {}
    for t in ("A", "B"):
        cv = np.concatenate([per[(t, "cvae", c)]["rmse"] for c in range(4)])
        gm = np.concatenate([per[(t, "gam", c)]["rmse"] for c in range(4)])
        mc = [np.median(per[(t, "cvae", c)]["rmse"]) for c in range(4)]
        mg = [np.median(per[(t, "gam", c)]["rmse"]) for c in range(4)]
        r13[t] = {"median_all_rel_cvae": 100 * (np.median(cv) - np.median(gm)) / np.median(cv),
                  "median_all_rel_gam": 100 * (np.median(cv) - np.median(gm)) / np.median(gm),
                  "mean_all_rel_cvae": 100 * (cv.mean() - gm.mean()) / cv.mean(),
                  "mean_of_cluster_medians_rel_cvae": 100 * (np.mean(mc) - np.mean(mg)) / np.mean(mc),
                  "mean_of_cluster_medians_rel_gam": 100 * (np.mean(mc) - np.mean(mg)) / np.mean(mg),
                  "per_cluster_median_rel_cvae": [100 * (a - b) / a for a, b in zip(mc, mg)],
                  "frac_days_gam_better": float(np.mean(gm < cv))}
    res["rmse_13pct_recomputed"] = r13
    log(f"13% readings: {json.dumps(r13, default=float)[:400]}")
    # ------------------------------------------------------------------ boot (paired over common test days)
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, 92, (4000, 92))
    boot = {}
    for m in ("cvae", "gam"):
        for c in range(4):
            for k in ("rmse", "es_split", "vg_ord"):
                a, b = per[("A", m, c)][k], per[("B", m, c)][k]
                dA, dB = np.median(a[idx], 1), np.median(b[idx], 1)
                diff = dB - dA
                ratio = dB / dA
                boot[f"{m}_c{c}_{k}"] = {"median_A": float(np.median(a)), "median_B": float(np.median(b)),
                                         "diff_B_minus_A": float(np.median(b) - np.median(a)),
                                         "diff_ci95": [float(np.quantile(diff, .025)), float(np.quantile(diff, .975))],
                                         "ratio_B_over_A": float(np.median(b) / np.median(a)),
                                         "ratio_ci95": [float(np.quantile(ratio, .025)),
                                                        float(np.quantile(ratio, .975))],
                                         "paired_day_wilcoxon_p": float(stats.wilcoxon(a, b).pvalue)}
    # gap CVAE-GAM per track and its change
    for c in range(4):
        for k in ("rmse", "es_split", "vg_ord"):
            ca, ga = per[("A", "cvae", c)][k], per[("A", "gam", c)][k]
            cb, gb = per[("B", "cvae", c)][k], per[("B", "gam", c)][k]
            rA = np.median(ca[idx], 1) / np.median(ga[idx], 1)
            rB = np.median(cb[idx], 1) / np.median(gb[idx], 1)
            boot[f"gap_c{c}_{k}"] = {"cvae_over_gam_A": float(np.median(ca) / np.median(ga)),
                                     "cvae_over_gam_B": float(np.median(cb) / np.median(gb)),
                                     "ci95_A": [float(np.quantile(rA, .025)), float(np.quantile(rA, .975))],
                                     "ci95_B": [float(np.quantile(rB, .025)), float(np.quantile(rB, .975))],
                                     "diff_B_minus_A_ci95": [float(np.quantile(rB - rA, .025)),
                                                             float(np.quantile(rB - rA, .975))]}
    res["stages"]["paired_bootstrap_days"] = boot
    log("bootstrap done")
    # ------------------------------------------------------------------ restarts
    import torch
    from lib import cvae_torch as ct
    torch.set_num_threads(1)
    rows = []
    regen = {}
    for t in ("A", "B"):
        te = read(TEST[t])
        for c in range(4):
            blk = block(te, c)
            Y = obs[t][c]
            sel_seed = int(ct.load_model(OUT[t] + f"/models/cvae_selected_c{c}.pt").meta["seed"])
            gam_mean = np.load(OUT[t] + f"/gam_samples_c{c}.npy").mean(0)
            for s in range(50):
                p = OUT[t] + f"/models/cvae_c{c}_s{s:02d}.pt"
                model = ct.load_model(p)
                cols = model.meta["cond_cols"]
                C = np.array(blk[cols].to_numpy(dtype=np.float64), copy=True)
                eps = np.random.default_rng([GEN_SEED, c]).standard_normal((N_SAMPLES, len(C), model.latent_dim))
                S = ct.generate(model, C, N_SAMPLES, eps=eps)
                if s == sel_seed:
                    regen[f"{t}_c{c}"] = float(np.abs(S - np.load(OUT[t] + f"/cvae_samples_c{c}.npy")).max())
                sc = my_scores(S, Y, with_u=False)
                dg = my_diag(S, Y, gam_mean=gam_mean)
                rows.append({"track": t, "cluster": c, "seed": s, "selected": s == sel_seed,
                             "rmse_median": float(np.median(sc["rmse"])), "rmse_mean": float(sc["rmse"].mean()),
                             "es_split_median": float(np.median(sc["es_split"])),
                             "vg_ord_median": float(np.median(sc["vg_ord"])),
                             **{k: v for k, v in dg.items()}})
            log(f"restarts track {t} c{c} done (selected seed {sel_seed}, regen max diff {regen[f'{t}_c{c}']})")
    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT["B"] + "/scores_restarts_AB.csv", index=False)
    comp = {}
    metrics = ["rmse_median", "es_split_median", "vg_ord_median", "spread_skill", "coverage90", "roughness_ratio",
               "lag1_across_samples", "profile_lag1_samples", "sample_sd_kwh_mean"]
    for c in range(4):
        for k in metrics:
            a = rdf.loc[(rdf.track == "A") & (rdf.cluster == c), k].to_numpy()
            b = rdf.loc[(rdf.track == "B") & (rdf.cluster == c), k].to_numpy()
            sa = float(rdf.loc[(rdf.track == "A") & (rdf.cluster == c) & rdf.selected, k].iloc[0])
            sb = float(rdf.loc[(rdf.track == "B") & (rdf.cluster == c) & rdf.selected, k].iloc[0])
            mw = stats.mannwhitneyu(a, b)
            comp[f"c{c}_{k}"] = {"A_selected": sa, "B_selected": sb,
                                 "A_restarts_median": float(np.median(a)), "B_restarts_median": float(np.median(b)),
                                 "A_restarts_q10_q90": [float(np.quantile(a, .1)), float(np.quantile(a, .9))],
                                 "B_restarts_q10_q90": [float(np.quantile(b, .1)), float(np.quantile(b, .9))],
                                 "A_restarts_sd": float(a.std(ddof=1)), "B_restarts_sd": float(b.std(ddof=1)),
                                 "mannwhitney_p": float(mw.pvalue),
                                 "prob_B_restart_gt_A_restart": float(np.mean(b[:, None] > a[None, :])),
                                 "B_selected_pct_rank_in_A_restarts": float(100 * np.mean(a < sb)),
                                 "A_selected_pct_rank_in_B_restarts": float(100 * np.mean(b < sa))}
    res["stages"]["restarts"] = {"csv": OUT["B"] + "/scores_restarts_AB.csv",
                                 "selected_regenerated_max_abs_diff_vs_saved": regen, "comparison": comp}
    log("restart comparison done")
    # ------------------------------------------------------------------ GAM Monte-Carlo noise
    from lib.gam_generator import generate as gam_generate
    gmc = {}
    for t in ("A", "B"):
        te = read(TEST[t])
        for c in range(4):
            with open(OUT[t] + f"/models/gam_params_c{c}.pkl", "rb") as fh:
                P = pickle.load(fh)
            tec = block(te, c)
            Y = obs[t][c]
            seed_samples, _ = np.random.SeedSequence([GAM_BASE_SEED, c]).spawn(2)
            S0 = gam_generate(P, tec, N_SAMPLES, np.random.default_rng(seed_samples))
            same = float(np.abs(S0 - np.load(OUT[t] + f"/gam_samples_c{c}.npy")).max())
            vals = {k: [] for k in ("rmse", "es_split", "vg_ord")}
            for r in range(5):
                S = gam_generate(P, tec, N_SAMPLES, np.random.default_rng([777, r, c]))
                sc = my_scores(S, Y, with_u=False)
                for k in vals:
                    vals[k].append(float(np.median(sc[k])))
            gmc[f"{t}_c{c}"] = {"regen_phase1_seed_max_abs_diff": same,
                                **{f"{k}_median_over_5_seeds": v for k, v in vals.items()},
                                **{f"{k}_mc_sd": float(np.std(v, ddof=1)) for k, v in vals.items()}}
            log(f"gam mc {t} c{c}: regen diff {same:.2e}, sd rmse {gmc[f'{t}_c{c}']['rmse_mc_sd']:.2e}, "
                f"es {gmc[f'{t}_c{c}']['es_split_mc_sd']:.2e}, vg {gmc[f'{t}_c{c}']['vg_ord_mc_sd']:.3f}")
    res["stages"]["gam_mc"] = gmc
    res["runtime_s"] = time.perf_counter() - t0
    with open(OUT["B"] + "/scores_independent_check.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"wrote scores_independent_check.json and scores_restarts_AB.csv in {res['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
