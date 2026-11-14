"""Track A: the authors' CVAE (PyTorch float64 port, lib/cvae_torch.py) trained on the authors' own prepared data.

Inputs (read-only, absolute paths; the upstream repo is never imported or executed):
  UPSTREAM/data/train_nmf.feather, test_nmf.feather   exact 273/92-day split, 4 NMF clusters
  UPSTREAM/data/power_consump_nmf.feather              365 days x 4 clusters (consumption PCA claim)
  UPSTREAM/data/sum_up_nmf.feather                     cluster sizes

Stages (run in this order with --stage all, the default; every stage is re-runnable):
  pca       "4 components explain > 80% of the variance" -> outputs/trackA/cvae_pca_consumption.json
  train     4 clusters x 50 restarts (seeds 0..49) in a pool of 4 worker processes (1 torch thread each).
            Per restart: outputs/trackA/models/cvae_c<c>_s<seed>.pt + .json. Finished restarts are skipped
            unless --force (so an interrupted run resumes).
  collect   outputs/trackA/cvae_restarts.csv (+ cvae_restarts_epoch_loss.csv)
  select    per cluster the restart with the lowest test MSE (authors' compute_mse: z ~ encoder posterior,
            100*sqrt(mean_h err^2) on [0,1]-scaled data, averaged over the 92 test days; 1 draw, fixed RNG)
            -> outputs/trackA/models/cvae_selected_c<c>.pt, outputs/trackA/cvae_selection.json
  generate  200 samples per test day, z ~ N(0, I_4) -> outputs/trackA/cvae_samples_c<c>.npy (200, 92, 48) kWh;
            tariff scenarios -> outputs/trackA/cvae_scen_<Normal|Low|High>_c<c>.npy (Normal = all 96 tariff
            dummies 0; Low = low slots 9-19 = 04:00-09:30 GMT; High = high slots 39-44 = 19:00-22:00 GMT; windows
            from lib.effects, REPRO_SPEC section 7); outputs/trackA/test_dates.csv. The same z draws are used for the
            real-condition samples and for the three scenarios of a cluster (common random numbers), so scenario
            differences carry no MC noise from z. An array identical to the stored one is not rewritten.
  effects   scenario mean differences (in-window, side, rebound; lib.effects.phase1_regions)
            -> outputs/trackA/cvae_scenario_effects.{json,csv}
  robust    the same region effects for every one of the 50 restarts per cluster (is the selected restart's tariff
            response typical?) -> outputs/trackA/cvae_scenario_effects_all_restarts.{csv,json}
  quirks    upstream-quirk emulation, 20 restarts per cluster: identical Glorot init for every restart (a seeded TF2
            initializer is stateless) + Adam without bias correction (one Keras Adam shared by all restarts)
            -> outputs/trackA/models/cvae_upq_c<c>_s<seed>.{pt,json}, outputs/trackA/cvae_sensitivity_upstream_quirks.json
            (restarts already on disk are only re-scored, not retrained)
  figures   outputs/figures/trackA_cvae_*.png (Fig. 5 replica for clusters 3 and 2: the paper's Fig. 5 shows cluster 2)
  summary   outputs/trackA/cvae_summary.json

Usage (from PROJECT or REPRO):  python -P scripts/10_trackA_cvae.py [--stage all|pca|train|...] [--workers 4]
                                [--no-train]   (refuse to train anything: re-generation / re-scoring only)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
if REPRO not in sys.path:
    sys.path.insert(0, REPRO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import cvae_torch as ct  # noqa: E402
from lib import effects as ef  # noqa: E402  (scenario windows and regions: single source of truth)

DATA = REPRO + "/data/authors"  # copy of the authors' repo data/*.feather (byte-identical; read as data only)
TRAIN_PATH = DATA + "/train_nmf.feather"
TEST_PATH = DATA + "/test_nmf.feather"
POWER_PATH = DATA + "/power_consump_nmf.feather"
SUMUP_PATH = DATA + "/sum_up_nmf.feather"
OUT = REPRO + "/outputs/trackA"
MODELS = OUT + "/models"
FIGS = REPRO + "/outputs/figures"
LOG_PATH = OUT + "/cvae_run_log.txt"

CLUSTERS = (0, 1, 2, 3)
N_RESTARTS = 50
N_SAMPLES = 200
GEN_SEED = 20200801            # z ~ N(0, I) for samples and scenarios: rng = default_rng([GEN_SEED, cluster])
LOW_SLOTS = list(ef.LOW_WINDOW)     # slots 9-19 = 04:00-09:30 GMT (lib.effects; REPRO_SPEC section 7)
HIGH_SLOTS = list(ef.HIGH_WINDOW)   # slots 39-44 = 19:00-22:00 GMT
SCENARIOS = ("Normal", "Low", "High")
FIG5_CLUSTERS = (3, 2)          # Fig. 5 replicas: 3 (the caption's 158 households) and 2 (what the paper's Fig. 5 shows)
FIG5_NOTE = {2: "the paper's Fig. 5 shows this cluster, 229 households",
             3: "the cluster named in the paper's caption; the figure itself shows cluster 2"}

# chart tokens (dataviz reference palette, light surface)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SCEN_COLOR = {"Normal": BLUE, "Low": AQUA, "High": ORANGE}
SCEN_LABEL = {"Normal": "Normal tariff all day",
              "Low": f"Low {ef.WINDOW_CLOCK['Low']} GMT (slots {ef.WINDOW_SPAN['Low']}), Normal otherwise",
              "High": f"High {ef.WINDOW_CLOCK['High']} GMT (slots {ef.WINDOW_SPAN['High']}), Normal otherwise"}
# x axis of the time-of-day figures: slot h is drawn at h/2 h, the END of its half-hour (end-labelled slots, GMT)
TIME_AXIS_LABEL = "Time of day, GMT (the point at time t is the half-hour ending at t)"


# ------------------------------------------------------------------------------------------------------------------
def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load(path: str) -> pd.DataFrame:
    df = pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"])
    if "cluster" in df.columns:
        df["cluster"] = df["cluster"].astype(int)
    return df


def model_path(c: int, s: int) -> str:
    return f"{MODELS}/cvae_c{c}_s{s:02d}.pt"


def json_path(c: int, s: int) -> str:
    return f"{MODELS}/cvae_c{c}_s{s:02d}.json"


def selected_path(c: int) -> str:
    return f"{MODELS}/cvae_selected_c{c}.pt"


def cluster_sizes() -> dict:
    su = pd.read_feather(SUMUP_PATH)
    col = [k for k in su.columns if k != "cluster"][0]
    return {int(r["cluster"]): int(r[col]) for _, r in su.iterrows()}


def test_block(te: pd.DataFrame, c: int) -> pd.DataFrame:
    return te.loc[te["cluster"] == c].sort_values("date", kind="stable").reset_index(drop=True)


def save_array(path: str, arr: np.ndarray) -> str:
    """np.save, unless the file already holds a byte-identical array (then it is left untouched).
    Returns 'unchanged', 'changed' or 'new'."""
    arr = np.ascontiguousarray(arr)
    if os.path.exists(path):
        old = np.load(path)
        if old.dtype == arr.dtype and old.shape == arr.shape and old.tobytes() == arr.tobytes():
            return "unchanged"
        np.save(path, arr)
        return "changed"
    np.save(path, arr)
    return "new"


def scenario_cond(C: np.ndarray, cond_cols: list, name: str) -> np.ndarray:
    """Only the 96 tariff dummies change. Normal: all 0; Low: low_09..low_19 = 1 (04:00-09:30 GMT);
    High: high_39..high_44 = 1 (19:00-22:00 GMT); everything else = the real test-day conditions.
    Windows: lib.effects.LOW_WINDOW / HIGH_WINDOW (REPRO_SPEC section 7)."""
    C = np.array(C, dtype=np.float64, copy=True)
    il = [cond_cols.index(f"low_{h:02d}") for h in range(48)]
    ih = [cond_cols.index(f"high_{h:02d}") for h in range(48)]
    C[:, il] = 0.0
    C[:, ih] = 0.0
    if name == "Low":
        C[:, [il[h] for h in LOW_SLOTS]] = 1.0
    elif name == "High":
        C[:, [ih[h] for h in HIGH_SLOTS]] = 1.0
    elif name == "AllHigh":                       # diagnostic only (never-seen tariff profile), not saved
        C[:, ih] = 1.0
    elif name != "Normal":
        raise ValueError(name)
    return C


# ------------------------------------------------------------------------------------------------------------------
# PCA claim
# ------------------------------------------------------------------------------------------------------------------
def _pca_stats(X: np.ndarray, center: bool = True) -> dict:
    X = np.asarray(X, dtype=np.float64)
    if center:
        X = X - X.mean(0)
    s = np.linalg.svd(X, compute_uv=False)
    evr = s ** 2 / np.sum(s ** 2)
    cum = np.cumsum(evr)
    return {"cum_evr_1_to_8": [round(float(v), 5) for v in cum[:8]], "cum_evr_at_4": float(cum[3]),
            "n_components_for_80pct": int(np.searchsorted(cum, 0.80) + 1),
            "n_components_for_90pct": int(np.searchsorted(cum, 0.90) + 1)}


def stage_pca(args) -> dict:
    from sklearn.decomposition import PCA
    pw = load(POWER_PATH)
    tr = load(TRAIN_PATH)
    res = {"claim": "Sec. V-B-2: a PCA on the consumption data: 4 components explain more than 80% of the variance",
           "data": "power_consump_nmf.feather: 365 daily 48-half-hour profiles per cluster (cluster-mean kWh)",
           "method": "PCA = SVD of the column-centred matrix (sklearn PCA cross-check on raw data); "
                     "'uncentred' = share of the total sum of squares (no centring)",
           "clusters": {}}
    for c in CLUSTERS:
        Y = pw.loc[pw["cluster"] == c].sort_values("date")[ct.COLS_DATA].to_numpy(dtype=np.float64)
        assert Y.shape == (365, 48), Y.shape
        Ytr = tr.loc[tr["cluster"] == c][ct.COLS_DATA].to_numpy(dtype=np.float64)
        tmin, tmax = Ytr.min(), Ytr.max()
        sd = Y.std(0, ddof=0)
        variants = {
            "raw_kWh": (Y, True),
            "minmax_global_train_minmax (CVAE scaling)": ((Y - tmin) / (tmax - tmin), True),
            "minmax_per_halfhour": ((Y - Y.min(0)) / (Y.max(0) - Y.min(0)), True),
            "standardized_per_halfhour": ((Y - Y.mean(0)) / sd, True),
            "raw_kWh_uncentred": (Y, False),
            "raw_kWh_train_days_only": (Ytr, True),
        }
        out = {k: _pca_stats(X, center=cen) for k, (X, cen) in variants.items()}
        sk = PCA().fit(Y)
        sk_cum4 = float(np.cumsum(sk.explained_variance_ratio_)[3])
        assert abs(sk_cum4 - out["raw_kWh"]["cum_evr_at_4"]) < 1e-10
        out["sklearn_PCA_raw_cum_evr_at_4"] = sk_cum4
        res["clusters"][str(c)] = out
        log(f"[pca] cluster {c}: cum EVR@4 raw {out['raw_kWh']['cum_evr_at_4']:.4f} | global min-max "
            f"{out['minmax_global_train_minmax (CVAE scaling)']['cum_evr_at_4']:.4f} | per-half-hour min-max "
            f"{out['minmax_per_halfhour']['cum_evr_at_4']:.4f} | standardized "
            f"{out['standardized_per_halfhour']['cum_evr_at_4']:.4f} | n for 80% (raw) "
            f"{out['raw_kWh']['n_components_for_80pct']} | uncentred {out['raw_kWh_uncentred']['cum_evr_at_4']:.4f}")
    allY = pw.sort_values(["cluster", "date"])[ct.COLS_DATA].to_numpy(dtype=np.float64)
    res["all_clusters_stacked_raw"] = _pca_stats(allY)
    # extra context (not this agent's claim): temperature PCA "3 components explain 98%"
    t0 = pw.loc[pw["cluster"] == 0].sort_values("date")
    T = t0[[f"temp_{i:02d}" for i in range(48)] + ["temp_smooth"]].to_numpy(dtype=np.float64)
    res["extra_temperature_pca_48temps_plus_smooth"] = {
        "raw_degC": _pca_stats(T), "standardized": _pca_stats((T - T.mean(0)) / T.std(0)),
        "note": "context only; the temperature PCA claim (3 components ~98%) is checked by the conditions task"}
    vals = [res["clusters"][str(c)]["raw_kWh"]["cum_evr_at_4"] for c in CLUSTERS]
    vals_std = [res["clusters"][str(c)]["standardized_per_halfhour"]["cum_evr_at_4"] for c in CLUSTERS]
    res["verdict_inputs"] = {"cum_evr_at_4_raw_per_cluster": vals, "all_above_80pct_raw": bool(min(vals) > 0.8),
                             "cum_evr_at_4_standardized_per_cluster": vals_std,
                             "all_above_80pct_standardized": bool(min(vals_std) > 0.8)}
    with open(OUT + "/cvae_pca_consumption.json", "w") as f:
        json.dump(res, f, indent=1)
    log(f"[pca] all-clusters stacked raw cum EVR@4 {res['all_clusters_stacked_raw']['cum_evr_at_4']:.4f}; "
        f"temperature PCA cum EVR@3 raw "
        f"{res['extra_temperature_pca_48temps_plus_smooth']['raw_degC']['cum_evr_1_to_8'][2]:.4f}")
    return res


# ------------------------------------------------------------------------------------------------------------------
# training (pool of workers)
# ------------------------------------------------------------------------------------------------------------------
def stage_train(args) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(MODELS, exist_ok=True)
    jobs = []
    for s in range(args.restarts):
        for c in args.clusters:
            if not args.force and os.path.exists(json_path(c, s)) and os.path.exists(model_path(c, s)):
                continue
            jobs.append({"train_path": TRAIN_PATH, "test_path": TEST_PATH, "cluster": c, "seed": s,
                         "model_path": model_path(c, s), "json_path": json_path(c, s), "epochs": args.epochs})
    log(f"[train] {len(jobs)} restarts to run with {args.workers} workers (epochs={args.epochs})")
    if not jobs:
        return
    if args.no_train:
        raise SystemExit(f"[train] --no-train: {len(jobs)} restarts would be trained; refusing")
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=ct.worker_init) as ex:
        futs = {ex.submit(ct.run_restart_job, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            log(f"[train] {done:3d}/{len(jobs)} c{r['cluster']} s{r['seed']:02d} test_mse_post {r['test_mse_post']:.4f} "
                f"prior {r['test_mse_prior']:.4f} last-epoch loss {r['train_loss_last_epoch']:.4f} "
                f"train KL {r['train_kl_final']:.2e} | {r['runtime_train_s']:.1f}s | wall {time.perf_counter() - t0:.0f}s")
    wall = time.perf_counter() - t0
    rec_path = OUT + "/cvae_train_walltime.json"
    hist = []
    if os.path.exists(rec_path):
        with open(rec_path) as f:
            hist = json.load(f).get("invocations", [])
    hist.append({"n_jobs": len(jobs), "workers": args.workers, "epochs": args.epochs, "wall_s": wall,
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    with open(rec_path, "w") as f:
        json.dump({"invocations": hist}, f, indent=1)
    log(f"[train] done: {len(jobs)} restarts in {wall:.0f}s wall")


def stage_collect(args) -> pd.DataFrame:
    rows, curves = [], []
    missing = []
    for c in CLUSTERS:
        for s in range(N_RESTARTS):
            p = json_path(c, s)
            if not os.path.exists(p):
                missing.append((c, s))
                continue
            with open(p) as f:
                r = json.load(f)
            row = {k: v for k, v in r.items() if isinstance(v, (int, float, str)) and k != "model_path"}
            for name in ("train_kl_per_dim", "train_posterior_std_mean", "train_posterior_mean_std"):
                for i, v in enumerate(r[name]):
                    row[f"{name}_{i}"] = v
            row["epoch_loss_first"] = r["epoch_loss"][0]
            row["epoch_loss_rel_change_last5"] = (r["epoch_loss"][-6] - r["epoch_loss"][-1]) / r["epoch_loss"][-6]
            rows.append(row)
            curves += [{"cluster": c, "seed": s, "epoch": e + 1, "mean_step_loss": v}
                       for e, v in enumerate(r["epoch_loss"])]
    if missing:
        log(f"[collect] WARNING {len(missing)} restarts missing: {missing[:10]}")
    df = pd.DataFrame(rows).sort_values(["cluster", "seed"]).reset_index(drop=True)
    df.to_csv(OUT + "/cvae_restarts.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT + "/cvae_restarts_epoch_loss.csv", index=False)
    log(f"[collect] {len(df)} restarts -> cvae_restarts.csv")
    return df


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def stage_select(args) -> dict:
    df = pd.read_csv(OUT + "/cvae_restarts.csv")
    sel = {"selection_metric": "test_mse_post = authors' compute_mse on the 92 test days (posterior z, one draw per "
                               "day, fixed RNG seed 0, common to all restarts); lowest wins",
           "clusters": {}}
    for c in CLUSTERS:
        d = df.loc[df["cluster"] == c].reset_index(drop=True)
        if len(d) != N_RESTARTS:
            raise RuntimeError(f"cluster {c}: {len(d)} restarts found, expected {N_RESTARTS}")
        ib = int(d["test_mse_post"].idxmin())
        b = d.loc[ib]
        s = int(b["seed"])
        shutil.copyfile(model_path(c, s), selected_path(c))
        m = d["test_mse_post"]
        avg = d["test_mse_post_avg20"]
        noise_sd = float(((d["test_mse_post"] - d["test_mse_post_seed1"]) / np.sqrt(2.0)).std(ddof=1))
        rm = d["test_rmse_mean200_kwh_median"]
        info = {
            "selected_seed": s,
            "test_mse_post_selected": float(b["test_mse_post"]),
            "test_mse_post_median": float(m.median()), "test_mse_post_mean": float(m.mean()),
            "test_mse_post_worst": float(m.max()), "test_mse_post_sd": float(m.std(ddof=1)),
            "test_mse_post_q10_q90": [float(m.quantile(0.1)), float(m.quantile(0.9))],
            "best_vs_median_rel_pct": float(100.0 * (m.median() - b["test_mse_post"]) / m.median()),
            "single_draw_metric_noise_sd": noise_sd,
            "between_restart_sd_over_metric_noise_sd": float(m.std(ddof=1) / noise_sd) if noise_sd > 0 else None,
            "selected_test_mse_post_seed1_redraw": float(b["test_mse_post_seed1"]),
            "selected_test_mse_post_avg20": float(b["test_mse_post_avg20"]),
            "avg20_best": float(avg.min()), "avg20_median": float(avg.median()),
            "selected_rank_in_avg20_metric (1=best)": int((avg < b["test_mse_post_avg20"]).sum() + 1),
            "selected_test_mse_prior": float(b["test_mse_prior"]),
            "test_mse_prior_median": float(d["test_mse_prior"].median()),
            "selected_rmse_mean200_kwh_median": float(b["test_rmse_mean200_kwh_median"]),
            "rmse_mean200_kwh_median_best": float(rm.min()), "rmse_mean200_kwh_median_median": float(rm.median()),
            "selected_rank_in_rmse_mean200 (1=best)": int((rm < b["test_rmse_mean200_kwh_median"]).sum() + 1),
            "spearman_selection_metric_vs_avg20": _spearman(m, avg),
            "spearman_selection_metric_vs_redraw_seed1": _spearman(m, d["test_mse_post_seed1"]),
            "spearman_selection_metric_vs_rmse_mean200": _spearman(m, rm),
            "selected_train_loss_last_epoch": float(b["train_loss_last_epoch"]),
            "selected_train_loss_final_fixed_eps": float(b["train_loss_final"]),
            "train_loss_last_epoch_median": float(d["train_loss_last_epoch"].median()),
            "selected_train_kl_final": float(b["train_kl_final"]),
            "train_kl_final_median": float(d["train_kl_final"].median()),
            "train_kl_final_max": float(d["train_kl_final"].max()),
            "selected_posterior_std_mean_per_dim": [float(b[f"train_posterior_std_mean_{i}"]) for i in range(4)],
            "selected_sample_sd_kwh_mean": float(b["test_sample_sd_kwh_mean"]),
            "epoch_loss_rel_change_last5_median": float(d["epoch_loss_rel_change_last5"].median()),
            "runtime_train_s_mean": float(d["runtime_train_s"].mean()),
            "runtime_train_s_total": float(d["runtime_train_s"].sum()),
        }
        sel["clusters"][str(c)] = info
        log(f"[select] c{c}: seed {s} test MSE {info['test_mse_post_selected']:.4f} (median {info['test_mse_post_median']:.4f}, "
            f"worst {info['test_mse_post_worst']:.4f}, best-vs-median {info['best_vs_median_rel_pct']:.1f}%); "
            f"metric noise sd {noise_sd:.4f} vs between-restart sd {info['test_mse_post_sd']:.4f}; "
            f"avg20 rank {info['selected_rank_in_avg20_metric (1=best)']}; RMSE(mean200) {info['selected_rmse_mean200_kwh_median']:.4f} kWh "
            f"(median restart {info['rmse_mean200_kwh_median_median']:.4f}); train KL {info['selected_train_kl_final']:.2e}")
    with open(OUT + "/cvae_selection.json", "w") as f:
        json.dump(sel, f, indent=1)
    return sel


# ------------------------------------------------------------------------------------------------------------------
# generation
# ------------------------------------------------------------------------------------------------------------------
def stage_generate(args) -> None:
    te = load(TEST_PATH)
    blocks = {c: test_block(te, c) for c in CLUSTERS}
    dates = blocks[0]["date"]
    assert len(dates) == 92 and dates.is_monotonic_increasing and dates.is_unique
    for c in CLUSTERS:
        assert blocks[c]["date"].equals(dates)
        dc = np.abs(blocks[c][ct.COLS_COND].to_numpy(float) - blocks[0][ct.COLS_COND].to_numpy(float)).max()
        assert dc == 0.0, f"conditions differ between clusters 0 and {c}: {dc}"
    pd.DataFrame({"day_index": np.arange(len(dates)), "date": dates.dt.strftime("%Y-%m-%d")}).to_csv(
        OUT + "/test_dates.csv", index=False)
    real_low = blocks[0][ct.LOW_COLS].to_numpy(float)
    real_high = blocks[0][ct.HIGH_COLS].to_numpy(float)
    all_normal_days = np.where((real_low.sum(1) == 0) & (real_high.sum(1) == 0))[0]
    log(f"[generate] test days {dates.iloc[0].date()} .. {dates.iloc[-1].date()}; {len(all_normal_days)} of 92 test "
        f"days have an all-Normal real tariff; conditions identical across clusters")
    log(f"[generate] scenarios: Low = low {ef.WINDOW_TEXT['Low']}, High = high {ef.WINDOW_TEXT['High']}")
    for c in CLUSTERS:
        model = ct.load_model(selected_path(c))
        cols = model.meta["cond_cols"]
        assert cols == ct.COLS_COND
        C = blocks[c][cols].to_numpy(dtype=np.float64)
        Y = blocks[c][ct.COLS_DATA].to_numpy(dtype=np.float64)
        eps = np.random.default_rng([GEN_SEED, c]).standard_normal((N_SAMPLES, len(C), model.latent_dim))
        S = ct.generate(model, C, N_SAMPLES, eps=eps)
        assert S.shape == (N_SAMPLES, 92, 48) and np.isfinite(S).all()
        status = {"samples": save_array(OUT + f"/cvae_samples_c{c}.npy", S)}
        scen = {}
        for name in SCENARIOS:
            Cs = scenario_cond(C, cols, name)
            keep = [i for i, k in enumerate(cols) if not k.startswith(("low_", "high_"))]
            assert np.array_equal(Cs[:, keep], C[:, keep])
            scen[name] = ct.generate(model, Cs, N_SAMPLES, eps=eps)
            assert scen[name].shape == (N_SAMPLES, 92, 48) and np.isfinite(scen[name]).all()
            status[name] = save_array(OUT + f"/cvae_scen_{name}_c{c}.npy", scen[name])
        # check: on days whose real tariff is all Normal, Normal scenario == real-condition samples (same z)
        if len(all_normal_days):
            assert np.array_equal(scen["Normal"][:, all_normal_days], S[:, all_normal_days])
        dev = S.mean(0) - Y
        rm = np.sqrt((dev ** 2).mean(1))
        log(f"[generate] c{c} seed {model.meta['seed']}: samples mean {S.mean():.4f} kWh vs obs {Y.mean():.4f}; "
            f"RMSE(mean of 200) median over days {np.median(rm):.4f} kWh (mean {rm.mean():.4f}); "
            f"mean across-sample sd {S.std(0, ddof=1).mean():.4f} kWh; scenario means N/L/H "
            f"{scen['Normal'].mean():.4f}/{scen['Low'].mean():.4f}/{scen['High'].mean():.4f}; files vs stored: {status}")


# ------------------------------------------------------------------------------------------------------------------
# scenario effects (quantitative support for the qualitative claim)
# ------------------------------------------------------------------------------------------------------------------
# phase-1 regions from the scenario windows (lib.effects): Low in-window 9-19, 3 slots before (6-8) / after (20-22),
# afternoon-evening 24-47; High in-window 39-44, 3 slots before (36-38) / after (45-47), afternoon 24-35
REGIONS = ef.phase1_regions()
# direction claimed by the paper (Sec. VII-C) for each region; None = no claim
CLAIMED_SIGN = dict(ef.PHASE1_CLAIMED_SIGN)
K_LOW_IN, K_LOW_AE = ef.PHASE1_KEY["Low", "in_window"], ef.PHASE1_KEY["Low", "afternoon_evening"]
K_HIGH_IN, K_HIGH_BEF = ef.PHASE1_KEY["High", "in_window"], ef.PHASE1_KEY["High", "before"]
K_HIGH_AFT, K_HIGH_PM = ef.PHASE1_KEY["High", "after"], ef.PHASE1_KEY["High", "afternoon"]


def _train_tariff_context() -> dict:
    tr = load(TRAIN_PATH)
    b = tr.loc[tr["cluster"] == 0].sort_values("date")
    L = b[ct.LOW_COLS].to_numpy(int)
    H = b[ct.HIGH_COLS].to_numpy(int)

    def runs(row):
        out, h = [], 0
        while h < 48:
            if row[h]:
                s = h
                while h < 48 and row[h]:
                    h += 1
                out.append((s, h - 1))
            else:
                h += 1
        return out
    low_runs = [r for row in L for r in runs(row)]
    high_runs = [r for row in H for r in runs(row)]
    from collections import Counter
    lw, hw = (LOW_SLOTS[0], LOW_SLOTS[-1]), (HIGH_SLOTS[0], HIGH_SLOTS[-1])
    lk, hk = f"{lw[0]}_{lw[1]}", f"{hw[0]}_{hw[1]}"
    return {"train_days": int(len(b)), "train_days_with_low": int((L.sum(1) > 0).sum()),
            "train_days_with_high": int((H.sum(1) > 0).sum()),
            "train_days_with_both": int(((L.sum(1) > 0) & (H.sum(1) > 0)).sum()),
            "low_slot_frequency": L.sum(0).tolist(), "high_slot_frequency": H.sum(0).tolist(),
            "most_common_low_runs (start,end slots)": [[list(k), v] for k, v in Counter(low_runs).most_common(6)],
            "most_common_high_runs (start,end slots)": [[list(k), v] for k, v in Counter(high_runs).most_common(6)],
            f"train_days_with_low_run_exactly_{lk}": int(sum(1 for row in L if lw in runs(row))),
            f"train_days_with_high_run_exactly_{hk}": int(sum(1 for row in H if hw in runs(row))),
            f"train_days_with_low_covering_{lk}": int((L[:, LOW_SLOTS].sum(1) == len(LOW_SLOTS)).sum()),
            f"train_days_with_high_covering_{hk}": int((H[:, HIGH_SLOTS].sum(1) == len(HIGH_SLOTS)).sum()),
            "max_high_slots_in_a_train_day": int(H.sum(1).max()), "max_low_slots_in_a_train_day": int(L.sum(1).max())}


def stage_effects(args) -> dict:
    te = load(TEST_PATH)
    rows = []
    res = {"definition": "diff = mean over the 200 samples of (scenario - Normal), per test day and slot, kWh per "
                         "household per half-hour (same z draws in both scenarios); region stats average the slots of "
                         "the region; 'pct' is relative to the Normal mean over the same slots and days",
           "scenario_windows": {k: {"slots": v, "clock_gmt": ef.WINDOW_CLOCK[k]} for k, v in ef.WINDOWS.items()},
           "regions": REGIONS,
           "claimed_sign": {f"{k[0]}:{k[1]}": v for k, v in CLAIMED_SIGN.items()},
           "train_tariff_context": _train_tariff_context(), "clusters": {}}
    for c in CLUSTERS:
        Nm = np.load(OUT + f"/cvae_scen_Normal_c{c}.npy").mean(0)          # (92, 48)
        out = {}
        for name in ("Low", "High"):
            D = np.load(OUT + f"/cvae_scen_{name}_c{c}.npy").mean(0) - Nm  # (92, 48)
            reg = {}
            for rname, slots in REGIONS[name].items():
                per_day = D[:, slots].mean(1)
                base = Nm[:, slots].mean()
                sign = CLAIMED_SIGN.get((name, rname))
                reg[rname] = {"mean_diff_kwh": float(per_day.mean()), "pct_of_normal": float(100 * per_day.mean() / base),
                              "frac_days_positive": float((per_day > 0).mean()),
                              "claimed_sign": sign,
                              "matches_claim": (None if sign is None else bool(np.sign(per_day.mean()) == sign))}
            reg["daily_energy_change_kwh"] = float(D.sum(1).mean())
            reg["daily_energy_change_pct"] = float(100 * D.sum(1).mean() / Nm.sum(1).mean())
            reg["day1_profile_diff_kwh"] = D[0].round(5).tolist()
            out[name] = reg
            q25, q75 = np.quantile(D, 0.25, axis=0), np.quantile(D, 0.75, axis=0)
            for h in range(48):
                rows.append({"cluster": c, "scenario": name, "slot": h, "normal_mean_kwh": float(Nm[:, h].mean()),
                             "diff_mean_kwh": float(D[:, h].mean()), "diff_q25_kwh": float(q25[h]),
                             "diff_q75_kwh": float(q75[h]), "frac_days_positive": float((D[:, h] > 0).mean())})
        # never-seen tariff profile (diagnostic only): High all day
        model = ct.load_model(selected_path(c))
        C = test_block(te, c)[model.meta["cond_cols"]].to_numpy(float)
        eps = np.random.default_rng([GEN_SEED, c]).standard_normal((N_SAMPLES, len(C), model.latent_dim))
        AH = ct.generate(model, scenario_cond(C, model.meta["cond_cols"], "AllHigh"), N_SAMPLES, eps=eps).mean(0)
        out["AllHigh_diagnostic"] = {
            "daily_energy_ratio_vs_normal": float(AH.sum(1).mean() / Nm.sum(1).mean()),
            "mean_profile_corr_vs_normal": float(np.corrcoef(AH.mean(0), Nm.mean(0))[0, 1]),
            "min_halfhour_kwh": float(AH.min()), "frac_negative_halfhours": float((AH < 0).mean()),
            "mean_profile_kwh": AH.mean(0).round(5).tolist(), "normal_mean_profile_kwh": Nm.mean(0).round(5).tolist()}
        res["clusters"][str(c)] = out
        lo, hi = out["Low"], out["High"]
        log(f"[effects] c{c}: Low in-window {lo[K_LOW_IN]['mean_diff_kwh']:+.4f} kWh ({lo[K_LOW_IN]['pct_of_normal']:+.1f}%), "
            f"afternoon/evening {lo[K_LOW_AE]['mean_diff_kwh']:+.4f} ({lo[K_LOW_AE]['pct_of_normal']:+.1f}%), "
            f"day energy {lo['daily_energy_change_pct']:+.1f}% | High in-window {hi[K_HIGH_IN]['mean_diff_kwh']:+.4f} "
            f"({hi[K_HIGH_IN]['pct_of_normal']:+.1f}%), before {hi[K_HIGH_BEF]['mean_diff_kwh']:+.4f}, "
            f"after {hi[K_HIGH_AFT]['mean_diff_kwh']:+.4f}, afternoon {hi[K_HIGH_PM]['mean_diff_kwh']:+.4f}, "
            f"day energy {hi['daily_energy_change_pct']:+.1f}% | AllHigh energy ratio "
            f"{out['AllHigh_diagnostic']['daily_energy_ratio_vs_normal']:.3f}")
    pd.DataFrame(rows).to_csv(OUT + "/cvae_scenario_effects.csv", index=False)
    with open(OUT + "/cvae_scenario_effects.json", "w") as f:
        json.dump(res, f, indent=1)
    return res


# ------------------------------------------------------------------------------------------------------------------
# robustness of the scenario effects across restarts, and upstream-quirk sensitivity
# ------------------------------------------------------------------------------------------------------------------
N_QUIRK = 20                  # restarts per cluster for the upstream-quirk emulation
QUIRK_ADAM_ITER0 = 10 ** 6    # shared Keras Adam: iteration counter far past 0 -> bias correction off


def quirk_model_path(c: int, s: int) -> str:
    return f"{MODELS}/cvae_upq_c{c}_s{s:02d}.pt"


def quirk_json_path(c: int, s: int) -> str:
    return f"{MODELS}/cvae_upq_c{c}_s{s:02d}.json"


def _region_effects(model, C: np.ndarray, eps: np.ndarray) -> dict:
    cols = model.meta["cond_cols"]
    M = {name: ct.generate(model, scenario_cond(C, cols, name), eps.shape[0], eps=eps).mean(0) for name in SCENARIOS}
    out = {}
    for name in ("Low", "High"):
        D = M[name] - M["Normal"]
        for rname, slots in REGIONS[name].items():
            out[f"{name}:{rname}:kwh"] = float(D[:, slots].mean())
            out[f"{name}:{rname}:pct"] = float(100 * D[:, slots].mean() / M["Normal"][:, slots].mean())
        out[f"{name}:daily_energy_pct"] = float(100 * D.sum(1).mean() / M["Normal"].sum(1).mean())
    return out


def _effects_summary(df: pd.DataFrame) -> dict:
    out = {"n_restarts": int(len(df))}
    for (name, rname), sign in CLAIMED_SIGN.items():
        v = df[f"{name}:{rname}:pct"]
        out[f"{name}:{rname}"] = {"frac_restarts_matching_claimed_sign": float((np.sign(v) == sign).mean()),
                                  "median_pct": float(v.median()), "q10_pct": float(v.quantile(0.1)),
                                  "q90_pct": float(v.quantile(0.9))}
    for name in ("Low", "High"):
        v = df[f"{name}:daily_energy_pct"]
        out[f"{name}:daily_energy_pct"] = {"median": float(v.median()), "q10": float(v.quantile(0.1)),
                                           "q90": float(v.quantile(0.9))}
    if len(df) > 2:
        out["spearman_test_mse_vs_Low_in_window_pct"] = _spearman(df["test_mse_post"], df[f"Low:{K_LOW_IN}:pct"])
        out["spearman_test_mse_vs_High_in_window_pct"] = _spearman(df["test_mse_post"], df[f"High:{K_HIGH_IN}:pct"])
    return out


def stage_robust(args) -> dict:
    te = load(TEST_PATH)
    rs = pd.read_csv(OUT + "/cvae_restarts.csv")
    with open(OUT + "/cvae_selection.json") as f:
        sel = json.load(f)["clusters"]
    with open(OUT + "/cvae_scenario_effects.json") as f:
        eff_sel = json.load(f)["clusters"]
    rows = []
    for c in CLUSTERS:
        C = test_block(te, c)[ct.COLS_COND].to_numpy(dtype=np.float64)
        eps = np.random.default_rng([GEN_SEED, c]).standard_normal((N_SAMPLES, len(C), ct.LATENT_DIM))
        for s in range(N_RESTARTS):
            m = ct.load_model(model_path(c, s))
            r = rs.loc[(rs["cluster"] == c) & (rs["seed"] == s)].iloc[0]
            rows.append({"cluster": c, "seed": s, "selected": s == sel[str(c)]["selected_seed"],
                         "test_mse_post": float(r["test_mse_post"]), **_region_effects(m, C, eps)})
        # consistency: the selected restart's row reproduces the effects stage exactly
        srow = [x for x in rows if x["cluster"] == c and x["selected"]][0]
        ref = eff_sel[str(c)]["Low"][K_LOW_IN]["mean_diff_kwh"]
        assert abs(srow[f"Low:{K_LOW_IN}:kwh"] - ref) < 1e-12, (srow[f"Low:{K_LOW_IN}:kwh"], ref)
    df = pd.DataFrame(rows)
    df.to_csv(OUT + "/cvae_scenario_effects_all_restarts.csv", index=False)
    res = {"definition": "per restart: same scenario construction and z draws as cvae_scenario_effects.json; region "
                         "means in % of the Normal scenario over the same slots; claimed signs from Sec. VII-C",
           "scenario_windows": {k: {"slots": v, "clock_gmt": ef.WINDOW_CLOCK[k]} for k, v in ef.WINDOWS.items()},
           "clusters": {}}
    for c in CLUSTERS:
        d = df.loc[df["cluster"] == c]
        res["clusters"][str(c)] = _effects_summary(d)
        sm = res["clusters"][str(c)]
        log(f"[robust] c{c}: Low in-window >0 in {100 * sm[f'Low:{K_LOW_IN}']['frac_restarts_matching_claimed_sign']:.0f}% "
            f"of restarts (median {sm[f'Low:{K_LOW_IN}']['median_pct']:+.1f}%), High in-window <0 in "
            f"{100 * sm[f'High:{K_HIGH_IN}']['frac_restarts_matching_claimed_sign']:.0f}% (median "
            f"{sm[f'High:{K_HIGH_IN}']['median_pct']:+.1f}%), High before <0 "
            f"{100 * sm[f'High:{K_HIGH_BEF}']['frac_restarts_matching_claimed_sign']:.0f}%, "
            f"after <0 {100 * sm[f'High:{K_HIGH_AFT}']['frac_restarts_matching_claimed_sign']:.0f}%, Low aft/eve <0 "
            f"{100 * sm[f'Low:{K_LOW_AE}']['frac_restarts_matching_claimed_sign']:.0f}%, High afternoon >0 "
            f"{100 * sm[f'High:{K_HIGH_PM}']['frac_restarts_matching_claimed_sign']:.0f}%")
    with open(OUT + "/cvae_scenario_effects_all_restarts.json", "w") as f:
        json.dump(res, f, indent=1)
    return res


def stage_quirks(args) -> dict:
    """Upstream-quirk emulation: every restart starts from the SAME Glorot weights (init_seed 0; a seeded TF2
    initializer is stateless) and Adam runs without bias correction (one Keras Adam shared by all restarts, its
    iteration counter already far past 0). Noise seeds 0..N_QUIRK-1 equal the main run's seeds, so restart 0 of
    both runs differs only by the Adam quirk."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = []
    for s in range(N_QUIRK):
        for c in args.clusters:
            if not args.force and os.path.exists(quirk_json_path(c, s)) and os.path.exists(quirk_model_path(c, s)):
                continue
            jobs.append({"train_path": TRAIN_PATH, "test_path": TEST_PATH, "cluster": c, "seed": s, "init_seed": 0,
                         "adam_iterations0": QUIRK_ADAM_ITER0, "epochs": args.epochs,
                         "model_path": quirk_model_path(c, s), "json_path": quirk_json_path(c, s)})
    log(f"[quirks] {len(jobs)} upstream-quirk restarts to run with {args.workers} workers")
    if jobs and args.no_train:
        raise SystemExit(f"[quirks] --no-train: {len(jobs)} upstream-quirk restarts would be trained; refusing")
    if jobs:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=ct.worker_init) as ex:
            futs = [ex.submit(ct.run_restart_job, j) for j in jobs]
            for k, fut in enumerate(as_completed(futs)):
                r = fut.result()
                log(f"[quirks] {k + 1:3d}/{len(jobs)} c{r['cluster']} s{r['seed']:02d} test_mse_post "
                    f"{r['test_mse_post']:.4f} | {r['runtime_train_s']:.1f}s | wall {time.perf_counter() - t0:.0f}s")
    te = load(TEST_PATH)
    main = pd.read_csv(OUT + "/cvae_restarts.csv")
    rob = pd.read_csv(OUT + "/cvae_scenario_effects_all_restarts.csv")
    res = {"setup": {"n_restarts_per_cluster": N_QUIRK, "init_seed": 0, "adam_iterations0": QUIRK_ADAM_ITER0,
                     "noise_seeds": f"0..{N_QUIRK - 1} (same as the main run)",
                     "scenario_windows": {k: {"slots": v, "clock_gmt": ef.WINDOW_CLOCK[k]}
                                          for k, v in ef.WINDOWS.items()}}, "clusters": {}}
    for c in CLUSTERS:
        recs = []
        for s in range(N_QUIRK):
            with open(quirk_json_path(c, s)) as f:
                recs.append(json.load(f))
        q = pd.DataFrame([{k: v for k, v in r.items() if isinstance(v, (int, float))} for r in recs])
        m20 = main.loc[(main["cluster"] == c) & (main["seed"] < N_QUIRK)]
        m50 = main.loc[main["cluster"] == c]
        qb = q.loc[q["test_mse_post"].idxmin()]
        C = test_block(te, c)[ct.COLS_COND].to_numpy(dtype=np.float64)
        eps = np.random.default_rng([GEN_SEED, c]).standard_normal((N_SAMPLES, len(C), ct.LATENT_DIM))
        qeff = []
        for s in range(N_QUIRK):
            qeff.append({"test_mse_post": float(q.loc[q["seed"] == s, "test_mse_post"].iloc[0]),
                         **_region_effects(ct.load_model(quirk_model_path(c, s)), C, eps)})
        qeff = pd.DataFrame(qeff)
        best_eff = qeff.loc[qeff["test_mse_post"].idxmin()]
        main_sel = rob.loc[(rob["cluster"] == c) & rob["selected"]].iloc[0]
        m0 = m20.loc[m20["seed"] == 0].iloc[0]
        q0 = q.loc[q["seed"] == 0].iloc[0]
        out = {
            "quirk_test_mse_post": {"best": float(q["test_mse_post"].min()), "median": float(q["test_mse_post"].median()),
                                    "worst": float(q["test_mse_post"].max()), "sd": float(q["test_mse_post"].std(ddof=1))},
            "main_first20_test_mse_post": {"best": float(m20["test_mse_post"].min()),
                                           "median": float(m20["test_mse_post"].median()),
                                           "worst": float(m20["test_mse_post"].max()),
                                           "sd": float(m20["test_mse_post"].std(ddof=1))},
            "main_all50_best": float(m50["test_mse_post"].min()),
            "quirk_best_seed": int(qb["seed"]),
            "quirk_best_rmse_mean200_kwh_median": float(qb["test_rmse_mean200_kwh_median"]),
            "main_selected_rmse_mean200_kwh_median": float(m50["test_rmse_mean200_kwh_median"].loc[m50["test_mse_post"].idxmin()]),
            "paired_seed0_only_adam_differs": {"main": float(m0["test_mse_post"]), "quirk": float(q0["test_mse_post"])},
            "quirk_train_kl_final_median": float(q["train_kl_final"].median()),
            "quirk_best_effects_pct": {k: float(best_eff[k]) for k in best_eff.index if k.endswith(":pct")},
            "main_selected_effects_pct": {k: float(main_sel[k]) for k in main_sel.index if str(k).endswith(":pct")},
            "quirk_effects_across_restarts": _effects_summary(qeff),
        }
        res["clusters"][str(c)] = out
        log(f"[quirks] c{c}: quirk best/median {out['quirk_test_mse_post']['best']:.4f}/{out['quirk_test_mse_post']['median']:.4f} "
            f"(sd {out['quirk_test_mse_post']['sd']:.3f}) vs main first-20 {out['main_first20_test_mse_post']['best']:.4f}/"
            f"{out['main_first20_test_mse_post']['median']:.4f} (sd {out['main_first20_test_mse_post']['sd']:.3f}); "
            f"seed0 main {out['paired_seed0_only_adam_differs']['main']:.4f} vs quirk {out['paired_seed0_only_adam_differs']['quirk']:.4f}; "
            f"quirk-best Low in-window {out['quirk_best_effects_pct'][f'Low:{K_LOW_IN}:pct']:+.1f}% High in-window "
            f"{out['quirk_best_effects_pct'][f'High:{K_HIGH_IN}:pct']:+.1f}%")
    with open(OUT + "/cvae_sensitivity_upstream_quirks.json", "w") as f:
        json.dump(res, f, indent=1)
    return res


# ------------------------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 8, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round", "axes.titlelocation": "left"})
    return plt


def _headroom(ax, frac: float = 0.10) -> None:
    """Extra space above the curves for the window labels drawn at the top of the axes."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def _shade_windows(ax, day_offsets=(0.0,), label=True, ymax_frac=0.97):
    """Shade the scenario windows over their physical intervals (lib.effects.WINDOW_HOURS: Low 04:00-09:30, High
    19:00-22:00 GMT); the curves put slot h at h/2 h, the end of its half-hour."""
    for off in day_offsets:
        for name, color in (("Low", AQUA), ("High", ORANGE)):
            a, b = ef.WINDOW_HOURS[name]
            ax.axvspan(off + a, off + b, color=color, alpha=0.10, lw=0)
            if label:
                ax.text(off + (a + b) / 2, ymax_frac, f"{name} window", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", color=INK2, fontsize=7.5)


def fig5_replica(plt, c: int = 3) -> str:
    sizes = cluster_sizes()
    te = load(TEST_PATH)
    blk = test_block(te, c)
    S = np.load(OUT + f"/cvae_samples_c{c}.npy")
    Y = blk[ct.COLS_DATA].to_numpy(float)
    x = np.arange(144) / 2.0
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    for k in range(20):
        ax.plot(x, S[k, :3].reshape(-1), color=BLUE, lw=0.6, alpha=0.45, label="20 generated profiles" if k == 0 else None)
    ax.plot(x, S[:, :3].mean(0).reshape(-1), color=BLUE, lw=2.0, label="Mean of 200 generated profiles")
    ax.plot(x, Y[:3].reshape(-1), color=INK, lw=1.4, label="Observed")
    for b in (24, 48):
        ax.axvline(b, color=AXIS, lw=0.8)
    for d in range(3):
        ax.text(24 * d + 0.5, 1.0, blk["date"].iloc[d].strftime("%a %d %b %Y"), transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", color=INK2, fontsize=8)
    ticks = np.arange(0, 73, 6)
    ax.set_xticks(ticks, [f"{int(t % 24):02d}:00" for t in ticks])
    ax.set_xlim(0, 72)
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("kWh per household per half-hour")
    ax.set_title(f"Track A CVAE, cluster {c} ({sizes[c]} households): first three test days\n"
                 f"Replica of paper Fig. 5, left ({FIG5_NOTE.get(c, 'not the cluster of the paper figure')})", pad=16)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=3)
    path = FIGS + f"/trackA_cvae_fig5_cluster{c}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig6_replica(plt, c: int = 3, day: int = 0) -> str:
    sizes = cluster_sizes()
    te = load(TEST_PATH)
    date = test_block(te, c)["date"].iloc[day]
    x = np.arange(48) / 2.0
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    _shade_windows(ax)
    for name in SCENARIOS:
        S = np.load(OUT + f"/cvae_scen_{name}_c{c}.npy")[:, day]
        for k in range(20):
            ax.plot(x, S[k], color=SCEN_COLOR[name], lw=0.6, alpha=0.5,
                    label=f"{SCEN_LABEL[name]}: 20 profiles" if k == 0 else None)
    for name in SCENARIOS:
        S = np.load(OUT + f"/cvae_scen_{name}_c{c}.npy")[:, day]
        ax.plot(x, S.mean(0), color=SCEN_COLOR[name], lw=2.0, label=f"{name}: mean of 200")
    _headroom(ax)
    ax.set_xticks(np.arange(0, 25, 2), [f"{h:02d}:00" for h in range(0, 25, 2)])
    ax.set_xlim(0, 23.5)
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("kWh per household per half-hour")
    ax.set_title(f"Track A CVAE, cluster {c} ({sizes[c]} households), {date:%a %d %b %Y}: tariff scenarios "
                 f"(replica of paper Fig. 6, left)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.14), ncol=2)
    path = FIGS + f"/trackA_cvae_fig6_cluster{c}_day1.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig8_replica(plt) -> str:
    sizes = cluster_sizes()
    te = load(TEST_PATH)
    dates = test_block(te, 0)["date"]
    x = np.arange(144) / 2.0
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 10), sharex=True)
    for c, ax in zip(CLUSTERS, axes):
        _shade_windows(ax, day_offsets=(0, 24, 48), label=False)   # windows named in the legend (no clash with curves)
        for name in SCENARIOS:
            M = np.load(OUT + f"/cvae_scen_{name}_c{c}.npy")[:, :3].mean(0).reshape(-1)
            ax.plot(x, M, color=SCEN_COLOR[name], lw=1.8, label=SCEN_LABEL[name])
        for b in (24, 48):
            ax.axvline(b, color=AXIS, lw=0.8)
        ax.set_title(f"Cluster {c} ({sizes[c]} households)")
        ax.set_ylabel("kWh / household / half-hour")
    ticks = np.arange(0, 73, 6)
    axes[-1].set_xticks(ticks, [f"{int(t % 24):02d}:00" for t in ticks])
    axes[-1].set_xlim(0, 72)
    axes[-1].set_xlabel(f"Time of day over the first three test days ({dates.iloc[0]:%d %b} - {dates.iloc[2]:%d %b %Y}), "
                        f"GMT; the point at time t is the half-hour ending at t")
    h, lab = axes[0].get_legend_handles_labels()
    h += [Patch(color=AQUA, alpha=0.25, lw=0), Patch(color=ORANGE, alpha=0.25, lw=0)]
    lab += [f"Low window ({ef.WINDOW_CLOCK['Low']} GMT)", f"High window ({ef.WINDOW_CLOCK['High']} GMT)"]
    axes[0].legend(h, lab, loc="lower left", bbox_to_anchor=(0.0, 1.12), ncol=2)
    fig.suptitle("Track A CVAE: mean of 200 generated profiles per tariff scenario (replica of paper Fig. 8, left)",
                 x=0.01, ha="left", y=1.0, fontsize=10.5)
    fig.tight_layout()
    path = FIGS + "/trackA_cvae_fig8_scenario_means.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_effects(plt) -> str:
    sizes = cluster_sizes()
    eff = pd.read_csv(OUT + "/cvae_scenario_effects.csv")
    x = np.arange(48) / 2.0
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex=True)
    for c, ax in zip(CLUSTERS, axes.ravel()):
        _shade_windows(ax, label=(c == 0))
        ax.axhline(0.0, color=AXIS, lw=1.0)
        for name in ("Low", "High"):
            e = eff.loc[(eff["cluster"] == c) & (eff["scenario"] == name)].sort_values("slot")
            ax.fill_between(x, e["diff_q25_kwh"], e["diff_q75_kwh"], color=SCEN_COLOR[name], alpha=0.12, lw=0)
            ax.plot(x, e["diff_mean_kwh"], color=SCEN_COLOR[name], lw=2.0, label=f"{name} minus Normal")
        _headroom(ax)
        ax.set_title(f"Cluster {c} ({sizes[c]} households)")
        ax.set_xticks(np.arange(0, 25, 4), [f"{h:02d}:00" for h in range(0, 25, 4)])
        ax.set_xlim(0, 23.5)
    for ax in axes[1]:
        ax.set_xlabel("Time of day, GMT (point at t = half-hour ending at t)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Scenario minus Normal (kWh)")
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.10), ncol=2)
    fig.suptitle("Track A CVAE: tariff effect, mean over the 92 test days of the 200-sample mean difference "
                 "(band = interquartile range across days)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    path = FIGS + "/trackA_cvae_scenario_effects.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_restarts(plt) -> str:
    df = pd.read_csv(OUT + "/cvae_restarts.csv")
    with open(OUT + "/cvae_selection.json") as f:
        sel = json.load(f)["clusters"]
    rng = np.random.default_rng(0)
    panels = (("test_mse_post", "Selection metric: test MSE\n(100 x RMSE on scaled data, posterior z, one draw)"),
              ("test_mse_post_avg20", "Same metric averaged over 20 draws\n(less metric noise)"),
              ("test_rmse_mean200_kwh_median", "Median daily RMSE (kWh) of the\nmean of 200 prior samples"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, (col, title) in zip(axes, panels):
        for c in CLUSTERS:
            d = df.loc[df["cluster"] == c]
            jit = rng.uniform(-0.18, 0.18, len(d))
            ax.scatter(c + jit, d[col], s=18, color=BLUE, alpha=0.75, edgecolors=SURFACE, linewidths=0.8,
                       label="Restart (seeds 0-49)" if c == 0 else None, zorder=3)
            sd = d.loc[d["seed"] == sel[str(c)]["selected_seed"], col].iloc[0]
            ax.scatter([c], [sd], s=80, facecolors="none", edgecolors=INK, linewidths=1.5,
                       label="Selected (lowest selection metric)" if c == 0 else None, zorder=4)
        ax.set_xticks(list(CLUSTERS), [f"Cluster {c}" for c in CLUSTERS])
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, -0.32), ncol=2)
    fig.suptitle("Track A CVAE: spread across the 50 restarts per cluster", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    path = FIGS + "/trackA_cvae_restarts.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _effects_restart_panels() -> tuple:
    """(column, color, title) of the three panels of the per-restart effect figure (phase-1 regions)."""
    bef = REGIONS["High"][K_HIGH_BEF]
    return ((f"Low:{K_LOW_IN}:pct", AQUA, f"Low minus Normal inside the Low window\n(slots {ef.WINDOW_SPAN['Low']}, "
                                          f"{ef.WINDOW_CLOCK['Low']} GMT), % of Normal"),
            (f"High:{K_HIGH_IN}:pct", ORANGE, f"High minus Normal inside the High window\n(slots "
                                              f"{ef.WINDOW_SPAN['High']}, {ef.WINDOW_CLOCK['High']} GMT), % of Normal"),
            (f"High:{K_HIGH_BEF}:pct", ORANGE, f"High minus Normal just before the window\n(slots {ef.span(bef)}, "
                                               f"{ef.slots_clock(bef)} GMT), % of Normal"))


def fig_effects_restarts(plt) -> str | None:
    p = OUT + "/cvae_scenario_effects_all_restarts.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    rng = np.random.default_rng(1)
    panels = _effects_restart_panels()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, (col, color, title) in zip(axes, panels):
        ax.axhline(0.0, color=AXIS, lw=1.0)
        for c in CLUSTERS:
            d = df.loc[df["cluster"] == c]
            ax.scatter(c + rng.uniform(-0.18, 0.18, len(d)), d[col], s=18, color=color, alpha=0.8, edgecolors=SURFACE,
                       linewidths=0.8, zorder=3, label="Restart (seeds 0-49)" if c == 0 else None)
            ax.scatter([c], [d.loc[d["selected"], col].iloc[0]], s=80, facecolors="none", edgecolors=INK,
                       linewidths=1.5, zorder=4, label="Selected restart" if c == 0 else None)
        ax.set_xticks(list(CLUSTERS), [f"Cluster {c}" for c in CLUSTERS])
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, -0.32), ncol=2)
    fig.suptitle("Track A CVAE: tariff effect of every restart (mean over the 92 test days); paper claims Low > 0, "
                 "High < 0 (also just before the window)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    path = FIGS + "/trackA_cvae_scenario_effects_restarts.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def stage_figures(args) -> list:
    os.makedirs(FIGS, exist_ok=True)
    plt = _plt()
    paths = [fig5_replica(plt, c) for c in FIG5_CLUSTERS]
    paths += [fig6_replica(plt, 3, 0), fig8_replica(plt), fig_effects(plt), fig_restarts(plt)]
    extra = fig_effects_restarts(plt)
    if extra:
        paths.append(extra)
    for p in paths:
        log(f"[figures] wrote {p}")
    return paths


# ------------------------------------------------------------------------------------------------------------------
def stage_summary(args) -> dict:
    import torch
    out = {"track": "A (authors' prepared data: train_nmf / test_nmf)", "torch": torch.__version__,
           "python": sys.version.split()[0], "n_restarts": N_RESTARTS, "n_samples": N_SAMPLES, "gen_seed": GEN_SEED,
           "scenario_windows": {k: {"slots": v, "clock_gmt": ef.WINDOW_CLOCK[k]} for k, v in ef.WINDOWS.items()},
           "scenario_windows_note": "Low 9-19 / High 39-44 (REPRO_SPEC section 7); the phase-1/2 runs used 9-18 / "
                                    "39-43 (outputs/archive_windows_9-18_39-43/)"}
    for name, fn in (("selection", "cvae_selection.json"), ("pca", "cvae_pca_consumption.json"),
                     ("effects", "cvae_scenario_effects.json"), ("walltime", "cvae_train_walltime.json"),
                     ("effects_all_restarts", "cvae_scenario_effects_all_restarts.json"),
                     ("upstream_quirks_sensitivity", "cvae_sensitivity_upstream_quirks.json")):
        p = f"{OUT}/{fn}"
        if os.path.exists(p):
            with open(p) as f:
                out[name] = json.load(f)
    if "effects" in out:
        for c in out["effects"]["clusters"].values():
            for k in ("Low", "High"):
                c[k].pop("day1_profile_diff_kwh", None)
            c["AllHigh_diagnostic"].pop("mean_profile_kwh", None)
            c["AllHigh_diagnostic"].pop("normal_mean_profile_kwh", None)
    out["deviations_from_upstream"] = [
        "PyTorch float64 instead of TF2/Keras float64 (same layers, loss, initialiser family, Keras-Adam update rule).",
        "Each restart uses its own seed (seeds 0..49) for the Glorot init and the reparameterisation noise; upstream "
        "reuses one glorot_uniform(seed=0) object and unseeded tf.random.normal.",
        "Fresh Adam per restart; upstream shares ONE Adam instance across all restarts and clusters (Keras iteration "
        "counter keeps growing -> later restarts start almost without bias correction, first steps ~3.16x larger).",
        "Selection metric evaluated with a fixed RNG (seed 0, common to all restarts); upstream draws are unseeded.",
        "Generation: z drawn with numpy (default_rng([20200801, cluster])), same z for real-condition samples and the "
        "three tariff scenarios (common random numbers).",
        "Upstream selection code is not in the repo; selection = lowest compute_mse on the test set as described in "
        "the paper (Sec. V-D).",
    ]
    with open(OUT + "/cvae_summary.json", "w") as f:
        json.dump(out, f, indent=1)
    log("[summary] wrote cvae_summary.json")
    return out


STAGES = ("pca", "train", "collect", "select", "generate", "effects", "robust", "quirks", "figures", "summary")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", help="all or a comma list of: " + ",".join(STAGES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--restarts", type=int, default=N_RESTARTS)
    ap.add_argument("--epochs", type=int, default=ct.EPOCHS)
    ap.add_argument("--clusters", type=int, nargs="+", default=list(CLUSTERS))
    ap.add_argument("--force", action="store_true", help="retrain restarts that already have results")
    ap.add_argument("--no-train", action="store_true",
                    help="refuse to train anything (train / quirks stages fail if a restart is missing)")
    args = ap.parse_args()
    if args.workers > 4:
        raise SystemExit("max 4 workers (shared machine)")
    if args.no_train and args.force:
        raise SystemExit("--no-train and --force are incompatible")
    stages = STAGES if args.stage == "all" else tuple(s.strip() for s in args.stage.split(","))
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {STAGES}")
    os.makedirs(MODELS, exist_ok=True)
    log(f"=== 10_trackA_cvae.py stages={','.join(stages)} workers={args.workers} restarts={args.restarts} "
        f"epochs={args.epochs} clusters={args.clusters} no_train={args.no_train}; scenario windows Low "
        f"{ef.WINDOW_TEXT['Low']}, High {ef.WINDOW_TEXT['High']}")
    fns = {"pca": stage_pca, "train": stage_train, "collect": stage_collect, "select": stage_select,
           "generate": stage_generate, "effects": stage_effects, "robust": stage_robust, "quirks": stage_quirks,
           "figures": stage_figures, "summary": stage_summary}
    for s in stages:
        t0 = time.perf_counter()
        fns[s](args)
        log(f"--- stage {s} done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
