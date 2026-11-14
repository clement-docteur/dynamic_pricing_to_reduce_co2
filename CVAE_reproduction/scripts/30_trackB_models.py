"""Track B: the CVAE (lib/cvae_torch.py) and the GAM benchmark (lib/gam_generator.py) trained and sampled on OUR
data, mirroring Track A (scripts/10_trackA_cvae.py, scripts/11_trackA_gam.py) setting for setting.

Inputs (read-only; built by scripts/05_build_trackB_data.py from our run-(a) ToU clusters)
  data/train_ours.parquet, data/test_ours.parquet  4 clusters relabelled to the authors' labels 0-3 (Hungarian match of
                                                    the cluster-mean series; 285/299/299/149 households), the authors'
                                                    exact 273/92-day split, ERA5 temperatures, Tariffs.xlsx tariffs in
                                                    the physically right ('loc') alignment; authors' schema
                                                    (consumption_00..47 + the 102 COLS_COND)
  data/power_consump_ours.parquet                   365 days x 4 clusters (consumption-PCA claim, GAM sigma variant)
  outputs/clustering_results.json ["trackB_data"]   relabel diagnostics (which Track B cluster matches the authors' 3)
  outputs/trackA/*, UPSTREAM/data/{train,test}_nmf  Track A results and data, for the sanity comparison and the mirror
                                                    check (feather files read by absolute path; nothing upstream runs)

Mirrored exactly (verified at run time by --stage mirror)
  CVAE  seeds 0..49 per cluster (lib.cvae_torch._seed_pair -> init + noise), 50 epochs, batch 1, rows in file order
        (= ascending date, the same 273-date sequence as Track A), Keras-exact Adam, eta 10; selection = lowest
        test_mse_post (authors' compute_mse: posterior z, one draw, RNG seed 0); 200 samples per test day with
        z = default_rng([20200801, c]).standard_normal((200, 92, 4)), the SAME z for the real-condition samples and the
        three tariff scenarios (common random numbers).
  GAM   fit_gam_generator(method="gcv") on the 273 training days of each cluster; samples from stream 0 and the three
        scenarios from stream 1 of SeedSequence([20200727, c]).spawn(2) (the scenarios share one stream: CRN).
  Scenarios: only the 96 tariff dummies change; Normal = all 0; Low = low_09..19 = 1 (04:00-09:30 GMT);
        High = high_39..44 = 1 (19:00-22:00 GMT); everything else = the real test-day conditions. Windows from
        lib.effects (single source of truth, REPRO_SPEC section 7); phases 1-2 used 9-18 / 39-43
        (outputs/archive_windows_9-18_39-43/).
  Arrays (200, 92, 48) kWh, test days ascending (outputs/trackB/test_dates.csv); file names as in Track A.

Stages (--stage all, the default, runs them in this order; every stage is re-runnable)
  mirror    proof of the mirroring: constants and scenario construction equal to the Track A scripts' (imported as
            modules, their main() is not run); this script's CVAE generation code regenerates Track A's arrays from
            Track A's selected models; its job builder retrains Track A's selected cluster-3 restart; its GAM code refits
            Track A's cluster 3 -> outputs/trackB/cvae_gam_mirror_check_trackA.json (models/mirror_trackA_*)
  pca       consumption-PCA claim on our cluster series -> cvae_pca_consumption.json
  train     4 clusters x 50 restarts in a pool of 4 workers (1 torch thread each) -> models/cvae_c<c>_s<ss>.{pt,json};
            finished restarts are skipped unless --force
  collect   cvae_restarts.csv, cvae_restarts_epoch_loss.csv
  select    lowest test MSE per cluster -> models/cvae_selected_c<c>.pt, cvae_selection.json,
            cvae_selection_table.csv (Track B next to Track A)
  verify    retrains the selected restarts and any restart with an implausible recorded runtime (machine suspended
            mid-run) from scratch and checks they are bitwise identical to the stored ones -> cvae_verify_retrain.json
            (models/verify_*); the 200 restarts were trained in two invocations
  generate  cvae_samples_c<c>.npy, cvae_scen_<Normal|Low|High>_c<c>.npy, test_dates.csv
  effects   cvae_scenario_effects.{json,csv} (Track A region definitions)
  robust    the same effects for all 50 restarts -> cvae_scenario_effects_all_restarts.{csv,json}
  gam       GAM per cluster -> models/gam_params_c<c>.pkl, gam_samples_c<c>.npy, gam_scen_<S>_c<c>.npy,
            gam_f_test_c<c>.npy, gam_diagnostics.json, gam_smoothing.csv, gam_sigma.csv (log: gam_run_log.txt)
  sanity    RMSE of the 200-sample mean per cluster vs Track A and the paper, independent recomputations, posterior
            collapse, dispersion, scenario effects of both generators on both tracks (day-bootstrap CIs), what the
            Fig. 6 replica day shows vs the other test days -> cvae_gam_sanity_vs_trackA.json
  figures   outputs/figures/trackB_cvae_*.png, trackB_gam_*.png (Fig. 5/6/7/8 replicas, Track A comparisons, tariff
            effect of every restart for the cluster matched to the authors' cluster 3; Fig. 5 replicas also for
            cluster 2, the cluster the paper's Fig. 5 shows)
  summary   cvae_summary.json

Extra stages (not part of 'all'; nothing is trained or fitted):
  mirror_regen  the mirror check without retraining or refitting: constants and scenario construction equal to the
                Track A scripts'; this script's CVAE generation code and GAM generation code (stored Track A GAMs)
                regenerate Track A's arrays bitwise; the training / refit results of the last full 'mirror' run are
                carried over -> cvae_gam_mirror_check_trackA.json
  gamscen       the GAM's tariff scenarios regenerated from the stored models/gam_params_c<c>.pkl with the 'gam'
                stage's seeds (stored samples checked bitwise; identical arrays are not rewritten); the scenario
                entries of gam_diagnostics.json recomputed

Not run for Track B: Track A's upstream-quirk sensitivity (stage quirks). It probes the training code, not the data,
and Track A already measured it.

Usage (from PROJECT or REPRO):  python -P scripts/30_trackB_models.py [--stage all|mirror|pca|train|...] [--workers 4]
                                [--no-train]   (refuse every stage that would train or fit a model)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
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

DATA = REPRO + "/data"
TRAIN_PATH = DATA + "/train_ours.parquet"
TEST_PATH = DATA + "/test_ours.parquet"
POWER_PATH = DATA + "/power_consump_ours.parquet"
CLUSTERING_JSON = REPRO + "/outputs/clustering_results.json"
OUT = REPRO + "/outputs/trackB"
MODELS = OUT + "/models"
FIGS = REPRO + "/outputs/figures"
TRACKA = REPRO + "/outputs/trackA"
UPSTREAM_DATA = REPRO + "/data/authors"  # copy of the authors' repo data/*.feather (byte-identical; read as data only)
A_TRAIN_PATH = UPSTREAM_DATA + "/train_nmf.feather"
A_TEST_PATH = UPSTREAM_DATA + "/test_nmf.feather"
A_SIZES = {0: 325, 1: 295, 2: 229, 3: 158}
CVAE_LOG = OUT + "/cvae_run_log.txt"
GAM_LOG = OUT + "/gam_run_log.txt"

CLUSTERS = (0, 1, 2, 3)
MODEL_NAMES = ("cvae", "gam")
N_RESTARTS = 50
N_SAMPLES = 200
GEN_SEED = 20200801            # CVAE z ~ N(0, I) for samples and scenarios: rng = default_rng([GEN_SEED, cluster])
GAM_BASE_SEED = 20200727       # GAM: SeedSequence([GAM_BASE_SEED, cluster]).spawn(2) -> [samples, scenarios]
LOW_SLOTS = list(ef.LOW_WINDOW)     # slots 9-19 = 04:00-09:30 GMT (lib.effects; REPRO_SPEC section 7)
HIGH_SLOTS = list(ef.HIGH_WINDOW)   # slots 39-44 = 19:00-22:00 GMT
SCENARIOS = ("Normal", "Low", "High")
H = 48
FIG_CLUSTER = 3                # Track B label matched to the authors' cluster 3 (paper Fig. 6); checked in figures
FIG5_CLUSTERS = (3, 2)         # Fig. 5 replicas: 3 (named in the paper's caption) and 2 (what the paper's Fig. 5 shows)
FIG5_NOTE = {2: "the paper's Fig. 5 shows the authors' cluster 2",
             3: "the paper's caption names the authors' cluster 3; its figure shows cluster 2"}
PAPER_FIG4 = {  # approximate medians read off Fig. 4 (spec section 3), clusters 0..3 (same dict as Track A's GAM)
    "rmse_gam": [0.015, 0.016, 0.015, 0.018], "rmse_cvae": [0.020, 0.022, 0.018, 0.025],
    "es_gam": [0.155, 0.150, 0.175, 0.135], "es_cvae": [0.150, 0.145, 0.170, 0.130],
    "vg_gam": [4, 4, 5, 5], "vg_cvae": [6, 7, 7, 10],
}

# chart tokens (dataviz reference palette, light surface; same as Track A)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SCEN_COLOR = {"Normal": BLUE, "Low": AQUA, "High": ORANGE}
SCEN_LABEL = {"Normal": "Normal tariff all day",
              "Low": f"Low {ef.WINDOW_CLOCK['Low']} GMT (slots {ef.WINDOW_SPAN['Low']}), Normal otherwise",
              "High": f"High {ef.WINDOW_CLOCK['High']} GMT (slots {ef.WINDOW_SPAN['High']}), Normal otherwise"}
# x axis of the time-of-day figures: slot h is drawn at h/2 h, the END of its half-hour (end-labelled slots, GMT)
TIME_AXIS_LABEL = "Time of day, GMT (the point at time t is the half-hour ending at t)"
MODEL_COLOR = {"cvae": BLUE, "gam": ORANGE}
MODEL_LABEL = {"cvae": "CVAE", "gam": "GAM"}

_LOG_FILES = [CVAE_LOG]


# ------------------------------------------------------------------------------------------------------------------
def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    for p in _LOG_FILES:
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def jdump(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"])
    if "cluster" in df.columns:
        df["cluster"] = df["cluster"].astype(int)
    return df


def cols(prefix: str) -> list:
    return [f"{prefix}_{h:02d}" for h in range(H)]


def model_path(c: int, s: int) -> str:
    return f"{MODELS}/cvae_c{c}_s{s:02d}.pt"


def json_path(c: int, s: int) -> str:
    return f"{MODELS}/cvae_c{c}_s{s:02d}.json"


def selected_path(c: int) -> str:
    return f"{MODELS}/cvae_selected_c{c}.pt"


def cluster_sizes() -> dict:
    tr = load(TRAIN_PATH)
    g = tr.groupby("cluster")["n_households"]
    if not (g.min() == g.max()).all():
        raise RuntimeError("n_households varies within a cluster")
    return {int(c): int(v) for c, v in g.min().items()}


def matched_cluster(authors_label: int = FIG_CLUSTER) -> dict:
    """Track B label of the cluster matched to the authors' cluster `authors_label`. 05_build_trackB_data.py relabels
    our run-(a) clusters to the authors' labels when the three matchings agree ('match_good'), so it is the same
    label; otherwise (size order) this raises."""
    with open(CLUSTERING_JSON, encoding="utf-8") as f:
        rel = json.load(f)["trackB_data"]["relabel"]
    if not rel.get("match_good"):
        raise RuntimeError("Track B clusters are not matched to the authors' labels (size-ordered fallback)")
    mp = {int(k): int(v) for k, v in rel["mapping_ours_to_final"].items()}
    ours = [k for k, v in mp.items() if v == authors_label][0]
    T = np.asarray(rel["contingency_ours_rows_authors_cols_1007"])      # rows: run-(a) labels, cols: authors' labels
    return {"trackB_label": authors_label, "run_a_label": ours, "relabel_rule": rel["relabel_rule"],
            "corr_cluster_mean_series_raw": float(rel["matched_corr_raw"][ours]),
            "corr_cluster_mean_series_deviation_from_all_mean": float(rel["matched_corr_deviation"][ours]),
            "n_households_trackB": cluster_sizes()[authors_label], "n_households_authors": A_SIZES[authors_label],
            "overlap_on_the_authors_1007": {"ours_in_1007": int(T[ours].sum()),
                                            "shared_with_authors_cluster": int(T[ours, authors_label]),
                                            "authors_cluster_size": int(T[:, authors_label].sum())}}


def test_block(te: pd.DataFrame, c: int) -> pd.DataFrame:
    return te.loc[te["cluster"] == c].sort_values("date", kind="stable").reset_index(drop=True)


def date_strings(s: pd.Series) -> list:
    return pd.to_datetime(s).dt.strftime("%Y-%m-%d").tolist()


def obs_blocks(path: str) -> tuple[dict, pd.Series]:
    te = load(path)
    out = {c: test_block(te, c)[ct.COLS_DATA].to_numpy(dtype=np.float64) for c in CLUSTERS}
    return out, test_block(te, CLUSTERS[0])["date"]


def save_array(path: str, arr: np.ndarray) -> str:
    """np.save, unless the file already holds a byte-identical array (then it is left untouched).
    Returns 'unchanged', 'changed' or 'new'. (Copy of Track A's.)"""
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
    Windows: lib.effects.LOW_WINDOW / HIGH_WINDOW (REPRO_SPEC section 7). (Copy of Track A's.)"""
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


def make_job(train_path: str, test_path: str, c: int, s: int, mpath: str, jpath: str, epochs: int) -> dict:
    """Restart job for lib.cvae_torch.run_restart_job (same keys as Track A's jobs)."""
    return {"train_path": train_path, "test_path": test_path, "cluster": c, "seed": s,
            "model_path": mpath, "json_path": jpath, "epochs": epochs}


def cvae_draws(c: int, n_days: int, latent_dim: int) -> np.ndarray:
    return np.random.default_rng([GEN_SEED, c]).standard_normal((N_SAMPLES, n_days, latent_dim))


def cvae_generate_cluster(model, C: np.ndarray, c: int) -> tuple[np.ndarray, dict]:
    """Track A's generate stage for one cluster: real-condition samples and the three scenarios, same z (CRN)."""
    cols_ = model.meta["cond_cols"]
    C = np.array(C, dtype=np.float64, copy=True)   # writable copy (pandas 3 returns read-only views; same values)
    eps = cvae_draws(c, len(C), model.latent_dim)
    S = ct.generate(model, C, N_SAMPLES, eps=eps)
    keep = [i for i, k in enumerate(cols_) if not k.startswith(("low_", "high_"))]
    scen = {}
    for name in SCENARIOS:
        Cs = scenario_cond(C, cols_, name)
        assert np.array_equal(Cs[:, keep], C[:, keep])
        scen[name] = ct.generate(model, Cs, N_SAMPLES, eps=eps)
    return S, scen


# ------------------------------------------------------------------------------------------------------------------
# CVAE: PCA claim (Track A stage_pca on our cluster series)
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
           "data": "power_consump_ours.parquet: 365 daily 48-half-hour profiles per cluster (cluster-mean kWh, our "
                   "run-(a) clusters relabelled to the authors' 0-3)",
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
    t0 = pw.loc[pw["cluster"] == 0].sort_values("date")
    T = t0[cols("temp") + ["temp_smooth"]].to_numpy(dtype=np.float64)
    res["extra_temperature_pca_48temps_plus_smooth"] = {
        "raw_degC": _pca_stats(T), "standardized": _pca_stats((T - T.mean(0)) / T.std(0)),
        "note": "context only (ERA5 temperatures); the temperature PCA claim is checked by the conditions task"}
    vals = [res["clusters"][str(c)]["raw_kWh"]["cum_evr_at_4"] for c in CLUSTERS]
    vals_std = [res["clusters"][str(c)]["standardized_per_halfhour"]["cum_evr_at_4"] for c in CLUSTERS]
    res["verdict_inputs"] = {"cum_evr_at_4_raw_per_cluster": vals, "all_above_80pct_raw": bool(min(vals) > 0.8),
                             "cum_evr_at_4_standardized_per_cluster": vals_std,
                             "all_above_80pct_standardized": bool(min(vals_std) > 0.8)}
    jdump(res, OUT + "/cvae_pca_consumption.json")
    log(f"[pca] all-clusters stacked raw cum EVR@4 {res['all_clusters_stacked_raw']['cum_evr_at_4']:.4f}")
    return res


# ------------------------------------------------------------------------------------------------------------------
# CVAE: training (pool of workers), collect, select
# ------------------------------------------------------------------------------------------------------------------
def stage_train(args) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(MODELS, exist_ok=True)
    jobs = []
    for s in range(args.restarts):
        for c in args.clusters:
            if not args.force and os.path.exists(json_path(c, s)) and os.path.exists(model_path(c, s)):
                continue
            jobs.append(make_job(TRAIN_PATH, TEST_PATH, c, s, model_path(c, s), json_path(c, s), args.epochs))
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
    jdump({"invocations": hist}, rec_path)
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
            # Track B only: a few restarts were in progress while the machine was suspended (their runtime_train_s
            # is wall time including the suspension), so the median is the meaningful per-restart cost
            "runtime_train_s_median": float(d["runtime_train_s"].median()),
            "runtime_train_s_max": float(d["runtime_train_s"].max()),
            "n_restarts_runtime_over_10x_median": int((d["runtime_train_s"] > 10 * d["runtime_train_s"].median()).sum()),
        }
        sel["clusters"][str(c)] = info
        log(f"[select] c{c}: seed {s} test MSE {info['test_mse_post_selected']:.4f} (median {info['test_mse_post_median']:.4f}, "
            f"worst {info['test_mse_post_worst']:.4f}, best-vs-median {info['best_vs_median_rel_pct']:.1f}%); "
            f"metric noise sd {noise_sd:.4f} vs between-restart sd {info['test_mse_post_sd']:.4f}; "
            f"avg20 rank {info['selected_rank_in_avg20_metric (1=best)']}; RMSE(mean200) {info['selected_rmse_mean200_kwh_median']:.4f} kWh "
            f"(median restart {info['rmse_mean200_kwh_median_median']:.4f}); train KL {info['selected_train_kl_final']:.2e}; "
            f"runtime median {info['runtime_train_s_median']:.1f}s/restart")
    jdump(sel, OUT + "/cvae_selection.json")
    selection_table(sel)
    return sel


SEL_TABLE_COLS = ("selected_seed", "test_mse_post_selected", "test_mse_post_median", "test_mse_post_worst",
                  "test_mse_post_sd", "single_draw_metric_noise_sd", "best_vs_median_rel_pct",
                  "selected_rmse_mean200_kwh_median", "rmse_mean200_kwh_median_median",
                  "selected_rank_in_rmse_mean200 (1=best)", "spearman_selection_metric_vs_rmse_mean200",
                  "selected_train_kl_final", "train_kl_final_max", "selected_sample_sd_kwh_mean",
                  "epoch_loss_rel_change_last5_median")


def selection_table(sel: dict) -> pd.DataFrame:
    """One row per (track, cluster): the selection metric and its context, Track B next to Track A
    -> cvae_selection_table.csv."""
    pa = TRACKA + "/cvae_selection.json"
    tracks = [("B", sel["clusters"], cluster_sizes())]
    if os.path.exists(pa):
        with open(pa) as f:
            tracks.insert(0, ("A", json.load(f)["clusters"], A_SIZES))
    rows = []
    for track, cl, sizes in tracks:
        for c in CLUSTERS:
            v = cl[str(c)]
            row = {"track": track, "cluster": c, "n_households": sizes[c]}
            row.update({k: v.get(k) for k in SEL_TABLE_COLS})
            row["selected_posterior_std_mean"] = float(np.mean(v["selected_posterior_std_mean_per_dim"]))
            row["runtime_train_s_median"] = v.get("runtime_train_s_median",
                                                  v["runtime_train_s_total"] / N_RESTARTS)  # Track A: no suspension
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT + "/cvae_selection_table.csv", index=False)
    return df


def verify_path(c: int, s: int, ext: str) -> str:
    return f"{MODELS}/verify_c{c}_s{s:02d}.{ext}"


def stage_verify(args) -> dict:
    """Retrain from scratch, into models/verify_c<c>_s<ss>.*, (a) the selected restart of every cluster and (b) every
    restart whose recorded training time exceeds 10x the median (it was in progress while the machine was suspended),
    and compare with the stored restart: weights bitwise, epoch losses and test metrics exactly. The 200 restarts were
    trained in two invocations (an interruption after 143), so this is the evidence that the stored set is what the
    current data and code produce. -> cvae_verify_retrain.json"""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    df = pd.read_csv(OUT + "/cvae_restarts.csv")
    with open(OUT + "/cvae_selection.json") as f:
        sel = json.load(f)["clusters"]
    med = float(df["runtime_train_s"].median())
    targets = {}
    for c in CLUSTERS:
        targets[(c, int(sel[str(c)]["selected_seed"]))] = ["selected"]
    for _, r in df.loc[df["runtime_train_s"] > 10 * med].iterrows():
        targets.setdefault((int(r["cluster"]), int(r["seed"])), []).append("runtime_outlier")
    jobs = [make_job(TRAIN_PATH, TEST_PATH, c, s, verify_path(c, s, "pt"), verify_path(c, s, "json"), args.epochs)
            for (c, s) in sorted(targets)]
    if args.no_train:
        raise SystemExit(f"[verify] --no-train: the verify stage retrains {len(jobs)} restarts; refusing")
    log(f"[verify] retraining {len(jobs)} stored restarts from scratch with {args.workers} workers: "
        f"{[(c, s, '+'.join(w)) for (c, s), w in sorted(targets.items())]}")
    recs = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=ct.worker_init) as ex:
        futs = [ex.submit(ct.run_restart_job, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            recs[(r["cluster"], r["seed"])] = r
    out = {"definition": "retrain from scratch with lib.cvae_torch.run_restart_job (same job keys as the train stage), "
                         "compare with the stored restart", "median_runtime_train_s_all_restarts": med, "restarts": {}}
    ok_all = True
    for (c, s), why in sorted(targets.items()):
        r = recs[(c, s)]
        with open(json_path(c, s)) as f:
            st = json.load(f)
        mS, mV = ct.load_model(model_path(c, s)), ct.load_model(verify_path(c, s, "pt"))
        wdiff = max(float((a.detach() - b.detach()).abs().max()) for a, b in zip(mS.params_list(), mV.params_list()))
        same_metrics = {k: bool(r[k] == st[k]) for k in ("test_mse_post", "test_mse_prior", "train_kl_final",
                                                         "test_rmse_mean200_kwh_median", "conso_min", "conso_max")}
        rec = {"why": why, "max_abs_weight_diff": wdiff, "epoch_loss_identical": bool(r["epoch_loss"] == st["epoch_loss"]),
               "metrics_identical": same_metrics, "test_mse_post": r["test_mse_post"],
               "runtime_train_s_stored": st["runtime_train_s"], "runtime_train_s_retrain": r["runtime_train_s"]}
        ok = wdiff == 0.0 and rec["epoch_loss_identical"] and all(same_metrics.values())
        rec["identical"] = bool(ok)
        ok_all &= ok
        out["restarts"][f"c{c}_s{s:02d}"] = rec
        log(f"[verify] c{c} s{s:02d} ({'+'.join(why)}): max |weight diff| {wdiff:.1e}, epoch losses identical "
            f"{rec['epoch_loss_identical']}, metrics identical {all(same_metrics.values())}; runtime stored "
            f"{st['runtime_train_s']:.1f}s, retrain {r['runtime_train_s']:.1f}s")
    out["all_identical"] = bool(ok_all)
    jdump(out, OUT + "/cvae_verify_retrain.json")
    log(f"[verify] all retrained restarts identical to the stored ones: {ok_all}")
    if not ok_all:
        raise RuntimeError("stored restarts are not reproduced; see cvae_verify_retrain.json")
    return out


# ------------------------------------------------------------------------------------------------------------------
# CVAE: generation
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
    td = pd.DataFrame({"day_index": np.arange(len(dates)), "date": date_strings(dates)})
    td.to_csv(OUT + "/test_dates.csv", index=False)
    pa = TRACKA + "/test_dates.csv"
    if os.path.exists(pa):
        same = pd.read_csv(pa)["date"].astype(str).tolist() == td["date"].tolist()
        log(f"[generate] test dates identical to Track A's test_dates.csv: {same}")
        if not same:
            raise RuntimeError("Track B test dates differ from Track A's")
    real_low = blocks[0][ct.LOW_COLS].to_numpy(float)
    real_high = blocks[0][ct.HIGH_COLS].to_numpy(float)
    all_normal_days = np.where((real_low.sum(1) == 0) & (real_high.sum(1) == 0))[0]
    log(f"[generate] test days {dates.iloc[0].date()} .. {dates.iloc[-1].date()}; {len(all_normal_days)} of 92 test "
        f"days have an all-Normal real tariff; conditions identical across clusters; scenarios Low = low "
        f"{ef.WINDOW_TEXT['Low']}, High = high {ef.WINDOW_TEXT['High']}")
    for c in CLUSTERS:
        model = ct.load_model(selected_path(c))
        cols_ = model.meta["cond_cols"]
        assert cols_ == ct.COLS_COND
        C = blocks[c][cols_].to_numpy(dtype=np.float64)
        Y = blocks[c][ct.COLS_DATA].to_numpy(dtype=np.float64)
        S, scen = cvae_generate_cluster(model, C, c)
        assert S.shape == (N_SAMPLES, 92, 48) and np.isfinite(S).all()
        status = {"samples": save_array(OUT + f"/cvae_samples_c{c}.npy", S)}
        for name in SCENARIOS:
            assert scen[name].shape == (N_SAMPLES, 92, 48) and np.isfinite(scen[name]).all()
            status[name] = save_array(OUT + f"/cvae_scen_{name}_c{c}.npy", scen[name])
        if len(all_normal_days):
            assert np.array_equal(scen["Normal"][:, all_normal_days], S[:, all_normal_days])
        dev = S.mean(0) - Y
        rm = np.sqrt((dev ** 2).mean(1))
        log(f"[generate] c{c} seed {model.meta['seed']}: samples mean {S.mean():.4f} kWh vs obs {Y.mean():.4f}; "
            f"RMSE(mean of 200) median over days {np.median(rm):.4f} kWh (mean {rm.mean():.4f}); "
            f"mean across-sample sd {S.std(0, ddof=1).mean():.4f} kWh; scenario means N/L/H "
            f"{scen['Normal'].mean():.4f}/{scen['Low'].mean():.4f}/{scen['High'].mean():.4f}; files vs stored: {status}")


# ------------------------------------------------------------------------------------------------------------------
# CVAE: scenario effects (Track A definitions)
# ------------------------------------------------------------------------------------------------------------------
# phase-1 regions from the scenario windows (lib.effects, as Track A): Low in-window 9-19, 3 slots before (6-8) /
# after (20-22), afternoon-evening 24-47; High in-window 39-44, 3 slots before (36-38) / after (45-47), afternoon 24-35
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
    Hh = b[ct.HIGH_COLS].to_numpy(int)

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
    high_runs = [r for row in Hh for r in runs(row)]
    from collections import Counter
    lw, hw = (LOW_SLOTS[0], LOW_SLOTS[-1]), (HIGH_SLOTS[0], HIGH_SLOTS[-1])
    lk, hk = f"{lw[0]}_{lw[1]}", f"{hw[0]}_{hw[1]}"
    return {"train_days": int(len(b)), "train_days_with_low": int((L.sum(1) > 0).sum()),
            "train_days_with_high": int((Hh.sum(1) > 0).sum()),
            "train_days_with_both": int(((L.sum(1) > 0) & (Hh.sum(1) > 0)).sum()),
            "low_slot_frequency": L.sum(0).tolist(), "high_slot_frequency": Hh.sum(0).tolist(),
            "most_common_low_runs (start,end slots)": [[list(k), v] for k, v in Counter(low_runs).most_common(6)],
            "most_common_high_runs (start,end slots)": [[list(k), v] for k, v in Counter(high_runs).most_common(6)],
            f"train_days_with_low_run_exactly_{lk}": int(sum(1 for row in L if lw in runs(row))),
            f"train_days_with_high_run_exactly_{hk}": int(sum(1 for row in Hh if hw in runs(row))),
            f"train_days_with_low_covering_{lk}": int((L[:, LOW_SLOTS].sum(1) == len(LOW_SLOTS)).sum()),
            f"train_days_with_high_covering_{hk}": int((Hh[:, HIGH_SLOTS].sum(1) == len(HIGH_SLOTS)).sum()),
            "max_high_slots_in_a_train_day": int(Hh.sum(1).max()), "max_low_slots_in_a_train_day": int(L.sum(1).max())}


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
        eps = cvae_draws(c, len(C), model.latent_dim)
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
    jdump(res, OUT + "/cvae_scenario_effects.json")
    ctx = res["train_tariff_context"]
    lk, hk = f"{LOW_SLOTS[0]}_{LOW_SLOTS[-1]}", f"{HIGH_SLOTS[0]}_{HIGH_SLOTS[-1]}"
    log(f"[effects] Track B training tariffs: {ctx['train_days_with_low']} days with Low, {ctx['train_days_with_high']} "
        f"with High; days with a Low run exactly {ef.WINDOW_SPAN['Low']}: "
        f"{ctx[f'train_days_with_low_run_exactly_{lk}']}, High run exactly {ef.WINDOW_SPAN['High']}: "
        f"{ctx[f'train_days_with_high_run_exactly_{hk}']}; most common High runs "
        f"{ctx['most_common_high_runs (start,end slots)'][:3]}")
    return res


def _region_effects(model, C: np.ndarray, eps: np.ndarray) -> dict:
    cols_ = model.meta["cond_cols"]
    M = {name: ct.generate(model, scenario_cond(C, cols_, name), eps.shape[0], eps=eps).mean(0) for name in SCENARIOS}
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
        eps = cvae_draws(c, len(C), ct.LATENT_DIM)
        for s in range(N_RESTARTS):
            m = ct.load_model(model_path(c, s))
            r = rs.loc[(rs["cluster"] == c) & (rs["seed"] == s)].iloc[0]
            rows.append({"cluster": c, "seed": s, "selected": s == sel[str(c)]["selected_seed"],
                         "test_mse_post": float(r["test_mse_post"]), **_region_effects(m, C, eps)})
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
    jdump(res, OUT + "/cvae_scenario_effects_all_restarts.json")
    return res


# ------------------------------------------------------------------------------------------------------------------
# GAM generator (Track A's run_gam without the authors'-profile comparisons, which need the authors' clusters)
# ------------------------------------------------------------------------------------------------------------------
def energy_score(S, y):
    """eq. 14 per day; S (N, D, 48), y (D, 48) -> (D,)"""
    n2 = S.shape[0] // 2
    d1 = np.linalg.norm(S[:n2] - y[None], axis=2)
    d2 = np.linalg.norm(S[:n2] - S[n2:2 * n2], axis=2)
    return d1.mean(axis=0) - 0.5 * d2.mean(axis=0)


def variogram_score(S, y, p=0.5):
    """eq. 16 per day; S (N, D, 48), y (D, 48) -> (D,)"""
    out = np.empty(y.shape[0])
    for d in range(y.shape[0]):
        obs = np.abs(y[d][:, None] - y[d][None, :]) ** p
        pred = (np.abs(S[:, d, :, None] - S[:, d, None, :]) ** p).mean(axis=0)
        out[d] = ((obs - pred) ** 2).sum()
    return out


def day_rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean(axis=-1))


def gam_cluster_frames(tr: pd.DataFrame, te: pd.DataFrame, c: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    trc = tr[tr["cluster"] == c].sort_values("date").reset_index(drop=True)
    tec = te[te["cluster"] == c].sort_values("date").reset_index(drop=True)
    assert len(trc) == 273 and len(tec) == 92 and tec["date"].is_monotonic_increasing
    return trc, tec


def gam_fit_sample(trc: pd.DataFrame, tec: pd.DataFrame, c: int):
    """Track A's fit + sampling for one cluster: GCV fit; samples from stream 0, the three scenarios from stream 1 of
    SeedSequence([GAM_BASE_SEED, c]).spawn(2) (common random numbers across scenarios)."""
    from lib.gam_generator import SCENARIOS as GSCEN, fit_gam_generator, generate, tariff_scenario
    t0 = time.time()
    P = fit_gam_generator(trc, method="gcv")
    t_fit = time.time() - t0
    ss = np.random.SeedSequence([GAM_BASE_SEED, c])
    seed_samples, seed_scen = ss.spawn(2)
    S = generate(P, tec, N_SAMPLES, np.random.default_rng(seed_samples))
    assert S.shape == (N_SAMPLES, 92, H)
    scen = {}
    for name in GSCEN:  # common random numbers across the three scenarios
        scen[name] = generate(P, tariff_scenario(tec, name), N_SAMPLES, np.random.default_rng(seed_scen))
    return P, S, scen, seed_samples, t_fit


GAM_SCENARIO_TEXT = (f"Normal = all dummies 0; Low = low {ef.WINDOW_TEXT['Low']}; High = high {ef.WINDOW_TEXT['High']} "
                     f"(lib.effects windows, REPRO_SPEC section 7; phases 1-2 used 9-18 / 39-43)")


def gam_scenario_effects(scen: dict, info: dict) -> dict:
    """In-window effect and confinement checks of the GAM's Low / High scenarios (common random numbers)."""
    win = {k: np.array(v) for k, v in ef.WINDOWS.items()}
    scen_eff = {}
    for name in ("Low", "High"):
        diff = scen[name].mean(axis=0) - scen["Normal"].mean(axis=0)  # (92, 48)
        outside = np.setdiff1d(np.arange(H), win[name])
        xi = info["xi_low"] if name == "Low" else info["xi_high"]
        miss = info["xi_low_missing"] if name == "Low" else info["xi_high_missing"]
        scen_eff[name] = dict(
            window_slots=win[name].tolist(),
            window_clock_gmt=ef.WINDOW_CLOCK[name],
            mean_effect_in_window_kWh=float(diff[:, win[name]].mean()),
            mean_effect_in_window_pct=float(100 * diff[:, win[name]].mean()
                                            / scen["Normal"].mean(axis=0)[:, win[name]].mean()),
            xi_in_window=xi[win[name]],
            xi_missing_in_window=int(miss[win[name]].sum()),
            n_window_slots_with_expected_sign=int((xi[win[name]] > 0).sum() if name == "Low"
                                                  else (xi[win[name]] < 0).sum()),
            max_abs_mean_diff_outside_window_kWh=float(np.abs(diff[:, outside]).max()),
            max_abs_sample_diff_outside_window_kWh=float(
                np.abs(scen[name][:, :, outside] - scen["Normal"][:, :, outside]).max()),
            sample_sd_ratio_in_window=float(scen[name][:, :, win[name]].std(axis=0).mean()
                                            / scen["Normal"][:, :, win[name]].std(axis=0).mean()),
        )
    return scen_eff


def gam_scenario_log(se: dict) -> str:
    return (f"scenario Low: in-window mean effect {se['Low']['mean_effect_in_window_kWh']:+.4f} kWh "
            f"({se['Low']['mean_effect_in_window_pct']:+.1f}%), xi>0 in {se['Low']['n_window_slots_with_expected_sign']}/"
            f"{len(ef.LOW_WINDOW)} slots, max |diff| outside {se['Low']['max_abs_sample_diff_outside_window_kWh']:.2e}; "
            f"High: {se['High']['mean_effect_in_window_kWh']:+.4f} kWh ({se['High']['mean_effect_in_window_pct']:+.1f}%), "
            f"xi<0 in {se['High']['n_window_slots_with_expected_sign']}/{len(ef.HIGH_WINDOW)}, outside "
            f"{se['High']['max_abs_sample_diff_outside_window_kWh']:.2e}")


def stage_gamscen(args) -> dict:
    """No fitting: the stored GAMs (models/gam_params_c<c>.pkl) regenerate the samples (checked bitwise against the
    stored arrays, as is the deterministic mean f) and the three tariff scenarios with the 'gam' stage's seeds
    (SeedSequence([GAM_BASE_SEED, c]).spawn(2): samples stream 0, scenarios stream 1, common random numbers);
    the 'scenario_effects' entries and the scenario description of gam_diagnostics.json are recomputed."""
    from lib.gam_generator import SCENARIOS as GSCEN, generate, predict_mean, tariff_scenario
    te = load(TEST_PATH)
    diag_path = OUT + "/gam_diagnostics.json"
    with open(diag_path, encoding="utf-8") as f:
        diag = json.load(f)
    log(f"[gamscen] GAM scenarios from the stored models (no refit): {GAM_SCENARIO_TEXT}")
    for c in CLUSTERS:
        tec = te[te["cluster"] == c].sort_values("date").reset_index(drop=True)
        assert len(tec) == 92 and tec["date"].is_monotonic_increasing
        with open(MODELS + f"/gam_params_c{c}.pkl", "rb") as fh:
            P = pickle.load(fh)
        if not np.array_equal(predict_mean(P, tec), np.load(OUT + f"/gam_f_test_c{c}.npy")):
            raise RuntimeError(f"cluster {c}: the stored GAM does not reproduce gam_f_test_c{c}.npy")
        seed_samples, seed_scen = np.random.SeedSequence([GAM_BASE_SEED, c]).spawn(2)
        S = generate(P, tec, N_SAMPLES, np.random.default_rng(seed_samples))
        st = {"samples": save_array(OUT + f"/gam_samples_c{c}.npy", S)}
        if st["samples"] != "unchanged":
            raise RuntimeError(f"cluster {c}: regenerated GAM samples differ from the stored ones ({st['samples']})")
        scen = {}
        for name in GSCEN:  # common random numbers across the three scenarios
            scen[name] = generate(P, tariff_scenario(tec, name), N_SAMPLES, np.random.default_rng(seed_scen))
            st[name] = save_array(OUT + f"/gam_scen_{name}_c{c}.npy", scen[name])
        se = gam_scenario_effects(scen, P["info"])
        diag["clusters"][str(c)]["scenario_effects"] = jsonable(se)
        log(f"[gamscen] cluster {c}: arrays vs stored {st}; " + gam_scenario_log(se))
    diag["settings"]["scenarios"] = GAM_SCENARIO_TEXT
    diag["settings"]["scenario_entries_regenerated"] = (
        f"{time.strftime('%Y-%m-%d %H:%M')}: scenario arrays and 'scenario_effects' regenerated from the stored models "
        "(30_trackB_models.py --stage gamscen, no refit); every other entry is from the 'gam' stage run")
    jdump(diag, diag_path)
    return diag


def stage_gam(args) -> dict:
    from lib.causality import fit_causality
    from lib.gam_generator import fit_gam_generator, generate, predict_mean, set_sigma
    if args.no_train:
        raise SystemExit("[gam] --no-train: the gam stage fits the GAMs; use --stage gamscen to regenerate scenarios")
    os.makedirs(MODELS, exist_ok=True)
    tr, te, pw = load(TRAIN_PATH), load(TEST_PATH), load(POWER_PATH)
    sizes = cluster_sizes()
    diag = dict(clusters={}, settings={}, paper_fig4_approx_medians=PAPER_FIG4, cluster_sizes=sizes)
    smooth_rows, sigma_rows = [], []
    for c in CLUSTERS:
        trc, tec = gam_cluster_frames(tr, te, c)
        P, S, scen, seed_samples, t_fit = gam_fit_sample(trc, tec, c)
        with open(MODELS + f"/gam_params_c{c}.pkl", "wb") as fh:
            pickle.dump(P, fh)
        y_tr = trc[cols("consumption")].to_numpy(float)
        y_te = tec[cols("consumption")].to_numpy(float)
        f_te = predict_mean(P, tec)
        np.save(OUT + f"/gam_f_test_c{c}.npy", f_te)
        np.save(OUT + f"/gam_samples_c{c}.npy", S)
        for name in SCENARIOS:
            np.save(OUT + f"/gam_scen_{name}_c{c}.npy", scen[name])
        # ---- sensitivity: REML smoothing (fit quality only)
        P_reml = fit_gam_generator(trc, method="reml")
        f_te_reml = predict_mean(P_reml, tec)

        # ---- diagnostics
        rm_tr, rm_te = day_rmse(P["f_train"], y_tr), day_rmse(f_te, y_te)
        smean = S.mean(axis=0)
        rm_smean = day_rmse(smean, y_te)
        es, vg = energy_score(S, y_te), variogram_score(S, y_te)
        resid_te = y_te - f_te
        info = P["info"]
        sig_variants = {}
        pcc = pw[pw["cluster"] == c].sort_values("date")
        for vname in ("causality_all365_days", "causality_train_no_sd_floor"):
            if vname == "causality_all365_days":
                table = fit_causality(pcc[cols("consumption")].to_numpy(float), pcc[cols("temp")].to_numpy(float),
                                      pcc[cols("low")].to_numpy(float), pcc[cols("high")].to_numpy(float))["sigma"][0]
            else:
                table = fit_causality(y_tr, trc[cols("temp")].to_numpy(float), trc[cols("low")].to_numpy(float),
                                      trc[cols("high")].to_numpy(float), sd_floor=1e-9)["sigma"][0]
            Pv = set_sigma(dict(P), table, source=vname)
            Sv = generate(Pv, tec, N_SAMPLES, np.random.default_rng(seed_samples))
            sig_variants[vname] = dict(
                es_median=float(np.median(energy_score(Sv, y_te))),
                vg_median=float(np.median(variogram_score(Sv, y_te))),
                sample_sd_mean=float(Sv.std(axis=0, ddof=1).mean()),
                sigma_median=float(np.median(Pv["sigma"])))
        scen_eff = gam_scenario_effects(scen, info)
        rough = lambda Z: np.abs(np.diff(Z, axis=-1)).mean()  # noqa: E731
        lag1 = lambda Z: np.mean([np.corrcoef(Z.reshape(-1, H)[:, h], Z.reshape(-1, H)[:, h + 1])[0, 1]  # noqa: E731
                                  for h in range(H - 1)])
        dc = dict(
            fit_seconds=round(t_fit, 2),
            train_rmse_f=dict(mean=rm_tr.mean(), median=np.median(rm_tr)),
            test_rmse_f=dict(mean=rm_te.mean(), median=np.median(rm_te),
                             euclidean_median=np.median(rm_te) * np.sqrt(H)),
            test_rmse_f_reml=dict(mean=day_rmse(f_te_reml, y_te).mean(),
                                  median=np.median(day_rmse(f_te_reml, y_te))),
            train_rmse_f_reml_mean=day_rmse(P_reml["f_train"], y_tr).mean(),
            test_rmse_sample_mean=dict(mean=rm_smean.mean(), median=np.median(rm_smean),
                                       euclidean_median=np.median(np.linalg.norm(smean - y_te, axis=1))),
            energy_score=dict(
                mean=es.mean(), median=np.median(es),
                median_term_mean_dist_to_obs=np.median(
                    np.linalg.norm(S[:N_SAMPLES // 2] - y_te[None], axis=2).mean(axis=0)),
                median_term_mean_dist_between_samples=np.median(
                    np.linalg.norm(S[:N_SAMPLES // 2] - S[N_SAMPLES // 2:], axis=2).mean(axis=0))),
            variogram_score_p05=dict(mean=vg.mean(), median=np.median(vg)),
            sigma_variants_scores=sig_variants,
            dispersion=dict(
                mean_sample_sd=float(S.std(axis=0, ddof=1).mean()),
                rms_test_residual_y_minus_f=float(np.sqrt((resid_te ** 2).mean())),
                ratio_sample_sd_to_rms_residual=float(S.std(axis=0, ddof=1).mean() / np.sqrt((resid_te ** 2).mean())),
                train_std_resid_sd_mean=float(P["std_resid_sd"].mean()),
                train_std_resid_sd_range=[float(P["std_resid_sd"].min()), float(P["std_resid_sd"].max())],
            ),
            roughness_mean_abs_first_diff=dict(samples=rough(S), observed_test=rough(y_te), f_test=rough(f_te)),
            lag1_corr_across_halfhours=dict(samples_minus_f=lag1(S - f_te[None]),
                                            observed_residuals=lag1(resid_te[None])),
            frac_negative_samples=float((S < 0).mean()),
            sigma=dict(median_LNH=np.median(P["sigma"], axis=0), n_fallback=int(P["sigma_fallback"].sum()),
                       frac_at_floor_0p01=float(np.isclose(P["sigma"], 0.01).mean())),
            xi_missing=dict(low=int(info["xi_low_missing"].sum()), high=int(info["xi_high_missing"].sum())),
            smoothing_median=dict(g_tau_tbar_kappa=np.median(info["g"], axis=0),
                                  edf_tau_tbar_kappa=np.median(info["edf_smooth"], axis=0),
                                  edf_total=np.median(info["edf"])),
            frac_smoothing_at_bounds=float(np.mean(np.abs(info["g"]) >= 7.999)),
            frac_g_on_coarse_grid=float(np.mean(np.isclose(info["g"], np.round(info["g"] / 2) * 2, atol=1e-3))),
            Sigma_offdiag_lag1_mean=float(np.mean(np.diag(P["Sigma"], 1))),
            jitter_used=P["jitter_used"],
            scenario_effects=scen_eff,
        )
        diag["clusters"][c] = dc
        for h in range(H):
            smooth_rows.append(dict(cluster=c, h=h, g_tau=info["g"][h, 0], g_tbar=info["g"][h, 1],
                                    g_kappa=info["g"][h, 2], lam_tau=info["lam"][h, 0],
                                    lam_tbar=info["lam"][h, 1], lam_kappa=info["lam"][h, 2],
                                    edf_tau=info["edf_smooth"][h, 0], edf_tbar=info["edf_smooth"][h, 1],
                                    edf_kappa=info["edf_smooth"][h, 2], edf_total=info["edf"][h],
                                    gcv=info["crit"][h], b0=info["b0"][h], alpha_workday=info["alpha"][h],
                                    xi_low=info["xi_low"][h], xi_high=info["xi_high"][h],
                                    xi_low_missing=info["xi_low_missing"][h],
                                    xi_high_missing=info["xi_high_missing"][h]))
            cz = P["causality"]
            sigma_rows.append(dict(cluster=c, h=h, sigma_low=P["sigma"][h, 0], sigma_normal=P["sigma"][h, 1],
                                   sigma_high=P["sigma"][h, 2], fallback_low=P["sigma_fallback"][h, 0],
                                   fallback_high=P["sigma_fallback"][h, 2], n_low_train=cz["counts"][h, 0],
                                   n_normal_train=cz["counts"][h, 1], n_high_train=cz["counts"][h, 2],
                                   causality_edf=cz["edf"][h], std_resid_sd=P["std_resid_sd"][h]))
        log(f"   cluster {c} ({sizes[c]} households): fit {t_fit:.1f}s | RMSE f train {rm_tr.mean():.4f} test "
            f"{rm_te.mean():.4f} (median {np.median(rm_te):.4f}; REML {np.median(day_rmse(f_te_reml, y_te)):.4f}) | "
            f"sample-mean RMSE median {np.median(rm_smean):.4f} (paper ~{PAPER_FIG4['rmse_gam'][c]}) | "
            f"ES median {np.median(es):.4f} (paper ~{PAPER_FIG4['es_gam'][c]}) | VG median {np.median(vg):.2f} "
            f"(paper ~{PAPER_FIG4['vg_gam'][c]}) | std-resid sd mean {P['std_resid_sd'].mean():.3f} | "
            f"sample sd / rms resid {dc['dispersion']['ratio_sample_sd_to_rms_residual']:.2f}")
        log("      " + gam_scenario_log(scen_eff) + f"; sigma fallbacks {dc['sigma']['n_fallback']}, "
            f"xi missing L/H {dc['xi_missing']['low']}/{dc['xi_missing']['high']}")
    diag["settings"] = dict(
        gam=dict(k=10, basis="natural cubic regression spline (mgcv cr analogue), centred",
                 smoothing="GCV (mgcv default GCV.Cp), grid -6..6 step 2 per smooth + L-BFGS-B",
                 terms="b0 + s(temp_hh) + s(temp_smooth) + s(pos_y) + work_day + low_hh + high_hh",
                 fit_days="train_ours (273 days = the authors' train dates)"),
        sigma="causality model (LAML, per-series lambda, sd floor 0.01) on the cluster's 273 training days; "
              "NaN -> sigma(Normal)",
        Sigma="np.corrcoef of standardized training residuals (eqs. 37-40); Cholesky",
        seeds=dict(base=GAM_BASE_SEED, scheme="SeedSequence([base, cluster]).spawn(2): [samples, scenarios]; "
                   "the three scenarios share one stream (common random numbers)"),
        n_samples=N_SAMPLES, test_days="92, ascending dates",
        scenarios=GAM_SCENARIO_TEXT,
        sigma_variant_scores=("ES/VG medians with sigma replaced by the causality fit on all 365 days and by the "
                              "train fit without the 0.01 sd floor; same random stream. (Track A's variants based "
                              "on the authors' agreg_profiles 'variance' column do not apply to our clusters.)"),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
    )
    jdump(diag, OUT + "/gam_diagnostics.json")
    pd.DataFrame(smooth_rows).to_csv(OUT + "/gam_smoothing.csv", index=False)
    pd.DataFrame(sigma_rows).to_csv(OUT + "/gam_sigma.csv", index=False)
    return diag


# ------------------------------------------------------------------------------------------------------------------
# mirror check: this script's code paths reproduce Track A
# ------------------------------------------------------------------------------------------------------------------
def _import_script(name: str, path: str):
    """Import one of OUR Track A scripts as a module (top level only: constants and functions; main() is not run)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_mirror_regen(args) -> dict:
    """The mirror check without any training or fitting (scenario-window change): (a) constants and scenario
    construction equal to the Track A scripts'; (b) this script's CVAE generation code applied to Track A's selected
    models reproduces Track A's stored arrays bitwise; (c) this script's GAM generation code (gamscen path) applied to
    Track A's stored GAMs reproduces Track A's stored GAM arrays bitwise. The retraining / refit checks of the last full
    'mirror' run (window-independent, except that its GAM scenario comparison used the windows of that time) are
    carried over unchanged."""
    from lib import gam_generator as gg
    path = OUT + "/cvae_gam_mirror_check_trackA.json"
    prev = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            prev = json.load(f)
    carried = prev.get("carried_over_from_full_mirror_run") or {
        "checks": {k: prev["checks"][k] for k in ("cvae_training_job_reproduces_trackA_restart",
                                                  "gam_code_reproduces_trackA_cluster3") if k in prev.get("checks", {})},
        "all_pass": prev.get("all_pass"),
        "note": "from the last full 'mirror' stage run (2026-11-12 08:51, scenario windows 9-18 / 39-43 at the time): "
                "retraining Track A's selected cluster-3 restart and refitting its GAM are independent of the scenario "
                "windows; the GAM refit's scenario-array comparison was made with the old windows"}
    A = _import_script("trackA_cvae_script", REPRO + "/scripts/10_trackA_cvae.py")
    G = _import_script("trackA_gam_script", REPRO + "/scripts/11_trackA_gam.py")
    res = {"purpose": "run-time evidence that Track B uses exactly Track A's settings and code paths (re-check after "
                      "the scenario-window correction, without retraining or refitting)",
           "scenario_windows": {k: {"slots": v, "clock_gmt": ef.WINDOW_CLOCK[k]} for k, v in ef.WINDOWS.items()},
           "checks": {}}
    chk = res["checks"]
    consts = {"GEN_SEED": (A.GEN_SEED, GEN_SEED), "N_RESTARTS": (A.N_RESTARTS, N_RESTARTS),
              "N_SAMPLES": (A.N_SAMPLES, N_SAMPLES), "LOW_SLOTS": (A.LOW_SLOTS, LOW_SLOTS),
              "HIGH_SLOTS": (A.HIGH_SLOTS, HIGH_SLOTS), "SCENARIOS": (tuple(A.SCENARIOS), SCENARIOS),
              "CLUSTERS": (tuple(A.CLUSTERS), CLUSTERS), "REGIONS": (A.REGIONS, REGIONS),
              "CLAIMED_SIGN": (A.CLAIMED_SIGN, CLAIMED_SIGN), "gam.BASE_SEED": (G.BASE_SEED, GAM_BASE_SEED),
              "gam.N_SAMPLES": (G.N_SAMPLES, N_SAMPLES), "gam.PAPER_FIG4": (G.PAPER_FIG4, PAPER_FIG4),
              "LOW_SLOTS == lib.effects.LOW_WINDOW": (LOW_SLOTS, ef.LOW_WINDOW),
              "HIGH_SLOTS == lib.effects.HIGH_WINDOW": (HIGH_SLOTS, ef.HIGH_WINDOW),
              "gam_generator.SCENARIOS windows == lib.effects": (
                  [list(gg.SCENARIOS["Low"][1]), list(gg.SCENARIOS["High"][1])], [ef.LOW_WINDOW, ef.HIGH_WINDOW])}
    chk["constants_equal_to_trackA_scripts"] = {k: bool(a == b) for k, (a, b) in consts.items()}
    chk["gam_lib_scenario_order_equals_SCENARIOS"] = bool(tuple(gg.SCENARIOS) == SCENARIOS)
    teB = load(TEST_PATH)
    CB = test_block(teB, 0)[ct.COLS_COND].to_numpy(np.float64)
    chk["cvae_scenario_cond_equal_trackA_function_on_trackB_conditions"] = {
        n: bool(np.array_equal(scenario_cond(CB, ct.COLS_COND, n), A.scenario_cond(CB, ct.COLS_COND, n)))
        for n in SCENARIOS + ("AllHigh",)}
    teA = load(A_TEST_PATH)
    gen, gam = {}, {}
    for c in CLUSTERS:
        m = ct.load_model(TRACKA + f"/models/cvae_selected_c{c}.pt")
        CA = test_block(teA, c)[m.meta["cond_cols"]].to_numpy(np.float64)
        S, scen = cvae_generate_cluster(m, CA, c)
        eq = {"samples": bool(np.array_equal(S, np.load(TRACKA + f"/cvae_samples_c{c}.npy")))}
        for n in SCENARIOS:
            eq[n] = bool(np.array_equal(scen[n], np.load(TRACKA + f"/cvae_scen_{n}_c{c}.npy")))
        gen[str(c)] = eq
        with open(TRACKA + f"/models/gam_params_c{c}.pkl", "rb") as fh:
            PA = pickle.load(fh)
        tec = teA[teA["cluster"] == c].sort_values("date").reset_index(drop=True)
        s_samp, s_scen = np.random.SeedSequence([GAM_BASE_SEED, c]).spawn(2)
        g = {"samples": bool(np.array_equal(gg.generate(PA, tec, N_SAMPLES, np.random.default_rng(s_samp)),
                                             np.load(TRACKA + f"/gam_samples_c{c}.npy")))}
        for n in SCENARIOS:
            g[n] = bool(np.array_equal(gg.generate(PA, gg.tariff_scenario(tec, n), N_SAMPLES,
                                                   np.random.default_rng(s_scen)),
                                       np.load(TRACKA + f"/gam_scen_{n}_c{c}.npy")))
        gam[str(c)] = g
    chk["cvae_generation_code_reproduces_trackA_arrays_bitwise"] = gen
    chk["gam_generation_code_on_trackA_stored_gams_reproduces_trackA_arrays_bitwise"] = gam
    log(f"[mirror_regen] CVAE generation on Track A models bitwise equal: {gen}")
    log(f"[mirror_regen] GAM generation on Track A stored GAMs bitwise equal: {gam}")
    ok = (all(chk["constants_equal_to_trackA_scripts"].values()) and chk["gam_lib_scenario_order_equals_SCENARIOS"]
          and all(chk["cvae_scenario_cond_equal_trackA_function_on_trackB_conditions"].values())
          and all(all(v.values()) for v in gen.values()) and all(all(v.values()) for v in gam.values()))
    res["all_pass_without_retraining"] = bool(ok)
    res["carried_over_from_full_mirror_run"] = carried
    res["all_pass"] = bool(ok and carried.get("all_pass", False))
    jdump(res, path)
    log(f"[mirror_regen] all checks pass: {ok} (carried-over training checks pass: {carried.get('all_pass')})")
    if not ok:
        raise RuntimeError("mirror check failed; see cvae_gam_mirror_check_trackA.json")
    return res


def stage_mirror(args) -> dict:
    if args.no_train:
        raise SystemExit("[mirror] --no-train: the mirror stage retrains one restart and refits one GAM; use "
                         "--stage mirror_regen")
    A = _import_script("trackA_cvae_script", REPRO + "/scripts/10_trackA_cvae.py")
    G = _import_script("trackA_gam_script", REPRO + "/scripts/11_trackA_gam.py")
    res = {"purpose": "run-time evidence that Track B uses exactly Track A's settings and code paths", "checks": {}}
    chk = res["checks"]
    consts = {"GEN_SEED": (A.GEN_SEED, GEN_SEED), "N_RESTARTS": (A.N_RESTARTS, N_RESTARTS),
              "N_SAMPLES": (A.N_SAMPLES, N_SAMPLES), "LOW_SLOTS": (A.LOW_SLOTS, LOW_SLOTS),
              "HIGH_SLOTS": (A.HIGH_SLOTS, HIGH_SLOTS), "SCENARIOS": (tuple(A.SCENARIOS), SCENARIOS),
              "CLUSTERS": (tuple(A.CLUSTERS), CLUSTERS), "REGIONS": (A.REGIONS, REGIONS),
              "CLAIMED_SIGN": (A.CLAIMED_SIGN, CLAIMED_SIGN), "gam.BASE_SEED": (G.BASE_SEED, GAM_BASE_SEED),
              "gam.N_SAMPLES": (G.N_SAMPLES, N_SAMPLES), "gam.PAPER_FIG4": (G.PAPER_FIG4, PAPER_FIG4)}
    chk["constants_equal_to_trackA_scripts"] = {k: bool(a == b) for k, (a, b) in consts.items()}
    from lib import gam_generator as gg
    chk["gam_lib_scenario_order_equals_SCENARIOS"] = bool(tuple(gg.SCENARIOS) == SCENARIOS)
    # scenario construction on Track B's real test conditions
    teB = load(TEST_PATH)
    CB = test_block(teB, 0)[ct.COLS_COND].to_numpy(np.float64)
    chk["cvae_scenario_cond_equal_trackA_function_on_trackB_conditions"] = {
        n: bool(np.array_equal(scenario_cond(CB, ct.COLS_COND, n), A.scenario_cond(CB, ct.COLS_COND, n)))
        for n in SCENARIOS + ("AllHigh",)}
    # (a) this script's CVAE generation code applied to Track A's selected models reproduces Track A's arrays
    teA = load(A_TEST_PATH)
    gen = {}
    for c in CLUSTERS:
        m = ct.load_model(TRACKA + f"/models/cvae_selected_c{c}.pt")
        CA = test_block(teA, c)[m.meta["cond_cols"]].to_numpy(np.float64)
        S, scen = cvae_generate_cluster(m, CA, c)
        eq = {"samples": bool(np.array_equal(S, np.load(TRACKA + f"/cvae_samples_c{c}.npy")))}
        for n in SCENARIOS:
            eq[n] = bool(np.array_equal(scen[n], np.load(TRACKA + f"/cvae_scen_{n}_c{c}.npy")))
        gen[str(c)] = eq
    chk["cvae_generation_code_reproduces_trackA_arrays_bitwise"] = gen
    log(f"[mirror] CVAE generation on Track A models bitwise equal: {gen}")
    # (b) this script's job builder + lib training on Track A data reproduces Track A's selected cluster-3 restart
    with open(TRACKA + "/cvae_selection.json") as f:
        selA = json.load(f)["clusters"]
    c, s = FIG_CLUSTER, int(selA[str(FIG_CLUSTER)]["selected_seed"])
    job = make_job(A_TRAIN_PATH, A_TEST_PATH, c, s, MODELS + f"/mirror_trackA_c{c}_s{s:02d}.pt",
                   MODELS + f"/mirror_trackA_c{c}_s{s:02d}.json", ct.EPOCHS)
    os.makedirs(MODELS, exist_ok=True)
    t0 = time.perf_counter()
    r = ct.run_restart_job(job)
    with open(TRACKA + f"/models/cvae_c{c}_s{s:02d}.json") as f:
        rA = json.load(f)
    mA, mB = ct.load_model(TRACKA + f"/models/cvae_c{c}_s{s:02d}.pt"), ct.load_model(job["model_path"])
    wdiff = max(float((a.detach() - b.detach()).abs().max()) for a, b in zip(mA.params_list(), mB.params_list()))
    chk["cvae_training_job_reproduces_trackA_restart"] = {
        "cluster": c, "seed": s, "test_mse_post_trackA": rA["test_mse_post"], "test_mse_post_rerun": r["test_mse_post"],
        "max_abs_weight_diff": wdiff, "epoch_loss_identical": bool(r["epoch_loss"] == rA["epoch_loss"]),
        "runtime_s": time.perf_counter() - t0}
    log(f"[mirror] Track A c{c} s{s:02d} retrained: test MSE {r['test_mse_post']:.10f} vs Track A "
        f"{rA['test_mse_post']:.10f}; max |weight diff| {wdiff:.2e}")
    # (c) this script's GAM code on Track A's cluster 3 reproduces Track A's GAM arrays
    trA = load(A_TRAIN_PATH)
    trc, tec = gam_cluster_frames(trA, teA, c)
    P, S, scen, _, t_fit = gam_fit_sample(trc, tec, c)
    g = {"samples_max_abs_diff_kwh": float(np.abs(S - np.load(TRACKA + f"/gam_samples_c{c}.npy")).max())}
    for n in SCENARIOS:
        g[f"{n}_max_abs_diff_kwh"] = float(np.abs(scen[n] - np.load(TRACKA + f"/gam_scen_{n}_c{c}.npy")).max())
    with open(TRACKA + f"/models/gam_params_c{c}.pkl", "rb") as fh:
        PA = pickle.load(fh)
    g["smoothing_g_max_abs_diff"] = float(np.abs(P["info"]["g"] - PA["info"]["g"]).max())
    g["sigma_max_abs_diff"] = float(np.abs(P["sigma"] - PA["sigma"]).max())
    g["note"] = (f"this refit ran with OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')!s} vs 4 in "
                 "11_trackA_gam.py, so rounding-level differences would be acceptable; anything above 1e-8 kWh would not")
    g["fit_seconds"] = t_fit
    chk["gam_code_reproduces_trackA_cluster3"] = g
    log(f"[mirror] GAM Track A c{c} refit: max |sample diff| {g['samples_max_abs_diff_kwh']:.2e} kWh, scenarios "
        f"{max(g[f'{n}_max_abs_diff_kwh'] for n in SCENARIOS):.2e}")
    ok = (all(chk["constants_equal_to_trackA_scripts"].values()) and chk["gam_lib_scenario_order_equals_SCENARIOS"]
          and all(chk["cvae_scenario_cond_equal_trackA_function_on_trackB_conditions"].values())
          and all(all(v.values()) for v in gen.values())
          and chk["cvae_training_job_reproduces_trackA_restart"]["max_abs_weight_diff"] == 0.0
          and max(g["samples_max_abs_diff_kwh"], *[g[f"{n}_max_abs_diff_kwh"] for n in SCENARIOS]) < 1e-8)
    res["all_pass"] = bool(ok)
    jdump(res, OUT + "/cvae_gam_mirror_check_trackA.json")
    log(f"[mirror] all checks pass: {ok}")
    if not ok:
        raise RuntimeError("mirror check failed; see cvae_gam_mirror_check_trackA.json")
    return res


# ------------------------------------------------------------------------------------------------------------------
# sanity: Track B vs Track A (and the paper)
# ------------------------------------------------------------------------------------------------------------------
def _qs(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"median": float(np.median(x)), "mean": float(x.mean()), "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75))}


def _effects_compact(e: dict) -> dict:
    keep = {"Low": ("in_window", "pre2", "pre4", "post2", "post4", ef.KEY_LOW_REBOUND, "rebound_32_47", "daily_energy"),
            "High": ("in_window", "pre2", "pre4", "post2", "post4", "rebound_pre_24_38", "afternoon_24_34",
                     "daily_energy")}
    out = {}
    for name, regs in keep.items():
        r = e[name]["regions"]
        out[name] = {k: {"pct": r[k]["pct"], "pct_ci95": r[k]["pct_ci95"], "kwh": r[k]["kwh"],
                         "claim_status": r[k].get("claim_status")} for k in regs}
        out[name]["max_abs_outside_window_kwh"] = r["max_abs_outside_window_kwh"]
    return out


def stage_sanity(args) -> dict:
    import torch
    from lib import effects as ef
    torch.set_num_threads(1)
    obsA, datesA = obs_blocks(A_TEST_PATH)
    obsB, datesB = obs_blocks(TEST_PATH)
    if date_strings(datesA) != date_strings(datesB):
        raise RuntimeError("Track A and Track B test dates differ")
    sizes = cluster_sizes()
    res = {"definition": {
        "rmse_rootmean": "per test day sqrt(mean_h (mean of the 200 samples - observed)^2), kWh; summarized over the "
                         "92 test days (the Fig. 4 scale; spec section 3)",
        "rmse_euclid": "per test day ||mean of the 200 samples - observed||_2 over the 48 half-hours (eq. 12 literally)",
        "sample_sd": "SD across the 200 samples, averaged over days and half-hours (kWh)",
        "active_units": "latent dims with Var_days(posterior mean) > 0.01 on the training days (Burda et al. 2016)",
        "effects": "lib.effects.scenario_effects (day-bootstrap 95% CIs, B=2000, seed 1000+c); pct of the Normal "
                   "scenario over the same slots and days; scenario windows Low "
                   f"{ef.WINDOW_TEXT['Low']}, High {ef.WINDOW_TEXT['High']} (High post4 = 45-47, 3 slots)"},
        "matched_cluster_to_authors_3": matched_cluster(FIG_CLUSTER), "cluster_sizes_trackB": sizes,
        "cluster_sizes_trackA": A_SIZES, "rmse": {}, "observed": {}, "collapse": {}, "gam_dispersion": {},
        "scenario_effects": {}, "independent_recomputation": {}}
    # --- RMSE of the 200-sample mean, both models, both tracks (recomputed here from the arrays, two ways)
    tbl = []
    for model in MODEL_NAMES:
        for c in CLUSTERS:
            rec = {}
            for track, outdir, obs in (("A", TRACKA, obsA), ("B", OUT, obsB)):
                S = np.load(f"{outdir}/{model}_samples_c{c}.npy")
                Y = obs[c]
                assert S.shape == (N_SAMPLES, 92, 48) and np.isfinite(S).all()
                r = np.sqrt(((S.mean(0) - Y) ** 2).mean(1))
                r_loop = np.array([np.linalg.norm(S[:, d, :].sum(0) / N_SAMPLES - Y[d]) / np.sqrt(48.0)
                                   for d in range(Y.shape[0])])
                assert np.allclose(r, r_loop, rtol=1e-10, atol=0), np.abs(r - r_loop).max()
                eu = np.linalg.norm(S.mean(0) - Y, axis=1)
                rec[track] = {"rmse_rootmean": _qs(r), "rmse_euclid": _qs(eu),
                              "sample_sd_mean_kwh": float(S.std(0, ddof=1).mean()),
                              "mean_level_samples_kwh": float(S.mean()), "mean_level_observed_kwh": float(Y.mean())}
            rec["paper_fig4_median_approx"] = PAPER_FIG4[f"rmse_{model}"][c]
            rec["B_over_A_median_rmse"] = rec["B"]["rmse_rootmean"]["median"] / rec["A"]["rmse_rootmean"]["median"]
            rec["B_over_paper_median_rmse"] = rec["B"]["rmse_rootmean"]["median"] / rec["paper_fig4_median_approx"]
            res["rmse"][f"{model}_c{c}"] = rec
            tbl.append(f"{MODEL_LABEL[model]:4s} c{c}: B {rec['B']['rmse_rootmean']['median']:.4f} | A "
                       f"{rec['A']['rmse_rootmean']['median']:.4f} | paper ~{rec['paper_fig4_median_approx']:.3f} | "
                       f"B/A {rec['B_over_A_median_rmse']:.3f} | sample sd B {rec['B']['sample_sd_mean_kwh']:.4f} "
                       f"A {rec['A']['sample_sd_mean_kwh']:.4f}")
    for line in tbl:
        log(f"[sanity] RMSE(mean of 200), median over test days (kWh): {line}")
    for track in ("A", "B"):
        g = np.array([res["rmse"][f"gam_c{c}"][track]["rmse_rootmean"]["median"] for c in CLUSTERS])
        v = np.array([res["rmse"][f"cvae_c{c}"][track]["rmse_rootmean"]["median"] for c in CLUSTERS])
        res["rmse"][f"gam_vs_cvae_track{track}"] = {
            "per_cluster_cvae_over_gam": (v / g).tolist(), "per_cluster_gam_lower_pct": (100 * (1 - g / v)).tolist(),
            "mean_of_medians_gam_lower_pct": float(100 * (1 - g.mean() / v.mean())),
            "gam_lower_in_all_clusters": bool((g < v).all())}
        log(f"[sanity] track {track}: GAM RMSE lower than CVAE by {np.round(100 * (1 - g / v), 1).tolist()} % "
            f"(mean of medians {100 * (1 - g.mean() / v.mean()):.1f}%; paper abstract: 13%)")
    # --- observed series: how different are our cluster series from the authors' (test days)?
    for c in CLUSTERS:
        a, b = obsA[c], obsB[c]
        res["observed"][str(c)] = {"mean_kwh_A": float(a.mean()), "mean_kwh_B": float(b.mean()),
                                   "corr_92x48": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
                                   "rms_diff_kwh": float(np.sqrt(((a - b) ** 2).mean())),
                                   "day_to_day_sd_of_daily_mean_A": float(a.mean(1).std(ddof=1)),
                                   "day_to_day_sd_of_daily_mean_B": float(b.mean(1).std(ddof=1)),
                                   "n_households_A": A_SIZES[c], "n_households_B": sizes[c]}
        o = res["observed"][str(c)]
        log(f"[sanity] observed test series c{c}: mean A {o['mean_kwh_A']:.4f} / B {o['mean_kwh_B']:.4f} kWh, corr "
            f"{o['corr_92x48']:.4f}, rms diff {o['rms_diff_kwh']:.4f} kWh ({A_SIZES[c]} vs {sizes[c]} households)")
    # --- posterior collapse (all 50 restarts of both tracks; selected models recomputed from scratch)
    teB, trB = load(TEST_PATH), load(TRAIN_PATH)
    for track, outdir in (("A", TRACKA), ("B", OUT)):
        df = pd.read_csv(outdir + "/cvae_restarts.csv")
        with open(outdir + "/cvae_selection.json") as f:
            sel = json.load(f)["clusters"]
        for c in CLUSTERS:
            d = df.loc[df["cluster"] == c]
            mstd = d[[f"train_posterior_mean_std_{i}" for i in range(4)]].to_numpy(float)
            pstd = d[[f"train_posterior_std_mean_{i}" for i in range(4)]].to_numpy(float)
            s = int(sel[str(c)]["selected_seed"])
            b = d.loc[d["seed"] == s].iloc[0]
            res["collapse"][f"track{track}_c{c}"] = {
                "train_kl_final_median": float(d["train_kl_final"].median()),
                "train_kl_final_max": float(d["train_kl_final"].max()),
                "test_kl_final_max": float(d["test_kl_final"].max()),
                "posterior_std_min_over_restarts_dims": float(pstd.min()),
                "posterior_std_max_over_restarts_dims": float(pstd.max()),
                "posterior_mean_sd_max_over_restarts_dims": float(mstd.max()),
                "n_restarts_with_any_active_unit": int(((mstd ** 2) > 0.01).any(axis=1).sum()),
                "selected_seed": s, "selected_train_kl_final": float(b["train_kl_final"]),
                "selected_sample_sd_kwh_mean": float(b["test_sample_sd_kwh_mean"]),
                "selected_rmse_mean200_kwh_median": float(b["test_rmse_mean200_kwh_median"]),
                "selected_sd_over_rmse": float(b["test_sample_sd_kwh_mean"] / b["test_rmse_mean200_kwh_median"])}
        cl = [res["collapse"][f"track{track}_c{c}"] for c in CLUSTERS]
        log(f"[sanity] collapse track {track}: train KL max over 200 restarts {max(x['train_kl_final_max'] for x in cl):.2e}; "
            f"posterior std range [{min(x['posterior_std_min_over_restarts_dims'] for x in cl):.4f}, "
            f"{max(x['posterior_std_max_over_restarts_dims'] for x in cl):.4f}]; restarts with an active unit "
            f"{sum(x['n_restarts_with_any_active_unit'] for x in cl)}/200; selected sample sd / RMSE "
            f"{[round(x['selected_sd_over_rmse'], 3) for x in cl]}")
    # independent recomputation for Track B's selected models: stored metrics vs fresh evaluation of the saved model
    with open(OUT + "/cvae_selection.json") as f:
        selB = json.load(f)["clusters"]
    rsB = pd.read_csv(OUT + "/cvae_restarts.csv")
    for c in CLUSTERS:
        m = ct.load_model(selected_path(c))
        s = int(selB[str(c)]["selected_seed"])
        with open(json_path(c, s)) as f:
            js = json.load(f)
        trc = ct.cluster_rows(trB, c)
        tec = ct.cluster_rows(teB, c, sort_by_date=True)
        lo = ct.loss_on(m, trc, seed=0)
        Y = np.array(tec[ct.COLS_DATA].to_numpy(np.float64), copy=True, order="C")
        C = np.array(tec[ct.COLS_COND].to_numpy(np.float64), copy=True, order="C")
        z0 = ct.generate(m, C, 1, eps=np.zeros((1, len(C), m.latent_dim)))[0]       # decoder at z = 0
        S = np.load(OUT + f"/cvae_samples_c{c}.npy")
        Sre = ct.generate(m, C, N_SAMPLES, eps=cvae_draws(c, len(C), m.latent_dim))
        with torch.no_grad():
            mu, ls = m.encode(torch.from_numpy(m.scale(Y)), torch.from_numpy(C))
        dsel = rsB.loc[rsB["cluster"] == c]
        res["independent_recomputation"][str(c)] = {
            "selected_seed": s, "is_argmin_of_stored_test_mse": bool(
                s == int(dsel.loc[dsel["test_mse_post"].idxmin(), "seed"])),
            "test_mse_post_stored": js["test_mse_post"], "test_mse_post_recomputed": ct.test_mse(m, tec, "posterior", 0),
            "train_kl_stored": js["train_kl_final"], "train_kl_recomputed": lo["kl"],
            "test_kl_recomputed_mean": float((0.5 * (torch.exp(2 * ls) + mu ** 2 - 1 - 2 * ls)).sum(1).mean()),
            "test_posterior_std_mean": float(torch.exp(ls).mean()),
            "test_posterior_mean_abs_max": float(mu.abs().max()),
            "samples_regenerated_bitwise": bool(np.array_equal(S, Sre)),
            "max_abs_sample_minus_decoder_at_z0_kwh": float(np.abs(S - z0[None]).max()),
            "rmse_decoder_at_z0_median_kwh": float(np.median(np.sqrt(((z0 - Y) ** 2).mean(1)))),
        }
        x = res["independent_recomputation"][str(c)]
        log(f"[sanity] c{c} selected seed {s} (argmin: {x['is_argmin_of_stored_test_mse']}): test MSE stored "
            f"{x['test_mse_post_stored']:.10f} recomputed {x['test_mse_post_recomputed']:.10f}; train KL "
            f"{x['train_kl_recomputed']:.2e}, test KL {x['test_kl_recomputed_mean']:.2e}, test posterior std "
            f"{x['test_posterior_std_mean']:.5f}; samples regenerate bitwise {x['samples_regenerated_bitwise']}; "
            f"max |sample - decoder(z=0)| {x['max_abs_sample_minus_decoder_at_z0_kwh']:.4f} kWh")
    # --- GAM dispersion (sample sd vs rms residual of f on the test days)
    for track, outdir, obs in (("A", TRACKA, obsA), ("B", OUT, obsB)):
        for c in CLUSTERS:
            S = np.load(f"{outdir}/gam_samples_c{c}.npy")
            f_te = np.load(f"{outdir}/gam_f_test_c{c}.npy")
            res["gam_dispersion"][f"track{track}_c{c}"] = {
                "mean_sample_sd": float(S.std(0, ddof=1).mean()),
                "rms_residual_y_minus_f": float(np.sqrt(((obs[c] - f_te) ** 2).mean())),
                "ratio": float(S.std(0, ddof=1).mean() / np.sqrt(((obs[c] - f_te) ** 2).mean())),
                "max_abs_sample_mean_minus_f_kwh": float(np.abs(S.mean(0) - f_te).max())}
    log("[sanity] GAM sample sd / rms residual: A " + str([round(res["gam_dispersion"][f"trackA_c{c}"]["ratio"], 3)
                                                         for c in CLUSTERS])
        + " B " + str([round(res["gam_dispersion"][f"trackB_c{c}"]["ratio"], 3) for c in CLUSTERS]))
    # --- scenario effects, both generators, both tracks (lib.effects regions + day-bootstrap CIs)
    for model in MODEL_NAMES:
        for track, outdir in (("A", TRACKA), ("B", OUT)):
            for c in CLUSTERS:
                A_ = {s_: np.load(f"{outdir}/{model}_scen_{s_}_c{c}.npy") for s_ in SCENARIOS}
                e = ef.scenario_effects(A_["Normal"], {"Low": A_["Low"], "High": A_["High"]}, B=2000, seed=1000 + c)
                res["scenario_effects"][f"{model}_track{track}_c{c}"] = _effects_compact(e)
    for model in MODEL_NAMES:
        for c in CLUSTERS:
            parts = []
            for track in ("A", "B"):
                e = res["scenario_effects"][f"{model}_track{track}_c{c}"]
                parts.append(f"{track}: Low in {e['Low']['in_window']['pct']:+.1f}% High in "
                             f"{e['High']['in_window']['pct']:+.1f}% pre2 {e['High']['pre2']['pct']:+.1f}% post2 "
                             f"{e['High']['post2']['pct']:+.1f}% Low reb{ef.KEY_LOW_REBOUND[8:].replace('_', '-')} "
                             f"{e['Low'][ef.KEY_LOW_REBOUND]['pct']:+.1f}% "
                             f"High aft24-34 {e['High']['afternoon_24_34']['pct']:+.1f}%")
            log(f"[sanity] effects {MODEL_LABEL[model]} c{c}: " + " | ".join(parts))
    # --- what the Fig. 6 replica shows (test day 1, cluster matched to the authors' 3) and how typical that day is
    c = FIG_CLUSTER
    pre3 = REGIONS["High"][K_HIGH_BEF]                       # 36-38
    post3 = REGIONS["High"][K_HIGH_AFT]                      # 45-47
    day_regions = (("Low", K_LOW_IN, LOW_SLOTS), ("Low", "evening_32_47", list(range(32, 48))),
                   ("High", K_HIGH_IN, HIGH_SLOTS), ("High", f"pre_{pre3[0]}_{pre3[-1]}", pre3),
                   ("High", f"post_{post3[0]}_{post3[-1]}", post3), ("High", "afternoon_24_34", list(range(24, 35))))
    res["fig6_day1"] = {"definition": "per test day: 100 x (scenario - Normal) / Normal of the 200-sample means, "
                                      "averaged over the region's slots; day1 = 2013-01-01 (the day of the Fig. 6 "
                                      "replica); quantiles and sign over the 92 test days",
                        "cluster": c}
    for model in MODEL_NAMES:
        for track, outdir in (("A", TRACKA), ("B", OUT)):
            M = {s_: np.load(f"{outdir}/{model}_scen_{s_}_c{c}.npy").mean(0) for s_ in SCENARIOS}
            rec = {}
            for name, rname, slots in day_regions:
                p = 100 * (M[name] - M["Normal"])[:, slots].mean(1) / M["Normal"][:, slots].mean(1)
                rec[f"{name}:{rname}"] = {"day1_pct": float(p[0]), "per_day_median_pct": float(np.median(p)),
                                          "per_day_q10_q90_pct": [float(np.quantile(p, 0.1)), float(np.quantile(p, 0.9))],
                                          "frac_days_negative": float((p < 0).mean())}
            res["fig6_day1"][f"{model}_track{track}"] = rec
            log(f"[sanity] Fig. 6 day 1, {MODEL_LABEL[model]} track {track} c{c}: " + ", ".join(
                f"{k} {v['day1_pct']:+.1f}% (median day {v['per_day_median_pct']:+.1f}%, <0 on "
                f"{100 * v['frac_days_negative']:.0f}% of days)" for k, v in rec.items()))
    # --- restart-level robustness of the CVAE effects (phase-1 regions), both tracks
    res["cvae_effects_all_restarts"] = {}
    for track, outdir in (("A", TRACKA), ("B", OUT)):
        p = outdir + "/cvae_scenario_effects_all_restarts.json"
        if os.path.exists(p):
            with open(p) as f:
                res["cvae_effects_all_restarts"][f"track{track}"] = json.load(f)["clusters"]
    jdump(res, OUT + "/cvae_gam_sanity_vs_trackA.json")
    log("[sanity] wrote cvae_gam_sanity_vs_trackA.json")
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


# ---- CVAE figures (Track A's figure code, Track B data)
def cvae_fig5_replica(plt, c: int = FIG_CLUSTER) -> str:
    sizes = cluster_sizes()
    blk = test_block(load(TEST_PATH), c)
    S = np.load(OUT + f"/cvae_samples_c{c}.npy")
    Y = blk[ct.COLS_DATA].to_numpy(float)
    x = np.arange(144) / 2.0
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    for k in range(20):
        ax.plot(x, S[k, :3].reshape(-1), color=BLUE, lw=0.6, alpha=0.45, label="20 generated profiles" if k == 0 else None)
    ax.plot(x, S[:, :3].mean(0).reshape(-1), color=BLUE, lw=1.6, label="Mean of 200 generated profiles")
    ax.plot(x, Y[:3].reshape(-1), color=INK, lw=1.1, label="Observed")
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
    ax.set_title(f"Track B CVAE, cluster {c} ({sizes[c]} households; matched to the authors' cluster {c}): first "
                 f"three test days\nReplica of paper Fig. 5, left ({FIG5_NOTE.get(c, 'not the paper figure cluster')})",
                 pad=16)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=3)
    path = FIGS + f"/trackB_cvae_fig5_cluster{c}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def cvae_fig6_replica(plt, c: int = FIG_CLUSTER, day: int = 0) -> str:
    sizes = cluster_sizes()
    date = test_block(load(TEST_PATH), c)["date"].iloc[day]
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
        ax.plot(x, S.mean(0), color=SCEN_COLOR[name], lw=1.6, label=f"{name}: mean of 200")
    _headroom(ax)
    ax.set_xticks(np.arange(0, 25, 2), [f"{h:02d}:00" for h in range(0, 25, 2)])
    ax.set_xlim(0, 23.5)
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel("kWh per household per half-hour")
    ax.set_title(f"Track B CVAE, cluster {c} ({sizes[c]} households), {date:%a %d %b %Y}: tariff scenarios "
                 f"(replica of paper Fig. 6, left)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.14), ncol=2)
    path = FIGS + f"/trackB_cvae_fig6_cluster{c}_day1.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def cvae_fig8_replica(plt) -> str:
    sizes = cluster_sizes()
    dates = test_block(load(TEST_PATH), 0)["date"]
    x = np.arange(144) / 2.0
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 10), sharex=True)
    for c, ax in zip(CLUSTERS, axes):
        _shade_windows(ax, day_offsets=(0, 24, 48), label=False)   # windows named in the legend (no clash with curves)
        for name in SCENARIOS:
            M = np.load(OUT + f"/cvae_scen_{name}_c{c}.npy")[:, :3].mean(0).reshape(-1)
            ax.plot(x, M, color=SCEN_COLOR[name], lw=1.4, label=SCEN_LABEL[name])
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
    fig.suptitle("Track B CVAE: mean of 200 generated profiles per tariff scenario (replica of paper Fig. 8, left)",
                 x=0.01, ha="left", y=1.0, fontsize=10.5)
    fig.tight_layout()
    path = FIGS + "/trackB_cvae_fig8_scenario_means.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def cvae_fig_effects(plt) -> str:
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
            ax.plot(x, e["diff_mean_kwh"], color=SCEN_COLOR[name], lw=1.6, label=f"{name} minus Normal")
        _headroom(ax)
        ax.set_title(f"Cluster {c} ({sizes[c]} households)")
        ax.set_xticks(np.arange(0, 25, 4), [f"{h:02d}:00" for h in range(0, 25, 4)])
        ax.set_xlim(0, 23.5)
    for ax in axes[1]:
        ax.set_xlabel("Time of day, GMT (point at t = half-hour ending at t)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Scenario minus Normal (kWh)")
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.10), ncol=2)
    fig.suptitle("Track B CVAE: tariff effect, mean over the 92 test days of the 200-sample mean difference "
                 "(band = interquartile range across days)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    path = FIGS + "/trackB_cvae_scenario_effects.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def cvae_fig_restarts(plt) -> str:
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
    fig.suptitle("Track B CVAE: spread across the 50 restarts per cluster", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    path = FIGS + "/trackB_cvae_restarts.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def cvae_fig_effects_restarts(plt) -> str | None:
    p = OUT + "/cvae_scenario_effects_all_restarts.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    rng = np.random.default_rng(1)
    bef = REGIONS["High"][K_HIGH_BEF]
    panels = ((f"Low:{K_LOW_IN}:pct", AQUA, f"Low minus Normal inside the Low window\n(slots {ef.WINDOW_SPAN['Low']}, "
                                            f"{ef.WINDOW_CLOCK['Low']} GMT), % of Normal"),
              (f"High:{K_HIGH_IN}:pct", ORANGE, f"High minus Normal inside the High window\n(slots "
                                                f"{ef.WINDOW_SPAN['High']}, {ef.WINDOW_CLOCK['High']} GMT), % of Normal"),
              (f"High:{K_HIGH_BEF}:pct", ORANGE, f"High minus Normal just before the window\n(slots {ef.span(bef)}, "
                                                 f"{ef.slots_clock(bef)} GMT), % of Normal"))
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
    fig.suptitle("Track B CVAE: tariff effect of every restart (mean over the 92 test days); paper claims Low > 0, "
                 "High < 0 (also just before the window)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    path = FIGS + "/trackB_cvae_scenario_effects_restarts.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- GAM figures (Track A's figure code, Track B data)
C_NORMAL, C_HIGH, C_LOW = BLUE, ORANGE, AQUA
C_SAMPLE = "#9a9893"
# GAM scenario curves are drawn from one slot before to one slot after their window (slice a:b): with common random
# numbers they equal the Normal scenario exactly everywhere else
SCEN_SEGMENTS = tuple((name, col, ef.WINDOWS[name][0] - 1, ef.WINDOWS[name][-1] + 2, f"{name} {ef.WINDOW_CLOCK[name]} GMT")
                      for name, col in (("Low", C_LOW), ("High", C_HIGH)))
GAM_TIME_AXIS_LABEL = "time of day, GMT (the point at time t is the half-hour ending at t)"


def gam_shade_windows(ax, offsets=(0.0,)):
    for off in offsets:
        for name, col in (("Low", C_LOW), ("High", C_HIGH)):
            a, b = ef.WINDOW_HOURS[name]
            ax.axvspan(off + a, off + b, color=col, alpha=0.10, lw=0)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, width=0.6)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def gam_store(c: int) -> dict:
    tec = test_block(load(TEST_PATH), c)
    return {"S": np.load(OUT + f"/gam_samples_c{c}.npy"),
            "scen": {n: np.load(OUT + f"/gam_scen_{n}_c{c}.npy") for n in SCENARIOS},
            "y_te": tec[ct.COLS_DATA].to_numpy(float), "dates": tec["date"].to_numpy()}


def plot_three_days(ax, st, n_days=3, n_lines=20, title=None, legend=True):
    S, y = st["S"][:, :n_days], st["y_te"][:n_days]
    x = np.arange(n_days * H) / 2.0  # end of each half-hour, hours since 00:00 GMT of test day 1
    for i in range(n_lines):
        ax.plot(x, S[i].ravel(), color=C_SAMPLE, lw=0.5, alpha=0.55, label="20 GAM samples" if i == 0 else None)
    ax.plot(x, S.mean(axis=0).ravel(), color=C_NORMAL, lw=1.6, label="mean of 200 samples")
    ax.plot(x, y.ravel(), color=INK, lw=1.1, label="observed")
    for d in range(1, n_days):
        ax.axvline(24 * d, color=AXIS, lw=0.8)
    ticks = np.arange(0, 24 * n_days + 1, 6)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t % 24):02d}h" for t in ticks])
    dates = pd.to_datetime(st["dates"][:n_days])
    for d, dt in enumerate(dates):
        ax.text(24 * d + 12, 1.0, dt.strftime("%a %d %b %Y"), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xlim(0, 24 * n_days)
    ax.set_xlabel(GAM_TIME_AXIS_LABEL)
    ax.set_ylabel("cluster-mean consumption (kWh per half-hour)")
    style_axes(ax)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left", pad=16)
    if legend:
        ax.legend(fontsize=8, frameon=False, loc="upper left")


def plot_scenarios_day(ax, st, day=0, n_lines=20, legend=True, title=None):
    x = np.arange(H) / 2.0  # end of each half-hour (GMT)
    gam_shade_windows(ax)
    Zn = st["scen"]["Normal"][:, day]
    for i in range(n_lines):
        ax.plot(x, Zn[i], color=C_NORMAL, lw=0.6, alpha=0.45, label="Normal all day: 20 samples" if i == 0 else None)
    ax.plot(x, Zn.mean(axis=0), color=C_NORMAL, lw=1.6)
    for name, col, a, b, lab in SCEN_SEGMENTS:
        Z = st["scen"][name][:, day, a:b]
        for i in range(n_lines):
            ax.plot(x[a:b], Z[i], color=col, lw=0.6, alpha=0.8, label=f"{lab}: 20 samples" if i == 0 else None)
        ax.plot(x[a:b], Z.mean(axis=0), color=col, lw=1.6)
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_, hi_ + 0.2 * (hi_ - lo_))      # room for the legend and window labels above the curves
    top = ax.get_ylim()[1]
    for name in ("Low", "High"):
        a, b = ef.WINDOW_HOURS[name]
        ax.text((a + b) / 2, top, f"{name} tariff window", ha="center", va="top", fontsize=8, color=INK2)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xticklabels([f"{t:02d}h" for t in range(0, 25, 3)])
    ax.set_xlabel(GAM_TIME_AXIS_LABEL)
    ax.set_ylabel("cluster-mean consumption (kWh per half-hour)")
    style_axes(ax)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    if legend:
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.41, 1.0))


def plot_scenario_means(ax, st, n_days=3, title=None, legend=True):
    x = np.arange(n_days * H) / 2.0  # end of each half-hour (GMT)
    gam_shade_windows(ax, offsets=[24.0 * d for d in range(n_days)])
    ax.plot(x, st["scen"]["Normal"][:, :n_days].mean(axis=0).ravel(), color=C_NORMAL, lw=1.2, label="Normal all day")
    for name, col, a, b, lab in SCEN_SEGMENTS:
        m = st["scen"][name][:, :n_days].mean(axis=0)  # (n_days, 48)
        for d in range(n_days):
            ax.plot(x[d * H + a:d * H + b], m[d, a:b], color=col, lw=1.2, label=lab if d == 0 else None)
    for d in range(1, n_days):
        ax.axvline(24 * d, color=AXIS, lw=0.8)
    ticks = np.arange(0, 24 * n_days + 1, 6)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t % 24):02d}h" for t in ticks])
    ax.set_xlim(0, 24 * n_days)
    ax.set_xlabel(GAM_TIME_AXIS_LABEL.replace("time of day", "time of day over 3 test days"))
    ax.set_ylabel("mean of 200 samples (kWh)")
    style_axes(ax)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    if legend:
        ax.legend(fontsize=8, frameon=False, loc="upper left")


def gam_figures(plt) -> list:
    sizes = cluster_sizes()
    store = {c: gam_store(c) for c in CLUSTERS}
    c3 = FIG_CLUSTER
    paths = []
    for c5 in FIG5_CLUSTERS:
        fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150)
        plot_three_days(ax, store[c5], title=f"Track B, GAM generator (paper Fig. 5 right): cluster {c5} "
                                             f"({sizes[c5]} households), first three test days\n({FIG5_NOTE[c5]})")
        fig.tight_layout()
        paths.append(FIGS + f"/trackB_gam_fig5_cluster{c5}.png")
        fig.savefig(paths[-1])
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    d0 = pd.to_datetime(store[c3]["dates"][0]).strftime("%d %b %Y")
    plot_scenarios_day(ax, store[c3], day=0,
                       title=f"Track B, GAM generator (paper Fig. 6 right): cluster {c3}, test day 1 ({d0})\n"
                             "20 samples per tariff scenario, bold = mean of 200. Outside the shaded windows "
                             "the scenarios coincide\n(same random draws), so only Normal is drawn there.")
    fig.tight_layout()
    paths.append(FIGS + f"/trackB_gam_fig6_cluster{c3}.png")
    fig.savefig(paths[-1])
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), dpi=130)
    for c, ax in zip(CLUSTERS, axes):
        plot_three_days(ax, store[c], title=f"cluster {c} ({sizes[c]} households)", legend=(c == 0))
    fig.suptitle("Track B, GAM generator (paper Fig. 7 right): 20 samples, mean of 200, observed", fontsize=10, color=INK)
    fig.tight_layout()
    paths.append(FIGS + "/trackB_gam_fig7_allclusters.png")
    fig.savefig(paths[-1])
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), dpi=130)
    for c, ax in zip(CLUSTERS, axes):
        plot_scenario_means(ax, store[c], title=f"cluster {c} ({sizes[c]} households)", legend=(c == 0))
    fig.suptitle("Track B, GAM generator (paper Fig. 8 right): mean of 200 samples under three tariff "
                 "scenarios\n(Low/High curves drawn around their windows; elsewhere they equal Normal exactly)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    paths.append(FIGS + "/trackB_gam_fig8_allclusters.png")
    fig.savefig(paths[-1])
    plt.close(fig)
    return paths


# ---- Track A vs Track B comparison figures (cluster matched to the authors' cluster 3)
def _hour_ticks(ax, n_days):
    ticks = np.arange(0, 24 * n_days + 1, 6)
    ax.set_xticks(ticks, [f"{int(t % 24):02d}:00" for t in ticks])
    ax.set_xlim(0, 24 * n_days)


def fig_compare_fig5(plt, c: int = FIG_CLUSTER) -> str:
    """Paper Fig. 5 layout (CVAE left, GAM right) for Track A's cluster c (top row) and Track B's (bottom row)."""
    obsA, datesA = obs_blocks(A_TEST_PATH)
    obsB, _ = obs_blocks(TEST_PATH)
    sizes = cluster_sizes()
    x = np.arange(144) / 2.0          # end of each half-hour (GMT)
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.8), sharey=True)
    rows = (("Track A (authors' data)", TRACKA, obsA, A_SIZES[c]), ("Track B (our data)", OUT, obsB, sizes[c]))
    for i, (tname, outdir, obs, n) in enumerate(rows):
        for j, model in enumerate(MODEL_NAMES):
            ax = axes[i, j]
            S = np.load(f"{outdir}/{model}_samples_c{c}.npy")[:, :3]
            for k in range(20):
                ax.plot(x, S[k].reshape(-1), color=MODEL_COLOR[model], lw=0.5, alpha=0.35,
                        label="20 generated profiles" if k == 0 else None)
            ax.plot(x, S.mean(0).reshape(-1), color=MODEL_COLOR[model], lw=1.5, label="mean of 200 profiles")
            ax.plot(x, obs[c][:3].reshape(-1), color=INK, lw=1.0, label="observed")
            for b in (24, 48):
                ax.axvline(b, color=AXIS, lw=0.8)
            _hour_ticks(ax, 3)
            ax.set_title(f"{tname}: {MODEL_LABEL[model]}, cluster {c} ({n} households)", fontsize=9.5)
            if j == 0:
                ax.set_ylabel("kWh per household per half-hour")
            if i == 1:
                ax.set_xlabel(f"Time of day, first three test days ({datesA.iloc[0]:%d %b} - {datesA.iloc[2]:%d %b %Y})\n"
                              f"GMT; the point at time t is the half-hour ending at t")
            if i == 0:
                ax.legend(loc="upper left", ncol=3, fontsize=7.5)
    fig.suptitle(f"Replica of paper Fig. 5 (CVAE left, GAM right) for cluster {c} ({FIG5_NOTE.get(c, '')}): "
                 "Track A (top) vs Track B (bottom)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    path = FIGS + f"/trackB_cvae_gam_fig5_cluster{c}_vs_trackA.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_compare_fig6(plt, c: int = FIG_CLUSTER, day: int = 0) -> str:
    """Paper Fig. 6 layout (CVAE left, GAM right), test day 1, Track A (top) vs Track B (bottom). CVAE: the decoder
    sees the whole tariff vector, so the three scenarios differ at every slot and are drawn all day. GAM: with common
    random numbers Low/High equal Normal exactly outside their windows, so (as in Track A's GAM figure) they are drawn
    only from one slot before to one slot after their window; elsewhere the Normal curves are the Low/High ones."""
    _, datesA = obs_blocks(A_TEST_PATH)
    sizes = cluster_sizes()
    x = np.arange(48) / 2.0           # end of each half-hour (GMT)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharey=True)
    rows = (("Track A (authors' data)", TRACKA, A_SIZES[c]), ("Track B (our data)", OUT, sizes[c]))
    for i, (tname, outdir, n) in enumerate(rows):
        for j, model in enumerate(MODEL_NAMES):
            ax = axes[i, j]
            _shade_windows(ax, label=(i == 0))
            for name in SCENARIOS:
                S = np.load(f"{outdir}/{model}_scen_{name}_c{c}.npy")[:, day]
                a, b = (0, 48)
                if model == "gam" and name != "Normal":
                    a, b = [(a_, b_) for nm, _, a_, b_, _ in SCEN_SEGMENTS if nm == name][0]
                for k in range(20):
                    ax.plot(x[a:b], S[k, a:b], color=SCEN_COLOR[name], lw=0.5, alpha=0.35)
                ax.plot(x[a:b], S.mean(0)[a:b], color=SCEN_COLOR[name], lw=1.5, label=SCEN_LABEL[name])
            ax.set_xticks(np.arange(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
            ax.set_xlim(0, 24)
            extra = "; Low/High drawn around their windows only" if model == "gam" else ""
            ax.set_title(f"{tname}: {MODEL_LABEL[model]}, cluster {c} ({n} households){extra}", fontsize=9.5)
            if j == 0:
                ax.set_ylabel("kWh per household per half-hour")
            if i == 1:
                ax.set_xlabel(f"Time of day, GMT (point at t = half-hour ending at t), test day 1 "
                              f"({datesA.iloc[day]:%a %d %b %Y})")
    _headroom(axes[0, 0], 0.08)                   # shared y: once, for the window labels of the top row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Replica of paper Fig. 6: 20 profiles (thin) and the mean of 200 (bold) per tariff scenario, "
                 "same random draws across scenarios", x=0.01, ha="left", fontsize=10.5, y=0.995)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = FIGS + f"/trackB_cvae_gam_fig6_cluster{c}_vs_trackA.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def cvae_fig_effect_profiles_restarts(plt, c: int = FIG_CLUSTER) -> str:
    """Scenario minus Normal per slot (mean over the 92 test days) for each of the 50 restarts (thin) and the selected
    one (bold), Track A (left) vs Track B (right); the GAM's effect (dashed) for reference. Same z draws as the
    generate stage, so the selected curves equal the ones in the effects figures."""
    sizes = cluster_sizes()
    x = np.arange(48) / 2.0           # end of each half-hour (GMT)
    with open(TRACKA + "/cvae_selection.json") as f:
        selA = json.load(f)["clusters"]
    with open(OUT + "/cvae_selection.json") as f:
        selB = json.load(f)["clusters"]
    tracks = (("A", "Track A (authors' data)", load(A_TEST_PATH), lambda s: f"{TRACKA}/models/cvae_c{c}_s{s:02d}.pt",
               int(selA[str(c)]["selected_seed"]), TRACKA, A_SIZES[c]),
              ("B", "Track B (our data)", load(TEST_PATH), lambda s: model_path(c, s), int(selB[str(c)]["selected_seed"]),
               OUT, sizes[c]))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), sharex=True, sharey="row")
    for j, (tr_, tname, te, mp, ssel, outdir, n) in enumerate(tracks):
        C = test_block(te, c)[ct.COLS_COND].to_numpy(dtype=np.float64)
        eps = cvae_draws(c, len(C), ct.LATENT_DIM)
        curves = {"Low": [], "High": []}
        for s in range(N_RESTARTS):
            m = ct.load_model(mp(s))
            M = {nm: ct.generate(m, scenario_cond(C, ct.COLS_COND, nm), N_SAMPLES, eps=eps).mean(0) for nm in SCENARIOS}
            for nm in ("Low", "High"):
                curves[nm].append(((M[nm] - M["Normal"]).mean(0), s == ssel))
        for i, nm in enumerate(("Low", "High")):
            ax = axes[i, j]
            _shade_windows(ax, label=False)
            ax.axhline(0.0, color=AXIS, lw=1.0)
            thin = [d for d, is_sel in curves[nm] if not is_sel]
            for k, d in enumerate(thin):
                ax.plot(x, d, color=SCEN_COLOR[nm], lw=0.5, alpha=0.3,
                        label=f"CVAE, each of the other {len(thin)} restarts" if k == 0 else None)
            d_sel = [d for d, is_sel in curves[nm] if is_sel][0]
            ax.plot(x, d_sel, color=INK, lw=1.5, label="CVAE, selected restart (lowest test MSE)")
            G = (np.load(f"{outdir}/gam_scen_{nm}_c{c}.npy").mean(0)
                 - np.load(f"{outdir}/gam_scen_Normal_c{c}.npy").mean(0)).mean(0)
            ax.plot(x, G, color=INK2, lw=1.1, ls=(0, (4, 2)), label="GAM")
            ax.set_title(f"{tname}, {n} households: {nm} scenario", fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"{nm} minus Normal (kWh)")
            if i == 1:
                ax.set_xlabel("Time of day, GMT (point at t = half-hour ending at t)")
            ax.set_xticks(np.arange(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
            ax.set_xlim(0, 24)
            ax.legend(loc="upper right" if nm == "Low" else "upper left", fontsize=7.5)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=AQUA, alpha=0.25, lw=0), Patch(color=ORANGE, alpha=0.25, lw=0)],
               [f"Low window of the Low scenario ({ef.WINDOW_CLOCK['Low']} GMT, slots {ef.WINDOW_SPAN['Low']})",
                f"High window of the High scenario ({ef.WINDOW_CLOCK['High']} GMT, slots {ef.WINDOW_SPAN['High']})"],
               loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2, frameon=False, fontsize=8)
    fig.suptitle(f"Tariff effect of every CVAE restart, cluster {c}: mean over the 92 test days of the 200-sample mean "
                 "difference (same random draws for all restarts)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = FIGS + f"/trackB_cvae_effect_profiles_restarts_cluster{c}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_compare_effects(plt, c: int = FIG_CLUSTER) -> str:
    """Scenario minus Normal, mean over the 92 test days, per slot: both generators, both tracks."""
    sizes = cluster_sizes()
    x = np.arange(48) / 2.0           # end of each half-hour (GMT)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
    for ax, name in zip(axes, ("Low", "High")):
        _shade_windows(ax, label=True)
        ax.axhline(0.0, color=AXIS, lw=1.0)
        for model in MODEL_NAMES:
            for track, outdir, ls, n in (("A", TRACKA, (0, (4, 2)), A_SIZES[c]), ("B", OUT, "-", sizes[c])):
                Nm = np.load(f"{outdir}/{model}_scen_Normal_c{c}.npy").mean(0)
                D = (np.load(f"{outdir}/{model}_scen_{name}_c{c}.npy").mean(0) - Nm).mean(0)
                ax.plot(x, D, color=MODEL_COLOR[model], lw=1.4, ls=ls,
                        label=f"{MODEL_LABEL[model]}, Track {track} ({n} households)")
        ax.set_xticks(np.arange(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
        ax.set_xlim(0, 24)
        ax.set_xlabel("Time of day, GMT (point at t = half-hour ending at t)")
        ax.set_title(f"{name} scenario minus Normal ({SCEN_LABEL[name]})", fontsize=9.5)
    _headroom(axes[0], 0.08)                      # shared y: once, for the window labels
    axes[0].set_ylabel("kWh per household per half-hour")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, -0.42), ncol=2)
    fig.suptitle(f"Tariff effect for cluster {c} (matched to the authors' cluster {c}): mean over the 92 test days of "
                 "the 200-sample mean difference", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    path = FIGS + f"/trackB_cvae_gam_effects_cluster{c}_vs_trackA.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_rmse_vs_trackA(plt) -> str | None:
    p = OUT + "/cvae_gam_sanity_vs_trackA.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        r = json.load(f)["rmse"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, model in zip(axes, MODEL_NAMES):
        for k, (track, off) in enumerate((("A", -0.12), ("B", 0.12))):
            med = np.array([r[f"{model}_c{c}"][track]["rmse_rootmean"]["median"] for c in CLUSTERS])
            q25 = np.array([r[f"{model}_c{c}"][track]["rmse_rootmean"]["q25"] for c in CLUSTERS])
            q75 = np.array([r[f"{model}_c{c}"][track]["rmse_rootmean"]["q75"] for c in CLUSTERS])
            xs = np.array(CLUSTERS) + off
            if track == "A":
                ax.errorbar(xs, med, yerr=[med - q25, q75 - med], fmt="o", ms=7, mfc=SURFACE, mec=INK2, ecolor=INK2,
                            elinewidth=1.0, capsize=0, label="Track A (authors' data): median, IQR over test days")
            else:
                ax.errorbar(xs, med, yerr=[med - q25, q75 - med], fmt="o", ms=7, color=MODEL_COLOR[model],
                            mec=SURFACE, elinewidth=1.0, capsize=0, label="Track B (our data): median, IQR")
        ax.scatter(CLUSTERS, PAPER_FIG4[f"rmse_{model}"], marker="_", s=260, color=INK, linewidths=1.6, zorder=4,
                   label="Paper Fig. 4 median (read off the plot)")
        ax.set_xticks(list(CLUSTERS), [f"Cluster {c}" for c in CLUSTERS])
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{MODEL_LABEL[model]}: RMSE of the mean of 200 samples", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=7.5)
    axes[0].set_ylabel("Daily RMSE, root-mean over 48 half-hours (kWh)")
    fig.suptitle("Track B vs Track A vs paper: RMSE per test day (92 days)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    path = FIGS + "/trackB_cvae_gam_rmse_vs_trackA.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def stage_figures(args) -> list:
    os.makedirs(FIGS, exist_ok=True)
    mc = matched_cluster(FIG_CLUSTER)
    ov = mc["overlap_on_the_authors_1007"]
    log(f"[figures] Track B cluster {FIG_CLUSTER} = run-(a) cluster {mc['run_a_label']} matched to the authors' "
        f"cluster {FIG_CLUSTER} (corr of cluster-mean series {mc['corr_cluster_mean_series_raw']:.4f}; "
        f"{mc['n_households_trackB']} vs {mc['n_households_authors']} households; {ov['shared_with_authors_cluster']} "
        f"of our {ov['ours_in_1007']} members among the authors' 1,007 are in the authors' cluster {FIG_CLUSTER})")
    plt = _plt()
    paths = [cvae_fig5_replica(plt, c) for c in FIG5_CLUSTERS]
    paths += [cvae_fig6_replica(plt), cvae_fig8_replica(plt), cvae_fig_effects(plt), cvae_fig_restarts(plt)]
    extra = cvae_fig_effects_restarts(plt)
    if extra:
        paths.append(extra)
    paths += gam_figures(plt)
    plt = _plt()
    paths += [fig_compare_fig5(plt, c) for c in FIG5_CLUSTERS]
    paths += [fig_compare_fig6(plt), fig_compare_effects(plt), cvae_fig_effect_profiles_restarts(plt)]
    extra = fig_rmse_vs_trackA(plt)
    if extra:
        paths.append(extra)
    for p in paths:
        log(f"[figures] wrote {p}")
    return paths


# ------------------------------------------------------------------------------------------------------------------
def stage_summary(args) -> dict:
    import torch
    out = {"track": "B (our data: data/train_ours.parquet / test_ours.parquet; run-(a) ToU clusters relabelled to the "
                    "authors' 0-3; ERA5 temperatures; Tariffs.xlsx 'loc' tariffs)",
           "torch": torch.__version__, "python": sys.version.split()[0], "n_restarts": N_RESTARTS,
           "n_samples": N_SAMPLES, "gen_seed": GEN_SEED, "gam_base_seed": GAM_BASE_SEED,
           "scenario_windows": {k: {"slots": v, "clock_gmt": ef.WINDOW_CLOCK[k]} for k, v in ef.WINDOWS.items()},
           "scenario_windows_note": "Low 9-19 / High 39-44 (REPRO_SPEC section 7); the phase-1/2 runs used 9-18 / "
                                    "39-43 (outputs/archive_windows_9-18_39-43/)",
           "cluster_sizes": cluster_sizes(), "authors_cluster_sizes": A_SIZES,
           "matched_to_authors_cluster_3": matched_cluster(FIG_CLUSTER)}
    import hashlib
    out["input_sha256"] = {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()
                           for p in (TRAIN_PATH, TEST_PATH, POWER_PATH)}
    for name, fn in (("selection", "cvae_selection.json"), ("pca", "cvae_pca_consumption.json"),
                     ("effects", "cvae_scenario_effects.json"), ("walltime", "cvae_train_walltime.json"),
                     ("effects_all_restarts", "cvae_scenario_effects_all_restarts.json"),
                     ("mirror_check", "cvae_gam_mirror_check_trackA.json"),
                     ("verify_retrain", "cvae_verify_retrain.json")):
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
    p = OUT + "/cvae_gam_sanity_vs_trackA.json"
    if os.path.exists(p):
        with open(p) as f:
            s = json.load(f)
        out["sanity_vs_trackA"] = {k: s[k] for k in ("rmse", "collapse", "independent_recomputation", "observed")}
    out["deviations_from_upstream"] = [
        "PyTorch float64 instead of TF2/Keras float64 (same layers, loss, initialiser family, Keras-Adam update rule).",
        "Each restart uses its own seed (seeds 0..49) for the Glorot init and the reparameterisation noise; upstream "
        "reuses one glorot_uniform(seed=0) object and unseeded tf.random.normal.",
        "Fresh Adam per restart; upstream shares ONE Adam instance across all restarts and clusters.",
        "Selection metric evaluated with a fixed RNG (seed 0, common to all restarts); upstream draws are unseeded.",
        "Generation: z drawn with numpy (default_rng([20200801, cluster])), same z for real-condition samples and the "
        "three tariff scenarios (common random numbers).",
    ]
    out["trackB_data_differences_vs_trackA"] = [
        "households: our 1,032 ToU households (the authors' 1,007 + 25), clustered by our run-(a) pipeline; ARI 0.58 "
        "with the authors' partition; sizes 285/299/299/149 vs 325/295/229/158",
        "temperatures: ERA5 (central London, bilinear) instead of NOAA; derived conditions r >= 0.915 with the authors'",
        "tariffs: Tariffs.xlsx in the physically right alignment ('loc'); differs from the authors' schedule on 17 days",
        "same 273/92 train/test dates as the authors'",
    ]
    out["run_notes"] = [
        "The 200 restarts were trained in two invocations of the train stage: 143 (seeds 0-35 of clusters 0-2, 0-34 of "
        "cluster 3) by a first process that stopped before finishing, then the remaining 57 by a second one (finished "
        "restarts are skipped; unfinished ones leave no record and are trained afresh). Three restarts of the first "
        "invocation were in progress while the machine was suspended and record ~4,640 s of wall time as "
        "runtime_train_s; use the per-cluster runtime_train_s_median. The verify stage retrains the four selected "
        "restarts and those three from scratch and finds them bitwise identical (cvae_verify_retrain.json).",
    ]
    jdump(out, OUT + "/cvae_summary.json")
    log("[summary] wrote cvae_summary.json")
    return out


STAGES = ("mirror", "pca", "train", "collect", "select", "verify", "generate", "effects", "robust", "gam", "sanity",
          "figures", "summary")
EXTRA_STAGES = ("mirror_regen", "gamscen")      # run only when named; they never train or fit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", help="all or a comma list of: " + ",".join(STAGES + EXTRA_STAGES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--restarts", type=int, default=N_RESTARTS)
    ap.add_argument("--epochs", type=int, default=ct.EPOCHS)
    ap.add_argument("--clusters", type=int, nargs="+", default=list(CLUSTERS))
    ap.add_argument("--force", action="store_true", help="retrain restarts that already have results")
    ap.add_argument("--no-train", action="store_true",
                    help="refuse every stage that would train or fit (train with missing restarts, verify, mirror, gam)")
    args = ap.parse_args()
    if args.workers > 4:
        raise SystemExit("max 4 workers (shared machine)")
    if args.no_train and args.force:
        raise SystemExit("--no-train and --force are incompatible")
    stages = STAGES if args.stage == "all" else tuple(s.strip() for s in args.stage.split(","))
    bad = [s for s in stages if s not in STAGES + EXTRA_STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {STAGES + EXTRA_STAGES}")
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    fns = {"mirror": stage_mirror, "pca": stage_pca, "train": stage_train, "collect": stage_collect,
           "select": stage_select, "verify": stage_verify, "generate": stage_generate, "effects": stage_effects,
           "robust": stage_robust, "gam": stage_gam, "sanity": stage_sanity, "figures": stage_figures,
           "summary": stage_summary, "mirror_regen": stage_mirror_regen, "gamscen": stage_gamscen}
    _LOG_FILES[:] = [CVAE_LOG]
    log(f"=== 30_trackB_models.py stages={','.join(stages)} workers={args.workers} restarts={args.restarts} "
        f"epochs={args.epochs} clusters={args.clusters} no_train={args.no_train}; scenario windows Low "
        f"{ef.WINDOW_TEXT['Low']}, High {ef.WINDOW_TEXT['High']}")
    for s in stages:
        _LOG_FILES[:] = [GAM_LOG] if s in ("gam", "gamscen") else [CVAE_LOG]
        t0 = time.perf_counter()
        fns[s](args)
        log(f"--- stage {s} done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
