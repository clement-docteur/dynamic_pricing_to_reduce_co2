"""Chronological-split test of both generators (project proposal, section 3: chronological train / validation / test).

The paper (and Tracks A/B) score on 92 test days drawn at random from 2013, interleaved with the training days.
Here, per track and cluster:
  train = 2013-01-01 .. 08-31 (243 days), validation = 09-01 .. 10-31 (61), test = 11-01 .. 12-31 (61).

Leak-free conditions: temp_smooth_n (min-max) and the 3 temperature PCA scores (PCA fit + one global min-max) are
refitted on the TRAINING days only and applied to all days (the authors fitted them on all 365 days). temp_smooth is a
causal filter and pos_y / work_day are calendar variables, so they are kept as they are.

Models (paper settings otherwise):
  * CVAE 'paper': all 102 conditions; R restarts (seeds 0..R-1); the restart with the lowest authors' selection metric
    (compute_mse, posterior z) on the VALIDATION days is kept.
  * CVAE 'no_pos_y': same without the position-in-year input pos_y (101 conditions): an ablation of extrapolation.
  * GAM: lib.gam_generator fitted on the training days (sigma from the causality model on the same days).
  * naive: for each test day, the mean observed profile over the last 28 validation days (Oct 4-31) of the same
    day type (work day / weekend).
  * reference: the random-split models already in outputs/track{A,B} (scores_long.csv) on their random test days that
    fall in Nov-Dec (in-period accuracy, the paper's setup).

Scores per test day: root-mean RMSE of the 200-sample mean (kWh, as Fig. 4) and the bias of daily energy (%).
Tariff response on the test days (paper scenarios, Low 9-19 / High 39-44, common random numbers): in-window % effect,
for the chrono CVAE and GAM and for the random-split selected CVAE (outputs/track*/models/cvae_selected_c*.pt).

Outputs: outputs/chrono/chrono_results.json, chrono_per_day.csv, chrono_restarts.csv.
Usage: python -P scripts/50_chrono_split.py [--restarts 20] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import conditions as C  # noqa: E402
from lib import cvae_torch as ct  # noqa: E402
from lib import gam_generator as gg  # noqa: E402
from lib.effects import HIGH_WINDOW, LOW_WINDOW  # noqa: E402

OUT = REPRO / "outputs" / "chrono"
FRAMES = {"A": REPRO / "data" / "authors" / "power_consump_nmf.feather", "B": REPRO / "data" / "power_consump_ours.parquet"}
SPLIT = {"train": ("2013-01-01", "2013-08-31"), "val": ("2013-09-01", "2013-10-31"), "test": ("2013-11-01", "2013-12-31")}
H, N_SAMPLES = 48, 200
TEMP = [f"temp_{h:02d}" for h in range(H)]
Y_COLS = ct.COLS_DATA
COND_NO_POSY = [c for c in ct.COLS_COND if c != "pos_y"]
VARIANTS = {"paper": list(ct.COLS_COND), "no_pos_y": COND_NO_POSY}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load(track: str) -> pd.DataFrame:
    p = FRAMES[track]
    df = pd.read_feather(p) if p.suffix == ".feather" else pd.read_parquet(p)
    df = df.assign(date=pd.to_datetime(df["date"]), cluster=df["cluster"].astype(int))
    return df.sort_values(["cluster", "date"]).reset_index(drop=True)


def split_masks(dates: pd.Series) -> dict:
    return {k: ((dates >= a) & (dates <= b)).to_numpy() for k, (a, b) in SPLIT.items()}


def leak_free_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Refit temp_smooth_n and pca_temp_0..2 on the training days (identical temperatures in every cluster)."""
    out = []
    for c, g in df.groupby("cluster", sort=True):
        g = g.copy()
        tr = split_masks(g["date"])["train"]
        ts = g["temp_smooth"].to_numpy(float)
        g["temp_smooth_n"] = (ts - ts[tr].min()) / (ts[tr].max() - ts[tr].min())
        fit = C.pca_fit(g[TEMP].to_numpy(float), 3, fit_rows=np.where(tr)[0])
        s = fit["scores"]
        s = (s - s[tr].min()) / (s[tr].max() - s[tr].min())
        for j in range(3):
            g[f"pca_temp_{j}"] = s[:, j]
        out.append(g)
    return pd.concat(out).reset_index(drop=True)


def rmse_rootmean(S: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """(n_samples, n_days, 48), (n_days, 48) -> per-day RMSE of the sample mean (kWh)."""
    return np.sqrt(((S.mean(0) - Y) ** 2).mean(1))


def energy_bias(S: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return 100 * (S.mean(0).sum(1) / Y.sum(1) - 1)


def scenario_cond(cdf: pd.DataFrame, cols: list[str], name: str) -> np.ndarray:
    d = cdf.copy()
    for h in range(H):
        d[f"low_{h:02d}"] = 0.0
        d[f"high_{h:02d}"] = 0.0
    if name == "Low":
        for h in LOW_WINDOW:
            d[f"low_{h:02d}"] = 1.0
    if name == "High":
        for h in HIGH_WINDOW:
            d[f"high_{h:02d}"] = 1.0
    return d[cols].to_numpy(float)


def window_effect(Sn: np.ndarray, Sx: np.ndarray, slots) -> float:
    a, b = Sx.mean(0)[:, slots], Sn.mean(0)[:, slots]
    return float(100 * (a.mean() - b.mean()) / b.mean())


def train_job(job: dict) -> dict:
    ct.worker_init()
    tr = job["train"]
    model, info = ct.train_cvae(tr, seed=job["seed"], cond_cols=job["cols"])
    val_metric = ct.test_mse(model, job["val"], z="posterior", seed=0)
    path = OUT / "models" / f"{job['track']}_c{job['cluster']}_{job['variant']}_s{job['seed']:02d}.pt"
    ct.save_model(model, str(path))
    return dict(track=job["track"], cluster=job["cluster"], variant=job["variant"], seed=job["seed"],
                val_mse=val_metric, train_loss_last=info["train_loss_last_epoch"], path=str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frames = {t: leak_free_conditions(load(t)) for t in "AB"}
    for t, f in frames.items():
        m = split_masks(f[f.cluster == 0]["date"])
        log(f"track {t}: train {m['train'].sum()} / val {m['val'].sum()} / test {m['test'].sum()} days; "
            f"test PCA scores outside the training [0, 1] range: "
            f"{float(((f.loc[f.cluster == 0, [f'pca_temp_{j}' for j in range(3)]].to_numpy()[m['test']] < 0) | (f.loc[f.cluster == 0, [f'pca_temp_{j}' for j in range(3)]].to_numpy()[m['test']] > 1)).mean()):.2%}")

    # ---------------------------------------------------------------- CVAE restarts (parallel)
    jobs = []
    for t, f in frames.items():
        for c in range(4):
            g = f[f.cluster == c]
            m = split_masks(g["date"])
            for v, cols in VARIANTS.items():
                for s in range(args.restarts):
                    jobs.append(dict(track=t, cluster=c, variant=v, seed=s, cols=cols,
                                     train=g[m["train"]].reset_index(drop=True), val=g[m["val"]].reset_index(drop=True)))
    log(f"training {len(jobs)} CVAE restarts on {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers, initializer=ct.worker_init) as ex:
        restarts = list(ex.map(train_job, jobs, chunksize=1))
    rdf = pd.DataFrame(restarts)
    rdf.to_csv(OUT / "chrono_restarts.csv", index=False)
    log(f"trained in {time.time() - t0:.0f} s")

    # ---------------------------------------------------------------- evaluation on the test days
    per_day, res = [], {"split": SPLIT, "restarts": args.restarts, "tracks": {}}
    ref_scores = {t: pd.read_csv(REPRO / f"outputs/track{t}/scores_long.csv", parse_dates=["date"]) for t in "AB"}
    for t, f in frames.items():
        orig = load(t)
        res["tracks"][t] = {}
        for c in range(4):
            g = f[f.cluster == c].reset_index(drop=True)
            go = orig[orig.cluster == c].reset_index(drop=True)
            m = split_masks(g["date"])
            te, va = g[m["test"]].reset_index(drop=True), g[m["val"]].reset_index(drop=True)
            Y = te[Y_COLS].to_numpy(float)
            eps = np.random.default_rng([20200801, c]).standard_normal((N_SAMPLES, len(te), ct.LATENT_DIM))
            out, rec = {}, {}
            # CVAE variants: selected restart on validation
            for v, cols in VARIANTS.items():
                sub = rdf[(rdf.track == t) & (rdf.cluster == c) & (rdf.variant == v)]
                best = sub.loc[sub.val_mse.idxmin()]
                model = ct.load_model(best.path)
                S = {sc: ct.generate(model, scenario_cond(te, cols, sc) if sc != "Real" else te[cols].to_numpy(float),
                                     N_SAMPLES, eps=eps) for sc in ("Real", "Normal", "Low", "High")}
                rec[f"cvae_{v}"] = rmse_rootmean(S["Real"], Y)
                out[f"cvae_{v}"] = dict(
                    selected_seed=int(best.seed), rmse_median=float(np.median(rec[f"cvae_{v}"])),
                    energy_bias_median_pct=float(np.median(energy_bias(S["Real"], Y))),
                    low_effect_pct=window_effect(S["Normal"], S["Low"], LOW_WINDOW),
                    high_effect_pct=window_effect(S["Normal"], S["High"], HIGH_WINDOW),
                    # spread of the test RMSE over all restarts (median of each restart's median)
                    restart_rmse_median=float(np.median([np.median(rmse_rootmean(ct.generate(
                        ct.load_model(p), te[cols].to_numpy(float), N_SAMPLES, eps=eps), Y)) for p in sub.path])))
            # GAM fitted on the training days
            gp = gg.fit_gam_generator(g[m["train"]].reset_index(drop=True))
            rng_seed = [20200727, c, 50]
            S = {sc: gg.generate(gp, te if sc == "Real" else gg.tariff_scenario(te, sc), N_SAMPLES,
                                 np.random.default_rng(rng_seed)) for sc in ("Real", "Normal", "Low", "High")}
            rec["gam"] = rmse_rootmean(S["Real"], Y)
            out["gam"] = dict(rmse_median=float(np.median(rec["gam"])),
                              energy_bias_median_pct=float(np.median(energy_bias(S["Real"], Y))),
                              low_effect_pct=window_effect(S["Normal"], S["Low"], LOW_WINDOW),
                              high_effect_pct=window_effect(S["Normal"], S["High"], HIGH_WINDOW))
            # naive: same day type, last 28 validation days
            last = va[va["date"] >= va["date"].max() - pd.Timedelta(days=27)]
            prof = {w: last.loc[last.work_day == w, Y_COLS].to_numpy(float).mean(0) for w in (0.0, 1.0)}
            P = np.stack([prof[float(w)] for w in te.work_day])
            rec["naive"] = np.sqrt(((P - Y) ** 2).mean(1))
            out["naive"] = dict(rmse_median=float(np.median(rec["naive"])),
                                energy_bias_median_pct=float(np.median(100 * (P.sum(1) / Y.sum(1) - 1))))
            # random-split models (paper setup): scored days in Nov-Dec, and their response on the chrono test days
            rs = ref_scores[t]
            rs = rs[(rs.cluster == c) & (rs.date >= SPLIT["test"][0])]
            sel = ct.load_model(str(REPRO / f"outputs/track{t}/models/cvae_selected_c{c}.pt"))
            teo = go[split_masks(go["date"])["test"]].reset_index(drop=True)
            Sr = {sc: ct.generate(sel, scenario_cond(teo, list(ct.COLS_COND), sc), N_SAMPLES, eps=eps)
                  for sc in ("Normal", "Low", "High")}
            out["random_split_reference"] = dict(
                n_days_in_nov_dec=int((rs.model == "cvae").sum()),
                cvae_rmse_median=float(rs.loc[rs.model == "cvae", "rmse_rootmean"].median()),
                gam_rmse_median=float(rs.loc[rs.model == "gam", "rmse_rootmean"].median()),
                cvae_selected_low_effect_pct_on_test_days=window_effect(Sr["Normal"], Sr["Low"], LOW_WINDOW),
                cvae_selected_high_effect_pct_on_test_days=window_effect(Sr["Normal"], Sr["High"], HIGH_WINDOW))
            out["mean_obs_kwh"] = float(Y.mean())
            res["tracks"][t][str(c)] = out
            for k, v in rec.items():
                per_day.append(pd.DataFrame({"track": t, "cluster": c, "model": k, "date": te["date"], "rmse": v}))
            log(f"track {t} c{c}: RMSE cvae {out['cvae_paper']['rmse_median']:.4f} | no_pos_y "
                f"{out['cvae_no_pos_y']['rmse_median']:.4f} | gam {out['gam']['rmse_median']:.4f} | naive "
                f"{out['naive']['rmse_median']:.4f} | random-split ref cvae {out['random_split_reference']['cvae_rmse_median']:.4f}"
                f" gam {out['random_split_reference']['gam_rmse_median']:.4f} (n={out['random_split_reference']['n_days_in_nov_dec']})")
    pd.concat(per_day).to_csv(OUT / "chrono_per_day.csv", index=False)
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(OUT / "chrono_results.json", "w", encoding="utf-8"), indent=1)
    log(f"done in {res['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
