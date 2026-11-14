"""v-collapse independent checks (scratch; not part of the deliverable script).

1. lib.cvae_torch.train_cvae (a second, separately written implementation) retrained at NON-default eta: weights must be
   bitwise equal to scripts/40_eta_sensitivity.py's runs -> "only eta changes" holds for every eta, not just eta = 10.
2. Pure-numpy forward pass (no torch maths) on the stored selected models: regenerate the stored Track A/B sample arrays
   with z = default_rng([20200801, c]).standard_normal((200, 92, 4)) (proves the z draws are mirrored) and recompute the
   collapse statistics and calibration numbers of C_cvae_low_variance / C_trackB_collapse with fresh score code.
3. Re-score every eta model from its saved weights with the numpy forward pass and differently written score code;
   compare with eta_sensitivity.csv.
4. Mechanism diagnostics per eta model: dead encoder units, decoder at z = 0 vs sample mean RMSE, R^2 of the active
   posterior means on the conditions, spread/skill and coverage if the Gaussian decoder noise implied by the loss
   (sigma^2 = eta / 2 on the [0, 1] scale) were added.
5. Linear-VAE collapse threshold: eigenvalues lambda_k of the training residual second-moment matrix (scaled units)
   around the collapsed eta = 10 decoder; a unit can be active at the optimum only if eta < 2 lambda_k.
Usage: python -P indep_checks.py [--part 1,2,3,4,5]
"""
import argparse
import json
import math
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
UP = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo/data"
SCR = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-collapse"
PATHS = {"A": (UP + "/train_nmf.feather", UP + "/test_nmf.feather"),
         "B": (REPRO + "/data/train_ours.parquet", REPRO + "/data/test_ours.parquet")}
STORED = {"A": REPRO + "/outputs/trackA", "B": REPRO + "/outputs/trackB"}
RUNS = REPRO + "/outputs/verify/v-collapse/runs"
CSV = REPRO + "/outputs/verify/v-collapse/eta_sensitivity.csv"
OUTJ = SCR + "/indep_checks.json"
CD = [f"consumption_{h:02d}" for h in range(48)]
CC = ["temp_smooth_n", "pca_temp_0", "pca_temp_1", "pca_temp_2", "pos_y", "work_day"] + \
     [n for h in range(48) for n in (f"low_{h:02d}", f"high_{h:02d}")]
ETAS = [10.0, 1.0, 0.1, 0.03, 0.01, 0.003, 0.001]


def tag(e):
    return f"{e:g}"


_T = {}


def data(track, c):
    if (track, c) in _T:
        return _T[(track, c)]
    tr = pd.read_parquet(PATHS[track][0]) if PATHS[track][0].endswith("parquet") else pd.read_feather(PATHS[track][0])
    te = pd.read_parquet(PATHS[track][1]) if PATHS[track][1].endswith("parquet") else pd.read_feather(PATHS[track][1])
    tr["date"] = pd.to_datetime(tr["date"])
    te["date"] = pd.to_datetime(te["date"])
    trc = tr.loc[tr["cluster"].astype(int) == c]
    tec = te.loc[te["cluster"].astype(int) == c].sort_values("date", kind="stable")
    Ytr = trc[CD].to_numpy(dtype=np.float64)
    Yte = tec[CD].to_numpy(dtype=np.float64)
    lo, hi = Ytr.min(), Ytr.max()
    d = dict(Ytr=Ytr, Yte=Yte, Ctr=trc[CC].to_numpy(dtype=np.float64), Cte=tec[CC].to_numpy(dtype=np.float64),
             lo=lo, hi=hi, Xtr=(Ytr - lo) / (hi - lo), Xte=(Yte - lo) / (hi - lo))
    _T[(track, c)] = d
    return d


def params(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    st = sd["state_dict"]
    return {k: v.detach().numpy().astype(np.float64) for k, v in st.items()}, sd.get("meta", {})


def enc(P, X, C):
    h = np.maximum(0.0, np.hstack([X, C]) @ P["enc_hidden.weight"].T + P["enc_hidden.bias"])
    o = h @ P["enc_out.weight"].T + P["enc_out.bias"]
    return o[:, :4], o[:, 4:], h


def dec(P, Z, C):
    h = np.maximum(0.0, np.hstack([Z, C]) @ P["dec_hidden.weight"].T + P["dec_hidden.bias"])
    return h @ P["dec_out.weight"].T + P["dec_out.bias"]


def gen(P, C, Z, lo, hi):
    N, D, k = Z.shape
    return dec(P, Z.reshape(-1, k), np.tile(C, (N, 1))).reshape(N, D, 48) * (hi - lo) + lo


def zdraw(c, D=92):
    return np.random.default_rng([20200801, c]).standard_normal((200, D, 4))


def scores(S, y):
    """Written differently from the deliverable: loops over days, unordered pairs x 2, np.percentile, np.linalg.norm."""
    N, D, H = S.shape
    m = S.mean(0)
    rmse = np.sqrt(np.mean((m - y) ** 2, axis=1))
    half = N // 2
    iu, ju = np.triu_indices(H, 1)
    es = np.empty(D)
    vg = np.empty(D)
    for t in range(D):
        A, B = S[:half, t], S[half:2 * half, t]
        es[t] = 2.0 / N * np.linalg.norm(A - y[t], axis=1).sum() - 1.0 / N * np.linalg.norm(A - B, axis=1).sum()
        dy = np.sqrt(np.abs(y[t, iu] - y[t, ju]))
        ds = np.sqrt(np.abs(S[:, t, iu] - S[:, t, ju])).mean(0)
        vg[t] = 2.0 * np.sum((dy - ds) ** 2)
    var = S.var(0, ddof=1)
    ss = math.sqrt(var.mean() * (N + 1) / N) / math.sqrt(np.mean((m - y) ** 2))
    p5, p25, p75, p95 = np.percentile(S, [5, 25, 75, 95], axis=0)
    below = (S < y[None]).sum(0)
    pit = (below + 0.5) / (N + 1)
    return {"rmse_median_kwh": float(np.median(rmse)), "es_median": float(np.median(es)),
            "vg_median": float(np.median(vg)), "spread_skill": ss,
            "coverage90": float(np.mean((y >= p5) & (y <= p95))), "coverage50": float(np.mean((y >= p25) & (y <= p75))),
            "frac_outside_range": float(np.mean((y < S.min(0)) | (y > S.max(0)))),
            "pit_var": float(pit.var()), "sample_sd_kwh_mean": float(np.sqrt(var).mean()),
            "rmse_pooled_kwh": float(math.sqrt(np.mean((m - y) ** 2)))}


def latent(P, X, C):
    mu, ls, h = enc(P, X, C)
    sd = np.exp(ls)
    kl = 0.5 * (sd ** 2 + mu ** 2 - 1.0 - 2.0 * ls)
    vmu = mu.var(0, ddof=1)
    return {"kl": float(kl.sum(1).mean()), "active": int((vmu > 0.01).sum()), "var_mu": vmu,
            "dead_enc_units": int((h.max(0) <= 0.0).sum()), "post_sd_min": float(sd.min()),
            "post_sd_max": float(sd.max()), "mu": mu, "sd": sd}


# ---------------------------------------------------------------------------------------------------------------------
def part1():
    sys.path.insert(0, REPRO)
    from lib import cvae_torch as ct   # second implementation (allowed: the task names this API for mirroring)
    out = []
    for track, c, eta, s in (("A", 3, 0.01, 0), ("B", 0, 0.1, 2), ("A", 1, 1.0, 4), ("B", 3, 0.001, 1)):
        t0 = time.perf_counter()
        tr = ct._read_table(PATHS[track][0])
        model, info = ct.train_cvae(ct.cluster_rows(tr, c), seed=s, eta=eta)
        P, meta = params(f"{RUNS}/{track}_c{c}_eta{tag(eta)}_s{s}.pt")
        names = ["enc_hidden.weight", "enc_hidden.bias", "enc_out.weight", "enc_out.bias", "dec_hidden.weight",
                 "dec_hidden.bias", "dec_out.weight", "dec_out.bias"]
        diff = max(float(np.abs(p.detach().numpy() - P[n]).max()) for p, n in zip(model.params_list(), names))
        js = json.load(open(f"{RUNS}/{track}_c{c}_eta{tag(eta)}_s{s}.json"))
        r = {"track": track, "cluster": c, "eta": eta, "seed": s, "max_abs_weight_diff_lib_vs_script": diff,
             "epoch_loss_identical": bool(info["epoch_loss"] == js["epoch_loss"]), "s": time.perf_counter() - t0}
        print("[part1]", r, flush=True)
        out.append(r)
    return out


def part2():
    out = {}
    for track in ("A", "B"):
        for c in range(4):
            d = data(track, c)
            P, meta = params(f"{STORED[track]}/models/cvae_selected_c{c}.pt")
            Z = zdraw(c)
            S = gen(P, d["Cte"], Z, d["lo"], d["hi"])
            Sst = np.load(f"{STORED[track]}/cvae_samples_c{c}.npy")
            f0 = gen(P, d["Cte"], np.zeros((1, 92, 4)), d["lo"], d["hi"])[0]
            shift = []
            for k in range(4):
                e = np.zeros((1, 92, 4))
                e[0, :, k] = 1.0
                shift.append(float(np.abs(gen(P, d["Cte"], e, d["lo"], d["hi"])[0] - f0).mean() * 1000))
            ltr = latent(P, d["Xtr"], d["Ctr"])
            lte = latent(P, d["Xte"], d["Cte"])
            sc = scores(Sst, d["Yte"])
            out[f"{track}|c{c}"] = {
                "selected_seed": meta.get("seed"), "regen_vs_stored_max_abs_kwh": float(np.abs(S - Sst).max()),
                "train_kl": ltr["kl"], "test_kl": lte["kl"], "active_units": ltr["active"],
                "max_var_post_mean": float(ltr["var_mu"].max()), "dead_encoder_units_of_15": ltr["dead_enc_units"],
                "post_sd_range": [ltr["post_sd_min"], ltr["post_sd_max"]],
                "unit_z_shift_wh_per_dim": shift, "mean_abs_samples_minus_f0_wh": float(np.abs(Sst - f0[None]).mean() * 1000),
                **sc}
            print("[part2]", f"{track}|c{c}", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out[f"{track}|c{c}"].items()}, flush=True)
    return out


def part3_4():
    df = pd.read_csv(CSV)
    rows = []
    for _, r in df.iterrows():
        track, c, eta, s = r["track"], int(r["cluster"]), float(r["eta"]), int(r["seed"])
        d = data(track, c)
        P, meta = params(f"{RUNS}/{track}_c{c}_eta{tag(eta)}_s{s}.pt")
        Z = zdraw(c)
        S = gen(P, d["Cte"], Z, d["lo"], d["hi"])
        sc = scores(S, d["Yte"])
        lt = latent(P, d["Xtr"], d["Ctr"])
        f0 = gen(P, d["Cte"], np.zeros((1, 92, 4)), d["lo"], d["hi"])[0]
        rm0 = float(np.median(np.sqrt(np.mean((f0 - d["Yte"]) ** 2, axis=1))))
        # R^2 of active posterior means on 8 condition summaries (train days)
        X8 = np.column_stack([d["Ctr"][:, :6], d["Ctr"][:, 6::2].sum(1), d["Ctr"][:, 7::2].sum(1), np.ones(len(d["Ctr"]))])
        r2 = []
        for k in np.where(lt["var_mu"] > 0.01)[0]:
            yk = lt["mu"][:, k]
            beta, *_ = np.linalg.lstsq(X8, yk, rcond=None)
            res = yk - X8 @ beta
            r2.append(float(1 - res.var() / yk.var()))
        # add the Gaussian decoder noise implied by the loss: sigma^2 = eta / 2 on the [0, 1] scale
        sig = math.sqrt(eta / 2.0) * (d["hi"] - d["lo"])
        Sn = S + sig * np.random.default_rng([4242, c, s]).standard_normal(S.shape)
        scn = scores(Sn, d["Yte"])
        row = {"track": track, "cluster": c, "eta": eta, "seed": s,
               **{f"indep_{k}": v for k, v in sc.items()},
               **{f"csv_{k}": float(r[k]) for k in ("rmse_median_kwh", "es_median", "vg_median", "spread_skill",
                                                     "coverage90", "coverage50", "frac_outside_range", "train_kl",
                                                     "active_units")},
               "indep_train_kl": lt["kl"], "indep_active": lt["active"], "dead_enc_units": lt["dead_enc_units"],
               "rmse_f0_median_kwh": rm0, "r2_active_mu_on_cond_max": max(r2) if r2 else None,
               "r2_active_mu_on_cond_min": min(r2) if r2 else None,
               "implied_noise_sd_kwh": sig, "noise_added_spread_skill": scn["spread_skill"],
               "noise_added_coverage90": scn["coverage90"], "noise_added_rmse_median_kwh": scn["rmse_median_kwh"]}
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(SCR + "/indep_rescore.csv", index=False)
    cmp = {}
    for k in ("rmse_median_kwh", "es_median", "vg_median", "spread_skill", "coverage90", "coverage50",
              "frac_outside_range", "train_kl"):
        a, b = out[f"indep_{k}"].to_numpy(), out[f"csv_{k}"].to_numpy()
        cmp[k] = {"max_abs_diff": float(np.abs(a - b).max()), "max_rel_diff": float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))}
    cmp["active_units_mismatches"] = int((out["indep_active"] != out["csv_active_units"]).sum())
    print("[part3] independent re-score vs CSV:", json.dumps(cmp), flush=True)
    g = out.groupby(["track", "eta"]).agg(
        dead_enc_min=("dead_enc_units", "min"), dead_enc_max=("dead_enc_units", "max"),
        rmse_f0=("rmse_f0_median_kwh", "median"), rmse_mean=("indep_rmse_median_kwh", "median"),
        r2max=("r2_active_mu_on_cond_max", "median"), r2min=("r2_active_mu_on_cond_min", "median"),
        noise_sd=("implied_noise_sd_kwh", "median"), ss_noise=("noise_added_spread_skill", "median"),
        cov90_noise=("noise_added_coverage90", "median"), ss=("indep_spread_skill", "median"),
        cov90=("indep_coverage90", "median"))
    print(g.to_string(), flush=True)
    return {"rescore_vs_csv": cmp, "by_track_eta": json.loads(g.reset_index().to_json(orient="records"))}


def part5():
    out = {}
    for track in ("A", "B"):
        for c in range(4):
            d = data(track, c)
            P, _ = params(f"{RUNS}/{track}_c{c}_eta10_s0.pt")
            f0 = dec(P, np.zeros((len(d["Ctr"]), 4)), d["Ctr"])
            R = d["Xtr"] - f0
            lam = np.sort(np.linalg.eigvalsh(R.T @ R / len(R)))[::-1][:6]
            Rc = R - R.mean(0)
            lamc = np.sort(np.linalg.eigvalsh(Rc.T @ Rc / len(R)))[::-1][:6]
            out[f"{track}|c{c}"] = {"lambda_top6_second_moment": lam.tolist(), "lambda_top6_cov": lamc.tolist(),
                                    "eta_threshold_2lambda_top4": (2 * lam[:4]).tolist(),
                                    "resid_rms_scaled": float(np.sqrt(np.mean(R ** 2)))}
            print("[part5]", f"{track}|c{c}", "2*lambda_1..4 =", np.round(2 * lam[:4], 4).tolist(),
                  "resid RMS (scaled)", round(float(np.sqrt(np.mean(R ** 2))), 4), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="1,2,3,4,5")
    a = ap.parse_args()
    parts = a.part.split(",")
    res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
    if "1" in parts:
        res["part1_lib_retrain"] = part1()
    if "2" in parts:
        res["part2_stored_selected"] = part2()
    if "3" in parts or "4" in parts:
        res["part3_4"] = part3_4()
    if "5" in parts:
        res["part5_threshold"] = part5()

    def js(o):
        if isinstance(o, dict):
            return {k: js(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [js(v) for v in o]
        if isinstance(o, np.ndarray):
            return js(o.tolist())
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    json.dump(js(res), open(OUTJ, "w"), indent=1)
    print("wrote", OUTJ)
