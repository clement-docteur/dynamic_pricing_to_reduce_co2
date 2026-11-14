"""v-code: is the GAM generator's over-dispersion (spread/skill ~1.4) an artifact of our sigma^h(p)?
Independent of lib: my own natural-cubic-spline regressions (numpy only).
 (a) my causality model on the cluster's TRAINING aggregate: per slot, Y = ns(tau^h) + xi_p, sd = gamma_p
     (iteratively re-weighted LS, gamma_p = RMS residual within tariff p, ML), with and without the 0.01 floor;
 (b) my GAM mean f: per slot, ns(tau^h) + ns(taubar) + ns(kappa) + w + 1Low + 1High on the training days;
 (c) compare sigma^h(p) (saved table outputs/trackA/gam_sigma.csv, and mine) with the RMS of TEST residuals
     around f (saved f_test and my f), and with the saved samples' per-slot SD.
"""
import os, json
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
import numpy as np, pandas as pd

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
UP = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo"
OUTD = os.path.dirname(os.path.abspath(__file__))
H = 48
cd = lambda p: [f"{p}_{h:02d}" for h in range(H)]

def ns_basis(x, knots):
    """Natural cubic spline basis (ESL 5.4), without intercept: [x, N_3..N_K]."""
    K = len(knots)
    def d(k):
        return (np.clip(x - knots[k], 0, None) ** 3 - np.clip(x - knots[K - 1], 0, None) ** 3) / (knots[K - 1] - knots[k])
    cols = [x] + [d(k) - d(K - 2) for k in range(K - 2)]
    return np.column_stack(cols)

def knots_q(x, K):
    return np.quantile(x, np.linspace(0, 1, K))

def load(path, c):
    df = pd.read_feather(path)
    df = df[df["cluster"] == c].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

res = {}
dates_csv = pd.read_csv(REPRO + "/outputs/trackA/test_dates.csv")
for c in range(4):
    tr, te = load(UP + "/data/train_nmf.feather", c), load(UP + "/data/test_nmf.feather", c)
    assert (te["date"].dt.strftime("%Y-%m-%d").to_numpy() == dates_csv["date"].to_numpy()).all()
    Ytr, Yte = tr[cd("consumption")].to_numpy(float), te[cd("consumption")].to_numpy(float)
    Ttr, Tte = tr[cd("temp")].to_numpy(float), te[cd("temp")].to_numpy(float)
    Ltr, Lte = tr[cd("low")].to_numpy(float), te[cd("low")].to_numpy(float)
    Htr, Hte = tr[cd("high")].to_numpy(float), te[cd("high")].to_numpy(float)
    ptr = np.where(Ltr > 0, 0, np.where(Htr > 0, 2, 1)); pte = np.where(Lte > 0, 0, np.where(Hte > 0, 2, 1))
    K = 6
    # ---------- (a) my causality model on the training aggregate
    sig_mine = np.full((H, 3), np.nan)
    for h in range(H):
        kn = knots_q(Ttr[:, h], K)
        B = ns_basis(Ttr[:, h], kn)
        B = B - B.mean(0)
        X = np.column_stack([B, (ptr[:, h] == 0), (ptr[:, h] == 1), (ptr[:, h] == 2)]).astype(float)
        keep = X.std(0) > 0
        keep[-3:] = X[:, -3:].sum(0) > 0
        X = X[:, keep]
        w = np.ones(len(Ytr))
        for it in range(50):
            beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], Ytr[:, h] * np.sqrt(w), rcond=None)[0]
            r = Ytr[:, h] - X @ beta
            g = np.array([np.sqrt(np.mean(r[ptr[:, h] == p] ** 2)) if (ptr[:, h] == p).sum() >= 2 else np.nan for p in range(3)])
            gg = np.where(np.isnan(g), g[1], g)
            w_new = 1.0 / gg[ptr[:, h]] ** 2
            if np.allclose(w_new, w, rtol=1e-10):
                break
            w = w_new
        sig_mine[h] = g
    sig_mine_floor = np.maximum(sig_mine, 0.01)
    # ---------- saved sigma table
    st = pd.read_csv(REPRO + "/outputs/trackA/gam_sigma.csv")
    st = st[st["cluster"] == c].sort_values("h")
    sig_saved = st[["sigma_low", "sigma_normal", "sigma_high"]].to_numpy(float)
    # ---------- (b) my GAM mean f (unpenalized natural splines, 6 knots each)
    f_tr_mine = np.empty_like(Ytr); f_te_mine = np.empty_like(Yte)
    kb, kk = knots_q(tr["temp_smooth"].to_numpy(float), K), knots_q(tr["pos_y"].to_numpy(float), K)
    def design(df_T, df, L, Hh, h, kn):
        return np.column_stack([np.ones(len(df)), ns_basis(df_T[:, h], kn),
                                ns_basis(df["temp_smooth"].to_numpy(float), kb),
                                ns_basis(df["pos_y"].to_numpy(float), kk),
                                df["work_day"].to_numpy(float), L[:, h], Hh[:, h]])
    for h in range(H):
        kn = knots_q(Ttr[:, h], K)
        Xtr = design(Ttr, tr, Ltr, Htr, h, kn); Xte = design(Tte, te, Lte, Hte, h, kn)
        beta = np.linalg.lstsq(Xtr, Ytr[:, h], rcond=None)[0]
        f_tr_mine[:, h] = Xtr @ beta; f_te_mine[:, h] = Xte @ beta
    f_te_saved = np.load(REPRO + f"/outputs/trackA/gam_f_test_c{c}.npy")
    S = np.load(REPRO + f"/outputs/trackA/gam_samples_c{c}.npy")
    # ---------- (c) comparisons
    sd_sig_test_saved = sig_saved[np.arange(H)[None, :], pte]          # sigma^h(p_t^h) on test days
    sd_sig_test_mine = sig_mine_floor[np.arange(H)[None, :], np.where(np.isnan(sig_mine_floor[np.arange(H)[None, :], pte]), 1, pte)]
    sd_sig_test_mine_nofloor = sig_mine[np.arange(H)[None, :], np.where(np.isnan(sig_mine[np.arange(H)[None, :], pte]), 1, pte)]
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    r = {
        "sigma_saved_vs_mine_floor_corr": float(np.corrcoef(sig_saved[:, 1], sig_mine_floor[:, 1])[0, 1]),
        "sigma_saved_over_mine_floor_median_LNH": [float(np.nanmedian(sig_saved[:, p] / sig_mine_floor[:, p])) for p in range(3)],
        "frac_cells_floor_binding_mine": float(np.nanmean(sig_mine < 0.01)),
        "rms_sigma_saved_on_test_cells": rms(sd_sig_test_saved),
        "rms_sigma_mine_on_test_cells": rms(sd_sig_test_mine),
        "rms_sigma_mine_nofloor_on_test_cells": rms(sd_sig_test_mine_nofloor),
        "rms_test_resid_saved_f": rms(Yte - f_te_saved),
        "rms_test_resid_my_f": rms(Yte - f_te_mine),
        "rms_train_resid_my_f": rms(Ytr - f_tr_mine),
        "corr_saved_f_my_f_test": float(np.corrcoef(f_te_saved.ravel(), f_te_mine.ravel())[0, 1]),
        "samples_sd_mean": float(S.std(0, ddof=1).mean()),
        "samples_rms_sd": float(np.sqrt(S.var(0, ddof=1).mean())),
        "rms_err_of_sample_mean": rms(S.mean(0) - Yte),
        "sample_mean_vs_saved_f_maxabs": float(np.abs(S.mean(0) - f_te_saved).max()),
    }
    r["ratio_sigma_saved_over_test_resid"] = r["rms_sigma_saved_on_test_cells"] / r["rms_test_resid_saved_f"]
    r["ratio_sigma_mine_over_test_resid_my_f"] = r["rms_sigma_mine_on_test_cells"] / r["rms_test_resid_my_f"]
    r["ratio_sigma_mine_nofloor_over_test_resid_my_f"] = r["rms_sigma_mine_nofloor_on_test_cells"] / r["rms_test_resid_my_f"]
    r["spread_skill_rms"] = r["samples_rms_sd"] / r["rms_err_of_sample_mean"]
    # per-slot: sigma vs RMS test residual (Normal-tariff cells only, all slots)
    per_slot = []
    for h in range(H):
        m = pte[:, h] == 1
        per_slot.append([h, float(sig_saved[h, 1]), float(np.sqrt(np.mean((Yte[m, h] - f_te_saved[m, h]) ** 2))),
                         float(sig_mine[h, 1])])
    ps = np.array(per_slot)
    r["per_slot_ratio_sigmaN_over_testRMS_N_median_min_max"] = [float(np.median(ps[:, 1] / ps[:, 2])),
                                                               float((ps[:, 1] / ps[:, 2]).min()), float((ps[:, 1] / ps[:, 2]).max())]
    # the causality-model residual RMS on TEST days (what sigma estimates) vs the full-GAM residual RMS on test days
    caus_te = []
    for h in range(H):
        kn = knots_q(Ttr[:, h], K)
        B = ns_basis(Ttr[:, h], kn); mu0 = B.mean(0)
        X = np.column_stack([B - mu0, (ptr[:, h] == 0), (ptr[:, h] == 1), (ptr[:, h] == 2)]).astype(float)
        keepc = X[:, -3:].sum(0) > 0
        X = np.column_stack([X[:, :-3], X[:, -3:][:, keepc]])
        beta = np.linalg.lstsq(X, Ytr[:, h], rcond=None)[0]
        Bt = ns_basis(Tte[:, h], kn) - mu0
        Xt = np.column_stack([Bt, np.column_stack([(pte[:, h] == p) for p in range(3)])[:, keepc].astype(float)])
        caus_te.append(Yte[:, h] - Xt @ beta)
    r["rms_test_resid_causality_model_mine"] = rms(np.column_stack(caus_te))
    r["sigma_mine_floor_on_test_over_caus_test_resid"] = r["rms_sigma_mine_on_test_cells"] / r["rms_test_resid_causality_model_mine"]
    res[c] = r
    print(c, json.dumps(r), flush=True)
json.dump(res, open(OUTD + "/gam_check.json", "w"), indent=1)
