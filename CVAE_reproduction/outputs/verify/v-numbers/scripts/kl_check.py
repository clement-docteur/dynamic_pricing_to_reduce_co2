"""v-numbers: posterior collapse. Train-day KL(q(z|y,x) || N(0, I)) of the selected CVAEs, own numpy forward pass.

torch is used only to read the saved state_dict (torch.load, weights_only=True); lib/ is not imported.
Also: decoder at z = 0 vs the saved sample mean (links the arrays to each test day's conditions), latent sensitivity,
and an attempt to regenerate the saved arrays with the documented z draws (default_rng([20200801, c])).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"
import json
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
SP = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad"
UPDATA = SP + "/upstream/repo/data/"
OUTDIR = SP + "/verify/v-numbers"
CONS = [f"consumption_{h:02d}" for h in range(48)]


def table(track, part):
    if track == "trackA":
        df = pd.read_feather(UPDATA + f"{part}_nmf.feather")
    else:
        df = pd.read_parquet(REPRO + f"/data/{part}_ours.parquet")
    return df.assign(cluster=df["cluster"].to_numpy(dtype=np.float64).astype(int), date=pd.to_datetime(df["date"]))


relu = lambda a: np.maximum(a, 0.0)
res = {}
for track in ("trackA", "trackB"):
    tr, te = table(track, "train"), table(track, "test")
    res[track] = {}
    for c in range(4):
        ck = torch.load(REPRO + f"/outputs/{track}/models/cvae_selected_c{c}.pt", map_location="cpu", weights_only=True)
        sd = {k: v.detach().numpy().astype(np.float64) for k, v in ck["state_dict"].items()}
        meta = ck["meta"]
        d = int(ck["arch"]["latent_dim"])
        cc = list(meta["cond_cols"])
        sub = tr.loc[tr.cluster == c].sort_values("date", kind="stable")
        Y = sub[CONS].to_numpy(np.float64)
        C = sub[cc].to_numpy(np.float64)
        ymin, ymax = Y.min(), Y.max()
        X = (Y - meta["conso_min"]) / (meta["conso_max"] - meta["conso_min"])
        o = relu(np.hstack([X, C]) @ sd["enc_hidden.weight"].T + sd["enc_hidden.bias"]) @ sd["enc_out.weight"].T + sd["enc_out.bias"]
        mu, ls = o[:, :d], o[:, d:]
        sig = np.exp(ls)
        kl_dim = 0.5 * (sig ** 2 + mu ** 2 - 1.0 - 2.0 * ls)  # (days, d), closed form KL to N(0,1)
        kl_day = kl_dim.sum(1)
        # test days, for reference
        st = te.loc[te.cluster == c].sort_values("date", kind="stable")
        Yt = st[CONS].to_numpy(np.float64)
        Ct = st[cc].to_numpy(np.float64)
        Xt = (Yt - meta["conso_min"]) / (meta["conso_max"] - meta["conso_min"])
        ot = relu(np.hstack([Xt, Ct]) @ sd["enc_hidden.weight"].T + sd["enc_hidden.bias"]) @ sd["enc_out.weight"].T + sd["enc_out.bias"]
        klt = (0.5 * (np.exp(2 * ot[:, d:]) + ot[:, :d] ** 2 - 1.0 - 2.0 * ot[:, d:])).sum(1)

        def dec(Z, Cm):
            out = relu(np.hstack([Z, Cm]) @ sd["dec_hidden.weight"].T + sd["dec_hidden.bias"]) @ sd["dec_out.weight"].T + sd["dec_out.bias"]
            return out * (meta["conso_max"] - meta["conso_min"]) + meta["conso_min"]

        S = np.load(REPRO + f"/outputs/{track}/cvae_samples_c{c}.npy")
        Sm = S.mean(0)
        d0 = dec(np.zeros((len(Ct), d)), Ct)
        sens = [float(np.abs(dec(np.eye(d)[[k] * len(Ct)], Ct) - d0).mean()) for k in range(d)]
        # regeneration with the documented draws
        eps = np.random.default_rng([20200801, c]).standard_normal((S.shape[0], len(Ct), d))
        regen = dec(eps.reshape(-1, d), np.tile(Ct, (S.shape[0], 1))).reshape(S.shape)
        r = {
            "train_days": int(len(Y)),
            "scaling_meta_equals_train_minmax": bool(np.isclose(meta["conso_min"], ymin, rtol=0, atol=1e-15) and np.isclose(meta["conso_max"], ymax, rtol=0, atol=1e-15)),
            "train_kl_per_day_mean_nats": float(kl_day.mean()),
            "train_kl_per_day_max_nats": float(kl_day.max()),
            "train_kl_per_dim_mean": kl_dim.mean(0).tolist(),
            "posterior_sd_min": float(sig.min()), "posterior_sd_max": float(sig.max()),
            "posterior_sd_mean_per_dim": sig.mean(0).tolist(),
            "posterior_mean_abs_max": float(np.abs(mu).max()),
            "posterior_mean_sd_across_days_per_dim": mu.std(0).tolist(),
            "test_kl_per_day_mean_nats": float(klt.mean()),
            "decoder_z0_vs_saved_sample_mean_max_abs_kwh": float(np.abs(d0 - Sm).max()),
            "decoder_z0_vs_saved_sample_mean_shift1day_median_abs_kwh": float(np.median(np.abs(d0 - np.roll(Sm, 1, axis=0)))),
            "decoder_z0_vs_saved_sample_mean_median_abs_kwh": float(np.median(np.abs(d0 - Sm))),
            "latent_unit_change_mean_abs_output_kwh": sens,
            "regenerated_with_documented_z_max_abs_diff_kwh": float(np.abs(regen - S).max()),
        }
        res[track][c] = r
        print(track, c, f"KL mean {r['train_kl_per_day_mean_nats']:.3e} max {r['train_kl_per_day_max_nats']:.3e} "
              f"sd [{r['posterior_sd_min']:.5f},{r['posterior_sd_max']:.5f}] |mu|max {r['posterior_mean_abs_max']:.2e} "
              f"testKL {r['test_kl_per_day_mean_nats']:.2e} z0-vs-mean {r['decoder_z0_vs_saved_sample_mean_max_abs_kwh']:.2e} "
              f"regen {r['regenerated_with_documented_z_max_abs_diff_kwh']:.2e} sens {np.round(sens, 5).tolist()} scaling_ok {r['scaling_meta_equals_train_minmax']}",
              flush=True)
json.dump(res, open(OUTDIR + "/kl_recomputed.json", "w"), indent=1)
