"""v-numbers: scenario effects for all 50 restarts per cluster and track, own numpy decoder.

Scenario conditions are built here from the test rows (sorted by date): Normal = all 96 tariff dummies 0; Low = low_09..18 = 1;
High = high_39..43 = 1; everything else from the real test-day conditions. z = default_rng([20200801, c]).standard_normal((200, 92, 4)),
shared by all scenarios and restarts. torch is used only to read the .pt files.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"
import json
import time
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
SP = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad"
UPDATA = SP + "/upstream/repo/data/"
OUTDIR = SP + "/verify/v-numbers"
REG = {"Low_in": ("Low", list(range(9, 19))), "High_in": ("High", list(range(39, 44))),
       "High_pre4": ("High", list(range(35, 39))), "High_post4": ("High", list(range(44, 48))),
       "High_pre2": ("High", [37, 38]), "High_post2": ("High", [44, 45])}
SIGN = {"Low_in": 1, "High_in": -1, "High_pre4": -1, "High_post4": -1, "High_pre2": -1, "High_post2": -1}
relu = lambda a: np.maximum(a, 0.0)
BOOT = np.random.default_rng(12345).integers(0, 92, size=(2000, 92))
t0 = time.time()


def load_sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.numpy().astype(np.float64) for k, v in ck["state_dict"].items()}, ck["meta"]


def decode(sd, meta, eps, Cm):
    n, D, d = eps.shape
    Z = eps.reshape(-1, d)
    Cr = np.tile(Cm, (n, 1))
    out = relu(np.hstack([Z, Cr]) @ sd["dec_hidden.weight"].T + sd["dec_hidden.bias"]) @ sd["dec_out.weight"].T + sd["dec_out.bias"]
    return (out * (meta["conso_max"] - meta["conso_min"]) + meta["conso_min"]).reshape(n, D, 48)


res = {}
for track in ("trackA", "trackB"):
    df = pd.read_feather(UPDATA + "test_nmf.feather") if track == "trackA" else pd.read_parquet(REPRO + "/data/test_ours.parquet")
    df = df.assign(cluster=df["cluster"].to_numpy(dtype=np.float64).astype(int), date=pd.to_datetime(df["date"]))
    sel = json.load(open(REPRO + f"/outputs/{track}/cvae_selection.json"))
    res[track] = {}
    for c in range(4):
        sd0, meta0 = load_sd(REPRO + f"/outputs/{track}/models/cvae_selected_c{c}.pt")
        cc = list(meta0["cond_cols"])
        st = df.loc[df.cluster == c].sort_values("date", kind="stable")
        C = st[cc].to_numpy(np.float64)
        tcols = [i for i, n in enumerate(cc) if n.startswith(("low_", "high_"))]
        assert len(tcols) == 96
        Cn = C.copy(); Cn[:, tcols] = 0.0
        Cl = Cn.copy(); Cl[:, [cc.index(f"low_{h:02d}") for h in range(9, 19)]] = 1.0
        Ch = Cn.copy(); Ch[:, [cc.index(f"high_{h:02d}") for h in range(39, 44)]] = 1.0
        eps = np.random.default_rng([20200801, c]).standard_normal((200, len(C), 4))
        # selected model: must reproduce the saved scenario arrays
        chk = {}
        for s, Cm in (("Normal", Cn), ("Low", Cl), ("High", Ch)):
            chk[s] = float(np.abs(decode(sd0, meta0, eps, Cm) - np.load(REPRO + f"/outputs/{track}/cvae_scen_{s}_c{c}.npy")).max())
        rows = []
        for seed in range(50):
            sd, meta = load_sd(REPRO + f"/outputs/{track}/models/cvae_c{c}_s{seed:02d}.pt")
            mN = decode(sd, meta, eps, Cn).mean(0)
            mS = {"Low": decode(sd, meta, eps, Cl).mean(0), "High": decode(sd, meta, eps, Ch).mean(0)}
            r = {"seed": seed, "is_selected_weights": bool(all(np.array_equal(sd[k], sd0[k]) for k in sd))}
            for name, (s, slots) in REG.items():
                num = (mS[s][:, slots] - mN[:, slots]).mean(1)
                den = mN[:, slots].mean(1)
                pct = 100 * num.mean() / den.mean()
                bp = 100 * num[BOOT].mean(1) / den[BOOT].mean(1)
                lo, hi = np.percentile(bp, [2.5, 97.5])
                r[name] = float(pct)
                r[name + "_appears"] = bool((SIGN[name] > 0 and lo > 0) or (SIGN[name] < 0 and hi < 0))
                r[name + "_contradicted"] = bool((SIGN[name] > 0 and hi < 0) or (SIGN[name] < 0 and lo > 0))
            rows.append(r)
        sel_seed = [r["seed"] for r in rows if r["is_selected_weights"]]
        P = lambda k: np.array([r[k] for r in rows])
        A = lambda k: np.array([r[k + "_appears"] for r in rows])
        summ = {
            "selected_model_regenerates_saved_scen_arrays_max_abs_kwh": chk,
            "selected_seed_by_weights": sel_seed,
            "pct_restarts_Low_in_positive": float(100 * (P("Low_in") > 0).mean()),
            "pct_restarts_High_in_negative": float(100 * (P("High_in") < 0).mean()),
            "pct_restarts_High_pre4_negative": float(100 * (P("High_pre4") < 0).mean()),
            "pct_restarts_High_post4_negative": float(100 * (P("High_post4") < 0).mean()),
            "n_both_directions_appear": int((A("Low_in") & A("High_in")).sum()),
            "n_side_effect_pre4_or_post4_appears": int((A("High_pre4") | A("High_post4")).sum()),
            "n_side_effect_pre4_and_post4_appear": int((A("High_pre4") & A("High_post4")).sum()),
            "n_side_effect_pre2_or_post2_appears": int((A("High_pre2") | A("High_post2")).sum()),
            "n_side_effect_pre2_and_post2_appear": int((A("High_pre2") & A("High_post2")).sum()),
            "median_Low_in_pct": float(np.median(P("Low_in"))), "median_High_in_pct": float(np.median(P("High_in"))),
        }
        res[track][c] = {"summary": summ, "restarts": rows}
        print(f"{time.time()-t0:5.1f}s {track} c{c} selected seed {sel_seed} regen {chk} | Low>0 {summ['pct_restarts_Low_in_positive']:.0f}% "
              f"High<0 {summ['pct_restarts_High_in_negative']:.0f}% pre4<0 {summ['pct_restarts_High_pre4_negative']:.0f}% "
              f"both {summ['n_both_directions_appear']} side(pre4|post4) {summ['n_side_effect_pre4_or_post4_appears']} "
              f"side(pre4&post4) {summ['n_side_effect_pre4_and_post4_appear']} side(pre2|post2) {summ['n_side_effect_pre2_or_post2_appears']} "
              f"side(pre2&post2) {summ['n_side_effect_pre2_and_post2_appear']} medLow {summ['median_Low_in_pct']:.1f} medHigh {summ['median_High_in_pct']:.1f}",
              flush=True)
json.dump(res, open(OUTDIR + "/restart_effects.json", "w"), indent=1)
print("done", time.time() - t0)
