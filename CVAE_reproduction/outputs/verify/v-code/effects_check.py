"""v-code: tariff-effect definitions. (1) windows and % base recomputed from the saved scenario arrays with my own
code and compared with outputs/trackA/effects_summary.json; (2) regenerate cluster-3 High-scenario days from the
selected decoder with MY numpy decoder and MY condition vectors (real test-day conditions, only the 96 dummies
changed) and compare with outputs/trackA/cvae_scen_High_c3.npy; (3) GAM/CVAE scenario arrays outside windows."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
import numpy as np, pandas as pd, torch
torch.set_num_threads(2)
REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
UP = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo"
OUTD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPRO)
from lib import cvae_torch as ct          # only to load the saved model
from lib import effects as EF            # only to read its window definitions for comparison
out = {}
LOW_W, HIGH_W = list(range(9, 19)), list(range(39, 44))
PRE4, POST4 = list(range(35, 39)), list(range(44, 48))
reg_lib_H = EF.scenario_regions("High"); reg_lib_L = EF.scenario_regions("Low")
out["lib_windows"] = {"High_in": reg_lib_H["in_window"], "High_pre4": reg_lib_H["pre4"], "High_post4": reg_lib_H["post4"],
                      "High_pre2": reg_lib_H["pre2"], "High_post2": reg_lib_H["post2"], "Low_in": reg_lib_L["in_window"]}
out["windows_match_task"] = (reg_lib_H["in_window"] == HIGH_W and reg_lib_H["pre4"] == PRE4 and
                             reg_lib_H["post4"] == POST4 and reg_lib_L["in_window"] == LOW_W)

def eff(model, c, scen, slots):
    N = np.load(REPRO + f"/outputs/trackA/{model}_scen_Normal_c{c}.npy").mean(0)
    S = np.load(REPRO + f"/outputs/trackA/{model}_scen_{scen}_c{c}.npy").mean(0)
    return float(100 * (S - N)[:, slots].mean() / N[:, slots].mean()), float((S - N)[:, slots].mean())

summ = json.load(open(REPRO + "/outputs/trackA/effects_summary.json"))
def find_pct(obj, model, c, scen, region):
    # effects_summary layout is not assumed: search for the region pct in the nested json
    hits = []
    def rec(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                rec(v, path + [str(k)])
        elif isinstance(o, (int, float)) and path and path[-1] == "pct":
            p = "/".join(path)
            if (model in p) and (f"c{c}" in p or f"/{c}/" in p) and scen in p and region in p:
                hits.append((p, o))
    rec(obj, [])
    return hits

table = {}
for model in ("cvae", "gam"):
    for c in range(4):
        row = {"Low_in": eff(model, c, "Low", LOW_W), "High_in": eff(model, c, "High", HIGH_W),
               "High_pre4": eff(model, c, "High", PRE4), "High_post4": eff(model, c, "High", POST4)}
        hits = {k: find_pct(summ, model, c, k.split("_")[0], {"in": "in_window", "pre4": "pre4", "post4": "post4"}[k.split("_")[1]])
                for k in row}
        row_cmp = {}
        for k, (pct, kwh) in row.items():
            h = hits[k]
            row_cmp[k] = {"mine_pct": round(pct, 3), "mine_kwh": kwh,
                          "summary_hits": [(p, round(v, 3)) for p, v in h[:2]]}
        table[f"{model}_c{c}"] = row_cmp
out["effects_recomputed"] = table

# ---------- (2) regenerate cluster-3 High scenario with my decoder ----------
cols_data = ['consumption_{:02}'.format(i) for i in range(48)]
cols_price = [n for i in range(48) for n in ('low_{:02}'.format(i), 'high_{:02}'.format(i))]
cols_cond = ['temp_smooth_n', 'pca_temp_0', 'pca_temp_1', 'pca_temp_2', 'pos_y', 'work_day'] + cols_price
te = pd.read_feather(UP + "/data/test_nmf.feather")
te["date"] = pd.to_datetime(te["date"])
t3 = te[te["cluster"] == 3].sort_values("date").reset_index(drop=True)
dates = pd.read_csv(REPRO + "/outputs/trackA/test_dates.csv")["date"].tolist()
assert t3["date"].dt.strftime("%Y-%m-%d").tolist() == dates
Creal = t3[cols_cond].to_numpy(float)
m = ct.load_model(REPRO + "/outputs/trackA/models/cvae_selected_c3.pt")
sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}
cmin, cmax = m.meta["conso_min"], m.meta["conso_max"]
def np_decode(z, cc):
    hdn = np.maximum(np.concatenate([z, cc], 1) @ sd["dec_hidden.weight"].T + sd["dec_hidden.bias"], 0)
    return (hdn @ sd["dec_out.weight"].T + sd["dec_out.bias"]) * (cmax - cmin) + cmin
eps = np.random.default_rng([20200801, 3]).standard_normal((200, 92, 4))
saved_H = np.load(REPRO + "/outputs/trackA/cvae_scen_High_c3.npy")
saved_L = np.load(REPRO + "/outputs/trackA/cvae_scen_Low_c3.npy")
saved_N = np.load(REPRO + "/outputs/trackA/cvae_scen_Normal_c3.npy")
saved_S = np.load(REPRO + "/outputs/trackA/cvae_samples_c3.npy")
lowi = [cols_cond.index(f"low_{h:02d}") for h in range(48)]; highi = [cols_cond.index(f"high_{h:02d}") for h in range(48)]
real_event = np.where(Creal[:, lowi + highi].sum(1) > 0)[0]
days = [0, int(real_event[0]), int(real_event[-1]), 45]
chk = []
for d in days:
    cH = Creal[d].copy(); cH[lowi + highi] = 0.0; cH[[highi[h] for h in HIGH_W]] = 1.0
    cL = Creal[d].copy(); cL[lowi + highi] = 0.0; cL[[lowi[h] for h in LOW_W]] = 1.0
    cN = Creal[d].copy(); cN[lowi + highi] = 0.0
    nontariff = [i for i in range(102) if i not in lowi + highi]
    assert np.array_equal(cH[nontariff], Creal[d][nontariff])
    gH = np_decode(eps[:, d, :], np.repeat(cH[None], 200, 0))
    gL = np_decode(eps[:, d, :], np.repeat(cL[None], 200, 0))
    gN = np_decode(eps[:, d, :], np.repeat(cN[None], 200, 0))
    gR = np_decode(eps[:, d, :], np.repeat(Creal[d][None], 200, 0))
    chk.append({"day": d, "date": dates[d], "real_day_has_events": bool(d in real_event),
                "n_real_low_slots": int(Creal[d][lowi].sum()), "n_real_high_slots": int(Creal[d][highi].sum()),
                "maxabs_High_vs_saved": float(np.abs(gH - saved_H[:, d]).max()),
                "maxabs_Low_vs_saved": float(np.abs(gL - saved_L[:, d]).max()),
                "maxabs_Normal_vs_saved": float(np.abs(gN - saved_N[:, d]).max()),
                "maxabs_real_vs_saved_samples": float(np.abs(gR - saved_S[:, d]).max())})
out["regen_c3"] = chk
# alternative hypothesis: scenarios built WITHOUT zeroing the real events -> would differ on event days
d = int(real_event[0])
cH_bad = Creal[d].copy(); cH_bad[[highi[h] for h in HIGH_W]] = 1.0
out["regen_c3_if_real_events_kept_maxabs_vs_saved"] = float(np.abs(np_decode(eps[:, d, :], np.repeat(cH_bad[None], 200, 0)) - saved_H[:, d]).max())
# Normal scenario == real samples on all-Normal days
alln = np.setdiff1d(np.arange(92), real_event)
out["normal_scen_equals_real_samples_on_all_normal_days_maxabs"] = float(np.abs(saved_N[:, alln] - saved_S[:, alln]).max())
out["n_all_normal_test_days"] = int(len(alln))
# (3) outside-window differences
for model in ("cvae", "gam"):
    for c in (0, 3):
        N = np.load(REPRO + f"/outputs/trackA/{model}_scen_Normal_c{c}.npy")
        Hs = np.load(REPRO + f"/outputs/trackA/{model}_scen_High_c{c}.npy")
        Ls = np.load(REPRO + f"/outputs/trackA/{model}_scen_Low_c{c}.npy")
        outH = [h for h in range(48) if h not in HIGH_W]; outL = [h for h in range(48) if h not in LOW_W]
        out[f"{model}_c{c}_maxabs_outside_window_High_Low"] = [float(np.abs(Hs[..., outH] - N[..., outH]).max()),
                                                               float(np.abs(Ls[..., outL] - N[..., outL]).max())]
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(OUTD + "/effects_check.json", "w"), indent=1, default=str)
