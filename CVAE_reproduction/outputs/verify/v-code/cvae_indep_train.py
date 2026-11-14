"""v-code: an independent CVAE trainer written from the upstream text (codes/cvae.py + cvae_generate.ipynb),
NOT importing lib. Keras-orientation kernels, Glorot-uniform / zero bias, float64, my own Keras-style Adam,
batch 1 in file order, 50 epochs, loss = SSE + eta*KL. Cluster 3 of train_nmf. Runs eta in {10, 1, 0.1, 0.01}
(plus eta=10 with the upstream quirks: no Adam bias correction) to test whether posterior collapse is set by the
paper's objective (eta=10 vs [0,1]-scaled SSE) or by the port."""
import os, sys, json, time, math
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
UP = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo"
OUTD = os.path.dirname(os.path.abspath(__file__))
cols_data = ['consumption_{:02}'.format(i) for i in range(48)]
cols_price = [n for i in range(48) for n in ('low_{:02}'.format(i), 'high_{:02}'.format(i))]
cols_cond = ['temp_smooth_n', 'pca_temp_0', 'pca_temp_1', 'pca_temp_2', 'pos_y', 'work_day'] + cols_price
c = int(sys.argv[1]) if len(sys.argv) > 1 else 3
tr = pd.read_feather(UP + "/data/train_nmf.feather"); te = pd.read_feather(UP + "/data/test_nmf.feather")
trc = tr.loc[tr["cluster"] == c]
tec = te.loc[te["cluster"] == c].assign(date=lambda x: pd.to_datetime(x["date"])).sort_values("date")
cmax = max(trc.loc[:, cols_data].max()); cmin = min(trc.loc[:, cols_data].min())
Xtr = torch.tensor(((trc[cols_data].to_numpy(float) - cmin) / (cmax - cmin)))
Ctr = torch.tensor(trc[cols_cond].to_numpy(float))
Yte = tec[cols_data].to_numpy(float)
Cte = torch.tensor(tec[cols_cond].to_numpy(float))
D = torch.float64
LAT = 4

def glorot(nin, nout, rng):
    lim = math.sqrt(6.0 / (nin + nout))
    return torch.tensor(rng.uniform(-lim, lim, size=(nin, nout)), dtype=D, requires_grad=True)

def init(seed):
    rng = np.random.default_rng(seed)
    P = [glorot(150, 15, rng), torch.zeros(15, dtype=D, requires_grad=True),
         glorot(15, 8, rng), torch.zeros(8, dtype=D, requires_grad=True),
         glorot(106, 15, rng), torch.zeros(15, dtype=D, requires_grad=True),
         glorot(15, 48, rng), torch.zeros(48, dtype=D, requires_grad=True)]
    return P

def enc(P, x, cc):
    h = torch.relu(torch.cat([x, cc], 1) @ P[0] + P[1]); o = h @ P[2] + P[3]
    return o[:, :LAT], o[:, LAT:]

def dec(P, z, cc):
    return torch.relu(torch.cat([z, cc], 1) @ P[4] + P[5]) @ P[6] + P[7]

def kl(mu, ls):
    return 0.5 * (torch.exp(ls) ** 2 + mu ** 2 - 1 - 2 * ls).sum(1)

def train(eta, seed, bias_correction=True, epochs=50):
    P = init(seed)
    m = [torch.zeros_like(p) for p in P]; v = [torch.zeros_like(p) for p in P]
    t = 0 if bias_correction else 10 ** 7
    g = torch.Generator().manual_seed(seed + 99)
    lr, b1, b2, ep = 1e-3, 0.9, 0.999, 1e-7
    for e in range(epochs):
        for i in range(Xtr.shape[0]):
            x, cc = Xtr[i:i + 1], Ctr[i:i + 1]
            mu, ls = enc(P, x, cc)
            z = torch.randn((1, LAT), generator=g, dtype=D) * torch.exp(ls) + mu
            loss = ((x - dec(P, z, cc)) ** 2).sum() + eta * kl(mu, ls).sum()
            grads = torch.autograd.grad(loss, P)
            t += 1
            lr_t = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
            with torch.no_grad():
                for k in range(len(P)):
                    m[k].mul_(b1).add_(grads[k], alpha=1 - b1)
                    v[k].mul_(b2).addcmul_(grads[k], grads[k], value=1 - b2)
                    P[k].sub_(lr_t * m[k] / (v[k].sqrt() + ep))
    return P

def evaluate(P):
    with torch.no_grad():
        mu, ls = enc(P, Xtr, Ctr)
        klv = kl(mu, ls)
        gz = torch.Generator().manual_seed(5)
        Z = torch.randn((200, Cte.shape[0], LAT), generator=gz, dtype=D)
        S = torch.stack([dec(P, Z[k], Cte) for k in range(200)]).numpy() * (cmax - cmin) + cmin   # (200, 92, 48) kWh
    dev = S.mean(0) - Yte
    rmse_rm = np.sqrt((dev ** 2).mean(1))
    sd = S.std(0, ddof=1)
    return {"train_kl_mean": float(klv.mean()), "post_std_mean": float(torch.exp(ls).mean()),
            "post_mean_sd_across_days": float(mu.std(0).mean()),
            "test_rmse_rootmean_median_kwh": float(np.median(rmse_rm)),
            "test_sample_sd_kwh_mean": float(sd.mean()),
            "spread_over_rmse": float(sd.mean() / np.sqrt((dev ** 2).mean()))}

res = {}
for name, eta, bc in [("eta10", 10.0, True), ("eta10_noBC_quirk", 10.0, False), ("eta1", 1.0, True),
                      ("eta0.1", 0.1, True), ("eta0.01", 0.01, True)]:
    t0 = time.perf_counter()
    P = train(eta, seed=11, bias_correction=bc)
    r = evaluate(P); r["train_s"] = time.perf_counter() - t0
    res[name] = r
    print(name, json.dumps(r), flush=True)
json.dump(res, open(OUTD + f"/cvae_indep_train_c{c}.json", "w"), indent=1)
