"""v-code: independent numpy re-implementation of the upstream CVAE forward pass + loss (codes/cvae.py, read as text),
compared with lib.cvae_torch on the selected Track A cluster-3 model. Also: KL / posterior stats on all training days,
selection-metric alternatives across the 200 restarts, and data-order/scaling checks against the feather files."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
import numpy as np, pandas as pd
import torch
torch.set_num_threads(2)

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
UP = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo"
sys.path.insert(0, REPRO)
from lib import cvae_torch as ct   # allowed: only to load the saved model

out = {}
# ---------- my own column lists, typed from the upstream notebook text ----------
cols_data = ['consumption_{:02}'.format(i) for i in range(48)]
cols_price = []
for i in range(48):
    cols_price.append('low_{:02}'.format(i)); cols_price.append('high_{:02}'.format(i))
cols_temp = ['temp_smooth_n'] + ['pca_temp_{:1}'.format(i) for i in range(3)]
cols_cond = cols_temp + ['pos_y', 'work_day'] + cols_price
out["cond_cols_identical_to_lib"] = (cols_cond == ct.COLS_COND)
out["n_cond"] = len(cols_cond)

tr = pd.read_feather(UP + "/data/train_nmf.feather")
te = pd.read_feather(UP + "/data/test_nmf.feather")
c = 3
trc = tr.loc[tr["cluster"] == c]          # file order, as upstream (df_train.loc[mask])
out["train_rows_c3"] = int(len(trc))
d = pd.to_datetime(trc["date"])
out["train_c3_file_order_is_ascending_date"] = bool(d.is_monotonic_increasing)
Y = trc[cols_data].to_numpy(dtype=np.float64)
C = trc[cols_cond].to_numpy(dtype=np.float64)
cmax = max(trc.loc[:, cols_data].max()); cmin = min(trc.loc[:, cols_data].min())   # upstream recipe verbatim
m = ct.load_model(REPRO + "/outputs/trackA/models/cvae_selected_c3.pt")
out["scaling_min_max_mine"] = [float(cmin), float(cmax)]
out["scaling_min_max_model_meta"] = [m.meta["conso_min"], m.meta["conso_max"]]
out["meta"] = {k: m.meta[k] for k in m.meta if k != "cond_cols"}
Xs = (Y - cmin) / (cmax - cmin)

# ---------- weights as Keras kernels (in, out) ----------
sd = {k: v.detach().numpy().astype(np.float64) for k, v in m.state_dict().items()}
out["state_dict_shapes"] = {k: list(v.shape) for k, v in sd.items()}
K1, b1 = sd["enc_hidden.weight"].T, sd["enc_hidden.bias"]
K2, b2 = sd["enc_out.weight"].T, sd["enc_out.bias"]
D1, a1 = sd["dec_hidden.weight"].T, sd["dec_hidden.bias"]
D2, a2 = sd["dec_out.weight"].T, sd["dec_out.bias"]
relu = lambda v: np.maximum(v, 0.0)
latent = 4

def np_encode(x, cc):
    o = relu(np.concatenate([x, cc], 1) @ K1 + b1) @ K2 + b2      # Dense(15, relu) -> Dense(8, linear)
    return o[:, :latent], o[:, latent:]                             # tf.split(..., 2, axis=1)

def np_decode(z, cc):
    return relu(np.concatenate([z, cc], 1) @ D1 + a1) @ D2 + a2     # Dense(15, relu) -> Dense(48, linear)

def np_kl(mean, logscale):
    # KL( MVNDiag(mean, scale_diag=exp(logscale)) || MVNDiag(0, 1) ), written from the textbook formula
    var = np.exp(logscale) ** 2
    return 0.5 * np.sum(var + mean ** 2 - 1.0 - np.log(var), axis=1)

def np_loss(x, cc, eps, eta=10.0):
    mean, logscale = np_encode(x, cc)
    z = eps * np.exp(logscale) + mean
    xh = np_decode(z, cc)
    rec = np.sum((x - xh) ** 2, axis=1)
    kl = np_kl(mean, logscale)
    return eta * kl + rec, rec, kl, xh

rng = np.random.default_rng(7)
idx = np.array([0, 1, 50, 100, 200, 272])
eps = rng.standard_normal((len(idx), latent))
L, R, KL, XH = np_loss(Xs[idx], C[idx], eps)
with torch.no_grad():
    Lt, Rt, KLt = m.loss_terms(torch.from_numpy(Xs[idx]), torch.from_numpy(C[idx]), torch.from_numpy(eps), 10.0)
    mu_t, ls_t = m.encode(torch.from_numpy(Xs[idx]), torch.from_numpy(C[idx]))
    XHt = m.decode(mu_t + torch.exp(ls_t) * torch.from_numpy(eps), torch.from_numpy(C[idx]))
out["few_days_idx"] = idx.tolist()
out["np_loss"] = L.tolist(); out["lib_loss"] = Lt.numpy().tolist()
out["np_rec"] = R.tolist(); out["np_kl"] = KL.tolist(); out["lib_kl"] = KLt.numpy().tolist()
out["max_abs_diff_loss"] = float(np.abs(L - Lt.numpy()).max())
out["max_abs_diff_kl"] = float(np.abs(KL - KLt.numpy()).max())
out["max_abs_diff_xhat"] = float(np.abs(XH - XHt.numpy()).max())
# torch.distributions as a third, independent KL
mu_np, ls_np = np_encode(Xs[idx], C[idx])
kl_td = torch.distributions.kl_divergence(
    torch.distributions.Independent(torch.distributions.Normal(torch.from_numpy(mu_np), torch.from_numpy(np.exp(ls_np))), 1),
    torch.distributions.Independent(torch.distributions.Normal(torch.zeros(latent, dtype=torch.float64), torch.ones(latent, dtype=torch.float64)), 1)).numpy()
out["max_abs_diff_kl_torchdist"] = float(np.abs(KL - kl_td).max())

# ---------- all training days: KL, posterior, dependence of the decoder on z ----------
mu_all, ls_all = np_encode(Xs, C)
kl_all = np_kl(mu_all, ls_all)
out["train_kl_mean"] = float(kl_all.mean()); out["train_kl_max"] = float(kl_all.max())
out["posterior_std_mean_per_dim"] = np.exp(ls_all).mean(0).tolist()
out["posterior_std_minmax"] = [float(np.exp(ls_all).min()), float(np.exp(ls_all).max())]
out["posterior_mean_abs_max"] = float(np.abs(mu_all).max())
out["posterior_mean_sd_across_days_per_dim"] = mu_all.std(0).tolist()
# mutual information proxy: does mu depend on x at all? perturb x by its residual scale
mu_shuf, _ = np_encode(Xs[rng.permutation(len(Xs))], C)
out["mu_change_when_x_shuffled_maxabs"] = float(np.abs(mu_shuf - mu_all).max())
# decoder sensitivity to z: SD over 2000 z draws vs SD of residuals, scaled units and kWh
zs = rng.standard_normal((2000, latent))
xs_day = np.stack([np_decode(zs, np.repeat(C[i:i+1], 2000, 0)) for i in range(0, 273, 30)])  # (10, 2000, 48)
sd_z = xs_day.std(1).mean()
xh0 = np_decode(np.zeros((len(Xs), latent)), C)
resid_rms = np.sqrt(((Xs - xh0) ** 2).mean())
out["decoder_sd_over_z_scaled_mean"] = float(sd_z)
out["decoder_sd_over_z_kwh_mean"] = float(sd_z * (cmax - cmin))
out["train_resid_rms_z0_scaled"] = float(resid_rms)
out["train_resid_rms_z0_kwh"] = float(resid_rms * (cmax - cmin))
# Jacobian norm of decoder wrt z at z=0 (kWh per unit z)
J = []
h = 1e-5
for k in range(latent):
    e = np.zeros((len(Xs), latent)); e[:, k] = h
    J.append((np_decode(e, C) - np_decode(-e, C)) / (2 * h))
J = np.stack(J, -1)  # (n, 48, 4)
out["decoder_dxhat_dz_rms_kwh"] = float(np.sqrt((J ** 2).mean()) * (cmax - cmin))
out["train_mean_SSE_scaled_z0"] = float(((Xs - xh0) ** 2).sum(1).mean())
# PPCA criterion: loss SSE + eta*KL <=> Gaussian decoder with sigma^2 = eta/2 = 5 (per slot, scaled units).
# A latent direction is worth using only if an eigenvalue of the per-day residual covariance exceeds eta/2.
R0 = Xs - xh0
ev = np.linalg.eigvalsh(np.cov(R0.T, bias=True))[::-1]
out["resid_cov_top_eigs_scaled"] = ev[:5].tolist()
out["resid_total_var_scaled"] = float(ev.sum())
Xc = Xs - Xs.mean(0)
ev2 = np.linalg.eigvalsh(np.cov(Xc.T, bias=True))[::-1]
out["raw_scaled_top_eigs_unconditional"] = ev2[:5].tolist()
out["ppca_threshold_eta_over_2"] = 5.0

# ---------- restarts: KL for all 200, selection-metric alternatives ----------
d = pd.read_csv(REPRO + "/outputs/trackA/cvae_restarts.csv")
out["all_restarts_train_kl_max"] = float(d["train_kl_final"].max())
out["all_restarts_train_kl_median"] = float(d["train_kl_final"].median())
psd = d[[f"train_posterior_std_mean_{i}" for i in range(4)]].to_numpy()
out["all_restarts_post_std_minmax"] = [float(psd.min()), float(psd.max())]
sel = {}
for cl in range(4):
    dc = d[d["cluster"] == cl]
    sel[cl] = {k: int(dc.loc[dc[k].idxmin(), "seed"]) for k in
               ["test_mse_post", "test_mse_prior", "test_mse_post_avg20", "test_mse_prior_avg20", "test_eq8_post",
                "test_rmse_mean200_kwh_median"]}
    sel[cl]["spearman_post_vs_prior"] = float(dc["test_mse_post"].rank().corr(dc["test_mse_prior"].rank()))
    sel[cl]["max_abs_post_minus_prior"] = float((dc["test_mse_post"] - dc["test_mse_prior"]).abs().max())
out["selection_alternatives"] = sel
q = json.load(open(REPRO + "/outputs/trackA/cvae_sensitivity_upstream_quirks.json"))
out["quirk_train_kl_median"] = {k: v.get("quirk_train_kl_final_median") for k, v in q["clusters"].items()}
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/cvae_check1.json", "w"), indent=1, default=float)
