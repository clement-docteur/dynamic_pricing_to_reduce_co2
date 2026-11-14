"""v-code: several eta=10 restarts of my independent trainer (cvae_indep_train.py functions, re-declared here) on
one cluster; reports compute_mse (posterior z, 100*RMSE scaled, mean over test days) and the 200-sample-mean RMSE
(kWh, root-mean, median over days), to compare with the port's 50-restart distribution."""
import os, sys, json, time, math, importlib.util
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, torch
torch.set_num_threads(1)
here = os.path.dirname(os.path.abspath(__file__))
c = int(sys.argv[1]); seeds = [int(s) for s in sys.argv[2].split(",")]
# load the trainer definitions without running its main loop: exec the file up to the 'res = {}' line
src = open(os.path.join(here, "cvae_indep_train.py")).read().split("res = {}")[0]
sys.argv = [sys.argv[0], str(c)]
g = {"__file__": os.path.join(here, "cvae_indep_train.py")}
exec(compile(src, "cvae_indep_train_defs", "exec"), g)
out = []
for s in seeds:
    t0 = time.perf_counter()
    P = g["train"](10.0, seed=1000 + s)
    r = g["evaluate"](P)
    with torch.no_grad():
        Xte = torch.tensor((g["Yte"] - g["cmin"]) / (g["cmax"] - g["cmin"]))
        mu, ls = g["enc"](P, Xte, g["Cte"])
        eps = torch.randn(mu.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
        xh = g["dec"](P, mu + torch.exp(ls) * eps, g["Cte"])
        r["compute_mse_post"] = float((100 * torch.sqrt(((Xte - xh) ** 2).mean(1))).mean())
    r["seed"] = s; r["train_s"] = time.perf_counter() - t0
    out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(os.path.join(here, f"cvae_indep_seeds_c{c}_{seeds[0]}.json"), "w"), indent=1)
