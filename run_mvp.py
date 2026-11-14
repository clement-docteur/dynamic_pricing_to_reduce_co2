"""Single reproducible entry point for the carbon-aware tariff MVP.

    .venv/bin/python run_mvp.py                      # full MVP (about 10 min on a MacBook Pro; IAI adds a few minutes)
    .venv/bin/python run_mvp.py --reproduce          # also re-run Clement's original Steps 2-3 and compare them with the
                                                     # committed outputs (carbon_tariff_pipeline/outputs/)
    .venv/bin/python run_mvp.py --stages policy,iai,outputs   # run selected stages only (inputs must exist)

Stages (in order):
  check     raw data present and readable, Python packages importable
  data      rebuild CVAE_reproduction/data/lcl2013_kwh_filled.npy from the raw LCL csv if it is missing (DuckDB, ~1 min),
            then verify it: cluster means must reproduce the committed power_consump_ours.parquet to < 1e-6 kWh
  forecast  Step 1 (next-day carbon-intensity forecast) and Step 1b (marginal emission factors)
  demand    Step 2 (demand-response simulator: fit, bootstrap, held-out validation, CVAE loader check)
  policy    Step 3 with the MVP settings: leak-free temperature feature; primary run with average carbon intensity,
            sensitivity run with marginal emission factors
  iai       Step 4: Interpretable AI Optimal Policy Trees (depths 1-3) vs the exact tree, main fold + rolling folds
  outputs   data-quality summary, processed Parquet, day x action cost matrix, comparison tables, 4 figures, sanity checks
  pacing    additional experiment: original vs budget-pacing feasibility rule, and remaining budget as a state variable
            (budget-aware trees), on all four rolling windows (results/pacing/, figures 5-6); needs the iai stage
  affordability  additional experiment: budget-aware trees with an expected-bill cap (0/5/10 %) or penalty, all four
            windows (results/affordability/, figure 7); needs the pacing stage
  summary   final cross-window comparison from the saved outputs (results/tables/final_comparison.md, figure 8)
  reproduce (only with --reproduce) re-run Clement's Steps 1 and 1b on this machine (results/reproduction/step1_mac) and his
            Steps 2-3 on his committed Step-1 inputs (results/reproduction/outputs), then compare with his committed outputs

Raw data are read from $CARBON_RAW_DIR (if set), else the repository root, else the folder that contains the repository,
and never modified.
MVP outputs: results/mvp/outputs (pipeline JSON/NPZ), results/figures, results/tables, results/processed (gitignored),
results/logs. Clement's committed outputs in carbon_tariff_pipeline/outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPE = ROOT / "carbon_tariff_pipeline"
REPRO = ROOT / "CVAE_reproduction"
RAW_DIRS = ([Path(os.environ["CARBON_RAW_DIR"])] if os.environ.get("CARBON_RAW_DIR") else []) + [ROOT, ROOT.parent]


def raw_file(*names):
    """First existing raw-data file among RAW_DIRS x names (else the first candidate)."""
    return next((d / n for d in RAW_DIRS for n in names if (d / n).exists()), RAW_DIRS[0] / names[0])


RES = ROOT / "results"
MVP_OUT = RES / "mvp" / "outputs"
LOGS = RES / "logs"
STAGES = ["check", "data", "forecast", "demand", "policy", "iai", "outputs", "pacing", "affordability", "summary"]
RAW_FILES = {"LCL smart meters": raw_file("CC_LCL-FullData.csv", "LCL_Data/CC_LCL-FullData.csv"),
             "NESO generation mix": raw_file("df_fuel_ckan.csv"), "ERA5 temperature": raw_file("temp UK 2013-14.csv"),
             "Tariffs": raw_file("Tariffs.xlsx")}
MVP_FLAGS = ["--temperature", "lagged", "--tag", "mvp"]


def say(msg):
    print(f"[run_mvp {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def env_for(out_dir, fig_dir):
    e = dict(os.environ)
    e.update(CARBON_OUT=str(out_dir), CARBON_FIG=str(fig_dir), PYTHONHASHSEED="0")      # raw data: same lookup in every script
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        e.setdefault(v, "4")
    return e


def run(name, args, cwd=PIPE, env=None, parallel=False):
    """Run `python -P <args>` in cwd with its output in results/logs/<name>.log; return the Popen if parallel."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log = open(LOGS / f"{name}.log", "w")
    say(f"{name}: python -P {' '.join(args)}")
    p = subprocess.Popen([sys.executable, "-P", *args], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if parallel:
        return p
    wait(name, p)


def wait(name, p):
    if p.wait() != 0:
        sys.exit(f"stage step '{name}' failed (exit {p.returncode}); see {LOGS / (name + '.log')}")
    say(f"{name}: ok")


# ------------------------------------------------------------------------------------------------ stages
def stage_check(a):
    bad = [f"{k}: {p}" for k, p in RAW_FILES.items() if not (p.is_file() and os.access(p, os.R_OK))]
    if bad:
        sys.exit("missing or unreadable raw data (put it in $CARBON_RAW_DIR, the repository root or its parent folder):\n  "
                 + "\n  ".join(bad))
    for k, p in RAW_FILES.items():
        say(f"raw data ok: {k} ({p.stat().st_size / 1e6:,.0f} MB)")
    missing = []
    for m in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "duckdb", "pyarrow", "matplotlib", "openpyxl"):
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    if missing:
        sys.exit(f"missing Python packages {missing}: pip install -r requirements.txt")
    say(f"python {sys.version.split()[0]} with all required packages")


def stage_data(a):
    filled = REPRO / "data" / "lcl2013_kwh_filled.npy"
    if not filled.exists():
        run("data_01_build_lcl2013", ["scripts/01_build_lcl2013.py", "--threads", "8"], cwd=REPRO, env=env_for(MVP_OUT, RES / "figures"))
        for f in ("lcl2013_build_stats.json", "lcl2013_fill_stats.json"):   # keep the committed (path-free) versions
            subprocess.run(["git", "checkout", "--", f"CVAE_reproduction/data/{f}"], cwd=ROOT, capture_output=True)
    else:
        say(f"data: {filled.name} present, not rebuilt")
    code = ("import sys; sys.path.insert(0, '.'); import numpy as np\n"
            "from lib.common import *\n"
            "hh = load_households(); s, w, Yc, _ = cluster_aggregates(hh)\n"
            "pc = read_parquet(REPRO / 'data' / 'power_consump_ours.parquet')\n"
            "d = max(np.abs(pc[pc.cluster == c].sort_values('date')[cols('consumption')].to_numpy(float) - Yc[c]).max() for c in range(4))\n"
            "assert np.isfinite(hh['Y_tou']).all() and (hh['Y_tou'] >= 0).all() and d < 1e-6, d\n"
            "print(f'households ToU {hh[\"Y_tou\"].shape[0]}, Std {hh[\"Y_std\"].shape[0]}; max |cluster mean - committed| = {d:.2e} kWh')\n")
    r = subprocess.run([sys.executable, "-P", "-c", code], cwd=PIPE, env=env_for(MVP_OUT, RES / "figures"), capture_output=True, text=True)
    if r.returncode:
        sys.exit("household matrix verification failed:\n" + r.stdout + r.stderr)
    say("data verified: " + r.stdout.strip())


def stage_forecast(a):
    e = env_for(MVP_OUT, RES / "mvp" / "figures")
    run("mvp_01_ci_forecast", ["scripts/01_ci_forecast.py"], env=e)
    run("mvp_01b_marginal_emissions", ["scripts/01b_marginal_emissions.py"], env=e)


def stage_demand(a):
    run("mvp_02_demand_response", ["scripts/02_demand_response.py"], env=env_for(MVP_OUT, RES / "mvp" / "figures"))


def stage_policy(a):
    e = env_for(MVP_OUT, RES / "mvp" / "figures")
    ps = [("mvp_03_policy_average", run("mvp_03_policy_average", ["scripts/03_policy.py", "--metric", "average", *MVP_FLAGS], env=e, parallel=True)),
          ("mvp_03_policy_marginal", run("mvp_03_policy_marginal", ["scripts/03_policy.py", "--metric", "marginal", *MVP_FLAGS], env=e, parallel=True))]
    for n, p in ps:
        wait(n, p)


def stage_iai(a):
    if a.skip_iai:
        say("iai: skipped (--skip-iai); the exact tree is still evaluated by the outputs stage")
        return
    if shutil.which(a.julia.split()[0]) is None:
        say(f"iai: '{a.julia}' not found; skipping IAI (install Interpretable AI or pass --julia)")
        return
    run("mvp_07_iai_opt", ["scripts/07_iai_opt.py", "--metric", "average", *MVP_FLAGS, "--folds", "5,7,9,11", "--julia", a.julia],
        env=env_for(MVP_OUT, RES / "mvp" / "figures"))


def stage_outputs(a):
    run("mvp_08_outputs", ["scripts/08_mvp_outputs.py"], env=env_for(MVP_OUT, RES / "figures"))


def stage_pacing(a):
    args = ["scripts/10_pacing_experiment.py", "--julia", a.julia] + (["--skip-iai"] if a.skip_iai else [])
    run("mvp_10_pacing", args, env=env_for(MVP_OUT, RES / "figures"))


def stage_affordability(a):
    solvers = "exact" if a.skip_iai else "iai,exact"
    run("mvp_11_bill_constraint", ["scripts/11_bill_constraint.py", "--julia", a.julia, "--solvers", solvers],
        env=env_for(MVP_OUT, RES / "figures"))


def stage_summary(a):
    run("mvp_12_final_summary", ["scripts/12_final_summary.py"], env=env_for(MVP_OUT, RES / "figures"))


def stage_reproduce(a):
    mac1 = RES / "reproduction" / "step1_mac"                # Clement's Steps 1 and 1b re-run here (incl. the net-load factor)
    mac1.mkdir(parents=True, exist_ok=True)
    e1 = env_for(mac1, RES / "reproduction" / "figures")
    run("repro_01_ci_forecast", ["scripts/01_ci_forecast.py"], env=e1)
    run("repro_01b_marginal_emissions", ["scripts/01b_marginal_emissions.py"], env=e1)
    out = RES / "reproduction" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("step1_metrics.json", "step1b_mef.json", "ci_forecasts_2013.npz", "mef_2013.npz"):
        shutil.copy2(PIPE / "outputs" / f, out / f)        # Clement's committed Step-1 outputs (see MVP_REPORT.md)
    e = env_for(out, RES / "reproduction" / "figures")
    run("repro_02_demand_response", ["scripts/02_demand_response.py"], env=e)
    ps = [(n, run(n, ["scripts/03_policy.py", "--metric", m], env=e, parallel=True))
          for n, m in (("repro_03_marginal", "marginal"), ("repro_03_average", "average"))]
    for n, p in ps:
        wait(n, p)
    run("repro_09_check", ["scripts/09_reproduction_check.py"], env=e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="all", help=f"comma-separated subset of {STAGES} (default all)")
    ap.add_argument("--reproduce", action="store_true", help="also reproduce Clement's original Steps 2-3 and compare")
    ap.add_argument("--skip-iai", action="store_true")
    ap.add_argument("--julia", default="julia +1.12.6/IAI/3.2.2", help="command that starts Julia with IAI")
    a = ap.parse_args()
    todo = STAGES if a.stages == "all" else [s.strip() for s in a.stages.split(",")]
    unknown = set(todo) - set(STAGES) - {"reproduce"}
    if unknown:
        sys.exit(f"unknown stages {unknown}")
    MVP_OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for s in STAGES + ["reproduce"]:
        if s in todo or (s == "reproduce" and a.reproduce):
            say(f"=== stage {s}")
            globals()[f"stage_{s}"](a)
    say(f"finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
