# dynamic_pricing_to_reduce_co2

Interpretable carbon-aware electricity pricing on the Low Carbon London 2013 dynamic-tariff trial (MIT 15.095).

The pipeline has five parts:
- a day-ahead grid carbon-intensity forecast;
- a demand-response simulator;
- Optimal Policy Trees that pick each day's tariff schedule under an average-price constraint;
- budget-aware trees;
- household-bill caps.

All emission effects are model-based estimates.

- **Results and methodology:** [MVP_REPORT.md](MVP_REPORT.md)
- **Summary for the team:** [TEAM_HANDOFF.md](TEAM_HANDOFF.md)
- **Audit of the original implementation:** [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)

## Repository layout

| Path | Contents |
|---|---|
| `carbon_tariff_pipeline/` | Clement's pipeline (`scripts/01`–`06`, `lib/`), plus the MVP scripts `07`–`12` (IAI trees, outputs, reproduction check, pacing, affordability, final summary). Committed outputs in `outputs/` and `figures/` are Clement's originals. |
| `CVAE_reproduction/` | Clement's reproduction of Brégère & Bessa's CVAE simulator: data build, clustering and trained checkpoints. |
| `run_mvp.py` | Single entry point for the whole MVP. |
| `tests/run_checks.py` | Automated checks: rule unit tests, preservation, reproduction, constraint compliance and consistency. |
| `results/` | MVP outputs: `tables/`, `figures/`, `mvp/outputs/`, `pacing/`, `affordability/`, `reproduction/`. `results/processed/` and `results/logs/` are gitignored and regenerated. |
| `final_report/` | LaTeX report draft. |

## 1. Environment

The pipeline needs Python 3.12. It was run on macOS arm64; Windows also works (Clement's original runs).

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

Optional: Interpretable AI 3.2.2 with an academic licence, for the IAI Optimal Policy Trees, via the juliaup channel
`1.12.6/IAI/3.2.2` (`julia +1.12.6/IAI/3.2.2`). Without IAI, pass `--skip-iai`. The exact-solver trees still run, but the
IAI-based outputs are not produced.

## 2. Raw data (not in the repository)

The pipeline reads these from `$CARBON_RAW_DIR`. By default that is the folder that **contains** this repository. They
are never modified.

| File | Content | Size |
|---|---|---|
| `LCL_Data/CC_LCL-FullData.csv` | Low Carbon London smart meters, 2011–2014 (167.9 M rows) | 8.5 GB |
| `df_fuel_ckan.csv` | NESO historic generation mix and carbon intensity (2009–) | 63 MB |
| `temp UK 2013-14.csv` | ERA5 2 m temperature, hourly, UK grid | 330 MB |
| `Tariffs.xlsx` | 2013 dynamic TOU schedule | 0.2 MB |

## 3. Run

```bash
.venv/bin/python run_mvp.py --reproduce
```

This runs every stage and also re-runs Clement's original Steps 2–3. It takes about 11 minutes on a MacBook Pro. The
first run also rebuilds the gitignored household matrix from the raw CSV (about 1 minute).

```bash
.venv/bin/python tests/run_checks.py
```

This runs the 20 automated checks; all must pass.

```bash
.venv/bin/python run_mvp.py --stages policy,iai,outputs
```

This runs selected stages only; their inputs must already exist.

| Stage | Script | Output |
|---|---|---|
| `check`, `data` | `CVAE_reproduction/scripts/01_build_lcl2013.py` (only if the matrix is missing) | raw data and environment checks; verified household matrix |
| `forecast` | `01_ci_forecast.py`, `01b_marginal_emissions.py` | `results/mvp/outputs/step1*`, `ci_forecasts_2013.npz` |
| `demand` | `02_demand_response.py` | `step2_results.json`, `response_theta.npz` |
| `policy` | `03_policy.py --temperature lagged --tag mvp` (average CI; marginal as sensitivity) | `step3_results{_average}_mvp.json` |
| `iai` | `07_iai_opt.py` + `07_iai_fit.jl` | `iai_opt_results_average_mvp.json` |
| `outputs` | `08_mvp_outputs.py` | `results/tables/*`, figures 1–4, `results/processed/*.parquet`, sanity checks |
| `pacing` | `10_pacing_experiment.py` | `results/pacing/*`, figures 5–6 |
| `affordability` | `11_bill_constraint.py` | `results/affordability/*`, figure 7 |
| `summary` | `12_final_summary.py` (reads saved outputs only) | `results/tables/final_comparison.md`, figure 8 |
| `reproduce` (with `--reproduce`) | Clement's `02`/`03` + `09_reproduction_check.py` | `results/reproduction/*` |

**Expected outputs.**
- **Figures:** `results/figures/fig1_*.png` … `fig8_*.png`.
- **Tables:** `results/tables/final_comparison.md`, `policy_comparison.md`, `data_quality.md` and `sanity_checks.md`
  (20/20 pass).
- **Reproduction:** `results/reproduction/reproduction_check.md` (0 mismatches for Steps 2–3).
- **Determinism:** all runs use fixed seeds. A full re-run reproduced every committed result file byte for byte.
  XGBoost forecasts differ slightly between Windows and macOS (test MAE 27.2 vs 27.6 g/kWh).

Clement's original pipeline can still be run as documented in `carbon_tariff_pipeline/README.md`. Set
`CARBON_OUT=<dir>` to avoid overwriting the committed outputs.
