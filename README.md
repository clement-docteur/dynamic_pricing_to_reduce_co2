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
- **Notebooks:** [project_implementation.ipynb](project_implementation.ipynb) (the pipeline written out in executable
  cells) and [project_walkthrough.ipynb](project_walkthrough.ipynb) (a read-only tour of the saved results); see
  section 4.

## Repository layout

| Path | Contents |
|---|---|
| `carbon_tariff_pipeline/` | Clement's pipeline (`scripts/01`–`06`, `lib/`), plus the MVP scripts `07`–`12` (IAI trees, outputs, reproduction check, pacing, affordability, final summary). Committed outputs in `outputs/` and `figures/` are Clement's originals. |
| `CVAE_reproduction/` | Clement's reproduction of Brégère & Bessa's CVAE simulator: data build, clustering and trained checkpoints. |
| `run_mvp.py` | Single entry point for the whole MVP (the canonical pipeline). |
| `project_implementation.ipynb` | Implementation notebook: preprocessing, forecasting, demand model, tariff optimisation, policy trees, budget-aware trees and bill caps as editable, executable code cells, checked against the pipeline's outputs. |
| `project_walkthrough.ipynb` | Read-only tour of the saved results, tables and figures. |
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

## 4. Notebooks

```bash
.venv/bin/pip install -r requirements-notebook.txt
```

This adds the Jupyter kernel and runner (`ipykernel`, `nbformat`, `nbconvert`, `nbclient`) to the project `.venv`. The
modelling packages come from `requirements.txt`. To work interactively, open a notebook in VS Code or Jupyter, select the
`.venv` kernel and run all cells from the top.

### `project_implementation.ipynb` — the pipeline in executable cells

The notebook writes out the ML pipeline step by step in ordinary code cells, so each step can be run, edited and
debugged:
1. data preprocessing (household selection, gap filling, cluster aggregation, tariff, temperature and carbon intensity);
2. carbon-intensity forecasting (features, Ridge / Lasso / random forest / XGBoost, hyper-parameter search, monthly
   refits);
3. the demand-response regression;
4. tariff optimisation (98 candidate schedules, menu selection, the day × schedule cost table, the feasibility rule);
5. Clement's exact policy tree and the IAI Optimal Policy Tree;
6. budget-aware trees and the pacing rule;
7. household-bill caps;
8. the final comparison.

Intermediate objects stay in memory for inspection: household matrices, `ci_frame`, the fitted models, the feature
matrix `X`, menus, cost tables, trees and the `results` / `final` tables.

**Faithful copies, not the canonical pipeline.** The functions in the notebook are copies of the logic in the `.py`
modules, with the same specifications, seeds and constraints. The `.py` scripts driven by `run_mvp.py` remain the
canonical pipeline, and all reported results come from them. Faithfulness is enforced by evidence, not by sharing code:
- each stage ends with `check(...)` cells that compare the notebook's result with the pipeline's saved outputs, and a
  mismatch stops the notebook;
- if a `.py` script changes later, the notebook is not updated automatically, and its checks will flag any difference.

**Two modes.** Review is the default. For Full, set `MODE = "full"` in the first code cell, or set
`PROJECT_NOTEBOOK_MODE=full`.

| | Review | Full |
|---|---|---|
| Preprocessing (selection, gap filling, aggregation, features) | runs | runs |
| Demand model, menu, exact policy trees, budget-aware and bill-penalised trees | trained live for the Nov–Dec window | trained live for all four rolling windows |
| Carbon forecaster | refits the selected model for Nov and Dec; earlier out-of-sample forecasts are loaded from the pipeline | full hyper-parameter search and all 12 monthly refits of the four models |
| IAI Optimal Policy Trees | reads the pipeline's saved IAI fits; selection and evaluation run in the notebook | refits IAI in Julia (temporary folder) |
| Raw 8.5 GB csv → household matrix | uses the pipeline's matrix | also rebuilds it in a temporary folder and compares |
| Checks against the pipeline | 54, all match | 138, all match |
| Runtime (MacBook Pro) | about 20 s | about 5 min |

Neither mode writes to the repository. Full Mode uses temporary folders, and the last cell verifies that no repository
file changed. Cells are labelled `TRAINING` or `EVALUATION`, and Review Mode prints `[review]` wherever it loads a saved
artefact instead of training.

Command-line runs:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 project_implementation.ipynb
```

```bash
PROJECT_NOTEBOOK_MODE=full .venv/bin/python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir /tmp --output project_implementation_full.ipynb project_implementation.ipynb
```

The first refreshes the Review Mode outputs saved in the notebook. The second writes the Full Mode run to `/tmp`, so the
committed copy keeps its Review outputs.

**Requirements and limitations.**
- **Raw data:** both modes read the raw data (the first rows of the LCL csv and the NESO file) from `$CARBON_RAW_DIR`.
- **Gitignored artefacts:** both modes need the household matrices `CVAE_reproduction/data/lcl2013_kwh.npy` and
  `lcl2013_kwh_filled.npy`. Review Mode also needs the saved IAI fits under `results/mvp/outputs/iai_*_average_mvp/`.
  On a fresh clone, run `.venv/bin/python run_mvp.py --reproduce` once to create them.
- **Julia/IAI:** Full Mode needs Julia with Interpretable AI to refit the IAI trees. Without it, it uses the saved IAI
  fits in all three IAI sections; only the first IAI section prints a `[review]` note.
- **What stays in external scripts:**
  - the one-pass DuckDB conversion of the raw csv (`01_build_lcl2013.py`; called in Full Mode);
  - Clement's behavioural clustering and ERA5 temperature interpolation (`04_clustering.py`, `03_conditions.py`; their
    outputs are loaded as data);
  - the IAI solver (`07_iai_fit.jl`);
  - the CVAE, which is not retrained;
  - bootstrap intervals and sensitivity runs (`run_mvp.py` produces them).
- **Platform:** XGBoost forecasts are platform-dependent (Windows vs macOS). The notebook's checks are against the
  pipeline run on the same machine.

### `project_walkthrough.ipynb` — results tour

This notebook loads the saved tables and figures and walks through the findings. It trains nothing and runs in a few
seconds.
