# Dynamic pricing to reduce CO2

Course project (MIT 15.095, Machine Learning, Fall 2026): carbon-aware dynamic electricity tariffs for households, built
on the Low Carbon London (LCL) 2013 dynamic time-of-use trial.

The pipeline has these parts:
- a day-ahead grid carbon-intensity forecast;
- marginal emission factors;
- a demand-response simulator;
- Optimal Policy Trees that pick each day's tariff schedule under an average-price constraint;
- budget-aware trees;
- household-bill caps.

All emission effects are model-based estimates.

- **Results and methodology of the MVP integration:** [MVP_REPORT.md](MVP_REPORT.md)
- **Summary for the team:** [TEAM_HANDOFF.md](TEAM_HANDOFF.md)
- **Audit of the original implementation:** [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
- **Notebooks:** [project_implementation.ipynb](project_implementation.ipynb) (the pipeline in executable cells) and
  [project_walkthrough.ipynb](project_walkthrough.ipynb) (a tour of the saved results); see section 5.

## Repository layout

| Folder / file | What it is |
|---|---|
| `carbon_tariff_pipeline/` | Main project. Clement's scripts `01`–`06` and `lib/`: day-ahead carbon-intensity forecast, marginal emission factors (incl. the net-load variant), demand-response simulator, optimal policy trees. Results in `carbon_tariff_pipeline/report.html`; details in [its README](carbon_tariff_pipeline/README.md). The MVP adds scripts `07`–`12`: IAI trees, outputs, reproduction check, pacing, affordability, final summary. |
| `demand_response/` | Demand-response baseline, summary of the CVAE investigation, and all demand predictions in one shared format (`outputs/predictions.csv.gz`). Self-contained; see [its README](demand_response/README.md). |
| `CVAE_reproduction/` | Reproduction of Brégère & Bessa (2020), *Simulating tariff impact ... with conditional variational autoencoders*. It also builds the household data, clusters and trained CVAEs that the main pipeline uses. Summary notebook: `CVAE_reproduction/CVAE_reproduction.ipynb`; spec: `CVAE_reproduction/REPRO_SPEC.md`. |
| `run_mvp.py` | Single entry point for the MVP pipeline (section 3). It is the canonical pipeline behind the MVP results. |
| `tests/run_checks.py` | Automated checks: rule unit tests, preservation of the original results, exact reproduction, constraint compliance and consistency. |
| `results/` | MVP outputs: `tables/`, `figures/`, `mvp/outputs/`, `pacing/`, `affordability/`, `reproduction/`. `results/processed/`, `results/logs/` and the IAI fit folders are gitignored and regenerated. |
| `project_implementation.ipynb`, `project_walkthrough.ipynb` | Notebooks (section 5). |
| `final_report/` | LaTeX report (`main.tex` + `figures/`, also zipped for Overleaf). |
| `EDA_df_fuel_ckan.ipynb` | Exploration of the NESO generation-mix / carbon-intensity data against the LCL tariff schedule. |
| `test.ipynb` | Early look at the raw LCL CSV. |
| `Tariffs.xlsx` | LCL dynamic ToU tariff schedule for 2013 (included). |
| `project_proposal_ML.pdf`, `Energy_Household_Prescriptive_ML_Proposal_Revised.docx` | Project proposal. |

All results (JSON, CSV, figures, trained CVAE checkpoints in `CVAE_reproduction/outputs/chrono/models/`) are committed,
so the reports and figures can be read without running anything.

## 1. Setup

Python >= 3.12 with pandas >= 3. Clement tested with Python 3.14 on Windows; the MVP was run with Python 3.12.14 on
macOS arm64. The exact versions are listed in `requirements.txt`.

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

- `torch` is only needed to retrain the CVAEs (`CVAE_reproduction/`). The main pipeline and `run_mvp.py` run without it.
- Optional: Interpretable AI 3.2.2 with an academic licence, for the IAI Optimal Policy Trees, via the juliaup channel
  `1.12.6/IAI/3.2.2` (`julia +1.12.6/IAI/3.2.2`). Without it, `run_mvp.py --skip-iai` still runs the exact-solver trees.

Run Clement's scripts with `python -P scripts/<name>.py` from the folder that contains `scripts/`. The `-P` flag keeps
the script's own folder off `sys.path`; the scripts add the right folder themselves.

## 2. Data not in the repository

These files are too large for git; see `.gitignore`. Put them under exactly these names in **one** of these folders:
- the folder named by `$CARBON_RAW_DIR`;
- the **repository root** (Clement's layout);
- the folder that **contains** the repository (Panle's layout; the LCL file may also be in an `LCL_Data/` subfolder).

The code looks in that order and only reads the files.

| File | Size | Needed by | Source |
|---|---|---|---|
| `df_fuel_ckan.csv` | 63 MB | `carbon_tariff_pipeline` steps 1, 1b, 3; `run_mvp.py`; `project_implementation.ipynb`; `EDA_df_fuel_ckan.ipynb` | NESO *Historic generation mix & carbon intensity* (half-hourly, 2009 onwards): <https://www.neso.energy/data-portal/historic-generation-mix>. Keep the file name `df_fuel_ckan.csv`. |
| `CC_LCL-FullData.csv` | 8.5 GB | `CVAE_reproduction/scripts/01_build_lcl2013.py`, `project_implementation.ipynb`, `test.ipynb` | London Datastore, *SmartMeter energy consumption data in London households* (the single full CSV): <https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households> |
| `Small LCL Data/LCL-June2015v2_*.csv` | 8 GB (168 files) | `CVAE_reproduction/scripts/02_lcl_checks.py` only | Same London Datastore page (the same data split into 168 files). |
| `temp UK 2013-14.csv` | 330 MB | `CVAE_reproduction/scripts/03_conditions.py`, `run_mvp.py` (data-quality summary) | Copernicus CDS, *ERA5 hourly data on single levels*: 2 m temperature and 2 m dewpoint, 2013-2014, area N 54.5 / W -3.0 / S 50.75 / E 0.25, 0.25° grid, downloaded as GRIB and flattened to CSV with columns `time,latitude,longitude,number,step,surface,valid_time,d2m,t2m` (Kelvin). |

Some derived files are also git-ignored but needed:

| File | Size | Built by |
|---|---|---|
| `CVAE_reproduction/data/lcl2013_kwh_filled.npy` | 145 MB | `CVAE_reproduction/scripts/01_build_lcl2013.py`, or the `data` stage of `run_mvp.py`. It needs `CC_LCL-FullData.csv` and is the gap-filled 2013 matrix of the 2,064 selected households. If you don't want to download the 8.5 GB CSV, get a copy from a teammate. |
| `CVAE_reproduction/data/lcl2013_kwh.npy` | 390 MB | Same script (the matrix before gap filling). Used by `02_lcl_checks.py` and `project_implementation.ipynb`. |

`01_build_lcl2013.py` also writes `lcl2013_long.parquet` (git-ignored). Large intermediates go to `$LCL_WORK` (default:
`<system temp>/lcl_work`).

## 3. How to run

### MVP pipeline (about 13 minutes on a MacBook Pro)

```bash
.venv/bin/python run_mvp.py --reproduce
```

This runs every stage below. `--reproduce` also re-runs Clement's original Steps 1, 1b, 2 and 3 and compares them with
his committed outputs. On the first run, the `data` stage rebuilds the gitignored household matrix from the raw CSV
(about 1 minute). The MVP writes to `results/` and never overwrites Clement's committed outputs in
`carbon_tariff_pipeline/outputs/`.

```bash
.venv/bin/python tests/run_checks.py
```

This runs the automated checks; all must pass.

```bash
.venv/bin/python run_mvp.py --stages policy,iai,outputs
```

This runs selected stages only; their inputs must already exist.

| Stage | Script | Output |
|---|---|---|
| `check`, `data` | `CVAE_reproduction/scripts/01_build_lcl2013.py` (only if the matrix is missing) | raw data and environment checks; verified household matrix |
| `forecast` | `01_ci_forecast.py`, `01b_marginal_emissions.py` | `results/mvp/outputs/step1*`, `ci_forecasts_2013.npz`, `mef_2013.npz` |
| `demand` | `02_demand_response.py` | `step2_results.json`, `response_theta.npz` |
| `policy` | `03_policy.py --temperature lagged --tag mvp` (average CI; marginal as sensitivity) | `step3_results{_average}_mvp.json` |
| `iai` | `07_iai_opt.py` + `07_iai_fit.jl` | `iai_opt_results_average_mvp.json` |
| `outputs` | `08_mvp_outputs.py` | `results/tables/*`, figures 1–4, `results/processed/*.parquet`, sanity checks |
| `pacing` | `10_pacing_experiment.py` | `results/pacing/*`, figures 5–6 |
| `affordability` | `11_bill_constraint.py` | `results/affordability/*`, figure 7 |
| `summary` | `12_final_summary.py` (reads saved outputs only) | `results/tables/final_comparison.md`, figure 8 |
| `reproduce` (with `--reproduce`) | Clement's `01`, `01b`, `02`, `03` + `09_reproduction_check.py` | `results/reproduction/*` |

**Expected outputs.**
- **Figures:** `results/figures/fig1_*.png` … `fig8_*.png`.
- **Tables:** `results/tables/final_comparison.md`, `policy_comparison.md`, `data_quality.md` and `sanity_checks.md`.
- **Reproduction:** `results/reproduction/reproduction_check.md`. Clement's Steps 1b, 2 and 3 reproduce with 0
  mismatches. XGBoost forecasts differ slightly between Windows and macOS (test MAE 27.2 vs 27.6 g/kWh).
- **Determinism:** all runs use fixed seeds.

### Clement's original pipeline (about 30 minutes on CPU)

It needs `df_fuel_ckan.csv` and `CVAE_reproduction/data/lcl2013_kwh_filled.npy`; everything else it reads is committed.
These commands rewrite the committed outputs in `carbon_tariff_pipeline/outputs/`. Set `CARBON_OUT=<dir>` (and
`CARBON_FIG=<dir>`) to write elsewhere.

```bash
cd carbon_tariff_pipeline
python -P scripts/01_ci_forecast.py
python -P scripts/01b_marginal_emissions.py
python -P scripts/02_demand_response.py
python -P scripts/03_policy.py --metric marginal
python -P scripts/03_policy.py --metric average
python -P scripts/06_tree_check.py
python -P scripts/04_figures.py
python -P scripts/05_report.py
```

Re-running this from a clean clone (Nov 2026) reproduced the committed outputs, figures and `report.html`. The optional
comparison with Interpretable AI's Optimal Policy Trees (`06_iai_fit.jl`) is described in
[carbon_tariff_pipeline/README.md](carbon_tariff_pipeline/README.md); the MVP's IAI experiments are in `07_iai_opt.py`.

### Demand-response baseline

See [demand_response/README.md](demand_response/README.md).

### CVAE reproduction (several hours on CPU)

Track A uses the authors' prepared data (committed in `CVAE_reproduction/data/authors/`). Track B rebuilds everything
from the raw LCL and ERA5 files. Each script's docstring lists its inputs, outputs and options.

```bash
cd CVAE_reproduction
python -P scripts/01_build_lcl2013.py        # raw LCL CSV -> data/lcl2013_*.npy / .parquet
python -P scripts/02_lcl_checks.py           # data and alignment checks (uses Small LCL Data/)
python -P scripts/03_conditions.py           # ERA5 + tariffs -> data/conditions_ours.parquet
python -P scripts/04_clustering.py           # causality profiles, NMF + k-medoids clusters, Table 3
python -P scripts/05_build_trackB_data.py    # track B datasets in the authors' schema
python -P scripts/06_table3.py               # Table 3 with the identified Calinski-Harabasz variant
python -P scripts/10_trackA_cvae.py          # track A CVAE, 4 clusters x 50 restarts (~20 min with 4 workers)
python -P scripts/11_trackA_gam.py           # track A GAM benchmark
python -P scripts/20_scores.py --track A
python -P scripts/21_tariff_effects.py --track A
python -P scripts/30_trackB_models.py        # track B CVAE + GAM
python -P scripts/20_scores.py --track B
python -P scripts/21_tariff_effects.py --track B
python -P scripts/40_eta_sensitivity.py      # KL-weight (eta) sensitivity
python -P scripts/50_chrono_split.py         # chronological split; writes the CVAE checkpoints the main pipeline uses
```

The generated sample arrays (`outputs/**/*.npy`) are git-ignored, so `CVAE_reproduction.ipynb` needs
`10_trackA_cvae.py` and `11_trackA_gam.py` to have been run first.

The scripts under `CVAE_reproduction/outputs/verify/*/scripts/` and `outputs/trackB/aux_scripts/` are one-off
verification scripts kept as a record. They still contain absolute paths from the original working folder and are not
meant to be re-run as they are.

## 4. Report

`final_report/main.tex` is self-contained (figures in `final_report/figures/`, bibliography inline). Upload
`final_report/final_report_overleaf.zip` to Overleaf, or run `pdflatex main.tex` in `final_report/`. The MVP write-up is
in `MVP_REPORT.md`.

## 5. Notebooks

The notebooks run in the project `.venv`. `jupyterlab` and `ipykernel` come from `requirements.txt`;
`requirements-notebook.txt` lists the minimal runner (`ipykernel`, `nbformat`, `nbconvert`, `nbclient`) for
command-line execution. To work interactively, open a notebook in VS Code or Jupyter, select the `.venv` kernel and run
all cells from the top.

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
- **Raw data:** both modes read the raw data (the first rows of the LCL csv and the NESO file), located as in section 2.
- **Gitignored artefacts:** both modes need `CVAE_reproduction/data/lcl2013_kwh.npy` and `lcl2013_kwh_filled.npy`.
  Review Mode also needs the saved IAI fits under `results/mvp/outputs/iai_*_average_mvp/`. On a fresh clone, run
  `run_mvp.py --reproduce` once.
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
