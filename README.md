# Dynamic pricing to reduce CO2

Course project (MIT, Machine Learning, Fall 2026): carbon-aware dynamic electricity tariffs for households, built on the
Low Carbon London (LCL) 2013 dynamic time-of-use trial.

| Folder / file | What it is |
|---|---|
| `carbon_tariff_pipeline/` | Main project: day-ahead carbon-intensity forecast, marginal emission factors, demand-response simulator, optimal policy trees. Results in `carbon_tariff_pipeline/report.html`; details in [its README](carbon_tariff_pipeline/README.md). |
| `CVAE_reproduction/` | Reproduction of Brégère & Bessa (2020), *Simulating tariff impact ... with conditional variational autoencoders*. It also builds the household data, clusters and trained CVAEs that the main pipeline uses. Summary notebook: `CVAE_reproduction/CVAE_reproduction.ipynb`; spec: `CVAE_reproduction/REPRO_SPEC.md`. |
| `final_report/` | LaTeX report (`main.tex` + `figures/`, also zipped for Overleaf). |
| `EDA_df_fuel_ckan.ipynb` | Exploration of the NESO generation-mix / carbon-intensity data against the LCL tariff schedule. |
| `test.ipynb` | Early look at the raw LCL CSV. |
| `Tariffs.xlsx` | LCL dynamic ToU tariff schedule for 2013 (included). |
| `project_proposal_ML.pdf`, `Energy_Household_Prescriptive_ML_Proposal_Revised.docx` | Project proposal. |

All results (JSON, CSV, figures, trained CVAE checkpoints in `CVAE_reproduction/outputs/chrono/models/`) are committed,
so the reports and figures can be read without running anything.

## Setup

Python 3.14 (tested with the versions in `requirements.txt`; pandas >= 3 is required).

```bash
pip install -r requirements.txt
```

Run every script with `python -P scripts/<name>.py` from the folder that contains `scripts/`
(`-P` keeps the script's own folder off `sys.path`; the scripts add the right folder themselves).

## Data not in the repository

These files are too large for git (see `.gitignore`). Put them in the **repository root** under exactly these names.

| File | Size | Needed by | Source |
|---|---|---|---|
| `df_fuel_ckan.csv` | 63 MB | `carbon_tariff_pipeline` steps 1, 1b, 3; `EDA_df_fuel_ckan.ipynb` | NESO *Historic generation mix & carbon intensity* (half-hourly, 2009 onwards): <https://www.neso.energy/data-portal/historic-generation-mix>. Keep the file name `df_fuel_ckan.csv`. |
| `CC_LCL-FullData.csv` | 8.5 GB | `CVAE_reproduction/scripts/01_build_lcl2013.py`, `test.ipynb` | London Datastore, *SmartMeter energy consumption data in London households* (the single full CSV): <https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households> |
| `Small LCL Data/LCL-June2015v2_*.csv` | 8 GB (168 files) | `CVAE_reproduction/scripts/02_lcl_checks.py` only | Same London Datastore page (the same data split into 168 files). |
| `temp UK 2013-14.csv` | 330 MB | `CVAE_reproduction/scripts/03_conditions.py` | Copernicus CDS, *ERA5 hourly data on single levels*: 2 m temperature and 2 m dewpoint, 2013-2014, area N 54.5 / W -3.0 / S 50.75 / E 0.25, 0.25° grid, downloaded as GRIB and flattened to CSV with columns `time,latitude,longitude,number,step,surface,valid_time,d2m,t2m` (Kelvin). |

One derived file is also git-ignored but needed by the main pipeline:

| File | Size | Built by |
|---|---|---|
| `CVAE_reproduction/data/lcl2013_kwh_filled.npy` | 145 MB | `CVAE_reproduction/scripts/01_build_lcl2013.py` (needs `CC_LCL-FullData.csv`; the gap-filled 2013 matrix of the 2,064 selected households). If you don't want to download the 8.5 GB CSV, get a copy of this file from a teammate. |

`01_build_lcl2013.py` also writes `lcl2013_kwh.npy` and `lcl2013_long.parquet` (git-ignored, used by `02_lcl_checks.py`).
Large intermediates go to `$LCL_WORK` (default: `<system temp>/lcl_work`).

## How to run

### Main pipeline (about 30 minutes on CPU)

Needs `df_fuel_ckan.csv` and `CVAE_reproduction/data/lcl2013_kwh_filled.npy`; everything else it reads is committed.

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

Re-running this from a clean clone (Nov 2026) reproduced the committed outputs, figures and `report.html`.
The optional comparison with Interpretable AI's Optimal Policy Trees (`06_iai_fit.jl`) is described in
[carbon_tariff_pipeline/README.md](carbon_tariff_pipeline/README.md).

### CVAE reproduction (several hours on CPU)

Track A uses the authors' prepared data (committed in `CVAE_reproduction/data/authors/`); track B rebuilds everything from
the raw LCL and ERA5 files. Each script's docstring lists its inputs, outputs and options.

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

## Report

`final_report/main.tex` is self-contained (figures in `final_report/figures/`, bibliography inline): upload
`final_report/final_report_overleaf.zip` to Overleaf, or run `pdflatex main.tex` in `final_report/`.
