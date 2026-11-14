# Implementation Status — audit of the existing repository

Audit date: 2026-11-14. Branch: `feature/mvp-integration` (clean, 2 commits: `043a03b`, `6c22f28`).
Auditor: Claude Code, on behalf of Panle Chen. Nothing in the repository was modified apart from adding this file.

## Status after MVP integration (2026-11-14)

All P0/P1 items below are done; see `MVP_REPORT.md`. Portable paths, `requirements.txt` and `run_mvp.py` added;
household matrix rebuilt and verified; Clement's Steps 2-3 reproduce exactly (`results/reproduction/reproduction_check.md`);
average CI is the primary metric (marginal = sensitivity); temperature leakage fixed (`--temperature lagged`); IAI Optimal
Policy Trees fitted at depths 1-3 on four folds (`scripts/07_iai_opt.py`); four required figures, data-quality summary,
cost matrix and 20 sanity checks generated (`scripts/08_mvp_outputs.py`). Later additions: pacing rule and budget-aware IAI trees
(`scripts/10_pacing_experiment.py`), household-bill caps (`scripts/11_bill_constraint.py`), final summary
(`scripts/12_final_summary.py`) and automated checks (`tests/run_checks.py`); see `TEAM_HANDOFF.md`.
The audit below is kept as written before the work.

## TL;DR

Clement has already built a **complete, documented end-to-end pipeline** (`carbon_tariff_pipeline/`), with results,
figures, an HTML report and a LaTeX final-report draft (`final_report/main.tex`). It was built and run **only on
Clement's Windows machine**. On this Mac it **cannot run yet**:

1. The core household matrix `CVAE_reproduction/data/lcl2013_kwh_filled.npy` is gitignored and absent; it must be
   rebuilt from `../LCL_Data/CC_LCL-FullData.csv` (8.5 GB) by `CVAE_reproduction/scripts/01_build_lcl2013.py`.
2. `CVAE_reproduction/lib/lcl_data.py` hard-codes Windows paths (`C:/Users/cleme/...`) for the raw CSV and scratch.
3. `carbon_tariff_pipeline/lib/common.py` reads `df_fuel_ckan.csv` from the repo root; here it lives in the parent folder.
4. No Python environment has the full stack: `~/.venvs/general-py312` has numpy/pandas/scipy/sklearn/xgboost/
   matplotlib/statsmodels but **not duckdb, pyarrow or openpyxl** (all required). No `requirements.txt` exists.
5. There is no single entry point (`run_mvp.py`) — scripts are run one by one.

So the fastest path to the MVP is **not** to rewrite anything: make the existing pipeline portable, rebuild the one
missing input, re-run it here, verify the numbers reproduce, and then fill the specific gaps listed below.

## Component table

| Component | Existing implementation | Status | Evidence | Missing work | Priority |
|---|---|---|---|---|---|
| **Data processing** | `CVAE_reproduction/scripts/01_build_lcl2013.py` (DuckDB scan of the 167.9 M-row CSV → 2013 matrix; 1,032 ToU households with >95 % completeness + 1,032 sampled Std; documented gap-filling rule), `03_conditions.py` (ERA5 temperature → slot-aligned conditions), `lib/tariffs.py` (tariff alignment). Committed outputs: `conditions_ours.parquet`, `clusters_run_a.parquet`, `selection.parquet`, `lcl2013_households.parquet`, build/fill stats JSON. | **Implemented and run (Windows); not runnable here** | `lcl2013_build_stats.json`: 167,932,474 rows, 5,566 households (4,443 Std / 1,123 ToU), 0 unparseable timestamps, 12 off-grid rows dropped in 2013, DST checked (LCL timestamps are GMT, no gap/duplication). REPRO_SPEC §7: tariff alignment verified by event study; authors' schedule preferred over `Tariffs.xlsx` (differs on 17 days). | Rebuild `lcl2013_kwh_filled.npy` (gitignored, ~145 MB); remove hard-coded Windows paths; verify the rebuild against the built-in checksum (cluster means reproduce `power_consump_ours.parquet` to < 4e-7 kWh). Write a short data-quality summary and Parquet copies of the processed panel as the prompt requests. | **P0** (blocks everything) |
| **Carbon forecasting** | `carbon_tariff_pipeline/scripts/01_ci_forecast.py`: day-ahead (decision at 12:00 GMT on d-1, horizon 12–36 h) forecast of 48 half-hourly CI values; Ridge, Lasso, RF, XGBoost vs 3 persistence baselines; NESO 2009–2013; monthly expanding refits so every 2013 day has an out-of-sample forecast; weather experiment flagged as retrospective (realised ERA5 = perfect forecast). `01b_marginal_emissions.py`: marginal emission factors. | **Implemented, run, outputs committed; not re-run here** | `outputs/step1_metrics.json`: selected XGBoost (by val). Val MAE/RMSE 24.0/31.5 g/kWh; test (Nov–Dec) 27.2/35.0 vs persistence 28.4/38.6. RF is slightly better on test (26.0/34.3). Weather adds nothing (val RMSE 32.62 → 32.56). `outputs/ci_forecasts_2013.npz` present. | Re-run on Mac and confirm metrics reproduce. Note honestly that the gain over persistence on the test period is small (MAE −4 %). | P1 |
| **CVAE demand simulation** | `CVAE_reproduction/` — full reproduction of Brégère & Bessa (2020): torch CVAE (`lib/cvae_torch.py`), GAM baseline, NMF + k-medoids clustering, scores, tariff-effect analysis. `scripts/50_chrono_split.py` retrains on the chronological split (train Jan–Aug). **320 trained checkpoints** in `outputs/chrono/models/` (2 tracks × 4 clusters × 2 variants × 20 restarts) + 568 files in `outputs/trackA/models`. Numpy decoder in `carbon_tariff_pipeline/lib/demand.CVAEEnsemble` (no torch needed). | **Trained and usable; demoted to sensitivity analysis** | `chrono_results.json`: on Nov–Dec, CVAE is not better than a naive baseline in several clusters, has energy bias of −13 % to −30 %, and wrong-sign tariff effects (e.g. cluster 0 High +2.8 %). `step2_results.json → cvae_loader_check`: numpy decoder reproduces torch RMSE exactly. Accepts arbitrary 48-slot Low/High schedules. | None for the MVP (no retraining). Keep it as the "alternative demand-response assumption" sensitivity — already wired into Step 3. | P2 |
| **Demand response (main simulator)** | `carbon_tariff_pipeline/lib/demand.py: ResponseModel` + `scripts/02_demand_response.py`: per-cluster ridge regression of log(ToU cluster mean / flat-tariff Std mean) on slot×day-type intercepts, temperature×band, and tariff terms (in-block High/Low by time-of-day band + pre/post-event edge terms → captures conservation, pre-Low dip, post-Low rebound, pre-High). Difference-in-differences against the Std control group. Counterfactual: `y(z) = y_obs · exp(m(z) − m(z_hist))`, depending on the **whole** day schedule. Spec chosen on Sep–Oct; 200-draw weekly-block bootstrap. | **Implemented, validated on held-out events, run (Windows)** | `step2_results.json`: chosen spec `main` (val RMSE 0.0803 vs 0.0806 full / 0.0804 in-block). Held-out check: e.g. cluster 0 in-block Low observed +3.8 % vs model +3.6 %; in-block High observed −1.7 % vs model −4.5 % (model overstates High response there). Peak-band High effect ≈ −5 % per cluster. | Re-run here. Make the validation figure required by the prompt (predicted vs observed profiles, daily totals, H/N/L-period demand, peak). State that coefficients are observational, not causal. | P1 |
| **Candidate tariffs** | `lib/policy.candidate_pool()`: ~150 single-block schedules (High 2/3/4 h between 01–20 h; Low 3/6 h; evening-High+night-Low combos), prices 3.99/11.76/67.20 p/kWh. `select_menu()`: greedy forward selection of ≤ 6 schedules on **training days only**. | **Implemented, run** | Selected menu (main fold): Normal all day (11.76), High 17:00–20:00 (18.69), High 18:00–20:00 (16.38), Low 02:00–05:00 (10.79). Every menu item's price is checked against the target in `feasibility_rule` (all-Normal day ≤ target asserted). | Report the menu's time-average prices and a feasibility table explicitly in the MVP report. | P2 |
| **Counterfactual costs** | `03_policy.costs()`: day × action matrix of emissions with realised factor (`E`), decision-time forecast factor + consumption forecast (`Ef`), perfect-CI variant, and price paid. | **Implemented, run** | Used for menu selection, tree fitting, direct optimisation; formula matches the proposal's Ĉ_d(z) = (1/1000) Σ_t CÎ_dt ŷ_dt(z). | Not saved as a standalone table. Export the day-by-action cost matrix (features, action, emissions, avg price, bill, kWh, peak) to Parquet as the prompt asks. | P1 |
| **Direct optimization** | `03_policy.run_policies()`: per-day argmin of forecast cost + λ·price (λ tuned on validation), then the online feasibility rule; also a perfect-CI variant, a budget-aware variant and a perfect-hindsight oracle (labelled as such). | **Implemented, run** | Test (Nov–Dec): −0.90 % emissions (marginal factors), identical schedules to the tree. | Re-run here. | P2 |
| **Optimal Policy Tree** | `lib/policy.PolicyTree`: a hand-written **exact prescriptive tree** (minimises summed model-based action costs, not a CART imitation) of depth 1–3, exhaustive thresholds at the last level and 12 quantile thresholds above; depth and λ chosen on validation; refit on train+val. Budget-aware variant adds remaining-budget as a feature. `scripts/06_tree_check.py` + `06_iai_fit.jl` prepared to compare with Interpretable AI's `OptimalTreePolicyMinimizer`, **never run** (IAI blocked on Windows). | **Implemented, run; IAI comparison pending** | Selected depth 1 (val emissions 271.60 vs 271.67/271.65 kg for depth 2/3). Split: "recent consumption ≤ 0.171 kWh/half-hour → Normal, else High 17:00–20:00". On test it prescribes High 17–20 on **all 61 days**; the feasibility rule cancels 45 of them. **IAI 3.2.2 runs on this Mac** (juliaup channel `1.12.6/IAI/3.2.2`, academic licence `~/iai.lic` accepted; `OptimalTreePolicyMinimizer` reaches the fit stage). `06_iai_fit.jl` needs a one-line fix: `using IAI` fails because the system image preloads `IAI` — drop it from the `using` line. | (a) If IAI runs here, fit the true OPT on the exported inputs and compare. (b) Document that the main tree is effectively "High 17–20 while budget allows", split on a consumption-level proxy, not a carbon feature. (c) Leakage: tree feature "temperature, day mean" is the **realised** ERA5 temperature of day d — not available at decision time. Not used by the chosen tree, but must be replaced by a lagged value or labelled. | P1 |
| **Evaluation** | `03_policy.py`: Historical TOU, carbon-only rule, tree, direct opt, perfect-CI direct opt, oracle, budget-aware tree/direct. Metrics: emissions change, mean announced price, price paid, kWh, daily peak, bill change distribution (p10/median/p90), adjustments. Sensitivities: 5 price targets (incl. 13.517, 14.228, test-matched 12.795), 5 demand worlds (×0.5, ×1.5, in-block, full, CVAE), bootstrap CIs, rolling 4-window evaluation, marginal vs average CI accounting. `04_figures.py` (10 figures), `05_report.py` → `report.html`. | **Implemented, run (Windows)** | `step3_results.json` (marginal) and `step3_results_average.json` (average CI). Main test: tree −0.90 % (bootstrap −1.01 to −0.80 %), oracle −0.97 %, carbon-only rule **+1.68 %**. With average CI: tree −0.76 %, direct −0.84 %. Bills **+9.2 %** for the tree. | (1) **Headline metric mismatch**: the main run uses *marginal* emission factors and expresses the change as % of the footprint at average CI; the proposal/prompt define the primary metric with realised *average* CI as 100·(E_hist − E_pol)/E_hist. Needs a decision (see below). (2) The main price target (13.608, Jan–Oct) is above the historical test-period price (12.795), so policies get ~6 % more price allowance than the TOU baseline on test; the matched-target result (−0.71 %) must be shown next to the headline. (3) Produce the 4 required figures in `results/figures/` and a single comparison table. | P0/P1 |

## Other observations

- **Final report draft exists** (`final_report/main.tex`, 478 lines, abstract states the −0.90 % result). The MVP
  report should be consistent with it, or flag where re-running changes the numbers.
- `test.ipynb` is a scratch notebook (GRIB → CSV conversion, CSV peek). `EDA_df_fuel_ckan.ipynb` is executed NESO EDA.
  `CVAE_reproduction/CVAE_reproduction.ipynb` is an unexecuted summary notebook of the reproduction.
- `CVAE_reproduction/scripts/02_lcl_checks.py` needs `Small LCL Data/` (168 chunk files) which we do not have; it is
  a reproduction check, not needed by the pipeline.
- Raw data verified readable: `CC_LCL-FullData.csv` (8.5 GB; columns `LCLid, stdorToU, DateTime, KWH/hh`),
  `df_fuel_ckan.csv` (312,816 rows, 2009-01-01 → 2026-11-04, UTC, `CARBON_INTENSITY` present; 2013: 17,520 rows,
  48 per day every day, no nulls, CI mean 475 g/kWh, range 259–622), `temp UK 2013-14.csv` (ERA5 hourly, 0.25° grid,
  Kelvin, `step = 0 days` → reanalysis, i.e. realised weather, not forecasts), `Tariffs.xlsx` (identical to the repo copy).
- The repo's `project_proposal_ML.pdf` differs from `../project_proposal_ML.pdf` (byte-level); the parent copy was used
  for this audit.
- `.gitignore` already excludes raw CSVs and the large `.npy` files but not virtualenvs, `results/` intermediates,
  scratch parquet or checkpoints outside `outputs/**/*.npy` (all `.pt` files are currently committed, ~tens of MB).

## Scientific caveats already visible in the existing results

1. The effect is small (~1 % of household emissions) and comes from **conservation during evening High events**, not
   from shifting load into clean hours; Low events increase consumption, which is why the carbon-only rule *raises*
   emissions.
2. The tree is depth 1 and its split variable is a consumption-level (season) proxy; in the test window it prescribes
   the same action every day and the feasibility rule does the rest (front-loads events until the budget runs out).
   Interpretability "costs nothing" largely because the decision problem is nearly trivial under this menu.
3. A time-average price constraint does not cap bills: the tree raises the mean bill by ~9 %.
4. Counterfactuals rely on observational (DiD vs flat-tariff households) response coefficients; they are model-based,
   not causal, and the High response is overstated in some held-out checks.
