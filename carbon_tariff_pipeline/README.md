# Carbon-aware tariff pipeline

End-to-end run of the project proposal (`../project_proposal_ML.pdf`) on Low Carbon London 2013.
**Read the results in `report.html`** (self-contained; open it in a browser).

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 1 | `scripts/01_ci_forecast.py` | Next-day half-hourly carbon-intensity forecast (Ridge, Lasso, random forest, XGBoost vs persistence), NESO 2009-2013, monthly expanding refits, weather test | `outputs/step1_metrics.json`, `outputs/ci_forecasts_2013.npz` |
| 1b | `scripts/01b_marginal_emissions.py` | Marginal emission factors (hour x season regressions of emission changes on generation changes; pumped storage charged at the overnight factor): 2013 estimates and monthly decision-time estimates | `outputs/step1b_mef.json`, `outputs/mef_2013.npz` |
| 2 | `scripts/02_demand_response.py` | Demand-response simulator: per-cluster response model with a flat-tariff control (main) and the reproduction's CVAEs (sensitivity); held-out checks | `outputs/step2_results.json`, `outputs/response_theta.npz` |
| 3 | `scripts/03_policy.py --metric marginal` (main) and `--metric average` | Schedule menu, exact optimal policy tree (depth <= 3), carbon-only rule, direct optimisation, oracle, feasibility rule, bills, sensitivity, cross-scoring with each emission factor, rolling evaluation on four 2-month test windows (May-Dec) | `outputs/step3_results{,_average}.json`, `outputs/step3_test_schedules{,_average}.npz` |
| check | `scripts/06_tree_check.py` (`--export` first if IAI is to be compared) and `scripts/06_iai_fit.jl` | Tree-solver check: refits every Step 3 tree with a finer (exhaustive at depth <= 2) search and, when its fits are present, Interpretable AI's Optimal Policy Tree on the same inputs | `outputs/tree_check.json`, `outputs/iai/` |
| - | `scripts/04_figures.py` | All figures | `figures/*.png` |
| - | `scripts/05_report.py` | Builds `report.html` from the JSON outputs | `report.html` |

| Module | Contents |
|---|---|
| `lib/common.py` | Paths, prices, splits, time conventions (`hours()`), data loaders, cluster aggregates, torch-free checkpoint reader |
| `lib/demand.py` | Tariff regressors, `ResponseModel` (main simulator), `CVAEEnsemble` (sensitivity); both share `log_effect()` |
| `lib/policy.py` | Policy names, candidate menu, exact `PolicyTree`, feasibility rule, lambda helpers, carbon-only rule |
| `lib/report_html.py`, `lib/report_style.css` | HTML helpers and page style for the report |

Run from this folder, in order, with `python -P scripts/<name>.py` (about 20 minutes: Step 1 about 5, each Step 3 run about 6).
Splits: train Jan-Aug 2013, validation Sep-Oct, test Nov-Dec (used only for the final evaluation). The rolling
evaluation repeats this on test windows May-Jun, Jul-Aug, Sep-Oct and Nov-Dec (validation = the 2 months before each).
`lib/` deliberately has no `__init__.py`: `lib/demand.cvae_conditions` imports `conditions.py` from `CVAE_reproduction/lib`.

Inputs come from the parent project folder: `df_fuel_ckan.csv` (NESO) and, from `CVAE_reproduction/data`, the gap-filled
household matrix, cluster labels, ERA5 conditions and the authors' tariff schedule; the trained CVAEs come from
`CVAE_reproduction/outputs/chrono/models`.

Environment note (Nov 2026): Windows Application Control blocks the torch DLLs and pyarrow's parquet/feather modules on
this machine. Parquet is read through DuckDB, and CVAE checkpoints are read without torch
(`lib/common.load_torch_checkpoint`) and run in numpy (`lib/demand.CVAEEnsemble`); the numpy decoder reproduces the
reproduction's test RMSE exactly (`cvae_loader_check` in `outputs/step2_results.json`). `lib/common` caps the BLAS thread
pool at 4, so scripts import it before numpy.

Interpretable AI (IAI 3.2.2, juliaup channel `1.11.6/IAI/3.2.2`) is installed with a licence, but Smart App Control also
blocks the compiled package caches and the libraries bundled with its system image (e.g. `libiconv-2.dll`), so it cannot
start here. To compare with IAI's Optimal Policy Trees:

    python -P scripts/06_tree_check.py --export          # here: writes outputs/iai/ (24 training sets, 5 MB)
    julia scripts/06_iai_fit.jl iai                      # on a machine where IAI runs, with copies of outputs/iai/
                                                         # (as ./iai) and scripts/06_iai_fit.jl; IAI's julia channel,
                                                         # e.g. julia +1.11.6/IAI/3.2.2
    python -P scripts/06_tree_check.py                    # here, after copying that iai/ folder back to outputs/iai/
    python -P scripts/05_report.py                        # adds IAI columns to the solver-check table

`06_iai_fit.jl` has not been run yet (IAI cannot start here). `06_tree_check.py` checks that IAI's exported node tables
reproduce IAI's own predictions before using them.
