# Team handoff — MVP integration (branch `feature/mvp-integration`)

Clement, this is a short guide to what changed. The full story is in `MVP_REPORT.md`; reproduction steps are in
`README.md`. All numbers are model-based estimates, using average carbon intensity (CI) unless stated otherwise.

## What Clement originally implemented (all kept and reproduced)

- **CVAE reproduction (`CVAE_reproduction/`).** LCL 2013 data build, clustering, and CVAE and GAM simulators with 320
  chronological checkpoints.
- **Pipeline (`carbon_tariff_pipeline/`).**
  - Step 1: CI forecast with Ridge, Lasso, RF and XGBoost, plus marginal emission factors.
  - Step 2: difference-in-differences demand-response simulator, with the CVAE as a sensitivity.
  - Step 3: menu, exact Optimal Policy Tree, carbon rule, direct optimisation, oracle, feasibility rule, budget-aware
    tree, sensitivities and rolling windows.
  - Figures, `report.html` and `final_report/main.tex`.
- **Your latest `main` (`0a2f9af`), merged into this branch.** It brings the fresh-clone fix, the net-load
  (merit-order) marginal-emission variant, `demand_response/` and the toned-down `main.tex`.
- **Reproduction.** Your Steps 1b (incl. the net-load factor), 2 and 3 reproduce **exactly** on macOS: 0 mismatches.
  XGBoost forecasts differ slightly across platforms (test MAE 27.2 vs 27.6 g/kWh). `tests/run_checks.py` verifies
  three things:
  - every value in your original outputs (`6c22f28`) is still there;
  - your latest outputs, report, docs and `demand_response/` are untouched by this branch;
  - your scripts are identical to your `main`, except the MVP's opt-in flags in `03_policy.py` and the raw-data path
    lookup in `03_conditions.py` / `lcl_data.py`.

## How the merge was resolved

| Conflict | Resolution |
|---|---|
| `CVAE_reproduction/lib/lcl_data.py` | Your version, plus a raw-data lookup that accepts both layouts: `$CARBON_RAW_DIR`, then **the repository root (your layout)**, then the folder containing the repository (Panle's, incl. `LCL_Data/`). Your removal of `UPSTREAM` and your temp-folder default for `$LCL_WORK` are kept. |
| `README.md` | Combined. Your repo map, data sources, run commands, CVAE steps and report section, plus the MVP pipeline, checks and notebooks. |
| `requirements.txt` | Your `>=` floors, `torch` and Jupyter. `statsmodels` dropped (unused). The MVP's exact macOS versions are listed as a comment. |

The same raw-data lookup was applied to `lib/common.py`, `03_conditions.py`, `run_mvp.py`, `08_mvp_outputs.py` and the
implementation notebook, so your fresh-clone layout and Panle's layout both work. `03_policy.py` merged automatically:
your `marginal_netload` re-scoring now also appears in the MVP results, with no existing value changed.

## What Panle added or changed

| Area | Change | Files |
|---|---|---|
| Portability | Repository-relative raw-data paths (`$CARBON_RAW_DIR`), overridable output folders (`CARBON_OUT`, `CARBON_FIG`), pinned `requirements.txt`, single entry point `run_mvp.py`. | `CVAE_reproduction/lib/lcl_data.py`, `scripts/03_conditions.py`, `carbon_tariff_pipeline/lib/common.py` (paths only) |
| Metric | Average CI as the primary metric (as in the proposal); marginal factors as sensitivity. | `run_mvp.py` (`policy` stage) |
| Leakage fix | The realised same-day temperature feature is replaced by a lagged value (24 h to 12:00 on d−1). | `03_policy.py --temperature lagged` (opt-in flag; your defaults unchanged) |
| IAI OPT | IAI `OptimalTreePolicyMinimizer` at depths 1–3, four rolling windows, same protocol as your exact tree. | `scripts/07_iai_opt.py`, `07_iai_fit.jl` |
| Outputs | Data-quality summary, day × action cost matrix, 20 sanity/leakage checks, figures 1–4. | `scripts/08_mvp_outputs.py`, `09_reproduction_check.py` |
| Pacing rule | New `pacing_rule`; original vs pacing, and budget-aware IAI trees. | `lib/policy.py` (additive), `scripts/10_pacing_experiment.py` |
| Affordability | New `multi_budget_rule`; bill caps of 0/5/10 % (rule only, or penalty + rule). | `lib/policy.py` (additive), `scripts/11_bill_constraint.py` |
| Summary and tests | Final cross-window table and figure 8; 25 automated checks. | `scripts/12_final_summary.py`, `tests/run_checks.py` |
| Notebooks | Implementation notebook (the pipeline in executable cells, with Review and Full modes) and a read-only results walkthrough. | `project_implementation.ipynb`, `project_walkthrough.ipynb`, `requirements-notebook.txt` |

## Notebooks

- **`project_implementation.ipynb`.** This is for reading, running and modifying the method step by step.
  - It covers preprocessing, forecasting, demand model, tariff optimisation, your exact tree, IAI trees, budget-aware
    trees and bill caps, as ordinary code cells.
  - The code is a **faithful copy** of the logic in the `.py` modules. The `.py` scripts (via `run_mvp.py`) remain the
    **canonical pipeline** behind every reported number.
  - Each stage ends with checks against the pipeline's saved outputs.
  - **Review Mode** (default) trains the Nov–Dec window live in about 20 s, with 54 checks.
  - **Full Mode** (`MODE = "full"`) retrains everything for all four windows, including the forecaster grid search, the
    IAI refits in Julia and the raw-CSV rebuild. It takes about 5 min, with 138 checks.
  - Both modes matched with 0 mismatches and wrote nothing to the repository.
  - It needs the gitignored household matrices and saved IAI fits. Run `run_mvp.py --reproduce` once on a fresh clone.
- **`project_walkthrough.ipynb`.** A read-only tour of the saved tables and figures, useful for the presentation.

If you change a `.py` script, the notebook is **not** updated automatically. Re-run it, and its checks will show any
difference.

## Key findings (four rolling windows, deployable targets)

| Policy | Mean reduction (range) | Mean bill change | Overrides per window |
|---|---|---|---|
| Carbon heuristic | −0.21 % (−0.58 to +0.32) | +5.6 % | 0 |
| Exact tree (yours, leak-free features) | 0.295 % (−0.38 to +0.76) | −4.3 % | 0–49 |
| IAI OPT | 0.303 % (−0.38 to +0.80) | −4.0 % | 0–45 |
| Direct optimisation | 0.477 % | +3.6 % | — |
| Budget-aware IAI OPT, no bill cap | 0.610 % (0.26 to 0.87) | +10.8 % | 1–17 |
| **Budget-aware IAI OPT, +5 % bill cap — primary candidate** | **0.585 % (0.23 to 0.78)** | **+7.3 %** | **0–1** |
| Oracle (hindsight) | 0.670 % | +10.2 % | — |

1. **IAI and your exact solver are equivalent.** Training objectives are identical at depth 1 and differ by ≤ 0.04 kg at
   depths 2–3. IAI does not meaningfully improve test results.
2. **The fixed trees are dominated by the feasibility rule.** They are overridden on 42–49 of 61 days in Nov–Dec and fail
   in May–Jun. Pacing spreads events evenly but adds only about +0.03 pp.
3. **Remaining budget as a state feature (your budget-aware tree) is the key improvement.** It is positive in all four
   windows and roughly doubles the mean reduction.
4. **The +5 % bill cap (penalty + rule)** keeps 96 % of the unconstrained reduction, cuts the mean bill increase from
   10.8 % to 7.3 %, and almost never needs the rule.
   - The cap is defined against a **pre-set reference price**: historical TOU price paid over train+val.
   - Against the test window's *realised* historical TOU bill, it holds in only 1 of 4 windows (Jul–Aug +20 %). This
     cannot be guaranteed by any deployable rule.
5. **Carbon impact is small, and it comes from evening conservation, not from following daily CI forecasts.** In three of
   four windows the primary tree is a single split on the remaining budget. The household transfer is about
   £4,800–8,600 per tCO₂ avoided.

## Recommended figures and tables for the final report

| Item | Use |
|---|---|
| Fig. 1 `fig1_carbon_forecast.png` | Forecast accuracy |
| Fig. 2 `fig2_demand_response.png` | Simulator validation |
| Fig. 5 `fig5_pacing_timeline.png` | Why the feasibility rule dominates the fixed trees, and what pacing and budget-aware trees change |
| Fig. 7 `fig7_carbon_vs_bills.png` | Carbon vs bill trade-off |
| Fig. 8 `fig8_final_comparison.png` | Headline comparison across windows |
| Fig. 4 `fig4_policy_tree.png` | Tree diagram; could be swapped for a diagram of the primary one-split tree (text in `MVP_REPORT.md` §7.5) |
| Table `results/tables/final_comparison.md` | Main results and the primary policy's constraint checks |
| `MVP_REPORT.md` §5.1 table | Nov–Dec detail with bootstrap |
| `MVP_REPORT.md` §7.2–7.3 tables | Bill caps |
| Optional | Sensitivity tables in `results/tables/policy_comparison.md` (marginal factors, demand worlds, price targets) |

## Limitations and decisions to discuss

1. **Primary metric.** Average CI (the proposal) vs marginal factors (your original framing). Results and rankings are
   similar either way.
2. **Bill-cap reference.** Train+val TOU price paid (current) vs the flat Std tariff (14.228) vs a revenue-neutral rebate.
   This determines how "affordability" is defined in the report.
3. **"Carbon-aware" claim.** The trees barely use the CI forecast. Should the report say so plainly, or should we test
   carbon-only feature sets?
4. **Uncertainty.** Bootstraps exist only for Nov–Dec, and hyper-parameter selection relies on single 2-month validation
   windows. Do we extend this before the final report?
5. **The demand model drives everything.** Cluster 3's large effects and the absence of fatigue or income effects are
   worth a robustness paragraph.
6. **`final_report/main.tex`** now reports the price-matched 0.71 % next to the marginal-factor 0.90 % headline. It still
   needs average CI as the primary metric, the budget-aware IAI tree and the affordability results. There is an
   untracked `final_report/final_report_overleaf/` folder locally, which was left untouched and not committed.
7. **Your net-load factor.** The MVP uses it only to re-score schedules (Nov–Dec: tree 0.82 %, direct optimisation
   0.85 %), because there is no decision-time net-load forecast yet. Should it become the marginal-factor sensitivity of
   record?
8. **`demand_response/`** vs the MVP's demand simulator. Do we compare them in the final report?

## Exact reproduction commands

Run these from the repository root. Raw data must be in the parent folder, or set `CARBON_RAW_DIR`.

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python run_mvp.py --reproduce
```

The full run takes about 12 minutes and includes the final summary table and figure 8. Add `--skip-iai` if Interpretable AI is not installed.

```bash
.venv/bin/python tests/run_checks.py
```

The last command should report 25/25 checks passed.

Optional: run the implementation notebook. These commands install the Jupyter runtime and run Review Mode in place:

```bash
.venv/bin/pip install -r requirements-notebook.txt
```

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 project_implementation.ipynb
```

For Full Mode, set `PROJECT_NOTEBOOK_MODE=full` and write the result elsewhere (`--output-dir /tmp --output
project_implementation_full.ipynb`, without `--inplace`). The README's section 4 has the details.
