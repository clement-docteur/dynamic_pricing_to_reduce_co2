# MVP Report — Interpretable Carbon-Aware Electricity Pricing

MIT 15.095 project (Clement Docteur, Panle Chen). MVP integration run: 2026-11-14, branch `feature/mvp-integration`,
macOS (Apple Silicon), Python 3.12, Interpretable AI 3.2.2.

**Research question.** Can an interpretable, carbon-aware pricing policy reduce estimated household electricity-related
emissions relative to the historical time-of-use (TOU) tariff while keeping a comparable average announced price?

**Short answer (model-based).** On the held-out test window (Nov–Dec 2013, 61 days), a shallow policy tree reduces
**estimated** emissions by **0.76 %** (exact tree, depth 2; 95 % demand-model bootstrap 0.67–0.85 %). The Interpretable AI
Optimal Policy Tree (depth 3) reduces them by **0.80 %**. Both stay within the time-average price target of 13.61 p/kWh,
and both come close to direct optimisation (0.84 %) and to a perfect-hindsight oracle (0.89 %). Three caveats come with
this:

- **The gain is small, and conservation drives it.** Households use less during evening High blocks; there is very little
  load shifting into clean hours.
- **The price budget, not the tree, decides most days.** The feasibility rule overrides the tree on 42–49 of 61 days.
- **Bills go up.** The mean predicted household bill rises 8–10 %, because a time-average price cap does not cap bills.

A carbon-only heuristic that ignores demand response **raises** emissions (−0.29 %, i.e. +0.29 %), because Low-price
blocks increase consumption. All numbers are model-based counterfactual estimates, not measured or causal effects.

---

## 1. How to run

```bash
/opt/homebrew/bin/python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_mvp.py                 # full MVP, ~4 min (+ ~1 min to rebuild the household matrix the first time)
.venv/bin/python run_mvp.py --reproduce     # also re-run Clement's original Steps 2-3 and compare with his committed outputs
```

Raw data are read from `$CARBON_RAW_DIR`, which defaults to the folder that contains the repository (`../LCL_Data/CC_LCL-FullData.csv`,
`../df_fuel_ckan.csv`, `../temp UK 2013-14.csv`, `../Tariffs.xlsx`), and are never modified. IAI runs through
`julia +1.12.6/IAI/3.2.2` (`--julia` to change, `--skip-iai` to skip). Stages, outputs and logs are documented in
`run_mvp.py`; per-stage logs are in `results/logs/` (gitignored).

| Output | Location |
|---|---|
| Comparison tables (main + all sensitivities) | `results/tables/policy_comparison.md` (CSV versions alongside) |
| Data-quality summary | `results/tables/data_quality.md` |
| Demand-model validation | `results/tables/demand_validation.json`, Figure 2 |
| Sanity / leakage checks (20/20 pass) | `results/tables/sanity_checks.md` |
| Four required figures | `results/figures/fig1_carbon_forecast.png` … `fig4_policy_tree.png` |
| MVP pipeline outputs (JSON/NPZ) | `results/mvp/outputs/` |
| Reproduction of Clement's pipeline | `results/reproduction/reproduction_check.md` |
| Processed Parquet (gitignored, regenerated) | `results/processed/panel_2013_halfhourly.parquet`, `household_daily_kwh_2013.parquet`, `cost_matrix_day_x_action_mvp.parquet` |

---

## 2. What is reproduced vs what is new

### 2.1 Reproduced (Clement's pipeline, unchanged methodology)

| Item | Result |
|---|---|
| Household data rebuild from the 8.5 GB raw CSV (`CVAE_reproduction/scripts/01_build_lcl2013.py`) | Identical: same 1,032 ToU + 1,032 Std households. Cluster means match the committed `power_consump_ours.parquet` to 3.6e-7 kWh, and the build/fill statistics match apart from float rounding at 1e-15. |
| Step 1 carbon forecast (Ridge, Lasso, RF, 3 persistence baselines) | Identical metrics. |
| Step 1 XGBoost (the selected model) | Reproduces only approximately: test MAE 27.23 → 27.56, RMSE 34.95 → 35.46 g/kWh; forecasts differ by 2.5 g/kWh on average. The same hyper-parameters are selected. This is platform floating-point behaviour (Apple Silicon vs Windows x86). |
| Step 1b marginal emission factors | Identical. |
| Steps 2–3 run on Clement's committed Step-1 forecasts (both emission metrics) | **Exactly reproduced.** 2,738 numbers compared, 0 mismatches; arrays agree to 1e-14 (`results/reproduction/reproduction_check.md`). Clement's headline (marginal factors): tree −0.90 % change, oracle −0.97 %, carbon rule +1.68 %. |

### 2.2 New in the MVP

1. **Portability and a single entry point.**
   - `run_mvp.py` and a pinned `requirements.txt`.
   - Raw-data paths are relative to the repository instead of Windows paths.
   - Output folders can be overridden (`CARBON_OUT`, `CARBON_FIG`), so Clement's committed outputs are never overwritten.
2. **Primary metric switched to average carbon intensity**, as in the proposal. Reduction (%) = 100 × (E_hist − E_policy) / E_hist,
   with realised NESO average CI for evaluation and the forecast CI for decisions. Marginal emission factors are kept as a
   sensitivity analysis.
3. **Temperature leakage fixed.** The policy feature "temperature, day mean" used the *realised* ERA5 temperature of the
   tariff day, which is not known at the day-ahead announcement (12:00 GMT on d−1). It is replaced by the mean temperature
   of the 24 h ending 12:00 GMT on d−1 (`03_policy.py --temperature lagged`; Clement's default is unchanged). A perturbation
   test confirms two things:
   - the new features of day d do not change when any data after 12:00 GMT on d−1 is perturbed;
   - the original feature did leak.
4. **IAI Optimal Policy Trees** (`scripts/07_iai_opt.py`, `07_iai_fit.jl`).
   - IAI's `OptimalTreePolicyMinimizer` is fitted at depths 1–3 under exactly the same protocol as Clement's exact tree:
     fit on the training days, choose depth and λ on validation under the feasibility rule, refit on train+val, test once.
   - Main fold plus the three earlier rolling folds; 312 IAI fits in total.
5. **The price-matched comparison is reported as a retrospective fairness check** (Section 5.3), not as a deployable policy.
6. **Deliverables:**
   - a day × action cost matrix (Parquet);
   - a data-quality summary;
   - a held-out demand-model validation against observed consumption;
   - the four required figures;
   - 20 automated sanity and leakage checks.

---

## 3. Data (details: `results/tables/data_quality.md`)

- **Low Carbon London.**
  - 167.9 M rows, 5,566 households (1,123 dynamic-ToU, 4,443 flat Std). Units are kWh per half-hour; timestamps are GMT
    (no DST gap or duplication).
  - Eligibility is more than 95 % valid 2013 half-hours. That keeps all **1,032 ToU** households and a seeded random sample
    of **1,032 Std** households (out of 4,034 eligible), which serve as the control group.
  - Cleaning: 12 off-grid rows dropped; 63,608 exact duplicate rows (identical values) averaged; 0.062 % of half-hours
    gap-filled, with nothing left missing.
  - ToU households are grouped into 4 clusters (sizes 285, 299, 299 and 149), from Clement's NMF + k-medoids reproduction.
- **NESO.** 2013 has 17,520 half-hours: 48 every day, UTC, no missing CI. Mean CI 475 g/kWh (range 259–622).
  Data from 2009–2013 is used to train the forecaster.
- **ERA5.** Hourly 2 m temperature on a 0.25° grid for 2013–14. This is reanalysis (realised weather), not a forecast, so it is
  used for the demand model's weather adjustment and, lagged, as a policy feature.
- **Tariff.**
  - Prices: Low 3.99, Normal 11.76, High 67.20 p/kWh.
  - The schedule is the authors' slot-aligned 2013 version, which the event study validated. `Tariffs.xlsx` (London clock
    time) differs from it on 17 days.
  - Full-year time-average price: 13.472 p/kWh for the schedule used, 13.517 for `Tariffs.xlsx` (the proposal's
    reference value).
- **Chronological splits.**
  - Train: Jan–Aug (243 days, average announced price 13.630).
  - Validation: Sep–Oct (61 days, 13.520).
  - Test: Nov–Dec (61 days, 12.795; 7 High-event and 18 Low-event days).
  - **Price target** = train+val historical average = **13.608 p/kWh**.

---

## 4. Methods (brief)

**Step 1 — Carbon-intensity forecast** (`01_ci_forecast.py`, Clement's code).
- Day-ahead forecast of the 48 half-hourly CI values, made at 12:00 GMT on d−1 (horizon 12–36 h).
- Features: lagged CI and generation mix up to the cut-off, plus calendar variables. No realised next-day data is used.
- Models: Ridge, Lasso, RF and XGBoost, against persistence baselines. Hyper-parameters are tuned on Sep–Oct.
- Monthly expanding refits, so every 2013 day has an out-of-sample forecast.

**Step 2 — Demand-response simulator** (`02_demand_response.py`, Clement's code). This replaces the CVAE as the main simulator.
- Per cluster, a ridge regression of log(ToU mean / Std mean) on:
  - half-hour × day-type intercepts;
  - temperature × time-of-day band;
  - tariff terms: in-block High/Low effects by band, a 2 h dip before Low, the half-hour after Low, and 2 h before High.
- The flat-tariff households act as a contemporaneous control, a difference-in-differences design.
- Counterfactual demand under a schedule z is y(z) = y_obs × exp(m(z) − m(z_hist)). It depends on the **whole** daily
  schedule, so it captures conservation, pre-event dips and rebounds.
- Fitted on Jan–Oct; specification chosen on Sep–Oct.
- The CVAE ensemble (320 trained checkpoints, not retrained) is kept as a sensitivity "world". On chronological hold-out it
  shows −13 % to −30 % energy bias and wrong-signed tariff effects.

**Step 3 — Candidate schedules, costs and policies** (`03_policy.py`).
- **Candidate pool:** 150 single-block schedules (High 2–4 h; Low 3 or 6 h).
- **Menu:** up to 6 schedules selected greedily on *training days only*. The Nov–Dec menu is:

| Schedule | Time-average price (p/kWh) |
|---|---|
| Normal all day | 11.76 |
| High 17:00–20:00 | 18.69 |
| Low 02:00–05:00 | 10.79 |
| High 18:00–20:00 | 16.38 |
| High 16:00–20:00 | 21.00 |

- **Cost of schedule z on day d:** C_d(z) = (1/1000) Σ_t CI_dt ŷ_dt(z), in kg CO₂ per household.
- The day × action matrix is exported to `cost_matrix_day_x_action_mvp.parquet`, with features, emissions, announced price,
  predicted bill, kWh and peak.
- Policies compared:

| Policy | How it chooses each day's schedule |
|---|---|
| A. Historical TOU | The observed 2013 schedule. |
| B. Carbon heuristic | High on the forecast-dirtiest 3 h, Low on the cleanest window, with as many High slots as fit the target every day. It does not use the demand model. |
| C. Direct optimisation | Minimises forecast emissions + λ·price for each day, with the feasibility rule applied. |
| D. Policy tree | A prescriptive tree on decision-time features, minimising realised training costs + λ·price. Exact solver (Clement's `PolicyTree`) or IAI. |
| Oracle | Perfect hindsight, with an exact price constraint. Not deployable. |
| Extension: budget-aware tree | Clement's tree with the remaining price budget as an extra feature. |

- **Tree features (all available at 12:00 GMT on d−1):**
  - forecast-CI summaries (day mean, 16–21 h mean, evening minus night, max − min, start of the dirtiest 3 h);
  - lagged temperature;
  - weekend/holiday flag;
  - recent consumption (days d−8 to d−2).

**Average-price feasibility rule** (proposal §2). Before announcing day k, the rule takes the first action in the policy's
ranking that keeps the period target attainable, assuming every remaining day is all-Normal. The all-Normal day
(11.76 ≤ 13.608) is always feasible, so the constraint can always be met. Every policy uses the same target, tolerance
(1e-9) and rule. Historical TOU is reported at its observed prices.

---

## 5. Results (held-out Nov–Dec 2013; model-based estimates)

### 5.1 Carbon forecast (Figure 1)

| Model | Val MAE | Test MAE | Test RMSE |
|---|---|---|---|
| Persistence (same slot, latest day) | 31.6 | 28.4 | 38.6 |
| Ridge | 25.8 | 29.5 | 37.8 |
| Random forest | 24.3 | **26.0** | **34.3** |
| **XGBoost (selected on validation)** | **24.0** | 27.6 | 35.5 |

Units are g/kWh. On the test period the forecaster beats persistence by only about 3 % in MAE (8 % in RMSE). RF would
have been slightly better on test, but selection is done on validation, so XGBoost is kept. ERA5 temperature added nothing
(Clement's weather experiment).

### 5.2 Demand-model validation (Figure 2, `results/tables/demand_validation.json`)

The test is out of sample: the model is fitted on Jan–Oct and predicts Nov–Dec ToU consumption from the same day's
flat-tariff consumption.

| Metric | Observed | Predicted |
|---|---|---|
| Daily total (kWh/household) | 10.56 | 10.62 (MAPE 1.3 %, bias +0.5 %) |
| High-price half-hours | 0.2465 | 0.2476 (0.2650 without tariff terms) |
| Low-price half-hours | 0.2249 | 0.2263 (0.2096 without tariff terms) |
| Normal half-hours | 0.2186 | 0.2197 |
| Mean daily peak half-hour | 0.360 | 0.359 |

Half-hour values are kWh per household per half-hour.

Held-out tariff effects (observed vs model):

| Effect | Observed | Model |
|---|---|---|
| In-block High | −7.7 % [−9.0, −5.9] | −7.5 % |
| In-block Low | +6.5 % [3.5, 9.2] | +7.7 % |
| 2 h before Low | −2.5 % | −3.0 % |
| 2 h before High | +4.2 % | +4.5 % |

This is consistent with the LCL trial findings: clear conservation during High events, increased use during Low events,
and only modest shifting. The CVAE does far worse on the in-block High effect (−1.1 %).

### 5.3 Main policy comparison (Figure 3; full table in `results/tables/policy_comparison.md`)

| Policy | Emissions (kg/household, 61 days) | **Reduction vs TOU** | 95 % simulator CI | Avg announced price (target 13.608) | Mean bill change | kWh change | Daily peak change | Feasibility overrides |
|---|---|---|---|---|---|---|---|---|
| A. Historical TOU | 298.23 | — | — | 12.795 (observed) | — | — | — | — |
| B. Carbon heuristic | 299.10 | **−0.29 %** | [−0.42, −0.16] | 13.606 | +4.7 % | +0.29 % | +2.0 % | 0 |
| C. Direct optimisation | 295.72 | **0.84 %** | [0.75, 0.93] | 13.594 | +14.0 % | −0.84 % | −2.5 % | 43 |
| D. Exact policy tree (depth 2, 4 leaves) | 295.96 | **0.76 %** | [0.67, 0.85] | 13.578 | +8.2 % | −0.76 % | −1.4 % | 49 |
| D'. IAI Optimal Policy Tree (depth 3, 7 leaves) | 295.86 | **0.80 %** | n/a | 13.578 | +9.5 % | −0.80 % | −1.8 % | 42 |
| Extension: budget-aware tree (depth 2) | 295.84 | 0.80 % | [0.71, 0.89] | 13.575 | +14.4 % | −0.83 % | −1.9 % | 3 |
| Reference: oracle (hindsight) | 295.58 | 0.89 % | [0.80, 0.97] | 13.551 | +14.9 % | −0.87 % | −3.1 % | — |

All proposed policies meet the target. The test-period average price of every proposed policy is at most 13.606 p/kWh,
against the 13.608 target, so no deviations had to be tolerated.

**How the trees behave.**
- Both trees split on decision-time carbon and consumption features: the root of the exact tree is the forecast CI for
  16:00–21:00 (Figure 4).
- In the test window the exact tree prescribes "High 16:00–20:00" on all 61 days.
- The feasibility rule turns this into a **front-loaded** sequence: events on Nov 1–12, until the budget is spent, then 49
  Normal days.
- The IAI tree's extra 0.03 percentage points come from prescribing cheaper 2–3 h High blocks on some days. That makes the
  budget last 19 event days instead of 12, with 42 overrides instead of 49.
- The budget-aware tree paces the budget (3 overrides, mostly night-Low days plus 16 High days). Its emissions are similar,
  but bills are higher.

**Price-matched fairness check (retrospective, not deployable).**
- The deployable target (13.608) is above the historical TOU price in the test window (12.795), so the proposed policies
  get about 6 % more price allowance than the baseline.
- Re-running every policy with the test window's own historical price as the target gives: exact tree 0.61 %, direct
  optimisation 0.67 %, oracle 0.69 %, carbon heuristic −0.21 %.
- This uses future information and is reported only as a fairness check. **About a fifth of the headline reduction comes
  from the higher price allowance.**

### 5.4 Does the IAI Optimal Policy Tree improve on Clement's tree?

**No, not meaningfully** (`results/tables/policy_comparison.md` → "IAI Optimal Policy Tree vs exact tree").

| Test window | Exact tree (selected depth) | IAI OPT (selected depth) | Overrides, exact / IAI |
|---|---|---|---|
| May–Jun | −0.38 % (d1) | −0.38 % (d1, identical tree) | 0 / 0 |
| Jul–Aug | +0.30 % (d1) | +0.30 % (d1, identical) | 0 / 0 |
| Sep–Oct | +0.50 % (d1) | +0.50 % (d1, identical) | 45 / 45 |
| **Nov–Dec (main)** | **+0.76 %** (d2) | **+0.80 %** (d3) | **49 / 42** |

Values are emissions reductions on the test window.

- **Training objective.** At depth 1 both solvers find the same, globally optimal, split. At depths 2–3 their training
  costs differ by at most 0.04 kg on objectives of 700–1,400 kg, and IAI is sometimes better and sometimes worse. Clement's
  solver is therefore effectively optimal for these problem sizes.
- **Test results.** In three of four windows the validation-selected trees are identical. In Nov–Dec, IAI's +0.03 pp is
  well inside the demand-model bootstrap width (about ±0.09 pp). It comes from a deeper (7-leaf) tree that the feasibility
  rule still overrides on 42 of 61 days.
- **Depth.** In every window, depths 1–3 differ by at most 0.1 pp on test.
- **Recommendation.** Present the IAI depth-1/2 tree, which is identical to the exact tree, as the interpretable policy.
  Present depth 3 as a sensitivity, not as an improvement.

### 5.5 Sensitivity (`results/tables/policy_comparison.md`)

**Price target** (exact-tree reduction):

| Target (p/kWh) | Basis | Reduction |
|---|---|---|
| 12.795 | Retrospective, price-matched | 0.61 % |
| 13.517 | Proposal reference | 0.75 % |
| **13.608** | **Main** | **0.76 %** |
| 14.228 | Flat Std tariff | 0.88 % |
| Cap on price *paid* (13.784) | Consumption-weighted instead of time-average | 0.67 % |

A higher price allowance buys more High events and therefore more conservation.

**Demand response** (policies re-trained in each world):

| Demand-response world | Tree reduction |
|---|---|
| Response ×0.5 | 0.42 % |
| Response ×1.5 | 1.08 % |
| In-block terms only | 0.77 % |
| Full model | 1.04 % |
| CVAE ensemble | 1.17 % (carbon heuristic −2.80 %) |

The conclusion holds in every world: demand-aware policies reduce emissions and the carbon-only rule does not. The
magnitude scales roughly linearly with the assumed response.

**Emission factor.** With marginal emission factors (Clement's original metric, now with leak-free features), the tree
gives 0.90 % and the carbon heuristic −1.68 %. Policies optimised for average CI and scored with marginal factors give
0.87 %. The ranking of policies does not depend on the factor.

**Rolling windows** (each with its own train/val/target):

| Test window | Exact-tree reduction |
|---|---|
| May–Jun | −0.38 % |
| Jul–Aug | +0.30 % |
| Sep–Oct | +0.50 % |
| Nov–Dec | +0.76 % |

The fixed tree **increases** emissions in May–Jun. There it chooses night-Low days, which lower prices and bills (−27 %)
but raise consumption. The budget-aware tree is positive in every window: +0.26, +0.70, +0.58 and +0.80 %.

### 5.6 Clement's committed average-CI results vs this MVP run

The direct-optimisation, tree, budget-aware-tree and oracle reductions are unchanged: 0.841, 0.763, 0.803 and 0.890 %.
The selected tree depth changes from 1 to 2 when the realised temperature is replaced by the lagged one, but the test
schedules are the same. The carbon heuristic moves from −0.18 % to −0.29 %, because it uses the re-run XGBoost forecast.
**The leakage fix therefore did not change the headline numbers.** The leaking feature was not used by Clement's selected
tree.

---

## 6. Verification (`results/tables/sanity_checks.md`, 20/20 pass)

- **Data alignment:** NESO has 48 half-hours on all 365 days of 2013. Tariff, CI, temperature and consumption are aligned on
  one documented GMT slot convention.
- **Units:** prices take only the values 3.99, 11.76 and 67.20 p/kWh, and Low and High never overlap. CI is in g/kWh with a
  2013 mean of 475. Consumption is in kWh per half-hour (mean 0.20). The `Tariffs.xlsx` full-year mean is 13.517.
- **Predicted consumption** is non-negative and finite for every menu action on every day. The cost-matrix formula
  reproduces the optimisation costs exactly.
- **No leakage:**
  - The splits are chronological and disjoint.
  - The target uses train+val only; the menu uses training days only; depth and λ are chosen on validation; the test window
    is used once.
  - Perturbing anything after 12:00 GMT on d−1 leaves the features of day d unchanged (40 random days).
  - The original same-day temperature feature is shown to leak.
- **Price constraint:** every proposed policy's realised average announced price is at or below the target, and the
  all-Normal day guarantees feasibility.
- **Reproduction:** Clement's Steps 2–3 reproduce exactly.

---

## 7. Limitations

1. **Counterfactuals are model-based, not causal.**
   - The demand-response coefficients are observational: a difference-in-differences against flat-tariff households in a
     trial where schedules were not randomised day by day. They are identified only for schedule shapes that occurred in
     2013.
   - Candidate schedules are restricted to similar single blocks, but timings never used in the trial (e.g. 16:00–20:00
     every day) are extrapolations.
   - Effects are assumed constant across repeated events (no fatigue). Cluster 3 (149 households) has very large estimated
     effects (High in the peak band −29 %, Low +41 %), which dominate the aggregate response.
2. **Small effect size.** Under every assumption the estimated reduction is about 0.4–1.2 % of household emissions.
   Bootstrap intervals cover only demand-model parameter uncertainty, not specification or forecast uncertainty.
3. **The feasibility rule dominates.**
   - It is greedy, so it spends the price budget in the first days of the window and then cancels events (42–49 overrides
     out of 61).
   - The tree's interpretability is therefore partly cosmetic in Nov–Dec. The budget-aware tree fixes the pacing, but bills
     rise 14 %.
4. **Bills.** A time-average announced-price cap does not protect bills: they rise 8–14 % for the demand-aware policies,
   because High blocks are placed where consumption is high. A cap on the price paid lowers the gain to 0.67 %.
5. **Price allowance.** The deployable target sits 6 % above the test-window historical price. Price-matched results are
   about 20 % lower.
6. **Carbon forecast.** The forecaster is only slightly better than persistence on the test period. Since direct
   optimisation with a perfect CI forecast performs about the same, forecast accuracy is not the binding constraint.
7. **Single year, single region**, and 2013 grid conditions (coal-heavy). The ERA5 temperature is reanalysis; operational
   use would need a real weather forecast.
8. **CVAE.** It was not retrained. Its chronological-split performance is poor, so it is used only as a sensitivity world,
   which departs from the proposal's plan.

---

## 8. Prioritised remaining work for the final project

1. **Pacing-aware feasibility rule** — the largest lever on both the policy's behaviour and its interpretability. Options:
   - spread the budget evenly (a per-day allowance), or
   - make the budget-aware IAI tree the main policy, with the remaining budget as a feature.
   Re-evaluate bills under the new rule.
2. **Report rolling-window results as the primary robustness evidence**, and explain the May–Jun failure: the night-Low
   rebound increases emissions.
3. **Bill-aware constraint or objective.** Add a constraint on the consumption-weighted price, or a bill-change penalty, to
   the Lagrangian. Report the emissions–bill trade-off curve.
4. **Uncertainty.**
   - Add the bootstrap to the IAI tree.
   - Propagate CI-forecast error, for example by re-scoring with persistence forecasts.
   - Add a household-level bootstrap of the clusters.
5. **Demand-model robustness.**
   - Check cluster 3's large effects, e.g. by excluding extreme households.
   - Test effect decay over repeated events.
   - Compare with published LCL elasticities (Schofield et al. 2015).
6. **Final report.** Update `final_report/main.tex` so that average CI is the primary metric and marginal factors are a
   sensitivity, and add the IAI comparison and the price-matched fairness check.
7. Optionally, rerun the CVAE only if a better-calibrated chronological model becomes available. It is not needed for the
   conclusions.
