# Interpretable Carbon-Aware Electricity Pricing — Final Project Report (MVP)

MIT 15.095 · Clement Docteur and Panle Chen · branch `feature/mvp-integration` · finalised 2026-11-14

All emission and bill effects in this report are **model-based counterfactual estimates**: demand from a demand-response
simulator, multiplied by realised grid carbon intensity. They are not measured or causal effects. Unless stated otherwise,
emissions use **realised average carbon intensity (CI)**, the primary metric. Marginal emission factors are a sensitivity
analysis. Reduction (%) = 100 × (E_TOU − E_policy) / E_TOU, so positive means fewer emissions than the historical
time-of-use (TOU) tariff on the same days.

## Results at a glance

**Three kinds of result are kept separate throughout.**

| Category | Contents |
|---|---|
| **Reproduced** | Clement's pipeline (data build, carbon forecast, demand simulator, exact policy tree, direct optimisation, budget-aware tree, sensitivities), re-run on macOS. Steps 1b, 2 and 3 reproduce exactly: 2,754 numbers in Steps 2–3 and 355 in Step 1b, with 0 mismatches. After merging Clement's latest `main` (`0a2f9af`), every value in his original committed outputs is preserved; his update only adds a net-load marginal-emission variant and a separate `demand_response/` module. |
| **New experiments** | Average CI as the primary metric; leak-free decision features; Interpretable AI (IAI) Optimal Policy Trees; a budget-pacing feasibility rule; budget-aware IAI trees; household-bill caps and penalties. |
| **Retrospective sensitivity analyses** (use hindsight; not deployable) | Price targets set to the test window's own historical TOU price; bill caps set against the test window's own historical TOU bills; the perfect-hindsight oracle and the hindsight trade-off sweep. |

**Primary candidate policy: the budget-aware IAI Optimal Policy Tree with a +5 % household-bill cap, enforced with a
training penalty plus the feasibility rule.**

| | May–Jun | Jul–Aug | Sep–Oct | Nov–Dec | **Mean** |
|---|---|---|---|---|---|
| Emissions reduction vs historical TOU | 0.23 % | 0.77 % | 0.56 % | 0.78 % | **0.585 %** |
| Mean household bill change vs historical TOU | −7.6 % | +20.0 % | +7.3 % | +9.6 % | **+7.3 %** |
| Average announced price (target) | 13.58 (13.63) | 14.07 (14.09) | 13.58 (13.63) | 13.58 (13.61) | 13.70 |
| Feasibility-rule overrides | 0/61 | 1/62 | 1/61 | 0/61 | 0.5 |

- **Price constraint.** The announced-price target is met in every window.
- **Bill cap.** Predicted and simulated price paid stay within the +5 % cap in every window. The cap is defined against a
  **reference price fixed before the test window**: the historical TOU price paid over the training + validation days.
  It is **not** defined against the historical TOU bill realised in the test window. Against that realised bill, the
  policy is within +5 % in 1 of 4 windows (Section 7.3).
- **Comparison.** For reference, Clement's exact tree averages 0.295 % across windows, direct optimisation 0.477 %, the
  unconstrained budget-aware IAI tree 0.610 % (bills +10.8 %), and the hindsight oracle 0.670 %.
- **Why this policy.** It is a reasonable trade-off, not a proven optimum. It keeps 96 % of the unconstrained budget-aware
  tree's mean reduction while cutting the mean bill increase from +10.8 % to +7.3 %. The feasibility rule almost never has
  to step in. In three of four windows it is a one-split rule a customer could read.
- **What the evidence does not show.** It does not show that this policy is statistically better than its neighbours:
  differences of a few hundredths of a percentage point are within model uncertainty.

---

## 1. Research question and motivation

The research question is whether an interpretable, carbon-aware pricing policy can reduce estimated household
electricity emissions relative to the historical TOU tariff while keeping a comparable average announced price.

Grid carbon intensity varies within and across days. Day-ahead dynamic prices — Low 3.99, Normal 11.76 and High
67.20 p/kWh in the 2013 Low Carbon London (LCL) trial — can change when households consume. The proposal combines three
steps:

1. next-day CI forecasting;
2. a demand-response simulator;
3. a shallow Optimal Policy Tree (Lecture 11) that picks each day's schedule from a small menu, under an average-price
   constraint.

Household bills were added as a second policy objective (Section 7).

## 2. Data and preprocessing

Details are in `results/tables/data_quality.md`. All raw files are read in place from `$CARBON_RAW_DIR` (by default, the
folder containing the repository) and are never modified.

**Low Carbon London smart meters** (`LCL_Data/CC_LCL-FullData.csv`).
- 167.9 M rows and 5,566 households: 1,123 dynamic-TOU and 4,443 flat-tariff (Std).
- Units are kWh per half-hour. Timestamps are GMT, with no daylight-saving gaps or duplicates, so every 2013 day has 48
  half-hours.
- Cleaning in the 2013 window:
  - 12 off-grid rows dropped;
  - 63,608 exact duplicate rows averaged (all had identical values);
  - 0.062 % of half-hours gap-filled, with nothing left missing.
- Sample: all **1,032** TOU households with more than 95 % valid 2013 data, plus **1,032** Std households sampled with
  seed 0 from 4,034 eligible. The Std households are the control group.
- TOU households fall into 4 behavioural clusters (285, 299, 299 and 149 households), from Clement's NMF + k-medoids
  reproduction of Brégère & Bessa.

**NESO generation mix** (`df_fuel_ckan.csv`). Times are UTC. 2013 has 17,520 half-hours, 48 every day, and no missing
CI. The 2013 mean CI is 475 g/kWh (range 259–622). The forecaster is trained on 2009–2013.

**ERA5 temperature** (`temp UK 2013-14.csv`). Hourly reanalysis, which is *realised* weather, not a forecast. It enters
the demand model's weather adjustment and, lagged, the policy features.

**Tariff.**
- The authors' slot-aligned 2013 schedule, validated by an event study. `Tariffs.xlsx`, which uses London clock time,
  differs on 17 days.
- Full-year time-average price: 13.472 p/kWh for the schedule used, 13.517 for `Tariffs.xlsx` (the proposal's
  reference value).

**Chronological evaluation.**
- **Main split:** train Jan–Aug (243 days), validation Sep–Oct (61), test Nov–Dec (61).
- **Rolling evaluation:** four test windows (May–Jun, Jul–Aug, Sep–Oct, Nov–Dec). Each window is validated on the two
  months before it and trained on everything earlier.
- **Hyper-parameters, menus, price targets and bill references** use only training and validation days. Each test window
  is evaluated once.

## 3. Carbon forecasting and demand-response modelling

**Step 1 — day-ahead CI forecast** (Clement's `01_ci_forecast.py`; reproduced).
- The forecast is made at the 12:00 GMT announcement on day d−1, so the horizon is 12–36 h.
- Inputs: lagged CI, the lagged generation mix and calendar variables.
- Models: Ridge, Lasso, random forest (RF) and XGBoost, against persistence baselines. Monthly expanding refits give an
  out-of-sample forecast for every day of 2013.
- XGBoost is selected on validation MAE (24.0 g/kWh). On the Nov–Dec test it scores MAE 27.6 / RMSE 35.5, against
  persistence at 28.4 / 38.6 and RF at 26.0 / 34.3 (Figure 1).
- The gain over persistence is small. Realised temperature adds nothing.
- The Mac re-run of XGBoost differs slightly from Clement's Windows run (test MAE 27.56 vs 27.23) because of
  platform-dependent floating point. All other models reproduce exactly.

**Step 2 — demand-response simulator** (Clement's `02_demand_response.py`; reproduced).
- Per cluster, a ridge regression of log(TOU mean / Std mean) on half-hour × day-type intercepts, temperature × band, and
  tariff terms:
  - in-block High/Low effects by time-of-day band;
  - a 2 h dip before Low, the half-hour after Low, and 2 h before High.
- The flat-tariff households are the control (a difference-in-differences design).
- Counterfactual demand under a whole-day schedule z is y(z) = y_obs · exp(m(z) − m(z_TOU)).
- Fitted on Jan–Oct; specification chosen on Sep–Oct.

Held-out Nov–Dec validation (Figure 2; `results/tables/demand_validation.json`):

| Check | Observed | Predicted |
|---|---|---|
| Daily-total MAPE | | 1.3 % (bias +0.5 %) |
| High-price half-hours (kWh) | 0.2465 | 0.2476 |
| Low-price half-hours (kWh) | 0.2249 | 0.2263 |
| In-block High effect | −7.7 % [−9.0, −5.9] | −7.5 % |
| In-block Low effect | +6.5 % [3.5, 9.2] | +7.7 % |

The pattern is clear conservation during High events, extra use during Low events, and modest shifting.

The CVAE of Brégère & Bessa (320 trained checkpoints; **not retrained**) is kept as a sensitivity "world". On the
chronological split it shows −13 % to −30 % energy bias and the wrong sign for the High-price effect.

## 4. Optimal Policy Tree methodology

**Actions.**
- Greedy forward selection on *training* days picks a menu of up to 6 daily schedules from 98 candidate schedules
  (single High and/or Low blocks).
- The Nov–Dec menu:

| Schedule | Time-average price (p/kWh) |
|---|---|
| Normal all day | 11.76 |
| High 17:00–20:00 | 18.69 |
| Low 02:00–05:00 | 10.79 |
| High 18:00–20:00 | 16.38 |
| High 16:00–20:00 | 21.00 |

**Costs.** C_d(z) = (1/1000) Σ_t CI_dt · ŷ_dt(z), in kg CO₂ per household. The day × action matrix, with features,
emissions, announced price, predicted bill, kWh and peak, is in `results/processed/cost_matrix_day_x_action_mvp.parquet`
(regenerated by `run_mvp.py`).

**Decision features.** All are known at 12:00 GMT on d−1:
- forecast-CI summaries (day mean, 16–21 h mean, evening minus night, range, start of the dirtiest 3 h);
- lagged temperature;
- a weekend/holiday flag;
- recent consumption (days d−8 to d−2).

**Leakage fix (new).** Clement's original feature set used the realised temperature of day d. The MVP replaces it with
the mean temperature over the 24 h ending 12:00 on d−1 (`03_policy.py --temperature lagged`; Clement's default is
unchanged). A perturbation test confirms two things: no data after 12:00 on d−1 can change day d's features, and the
original feature did leak. The leaking feature was never used by Clement's selected tree, so the headline numbers did not
change.

**Trees.**
- **Prescriptive trees:** they minimise the summed model-based cost E + λ·price of the prescribed actions.
  - Clement's exact solver (`PolicyTree`) is exhaustive at the last level, with 12 quantile thresholds above it.
  - IAI's `OptimalTreePolicyMinimizer` (new) uses max_depth 1–3, minbucket 10 and cp 0.
- **Selection:** depth (1–3) and λ are chosen on validation; the tree is refitted on train + validation and tested once.

**Average-price feasibility rule** (proposal §2).
- Day k takes the first action in the tree leaf's ranking that keeps the period target attainable, assuming every
  remaining day is all-Normal.
- The all-Normal day (11.76 ≤ target) is always admissible, so the constraint is always met.
- The *override* count is the number of days on which the rule replaced the tree's first choice.
- The price target is the historical TOU average over train + validation. For Nov–Dec it is 13.608 p/kWh.

## 5. Baseline results and identified limitations

### 5.1 Main test window (Nov–Dec 2013, primary metric)

| Policy | Reduction | 95 % demand-model bootstrap | Avg announced price | Mean bill change | Overrides |
|---|---|---|---|---|---|
| Historical TOU | — | — | 12.795 (observed) | — | — |
| Carbon heuristic (no demand model) | **−0.29 %** | [−0.42, −0.16] | 13.606 | +4.7 % | 0 |
| Direct optimisation | 0.84 % | [0.75, 0.93] | 13.594 | +14.0 % | 43 |
| Exact policy tree (depth 2) | 0.76 % | [0.67, 0.85] | 13.578 | +8.2 % | 49 |
| IAI OPT (depth 3, new) | 0.80 % | — | 13.578 | +9.5 % | 42 |
| Oracle (hindsight) | 0.89 % | [0.80, 0.97] | 13.551 | +14.9 % | — |

These come from `results/tables/policy_comparison.md`. Clement's committed average-CI results give the same tree,
direct-optimisation and oracle numbers. The carbon heuristic differs (−0.18 % vs −0.29 %) because of the XGBoost
platform difference.

**Sensitivity.**
- **Marginal emission factors:** tree 0.90 %, heuristic −1.68 %. The policy ranking is unchanged.
- **Clement's net-load marginal factors** (merged from his `main`). These are merit-order factors by season ×
  net-load decile. They exist only as realised factors, with no decision-time forecast, so they are used to re-score the
  same average-CI schedules, not to optimise. Re-scored this way: tree 0.82 %, direct optimisation 0.85 %, budget-aware
  tree 0.87 %, oracle 0.88 %, heuristic −0.30 %. The ranking is again unchanged.
- **Demand response scaled ×0.5 / ×1.5:** tree 0.42 % / 1.08 %.
- **CVAE world:** tree 1.17 %, heuristic −2.80 %.
- **Price target (retrospective, not deployable):** with the target set to the test window's own historical price
  (12.795), the tree gives 0.61 %. About a fifth of the headline gain comes from the deployable target being 6 % above
  that price.

**IAI vs exact solver.** The two are equivalent in practice.
- At depth 1 their training objectives are identical.
- At depths 2–3 they differ by at most 0.04 kg on objectives of 700–1,400 kg.
- In three of four windows they select the identical tree.
- In Nov–Dec, IAI's +0.03 pp is inside the bootstrap width.

### 5.2 Limitations identified in the baseline

1. **The effect is small and comes from conservation.** Households use less during evening High blocks. Low blocks
   increase use, which is why the carbon-only heuristic increases emissions.
2. **The feasibility rule, not the tree, makes most decisions.**
   - In Nov–Dec the tree prescribes "High 16:00–20:00" on all 61 days. The rule turns this into events on Nov 1–12 and
     then 49 Normal days.
   - The rule overrides the tree on 42–49 of 61 days.
3. **The fixed trees fail in summer.** In May–Jun they choose night-Low days, which raise use (−0.38 %). In Jul–Aug they
   never use the price allowance.
4. **Bills rise by 8–15 %**, because a time-average price cap does not cap bills.

## 6. Budget-aware policy improvements

There are two responses to limitation 2 (`scripts/10_pacing_experiment.py`; `results/pacing/`; Figures 5–6).

**(a) Budget-pacing rule** (new, `lib/policy.pacing_rule`).
- The rule also requires cumulative announced prices to stay within B (p/kWh × days) of an even spend of the target.
  B is chosen on validation.
- It uses only past announced prices, the day index and the target. With B = ∞ it is identical to the original rule
  (unit-tested).
- Results:
  - It spreads events evenly: Nov–Dec High days per quarter go from 12/0/0/0 to 7/6/6/5.
  - Gains are small where the budget binds. Exact tree +0.05 pp in Nov–Dec (paired bootstrap [+0.02, +0.08]); IAI +0.02.
  - Nothing changes in summer. Mean across windows: 0.295 → 0.325 %.
  - It makes the tree's choices *less* decisive: the first choice is followed on only 0–7 % of Sep–Dec days, and a
    constant (depth-0) policy does as well.

**(b) Remaining budget as a state variable** (Clement's budget-aware tree design, extended here to IAI).
- The tree sees the remaining price budget per remaining day. It is trained on days × 9 budget levels, with matching
  Lagrangian costs.
- **Mean reduction:** budget-aware IAI 0.610 % and exact 0.585 %, against 0.30–0.33 % for the fixed trees. That is 87–91 %
  of the oracle (0.670 %).
- **Robustness:** positive in all four windows.
- **Overrides:** the exact tree is overridden on 0–3 days per window; the IAI tree on 1–17. The rule becomes a safety net.
- **Pacing on top** adds nothing (−0.008 to +0.002 pp).
- **Learned rule:** bank budget with cheap night-Low days, then spend it on evening High events.
- **Cost:** mean bills +10.8 % (IAI), up to +22.7 % in Jul–Aug.

**Decision.** State-dependent (budget-aware) trees, with the original rule as a safety net, are the better design.

## 7. Household affordability constraints

This is a new experiment (`scripts/11_bill_constraint.py`; `results/affordability/`; Figure 7).

### 7.1 Formulation

- **Bill metric at decision time.** π̂(d, a) is the predicted price paid per kWh for schedule a on day d, using Step 3's
  decision-time demand forecast (days d−15 to d−2 of the same day type, multiplied by the tariff response). Simulated kWh
  changes are under 1 %, so the price paid tracks the bill.
- **Cap.** mean_d π̂(d, a_d) ≤ (1 + x) · π_ref, with x ∈ {0, 5, 10} %. π_ref is the historical TOU's mean daily price
  paid over **train + validation**: a pre-established reference, known before the test window.
- **Enforcement.** `lib/policy.multi_budget_rule` (new; equals `feasibility_rule` with one budget) enforces the cap
  online, together with the price target. The all-Normal day fits both budgets, so the rule never fails.
- **Penalty variant.** The budget-aware tree is retrained on E + μ·π + λ(budget)·price, with μ and depth chosen on
  validation. This is compared with "rule only", where the existing tree is capped by post-processing.

### 7.2 Results (IAI tree; mean of four windows; exact solver in brackets)

| Policy | Reduction | Mean bill change vs historical TOU | £ per household per window | Overrides |
|---|---|---|---|---|
| Unconstrained budget-aware | 0.610 % (0.585) | +10.8 % | +7.19 | 5.0 |
| Cap +10 %, penalty | 0.614 % (0.594) | +8.9 % | +5.99 | 5.0 |
| **Cap +5 %, penalty (primary)** | **0.585 %** (0.590) | **+7.3 %** | +4.70 | **0.5** |
| Cap +5 %, rule only | 0.568 % (0.547) | +8.6 % | +5.52 | 8.0 |
| Cap 0 %, penalty | 0.438 % (0.525) | +0.4 % | −0.46 | 5.8 |
| Cap 0 %, rule only | 0.462 % (0.463) | +3.7 % | +2.04 | 12.8 |

### 7.3 Is the constraint satisfied?

The answer depends on which quantity is constrained.

| Quantity | Satisfied? | Notes |
|---|---|---|
| (i) Predicted price paid (what the rule controls) | Always: every window, cap and variant | By construction |
| (ii) Simulated price paid | Primary policy: 4/4 windows. All IAI variants: 21 of 24 cases | The 3 misses are ≤ 0.07 p/kWh, which is forecast error |
| (iii) Bill vs historical TOU in the same test window | Primary policy: 1/4 windows (May–Jun) | See below |

Case (iii) **cannot be guaranteed by any deployable rule**. The benchmark's own price paid moves with the trial's event
schedule: 15.52, 12.16, 13.34 and 13.00 p/kWh across the four windows, against references of 13.75–14.43. In Jul–Aug the
historical TOU was unusually cheap, so a policy within +5 % of the reference is +20 % relative to that window's TOU.

With hindsight — caps set against each window's own historical TOU, a retrospective analysis — the caps are met in 4/4
windows. Mean reductions are then 0.30 / 0.46 / 0.53 % for caps of 0 / 5 / 10 %.

**A rigorous affordability cap must therefore be stated against a pre-committed reference price**, such as the train+val
TOU average used here or the flat Std tariff of 14.228 p/kWh.

### 7.4 Economic meaning

**Trade-off.** Moving from unconstrained to a 0 % cap (penalty) keeps 78–98 % of the reduction in the windows where
bills rose, and removes £3–8 of an £8–13 per-household bill increase per window. The +5 % cap with penalty costs
0.025 pp of mean reduction and cuts the mean bill increase by 3.5 pp, with almost no overrides.

**Per tonne.**
- The household transfer per tonne avoided, compared with historical TOU, is about **£4,800–8,600 per tCO₂**, and
  £16,000–100,000 per tCO₂ at the margin.
- For comparison, carbon prices are on the order of £50–100/t and appraisal values a few hundred £/t.
- The 1.4–2.6 kg avoided per household per two months is worth well under £1 at those values.

**Transfers, not resource costs.** Bill increases are transfers to the supplier. A revenue-neutral lump-sum rebate would
keep the price signal with zero mean bill change, but bills would still be widely spread. In Nov–Dec the p10/p90 change
moves from +7 % / +22 % to −31 % / +9 %, and 21 % of households would still pay over 5 % more. Affordability is therefore
a distributional issue.

### 7.5 Interpretability of the primary policy

- **Jul–Aug, Sep–Oct, Nov–Dec:** the tree is a single split on the remaining price budget per remaining day:
  "if budget < 2.5–4 p/kWh above Normal → Normal day, else → evening High block (16/17:00–20:00)".
- **May–Jun:** 7 leaves, also splitting on the budget, plus temperature, recent consumption and weekend.

**The policy does not use the day-ahead carbon forecast to time events.** Its carbon benefit comes from the
model-based costs, which favour evening conservation blocks.

## 8. Final results and comparisons

Mean of the four rolling windows. These are deployable policies under each window's own price target
(`results/tables/final_comparison.md`, Figure 8).

| Policy | Category | Reduction (mean; range) | Mean bill change | Avg announced price |
|---|---|---|---|---|
| Historical TOU | benchmark | 0 | 0 | 13.40 |
| Carbon heuristic | reproduced | −0.21 % (−0.58 to +0.32) | +5.6 % | 13.73 |
| Policy tree, exact solver | reproduced (leak-free features) | 0.295 % (−0.38 to +0.76) | −4.3 % | 12.43 |
| IAI OPT | new | 0.303 % (−0.38 to +0.80) | −4.0 % | 12.43 |
| Direct optimisation | reproduced | 0.477 % (−0.17 to +0.84) | +3.6 % | 13.19 |
| Budget-aware IAI OPT, no bill cap | new | 0.610 % (0.26 to 0.87) | +10.8 % | 13.70 |
| **Budget-aware IAI OPT, +5 % bill cap (primary)** | new | **0.585 % (0.23 to 0.78)** | **+7.3 %** | 13.70 |
| Oracle (hindsight) | reference | 0.670 % (0.36 to 0.89) | +10.2 % | 13.70 |

**Answer to the research question (model-based).**
- An interpretable policy can reduce estimated household emissions relative to the 2013 TOU at a comparable average
  announced price.
- The effect is small: about 0.2–0.8 % of household emissions over two months, or 0.5–2.3 kg CO₂ per household for the
  primary policy.
- It comes from evening conservation, not from following day-to-day carbon forecasts.
- Making the tree budget-aware is what makes it robust across seasons.
- Holding bills near a pre-set reference costs little carbon. Holding them to the historical TOU's realised bills cannot
  be guaranteed in advance.

## 9. Limitations and future work

**Limitations.**
1. **Counterfactual validity.**
   - The response coefficients are observational (a difference-in-differences against flat-tariff households) and
     identified only for schedule shapes used in 2013.
   - There are no fatigue or income effects.
   - Cluster 3 (149 households) has very large estimated effects (High in the peak band −29 %, Low +41 %).
2. **Uncertainty.**
   - Bootstrap intervals cover only demand-model parameters, for Nov–Dec.
   - Differences of a few hundredths of a pp between policies are not meaningful.
   - Hyper-parameters (λ, μ, depth, B) are chosen on single 2-month validation windows, which is noisy. For example, the
     IAI 0 %-cap policy chooses all-Normal days in May–Jun.
3. **Carbon awareness is weak.** The forecaster barely beats persistence. With perfect CI forecasts, direct
   optimisation barely changes (0.84 → 0.85 % in Nov–Dec). The selected trees rarely split on carbon features.
4. **Bills.**
   - The caps are relative to a train+val reference, not to the realised benchmark.
   - The bill budget is not a tree feature; the bill state enters only through the penalty and the rule.
   - Distributional effects are large.
5. **Scope.** One year (2013, a coal-heavy grid), one city and about 1,000 households. ERA5 temperature is reanalysis,
   not a forecast. The CVAE was not retrained and is used only as a sensitivity world.

**Future work** (prioritised).
1. A bill-budget state feature, or a dynamic-programming / rolling-horizon formulation of both budgets.
2. Multi-window or cross-validated hyper-parameter selection.
3. Bootstrap uncertainty for every window and policy, including CI-forecast error.
4. A demand-model robustness check: cluster 3 outliers, comparison with published LCL elasticities, and comparison with
   Clement's demand-response baseline in `demand_response/` (merged from his `main`; its predictions use a shared
   format in `demand_response/outputs/predictions.csv.gz`).
5. Pre-committed reference tariffs, such as the flat Std price, and rebate designs for distributional effects.
6. Update `final_report/main.tex`. Since Clement's latest update it reports the price-matched 0.71 % next to the
   marginal-factor 0.90 % headline. It still needs average CI as the primary metric, the budget-aware IAI tree and the
   affordability results.

---

## Appendix A. Reproduction

See `README.md`. In short:
- `.venv/bin/python run_mvp.py --reproduce` runs every stage (about 12 min), including a re-run of Clement's Steps 1
  and 1b on this machine.
- `.venv/bin/python tests/run_checks.py` runs 25 automated checks.

**Merge with Clement's `main` (`0a2f9af`).** After the merge, the full pipeline was re-run and every result file was
compared with a pre-merge backup:
- 2,018 of 2,032 files are byte-identical.
- 11 files only gained entries: the net-load factor in `mef_2013.npz` and `step1b_mef.json`, its re-scoring under
  `scored_with_each_factor` in the Step 3 JSONs, and one new column in `sensitivity_emission_factor.csv`.
- 3 reports (`reproduction_check.*`, `policy_comparison.md`) changed only in counts and the added column.

**No pre-existing result value changed.** `tests/run_checks.py` encodes this with three reference points:
- every value of Clement's original outputs (`6c22f28`) is preserved;
- his latest outputs and code (`0a2f9af`) are untouched by this branch, and his net-load factor reproduces exactly;
- every value of the pre-merge MVP results (`6c55a54`) is preserved.

## Appendix B. Figures and tables

| File | Content |
|---|---|
| `results/figures/fig1_carbon_forecast.png` | CI forecast vs realised, model errors |
| `results/figures/fig2_demand_response.png` | Held-out demand validation, tariff effects, simulated profiles |
| `results/figures/fig3_policy_comparison.png` | Nov–Dec policy comparison and price compliance |
| `results/figures/fig4_policy_tree.png` | Fixed IAI and exact trees (Nov–Dec) |
| `results/figures/fig5_pacing_timeline.png` | Event timing under the original and pacing rules |
| `results/figures/fig6_pacing_rolling.png` | Pacing vs original rule, four windows |
| `results/figures/fig7_carbon_vs_bills.png` | Carbon vs bill trade-off, four windows |
| `results/figures/fig8_final_comparison.png` | Final comparison, four windows |
| `results/tables/final_comparison.md` | Final cross-window table and primary-policy constraint checks |
| `results/tables/policy_comparison.md` | Nov–Dec tables, sensitivities, IAI vs exact, committed vs MVP |
| `results/pacing/pacing_tables.md` | Pacing experiment |
| `results/affordability/affordability_tables.md` | Affordability experiment |
| `results/tables/data_quality.md` | Data-quality summary |
| `results/tables/sanity_checks.md` | 20 sanity and leakage checks |
| `results/reproduction/reproduction_check.md` | Reproduction of Clement's results |
| `IMPLEMENTATION_STATUS.md` | Audit of the original repository (before this work) |
| `TEAM_HANDOFF.md` | Summary for the team |
