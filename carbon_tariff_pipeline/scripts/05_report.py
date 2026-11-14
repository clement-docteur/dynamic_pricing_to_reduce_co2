"""Build report.html (self-contained: figures embedded) from the pipeline outputs. Main results use marginal emission
factors (step3_results.json); the average-intensity run (step3_results_average.json) is the comparison. Numbers come
from those outputs; two background facts about the 2013 grid (mean carbon intensity, typical within-day range) are
cited from EDA_df_fuel_ckan.ipynb."""
from __future__ import annotations

import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import FIG, FLAT_PRICE, OUT, PIPE, PRICE, load_json, split_of, DATES_2013  # noqa: E402

import numpy as np  # noqa: E402

from lib.policy import POLICIES as P, carbon_rule_counts, carbon_rule_schedule, slot_time  # noqa: E402
from lib.report_html import esc, img, num, signed, table, tree_html  # noqa: E402

S1, S1b, S2, S3, S3a, TC = (load_json(OUT / f) for f in ("step1_metrics.json", "step1b_mef.json", "step2_results.json",
                                                          "step3_results.json", "step3_results_average.json",
                                                          "tree_check.json"))
assert S3["metric"] == "marginal" and S3a["metric"] == "average"
R, BOOT, Ra = S3["results"], S3["bootstrap_ci_emissions_change_pct"], S3a["results"]
menu_names = [m["name"] for m in S3["menu"]]
EDA = {"ci_mean_2013": 475, "within_day_range_pct": 13}       # EDA_df_fuel_ckan.ipynb, sections 2-3


def cut(x, nd=2):
    """'lowers by' wording: a negative % change shown as a positive reduction."""
    return num(-x, nd) + "%"


def ci(lo_hi, nd=2):
    return f"[{signed(lo_hi[0], nd)}, {signed(lo_hi[1], nd)}]"


def tree_features(node):
    return set() if node["leaf"] else {node["feature"]} | tree_features(node["left"]) | tree_features(node["right"])


def hours_mean(values, hrs):
    return float(np.mean([values[h] for h in hrs]))


# ------------------------------------------------------------------------------------------------ numbers used in text
tree, direct, oracle, crule, hist = R[P[2]], R[P[3]], R[P[5]], R[P[1]], R[P[0]]
T, n_test, days = S3["target_price"], S3["n_test_days"], S3["split_days"]
adj = S3["adjustments"]
sc, sw = S3["sensitivity_constraints"], S3["sensitivity_worlds"]
matched, paid, flat = (sc[k]["results"] for k in ("matched", "paid", "flat"))
paid_hist = S3["historical_price_paid"]
saved_kg = -tree["emissions_change_kg_per_hh"]
interp_cost = abs(tree["emissions_change_pct"] - direct["emissions_change_pct"])
same_as_direct = interp_cost < 5e-3 and (tree["high_slots"], tree["low_slots"]) == (direct["high_slots"], direct["low_slots"])
perfect_gain = abs(R[P[4]]["emissions_change_pct"] - direct["emissions_change_pct"])
budget_share = 1 - matched[P[2]]["emissions_change_pct"] / tree["emissions_change_pct"]     # part due to the larger budget
prescribed = max(S3["tree_prescribed_test_counts"], key=S3["tree_prescribed_test_counts"].get)
n_asks = S3["tree_prescribed_test_counts"][prescribed]
asks = "every test day" if n_asks == n_test else f"{n_asks} of the {n_test} test days"
ci_features = [f for j, f in enumerate(S3["features"]) if j in tree_features(S3["tree"]) and f.startswith("forecast")]
M1, LAB1, sel = S1["metrics"], S1["labels"], S1["selected_model"]
pers = M1["persist_latest"]
best_ml_val = min(M1[k]["val"]["MAE"] for k in ("xgb", "rf", "ridge", "lasso"))
wx = S1["weather_experiment"]
v_val, v_test = (S2["heldout_validation"][k] for k in ("Sep-Oct (validation; fitted Jan-Aug)", "Nov-Dec (test; fitted Jan-Oct)"))
eff3, sizes = S2["effects_main_janoct"]["3"], S2["cluster_sizes"]
OTHER_MAX = max(abs(v["pct"]) for c in "012" for k, v in S2["effects_main_janoct"][c].items() if "|" in k)
# marginal factors (winter, Oct-Mar, when the test period falls)
hr = S1b["hourly_2013"]
night_mef = hours_mean(hr["adjusted"]["Oct-Mar"], range(0, 5))
peak_mef = hours_mean(hr["adjusted"]["Oct-Mar"], range(16, 20))
peak_mef_u = hours_mean(hr["unadjusted"]["Oct-Mar"], range(16, 20))
peak_avg = hours_mean(hr["average_ci_test_by_hour"], range(16, 20))
fs = S1b["marginal_fuel_shares_2013"]
annual = S1b["annual"]
mt = S1b["checks"]["mean_test"]
by_policy = lambda S: {p: {f: v[p] for f, v in S["scored_with_each_factor"].items()} for p in P}
cross, cross_a = by_policy(S3), by_policy(S3a)          # policy -> factor -> emissions change (% of footprint)
paced = R[P[6]]                                          # budget-aware tree (remaining budget as an input)
_pt, _pf = S3["paced_tree"], S3["paced_features"]
assert (_pf[_pt["feature"]].startswith("recent consumption")
        and all(not n["leaf"] and _pf[n["feature"]].startswith("remaining budget") for n in (_pt["left"], _pt["right"]))),     "the text describing the budget-aware tree assumes: split on recent consumption, then on the remaining budget"
paced_counts = ", ".join(f"{esc(k)} ({v})" for k, v in sorted(S3["paced_test_counts"].items(), key=lambda t: -t[1]))
# rolling evaluation (four 2-month test windows)
roll = list(S3["rolling"].values())
rmean = lambda tid, p: float(np.mean([f[tid][p]["emissions_change_pct"] for f in roll]))
rvals = lambda tid, p: [f[tid][p]["emissions_change_pct"] for f in roll]
no_event_folds = [f["label"] for f in roll if f["tree_event_days"] == 0]
same_direct = sum(abs(f["main"][P[2]]["emissions_change_pct"] - f["main"][P[3]]["emissions_change_pct"]) < 5e-3 for f in roll)
assert roll[-1]["main"][P[2]]["emissions_change_pct"] == tree["emissions_change_pct"], "rolling Nov-Dec must equal the main run"
# tree-solver check (06_tree_check.py): finer search, and Interpretable AI's OPT once its fits are present
tc = list(TC["folds"].values())
HAS_IAI = TC["iai_cases"] == TC["n_cases"]
SOLVERS = ["step3", "fine"] + (["iai"] if HAS_IAI else [])
tc_fits = [(f["label"], kind, int(d), g) for f in tc for kind, by_d in f["fits"].items() for d, g in by_d.items()]


def gain_share(g, k):
    """Training gain over the best single action reached by solver k, % of the finer search's gain."""
    full = g["single_action_cost"] - g["cost"]["fine"]
    return 100 * (g["single_action_cost"] - g["cost"][k]) / full if full > 1e-12 else 100.0


share_d = {d: [gain_share(g, "step3") for _, _, dd, g in tc_fits if dd == d] for d in (2, 3)}
tc_chosen = [(f["label"], kind, f["chosen_depth"][kind], f["test"][kind]) for f in tc for kind in ("fixed", "paced")]
n_depth1 = sum(d == 1 for _, _, d, _ in tc_chosen)
tc_changed = [(lab, kind, t["step3"]["emissions_change_pct"], t["fine"]["emissions_change_pct"])
              for lab, kind, _, t in tc_chosen if abs(t["step3"]["emissions_change_pct"] - t["fine"]["emissions_change_pct"]) >= 5e-4]
assert all(a < 0 and b >= a for *_, a, b in tc_changed), "the text says the finer trees save slightly less on the test windows"
KIND = {"fixed": "tree", "paced": "budget-aware tree"}


# where the carbon-only rule puts its blocks under marginal factors (decision-time factors, test days)
mef_fc = np.load(OUT / "mef_2013.npz")["forecast"][split_of(DATES_2013) == "test"]
kh, kl = carbon_rule_counts(T)
rule = [carbon_rule_schedule(day, kh, kl) for day in mef_fc]
rule_high = Counter(slot_time(int(np.argmax(h))) for _, h in rule).most_common(1)[0][0]
rule_low = Counter(f"{slot_time(int(np.argmax(lo)))}-{slot_time(int(np.argmax(lo)) + kl)}" for lo, _ in rule).most_common(1)[0][0]


# ------------------------------------------------------------------------------------------------ tables
def t_step1():
    order = ["xgb", "rf", "ridge", "lasso", "persist_latest", "flat24h", "persist_7d"]
    rows = []
    for k in order:
        v, t = M1[k]["val"], M1[k]["test"]
        rho = t["within_day_spearman"]
        rows.append([LAB1[k] + (" <b>(selected)</b>" if k == sel else ""), num(v["MAE"], 1), num(t["MAE"], 1),
                     num(t["RMSE"], 1), "&ndash;" if rho != rho else num(rho, 2), num(t["spread_captured_3h"], 2)])
    params = "; ".join(f"{LAB1[k]} " + ", ".join(f"{a} = {b}" for a, b in p.items()) for k, p in S1["best_params"].items())
    return table(["Forecast of next-day CI", "Val. MAE", "Test MAE", "Test RMSE", "Within-day rank corr.", "3 h spread captured"],
                 rows, num_cols=(1, 2, 3, 4, 5), highlight_rows=(order.index(sel),),
                 caption="gCO2/kWh. Validation = Sep-Oct 2013, test = Nov-Dec 2013, both from monthly expanding refits. "
                         "Rank corr. = mean Spearman correlation between forecast and actual over the 48 half-hours of a day. "
                         "3 h spread captured = actual CI gap between the forecast dirtiest and cleanest 3 h windows, as a share "
                         f"of the true gap (1 = perfect). Hyperparameters chosen on validation: {params}.")


def t_mef():
    rows = [[str(y), num(v["mef_unadjusted"], 0), num(v["average_ci"], 0)] for y, v in annual.items()]
    return table(["Year", "Marginal factor (NESO, unadjusted)", "Average intensity"], rows, num_cols=(1, 2),
                 caption="g/kWh. Slope of half-hourly emission changes on generation changes, all hours. The margin moved "
                         "from coal to gas as coal became cheaper than gas in 2012-2013, so factors must be estimated "
                         "close to the period they describe.")


def t_fuels():
    rows = [[b, *(num(100 * fs[b][f], 0) + "%" for f in ("COAL", "GAS", "STORAGE", "IMPORTS"))] for b in fs]
    return table(["2013, hours UTC", "Coal", "Gas", "Pumped storage", "Imports"], rows, num_cols=(1, 2, 3, 4),
                 caption="Share of half-hourly changes in generation met by each source (regression slopes).")


def t_step2():
    rows = []
    for nm in ("in-block High", "in-block Low", "2 h before Low", "2 h before High"):
        for per, V in (("Sep-Oct", v_val), ("Nov-Dec", v_test)):
            for c, lab in (("all", "all 1,032"), ("3", f"cluster 3 ({sizes[3]})")):
                o = V[c][nm]
                rows.append([nm, per, lab, f"{signed(o['observed_pct'], 1)} {ci(o['observed_ci95'], 1)}",
                             signed(o["main_pct"], 1), signed(o["cvae_pct"], 1), str(o["n_days"])])
    return table(["Tariff situation", "Held-out days", "Households", "Observed (95% CI)", "Response model", "CVAE", "Event days"],
                 rows, num_cols=(3, 4, 5, 6),
                 caption="Effect on consumption relative to the same half-hour on event-free days, after removing the "
                         "Std-household control (difference-in-differences). Sep-Oct: models fitted on Jan-Aug. "
                         "Nov-Dec: models fitted on Jan-Oct; never used to fit or choose anything.")


def t_spec():
    lab = {"inblock": "In-block effects only", "main": "In-block + verified edge effects (main)",
           "full": "Full: + post-block and day-level terms"}
    rows = [[lab[k], ", ".join(S2["spec_terms"][k]) or "&ndash;", num(1000 * S2["spec_selection_val_rmse"][k]["household_weighted"], 2)]
            for k in lab]
    return table(["Specification", "Edge terms", "Val. RMSE &times;1000"], rows, num_cols=(2,),
                 highlight_rows=(list(lab).index(S2["chosen_spec"]),),
                 caption="RMSE of log(ToU / Std consumption) on Sep-Oct event days, models fitted on Jan-Aug.")


def t_menu():
    rows = [[esc(m["name"]), num(m["price"], 2), signed(m["price"] - T, 2, "")] for m in S3["menu"]]
    return table(["Daily schedule", "Time-average price (p/kWh)", "vs target"], rows, num_cols=(1, 2),
                 caption=f"Greedy forward selection from {S3['menu_pool_size']} plausible single-block schedules "
                         f"(High 2-4 h, Low 3-6 h, evening High + night Low), on Jan-Aug hindsight costs; selection "
                         f"stops when a schedule adds less than 2% of the first schedule's gain. Target = {num(T, 3)} p/kWh.")


def t_results():
    rows = []
    for i, p in enumerate(P):
        r = R[p]
        a = {2: adj["tree"], 3: adj["direct"], 4: adj["direct_perfect_ci"], 6: adj["tree_paced"], 7: adj["direct_paced"]}.get(i)
        ref = lambda s: "0" if i == 0 else s
        rows.append([p, "0 (reference)" if i == 0 else signed(r["emissions_change_pct"]), "&ndash;" if i == 0 else ci(BOOT[p]),
                     num(r["mean_price"], 2), "&ndash;" if a is None else f"{a}/{n_test}", num(r["price_paid"], 2),
                     ref(signed(r["bill_change_mean_pct"], 1)),
                     "&ndash;" if i == 0 else f"{signed(r['bill_change_p10_pct'], 1)} / {signed(r['bill_change_p90_pct'], 1)}",
                     ref(signed(r["kwh_change_pct"])), ref(signed(r["daily_peak_change_pct"], 1)), f"{r['high_slots']} / {r['low_slots']}"])
    return table(["Policy", "Emissions", "95% bootstrap", "Time-avg. price", "Rule adjusted", "Price paid", "Mean bill",
                  "Bill p10 / p90", "kWh", "Daily peak", "High / Low half-hours"], rows,
                 num_cols=tuple(range(1, 11)), highlight_rows=(2, 6),
                 caption=f"Test period Nov-Dec 2013 ({n_test} days), 1,032 ToU households, simulated with the main response "
                         f"model. Emissions = change in marginal-factor-weighted consumption vs the historical TOU tariff, as a "
                         f"share of the households' historical footprint at average intensity ({num(S3['reference_footprint_kg'], 1)} "
                         f"kg CO2 per household over the period). Prices in p/kWh; price paid = total bill / total kWh. "
                         f"Bootstrap = 200 weekly-block resamples of the response model.")


def t_cross():
    rows = []
    for p in P[1:]:
        rows.append([p, signed(cross[p]["marginal"]), signed(cross[p]["marginal_unadjusted"]), signed(cross[p]["average"]),
                     signed(cross_a[p]["average"]), signed(cross_a[p]["marginal"])])
    return table(["Policy", "Marginal-trained: marginal score (main)", "Marginal-trained: storage-unadjusted score",
                  "Marginal-trained: average score", "Average-trained: average score", "Average-trained: marginal score"], rows,
                 num_cols=(1, 2, 3, 4, 5), highlight_rows=(1,),
                 caption="Emissions vs the historical TOU tariff, Nov-Dec 2013, % of the historical footprint. Columns 2-4: "
                         "policies trained with marginal factors, scored with each factor. Columns 5-6: policies trained "
                         "with average intensity (the earlier analysis), scored with average and with marginal factors.")


def rule_text(node, menu):
    """One-line description of a fold's tree (menu = that fold's schedule names)."""
    feat = lambda n: f"{S3['features'][n['feature']]} &le; {S3['feature_formats'][n['feature']].format(n['threshold'])}"
    if node["leaf"]:
        return f"always {esc(menu[node['action']])}"
    if node["left"]["leaf"] and node["right"]["leaf"]:
        return f"{feat(node)}: {esc(menu[node['left']['action']])}; else {esc(menu[node['right']['action']])}"
    return f"deeper tree; first split on {feat(node)}"


def t_rolling():
    rows = []
    for f in roll:
        m, mt = f["main"], f["matched"]
        presc = ", ".join(f"{esc(k)} ({v})" for k, v in f["tree_prescribed_counts"].items())
        rows.append([f["label"], f"{f['days']['train']} / {f['days']['val']} / {f['days']['test']}", num(f["target"], 2),
                     num(f["historical_test_price"], 2), f"{f['depth']}: {rule_text(f['tree'], f['menu'])}", presc,
                     f"{f['tree_event_days']} / {f['matched_tree_event_days']}",
                     signed(m[P[2]]["emissions_change_pct"]), signed(mt[P[2]]["emissions_change_pct"]),
                     f"{f['paced_event_days']} / {f['matched_paced_event_days']}",
                     signed(m[P[6]]["emissions_change_pct"]), signed(mt[P[6]]["emissions_change_pct"]),
                     signed(mt[P[5]]["emissions_change_pct"]), signed(m[P[1]]["emissions_change_pct"]),
                     signed(m[P[2]]["bill_change_mean_pct"], 1), signed(m[P[6]]["bill_change_mean_pct"], 1)])
    return table(["Test window", "Days train / val / test", "Target", "2013 price", "Tree (depth: rule)", "Tree asks for (days)",
                  "Tree: days with an event (target / price-matched)", "Tree", "Tree, price-matched",
                  "Budget-aware tree: days with an event (target / price-matched)",
                  "Budget-aware tree", "Budget-aware tree, price-matched", "Oracle, price-matched", "Carbon rule", "Tree bill",
                  "Budget-aware tree bill"],
                 rows, num_cols=tuple(range(2, 16)), highlight_rows=(len(roll) - 1,),
                 caption="Each window repeats the whole protocol on the days before it: response model refitted, menu chosen "
                         "on the training days, depth and &lambda; on the 2-month validation window, target = historical "
                         "time-average price over training + validation (p/kWh). 'Price-matched' = target set to the 2013 "
                         "tariff's own time-average price in the test window. Emissions with marginal factors, % of the "
                         "window's historical footprint. Nov-Dec is the main analysis.")


def t_tree_check():
    names = {"step3": "Step 3", "fine": "finer", "iai": "IAI"}
    rows = []
    for f in tc:
        for kind, label in (("fixed", "Tree"), ("paced", "Budget-aware tree")):
            d = f["chosen_depth"][kind]
            g, t = f["fits"][kind][str(d)], f["test"][kind]
            rows.append([f["label"], label, str(d), str(g["rows"]), " / ".join(str(g["leaves"][k]) for k in SOLVERS)]
                        + [num(gain_share(g, k), 1) + "%" for k in SOLVERS if k != "fine"]
                        + [num(g["agree_with_step3_pct"][k], 1) + "%" for k in SOLVERS if k != "step3"]
                        + [signed(t[k]["emissions_change_pct"], 3) for k in SOLVERS])
    others = [k for k in SOLVERS if k != "step3"]
    header = (["Window", "Tree", "Depth", "Rows", "Leaves (" + " / ".join(names[k] for k in SOLVERS) + ")"]
              + [f"Gain, {names[k]}" for k in SOLVERS if k != "fine"]
              + [f"Same schedule, {names[k]}" for k in others]
              + [f"Test, {names[k]}" for k in SOLVERS])
    return table(header, rows, num_cols=tuple(range(2, len(header))), highlight_rows=(len(rows) - 2, len(rows) - 1),
                 caption="Each tree refitted on Step 3's own training set (Rows: training + validation days; for the "
                         "budget-aware tree, day &times; budget level) at the depth chosen in Step 3, by Step 3's search "
                         "and the finer search" + (" and IAI's OPT" if HAS_IAI else "") + ". Gain: training gain (cost of "
                         "the best single schedule for every row minus the tree's cost) as % of the finer search's. "
                         "Same schedule: % of training rows given the same schedule as Step 3's tree. Test: emissions "
                         "change vs the historical TOU tariff in the test window (marginal factors, % of footprint) under "
                         "the feasibility rule at the window's own target.")


def t_constraints():
    rows = []
    for v in sc.values():
        r = v["results"]
        g = lambda p, k, nd: signed(r[p][k], nd) if p in r else "&ndash;"
        rows.append([esc(v["label"])] + [g(p, "emissions_change_pct", 2) for p in (P[1], P[2], P[6], P[3], P[5])]
                    + [g(p, "bill_change_mean_pct", 1) for p in (P[2], P[6], P[3])])
    return table(["Constraint", "Carbon rule", "Tree", "Budget-aware tree", "Direct", "Oracle", "Tree bill",
                  "Budget-aware tree bill", "Direct bill"], rows,
                 num_cols=tuple(range(1, 9)), highlight_rows=(list(sc).index("main"),),
                 caption="Emissions (marginal factors) and mean bill vs the historical TOU tariff, Nov-Dec 2013, main "
                         "simulator. Tree depth fixed at the main run's; lambda re-chosen on validation for each constraint.")


def t_worlds():
    rows = [["Main response model"] + [signed(R[p]["emissions_change_pct"]) for p in (P[1], P[2], P[6], P[3])]
            + ["same", "same", "same", signed(R[P[5]]["emissions_change_pct"])]]
    for v in sw.values():
        f, rt = v["main_policies_evaluated_here"], v["retrained_here"]
        rows.append([esc(v["label"])] + [signed(f[p]) for p in (P[1], P[2], P[6], P[3])]
                    + [signed(rt[p]) for p in (P[2], P[6], P[3], P[5])])
    return table(["Simulated world", "Carbon rule", "Main tree", "Main budget-aware tree", "Main direct", "Tree re-trained",
                  "Budget-aware tree re-trained", "Direct re-trained", "Oracle"],
                 rows, num_cols=tuple(range(1, 9)), highlight_rows=(0,),
                 caption="Emissions (marginal factors) vs the historical TOU tariff, Nov-Dec 2013. 'Main' policies were "
                         "trained with the main simulator and are evaluated in each alternative world (robustness); "
                         "'re-trained' policies are trained and evaluated in that world (same menu and depth).")


# ------------------------------------------------------------------------------------------------ page
css = (PIPE / "lib" / "report_style.css").read_text(encoding="utf-8")
NO_CI_TEXT = ("No carbon-forecast feature enters the tree: in the training data, how much electricity households were "
              "using mattered more than the day's carbon intensity.")
direct_text = (f"Direct optimisation picks exactly the same schedules, so interpretability costs nothing here."
               if same_as_direct else
               f"Direct optimisation instead buys extra High days with cheap night Low days ({direct['high_slots']} High and "
               f"{direct['low_slots']} Low half-hours) and saves {num(interp_cost, 2)} points more than the tree.")

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Carbon-Aware Tariff Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
{css}
</style>
</head>
<body>
<main>
<header class="top">
  <div class="eyebrow">15.095 Machine Learning Under a Modern Optimization Lens &middot; project results</div>
  <h1>Carbon-aware tariffs for London households</h1>
  <p class="lede">We ran the full proposal on Low Carbon London 2013: a next-day carbon-intensity forecast, marginal emission
  factors for the 2013 grid, a demand-response simulator checked against held-out tariff events, and an optimal policy tree
  that picks each day's tariff schedule. Counting emissions at the margin, the tree beats the 2013 time-of-use schedule by
  about one percent of household emissions, and it raises bills.</p>
  <div class="byline">Clement Docteur and Panle Chen &middot; generated {date.today():%d %B %Y} from <code>carbon_tariff_pipeline/</code></div>
  <div class="prices">
    <span class="chip"><i style="background:var(--low)"></i>Low {PRICE['L']:.2f} p/kWh</span>
    <span class="chip"><i style="background:var(--normal)"></i>Normal {PRICE['N']:.2f} p/kWh</span>
    <span class="chip"><i style="background:var(--high)"></i>High {PRICE['H']:.2f} p/kWh</span>
    <span class="chip">Flat (Std) {FLAT_PRICE} p/kWh</span>
  </div>
</header>

<section>
  <div class="head"><div class="stepno">Research question</div>
  <h2>Can an interpretable, carbon-aware pricing policy cut household emissions at a comparable average price?</h2></div>
  <div class="answer">
    <p><b>Yes, slightly.</b> Counting emissions with marginal emission factors (the plants that actually ramp when demand
    changes), a depth-{S3['depth']} policy tree lowers the estimated emissions of the 1,032 trial households by
    <b>{cut(tree['emissions_change_pct'])}</b> of their footprint over the unseen test period, Nov-Dec 2013 (95% simulator
    bootstrap: {cut(BOOT[P[2]][1])} to {cut(BOOT[P[2]][0])}), about {num(saved_kg, 1)} kg CO2 per household over {n_test} days.
    Its time-average price is {num(tree['mean_price'], 2)} p/kWh against a target of {num(T, 2)}. Perfect hindsight saves
    {cut(oracle['emissions_change_pct'])}, so with this menu and the response the trial households actually showed, the
    ceiling is about one percent. Held to the 2013 tariff's own test-period average price
    ({num(S3['historical_mean_price']['test'], 2)} p/kWh), the tree still saves {cut(matched[P[2]]['emissions_change_pct'])}
    with a mean bill change of {signed(matched[P[2]]['bill_change_mean_pct'], 1)}.</p>
    <p><b>Nov-Dec is the best case, and a budget-aware tree recovers the rest.</b> Repeating the whole protocol on four
    2-month test windows from May to December, at the 2013 tariff's own price in each window, the tree's change averages
    {signed(rmean('matched', P[2]))} (from {signed(max(rvals('matched', P[2])))} to {signed(min(rvals('matched', P[2])))})
    against a hindsight ceiling of {signed(rmean('matched', P[5]))}: in {' and '.join(no_event_folds)} its seasonal rule
    switches events off entirely. Giving the tree one more input, the remaining price budget per remaining day, lifts the
    four-window average to <b>{signed(rmean('matched', P[6]))}</b>, close to the ceiling. The budget-aware tree spends
    unused budget in summer and paces itself in winter: in Nov-Dec it saves {cut(paced['emissions_change_pct'])} and the
    price rule overrides it on {adj['tree_paced']} days instead of {adj['tree']}. It also spends more of the budget, so
    the mean bill rises by {signed(paced['bill_change_mean_pct'], 1)}.</p>
    <p><b>Marginal accounting changes the size, not the policy.</b> At the margin, overnight power in 2013 came from coal
    (about {num(night_mef, 0)} g/kWh), while the evening peak was met by gas and pumped storage. Once storage output is charged
    with the overnight coal that refilled it, the evening peak runs at about {num(peak_mef, 0)} g/kWh, above the average
    intensity of those hours ({num(peak_avg, 0)}). So the tree's evening conservation is worth a little more than average
    accounting says ({cut(tree['emissions_change_pct'])} vs {cut(Ra[P[2]]['emissions_change_pct'])}), and the best policy is
    still an evening High block. The policy that suffers is a carbon-only rule that ignores demand response: following the
    marginal factors, it puts High prices at {rule_high}, when almost nobody uses power, and Low prices at {rule_low}, and it
    <i>raises</i> emissions by {cut(-crule['emissions_change_pct'])} (following average intensity: {signed(Ra[P[1]]['emissions_change_pct'])}).</p>
    <p><b>The saving comes from conservation, not timing.</b> High prices cut evening use without moving it elsewhere, and the
    evening peak is when use is highest. {direct_text} A perfect forecast of the marginal factors changes direct optimisation
    by {num(perfect_gain, 2)} points.</p>
    <p><b>A time-average price cap does not protect bills.</b> High blocks fall on the evening peak, so at the main target the
    average price actually paid rises from {num(hist['price_paid'], 2)} to {num(tree['price_paid'], 2)} p/kWh and the mean
    bill by {signed(tree['bill_change_mean_pct'], 1)}. Capping the average price <i>paid</i> at its Jan-Oct historical level
    ({num(paid_hist['trainval'], 2)} p/kWh) limits the increase to {signed(paid[P[2]]['bill_change_mean_pct'], 1)} while still
    saving {cut(paid[P[2]]['emissions_change_pct'])}.</p>
    <p><b>The size depends on the simulator.</b> Under the CVAE simulator, which barely reacts to High prices on held-out
    days, the main tree's saving falls to {cut(sw['cvae']['main_policies_evaluated_here'][P[2]])}; a tree re-trained on the
    CVAE saves {cut(sw['cvae']['retrained_here'][P[2]])}.</p>
  </div>
  <div class="kpis">
    <div class="kpi"><span class="v">{signed(tree['emissions_change_pct'])}</span><span class="l">Policy tree: emissions vs historical TOU (marginal factors), Nov-Dec 2013</span></div>
    <div class="kpi"><span class="v">{signed(Ra[P[2]]['emissions_change_pct'])}</span><span class="l">Same question with average carbon intensity</span></div>
    <div class="kpi"><span class="v">{signed(oracle['emissions_change_pct'])}</span><span class="l">Best possible with this menu (perfect hindsight)</span></div>
    <div class="kpi"><span class="v">{signed(rmean('matched', P[2]))}</span><span class="l">Tree, average over four test windows (May-Dec), price-matched</span></div>
    <div class="kpi"><span class="v">{signed(rmean('matched', P[6]))}</span><span class="l">Budget-aware tree, same four windows (hindsight ceiling {signed(rmean('matched', P[5]))})</span></div>
    <div class="kpi"><span class="v">{signed(crule['emissions_change_pct'])}</span><span class="l">Carbon-only rule following marginal factors</span></div>
    <div class="kpi"><span class="v">{signed(tree['bill_change_mean_pct'], 1)}</span><span class="l">Policy tree: mean household bill vs historical TOU</span></div>
    <div class="kpi"><span class="v">{adj['tree']}/{n_test}</span><span class="l">Test days on which the price rule cancelled the tree's High block</span></div>
  </div>
</section>

<section>
  <div class="head"><div class="stepno">Data and protocol</div><h2>What was used, and when each piece is allowed to look</h2></div>
  <ul class="tight">
    <li><b>Households:</b> 1,032 Low Carbon London households on the 2013 dynamic tariff with &gt;95% of 2013 readings
    (gaps filled), grouped into the four consumption clusters of the CVAE reproduction ({', '.join(str(s) for s in sizes)} households),
    plus 1,032 flat-tariff (Std) households used as a control. Cluster 3 ({sizes[3]} households) carries most of the price response.</li>
    <li><b>Tariff:</b> the authors' 2013 schedule (aligned to the meter half-hours; <code>Tariffs.xlsx</code> is wrong on 17 late-2013 days).</li>
    <li><b>Grid:</b> NESO half-hourly carbon intensity and generation mix, 2009-2013 (the 2013 grid is coal-heavy, mean {EDA['ci_mean_2013']} g/kWh).
    <b>Weather:</b> ERA5 2 m temperature at central London, 2013.</li>
    <li><b>Emissions:</b> marginal emission factors by hour and season (Step 1b), with pumped storage charged at the overnight
    marginal factor. Changes are reported as a share of the households' historical footprint at average intensity.</li>
    <li><b>Timing:</b> the tariff for day <i>d</i> is announced at 12:00 GMT on <i>d</i>&minus;1, so every forecast and policy input uses
    data up to that moment (CI horizon 12-36 h). Realised temperature stands in for a day-ahead forecast.</li>
    <li><b>Splits:</b> train Jan-Aug 2013 ({days['train']} days), validation Sep-Oct ({days['val']}), test Nov-Dec ({days['test']}). Models, the schedule menu,
    tree depth and the price multiplier are chosen on train and validation only; the test period is used once, at the end.</li>
    <li><b>Price target:</b> the historical TOU time-average price over Jan-Oct, {num(T, 3)} p/kWh. The historical tariff averaged
    only {num(S3['historical_mean_price']['test'], 3)} p/kWh in the test period, so a price-matched variant is reported too.</li>
  </ul>
</section>

<section>
  <div class="head"><div class="stepno">Step 1</div><h2>Forecasting next-day carbon intensity</h2></div>
  <p class="col">The day-ahead forecast of average carbon intensity (CI) feeds the policy tree's features and the
  average-accounting comparison. Each model predicts the deviation of every half-hour's CI from the mean of the last known
  24 h, from calendar features, lagged CI (latest known day, 2 and 7 days earlier), lagged fuel shares and lagged generation.
  Training uses NESO data from 2009, refitted monthly. On validation the tree ensembles cut the error of persistence by
  {num(100 * (1 - best_ml_val / pers['val']['MAE']), 0)}% ({num(best_ml_val, 1)} vs {num(pers['val']['MAE'], 1)} g/kWh), and
  {LAB1[sel]} was selected. On the test period the gain shrinks: XGBoost {num(M1['xgb']['test']['MAE'], 1)}, random forest
  {num(M1['rf']['test']['MAE'], 1)}, persistence {num(pers['test']['MAE'], 1)} g/kWh. All models rank the half-hours within a
  day poorly (rank correlation {num(M1['xgb']['test']['within_day_spearman'], 2)} for XGBoost vs
  {num(pers['test']['within_day_spearman'], 2)} for persistence), most likely because the swings follow wind output and plant
  availability, which lagged data do not anticipate. Adding temperature does not help: validation MAE
  {num(wx['no weather']['val']['MAE'], 1)} &rarr; {num(wx['with ERA5 temperature']['val']['MAE'], 1)}, test
  {num(wx['no weather']['test']['MAE'], 1)} &rarr; {num(wx['with ERA5 temperature']['test']['MAE'], 1)} g/kWh (2013-only
  training, as ERA5 covers 2013 only).</p>
  {t_step1()}
  {img(FIG / 'step1_ci_forecast.png', 'Forecast error by model and one test week of forecasts',
       'Forecast error by model (a) and one test week (b). The forecast follows the level of the coming day but misses most within-day swings.')}
</section>

<section>
  <div class="head"><div class="stepno">Step 1b</div><h2>Marginal emission factors: which plants respond to demand</h2></div>
  <p class="col">A tariff changes demand at the margin, so its emissions depend on the plants that ramp, not on the average
  mix. For each hour of day and season we regress the half-hourly change in system emissions on the change in generation
  (the method of Hawkes, 2010). NESO counts pumped-storage output as zero-carbon, but that energy was pumped overnight; in the
  main factors, storage output carries the overnight marginal factor divided by a {num(100 * S1b['round_trip_efficiency'], 0)}%
  round-trip efficiency ({num(S1b['storage_emission_factor'], 0)} g/kWh). Factors used to score schedules are estimated on 2013;
  the factors available at decision time are re-estimated each month on the previous 12 months (test-period error
  {num(S1b['checks']['forecast_mae_test_g_per_kwh'], 0)} g/kWh).</p>
  <p class="col">The 2013 margin looks nothing like the average. Overnight, coal met {num(100 * fs['night 00-05']['COAL'], 0)}% of
  changes in generation and the marginal factor is about {num(night_mef, 0)} g/kWh. At the evening peak, gas met
  {num(100 * fs['peak 16-20']['GAS'], 0)}% and pumped storage {num(100 * fs['peak 16-20']['STORAGE'], 0)}%: about
  {num(peak_mef_u, 0)} g/kWh if storage is counted as zero-carbon, {num(peak_mef, 0)} once it is charged for its overnight
  coal. Over the test period the factors average {num(mt['mef_adjusted'], 0)} g/kWh (unadjusted {num(mt['mef_unadjusted'], 0)}),
  against an average intensity of {num(mt['average_ci'], 0)}. The margin also moved between years, so factors from
  2009-2012 would have overstated 2013.</p>
  <div class="two">{t_mef()}{t_fuels()}</div>
  {img(FIG / 'step1b_marginal_emissions.png', 'Marginal emission factors by hour and by year',
       'Marginal emission factors by hour of day, Oct-Mar 2013, with and without the pumped-storage adjustment, against the average intensity of the test period (a); annual marginal factor and average intensity, 2009-2013 (b). Bands: plus or minus two standard errors.')}
</section>

<section>
  <div class="head"><div class="stepno">Step 2</div><h2>Simulating demand under other tariffs</h2></div>
  <p class="col">The proposal planned to adapt Br&eacute;g&egrave;re and Bessa's CVAE. Our reproduction found its mean profiles accurate but
  its tariff response unreliable: the latent variable collapses, the response changes sign between training runs, and its
  "side effects" around price blocks are not in the data. We therefore use it only as a sensitivity case. The main simulator
  is a <b>response model</b>: for each cluster, the log ratio of ToU to flat-tariff consumption is regressed on slot &times; day-type
  intercepts, temperature, and tariff terms (in-block High and Low effects by time of day, and edge effects). Flat-tariff households
  share the weather, holidays and season, so these cancel out. A schedule's simulated consumption is the observed consumption
  times the modelled ratio between that schedule and the historical one; only the tariff response is modelled.</p>
  <p class="col">Three specifications were compared on Sep-Oct event days; the one with only the edge effects that the
  reproduction had verified wins (2 h dip before Low blocks, the half-hour after a Low block, 2 h before High blocks).
  Inside blocks, cluster 3 cuts use by {num(-eff3['High|peak 16-20']['pct'], 0)}% under evening High prices and raises it by
  {num(eff3['Low|afternoon 12-16']['pct'], 0)}% under afternoon Low prices; in the other three clusters no in-block effect
  exceeds {num(OTHER_MAX, 0)}%.</p>
  <div class="two">{t_spec()}<div></div></div>
  {t_step2()}
  {img(FIG / 'step2_demand_response.png', 'Estimated tariff effects and held-out validation',
       'Estimated in-block effects (a). On held-out Nov-Dec event days (b) the response model matches the observed High and Low effects for all households and for cluster 3; the CVAE ensemble barely reacts to High prices.')}
</section>

<section>
  <div class="head"><div class="stepno">Step 3</div><h2>Learning an interpretable tariff policy</h2></div>
  <p class="col">Each day's action is one schedule from a small menu. Its model-based cost is the change in emissions,
  C<sub>d</sub>(z) = &Sigma;<sub>t</sub> MEF<sub>dt</sub> &middot; y&#770;<sub>dt</sub>(z) / 1000 kg CO2, with the 2013 marginal
  factors and simulated consumption on the fitting days; the direct optimiser and the carbon-only rule use the decision-time
  factors instead. The policy sees only decision-time features (CI forecast summaries, temperature, weekend or holiday,
  recent consumption). The price constraint enters as a Lagrangian penalty &lambda; &middot; price, and the proposal's
  feasibility rule keeps the test-period average attainable: a day keeps its recommendation only if the target could still be
  met with every remaining day on the Normal tariff; otherwise it falls back to the next affordable schedule in the leaf's
  ranking. The tree is an exact optimal prescriptive tree (Lecture 11) of depth 1-3; depth and &lambda; were chosen on
  validation, and the tree was refitted on Jan-Oct.</p>
  <div class="two">
    {t_menu()}
    <div style="display:flex;flex-direction:column;gap:10px;min-width:0">
      {tree_html(S3['tree'], menu_names, S3['features'], S3['feature_formats'])}
      <p class="note">Validation emissions (marginal-factor-weighted kg CO2 per household, Sep-Oct) at the best &lambda; for each depth:
      {', '.join(f"depth {d}: {num(g['val_emissions_kg'], 2)} ({g['leaves']} leaves)" for d, g in S3['selection_by_depth'].items())}.
      Depth {S3['depth']} is the shallowest within 0.01% of the best (&lambda; = {S3['lambda_tree']} kg per p/kWh).</p>
    </div>
  </div>
  <p class="col">The tree learned a seasonal rule: recent consumption above the threshold marks the darker, colder half of
  the year, when an evening High block saves the most. On {asks} the tree asks for <i>{esc(prescribed)}</i>. The price budget
  covers {S3['tree_event_days']} of them; after that the price rule cancels the event ({adj['tree']} of {n_test} days).
  {direct_text} {NO_CI_TEXT if not ci_features else "Carbon-forecast features used by the tree: " + esc(", ".join(ci_features)) + "."}</p>
  <h3 class="sub">A budget-aware tree</h3>
  <p class="col">The tree above cannot see how much of the price budget is left, so it asks for a High block every winter
  day and the price rule decides which days get one. The budget-aware tree has one more input: the remaining budget per
  remaining day, i.e. how much the average daily price can still exceed an all-Normal day ({num(PRICE['N'], 2)} p/kWh) without
  breaking the target. It is computed each day from the prices already announced. To train it, every fitting day is
  repeated at {len(S3['slack_grid'])} budget levels from 0 to {num(max(S3['slack_grid']), 1)} p/kWh. Each level is priced with
  the &lambda; that would make the policy spend that much per day on the fitting days (from {num(max(S3['slack_lambdas']), 4)}
  at zero budget to {num(min(S3['slack_lambdas']), 4)} at the top). A budget-aware direct optimiser uses the same mapping.
  Depth {S3['paced_depth']} was chosen on validation.</p>
  <div class="two">
    {tree_html(S3['paced_tree'], menu_names, S3['paced_features'], S3['paced_feature_formats'], unit='training cases (day &times; budget level)')}
    <p class="note">On low-consumption days the tree earns budget with a cheap night Low block unless plenty of budget is
    left; on high-consumption days it chooses the length of the evening High block according to the budget. In Nov-Dec it
    chose {paced_counts}, saving {cut(paced['emissions_change_pct'])} (bootstrap {cut(BOOT[P[6]][1])} to
    {cut(BOOT[P[6]][0])}) against {cut(tree['emissions_change_pct'])} for the tree above, with the price rule stepping in
    on {adj['tree_paced']} days. Mean bill: {signed(paced['bill_change_mean_pct'], 1)}.</p>
  </div>
  {t_results()}
  {img(FIG / 'step3_policy_results.png', 'Emissions and bill changes by policy',
       'Emissions (a, marginal factors) and household bills (b) relative to the historical TOU tariff, test period. Bootstrap bars reflect uncertainty in the response model only.')}
  {img(FIG / 'step3_schedules.png', 'Price schedules over the test period and mean profiles',
       'Price schedules over the test period (a-c). The historical tariff placed a few events at varied times. The tree spends the budget on evening High blocks in early November, then the rule cancels its events; the budget-aware tree earns budget with night Low blocks through November and spends it on evening High blocks in December. Mean consumption (d) drops in the High window. Panel e compares the marginal factors with the average intensity over the day.')}
</section>

<section>
  <div class="head"><div class="stepno">Marginal vs average</div><h2>What changes when emissions are counted at the margin</h2></div>
  <ul class="tight">
    <li><b>The policy barely changes.</b> Trained on average intensity, the tree asks for {esc(max(S3a['tree_prescribed_test_counts'], key=S3a['tree_prescribed_test_counts'].get))}
    and, scored with marginal factors, saves {cut(cross_a[P[2]]['marginal'])}; trained on marginal factors it saves
    {cut(cross[P[2]]['marginal'])}. Both are evening conservation policies.</li>
    <li><b>The measured saving is a little larger</b> ({cut(tree['emissions_change_pct'])} vs {cut(Ra[P[2]]['emissions_change_pct'])}),
    because the storage-adjusted evening margin is dirtier than the evening average.</li>
    <li><b>Pumped storage decides the sign of that difference.</b> Counting storage as zero-carbon, as NESO's intensity does,
    makes the evening peak look clean ({num(peak_mef_u, 0)} g/kWh) and shrinks the tree's saving to
    {cut(cross[P[2]]['marginal_unadjusted'])}.</li>
    <li><b>The carbon-only rule is the policy most affected.</b> Following marginal factors, it moves its High block to the
    night ({rule_high}), when consumption is lowest, and its Low block to {rule_low}, when far more power is used. It goes from
    {signed(Ra[P[1]]['emissions_change_pct'])} to {signed(crule['emissions_change_pct'])}: placing prices by carbon intensity
    alone, without a demand model, can backfire.</li>
  </ul>
  {t_cross()}
</section>

<section>
  <div class="head"><div class="stepno">Rolling evaluation</div><h2>Does the result hold outside one winter?</h2></div>
  <p class="col">The main test covers 61 winter days. To see how much rests on that window, we repeat the whole protocol on
  four 2-month test windows (May-Jun, Jul-Aug, Sep-Oct, Nov-Dec). For each window, the response model is refitted on the
  days before it, the menu is chosen on the training days, and depth and &lambda; are chosen on the 2 months just before
  the window, so no window sees its own data. Early windows have short histories (59 training days for May-Jun).</p>
  <ul class="tight">
    <li><b>Nov-Dec is the best window.</b> At each window's own target the tree's change is
    {', '.join(f"{f['label']} {signed(f['main'][P[2]]['emissions_change_pct'])}" for f in roll)} (mean {signed(rmean('main', P[2]))}).
    The targets differ from what the 2013 tariff actually charged, so the price-matched comparison is the fairer one: there
    the tree averages {signed(rmean('matched', P[2]))} and perfect hindsight {signed(rmean('matched', P[5]))}.</li>
    <li><b>In summer the tree does nothing.</b> In {' and '.join(no_event_folds)} the tree never prescribes an event. Its
    threshold on recent consumption is learned on the colder months before each window, so every summer day falls on the
    "Normal" side. The tree then only gains from dropping the 2013 tariff's events, and in May-Jun, when the 2013 tariff
    ran many High blocks ({num(roll[0]['historical_test_price'], 2)} p/kWh on average), it loses to it. Hindsight shows that
    summer High blocks would still have saved emissions.</li>
    <li><b>The budget-aware tree uses that summer budget.</b> Price-matched, it has events on
    {', '.join(f"{f['matched_paced_event_days']} days in {f['label']}" for f in roll)}, and its change is
    {', '.join(f"{f['label']} {signed(f['matched'][P[6]]['emissions_change_pct'])}" for f in roll)}
    (mean {signed(rmean('matched', P[6]))}, against {signed(rmean('matched', P[5]))} for hindsight).</li>
    <li><b>Direct optimisation matches the tree in {same_direct} of {len(roll)} windows</b>, so the cost of interpretability
    stays small throughout.</li>
    <li><b>The carbon-only rule raises emissions in every window</b> ({signed(min(rvals('main', P[1])))} to
    {signed(max(rvals('main', P[1])))}).</li>
  </ul>
  {t_rolling()}
  {img(FIG / 'step3_rolling.png', 'Emissions change in four rolling test windows',
       'Emissions change vs the historical TOU tariff in each rolling test window, at the window target (a) and at the 2013 tariff price in that window (b). Marginal factors.')}
</section>

<section>
  <div class="head"><div class="stepno">Sensitivity</div><h2>How much the answer depends on the price cap and the simulator</h2></div>
  <ul class="tight">
    <li><b>Price-matched to the historical test period</b> ({num(S3['historical_mean_price']['test'], 3)} p/kWh): the tree's
    change is {signed(matched[P[2]]['emissions_change_pct'])} and direct optimisation's {signed(matched[P[3]]['emissions_change_pct'])}.
    So {num(100 * budget_share, 0)}% of the main result comes from spending a larger price budget than the 2013 tariff did in
    Nov-Dec. At this target the tree's mean bill changes by {signed(matched[P[2]]['bill_change_mean_pct'], 1)}.</li>
    <li><b>Flat-tariff cap</b> ({FLAT_PRICE} p/kWh): tree {signed(flat[P[2]]['emissions_change_pct'])}, oracle
    {signed(flat[P[5]]['emissions_change_pct'])}. A larger budget buys more High blocks and more conservation.</li>
    <li><b>Cap on the average price paid</b> (&le; {num(paid_hist['trainval'], 2)} p/kWh, the Jan-Oct historical value):
    tree {signed(paid[P[2]]['emissions_change_pct'])} with mean bill {signed(paid[P[2]]['bill_change_mean_pct'], 1)}. Bills still
    rise because the 2013 tariff charged only {num(paid_hist['test'], 2)} p/kWh in Nov-Dec; a cap at that level would be
    bill-neutral by construction.</li>
    <li><b>Demand response:</b> halving the response or scaling it by 1.5 scales the savings almost proportionally
    ({signed(sw['x0.5']['main_policies_evaluated_here'][P[2]])} and {signed(sw['x1.5']['main_policies_evaluated_here'][P[2]])}
    for the tree). Dropping the edge effects changes little; the full model, with post-block and day-level terms, gives
    {signed(sw['full']['retrained_here'][P[2]])} for a re-trained tree.</li>
    <li><b>CVAE simulator:</b> the main tree's saving falls to {signed(sw['cvae']['main_policies_evaluated_here'][P[2]])}: the CVAE
    barely responds to High prices on held-out days (Step 2). Re-trained on the CVAE, the tree reaches
    {signed(sw['cvae']['retrained_here'][P[2]])} and the hindsight oracle {signed(sw['cvae']['retrained_here'][P[5]])}.</li>
  </ul>
  {t_constraints()}
  {t_worlds()}
  {img(FIG / 'step3_sensitivity.png', 'Sensitivity of emission changes',
       'Emissions (marginal factors) vs the historical TOU tariff under each price constraint and each simulated demand response (policies re-trained in each case).')}
</section>

<section>
  <div class="head"><div class="stepno">Solver check</div><h2>Would a stronger tree solver change the answer?</h2></div>
  <p class="col">The proposal planned Interpretable AI's Optimal Policy Trees (OPT). Our trees come from our own search,
  which tries every threshold at the last split but only 12 quantile thresholds at the splits above it. If that shortcut
  missed better trees, a stronger solver such as OPT could find them. We refitted every Step 3 tree ({len(tc)} test windows,
  fixed and budget-aware, depths 1-3: {TC['n_cases']} fits) on its own training set with a finer search: every threshold at
  every level for depths 1 and 2, which gives the true optimum, and {TC['fine_depth3_quantiles']} thresholds above the last
  level for depth 3 (the exhaustive depth-3 search takes hours). The refits with our original search reproduce Step 3's
  trees and test results exactly.</p>
  <ul class="tight">
    <li><b>Depth 1 is already exact.</b> Of the {len(tc_chosen)} trees chosen in Step 3, {n_depth1} have depth 1, including
    the main Nov-Dec tree, so no solver can improve them.</li>
    <li><b>At depth 2 our search gets {num(min(share_d[2]), 1)}-{num(max(share_d[2]), 0)}% of the optimum's training gain</b>
    (gain = cost of the best single schedule for every day minus the tree's cost). At depth 3 it gets
    {num(min(share_d[3]), 1)}-{num(max(share_d[3]), 1)}% of the finer search's. The finer grid is not a superset of ours, so
    it can do slightly worse.</li>
    <li><b>The extra training gain does not carry over to the test windows.</b> At the depths chosen in Step 3 the finer
    trees give the same test result in {len(tc_chosen) - len(tc_changed)} of {len(tc_chosen)} cases, and slightly smaller
    savings in the others ({'; '.join(f"{lab} {KIND[k]}: {signed(b, 3)} vs {signed(a, 3)}" for lab, k, a, b in tc_changed)}).</li>
    <li><b>Interpretable AI itself could not run on this PC.</b> IAI 3.2.2 and its licence are installed, but Windows Smart
    App Control blocks libraries bundled with its system image. <code>06_tree_check.py --export</code> writes the
    {TC['n_cases']} training sets, and <code>06_iai_fit.jl</code> fits OPT on them (same depth, minimum leaf size
    {TC['min_leaf']}, no complexity penalty, so only the search differs) on any machine where IAI runs. Rerunning the check
    then adds IAI columns to this table. {"IAI fits are included below." if HAS_IAI else "That run has not been done yet."}</li>
  </ul>
  {t_tree_check()}
</section>

<section>
  <div class="head"><div class="stepno">Changes from the proposal</div><h2>Where we departed from the plan, and why</h2></div>
  <ul class="tight">
    <li><b>Emission metric.</b> The proposal priced emissions at the average carbon intensity. A tariff changes demand at the
    margin, so the main results use marginal emission factors (storage-adjusted); average intensity is kept as a comparison.</li>
    <li><b>Main simulator.</b> The CVAE is kept as a sensitivity case; the main simulator is a cluster-level response model with a
    flat-tariff control, because the reproduction showed the CVAE's response to schedule changes is unreliable and the response
    model matches held-out tariff events.</li>
    <li><b>Action costs use measured factors</b> on the fitting days. Emission factors are not affected by the tariff, so the
    tree learns the mapping from forecasts to schedules against the outcomes it will face; the direct optimiser uses
    decision-time factors and forecasts, as planned.</li>
    <li><b>Lagrangian multiplier.</b> The average-price constraint is priced into each action's cost; &lambda; is a
    hyperparameter chosen with the tree depth on validation. Tuning &lambda; to hit the target on the training days made
    coarse trees spend nothing on the validation period.</li>
    <li><b>Feasibility rule floor.</b> "Attainable" is checked assuming every remaining day could be an all-Normal day, so
    once the budget is spent the rule cancels events. Using the cheapest schedule (a night Low block) as the floor instead let
    policies overspend early and then forced Low days, which add consumption.</li>
    <li><b>Budget-aware tree.</b> Added after the rolling evaluation showed the original tree switching events off in summer:
    the remaining budget per remaining day becomes a tree input, trained over a grid of budget levels.</li>
    <li><b>Extra rows we added:</b> direct optimisation with a perfect carbon forecast (value of better forecasts), a hindsight
    oracle (ceiling), a target matched to the historical test-period price, and a cap on the average price paid.</li>
    <li><b>Implementation.</b> Interpretable AI's Optimal Policy Trees are installed but blocked on this PC (Smart App
    Control). The tree is our own search over splits of depth &le; 3 (every threshold at the last level, 12 quantile
    thresholds above). The solver check shows that a finer search does not improve the test results.</li>
  </ul>
</section>

<section>
  <div class="head"><div class="stepno">Limitations</div><h2>What these numbers can and cannot say</h2></div>
  <ul class="tight">
    <li>All counterfactual emissions are model-based. The response model reproduces observed in-block and edge effects on held-out
    days, but new schedules are combinations of effects estimated from 2013 events.</li>
    <li>Marginal factors are short-run statistical estimates (hour &times; season regressions on system-level changes). They
    ignore transmission losses, interconnector effects beyond NESO's import accounting and any long-run effect on what gets
    built; the storage adjustment assumes a {num(100 * S1b['round_trip_efficiency'], 0)}% round-trip efficiency and overnight pumping.</li>
    <li>Each test window is 2 months long, and the early rolling windows have short training histories. The bootstrap
    intervals cover the main window only.</li>
    <li>Households are represented by four clusters; bill changes assume each household responds like its cluster.</li>
    <li>Weather enters as realised temperature (a perfect day-ahead forecast); wind speed, a major driver of CI swings, is not in the data.</li>
  </ul>
</section>

<section>
  <div class="head"><div class="stepno">Reproduce</div><h2>Files and commands</h2></div>
  <p class="col">Run from <code>carbon_tariff_pipeline/</code> with <code>python -P</code>, in order (about 20 minutes in total;
  Step 1 takes about 5, each Step 3 run about 6). Step 3 runs twice: <code>--metric marginal</code> (main) and <code>--metric average</code>.
  The solver check runs after Step 3 and adds about 13 minutes. The
  pipeline reuses the reproduction's household data, clusters and trained CVAEs from <code>CVAE_reproduction/</code>.</p>
  <div class="files">
    <span>scripts/01_ci_forecast.py</span><span>Step 1: CI forecasts, model selection, monthly refits, weather test</span>
    <span>scripts/01b_marginal_emissions.py</span><span>Step 1b: marginal emission factors (measured and decision-time)</span>
    <span>scripts/02_demand_response.py</span><span>Step 2: response model specs, bootstrap, CVAE ensemble, held-out checks</span>
    <span>scripts/03_policy.py</span><span>Step 3: menu, policy tree, baselines, feasibility rule, evaluation, sensitivity</span>
    <span>scripts/06_tree_check.py</span><span>Solver check: finer tree search; <code>--export</code> writes the inputs for IAI</span>
    <span>scripts/06_iai_fit.jl</span><span>Fits Interpretable AI's OPT on those inputs (needs a machine where IAI runs)</span>
    <span>scripts/04_figures.py</span><span>All figures</span>
    <span>scripts/05_report.py</span><span>This page</span>
    <span>lib/</span><span>Data loaders and time conventions, simulators, tree solver and price rule</span>
    <span>outputs/</span><span>JSON results of each step, forecasts, emission factors and test schedules</span>
  </div>
</section>
</main>
</body>
</html>
"""
(PIPE / "report.html").write_text(html, encoding="utf-8")
print("report.html written,", len(html) // 1024, "KB")
