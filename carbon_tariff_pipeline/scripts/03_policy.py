"""Step 3: learn an interpretable carbon-aware tariff policy and evaluate it on held-out 2013 days.

Model-based action cost of schedule a on day d (kg CO2 per household):
    C_d(a) = (1/1000) * sum_h F_dh * y_dh(a),   y_dh(a) = sum_c w_c * Y_c,dh * exp(m_c(a)_h - m_c(hist)_dh)
(Y_c = observed cluster consumption, m = the simulator's log tariff effect). F is the emission factor chosen with
--metric: 'marginal' (main; Step 1b's storage-adjusted marginal factors) or 'average' (NESO carbon intensity, CI).
Emission changes are F-weighted consumption changes vs the historical tariff, reported in kg and as % of the
households' historical footprint at average intensity (so both metrics share one denominator).
Average-price constraint: time-average price over the evaluation period <= target T, handled by a Lagrangian
(cost + lambda * price, lambda chosen on validation) plus the online feasibility rule, whose floor is the all-Normal
day: once the budget is spent, events are cancelled.

Protocol (a Fold): menu (<= 6 schedules) chosen greedily on the training days; tree depth (1-3) and Lagrange multiplier
chosen on the 2-month validation window (validation emissions under the feasibility rule); final tree refitted on
training + validation; the 2-month test window is used once. Target = historical TOU mean price over training + validation.
Main analysis: test Nov-Dec (train Jan-Aug, validation Sep-Oct), with
  policies: historical TOU, simple carbon-based tariff (no demand model), optimal policy tree, direct model-based
  optimisation (forecast factors + consumption forecast; also with perfect factors), perfect-hindsight oracle;
  sensitivity: price targets, a cap on the average price paid, demand-response worlds (x0.5, x1.5, in-block only,
  full model, CVAE), weekly-block bootstrap of the simulator.
Budget-aware variants: the tree and the direct optimiser also see the remaining budget per remaining day (extra price
above an all-Normal day that the target still allows). Training repeats each fitting day over a grid of budget levels,
each priced with the lambda that would make the policy spend that much on the fitting days.
Rolling evaluation: the same protocol on four test windows (May-Jun, Jul-Aug, Sep-Oct, Nov-Dec), each with the
response model refitted on the days before it.

Usage:   python -P scripts/03_policy.py [--metric marginal|average]
Outputs: outputs/step3_results{,_average}.json, outputs/step3_test_schedules{,_average}.npz (inputs of figure 4)
"""
from __future__ import annotations

import argparse
import calendar
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import (DATES_2013, FLAT_PRICE, H, OUT, cluster_aggregates, day_type, hours,  # noqa: E402
                        load_households, load_tariff_2013, load_temps_2013, make_log, price_paid, save_json,
                        schedule_price, split_of)

import numpy as np  # noqa: E402

from lib.demand import CVAEEnsemble, ResponseModel, tariff_regressors  # noqa: E402
from lib.policy import (NORMAL, POLICIES, PolicyTree, candidate_pool, carbon_rule_counts,  # noqa: E402
                        carbon_rule_schedule, describe, feasibility_rule, make_schedule, pick_near_best, tune_lambda)

_ap = argparse.ArgumentParser()
_ap.add_argument("--metric", choices=["marginal", "average"], default="marginal")
# MVP integration (2026-11-14): 'lagged' replaces the realised same-day temperature (not known when the tariff is
# announced at 12:00 GMT on d-1) by the mean temperature of the 24 h ending 12:00 GMT on d-1. Default = original.
_ap.add_argument("--temperature", choices=["realised", "lagged"], default="realised")
_ap.add_argument("--tag", default="", help="extra output-file suffix")
_ARGS = _ap.parse_args()
METRIC, TEMPERATURE, TAG = _ARGS.metric, _ARGS.temperature, _ARGS.tag
SUFFIX = ("" if METRIC == "marginal" else f"_{METRIC}") + (f"_{TAG}" if TAG else "")
log, _ = make_log()
CLIP = np.log(4.0)            # simulated consumption ratios are clipped to [1/4, 4]
K_MENU = 6
LAMBDAS = [0.0, 0.002, 0.004, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.013, 0.016, 0.02, 0.03]   # kg per (p/kWh)
TEMP_FEATURE = {"realised": "temperature, day mean (C)", "lagged": "temperature, 24 h to 12:00 on d-1 (C)"}[TEMPERATURE]
FEATURES = {"forecast CI, day mean (g/kWh)": "{:.0f} g/kWh", "forecast CI, 16:00-21:00 (g/kWh)": "{:.0f} g/kWh",
            "forecast CI, evening minus night (g/kWh)": "{:.0f} g/kWh", "forecast CI, max - min (g/kWh)": "{:.0f} g/kWh",
            "forecast dirtiest 3 h, start hour (GMT)": "{:.1f} h", TEMP_FEATURE: "{:.1f} &deg;C",
            "weekend or holiday": "{:.1f}", "recent consumption (kWh/half-hour)": "{:.3f}"}
NIGHT_LOW = (5, 10)           # Low 02:00-05:00, the menu's night Low block
SLACK_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 9.5]     # budget levels (p/kWh above Normal per remaining day)
PACED_FEATURES = {**FEATURES, "remaining budget per remaining day (p/kWh above Normal)": "{:.2f} p/kWh"}
MAIN_TEST_MONTH = 11          # main analysis: test Nov-Dec
ROLLING_TEST_MONTHS = (5, 7, 9, 11)

# ------------------------------------------------------------------------------------------------ data
hh = load_households(std=False)
LOW_H, HIGH_H, PRICE_H = load_tariff_2013()
ALL = np.arange(365)
CIZ = np.load(OUT / "ci_forecasts_2013.npz")
CI, CIF = CIZ["actual"], CIZ["selected"]                 # average CI: realised, and the Step 1 forecast (tree features)
MEFZ = np.load(OUT / "mef_2013.npz")
FACTORS = {"marginal": MEFZ["realised"], "marginal_unadjusted": MEFZ["realised_unadjusted"], "average": CI}
FAC = FACTORS[METRIC]                                    # emission factor of costs and evaluation (g/kWh)
FAC_F = {"marginal": MEFZ["forecast"], "average": CIF}[METRIC]   # its decision-time forecast
CL = hh["cluster"]
SIZES, W, YC, YALL = cluster_aggregates(hh)
DAYTYPE = day_type(DATES_2013)
THETA = np.load(OUT / "response_theta.npz")


def lagged_temperature(temp):
    """Mean temperature of the 48 half-hours ending 12:00 GMT on d-1 (slot 24 of d-1 ends at 12:00), i.e. the
    information available at the day-ahead announcement. 2013-01-01 has no such data (ERA5 starts 2013-01-01) and
    takes 2013-01-02's value; 2013-01-02 uses the 25 available half-hours (both are training days)."""
    flat = temp.ravel()
    out = np.empty(len(temp))
    for d in range(len(temp)):
        end = (d - 1) * H + 24
        out[d] = flat[max(0, end - 47): end + 1].mean() if end >= 0 else np.nan
    out[0] = out[1]
    return out


def features():
    temp = load_temps_2013()
    win = np.stack([np.convolve(r, np.ones(6), "valid") for r in CIF])
    daily = YALL.mean(1)
    f = np.zeros((365, len(FEATURES)))
    f[:, 0] = CIF.mean(1)
    f[:, 1] = CIF[:, hours(16, 21)].mean(1)
    f[:, 2] = f[:, 1] - CIF[:, hours(0, 6)].mean(1)
    f[:, 3] = CIF.max(1) - CIF.min(1)
    f[:, 4] = (win.argmax(1) - 1) / 2.0                     # window starting at slot i begins at (i-1)/2 h GMT
    f[:, 5] = temp.mean(1) if TEMPERATURE == "realised" else lagged_temperature(temp)
    f[:, 6] = DAYTYPE
    f[:, 7] = [daily[max(0, d - 8): max(1, d - 1)].mean() for d in range(365)]   # days d-8..d-2
    return f


X = features()


# ------------------------------------------------------------------------------------------------ worlds (simulators)
class World:
    """A demand-response simulator and the effect of the historical schedule, which counterfactuals are relative to."""

    def __init__(self, model):
        self.model = model
        self.m_hist = self.m(LOW_H, HIGH_H, ALL, timeline=True)

    def m(self, low, high, days, timeline=False):
        """(4, n_days, 48) log effect of the schedules low/high (n_days, 48) on `days`."""
        return np.stack([self.model.log_effect(c, low, high, days, timeline) for c in range(4)])

    def m_menu(self, menu):
        """(4, 365, A, 48) log effect of each menu schedule on every day."""
        return np.stack([self.m(np.tile(s["low"], (365, 1)), np.tile(s["high"], (365, 1)), ALL) for s in menu], axis=2)

    def ratio(self, m, days):
        return np.exp(np.clip(m - self.m_hist[:, days], -CLIP, CLIP))


def response_world(spec="main", scale=1.0, fit="janoct"):
    """Response-model world; fit = 'janoct' (fitted on Jan-Oct) or 'beforeMM' (days before month MM)."""
    model = ResponseModel(spec, scale=scale)
    model.theta = {c: THETA[f"{spec}_{fit}_c{c}"] for c in range(4)}
    return World(model)


def costs(world, M, menu):
    """Per day and menu action: emissions with the measured factor (E), with the decision-time factor and consumption
    forecasts (Ef), with a perfect factor forecast but forecast consumption, and the average price paid."""
    Y = np.einsum("c,cdh,cdah->dah", W, YC, world.ratio(M, ALL[:, None]))
    det = YC * np.exp(-world.m_hist)                       # consumption forecast: de-tariffed mean of the same day
    base = np.zeros_like(YC)                               # type over days d-15..d-2
    for d in range(365):
        prev = np.arange(max(0, d - 15), max(1, d - 1))
        same = prev[DAYTYPE[prev] == DAYTYPE[d]]
        base[:, d] = det[:, same if len(same) else prev].mean(1)
    Yf = np.einsum("c,cdh,cdah->dah", W, base, np.exp(M))
    p = np.stack([schedule_price(s["low"], s["high"]) for s in menu])          # (A, 48)
    return {"E": np.einsum("dh,dah->da", FAC, Y) / 1000, "Ef": np.einsum("dh,dah->da", FAC_F, Yf) / 1000,
            "Ef_perfect_ci": np.einsum("dh,dah->da", FAC, Yf) / 1000,
            "paid": price_paid(p[None], Y, axis=2), "paid_f": price_paid(p[None], Yf, axis=2)}


# ------------------------------------------------------------------------------------------------ folds and evaluation
class Fold:
    """Chronological split of 2013: training days, a 2-month validation window, then a 2-month test window
    starting in month `test_month`. Holds the test-day households and the historical (observed) reference."""

    def __init__(self, test_month):
        m = DATES_2013.month
        self.test_month = test_month
        self.train = np.where(m < test_month - 2)[0]
        self.val = np.where((m >= test_month - 2) & (m < test_month))[0]
        self.test = np.where((m >= test_month) & (m < test_month + 2))[0]
        self.trainval = np.concatenate([self.train, self.val])
        self.label = f"{calendar.month_abbr[test_month]}-{calendar.month_abbr[test_month + 1]}"
        self.Y = hh["Y_tou"][:, self.test].astype(float)    # households on the test days, for bills
        self.ref = evaluate(self, None, LOW_H[self.test], HIGH_H[self.test])
        self.footprint = emissions(self, self.ref["y"], CI)  # historical footprint at average intensity: % denominator


def test_consumption(fold, ratio):
    return np.einsum("c,cdh->dh", W, YC[:, fold.test] * ratio)


def emissions(fold, y, factor=None):
    """Factor-weighted test-window consumption, kg CO2 per household (differences between schedules = emissions change)."""
    return float(((FAC if factor is None else factor)[fold.test] * y).sum() / 1000)


def evaluate(fold, world, low, high):
    """Simulated test-window outcomes of a schedule sequence; world=None means the historical (observed) outcome."""
    te = fold.test
    ratio = np.ones((4, len(te), H)) if world is None else world.ratio(world.m(low, high, te), te)
    y = test_consumption(fold, ratio)
    price = schedule_price(low, high)
    return {"emissions_kg": emissions(fold, y), "mean_price": float(price.mean()), "price_paid": float(price_paid(price, y)),
            "kwh": float(y.sum()), "mean_daily_peak_kwh": float(y.max(1).mean()), "profile": y.mean(0), "y": y,
            "bills": np.einsum("ndh,ndh->n", fold.Y, (price[None] * ratio)[CL]) / 100,           # GBP per household
            "high_slots": int(high.sum()), "low_slots": int(low.sum())}


def summarise(fold, res):
    ref = fold.ref
    ch = 100 * (res["bills"] / ref["bills"] - 1)
    rel = lambda k: 100 * (res[k] / ref[k] - 1)
    change = res["emissions_kg"] - ref["emissions_kg"]
    return {"emissions_change_pct": 100 * change / fold.footprint, "emissions_change_kg_per_hh": change,
            "mean_price": res["mean_price"], "price_paid": res["price_paid"], "kwh_change_pct": rel("kwh"),
            "daily_peak_change_pct": rel("mean_daily_peak_kwh"),
            "bill_change_mean_pct": float(100 * (res["bills"].mean() / ref["bills"].mean() - 1)),
            "bill_change_p10_pct": float(np.percentile(ch, 10)), "bill_change_median_pct": float(np.median(ch)),
            "bill_change_p90_pct": float(np.percentile(ch, 90)), "high_slots": res["high_slots"], "low_slots": res["low_slots"]}


def evaluate_all(fold, world, scheds):
    raw = {p: (fold.ref if p == POLICIES[0] else evaluate(fold, world, *s)) for p, s in scheds.items()}
    return {p: summarise(fold, r) for p, r in raw.items()}, raw


def rescore(fold, raw):
    """The same schedules scored with each emission factor (% of the historical footprint)."""
    return {f: {p: 100 * (emissions(fold, r["y"], F) - emissions(fold, fold.ref["y"], F)) / fold.footprint
                for p, r in raw.items()} for f, F in FACTORS.items()}


# ------------------------------------------------------------------------------------------------ policies
def select_menu(fold, world, pool, Mp, target):
    """Greedy forward selection of <= K_MENU schedules on the fold's training days (Lagrangian hindsight costs)."""
    Ep = costs(world, Mp, pool)["E"][fold.train]
    Pp = np.broadcast_to([s["price"] for s in pool], Ep.shape)
    J = Ep + tune_lambda(lambda l: (Ep + l * Pp).argmin(1), Pp, target) * Pp
    menu_idx, gains = [NORMAL], []
    while len(menu_idx) < K_MENU:
        cur = J[:, menu_idx].min(1).sum()
        v, x = min((J[:, menu_idx + [x]].min(1).sum(), x) for x in range(len(pool)) if x not in menu_idx)
        if gains and cur - v < 0.02 * gains[0]:
            break                                          # stop when a schedule adds < 2 % of the first gain
        menu_idx.append(x)
        gains.append(cur - v)
    return menu_idx


def paced_training_set(E, P, days):
    """Budget-aware tree training rows: each of `days` repeated at every SLACK_GRID level, with the level as an extra
    feature and costs E + lambda * P, lambda being the multiplier that makes the per-day argmin spend that level on
    average over `days`. Returns (X, C, lambdas)."""
    pn = float(P[days][:, NORMAL].mean())
    lams = np.array([tune_lambda(lambda l: (E[days] + l * P[days]).argmin(1), P[days], pn + sl) for sl in SLACK_GRID])
    Xa = np.vstack([np.column_stack([X[days], np.full(len(days), sl)]) for sl in SLACK_GRID])
    Ca = np.vstack([E[days] + lam * P[days] for lam in lams])
    return Xa, Ca, lams


def run_policies(fold, world, menu, M, target, depth=None, paced_depth=None, constraint="time", verbose=True):
    """Train every menu policy under `world` (its own costs) and return their test-window action sequences.
    constraint 'time' = time-average price (the proposal); 'paid' = consumption-weighted price.
    The Lagrange multiplier lambda is a hyperparameter: for the tree it is chosen jointly with the depth by fitting on
    the training days and scoring validation emissions under the feasibility rule; the final tree is refitted on
    training + validation with the chosen (depth, lambda). The direct optimisers choose lambda the same way."""
    K = costs(world, M, menu)
    E = K["E"]
    if constraint == "time":
        P_fit = P_dec = np.broadcast_to([s["price"] for s in menu], E.shape)
    else:
        P_fit, P_dec = K["paid"], K["paid_f"]                     # realised (fitting) / forecast (decision time)
    tr, va, tv, te = fold.train, fold.val, fold.trainval, fold.test
    grid = []
    for dpt in ((1, 2, 3) if depth is None else (depth,)):
        for lam in LAMBDAS:
            tree = PolicyTree(dpt, feature_names=list(FEATURES)).fit(X[tr], E[tr] + lam * P_fit[tr])
            acts, adj = feasibility_rule(tree.rankings(X[va]), P_dec[va], target)
            grid.append({"depth": dpt, "lambda": lam, "val_emissions_kg": float(E[va, acts].sum()), "adjustments": adj,
                         "leaves": tree.n_leaves()})
    by_depth = {d: min((g for g in grid if g["depth"] == d), key=lambda g: g["val_emissions_kg"]) for d in sorted({g["depth"] for g in grid})}
    if verbose:
        for d, g in by_depth.items():
            log(f"  depth {d}: best lambda {g['lambda']}, val emissions {g['val_emissions_kg']:.3f} kg/hh, "
                f"adj {g['adjustments']}, leaves {g['leaves']}")
    pick = pick_near_best(grid, lambda g: (g["depth"], g["adjustments"], -g["lambda"]))     # shallowest, fewest adj.
    tree = PolicyTree(pick["depth"], feature_names=list(FEATURES)).fit(X[tv], E[tv] + pick["lambda"] * P_fit[tv])
    ranks = tree.rankings(X[te])
    out = {"depth": pick["depth"], "lambda": pick["lambda"], "tree": tree, "by_depth": by_depth,
           "tree_prescribed": [r[0] for r in ranks]}
    out["tree_actions"], out["tree_adjustments"] = feasibility_rule(ranks, P_dec[te], target)
    for key, Ecost in (("direct", K["Ef"]), ("direct_perfect_ci", K["Ef_perfect_ci"])):
        g2 = []
        for lam in LAMBDAS:
            acts, adj = feasibility_rule(np.argsort(Ecost[va] + lam * P_dec[va], axis=1, kind="stable").tolist(), P_dec[va], target)
            g2.append({"lambda": lam, "val_emissions_kg": float(E[va, acts].sum()), "adjustments": adj})
        lam_d = pick_near_best(g2, lambda g: (g["adjustments"], -g["lambda"]))["lambda"]
        ranks = np.argsort(Ecost[te] + lam_d * P_dec[te], axis=1, kind="stable").tolist()
        out[f"{key}_actions"], out[f"{key}_adjustments"] = feasibility_rule(ranks, P_dec[te], target)
    lam_o = tune_lambda(lambda l: (E[te] + l * P_fit[te]).argmin(1), P_fit[te], target)   # hindsight, exact constraint
    out["oracle_actions"] = (E[te] + lam_o * P_fit[te]).argmin(1)

    # ---- budget-aware policies: the remaining budget per remaining day is an input
    def fit_paced(days, dpt):
        Xa, Ca, lams = paced_training_set(E, P_fit, days)
        return PolicyTree(dpt, feature_names=list(PACED_FEATURES)).fit(Xa, Ca), lams

    def paced_rollout(rank_fn, days):
        first = []
        def ranked(k, sl):
            r = rank_fn(k, sl)
            first.append(r[0])
            return r
        acts, adj = feasibility_rule(ranked, P_dec[days], target)
        return acts, adj, first

    tree_rank = lambda tree, days: (lambda k, sl: tree.leaf_of(np.append(X[days[k]], sl))["ranking"])
    grid3 = []
    for dpt in ((1, 2, 3) if paced_depth is None else (paced_depth,)):
        tree, _ = fit_paced(tr, dpt)
        acts, adj, _ = paced_rollout(tree_rank(tree, va), va)
        grid3.append({"depth": dpt, "val_emissions_kg": float(E[va, acts].sum()), "adjustments": adj, "leaves": tree.n_leaves()})
    pick3 = pick_near_best(grid3, lambda g: (g["depth"], g["adjustments"]))
    tree, lams = fit_paced(tv, pick3["depth"])
    out["tree_paced_actions"], out["tree_paced_adjustments"], out["tree_paced_prescribed"] = paced_rollout(tree_rank(tree, te), te)
    out.update(paced_depth=pick3["depth"], paced_tree=tree, paced_by_depth={g["depth"]: g for g in grid3}, slack_lambdas=lams)
    direct_rank = lambda k, sl: np.argsort(K["Ef"][te[k]] + np.interp(sl, SLACK_GRID, lams) * P_dec[te[k]], kind="stable").tolist()
    out["direct_paced_actions"], out["direct_paced_adjustments"], _ = paced_rollout(direct_rank, te)
    if verbose:
        log(f"  budget-aware tree: depth {pick3['depth']} (val emissions by depth "
            + ", ".join(f"{g['depth']}: {g['val_emissions_kg']:.3f}" for g in grid3) + ")")
    return out


def policy_schedules(fold, menu, pol, target, constraint="time"):
    """Test-window (low, high) schedules of every policy; the carbon rule exists only for the time-average constraint."""
    te = fold.test
    from_menu = lambda acts: (np.stack([menu[a]["low"] for a in acts]), np.stack([menu[a]["high"] for a in acts]))
    scheds = {POLICIES[0]: (LOW_H[te], HIGH_H[te])}
    if constraint == "time":
        kh, kl = carbon_rule_counts(target)
        s = [carbon_rule_schedule(FAC_F[d], kh, kl) for d in te]
        scheds[POLICIES[1]] = (np.stack([x[0] for x in s]), np.stack([x[1] for x in s]))
    for p, key in zip(POLICIES[2:], ("tree", "direct", "direct_perfect_ci", "oracle", "tree_paced", "direct_paced")):
        scheds[p] = from_menu(pol[f"{key}_actions"])
    return scheds


def bootstrap_ci(fold, scheds, n=200):
    """95% interval of each policy's emissions change over the response-model bootstrap draws (policies fixed)."""
    model = ResponseModel("main")
    R_hist = tariff_regressors(LOW_H, HIGH_H, timeline=True)
    R_pol = {p: tariff_regressors(*s) for p, s in scheds.items() if p != POLICIES[0]}
    draws = {p: [] for p in R_pol}
    for b in range(n):
        th = [THETA[f"boot_main_janoct_c{c}"][b] for c in range(4)]
        m_hist = np.stack([model.combine(th[c], R_hist)[fold.test] for c in range(4)])
        for p, R in R_pol.items():
            m = np.stack([model.combine(th[c], R) for c in range(4)])
            y = test_consumption(fold, np.exp(np.clip(m - m_hist, -CLIP, CLIP)))
            draws[p].append(100 * (emissions(fold, y) - fold.ref["emissions_kg"]) / fold.footprint)
    return {p: list(np.percentile(v, [2.5, 97.5])) for p, v in draws.items()}


def rolling(pool):
    """The full protocol (menu, depth, lambda, policies) on each rolling fold, with the response model refitted on the
    days before its test window. Returns per-fold results at the fold's own target and at the test-window historical price."""
    out = {}
    for mm in ROLLING_TEST_MONTHS:
        fold, world = Fold(mm), response_world(fit=f"before{mm:02d}")
        Mp = world.m_menu(pool)
        target, matched = float(PRICE_H[fold.trainval].mean()), float(PRICE_H[fold.test].mean())
        menu_idx = select_menu(fold, world, pool, Mp, target)
        menu, M = [pool[i] for i in menu_idx], Mp[:, :, menu_idx]
        names = [s["name"] for s in menu]
        r = {"label": fold.label, "days": {"train": len(fold.train), "val": len(fold.val), "test": len(fold.test)},
             "target": target, "historical_test_price": matched, "menu": names}
        for tid, T in (("main", target), ("matched", matched)):
            pol = run_policies(fold, world, menu, M, T, depth=r.get("depth"), paced_depth=r.get("paced_depth"), verbose=False)
            table, raw = evaluate_all(fold, world, policy_schedules(fold, menu, pol, T))
            if tid == "main":
                r.update(depth=pol["depth"], lambda_tree=pol["lambda"], tree=pol["tree"].root,
                         tree_prescribed_counts=dict(Counter(names[a] for a in pol["tree_prescribed"])),
                         tree_event_days=int((pol["tree_actions"] != NORMAL).sum()), tree_adjustments=pol["tree_adjustments"],
                         paced_depth=pol["paced_depth"], paced_tree=pol["paced_tree"].root,
                         paced_event_days=int((pol["tree_paced_actions"] != NORMAL).sum()),
                         paced_adjustments=pol["tree_paced_adjustments"], scored_with_each_factor=rescore(fold, raw))
            else:                                          # the same counts at the price-matched target
                r.update(matched_tree_event_days=int((pol["tree_actions"] != NORMAL).sum()),
                         matched_tree_adjustments=pol["tree_adjustments"],
                         matched_paced_event_days=int((pol["tree_paced_actions"] != NORMAL).sum()),
                         matched_paced_adjustments=pol["tree_paced_adjustments"])
            r[tid] ={p: {k: v[k] for k in ("emissions_change_pct", "bill_change_mean_pct", "mean_price", "high_slots", "low_slots")}
                      for p, v in table.items()}
        out[str(mm)] = r
        log(f"rolling {fold.label}: depth {r['depth']}, " + ", ".join(
            f"{p.split(' (')[0]} {v['emissions_change_pct']:+.2f}%" for p, v in r["main"].items()))
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    fold = Fold(MAIN_TEST_MONTH)
    assert (split_of(DATES_2013[fold.test]) == "test").all() and (split_of(DATES_2013[fold.train]) == "train").all()
    P_hist = {k: float(PRICE_H[getattr(fold, k)].mean()) for k in ("train", "val", "test", "trainval")}
    paid_hist = {k: float(price_paid(PRICE_H[getattr(fold, k)], YALL[getattr(fold, k)])) for k in ("train", "val", "test", "trainval")}
    TARGET = P_hist["trainval"]
    log(f"metric: {METRIC}; historical TOU time-average price {P_hist}; price paid {paid_hist}; target = {TARGET:.3f} p/kWh")
    main_world = response_world()

    # ---------------- menu selection on the training days
    pool = candidate_pool()
    Mp = main_world.m_menu(pool)
    menu_idx = select_menu(fold, main_world, pool, Mp, TARGET)
    menu = [pool[i] for i in menu_idx]
    names = [s["name"] for s in menu]
    log("menu: " + " | ".join(f"{s['name']} ({s['price']:.2f})" for s in menu))
    M = Mp[:, :, menu_idx]

    # ---------------- main run
    pol = run_policies(fold, main_world, menu, M, TARGET)
    log(f"chosen depth {pol['depth']}; tree:\n" + pol["tree"].text(names))
    scheds = policy_schedules(fold, menu, pol, TARGET)
    table, raw = evaluate_all(fold, main_world, scheds)
    for k, v in table.items():
        log(f"{k:42s} E {v['emissions_change_pct']:+.2f}%  P {v['mean_price']:.3f}  paid {v['price_paid']:.2f}  "
            f"bill {v['bill_change_mean_pct']:+.2f}%  kWh {v['kwh_change_pct']:+.2f}%  peak {v['daily_peak_change_pct']:+.2f}%")
    log(f"adjustments: tree {pol['tree_adjustments']}, direct {pol['direct_adjustments']}, budget-aware tree "
        f"{pol['tree_paced_adjustments']}; budget-aware tree:\n" + pol["paced_tree"].text(names))
    rescored = rescore(fold, raw)
    boot = bootstrap_ci(fold, scheds)
    log("bootstrap 95% intervals: " + ", ".join(f"{p.split(' (')[0]} [{a:+.2f}, {b:+.2f}]%" for p, (a, b) in boot.items()))

    # ---------------- sensitivity 1: price targets and a cap on the average price paid (main world, depth fixed)
    variants = {"matched": ("time", P_hist["test"], "historical TOU, test period"),
                "full_year": ("time", 13.517, "full-year 2013 TOU in Tariffs.xlsx; proposal"),
                "main": ("time", TARGET, "Jan-Oct historical TOU; main"),
                "flat": ("time", FLAT_PRICE, "flat Std tariff"),
                "paid": ("paid", paid_hist["trainval"], "Jan-Oct historical level")}
    sens_constraints = {}
    for vid, (cons, T, note) in variants.items():
        p2 = pol if vid == "main" else run_policies(fold, main_world, menu, M, T, depth=pol["depth"], paced_depth=pol["paced_depth"],
                                                    constraint=cons, verbose=False)
        t2, _ = evaluate_all(fold, main_world, policy_schedules(fold, menu, p2, T, cons))
        label = f"{'Time-average price' if cons == 'time' else 'Average price paid'} <= {T:.3f} ({note})"
        sens_constraints[vid] = {"label": label, "results": {p: {k: v[k] for k in ("emissions_change_pct", "bill_change_mean_pct")}
                                                            for p, v in t2.items()}}
        log(f"{label}: " + ", ".join(f"{p.split(' (')[0]} {v['emissions_change_pct']:+.2f}% bill {v['bill_change_mean_pct']:+.1f}%"
                                     for p, v in t2.items()))

    # ---------------- sensitivity 2: demand-response worlds
    worlds = {"x0.5": ("Response x0.5", response_world(scale=0.5)), "x1.5": ("Response x1.5", response_world(scale=1.5)),
              "inblock": ("In-block effects only", response_world("inblock")),
              "full": ("Full model (+ post-block and day-level terms)", response_world("full")),
              "cvae": ("CVAE ensemble (proposal's simulator)", World(CVAEEnsemble()))}
    lo_night, _ = make_schedule(None, NIGHT_LOW)
    zero = np.zeros((len(fold.test), H))

    def night_low_effect(wd):
        """% change in test-period kWh if every test day had the night Low block instead of an all-Normal day."""
        return 100 * (evaluate(fold, wd, np.tile(lo_night, (len(fold.test), 1)), zero)["kwh"]
                      / evaluate(fold, wd, zero, zero)["kwh"] - 1)

    night_low = {"label": describe(None, NIGHT_LOW), "main": night_low_effect(main_world)}
    sens_worlds = {}
    for wid, (label, wd) in worlds.items():
        night_low[wid] = night_low_effect(wd)
        t_fixed, _ = evaluate_all(fold, wd, scheds)        # (a) main policies, evaluated in this world
        p3 = run_policies(fold, wd, menu, wd.m_menu(menu), TARGET, depth=pol["depth"], paced_depth=pol["paced_depth"], verbose=False)
        t3, _ = evaluate_all(fold, wd, policy_schedules(fold, menu, p3, TARGET))   # (b) policies re-trained in this world
        sens_worlds[wid] = {"label": label,
                            "main_policies_evaluated_here": {p: v["emissions_change_pct"] for p, v in t_fixed.items()},
                            "retrained_here": {p: v["emissions_change_pct"] for p, v in t3.items()}}
        log(f"world {label}: fixed " + ", ".join(f"{p.split(' (')[0]} {v:+.2f}%" for p, v in sens_worlds[wid]["main_policies_evaluated_here"].items())
            + " | retrained " + ", ".join(f"{p.split(' (')[0]} {v:+.2f}%" for p, v in sens_worlds[wid]["retrained_here"].items()))

    # ---------------- rolling evaluation over four test windows
    roll = rolling(pool)

    # ---------------- save
    count = lambda acts: dict(Counter(names[a] for a in acts))
    save_json({"metric": METRIC, "reference_footprint_kg": fold.footprint, "scored_with_each_factor": rescored,
               "target_price": TARGET, "historical_mean_price": P_hist, "historical_price_paid": paid_hist,
               "split_days": {k: len(getattr(fold, k)) for k in ("train", "val", "test")}, "n_test_days": len(fold.test),
               "menu": [{"name": s["name"], "price": s["price"]} for s in menu], "menu_pool_size": len(pool),
               "features": list(FEATURES), "feature_formats": list(FEATURES.values()),
               "depth": pol["depth"], "lambda_tree": pol["lambda"], "selection_by_depth": pol["by_depth"],
               "tree": pol["tree"].root, "tree_prescribed_test_counts": count(pol["tree_prescribed"]),
               "tree_test_counts": count(pol["tree_actions"]),
               "tree_event_days": int((pol["tree_actions"] != NORMAL).sum()),
               "adjustments": {k: pol[f"{k}_adjustments"] for k in ("tree", "direct", "direct_perfect_ci", "tree_paced", "direct_paced")},
               "paced_depth": pol["paced_depth"], "paced_tree": pol["paced_tree"].root, "paced_selection_by_depth": pol["paced_by_depth"],
               "paced_features": list(PACED_FEATURES), "paced_feature_formats": list(PACED_FEATURES.values()),
               "paced_prescribed_test_counts": count(pol["tree_paced_prescribed"]),
               "paced_test_counts": count(pol["tree_paced_actions"]),
               "paced_event_days": int((pol["tree_paced_actions"] != NORMAL).sum()),
               "slack_grid": SLACK_GRID, "slack_lambdas": pol["slack_lambdas"],
               "results": table, "bootstrap_ci_emissions_change_pct": boot,
               "sensitivity_constraints": sens_constraints, "sensitivity_worlds": sens_worlds,
               "night_low_day_kwh_change_pct": night_low, "rolling": roll}, OUT / f"step3_results{SUFFIX}.json")
    shown = POLICIES[0], POLICIES[2], POLICIES[6]                  # the policies drawn in figure 4
    np.savez_compressed(OUT / f"step3_test_schedules{SUFFIX}.npz", dates=DATES_2013[fold.test].values.astype("datetime64[D]"),
                        factor=FAC[fold.test], factor_forecast=FAC_F[fold.test], ci=CI[fold.test],
                        **{f"{p}|low": scheds[p][0] for p in shown}, **{f"{p}|high": scheds[p][1] for p in shown},
                        **{f"{p}|profile": raw[p]["profile"] for p in shown})
    log("done")


if __name__ == "__main__":
    main()
