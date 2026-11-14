"""Carbon reduction vs household affordability: budget-aware policy trees with an expected-bill constraint.
Additive: no existing output or implementation is modified.

Primary candidate: the budget-aware IAI Optimal Policy Tree (remaining announced-price budget as a state feature;
scripts/10_pacing_experiment.py) with the original feasibility rule as safety net. Here a second, BILL budget is added.

Bill metric (decision time). For day d and menu action a, the predicted consumption-weighted price paid is
    pi_hat[d, a] = sum_t p_t(a) y_hat_dt(a) / sum_t y_hat_dt(a)          (p/kWh; = 11.76 for the all-Normal day)
with y_hat the decision-time demand forecast of Step 3 (`costs()["paid_f"]`: de-tariffed mean of days d-15..d-2 of the
same day type x the simulator's tariff response) - no realised consumption of day d or later is used. A household's bill
is price paid x kWh, and simulated kWh changes are < 1 %, so the bill change ~= the change in price paid.
Bill constraint (deployable): mean over the evaluation days of pi_hat[d, a_d] <= (1 + x) * pi_ref, x in {0, 5, 10} %,
where pi_ref = mean daily price paid under the historical TOU over the TRAINING + VALIDATION days (observed consumption;
known before the test window). It is enforced online by lib/policy.multi_budget_rule together with the original
announced-price budget (time-average price <= T): the all-Normal day satisfies both, so the rule never fails and the
PREDICTED constraint holds by construction. Whether the SIMULATED bill change vs historical TOU in the test window stays
within x is an empirical question, reported separately (two gaps: forecast vs simulated demand, and pi_ref vs the test
window's own historical price paid).

Policies per window and target x:
  * budget-aware tree, rule only  (mu = 0): the existing tree, bill budget enforced by the rule (post-processing);
  * budget-aware tree, penalty + rule: training costs E + mu * pi (realised past days) + lambda(budget level) * price;
    mu and depth chosen on VALIDATION (minimum validation emissions under the capped rule, ties: shallower, fewer
    overrides, smaller mu);
  * unconstrained budget-aware tree (x = inf, mu = 0) = scripts/10 result (checked), and historical TOU;
  * hindsight frontier (not deployable): per-day argmin of E + mu * pi + lambda * price on the TEST days with realised
    costs, mu swept and lambda set to meet the price target exactly - the best attainable trade-off for this menu.
Retrospective sensitivity (not deployable): the same fitted policies with pi_ref replaced by the test window's own
historical TOU price paid, i.e. a cap that literally means "bills within x % of historical TOU".

Usage:  python -P scripts/11_bill_constraint.py [--solvers iai,exact] [--julia CMD]
Output: results/affordability/*.json|csv|md, results/figures/fig7_carbon_vs_bills.png
"""
from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--julia", default="julia +1.12.6/IAI/3.2.2")
_ap.add_argument("--solvers", default="iai,exact")
ARGS = _ap.parse_args()
HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(HERE.parent))
_spec = importlib.util.spec_from_file_location("pacing10", HERE / "10_pacing_experiment.py")
M10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M10)                             # FoldData, tree helpers; loads 07 (M10.M7) and Step 3 (M10.S3)
M7, S3 = M10.M7, M10.S3

from lib.common import (BLUE, GREEN, INK, INK2, MUTED, ORANGE, OUT, PRICE, PROJECT, VIOLET, load_json, make_log,  # noqa: E402
                        mpl_style, price_paid, save_json, schedule_price)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.policy import NORMAL, PolicyTree, feasibility_rule, multi_budget_rule, pick_near_best, tune_lambda  # noqa: E402

log, _ = make_log()
FOLDS = (5, 7, 9, 11)
XS = [0.0, 0.05, 0.10]                                    # bill-increase targets
MU_GRID = [0.0, 0.001, 0.002, 0.004, 0.007, 0.01, 0.015, 0.025, 0.05]   # kg CO2 per (p/kWh) of predicted price paid
DEPTHS = (1, 2, 3)
MIN_LEAF = 10
SOLVERS = [s for s in ARGS.solvers.split(",") if s]
RES = PROJECT / "results" / "affordability"
RES.mkdir(parents=True, exist_ok=True)
FIGD = PROJECT / "results" / "figures"
BILL_DIR = OUT / "iai_bill_average_mvp"
PACING = load_json(PROJECT / "results" / "pacing" / "pacing_results.json")
PACED_FEATS = list(S3.PACED_FEATURES)


# ------------------------------------------------------------------------------------------------ fold data
class Fold(M10.FoldData):
    def __init__(self, mm):
        super().__init__(mm)
        K = S3.costs(self.world, self.world.m_menu(self.menu), self.menu)
        assert np.allclose(K["E"], self.E)
        self.Pi_fit, self.Pi_dec = K["paid"], K["paid_f"]       # realised (past days: fitting) / decision-time forecast
        f = self.fold
        daily_paid = np.array([price_paid(S3.PRICE_H[d], S3.YALL[d]) for d in range(365)])
        self.pi_ref = float(daily_paid[f.trainval].mean())     # deployable reference (train + val, observed)
        self.pi_ref_test = float(daily_paid[f.test].mean())    # retrospective reference (test window)
        assert np.allclose(self.Pi_dec[:, NORMAL], PRICE["N"]) and np.allclose(self.Pi_fit[:, NORMAL], PRICE["N"])


def ba_set(fd, days, mu):
    """Budget-aware training set with bill penalty: E' = E + mu * pi (realised, past days); lambda per budget level."""
    return S3.paced_training_set(fd.E + mu * fd.Pi_fit, fd.P, days)


def export_iai(fds):
    BILL_DIR.mkdir(parents=True, exist_ok=True)
    cases = []
    for fd in fds:
        for mi, mu in enumerate(MU_GRID):
            tag = f"{fd.tag}m{mi}"
            for s, days in (("tr", fd.fold.train), ("tv", fd.fold.trainval)):
                Xa, Ca, _ = ba_set(fd, days, mu)
                cols = [f"f{j}" for j in range(Xa.shape[1])]
                pd.DataFrame(Xa, columns=cols).to_csv(BILL_DIR / f"{tag}_X_{s}.csv", index=False)
                pd.DataFrame(Ca, columns=[f"a{a}" for a in range(Ca.shape[1])]).to_csv(BILL_DIR / f"{tag}_E_{s}.csv", index=False)
                pd.DataFrame(Xa[:5], columns=cols).to_csv(BILL_DIR / f"{tag}_X_{'va' if s == 'tr' else 'te'}.csv", index=False)
                cases += [{"case": f"{tag}_{s}_d{d}_l00", "fold": tag, "fitset": s, "depth": d, "lambda": 0.0, "minbucket": MIN_LEAF}
                          for d in DEPTHS]
            pd.DataFrame({"price": np.zeros(fd.E.shape[1])}).to_csv(BILL_DIR / f"{tag}_price.csv", index=False)
    pd.DataFrame(cases).to_csv(BILL_DIR / "cases.csv", index=False)
    cmd = shlex.split(ARGS.julia) + [str(HERE / "07_iai_fit.jl"), str(BILL_DIR)]
    log(f"IAI bill-penalised budget-aware fits ({len(cases)}): {' '.join(cmd)}")
    with open(BILL_DIR / "julia.log", "w") as fh:
        if subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode:
            raise RuntimeError(f"IAI failed, see {BILL_DIR / 'julia.log'}")


_TREES = {}


def tree(fd, solver, mi, depth, s):
    """(root, one-row ranking function) of a budget-aware tree with bill penalty MU_GRID[mi], fitted on 'tr' or 'tv'."""
    key = (fd.mm, solver, mi, depth, s)
    if key not in _TREES:
        days = fd.fold.train if s == "tr" else fd.fold.trainval
        Xa, Ca, _ = ba_set(fd, days, MU_GRID[mi])
        if solver == "exact":
            t = PolicyTree(depth, min_leaf=MIN_LEAF, feature_names=PACED_FEATS).fit(Xa, Ca)
            _TREES[key] = (t.root, lambda x, t=t: t.leaf_of(x)["ranking"])
        else:
            old = M7.IAI_DIR
            M7.IAI_DIR = BILL_DIR
            try:
                root, rf = M7.iai_fitted(f"{fd.tag}m{mi}_{s}_d{depth}_l00", Xa)
            finally:
                M7.IAI_DIR = old
            _TREES[key] = (root, lambda x, rf=rf: rf(np.asarray(x, float)[None])[0])
    return _TREES[key]


def ranks(one, days):
    return lambda k, sl: one(np.append(S3.X[days[k]], sl))


def rollout(fd, one, days, T, cap):
    """Run the rule on `days`; cap None = original rule (announced-price budget only)."""
    budgets = [(fd.P[days], T)] + ([] if cap is None else [(fd.Pi_dec[days], cap)])
    first = []

    def rk(k, sl):
        r = ranks(one, days)(k, sl)
        first.append(int(r[0]))
        return r
    acts, adj, blocked = multi_budget_rule(rk, budgets)
    return acts, adj, blocked, np.array(first)


def select(fd, solver, cap, mus):
    """(mi, depth) on validation under the capped rule: min validation emissions; ties -> shallower, fewer overrides, smaller mu."""
    va = fd.fold.val
    grid = []
    for mi in mus:
        for d in DEPTHS:
            _, one = tree(fd, solver, mi, d, "tr")
            acts, adj, _, _ = rollout(fd, one, va, fd.target, cap)
            grid.append({"mi": mi, "mu": MU_GRID[mi], "depth": d, "val_emissions_kg": float(fd.E[va, acts].sum()), "adjustments": adj,
                         "val_pred_paid": float(fd.Pi_dec[va, acts].mean())})
    return pick_near_best(grid, lambda g: (g["depth"], g["adjustments"], g["mu"])), grid


def leaves(n):
    return 1 if n["leaf"] else leaves(n["left"]) + leaves(n["right"])


def uses_budget(n):
    return (not n["leaf"]) and (n["feature"] == len(PACED_FEATS) - 1 or uses_budget(n["left"]) or uses_budget(n["right"]))


def evaluate(fd, acts, adj, blocked, first, cap, cap_x, root=None):
    f = fd.fold
    te = f.test
    low = np.stack([fd.menu[a]["low"] for a in acts])
    high = np.stack([fd.menu[a]["high"] for a in acts])
    r = S3.evaluate(f, fd.world, low, high)
    s = S3.summarise(f, r)
    ok = f.ref["bills"] > 0                                  # households with a zero historical bill in the window are excluded
    ch = 100 * (r["bills"][ok] / f.ref["bills"][ok] - 1)     # from per-household % changes (their % change is undefined)
    rebate = float(r["bills"].mean() - f.ref["bills"].mean())  # revenue-neutral lump-sum rebate (GBP per household)
    ch_reb = 100 * ((r["bills"][ok] - rebate) / f.ref["bills"][ok] - 1)
    price = schedule_price(low, high)
    sim_paid = float(np.mean([price_paid(price[i], r["y"][i]) for i in range(len(te))]))
    pred_paid = float(fd.Pi_dec[te, acts].mean())
    names = [m["name"] for m in fd.menu]
    out = {"emissions_reduction_pct": -s["emissions_change_pct"], "emissions_change_kg_per_hh": s["emissions_change_kg_per_hh"],
           "mean_price": s["mean_price"], "bill_change_mean_pct": s["bill_change_mean_pct"],
           "bill_change_p10_pct": float(np.percentile(ch, 10)), "bill_change_p90_pct": float(np.percentile(ch, 90)),
           "households_zero_historical_bill": int((~ok).sum()),
           "lump_sum_rebate_gbp_per_hh": rebate, "after_rebate_bill_change_p10_pct": float(np.percentile(ch_reb, 10)),
           "after_rebate_bill_change_p90_pct": float(np.percentile(ch_reb, 90)),
           "after_rebate_share_households_worse_by_5pct": float(100 * np.mean(ch_reb > 5)),
           "bill_change_gbp_per_hh": float(r["bills"].mean() - f.ref["bills"].mean()),
           "historical_bill_gbp_per_hh": float(f.ref["bills"].mean()),
           "kwh_change_pct": s["kwh_change_pct"], "price_paid_consumption_weighted": s["price_paid"],
           "pred_daily_paid_mean": pred_paid, "sim_daily_paid_mean": sim_paid, "cap": cap,
           "pred_within_cap": None if cap is None else bool(pred_paid <= cap + 1e-9),
           "sim_within_cap": None if cap is None else bool(sim_paid <= cap + 1e-9),
           "bill_within_x_vs_historical": None if cap_x is None else bool(s["bill_change_mean_pct"] <= 100 * cap_x + 1e-9),
           "share_households_above_x_pct": None if cap_x is None else float(np.mean(ch > 100 * cap_x) * 100),
           "price_within_target": bool(s["mean_price"] <= fd.target + 1e-9) if cap is not None or True else None,
           "overrides": int(adj), "overrides_blocked_by_price": int(blocked[0]),
           "overrides_blocked_by_bill": int(blocked[1]) if len(blocked) > 1 else 0, "n_days": len(te),
           "high_days": int(high.any(1).sum()), "low_days": int(low.any(1).sum()),
           "realised_counts": dict(Counter(names[a] for a in acts)), "prescribed_counts": dict(Counter(names[a] for a in first)),
           "actions": acts.tolist()}
    if root is not None:
        out.update(leaves=leaves(root), uses_budget_feature=uses_budget(root), tree_text=M10.tree_text(root, names, PACED_FEATS))
    return out


def frontier(fd, n=60):
    """Hindsight (test-day realised costs) trade-off: argmin E + mu * pi + lambda * price, lambda meeting the price target."""
    te = fd.fold.test
    E, P, Pi = fd.E[te], fd.P[te], fd.Pi_fit[te]
    pts = []
    for mu in np.concatenate([[0.0], np.geomspace(1e-4, 0.2, n)]):
        lam = tune_lambda(lambda l: (E + mu * Pi + l * P).argmin(1), P, fd.target)
        acts = (E + mu * Pi + lam * P).argmin(1)
        low = np.stack([fd.menu[a]["low"] for a in acts])
        high = np.stack([fd.menu[a]["high"] for a in acts])
        s = S3.summarise(fd.fold, S3.evaluate(fd.fold, fd.world, low, high))
        pts.append({"mu": float(mu), "emissions_reduction_pct": -s["emissions_change_pct"], "bill_change_mean_pct": s["bill_change_mean_pct"],
                    "mean_price": s["mean_price"]})
    return pts


# ------------------------------------------------------------------------------------------------ main
def run_fold(fd):
    te = fd.fold.test
    res = {"label": fd.fold.label, "target": fd.target, "pi_ref_trainval": fd.pi_ref, "pi_ref_test_historical": fd.pi_ref_test,
           "historical_test_mean_price": fd.matched, "menu": fd.names, "policies": {}}
    log(f"=== {fd.fold.label}: price target {fd.target:.3f}; reference price paid train+val {fd.pi_ref:.3f}, "
        f"historical in test window {fd.pi_ref_test:.3f} p/kWh")
    for solver in SOLVERS:
        # unconstrained existing policy (mu = 0, original rule); must equal scripts/10's result
        pick, _ = select(fd, solver, None, [0])
        root, one = tree(fd, solver, 0, pick["depth"], "tv")
        acts, adj, bl, first = rollout(fd, one, te, fd.target, None)
        e = evaluate(fd, acts, adj, bl, first, None, None, root)
        want = PACING["folds"][str(fd.mm)]["policies"][f"{solver} budget-aware tree | original rule"]["main"]["emissions_reduction_pct"]
        assert abs(e["emissions_reduction_pct"] - want) < 1e-9, (solver, fd.mm, e["emissions_reduction_pct"], want)
        res["policies"][f"{solver}|unconstrained"] = {"x": None, "variant": "unconstrained", "solver": solver, "selection": pick, "main": e}
        log(f"  {solver:5s} unconstrained           red {e['emissions_reduction_pct']:+.3f}%  bill {e['bill_change_mean_pct']:+.1f}%  "
            f"paid(pred) {e['pred_daily_paid_mean']:.2f}  overrides {e['overrides']}  depth {pick['depth']} leaves {e['leaves']}")
        for x in XS:
            cap = (1 + x) * fd.pi_ref
            for variant, mus in (("rule only", [0]), ("penalty + rule", list(range(len(MU_GRID))))):
                pick, grid = select(fd, solver, cap, mus)
                root, one = tree(fd, solver, pick["mi"], pick["depth"], "tv")
                entry = {"x": x, "variant": variant, "solver": solver, "selection": pick, "val_grid": grid}
                acts, adj, bl, first = rollout(fd, one, te, fd.target, cap)
                entry["main"] = evaluate(fd, acts, adj, bl, first, cap, x, root)
                capm = (1 + x) * fd.pi_ref_test                     # retrospective: cap relative to the test window's TOU
                if capm >= PRICE["N"]:
                    acts, adj, bl, first = rollout(fd, one, te, fd.target, capm)
                    entry["matched"] = evaluate(fd, acts, adj, bl, first, capm, x, root)
                else:
                    entry["matched"] = {"infeasible": f"cap {capm:.3f} < all-Normal price 11.76: no menu sequence can satisfy it"}
                res["policies"][f"{solver}|{variant}|{int(100 * x)}%"] = entry
                e = entry["main"]
                log(f"  {solver:5s} x={int(100 * x):2d}% {variant:15s} red {e['emissions_reduction_pct']:+.3f}%  bill {e['bill_change_mean_pct']:+.1f}% "
                    f"(cap {cap:.2f}; pred {e['pred_daily_paid_mean']:.2f}, sim {e['sim_daily_paid_mean']:.2f})  overrides {e['overrides']} "
                    f"(price {e['overrides_blocked_by_price']}, bill {e['overrides_blocked_by_bill']})  mu {pick['mu']} depth {pick['depth']} "
                    f"leaves {e['leaves']} budget-split {e['uses_budget_feature']}")
    res["frontier"] = frontier(fd)
    return res


def tables_and_figure(out):
    rows = []
    for mm, r in out["folds"].items():
        rows.append({"window": r["label"], "policy": "historical TOU", "solver": "-", "x_pct": "-", "variant": "-", "reduction_pct": 0.0,
                     "bill_change_pct": 0.0, "bill_change_gbp": 0.0, "avg_price": r["historical_test_mean_price"],
                     "pred_paid": "-", "sim_paid": "-", "cap": "-", "pred_ok": "-", "sim_ok": "-", "bill_ok_vs_hist": "-",
                     "hh_above_x_pct": "-", "bill_p10_pct": "-", "bill_p90_pct": "-", "rebate_gbp": "-", "after_rebate_p10_pct": "-",
                     "after_rebate_p90_pct": "-", "overrides": "-", "by_bill": "-", "high_days": "-", "mu": "-", "depth": "-", "leaves": "-",
                     "budget_split": "-", "matched_reduction_pct": "-", "matched_bill_change_pct": "-", "matched_bill_ok": "-"})
        for k, p in r["policies"].items():
            e, m = p["main"], p.get("matched", {})
            rows.append({"window": r["label"], "policy": "budget-aware tree", "solver": p["solver"],
                         "x_pct": "none" if p["x"] is None else int(100 * p["x"]), "variant": p["variant"],
                         "reduction_pct": e["emissions_reduction_pct"], "bill_change_pct": e["bill_change_mean_pct"],
                         "bill_change_gbp": e["bill_change_gbp_per_hh"], "avg_price": e["mean_price"],
                         "pred_paid": e["pred_daily_paid_mean"], "sim_paid": e["sim_daily_paid_mean"], "cap": e["cap"] or "-",
                         "pred_ok": e["pred_within_cap"] if e["cap"] else "-", "sim_ok": e["sim_within_cap"] if e["cap"] else "-",
                         "bill_ok_vs_hist": e["bill_within_x_vs_historical"] if e["cap"] else "-",
                         "hh_above_x_pct": e["share_households_above_x_pct"] if e["cap"] else "-", "bill_p10_pct": e["bill_change_p10_pct"],
                         "bill_p90_pct": e["bill_change_p90_pct"], "rebate_gbp": e["lump_sum_rebate_gbp_per_hh"],
                         "after_rebate_p10_pct": e["after_rebate_bill_change_p10_pct"], "after_rebate_p90_pct": e["after_rebate_bill_change_p90_pct"],
                         "overrides": e["overrides"],
                         "by_bill": e["overrides_blocked_by_bill"], "high_days": e["high_days"], "mu": p["selection"]["mu"],
                         "depth": p["selection"]["depth"], "leaves": e["leaves"], "budget_split": e["uses_budget_feature"],
                         "matched_reduction_pct": m.get("emissions_reduction_pct", m.get("infeasible", "-")),
                         "matched_bill_change_pct": m.get("bill_change_mean_pct", "-"), "matched_bill_ok": m.get("bill_within_x_vs_historical", "-")})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "affordability_all.csv", index=False)
    pol = df[df.policy != "historical TOU"]
    summ = (pol.groupby(["solver", "x_pct", "variant"], sort=False)
            .agg(mean_reduction_pct=("reduction_pct", "mean"), min_reduction_pct=("reduction_pct", "min"),
                 mean_bill_change_pct=("bill_change_pct", "mean"), max_bill_change_pct=("bill_change_pct", "max"),
                 mean_bill_change_gbp=("bill_change_gbp", "mean"), mean_emissions_saved_kg=("reduction_pct", "size"),
                 windows_bill_within_x=("bill_ok_vs_hist", lambda v: f"{sum(x is True for x in v)}/{len(v)}" if v.iloc[0] != "-" else "-"),
                 windows_sim_within_cap=("sim_ok", lambda v: f"{sum(x is True for x in v)}/{len(v)}" if v.iloc[0] != "-" else "-"),
                 mean_overrides=("overrides", "mean"), mean_leaves=("leaves", "mean")).reset_index())
    # implied household transfer per tonne avoided, pooled over windows (GBP per tonne CO2)
    kg = {}
    for mm, r in out["folds"].items():
        for k, p in r["policies"].items():
            kg.setdefault((p["solver"], "none" if p["x"] is None else int(100 * p["x"]), p["variant"]), []).append(
                (-p["main"]["emissions_change_kg_per_hh"], p["main"]["bill_change_gbp_per_hh"]))
    summ["mean_emissions_saved_kg"] = [np.mean([a for a, _ in kg[(s, x, v)]]) for s, x, v in zip(summ.solver, summ.x_pct, summ.variant)]
    summ["gbp_per_tonne_avoided_pooled"] = [(sum(b for _, b in kg[(s, x, v)]) / sum(a for a, _ in kg[(s, x, v)]) * 1000)
                                            if sum(a for a, _ in kg[(s, x, v)]) > 0 else np.nan
                                            for s, x, v in zip(summ.solver, summ.x_pct, summ.variant)]
    summ.to_csv(RES / "affordability_summary.csv", index=False)

    plt = mpl_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    style = {("none", "unconstrained"): (INK, "o", "unconstrained budget-aware tree"),
             (10, "penalty + rule"): (BLUE, "s", "bill cap +10 %, penalty + rule"), (5, "penalty + rule"): (VIOLET, "s", "bill cap +5 %, penalty + rule"),
             (0, "penalty + rule"): (GREEN, "s", "bill cap 0 %, penalty + rule"),
             (10, "rule only"): (BLUE, "^", "bill cap +10 %, rule only"), (5, "rule only"): (VIOLET, "^", "bill cap +5 %, rule only"),
             (0, "rule only"): (GREEN, "^", "bill cap 0 %, rule only")}
    prim = SOLVERS[0]
    for ax, (mm, r) in zip(axes.ravel(), out["folds"].items()):
        fr = pd.DataFrame(r["frontier"]).sort_values("bill_change_mean_pct")
        best = fr.emissions_reduction_pct.cummax()
        ax.plot(fr.bill_change_mean_pct, best, color=MUTED, lw=1.5, label="hindsight Lagrangian sweep (approx. frontier; not deployable)")
        ax.scatter([0], [0], color=ORANGE, marker="D", s=40, zorder=4, label="historical TOU")
        for (x, var), (col, mk, lab) in style.items():
            key = f"{prim}|unconstrained" if x == "none" else f"{prim}|{var}|{x}%"
            if key in r["policies"]:
                e = r["policies"][key]["main"]
                ax.scatter(e["bill_change_mean_pct"], e["emissions_reduction_pct"], color=col, marker=mk, s=46, zorder=5,
                           label=lab, edgecolor="white", lw=0.5)
        for x, col in ((0, GREEN), (5, VIOLET), (10, BLUE)):
            bx = 100 * ((1 + x / 100) * r["pi_ref_trainval"] / r["pi_ref_test_historical"] - 1)
            ax.axvline(bx, color=col, ls=":", lw=1)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"{r['label']}: dotted = deployable caps 0/5/10 % (train+val reference)\nexpressed vs this window's historical TOU", fontsize=8.5)
        ax.set_xlabel("Mean household bill change vs historical TOU (%)")
        ax.set_ylabel("Emissions reduction vs TOU (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"Carbon reduction vs household bills: budget-aware {'IAI OPT' if prim == 'iai' else 'exact'} tree, four rolling test windows "
                 "(model-based estimates)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(FIGD / "fig7_carbon_vs_bills.png", bbox_inches="tight")
    plt.close(fig)
    md = lambda d: "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join(
        "| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |" for row in d.itertuples(index=False)) + "\n"
    txt = ["# Carbon vs affordability tables (generated by scripts/11_bill_constraint.py)", "",
           "Model-based estimates (simulated demand x realised average CI; simulated bills). bill_change = mean household bill vs "
           "historical TOU in the same test window. Deployable cap = (1 + x) x reference price paid (historical TOU over train+val). "
           "pred_paid / sim_paid = mean daily price paid (p/kWh) with decision-time / simulated demand. matched_* = retrospective "
           "cap relative to the test window's own historical TOU (not deployable).", "",
           "## Summary across the four windows", "", md(summ.round(3)), ""]
    for win in dict.fromkeys(df.window):
        txt += [f"## {win}", "", md(df[df.window == win].drop(columns=["window"]).round(3)), ""]
    (RES / "affordability_tables.md").write_text("\n".join(txt), encoding="utf-8")
    return df, summ


def main():
    fds = [Fold(mm) for mm in FOLDS]
    if "iai" in SOLVERS:
        export_iai(fds)
    out = {"bill_targets": XS, "mu_grid": MU_GRID, "solvers": SOLVERS, "folds": {}}
    for fd in fds:
        out["folds"][str(fd.mm)] = run_fold(fd)
    for r in out["folds"].values():                         # keep the json light: drop validation grids' long fields
        for p in r["policies"].values():
            p.pop("val_grid", None)
    save_json(out, RES / "affordability_results.json")
    df, summ = tables_and_figure(out)
    log("summary:\n" + summ.round(3).to_string())
    log("done")


if __name__ == "__main__":
    main()
