"""Pacing experiment: original (greedy) average-price feasibility rule vs a budget-PACING rule, and post-hoc rule vs
remaining budget as a policy STATE variable. Additive: no existing output is modified.

Rules (lib/policy.py):
  original  feasibility_rule: day k takes the tree's first-ranked action unless the period target would become
            unattainable even with all-Normal days afterwards -> spends the budget as early as possible.
  pacing    pacing_rule(borrow=B): additionally the cumulative announced price may run at most B (p/kWh x days) ahead of
            an even spend of the target, S_k + p_a <= (k+1) T + B. B is a hyper-parameter chosen on VALIDATION from
            BORROW_GRID (B = 0: save first, then spend). borrow = inf reproduces the original rule exactly (unit-tested).
Both rules use only information known at each announcement (past announced prices, day index, period length, target,
menu prices); the tree's rankings use only decision-time features (03_policy.features).

Policies (each fold = rolling test window with its own train/validation/target, menu and simulator, as in Step 3):
  * Exact tree / IAI OPT, "trained under original rule": the trees selected in 07_iai_opt.py (depth, lambda chosen on
    validation under the original rule), evaluated with the original rule and with the pacing rule (B chosen on
    validation for that fixed tree). Isolates the effect of the rule.
  * Exact tree / IAI OPT, "re-selected under pacing": depth (1-3), lambda and B chosen jointly on validation with the
    pacing rule (same fitted trees as 07_iai_opt.py; final refit on train+val).
  * Constant policy (depth-0 tree: one fixed ranking for every day), under each rule: a benchmark for whether the tree's
    splits matter at all.
  * Budget-aware trees (exact and IAI): remaining budget per remaining day (above an all-Normal day) is a FEATURE (state
    variable); training rows = each day x 9 budget levels with Lagrangian costs at the matching lambda
    (03_policy.paced_training_set). Depth chosen on validation under each rule (and B under pacing).
  * Direct optimisation (forecast costs) and budget-aware direct optimisation, under each rule; historical TOU and the
    perfect-hindsight oracle as references.
Evaluation: test window once, simulated demand x realised average CI (primary metric), bills, overrides (days the
rule replaced the first-ranked action), timing of High events, and whether High events land on forecast-dirtier evenings.
Price-matched sensitivity (retrospective, not deployable): the same fitted policies and hyper-parameters, with the
rule's target set to the historical TOU price of the test window.
Paired bootstrap (Nov-Dec only; 200 demand-model draws, policies fixed): difference pacing - original.

Usage:  python -P scripts/10_pacing_experiment.py [--julia "julia +1.12.6/IAI/3.2.2"] [--skip-iai]
Needs:  results of 07_iai_opt.py (OUT/iai_opt_results_average_mvp.json and its IAI fits in OUT/iai_opt_average_mvp/)
Output: results/pacing/pacing_results.json, results/pacing/*.md|csv, results/figures/fig5_*.png, fig6_*.png
"""
from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--julia", default="julia +1.12.6/IAI/3.2.2")
_ap.add_argument("--skip-iai", action="store_true")
ARGS = _ap.parse_args()
HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0], "--metric", "average", "--temperature", "lagged", "--tag", "mvp"]
sys.path.insert(0, str(HERE.parent))
_spec = importlib.util.spec_from_file_location("iai07", HERE / "07_iai_opt.py")
M7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M7)                              # loads Step 3 (M7.S3) with the MVP settings
S3 = M7.S3

from lib.common import (BLUE, DATES_2013, GREEN, H, INK, INK2, MUTED, ORANGE, OUT, PROJECT, SHADE, VIOLET,  # noqa: E402
                        hours, load_json, make_log, mpl_style, save_json)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.demand import ResponseModel, tariff_regressors  # noqa: E402
from lib.policy import NORMAL, PolicyTree, feasibility_rule, pacing_rule, pick_near_best, tune_lambda  # noqa: E402

log, _ = make_log()
FOLDS = (5, 7, 9, 11)
BORROW_GRID = [0.0, 5.0, 10.0, 20.0, 40.0]               # p/kWh x days ahead of an even spend (21.0 - 13.6 = 7.4 = one 4-h High day)
DEPTHS = (1, 2, 3)
MIN_LEAF = 10
FEATS = list(S3.FEATURES)
PACED_FEATS = list(S3.PACED_FEATURES)
RES = PROJECT / "results" / "pacing"
RES.mkdir(parents=True, exist_ok=True)
FIGD = PROJECT / "results" / "figures"
PACED_DIR = OUT / "iai_budgetaware_average_mvp"
IAI07 = load_json(OUT / "iai_opt_results_average_mvp.json")
KEEP = ("emissions_change_pct", "mean_price", "price_paid", "kwh_change_pct", "daily_peak_change_pct",
        "bill_change_mean_pct", "bill_change_p10_pct", "bill_change_median_pct", "bill_change_p90_pct")


def tree_text(node, names, feats, indent=""):
    if node["leaf"]:
        return f"{indent}-> {names[node['action']]}  (n={node['n']})\n"
    op = "<" if node.get("strict") else "<="
    return (f"{indent}if {feats[node['feature']]} {op} {node['threshold']:.4g}:\n" + tree_text(node["left"], names, feats, indent + "    ")
            + f"{indent}else:\n" + tree_text(node["right"], names, feats, indent + "    "))


def run_rule(rule, ranks, P, target, borrow=None):
    """Apply a rule; also return the first-ranked (prescribed) action of each day."""
    first = []
    if callable(ranks):
        def wrapped(k, sl):
            r = ranks(k, sl)
            first.append(int(r[0]))
            return r
        rk = wrapped
    else:
        rk = ranks
        first = [int(r[0]) for r in ranks]
    acts, adj = feasibility_rule(rk, P, target) if rule == "original" else pacing_rule(rk, P, target, borrow)
    return acts, adj, first


# ------------------------------------------------------------------------------------------------ fold set-up
class FoldData:
    def __init__(self, mm):
        self.mm = mm
        self.fold = S3.Fold(mm)
        f = self.fold
        self.world = S3.response_world() if mm == S3.MAIN_TEST_MONTH else S3.response_world(fit=f"before{mm:02d}")
        r7 = IAI07["folds"][str(mm)]
        pool = {s["name"]: s for s in M7.candidate_pool()}
        self.menu = [pool[n] for n in r7["menu"]]             # the menu chosen on training days in 07 / Step 3
        M = self.world.m_menu(self.menu)
        K = S3.costs(self.world, M, self.menu)
        self.E, self.Ef = K["E"], K["Ef"]
        self.P = np.broadcast_to([s["price"] for s in self.menu], self.E.shape)
        self.target = float(S3.PRICE_H[f.trainval].mean())
        assert abs(self.target - r7["target"]) < 1e-12
        self.matched = float(S3.PRICE_H[f.test].mean())
        self.tag = f.label.replace("-", "")
        self.r7 = r7
        self.names = [s["name"] for s in self.menu]

    def val_emissions(self, acts):
        return float(self.E[self.fold.val, acts].sum())


def rollout_grid(fd, ranks_val, rule):
    """Validation rollouts: original -> one entry; pacing -> one entry per B."""
    va = fd.fold.val
    Bs = [None] if rule == "original" else BORROW_GRID
    out = []
    for B in Bs:
        acts, adj, _ = run_rule(rule, ranks_val, fd.P[va], fd.target, B)
        out.append({"borrow": B, "val_emissions_kg": fd.val_emissions(acts), "adjustments": adj})
    return out


# ------------------------------------------------------------------------------------------------ policy builders
def static_tree(fd, solver, depth, lam, days_fit):
    """(root, rank_fn) of an exact or IAI tree fitted on days_fit ('tr' or 'tv') at (depth, lambda)."""
    f = fd.fold
    days = f.train if days_fit == "tr" else f.trainval
    if solver == "exact":
        t = PolicyTree(depth, min_leaf=MIN_LEAF, feature_names=FEATS).fit(S3.X[days], fd.E[days] + lam * fd.P[days])
        return t.root, t.rankings
    li = S3.LAMBDAS.index(lam)
    return M7.iai_fitted(f"{fd.tag}_{days_fit}_d{depth}_l{li:02d}", S3.X[days])


def select_static(fd, solver, rule, depths=DEPTHS):
    """Choose (depth, lambda[, B]) on validation under `rule` (trees fitted on training days), refit on train+val."""
    grid = []
    for d in depths:
        for lam in S3.LAMBDAS:
            if d == 0:
                root, rank_fn = static_tree(fd, "exact", 0, lam, "tr")
            else:
                root, rank_fn = static_tree(fd, solver, d, lam, "tr")
            for g in rollout_grid(fd, rank_fn(S3.X[fd.fold.val]), rule):
                grid.append({"depth": d, "lambda": lam, **g})
    pick = pick_near_best(grid, lambda g: (g["depth"], g["adjustments"], -g["lambda"], g["borrow"] or 0.0))
    root, rank_fn = static_tree(fd, "exact" if pick["depth"] == 0 else solver, pick["depth"], pick["lambda"], "tv")
    return pick, root, rank_fn


def budget_aware_tree(fd, solver, depth, days_fit, Xa, Ca):
    if solver == "exact":
        t = PolicyTree(depth, min_leaf=MIN_LEAF, feature_names=PACED_FEATS).fit(Xa, Ca)
        return t.root, (lambda x, t=t: t.leaf_of(x)["ranking"])
    old = M7.IAI_DIR
    M7.IAI_DIR = PACED_DIR
    try:
        root, rank_fn = M7.iai_fitted(f"{fd.tag}_{days_fit}_d{depth}_l00", Xa)
    finally:
        M7.IAI_DIR = old
    return root, (lambda x: rank_fn(np.asarray(x, float)[None])[0])


def paced_rank(fd, rank_one, days):
    return lambda k, sl: rank_one(np.append(S3.X[days[k]], sl))


def export_budget_aware(fds):
    """Write the budget-aware training sets (rows = day x budget level, costs already Lagrangian) for IAI."""
    PACED_DIR.mkdir(parents=True, exist_ok=True)
    cases = []
    for fd in fds:
        for s, days in (("tr", fd.fold.train), ("tv", fd.fold.trainval)):
            Xa, Ca, _ = S3.paced_training_set(fd.E, fd.P, days)
            pd.DataFrame(Xa, columns=[f"f{j}" for j in range(Xa.shape[1])]).to_csv(PACED_DIR / f"{fd.tag}_X_{s}.csv", index=False)
            pd.DataFrame(Ca, columns=[f"a{a}" for a in range(Ca.shape[1])]).to_csv(PACED_DIR / f"{fd.tag}_E_{s}.csv", index=False)
            dummy = "va" if s == "tr" else "te"                  # 07_iai_fit.jl predicts on these; we route ourselves
            pd.DataFrame(Xa[:5], columns=[f"f{j}" for j in range(Xa.shape[1])]).to_csv(PACED_DIR / f"{fd.tag}_X_{dummy}.csv", index=False)
            for d in DEPTHS:
                cases.append({"case": f"{fd.tag}_{s}_d{d}_l00", "fold": fd.tag, "fitset": s, "depth": d, "lambda": 0.0,
                              "minbucket": MIN_LEAF})
        pd.DataFrame({"price": np.zeros(fd.E.shape[1])}).to_csv(PACED_DIR / f"{fd.tag}_price.csv", index=False)
    pd.DataFrame(cases).to_csv(PACED_DIR / "cases.csv", index=False)
    cmd = shlex.split(ARGS.julia) + [str(HERE / "07_iai_fit.jl"), str(PACED_DIR)]
    log(f"IAI budget-aware fits ({len(cases)}): {' '.join(cmd)}")
    with open(PACED_DIR / "julia.log", "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"IAI failed, see {PACED_DIR / 'julia.log'}")


def select_budget_aware(fd, solver, rule):
    f = fd.fold
    Xtr, Ctr, _ = S3.paced_training_set(fd.E, fd.P, f.train)
    grid = []
    for d in DEPTHS:
        _, one = budget_aware_tree(fd, solver, d, "tr", Xtr, Ctr)
        for g in rollout_grid(fd, paced_rank(fd, one, f.val), rule):
            grid.append({"depth": d, "lambda": None, **g})
    pick = pick_near_best(grid, lambda g: (g["depth"], g["adjustments"], g["borrow"] or 0.0))
    Xtv, Ctv, lams = S3.paced_training_set(fd.E, fd.P, f.trainval)
    root, one = budget_aware_tree(fd, solver, pick["depth"], "tv", Xtv, Ctv)
    return pick, root, one, lams


def select_direct(fd, rule, budget_aware=False, lams=None):
    """Direct optimisation on decision-time (forecast) costs Ef; lambda (or the budget map) and B chosen on validation."""
    f = fd.fold
    grid = []
    if budget_aware:
        mk = lambda days: (lambda k, sl: np.argsort(fd.Ef[days[k]] + np.interp(sl, S3.SLACK_GRID, lams) * fd.P[days[k]], kind="stable").tolist())
        for g in rollout_grid(fd, mk(f.val), rule):
            grid.append({"lambda": None, **g})
        pick = pick_near_best(grid, lambda g: (g["adjustments"], g["borrow"] or 0.0))
        return pick, mk
    mk_l = lambda lam: (lambda days: np.argsort(fd.Ef[days] + lam * fd.P[days], axis=1, kind="stable").tolist())
    for lam in S3.LAMBDAS:
        for g in rollout_grid(fd, mk_l(lam)(f.val), rule):
            grid.append({"lambda": lam, **g})
    pick = pick_near_best(grid, lambda g: (g["adjustments"], -g["lambda"], g["borrow"] or 0.0))
    return pick, mk_l(pick["lambda"])


# ------------------------------------------------------------------------------------------------ evaluation
def evaluate(fd, acts, first, adj, target):
    f = fd.fold
    te = f.test
    D = len(te)
    low = np.stack([fd.menu[a]["low"] for a in acts])
    high = np.stack([fd.menu[a]["high"] for a in acts])
    s = S3.summarise(f, S3.evaluate(f, fd.world, low, high))
    hd = high.any(1)
    ld = low.any(1)
    q = [int(hd[idx].sum()) for idx in np.array_split(np.arange(D), 4)]
    fc_eve, re_eve = S3.X[te, 1], S3.CI[te][:, hours(16, 21)].mean(1)        # forecast / realised CI 16:00-21:00
    gap = lambda v, m: float(v[m].mean() - v[~m].mean()) if 0 < m.sum() < D else None
    first = np.asarray(first)
    out = {k: float(s[k]) for k in KEEP}
    out.update(emissions_reduction_pct=-float(s["emissions_change_pct"]), target=target, overrides=int(adj), n_days=D,
               follow_rate_pct=float(100 * (acts == first).mean()), high_event_days=int(hd.sum()), low_event_days=int(ld.sum()),
               high_days_by_quarter=q, high_share_first_half_pct=float(100 * hd[: D // 2].sum() / hd.sum()) if hd.any() else None,
               first_high_day=int(np.argmax(hd)) if hd.any() else None,
               last_high_day=int(D - 1 - np.argmax(hd[::-1])) if hd.any() else None,
               high_slots=int(high.sum()), distinct_prescribed=int(len(set(first.tolist()))),
               distinct_realised=int(len(set(acts.tolist()))),
               forecast_ci_evening_gap_high_vs_other=gap(fc_eve, hd), realised_ci_evening_gap_high_vs_other=gap(re_eve, hd),
               actions=acts.tolist(), prescribed=first.tolist())
    return out


def oracle(fd, target):
    te = fd.fold.test
    E, P = fd.E[te], fd.P[te]
    lam = tune_lambda(lambda l: (E + l * P).argmin(1), P, target)
    acts = (E + lam * P).argmin(1)
    return evaluate(fd, acts, acts, 0, target)


def historical(fd):
    te = fd.fold.test
    s = S3.summarise(fd.fold, fd.fold.ref)
    hd = S3.HIGH_H[te].any(1)
    D = len(te)
    return {"emissions_reduction_pct": 0.0, "mean_price": float(s["mean_price"]), "bill_change_mean_pct": 0.0,
            "high_event_days": int(hd.sum()), "high_days_by_quarter": [int(hd[i].sum()) for i in np.array_split(np.arange(D), 4)],
            "high_share_first_half_pct": float(100 * hd[: D // 2].sum() / hd.sum()) if hd.any() else None, "overrides": 0, "n_days": D,
            "high_flags": hd.astype(int).tolist()}


# ------------------------------------------------------------------------------------------------ one fold
def run_fold(fd, iai_ok):
    f = fd.fold
    te = f.test
    res = {"label": f.label, "target": fd.target, "matched_target": fd.matched, "menu": fd.names,
           "menu_prices": [s["price"] for s in fd.menu], "days": {"train": len(f.train), "val": len(f.val), "test": len(te)},
           "policies": {}, "historical": historical(fd)}
    pols = res["policies"]

    def add(name, kind, solver, rule, sel, ranks_test, extra=None):
        B = sel.get("borrow")
        entry = {"kind": kind, "solver": solver, "rule": rule, "selection": {k: v for k, v in sel.items()}, **(extra or {})}
        for tid, T in (("main", fd.target), ("matched", fd.matched)):
            acts, adj, first = run_rule(rule, ranks_test(T), fd.P[te], T, B)
            entry[tid] = evaluate(fd, acts, first, adj, T)
        pols[name] = entry
        m = entry["main"]
        log(f"  {name:55s} red {m['emissions_reduction_pct']:+.3f}%  P {m['mean_price']:.3f}  bill {m['bill_change_mean_pct']:+.1f}%  "
            f"overrides {m['overrides']}/{m['n_days']}  High days {m['high_event_days']} by quarter {m['high_days_by_quarter']}  "
            f"(B={B}, sel={ {k: v for k, v in sel.items() if k in ('depth', 'lambda')} })")

    solvers = ["exact"] + (["iai"] if iai_ok else [])
    for solver in solvers:
        # (1) trees selected under the original rule (07), evaluated with both rules; B chosen on val for that tree
        ch = fd.r7["chosen"][solver]
        _, rank_fn_tr = static_tree(fd, solver, ch["depth"], ch["lambda"], "tr")
        _, rank_fn_tv = static_tree(fd, solver, ch["depth"], ch["lambda"], "tv")
        base_sel = {"depth": ch["depth"], "lambda": ch["lambda"]}
        add(f"{solver} tree (selected under original rule) | original rule", "tree", solver, "original", base_sel,
            lambda T: rank_fn_tv(S3.X[te]))
        g = pick_near_best(rollout_grid(fd, rank_fn_tr(S3.X[f.val]), "pacing"), lambda g: (g["adjustments"], g["borrow"]))
        add(f"{solver} tree (selected under original rule) | pacing rule", "tree", solver, "pacing", {**base_sel, **g},
            lambda T: rank_fn_tv(S3.X[te]))
        # (2) trees re-selected (depth, lambda, B) under the pacing rule
        pick, root, rank_fn = select_static(fd, solver, "pacing")
        add(f"{solver} tree (re-selected under pacing) | pacing rule", "tree", solver, "pacing", pick, lambda T: rank_fn(S3.X[te]),
            {"tree_text": tree_text(root, fd.names, FEATS)})
        # (3) budget-aware trees (remaining budget as a state variable), each rule
        for rule in ("original", "pacing"):
            pick, root, one, _ = select_budget_aware(fd, solver, rule)
            add(f"{solver} budget-aware tree | {rule} rule", "budget-aware tree", solver, rule, pick,
                lambda T, one=one: paced_rank(fd, one, te), {"tree_text": tree_text(root, fd.names, PACED_FEATS)})
    # (4) constant policy (depth 0) and direct optimisation, each rule
    for rule in ("original", "pacing"):
        pick, _, rank_fn = select_static(fd, "exact", rule, depths=(0,))
        add(f"constant policy (depth 0) | {rule} rule", "constant", "exact", rule, pick, lambda T, rf=rank_fn: rf(S3.X[te]))
        pick, mk = select_direct(fd, rule)
        add(f"direct optimisation | {rule} rule", "direct", "-", rule, pick, lambda T, mk=mk: mk(te))
        lams = S3.paced_training_set(fd.E, fd.P, f.trainval)[2]
        pick, mk = select_direct(fd, rule, budget_aware=True, lams=lams)
        add(f"budget-aware direct optimisation | {rule} rule", "budget-aware direct", "-", rule, pick, lambda T, mk=mk: mk(te))
    res["oracle"] = {"main": oracle(fd, fd.target), "matched": oracle(fd, fd.matched)}
    # consistency: the original-rule trees reproduce 07's test results
    for solver in solvers:
        want = fd.r7["by_depth"][solver][str(fd.r7["chosen"][solver]["depth"])]["test"]["emissions_reduction_pct"]
        got = pols[f"{solver} tree (selected under original rule) | original rule"]["main"]["emissions_reduction_pct"]
        assert abs(want - got) < 1e-9, (solver, want, got)
    return res


# ------------------------------------------------------------------------------------------------ paired bootstrap (Nov-Dec)
def paired_bootstrap(fd, res, pairs, n=200):
    te = fd.fold.test
    model = ResponseModel("main")
    R_hist = tariff_regressors(S3.LOW_H, S3.HIGH_H, timeline=True)
    th = [[S3.THETA[f"boot_main_janoct_c{c}"][b] for c in range(4)] for b in range(n)]
    regs = {}
    for name in {p for pr in pairs for p in pr}:
        acts = np.array(res["policies"][name]["main"]["actions"])
        regs[name] = tariff_regressors(np.stack([fd.menu[a]["low"] for a in acts]), np.stack([fd.menu[a]["high"] for a in acts]))
    draws = {k: [] for k in regs}
    for b in range(n):
        m_hist = np.stack([model.combine(th[b][c], R_hist)[te] for c in range(4)])
        for k, R in regs.items():
            m = np.stack([model.combine(th[b][c], R) for c in range(4)])
            y = S3.test_consumption(fd.fold, np.exp(np.clip(m - m_hist, -S3.CLIP, S3.CLIP)))
            draws[k].append(-100 * (S3.emissions(fd.fold, y) - fd.fold.ref["emissions_kg"]) / fd.fold.footprint)
    out = {}
    for a, b in pairs:
        d = np.array(draws[b]) - np.array(draws[a])
        out[f"{b}  minus  {a}"] = {"mean_pp": float(d.mean()), "ci95_pp": list(np.percentile(d, [2.5, 97.5])),
                                   "share_draws_positive": float((d > 0).mean())}
    return out


# ------------------------------------------------------------------------------------------------ tables and figures
SHOW = ["exact tree (selected under original rule) | original rule", "exact tree (selected under original rule) | pacing rule",
        "exact tree (re-selected under pacing) | pacing rule", "iai tree (selected under original rule) | original rule",
        "iai tree (selected under original rule) | pacing rule", "iai tree (re-selected under pacing) | pacing rule",
        "exact budget-aware tree | original rule", "exact budget-aware tree | pacing rule",
        "iai budget-aware tree | original rule", "iai budget-aware tree | pacing rule",
        "constant policy (depth 0) | original rule", "constant policy (depth 0) | pacing rule",
        "direct optimisation | original rule", "direct optimisation | pacing rule",
        "budget-aware direct optimisation | original rule", "budget-aware direct optimisation | pacing rule"]


def md(df):
    f = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    return "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join(
        "| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n"


def tables(out):
    rows = []
    for mm, r in out["folds"].items():
        for tid in ("main", "matched"):
            h = r["historical"]
            rows.append({"window": r["label"], "target_type": tid, "policy": "historical TOU", "rule": "-",
                         "target": r["target"] if tid == "main" else r["matched_target"], "reduction_pct": 0.0,
                         "avg_price": h["mean_price"], "bill_change_pct": 0.0, "bill_p10_pct": 0.0, "bill_p90_pct": 0.0,
                         "overrides": "-", "follow_rate_pct": "-", "high_days": h["high_event_days"],
                         "high_days_by_quarter": "/".join(map(str, h["high_days_by_quarter"])),
                         "high_first_half_pct": h["high_share_first_half_pct"], "fcst_CI_gap_high_days": "-", "depth": "-", "lambda": "-", "B": "-"})
            for name in SHOW + ["oracle"]:
                if name == "oracle":
                    e, sel = r["oracle"][tid], {}
                    pol, rule = "oracle (hindsight)", "exact constraint"
                elif name in r["policies"]:
                    p = r["policies"][name]
                    e, sel = p[tid], p["selection"]
                    pol, rule = name.split(" | ")
                else:
                    continue
                rows.append({"window": r["label"], "target_type": tid, "policy": pol, "rule": rule, "target": e["target"],
                             "reduction_pct": e["emissions_reduction_pct"], "avg_price": e["mean_price"],
                             "bill_change_pct": e["bill_change_mean_pct"], "bill_p10_pct": e["bill_change_p10_pct"],
                             "bill_p90_pct": e["bill_change_p90_pct"], "overrides": e["overrides"],
                             "follow_rate_pct": round(e["follow_rate_pct"], 1), "high_days": e["high_event_days"],
                             "high_days_by_quarter": "/".join(map(str, e["high_days_by_quarter"])),
                             "high_first_half_pct": e["high_share_first_half_pct"],
                             "fcst_CI_gap_high_days": "-" if e["forecast_ci_evening_gap_high_vs_other"] is None else round(e["forecast_ci_evening_gap_high_vs_other"], 1),
                             "depth": sel.get("depth", "-") if sel.get("depth") is not None else "-",
                             "lambda": sel.get("lambda", "-") if sel.get("lambda") is not None else "-",
                             "B": sel.get("borrow", "-") if sel.get("borrow") is not None else "-"})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "pacing_all.csv", index=False)
    # summary across windows (main target)
    m = df[(df.target_type == "main") & (df.policy != "historical TOU")]
    summ = (m.groupby(["policy", "rule"], sort=False)
             .agg(mean_reduction_pct=("reduction_pct", "mean"), min_reduction_pct=("reduction_pct", "min"),
                  mean_bill_change_pct=("bill_change_pct", "mean"),
                  mean_overrides=("overrides", lambda v: np.mean([x for x in v if x != "-"]) if any(x != "-" for x in v) else np.nan),
                  mean_high_first_half_pct=("high_first_half_pct", lambda v: pd.to_numeric(v, errors="coerce").mean())).reset_index())
    mt = df[(df.target_type == "matched") & (df.policy != "historical TOU")]
    summ["mean_reduction_pct_matched_target"] = mt.groupby(["policy", "rule"], sort=False)["reduction_pct"].mean().to_numpy()
    summ.to_csv(RES / "pacing_summary.csv", index=False)
    return df, summ


def fig_timeline(out, fd):
    plt = mpl_style()
    r = out["folds"][str(S3.MAIN_TEST_MONTH)]
    show = [("historical TOU", None)] + [(n, n) for n in SHOW[:10] if n in r["policies"]]
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(show) + 1.4))
    names = r["menu"]
    D = r["days"]["test"]
    for i, (lab, key) in enumerate(show):
        y = len(show) - 1 - i
        if key is None:
            hd = np.array(r["historical"]["high_flags"])
            ax.scatter(np.where(hd)[0], [y] * hd.sum(), marker="s", s=28, color=INK2)
            txt = f"{hd.sum()} High days"
        else:
            e = r["policies"][key]["main"]
            acts, pres = np.array(e["actions"]), np.array(e["prescribed"])
            for d in range(D):
                nm = names[acts[d]]
                col = ORANGE if "High" in nm else (GREEN if "Low" in nm else None)
                if col:
                    ax.scatter(d, y, marker="s", s=28, color=col)
                if acts[d] != pres[d]:
                    ax.scatter(d, y - 0.28, marker="x", s=10, color=MUTED, lw=0.8)
            txt = f"{e['emissions_reduction_pct']:+.2f} %, {e['high_event_days']} High days, {e['overrides']} overrides"
        ax.text(D + 0.8, y, txt, va="center", fontsize=7.5)
    ax.set_yticks(range(len(show)), [s[0].replace(" (selected under original rule)", "").replace(" (re-selected under pacing)", " [re-selected]")
                                     for s in show][::-1], fontsize=7.5)
    ticks = np.arange(0, D, 7)
    ax.set_xticks(ticks, [str(DATES_2013[fd.fold.test[t]].date())[5:] for t in ticks], fontsize=7.5)
    ax.set_xlim(-1, D + 22)
    ax.grid(False)
    ax.set_title("Nov-Dec 2013 test window: realised High (orange) / Low (green) event days; x = feasibility override "
                 "(prescription replaced)\noriginal rule spends the price budget first; pacing rule spreads it", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGD / "fig5_pacing_timeline.png", bbox_inches="tight")
    plt.close(fig)


def fig_rolling(df):
    plt = mpl_style()
    pols = [("exact tree (selected under original rule)", "Exact tree"), ("iai tree (selected under original rule)", "IAI OPT"),
            ("exact budget-aware tree", "Budget-aware exact tree"), ("iai budget-aware tree", "Budget-aware IAI tree"),
            ("constant policy (depth 0)", "Constant policy (depth 0)"), ("direct optimisation", "Direct optimisation")]
    pols = [p for p in pols if (df.policy == p[0]).any()]
    wins = list(dict.fromkeys(df.window))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, tid in zip(axes, ("main", "matched")):
        x = np.arange(len(pols))
        w = 0.8 / len(wins)
        cols = [MUTED, BLUE, VIOLET, ORANGE]
        for j, win in enumerate(wins):
            for k, rule in enumerate(("original rule", "pacing rule")):
                v = [df[(df.window == win) & (df.target_type == tid) & (df.policy == p) & (df.rule == rule)].reduction_pct.iloc[0] for p, _ in pols]
                ax.bar(x + (j - len(wins) / 2 + 0.5) * w + (k - 0.5) * w * 0.45, v, w * 0.45, color=cols[j],
                       alpha=0.45 if k == 0 else 1.0, label=f"{win}, {rule}" if tid == "main" else None)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_xticks(x, [p[1] for p in pols], fontsize=7.5, rotation=15)
        ax.set_title("(a) Deployable target (train+val TOU average)" if tid == "main"
                     else "(b) Retrospective price-matched target (test-window TOU average)")
    axes[0].set_ylabel("Estimated emissions reduction vs TOU (%)")
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Original (light) vs pacing (dark) feasibility rule across the four rolling test windows (model-based estimates)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGD / "fig6_pacing_rolling.png", bbox_inches="tight")
    plt.close(fig)


def main():
    fds = [FoldData(mm) for mm in FOLDS]
    iai_ok = IAI07["iai_available"] and not ARGS.skip_iai
    if iai_ok:
        try:
            export_budget_aware(fds)
        except Exception as e:
            log(f"IAI budget-aware fits unavailable: {e}")
            iai_ok = False
    out = {"metric": "average carbon intensity (realised) for evaluation; forecast CI for decisions", "borrow_grid": BORROW_GRID,
           "folds": {}}
    for fd in fds:
        log(f"=== fold {fd.fold.label}: target {fd.target:.3f}, matched {fd.matched:.3f}, menu {fd.names}")
        out["folds"][str(fd.mm)] = run_fold(fd, iai_ok)
    main_fd = fds[-1]
    r = out["folds"][str(S3.MAIN_TEST_MONTH)]
    pairs = [(f"{s} tree (selected under original rule) | original rule", f"{s} tree (selected under original rule) | pacing rule")
             for s in (["exact", "iai"] if iai_ok else ["exact"])]
    pairs += [(f"{s} tree (selected under original rule) | original rule", f"{s} budget-aware tree | pacing rule")
              for s in (["exact", "iai"] if iai_ok else ["exact"])]
    pairs += [("exact budget-aware tree | original rule", "exact budget-aware tree | pacing rule")]
    out["paired_bootstrap_nov_dec"] = paired_bootstrap(main_fd, r, pairs)
    for k, v in out["paired_bootstrap_nov_dec"].items():
        log(f"bootstrap {k}: {v['mean_pp']:+.3f} pp [{v['ci95_pp'][0]:+.3f}, {v['ci95_pp'][1]:+.3f}]")
    save_json(out, RES / "pacing_results.json")
    df, summ = tables(out)
    fig_timeline(out, main_fd)
    fig_rolling(df)
    txt = ["# Pacing experiment tables (generated by scripts/10_pacing_experiment.py)", "",
           "Model-based estimates: simulated demand x realised average CI. Reduction (%) = 100 (E_hist - E_policy) / E_hist. "
           "Overrides = days the rule replaced the policy's first-ranked action. High days by quarter = High-event days in each "
           "quarter of the window. fcst_CI_gap_high_days = mean forecast CI 16-21 h on High-event days minus other days (g/kWh). "
           "B = borrowing allowance of the pacing rule (p/kWh x days), chosen on validation.", "",
           "## Summary across the four windows", "", md(summ.round(3)), ""]
    for mm, rr in out["folds"].items():
        for tid in ("main", "matched"):
            sub = df[(df.window == rr["label"]) & (df.target_type == tid)].drop(columns=["window", "target_type"])
            txt += [f"## {rr['label']} — {'deployable target' if tid == 'main' else 'price-matched target (retrospective)'} "
                    f"{rr['target'] if tid == 'main' else rr['matched_target']:.3f} p/kWh", "", md(sub), ""]
    txt += ["## Paired bootstrap, Nov-Dec (difference in reduction, percentage points; 200 demand-model draws)", "",
            md(pd.DataFrame([{"comparison": k, "mean_pp": v["mean_pp"], "ci95_low": v["ci95_pp"][0], "ci95_high": v["ci95_pp"][1],
                              "share_positive": v["share_draws_positive"]} for k, v in out["paired_bootstrap_nov_dec"].items()]))]
    (RES / "pacing_tables.md").write_text("\n".join(txt), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
