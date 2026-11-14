"""MVP Step 4: Interpretable AI's Optimal Policy Trees (Lecture 11 OPT) vs the pipeline's exact policy tree.

Both trees are fitted on exactly the same inputs as Step 3's fixed policy tree (scripts/03_policy.py):
  features X (decision-time information only, see 03_policy.features) and Lagrangian model-based action costs
  C = E + lambda * price, E = simulated emissions of each menu action on past days.
Protocol (identical for both solvers, fold = test window starting in month mm; main analysis mm = 11, Nov-Dec):
  1. for every depth in {1, 2, 3} and lambda in Step 3's grid: fit on the training days, then roll the tree's rankings
     through the online average-price feasibility rule on the validation days and record validation emissions;
  2. per depth, keep the best lambda; the overall choice (depth, lambda) uses Step 3's rule (validation emissions within
     1e-4 of the best, then shallowest, fewest adjustments, largest lambda);
  3. refit each depth on training + validation days at its lambda and evaluate it ONCE on the test window under the
     feasibility rule (realised emission factor, simulated demand) - all depths are reported, the chosen one is flagged.
IAI settings: OptimalTreePolicyMinimizer(max_depth = depth, minbucket = 10, cp = 0, random_seed = 1) (the exact tree uses
min_leaf = 10 and no complexity penalty too). IAI is run through scripts/07_iai_fit.jl.

"Overrides" = days on which the feasibility rule could not apply the tree's first-ranked action (the price budget would
no longer be attainable) and moved down the leaf's ranking, usually to the all-Normal day.

Usage:  python -P scripts/07_iai_opt.py [--metric average] [--temperature lagged] [--tag mvp] [--folds 11|5,7,9,11]
Output: OUT/iai_opt_results<suffix>.json and OUT/iai_opt<suffix>/ (IAI inputs, fitted trees, rankings)
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
_ap.add_argument("--metric", choices=["marginal", "average"], default="average")
_ap.add_argument("--temperature", choices=["realised", "lagged"], default="lagged")
_ap.add_argument("--tag", default="mvp")
_ap.add_argument("--folds", default="11", help="comma-separated first test months (11 = main Nov-Dec fold)")
_ap.add_argument("--julia", default="julia +1.12.6/IAI/3.2.2", help="command that starts IAI's Julia")
_ap.add_argument("--reuse", action="store_true", help="reuse existing IAI fits instead of refitting")
A = _ap.parse_args()
sys.argv = [sys.argv[0], "--metric", A.metric, "--temperature", A.temperature] + (["--tag", A.tag] if A.tag else [])
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import OUT, load_json, make_log, save_json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.policy import NORMAL, PolicyTree, candidate_pool, feasibility_rule, pick_near_best  # noqa: E402

_spec = importlib.util.spec_from_file_location("step3", Path(__file__).with_name("03_policy.py"))
S3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S3)                              # Step 3's data, features, simulator, costs and protocol

log, _ = make_log()
DEPTHS = (1, 2, 3)
MIN_LEAF = 10
FEATS = list(S3.FEATURES)
IAI_DIR = OUT / f"iai_opt{S3.SUFFIX}"
SOLVERS = ("exact", "iai")
KEEP = ("emissions_change_pct", "emissions_change_kg_per_hh", "mean_price", "price_paid", "kwh_change_pct",
        "daily_peak_change_pct", "bill_change_mean_pct", "bill_change_p10_pct", "bill_change_median_pct",
        "bill_change_p90_pct", "high_slots", "low_slots")


# ------------------------------------------------------------------------------------------------ fold inputs
def fold_inputs(mm):
    """Step 3's fold, simulator, menu (chosen on training days), realised costs E and prices P for test month mm."""
    fold = S3.Fold(mm)
    world = S3.response_world() if mm == S3.MAIN_TEST_MONTH else S3.response_world(fit=f"before{mm:02d}")
    pool = candidate_pool()
    Mp = world.m_menu(pool)
    target = float(S3.PRICE_H[fold.trainval].mean())
    menu_idx = S3.select_menu(fold, world, pool, Mp, target)
    menu = [pool[i] for i in menu_idx]
    E = S3.costs(world, Mp[:, :, menu_idx], menu)["E"]
    P = np.broadcast_to([s["price"] for s in menu], E.shape)
    return fold, world, menu, E, P, target


def export(tag, fold, E, P):
    sets = {"tr": fold.train, "tv": fold.trainval, "va": fold.val, "te": fold.test}
    for s, days in sets.items():
        pd.DataFrame(S3.X[days], columns=[f"f{j}" for j in range(len(FEATS))]).to_csv(IAI_DIR / f"{tag}_X_{s}.csv", index=False)
        if s in ("tr", "tv"):
            pd.DataFrame(E[days], columns=[f"a{a}" for a in range(E.shape[1])]).to_csv(IAI_DIR / f"{tag}_E_{s}.csv", index=False)
    pd.DataFrame({"price": P[0]}).to_csv(IAI_DIR / f"{tag}_price.csv", index=False)
    return [{"case": f"{tag}_{s}_d{d}_l{li:02d}", "fold": tag, "fitset": s, "depth": d, "lambda": lam, "minbucket": MIN_LEAF}
            for s in ("tr", "tv") for d in DEPTHS for li, lam in enumerate(S3.LAMBDAS)]


def run_iai(cases):
    pd.DataFrame(cases).to_csv(IAI_DIR / "cases.csv", index=False)
    if A.reuse and all((IAI_DIR / f"{c['case']}_nodes.csv").exists() for c in cases):
        log(f"reusing {len(cases)} IAI fits in {IAI_DIR}")
        return
    cmd = shlex.split(A.julia) + [str(Path(__file__).with_name("07_iai_fit.jl")), str(IAI_DIR)]
    log(f"running IAI ({len(cases)} fits): {' '.join(cmd)}")
    with open(IAI_DIR / "julia.log", "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"IAI failed (exit {r.returncode}); see {IAI_DIR / 'julia.log'}")


# ------------------------------------------------------------------------------------------------ IAI tree reader
def iai_tree(case):
    """IAI's fitted tree as a nested dict in PolicyTree's format (left = lower child). The split direction (x < t or
    x <= t goes to the lower child) is the one that reproduces IAI's own predictions on every fitting row."""
    nodes = pd.read_csv(IAI_DIR / f"{case}_nodes.csv").set_index("node")
    want = pd.read_csv(IAI_DIR / f"{case}_rank_fit.csv").to_numpy(int)[:, 0] - 0

    def build(t):
        r = nodes.loc[t]
        if bool(r["leaf"]):
            rank = [int(a[1:]) for a in str(r["ranking"]).split()]
            return {"leaf": True, "action": rank[0], "n": int(r["n"]), "ranking": rank}
        return {"leaf": False, "feature": int(str(r["feature"])[1:]), "threshold": float(r["threshold"]), "n": int(r["n"]),
                "left": build(int(r["lower"])), "right": build(int(r["upper"]))}

    root = build(1)
    return root, want


def route(node, x, strict):
    while not node["leaf"]:
        v, t = x[node["feature"]], node["threshold"]
        node = node["left"] if (v < t if strict else v <= t) else node["right"]
    return node


def n_leaves(node):
    return 1 if node["leaf"] else n_leaves(node["left"]) + n_leaves(node["right"])


def tree_text(node, names, indent=""):
    if node["leaf"]:
        return f"{indent}-> {names[node['action']]}  (n={node['n']})\n"
    op = "<" if node.get("strict") else "<="
    return (f"{indent}if {FEATS[node['feature']]} {op} {node['threshold']:.4g}:\n" + tree_text(node["left"], names, indent + "    ")
            + f"{indent}else:\n" + tree_text(node["right"], names, indent + "    "))


def mark_strict(node, strict):
    node["strict"] = strict
    if not node["leaf"]:
        mark_strict(node["left"], strict)
        mark_strict(node["right"], strict)
    return node


def iai_fitted(case, X_fit):
    """(root, rankings function) of an IAI fit, after checking that the node table reproduces IAI's predictions."""
    root, _ = iai_tree(case)
    ranks_fit = pd.read_csv(IAI_DIR / f"{case}_rank_fit.csv").to_numpy(int)
    for strict in (True, False):
        if all(route(root, x, strict)["ranking"][0] == r[0] for x, r in zip(X_fit, ranks_fit)):
            mark_strict(root, strict)
            return root, (lambda X, s=strict: [route(root, x, s)["ranking"] for x in np.asarray(X, float)])
    raise ValueError(f"{case}: the IAI node table does not reproduce IAI's own predictions")


# ------------------------------------------------------------------------------------------------ evaluation
def evaluate_test(fold, world, menu, P, target, ranks):
    te = fold.test
    names = [s["name"] for s in menu]
    acts, adj = feasibility_rule(ranks, P[te], target)
    low, high = np.stack([menu[a]["low"] for a in acts]), np.stack([menu[a]["high"] for a in acts])
    s = S3.summarise(fold, S3.evaluate(fold, world, low, high))
    out = {k: s[k] for k in KEEP}
    out.update(emissions_reduction_pct=-s["emissions_change_pct"], overrides=adj, n_days=len(te),
               override_rate_pct=100 * adj / len(te), event_days=int((acts != NORMAL).sum()),
               prescribed_counts=dict(Counter(names[r[0]] for r in ranks)), realised_counts=dict(Counter(names[a] for a in acts)),
               actions=acts.tolist(), prescribed=[int(r[0]) for r in ranks])
    return out


def run_fold(mm, iai_ok):
    fold, world, menu, E, P, target = fold_inputs(mm)
    names = [s["name"] for s in menu]
    tag = fold.label.replace("-", "")
    tr, va, tv, te = fold.train, fold.val, fold.trainval, fold.test
    log(f"fold {fold.label}: menu {names}, target {target:.3f} p/kWh, days train/val/test {len(tr)}/{len(va)}/{len(te)}")
    res = {"label": fold.label, "target": target, "historical_test_price": float(S3.PRICE_H[te].mean()), "menu": names,
           "menu_prices": [s["price"] for s in menu], "grid": {}, "chosen": {}, "by_depth": {}}
    for solver in SOLVERS:
        if solver == "iai" and not iai_ok:
            continue
        grid, fitted = [], {}
        for d in DEPTHS:
            for li, lam in enumerate(S3.LAMBDAS):
                C_tr = E[tr] + lam * P[tr]
                if solver == "exact":
                    tree = PolicyTree(d, min_leaf=MIN_LEAF, feature_names=FEATS).fit(S3.X[tr], C_tr)
                    root, rank_fn = tree.root, tree.rankings
                else:
                    root, rank_fn = iai_fitted(f"{tag}_tr_d{d}_l{li:02d}", S3.X[tr])
                first = np.array([r[0] for r in rank_fn(S3.X[tr])])
                acts, adj = feasibility_rule(rank_fn(S3.X[va]), P[va], target)
                grid.append({"depth": d, "lambda": lam, "li": li, "val_emissions_kg": float(E[va, acts].sum()), "adjustments": adj,
                             "leaves": n_leaves(root), "train_cost": float(C_tr[np.arange(len(tr)), first].sum())})
        pick = pick_near_best(grid, lambda g: (g["depth"], g["adjustments"], -g["lambda"]))
        res["grid"][solver] = grid
        res["chosen"][solver] = {"depth": pick["depth"], "lambda": pick["lambda"]}
        res["by_depth"][solver] = {}
        for d in DEPTHS:
            g = pick_near_best([x for x in grid if x["depth"] == d], lambda x: (x["adjustments"], -x["lambda"]))
            C_tv = E[tv] + g["lambda"] * P[tv]
            if solver == "exact":
                tree = PolicyTree(d, min_leaf=MIN_LEAF, feature_names=FEATS).fit(S3.X[tv], C_tv)
                root, rank_fn = tree.root, tree.rankings
            else:
                root, rank_fn = iai_fitted(f"{tag}_tv_d{d}_l{g['li']:02d}", S3.X[tv])
            first = np.array([r[0] for r in rank_fn(S3.X[tv])])
            entry = {"lambda": g["lambda"], "val_emissions_kg": g["val_emissions_kg"], "val_adjustments": g["adjustments"],
                     "trainval_cost": float(C_tv[np.arange(len(tv)), first].sum()),
                     "trainval_single_action_cost": float(C_tv.sum(0).min()), "trainval_per_day_best_cost": float(C_tv.min(1).sum()),
                     "leaves": n_leaves(root), "tree": root, "text": tree_text(root, names),
                     "test": evaluate_test(fold, world, menu, P, target, rank_fn(S3.X[te])),
                     "chosen": d == pick["depth"]}
            res["by_depth"][solver][d] = entry
            t = entry["test"]
            log(f"  {solver} depth {d}: lambda {g['lambda']}, val {g['val_emissions_kg']:.3f} kg (adj {g['adjustments']}), "
                f"leaves {entry['leaves']}, test reduction {t['emissions_reduction_pct']:+.3f}%, overrides {t['overrides']}/{t['n_days']}"
                + (" <- chosen" if entry["chosen"] else ""))
    if iai_ok:
        res["agreement_test_prescriptions_pct"] = {
            d: float(100 * np.mean(np.array(res["by_depth"]["exact"][d]["test"]["prescribed"])
                                   == np.array(res["by_depth"]["iai"][d]["test"]["prescribed"]))) for d in DEPTHS}
    return res


def main():
    IAI_DIR.mkdir(parents=True, exist_ok=True)
    folds = [int(x) for x in A.folds.split(",")]
    cases = []
    for mm in folds:
        fold, _, _, E, P, _ = fold_inputs(mm)
        cases += export(fold.label.replace("-", ""), fold, E, P)
    iai_ok, err = True, None
    try:
        run_iai(cases)
    except Exception as e:                                # keep the exact-tree results if IAI is unavailable
        iai_ok, err = False, str(e)
        log(f"IAI unavailable: {e}")
    out = {"metric": S3.METRIC, "temperature_feature": S3.TEMPERATURE, "features": FEATS, "depths": DEPTHS, "min_leaf": MIN_LEAF,
           "lambdas": S3.LAMBDAS, "iai_settings": "OptimalTreePolicyMinimizer(max_depth=depth, minbucket=10, cp=0, random_seed=1)",
           "iai_available": iai_ok, "iai_error": err, "folds": {str(mm): run_fold(mm, iai_ok) for mm in folds}}
    # consistency with Step 3: the exact tree chosen here is Step 3's tree
    s3_path = OUT / f"step3_results{S3.SUFFIX}.json"
    main = out["folds"].get(str(S3.MAIN_TEST_MONTH))
    if main and s3_path.exists():
        s3 = load_json(s3_path)
        ch = main["chosen"]["exact"]
        same = (ch["depth"] == s3["depth"] and ch["lambda"] == s3["lambda_tree"]
                and main["by_depth"]["exact"][ch["depth"]]["tree"] == s3["tree"])
        out["exact_matches_step3"] = bool(same)
        log(f"exact tree identical to Step 3's chosen tree: {same}")
    save_json(out, OUT / f"iai_opt_results{S3.SUFFIX}.json")
    log("done")


if __name__ == "__main__":
    main()
