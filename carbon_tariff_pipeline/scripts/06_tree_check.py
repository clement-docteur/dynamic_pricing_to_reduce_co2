"""Tree-solver check: is Step 3's policy tree limited by our search, and how does Interpretable AI's Optimal Policy
Tree (the proposal's OPT) compare on the same inputs?

For each rolling fold and each tree (fixed; budget-aware), Step 3's training set is rebuilt from its saved choices
(menu and lambda; features X and Lagrangian costs E + lambda * P on the training + validation days), and the tree is
refitted at depths 1-3 by
  * 'step3': our search as run in Step 3 (every threshold at the last level, 12 quantile thresholds above it),
  * 'fine':  every threshold at every level at depth <= 2 (the global optimum); at depth 3, FINE_D3 = 24 quantile
             thresholds above the last level, twice Step 3's (the exhaustive depth-3 search takes hours in Python),
  * 'iai':   Interpretable AI's OptimalTreePolicyMinimizer (same depth, minbucket 10, cp 0), if its fits are in
             outputs/iai/.
Reported per fit: training cost, leaves and agreement with Step 3's prescriptions; at the depth chosen in Step 3, the
test-window outcome under the feasibility rule (the fold's own price target).
Check: the 'step3' refits reproduce the saved Step 3 trees and their test results exactly.

IAI does not run on the PC this pipeline was built on (Smart App Control blocks the libraries bundled in its system
image), so `--export` writes its inputs to outputs/iai/ and scripts/06_iai_fit.jl fits it on another machine:
    python -P scripts/06_tree_check.py --export
    julia +1.11.6/IAI/3.2.2 scripts/06_iai_fit.jl outputs/iai
    python -P scripts/06_tree_check.py
Output: outputs/tree_check.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--export", action="store_true", help="write the IAI inputs to outputs/iai/ and stop")
EXPORT = _ap.parse_args().export
sys.argv = sys.argv[:1]                                   # 03_policy parses its own --metric (default: marginal)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import OUT, load_json, make_log, save_json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.policy import NORMAL, POLICIES, PolicyTree, candidate_pool, feasibility_rule  # noqa: E402

_spec = importlib.util.spec_from_file_location("step3", Path(__file__).with_name("03_policy.py"))
S3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S3)                              # Step 3's data, simulator, costs and protocol

log, _ = make_log()
IAI_DIR = OUT / "iai"
FINE_D3 = 24
MIN_LEAF = 10
DEPTHS = (1, 2, 3)
S3R = load_json(OUT / "step3_results.json")
POOL = {s["name"]: s for s in candidate_pool()}
TREES = {"fixed": POLICIES[2], "paced": POLICIES[6]}      # tree kind -> policy name in Step 3's results


def fold_setup(mm):
    """Step 3's tree training sets and evaluation inputs for the rolling fold whose test window starts in month mm."""
    r = S3R["rolling"][str(mm)]
    fold, world = S3.Fold(mm), S3.response_world(fit=f"before{mm:02d}")
    menu = [POOL[n] for n in r["menu"]]
    E = S3.costs(world, world.m_menu(menu), menu)["E"]
    P = np.broadcast_to([s["price"] for s in menu], E.shape)
    target = float(S3.PRICE_H[fold.trainval].mean())
    assert abs(target - r["target"]) < 1e-12
    tv = fold.trainval
    Xa, Ca, _ = S3.paced_training_set(E, P, tv)
    data = {"fixed": (S3.X[tv], E[tv] + r["lambda_tree"] * P[tv]), "paced": (Xa, Ca)}
    return r, fold, world, menu, P, target, data


def test_result(fold, world, menu, P, target, rank_of, paced):
    """Test-window outcome of a tree (rank_of: feature row -> action ranking) under the feasibility rule."""
    te = fold.test
    rankings = (lambda k, sl: rank_of(np.append(S3.X[te[k]], sl))) if paced else [rank_of(x) for x in S3.X[te]]
    acts, adj = feasibility_rule(rankings, P[te], target)
    low, high = np.stack([menu[a]["low"] for a in acts]), np.stack([menu[a]["high"] for a in acts])
    s = S3.summarise(fold, S3.evaluate(fold, world, low, high))
    return {"emissions_change_pct": s["emissions_change_pct"], "bill_change_mean_pct": s["bill_change_mean_pct"],
            "event_days": int((acts != NORMAL).sum()), "adjustments": adj}


def iai_tree(case, X):
    """(rank_of, n_leaves) of the IAI tree in outputs/iai/<case>_nodes.csv. IAI's split direction (lower child when
    x < t, or when x <= t) is the one that reproduces IAI's own predictions on every fitting row."""
    nodes = pd.read_csv(IAI_DIR / f"{case}_nodes.csv").set_index("node")
    want = pd.read_csv(IAI_DIR / f"{case}_fitrank.csv")["first"].str[1:].astype(int).to_numpy()

    def router(strict):
        def rank_of(x):
            t = 1
            while not nodes.at[t, "leaf"]:
                v, thr = x[int(nodes.at[t, "feature"][1:])], nodes.at[t, "threshold"]
                t = int(nodes.at[t, "lower"] if (v < thr if strict else v <= thr) else nodes.at[t, "upper"])
            return [int(a[1:]) for a in nodes.at[t, "ranking"].split()]
        return rank_of

    for strict in (True, False):
        rank_of = router(strict)
        if all(rank_of(x)[0] == w for x, w in zip(X, want)):
            return rank_of, int(nodes["leaf"].sum())
    raise ValueError(f"{case}: the IAI node table does not reproduce IAI's own predictions")


def export(cases):
    IAI_DIR.mkdir(exist_ok=True)
    for case, (X, C, _) in cases.items():
        pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])]).to_csv(IAI_DIR / f"{case}_X.csv", index=False)
        pd.DataFrame(C, columns=[f"a{a}" for a in range(C.shape[1])]).to_csv(IAI_DIR / f"{case}_C.csv", index=False)
    pd.DataFrame([{"case": c, "depth": d, "minbucket": MIN_LEAF} for c, (_, _, d) in cases.items()]).to_csv(
        IAI_DIR / "cases.csv", index=False)
    log(f"wrote {len(cases)} IAI cases to {IAI_DIR}")


def main():
    out, cases = {}, {}
    for mm in S3.ROLLING_TEST_MONTHS:
        r, fold, world, menu, P, target, data = fold_setup(mm)
        fo = {"label": fold.label, "chosen_depth": {"fixed": r["depth"], "paced": r["paced_depth"]}, "fits": {}, "test": {}}
        for kind, (Xk, Ck) in data.items():
            chosen = fo["chosen_depth"][kind]
            fo["fits"][kind] = {}
            for depth in DEPTHS:
                case = f"{fold.label.replace('-', '')}_{kind}_d{depth}"          # e.g. NovDec_paced_d2
                cases[case] = (Xk, Ck, depth)
                if EXPORT:
                    continue
                trees = {"step3": PolicyTree(depth, min_leaf=MIN_LEAF).fit(Xk, Ck),
                         "fine": PolicyTree(depth, min_leaf=MIN_LEAF, n_thresh=None if depth <= 2 else FINE_D3).fit(Xk, Ck)}
                rank_of = {k: (lambda x, t=t: t.leaf_of(x)["ranking"]) for k, t in trees.items()}
                leaves = {k: t.n_leaves() for k, t in trees.items()}
                if (IAI_DIR / f"{case}_nodes.csv").exists():
                    rank_of["iai"], leaves["iai"] = iai_tree(case, Xk)
                acts = {k: np.array([f(x)[0] for x in Xk]) for k, f in rank_of.items()}
                fo["fits"][kind][depth] = {
                    "rows": len(Xk), "single_action_cost": float(Ck.sum(0).min()), "per_row_best_cost": float(Ck.min(1).sum()),
                    "cost": {k: float(Ck[np.arange(len(Ck)), a].sum()) for k, a in acts.items()}, "leaves": leaves,
                    "agree_with_step3_pct": {k: float(100 * (a == acts["step3"]).mean()) for k, a in acts.items()}}
                if depth == chosen:
                    saved = r["tree"] if kind == "fixed" else r["paced_tree"]
                    assert json.loads(json.dumps(trees["step3"].root)) == saved, f"{case}: Step 3 tree not reproduced"
                    fo["test"][kind] = {k: test_result(fold, world, menu, P, target, f, kind == "paced") for k, f in rank_of.items()}
                    want = r["main"][TREES[kind]]["emissions_change_pct"]
                    assert abs(fo["test"][kind]["step3"]["emissions_change_pct"] - want) < 1e-9, f"{case}: test result not reproduced"
                g = fo["fits"][kind][depth]
                log(f"{case}: rows {g['rows']}, " + ", ".join(
                    f"{k} gain {g['single_action_cost'] - c:.3f} kg ({leaves[k]} leaves, {g['agree_with_step3_pct'][k]:.1f}% agree)"
                    for k, c in g["cost"].items()))
            if not EXPORT:
                log(f"{fold.label} {kind} (depth {chosen}) test: " + ", ".join(
                    f"{k} {v['emissions_change_pct']:+.3f}%" for k, v in fo["test"][kind].items()))
        out[str(mm)] = fo
    if EXPORT:
        export(cases)
        return
    n_iai = sum((IAI_DIR / f"{c}_nodes.csv").exists() for c in cases)
    save_json({"fine_depth3_quantiles": FINE_D3, "min_leaf": MIN_LEAF, "iai_cases": n_iai, "n_cases": len(cases),
               "iai_settings": "OptimalTreePolicyMinimizer(max_depth=depth, minbucket=10, cp=0, random_seed=1)",
               "folds": out}, OUT / "tree_check.json")
    log(f"done ({n_iai}/{len(cases)} IAI fits found)")


if __name__ == "__main__":
    main()
