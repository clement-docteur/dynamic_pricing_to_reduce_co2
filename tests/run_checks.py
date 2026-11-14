"""Automated checks for the MVP (no extra dependencies):  .venv/bin/python tests/run_checks.py

1. Unit tests of the feasibility rules in carbon_tariff_pipeline/lib/policy.py.
2. Clement's original code outputs and data artefacts are unchanged vs his commit 6c22f28.
3. Reproduction: Clement's Steps 2-3 re-run exactly (results/reproduction/reproduction_check.json).
4. MVP sanity / leakage checks all pass (results/tables/sanity_checks.json).
5. Constraint compliance in every saved experiment (announced-price target; predicted bill cap).
6. Cross-file consistency of the numbers reused across experiments and in the final summary table.
Requires the outputs of `run_mvp.py --reproduce` (all stages). Exit code 0 = all checks passed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "carbon_tariff_pipeline"))
import numpy as np  # noqa: E402

from lib.policy import NORMAL, feasibility_rule, multi_budget_rule, pacing_rule  # noqa: E402

CLEMENT_COMMIT = "6c22f28"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ 1. rules
def test_rules():
    rng = np.random.default_rng(0)
    P = np.tile([11.76, 18.69, 10.79, 16.38, 21.0], (61, 1))
    Pi = P * rng.uniform(0.95, 1.3, P.shape)
    Pi[:, NORMAL] = 11.76
    T = 13.608
    ok1 = ok2 = ok3 = ok4 = True
    for _ in range(200):
        R = [list(rng.permutation(5)) for _ in range(61)]
        a, j = feasibility_rule(R, P, T)
        b, i = pacing_rule(R, P, T, np.inf)
        c, k, _ = multi_budget_rule(R, [(P, T)])
        ok1 &= (a == b).all() and i == j
        ok2 &= (a == c).all() and k == j
        d, _ = pacing_rule(R, P, T, 0.0)
        ok3 &= bool((np.cumsum(P[0, d]) <= np.arange(1, 62) * T + 1e-9).all() and P[0, d].mean() <= T + 1e-9)
        e, _, _ = multi_budget_rule(R, [(P, T), (Pi, 13.0)])
        ok4 &= bool(P[np.arange(61), e].mean() <= T + 1e-9 and Pi[np.arange(61), e].mean() <= 13.0 + 1e-9)
    check("pacing_rule(borrow=inf) == feasibility_rule (200 random rankings)", ok1)
    check("multi_budget_rule(single budget) == feasibility_rule (200 random rankings)", ok2)
    check("pacing_rule(borrow=0) keeps the cumulative pace and the period target", ok3)
    check("multi_budget_rule satisfies both budgets (200 random rankings)", ok4)


# ------------------------------------------------------------------------------------------------ 2-4. preservation
def test_preservation():
    paths = ["carbon_tariff_pipeline/outputs", "carbon_tariff_pipeline/figures", "carbon_tariff_pipeline/report.html",
             "CVAE_reproduction/outputs", "CVAE_reproduction/data", "final_report/main.tex"]
    r = subprocess.run(["git", "diff", "--quiet", CLEMENT_COMMIT, "--", *paths], cwd=ROOT)
    check("Clement's committed outputs, figures, report and data artefacts unchanged since 6c22f28", r.returncode == 0, ", ".join(paths))
    r = subprocess.run(["git", "diff", "--quiet", CLEMENT_COMMIT, "--", "carbon_tariff_pipeline/scripts/0*.py",
                        ":(exclude)carbon_tariff_pipeline/scripts/03_policy.py", ":(exclude)carbon_tariff_pipeline/scripts/07_*",
                        ":(exclude)carbon_tariff_pipeline/scripts/08_*", ":(exclude)carbon_tariff_pipeline/scripts/09_*"], cwd=ROOT)
    check("Clement's pipeline scripts unchanged except the opt-in flags in 03_policy.py", r.returncode == 0)
    rc = load("results/reproduction/reproduction_check.json")["files"]
    s23 = {k: v for k, v in rc.items() if "results/reproduction/outputs" in k and isinstance(v, dict) and "mismatches" in v}
    check("Clement's Steps 2-3 reproduce exactly from his Step-1 inputs", s23 and all(v["mismatches"] == 0 for v in s23.values()),
          f"{sum(v['values'] for v in s23.values())} numbers compared")
    arr = {k: v for k, v in rc.items() if "results/reproduction/outputs" in k and isinstance(v, dict) and "max_abs_diff_per_array" in v}
    mx = max(max(x for x in v["max_abs_diff_per_array"].values() if isinstance(x, float)) for v in arr.values())
    check("Reproduced arrays agree to < 1e-12", mx < 1e-12, f"max abs diff {mx:.1e}")
    sc = load("results/tables/sanity_checks.json")
    check("MVP sanity / leakage checks all pass", all(c["passed"] for c in sc), f"{sum(c['passed'] for c in sc)}/{len(sc)}")


# ------------------------------------------------------------------------------------------------ 5-6. constraints and consistency
def test_constraints_and_consistency():
    s3 = load("results/mvp/outputs/step3_results_average_mvp.json")
    T = s3["target_price"]
    bad = {p: v["mean_price"] for p, v in s3["results"].items() if not p.startswith("Historical") and v["mean_price"] > T + 1e-9}
    check("Step 3 (Nov-Dec): every proposed policy meets the announced-price target", not bad, str(bad or ""))
    bad = [(m, p) for m, r in s3["rolling"].items() for p, v in r["main"].items()
           if not p.startswith("Historical") and v["mean_price"] > r["target"] + 1e-9]
    check("Step 3 rolling windows: every proposed policy meets its window's price target", not bad, str(bad or ""))
    i7 = load("results/mvp/outputs/iai_opt_results_average_mvp.json")
    check("IAI run available and exact tree identical to Step 3's chosen tree", i7["iai_available"] and i7.get("exact_matches_step3"))
    bad = [(m, s, d) for m, f in i7["folds"].items() for s in ("exact", "iai") for d, e in f["by_depth"][s].items()
           if e["test"]["mean_price"] > f["target"] + 1e-9]
    check("IAI vs exact (all depths, windows): price target met", not bad, str(bad or ""))
    ok = all(abs(-s3["rolling"][m]["main"]["Optimal policy tree"]["emissions_change_pct"]
                 - f["by_depth"]["exact"][str(f["chosen"]["exact"]["depth"])]["test"]["emissions_reduction_pct"]) < 1e-9
             for m, f in i7["folds"].items())
    check("Exact tree results identical in Step 3 rolling and 07_iai_opt", ok)
    pc = load("results/pacing/pacing_results.json")
    bad = [(m, n) for m, r in pc["folds"].items() for n, p in r["policies"].items() for tid in ("main", "matched")
           if p[tid]["mean_price"] > p[tid]["target"] + 1e-9]
    check("Pacing experiment: every policy x rule x target meets its price target", not bad, str(bad[:3] or ""))
    ok = all(abs(r["policies"][f"{s} tree (selected under original rule) | original rule"]["main"]["emissions_reduction_pct"]
                 - i7["folds"][m]["by_depth"][s][str(i7["folds"][m]["chosen"][s]["depth"])]["test"]["emissions_reduction_pct"]) < 1e-9
             for m, r in pc["folds"].items() for s in ("exact", "iai"))
    check("Pacing experiment reproduces 07's trees under the original rule", ok)
    af = load("results/affordability/affordability_results.json")
    bad = [(m, k) for m, r in af["folds"].items() for k, p in r["policies"].items()
           if p["main"]["cap"] is not None and not p["main"]["pred_within_cap"]]
    check("Affordability: predicted bill cap satisfied in every window and variant", not bad, str(bad or ""))
    bad = [(m, k) for m, r in af["folds"].items() for k, p in r["policies"].items() if p["main"]["mean_price"] > r["target"] + 1e-9]
    check("Affordability: announced-price target met in every window and variant", not bad, str(bad or ""))
    ok = all(abs(af["folds"][m]["policies"][f"{s}|unconstrained"]["main"]["emissions_reduction_pct"]
                 - pc["folds"][m]["policies"][f"{s} budget-aware tree | original rule"]["main"]["emissions_reduction_pct"]) < 1e-9
             for m in af["folds"] for s in ("iai", "exact"))
    check("Unconstrained budget-aware trees identical in pacing and affordability experiments", ok)
    import csv
    rows = list(csv.DictReader(open(ROOT / "results/tables/final_comparison.csv")))
    prim = [r for r in rows if "PRIMARY" in r["policy"]]
    ok = len(prim) == 4 and all(abs(float(r["reduction_pct"]) - af["folds"][m]["policies"]["iai|penalty + rule|5%"]["main"]["emissions_reduction_pct"]) < 1e-9
                                for r, m in zip(prim, ("5", "7", "9", "11")))
    check("Final summary table matches the affordability outputs for the primary policy", ok)


def main():
    test_rules()
    test_preservation()
    test_constraints_and_consistency()
    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
