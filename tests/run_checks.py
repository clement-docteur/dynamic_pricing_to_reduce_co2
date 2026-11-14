"""Automated checks for the MVP (no extra dependencies):  .venv/bin/python tests/run_checks.py

1. Unit tests of the feasibility rules (carbon_tariff_pipeline/lib/policy.py) and of the raw-data file lookup.
2. Preservation (three reference points):
   a. ORIGINAL_COMMIT - Clement's original version, the baseline of the MVP audit: every value in his original committed
      outputs is still present and identical (additions such as the net-load factor are allowed and listed); his
      figures, report and CVAE artefacts are byte-identical.
   b. CLEMENT_MAIN - Clement's main merged into this branch: his latest outputs, report, demand_response/ module and
      scripts are untouched by the MVP branch (except the MVP's opt-in flags in 03_policy.py and the raw-data path
      lookup in lcl_data.py / 03_conditions.py), and his new net-load outputs are valid and reproduce exactly.
   c. PREMERGE_MVP - the last MVP commit before the merge: every pre-merge MVP result value is unchanged.
3. Reproduction: Clement's Steps 1b, 2 and 3 re-run exactly (results/reproduction/reproduction_check.json).
4. MVP sanity / leakage checks all pass (results/tables/sanity_checks.json).
5. Constraint compliance in every saved experiment (announced-price target; predicted bill cap).
6. Cross-file consistency of the numbers reused across experiments and in the final summary table.
Requires the outputs of `run_mvp.py --reproduce` (all stages). Exit code 0 = all checks passed.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "carbon_tariff_pipeline"))
sys.path.insert(0, str(ROOT / "CVAE_reproduction"))
import numpy as np  # noqa: E402

from lib import common as COMMON, lcl_data as LCL  # noqa: E402
from lib.policy import NORMAL, feasibility_rule, multi_budget_rule, pacing_rule  # noqa: E402

ORIGINAL_COMMIT = "6c22f28"     # Clement's original version (audit and reproduction baseline)
CLEMENT_MAIN = "0a2f9af"        # Clement's main merged into this branch (net-load MEF, demand_response/, fresh-clone fix)
PREMERGE_MVP = "6c55a54"        # last MVP commit before merging Clement's main
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)


def git_show(rev, path):
    r = git("show", f"{rev}:{path}")
    return r.stdout if r.returncode == 0 else None


def git_files(rev, *paths):
    return [l for l in git("ls-tree", "-r", "--name-only", rev, "--", *paths).stdout.decode().splitlines() if l]


# ------------------------------------------------------------------------------------------------ value comparison
def _walk(old, new, path, out):
    if isinstance(old, dict):
        if not isinstance(new, dict):
            out["changed"].append(path); return
        for k in old:
            if k not in new:
                out["removed"].append(f"{path}/{k}")
            else:
                _walk(old[k], new[k], f"{path}/{k}", out)
        out["added"] += [f"{path}/{k}" for k in new if k not in old]
    elif isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            out["changed"].append(path); return
        for i, (a, b) in enumerate(zip(old, new)):
            _walk(a, b, f"{path}[{i}]", out)
    elif isinstance(old, (int, float)) and not isinstance(old, bool) and isinstance(new, (int, float)):
        out["values"] += 1
        if not ((np.isnan(old) and np.isnan(new)) or abs(old - new) <= 1e-9 * max(1.0, abs(old))):
            out["changed"].append(path)
    else:
        out["values"] += 1
        if old != new:
            out["changed"].append(path)


def compare_values(old_bytes, new_path):
    """Every value of the old file is present and identical in the new file (additions allowed).
    Returns dict(values, changed, removed, added)."""
    out = {"values": 0, "changed": [], "removed": [], "added": []}
    p = ROOT / new_path
    if not p.exists():
        out["removed"].append(new_path); return out
    suf = p.suffix
    if suf == ".json":
        _walk(json.loads(old_bytes), json.loads(p.read_text(encoding="utf-8")), "", out)
    elif suf == ".npz":
        a, b = np.load(io.BytesIO(old_bytes), allow_pickle=True), np.load(p, allow_pickle=True)
        out["added"] += [k for k in b.files if k not in a.files]
        for k in a.files:
            out["values"] += a[k].size
            if k not in b.files:
                out["removed"].append(k)
            elif a[k].shape != b[k].shape or not (np.array_equal(a[k], b[k], equal_nan=True) if a[k].dtype.kind in "fc" else np.array_equal(a[k], b[k])):
                out["changed"].append(k)
    elif suf == ".csv":
        ra = list(csv.DictReader(io.StringIO(old_bytes.decode("utf-8"))))
        rb = list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8"))))
        if len(ra) != len(rb):
            out["changed"].append(f"{new_path}: {len(ra)} -> {len(rb)} rows"); return out
        out["added"] += [c for c in (rb[0] if rb else {}) if c not in (ra[0] if ra else {})]
        for i, (x, y) in enumerate(zip(ra, rb)):
            for c, v in x.items():
                out["values"] += 1
                if c not in y:
                    out["removed"].append(c); continue
                try:
                    same = abs(float(v) - float(y[c])) <= 1e-9 * max(1.0, abs(float(v))) or (np.isnan(float(v)) and np.isnan(float(y[c])))
                except ValueError:
                    same = v == y[c]
                if not same:
                    out["changed"].append(f"row {i} {c}")
    else:
        out["values"] += 1
        if old_bytes != p.read_bytes():
            out["changed"].append(new_path)
    return out


def preserved(rev, paths, label, exclude=(), suffixes=None):
    files = [f for f in git_files(rev, *paths) if Path(f).name not in exclude and (suffixes is None or Path(f).suffix in suffixes)]
    tot = {"values": 0, "changed": [], "removed": [], "added": []}
    for f in files:
        r = compare_values(git_show(rev, f), f)
        tot["values"] += r["values"]
        tot["changed"] += [f"{f}:{c}" for c in r["changed"]][:5]
        tot["removed"] += [f"{f}:{c}" for c in r["removed"]][:5]
        tot["added"] += [f"{Path(f).name}:{c}" for c in r["added"]]
    ok = files and not tot["changed"] and not tot["removed"]
    added = sorted({a.split("[")[0] for a in tot["added"]})
    check(label, ok, f"{len(files)} files, {tot['values']:,} values compared; changed {tot['changed'][:3]}; removed {tot['removed'][:3]}; "
                     f"additions: {len(tot['added'])} ({', '.join(added[:6])}{' ...' if len(added) > 6 else ''})")


# ------------------------------------------------------------------------------------------------ 1. unit tests
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


def test_raw_lookup():
    """The three raw-data lookups (lib/common, lcl_data, run_mvp) prefer $CARBON_RAW_DIR, then the repository root
    (Clement's layout), then the parent folder (Panle's layout), and accept the LCL_Data/ subfolder."""
    spec = importlib.util.spec_from_file_location("run_mvp", ROOT / "run_mvp.py")
    RUN = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RUN)
    ok = True
    with tempfile.TemporaryDirectory() as t:
        env, root, parent = (Path(t) / x for x in ("env", "root", "parent"))
        for d in (env, root, parent, parent / "LCL_Data"):
            d.mkdir(parents=True)
        (root / "df_fuel_ckan.csv").touch(); (parent / "df_fuel_ckan.csv").touch(); (parent / "Tariffs.xlsx").touch()
        (parent / "LCL_Data" / "CC_LCL-FullData.csv").touch(); (env / "Tariffs.xlsx").touch()
        for mod, attr in ((COMMON, "RAW_DIRS"), (LCL, "RAW_DIRS"), (RUN, "RAW_DIRS")):
            old = getattr(mod, attr)
            try:
                setattr(mod, attr, [root, parent]); f = mod.raw_file
                ok &= f("df_fuel_ckan.csv") == root / "df_fuel_ckan.csv"                     # repository root first
                ok &= f("Tariffs.xlsx") == parent / "Tariffs.xlsx"                           # then the parent folder
                ok &= f("CC_LCL-FullData.csv", "LCL_Data/CC_LCL-FullData.csv") == parent / "LCL_Data" / "CC_LCL-FullData.csv"
                ok &= f("missing.csv") == root / "missing.csv"                               # clear error path
                setattr(mod, attr, [env, root, parent])
                ok &= f("Tariffs.xlsx") == env / "Tariffs.xlsx"                              # $CARBON_RAW_DIR wins
            finally:
                setattr(mod, attr, old)
    found = {k: str(COMMON.raw_file(n).relative_to(ROOT.parent)) for k, n in (("NESO", "df_fuel_ckan.csv"), ("tariffs", "Tariffs.xlsx"))}
    check("raw-data lookup ($CARBON_RAW_DIR > repo root > parent; LCL_Data/ accepted) in lib/common, lcl_data, run_mvp", ok,
          f"this machine: {found}, LCL {LCL.RAW_CSV.relative_to(ROOT.parent)}")


# ------------------------------------------------------------------------------------------------ 2-4. preservation, reproduction
def test_preservation():
    preserved(ORIGINAL_COMMIT, ["carbon_tariff_pipeline/outputs", "carbon_tariff_pipeline/figures", "carbon_tariff_pipeline/report.html",
                                "CVAE_reproduction/outputs", "CVAE_reproduction/data"],
              f"a. Clement's ORIGINAL results ({ORIGINAL_COMMIT}): every value preserved (additions allowed)")
    r = git("diff", "--quiet", CLEMENT_MAIN, "--", "carbon_tariff_pipeline/outputs", "carbon_tariff_pipeline/figures",
            "carbon_tariff_pipeline/report.html", "carbon_tariff_pipeline/README.md", "CVAE_reproduction/outputs", "CVAE_reproduction/data",
            "CVAE_reproduction/REPRO_SPEC.md", "demand_response", "final_report/main.tex", "final_report/final_report_overleaf.zip")
    check(f"b. Clement's LATEST outputs, report, docs and demand_response/ ({CLEMENT_MAIN}) untouched by the MVP branch", r.returncode == 0)
    r = git("diff", "--quiet", CLEMENT_MAIN, "--", "carbon_tariff_pipeline/scripts/0*.py", "carbon_tariff_pipeline/lib/demand.py",
            "CVAE_reproduction/scripts", ":(exclude)carbon_tariff_pipeline/scripts/03_policy.py", ":(exclude)carbon_tariff_pipeline/scripts/07_*",
            ":(exclude)carbon_tariff_pipeline/scripts/08_*", ":(exclude)carbon_tariff_pipeline/scripts/09_*",
            ":(exclude)CVAE_reproduction/scripts/03_conditions.py")
    pm = lambda a, b, f: sorted(l for l in git("diff", "--ignore-cr-at-eol", "-U0", a, *([b] if b else []), "--", f).stdout.decode().splitlines()
                               if l[:1] in "+-" and not l.startswith(("+++", "---")))
    same03 = pm(ORIGINAL_COMMIT, PREMERGE_MVP, "carbon_tariff_pipeline/scripts/03_policy.py") == pm(CLEMENT_MAIN, None, "carbon_tariff_pipeline/scripts/03_policy.py")
    check(f"b. Clement's scripts identical to {CLEMENT_MAIN} except the MVP's opt-in flags in 03_policy.py (same diff as before the merge) "
          "and raw-data paths in 03_conditions.py", r.returncode == 0 and same03)
    mef = np.load(ROOT / "carbon_tariff_pipeline/outputs/mef_2013.npz")
    nl = mef["realised_netload"]
    check("b. Clement's net-load MEF (committed): (365, 48), finite, plausible g/kWh",
          nl.shape == (365, 48) and np.isfinite(nl).all() and 0 < nl.mean() < 2000,
          f"mean {nl.mean():.0f}, range {nl.min():.0f}-{nl.max():.0f} g/kWh (realised hour-based MEF mean {mef['realised'].mean():.0f})")
    rc = load("results/reproduction/reproduction_check.json")["files"]
    s1b = rc.get("results/reproduction/step1_mac/step1b_mef.json", {})
    m = rc.get("results/reproduction/step1_mac/mef_2013.npz", {}).get("max_abs_diff_per_array", {})
    check("b. Clement's Step 1b incl. the net-load factor reproduces exactly on this machine",
          isinstance(s1b, dict) and s1b.get("mismatches") == 0 and not s1b.get("keys_only_in_committed") and m
          and all(isinstance(v, float) and v == 0.0 for v in m.values()) and "realised_netload" in m,
          f"step1b_mef.json {s1b.get('values')} numbers, 0 mismatches; mef arrays {sorted(m)}")
    preserved(PREMERGE_MVP, ["results/mvp/outputs", "results/pacing", "results/affordability", "results/tables",
                             "results/reproduction/outputs", "results/reproduction/step1_mac"],
              f"c. MVP results before the merge ({PREMERGE_MVP}): every value preserved (additions allowed)",
              exclude=("sanity_checks.json",),                     # descriptive text; its pass/fail is checked below
              suffixes=(".json", ".npz", ".csv"))                  # data files; the .md tables render these CSVs
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
    rows = list(csv.DictReader(open(ROOT / "results/tables/final_comparison.csv")))
    prim = [r for r in rows if "PRIMARY" in r["policy"]]
    ok = len(prim) == 4 and all(abs(float(r["reduction_pct"]) - af["folds"][m]["policies"]["iai|penalty + rule|5%"]["main"]["emissions_reduction_pct"]) < 1e-9
                                for r, m in zip(prim, ("5", "7", "9", "11")))
    check("Final summary table matches the affordability outputs for the primary policy", ok)


def main():
    test_rules()
    test_raw_lookup()
    test_preservation()
    test_constraints_and_consistency()
    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
