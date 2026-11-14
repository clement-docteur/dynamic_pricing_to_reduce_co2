"""Final cross-window comparison, assembled ONLY from saved outputs (no fitting, no optimisation).

Reads:  results/mvp/outputs/step3_results_average_mvp.json (Step 3 rolling windows: historical TOU, carbon heuristic,
        exact policy tree, direct optimisation, oracle), results/mvp/outputs/iai_opt_results_average_mvp.json (IAI OPT),
        results/affordability/affordability_results.json (budget-aware IAI tree, unconstrained and with bill caps).
Writes: results/tables/final_comparison.{csv,md}, results/figures/fig8_final_comparison.png

All policies use the deployable price target (historical TOU average announced price over each window's train+val days)
and the primary metric (realised average carbon intensity x simulated demand). Bill change = mean simulated household
bill vs the historical TOU in the same test window. Model-based estimates, not causal effects.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import BLUE, GREEN, INK, MUTED, ORANGE, PROJECT, VIOLET, load_json, mpl_style  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.policy import POLICIES  # noqa: E402

RES = PROJECT / "results"
MVP = RES / "mvp" / "outputs"
S3 = load_json(MVP / "step3_results_average_mvp.json")
I7 = load_json(MVP / "iai_opt_results_average_mvp.json")
AF = load_json(RES / "affordability" / "affordability_results.json")
WINDOWS = ["5", "7", "9", "11"]
PRIMARY = "Budget-aware IAI OPT, bill cap +5 % (penalty + rule) [PRIMARY]"
ORDER = ["Historical TOU", "Carbon heuristic", "Policy tree, exact solver (Clement)", "IAI OPT", "Direct optimisation",
         "Budget-aware IAI OPT (no bill cap)", PRIMARY, "Oracle (hindsight; reference only)"]


def rows():
    out = []
    for mm in WINDOWS:
        roll, f7, af = S3["rolling"][mm], I7["folds"][mm], AF["folds"][mm]
        lab = roll["label"]
        assert lab == f7["label"] == af["label"]
        base = {"window": lab, "price_target": roll["target"], "historical_avg_price": roll["historical_test_price"],
                "bill_cap_ref_price_paid": af["pi_ref_trainval"], "historical_price_paid_test": af["pi_ref_test_historical"]}
        m = roll["main"]
        add = lambda name, red, bill, price, ovr, kind: out.append({**base, "policy": name, "kind": kind, "reduction_pct": red,
                                                                   "bill_change_pct": bill, "avg_announced_price": price,
                                                                   "overrides": ovr})
        add("Historical TOU", 0.0, 0.0, m[POLICIES[0]]["mean_price"], "-", "benchmark (observed schedule)")
        add("Carbon heuristic", -m[POLICIES[1]]["emissions_change_pct"], m[POLICIES[1]]["bill_change_mean_pct"],
            m[POLICIES[1]]["mean_price"], 0, "baseline (no demand model)")
        add("Policy tree, exact solver (Clement)", -m[POLICIES[2]]["emissions_change_pct"], m[POLICIES[2]]["bill_change_mean_pct"],
            m[POLICIES[2]]["mean_price"], roll["tree_adjustments"], "reproduced method (leak-free features)")
        ch = f7["chosen"]["iai"]["depth"]
        t = f7["by_depth"]["iai"][str(ch)]["test"]
        add("IAI OPT", t["emissions_reduction_pct"], t["bill_change_mean_pct"], t["mean_price"], t["overrides"], "new")
        add("Direct optimisation", -m[POLICIES[3]]["emissions_change_pct"], m[POLICIES[3]]["bill_change_mean_pct"],
            m[POLICIES[3]]["mean_price"], "-", "benchmark (no interpretability constraint)")
        e = af["policies"]["iai|unconstrained"]["main"]
        add("Budget-aware IAI OPT (no bill cap)", e["emissions_reduction_pct"], e["bill_change_mean_pct"], e["mean_price"], e["overrides"], "new")
        p = af["policies"]["iai|penalty + rule|5%"]
        e = p["main"]
        add(PRIMARY, e["emissions_reduction_pct"], e["bill_change_mean_pct"], e["mean_price"], e["overrides"], "new (primary candidate)")
        out[-1].update(pred_price_paid=e["pred_daily_paid_mean"], sim_price_paid=e["sim_daily_paid_mean"], cap=e["cap"],
                       pred_within_cap=e["pred_within_cap"], sim_within_cap=e["sim_within_cap"],
                       bill_within_5pct_of_historical=e["bill_within_x_vs_historical"], depth=p["selection"]["depth"],
                       mu=p["selection"]["mu"], leaves=e["leaves"], bill_p10=e["bill_change_p10_pct"], bill_p90=e["bill_change_p90_pct"])
        add("Oracle (hindsight; reference only)", -m[POLICIES[5]]["emissions_change_pct"], m[POLICIES[5]]["bill_change_mean_pct"],
            m[POLICIES[5]]["mean_price"], "-", "hindsight reference")
    return pd.DataFrame(out)


def main():
    df = rows()
    df.to_csv(RES / "tables" / "final_comparison.csv", index=False)
    piv = lambda col: df.pivot_table(index="policy", columns="window", values=col, sort=False).reindex(ORDER)[["May-Jun", "Jul-Aug", "Sep-Oct", "Nov-Dec"]]
    red, bill, price = piv("reduction_pct"), piv("bill_change_pct"), piv("avg_announced_price")
    red["mean"], bill["mean"], price["mean"] = red.mean(axis=1), bill.mean(axis=1), price.mean(axis=1)
    tgt = df.drop_duplicates("window").set_index("window")
    fmt = lambda d: "| policy | " + " | ".join(d.columns) + " |\n|---|" + "---|" * len(d.columns) + "\n" + "\n".join(
        f"| {i} | " + " | ".join(f"{v:+.3f}" if isinstance(v, float) else str(v) for v in r) + " |" for i, r in d.iterrows()) + "\n"
    prim = df[df.policy == PRIMARY].set_index("window")
    txt = ["# Final comparison across the four rolling test windows (generated by scripts/12_final_summary.py)", "",
           "Assembled from saved outputs only. Deployable price target per window; primary metric = realised average carbon "
           "intensity x simulated demand; bill change vs the historical TOU in the same window. Model-based estimates.", "",
           "| window | price target | historical avg announced price | bill-cap reference price paid (train+val) | historical price paid (test window) |",
           "|---|---|---|---|---|"] + [f"| {w} | {r.price_target:.3f} | {r.historical_avg_price:.3f} | {r.bill_cap_ref_price_paid:.3f} | "
                                       f"{r.historical_price_paid_test:.3f} |" for w, r in tgt.iterrows()] + [
           "", "## Emissions reduction vs historical TOU (%)", "", fmt(red),
           "## Mean household bill change vs historical TOU (%)", "", fmt(bill),
           "## Average announced price (p/kWh)", "", fmt(price),
           "## Primary candidate: constraint checks per window", "",
           "| window | cap (p/kWh) | predicted price paid | simulated price paid | predicted within cap | simulated within cap | "
           "bill within +5 % of historical TOU | overrides | depth / leaves | mu | bill p10 / p90 (%) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"] + [
           f"| {w} | {r.cap:.3f} | {r.pred_price_paid:.3f} | {r.sim_price_paid:.3f} | {r.pred_within_cap} | {r.sim_within_cap} | "
           f"{r.bill_within_5pct_of_historical} | {r.overrides} | {int(r.depth)} / {int(r.leaves)} | {r.mu} | {r.bill_p10:+.1f} / {r.bill_p90:+.1f} |"
           for w, r in prim.iterrows()]
    (RES / "tables" / "final_comparison.md").write_text("\n".join(txt) + "\n", encoding="utf-8")

    plt = mpl_style()
    show = ORDER[1:]
    cols = [ORANGE, MUTED, VIOLET, INK, BLUE, GREEN, "#bdbdbd"]
    wins = ["May-Jun", "Jul-Aug", "Sep-Oct", "Nov-Dec", "mean"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.arange(len(wins))
    w = 0.8 / len(show)
    for ax, tab, lab in ((axes[0], red, "Emissions reduction vs historical TOU (%)"), (axes[1], bill, "Mean household bill change vs historical TOU (%)")):
        for j, (p, c) in enumerate(zip(show, cols)):
            ax.bar(x + (j - len(show) / 2 + 0.5) * w, tab.loc[p, wins].to_numpy(float), w, color=c, label=p,
                   edgecolor=INK if p == PRIMARY else "none", lw=1.2 if p == PRIMARY else 0)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_xticks(x, wins)
        ax.set_ylabel(lab)
    axes[0].set_title("(a) Estimated emissions reduction (model-based)")
    axes[1].set_title("(b) Household bills (bill cap is vs a train+val reference price, not this window's TOU)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(RES / "figures" / "fig8_final_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("\n".join(txt))


if __name__ == "__main__":
    main()
