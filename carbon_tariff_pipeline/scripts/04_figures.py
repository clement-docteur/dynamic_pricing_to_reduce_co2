"""All report figures, from the saved outputs of steps 1-3 (fast to re-run).

figures/step1_ci_forecast.png    : forecast error by model (validation and test) and one test week
figures/step1b_marginal_emissions.png: marginal emission factors by hour (with/without storage adjustment) and by year
figures/step2_demand_response.png: estimated in-block effects and the held-out check against observed effects
figures/step3_policy_results.png : emissions change vs the historical TOU (bootstrap 95% CI) and household bill changes
figures/step3_schedules.png      : price schedules over the test period (historical vs tree vs direct) + mean profiles
figures/step3_rolling.png        : emissions change in four rolling 2-month test windows
figures/step3_sensitivity.png    : emissions change under alternative price constraints and demand-response worlds
figures/pipeline.png             : schematic of the three-step pipeline (final report)
figures/data_overview.png        : cluster load profiles and 2013 tariff events by month (final report)
figures/step3_trees.png          : the selected policy tree and budget-aware tree, drawn as diagrams (final report)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import (AQUA, BLUE, DATES_2013, EDGE, FIG, GREEN, GRID, INK, INK2, MAGENTA, MUTED, ORANGE, OUT,  # noqa: E402
                        PRICE, SHADE, SPLIT, VIOLET, load_households, load_json, load_tariff_2013, mpl_style)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

from lib.policy import POLICIES as P, SHORT  # noqa: E402

R = load_json(OUT / "step3_results.json")
Z = np.load(OUT / "step3_test_schedules.npz")
COL = {P[1]: MUTED, P[2]: ORANGE, P[3]: BLUE, P[4]: AQUA, P[5]: VIOLET, P[6]: GREEN, P[7]: MAGENTA}
plt = mpl_style()


def fig_step1():
    m = load_json(OUT / "step1_metrics.json")
    z = np.load(OUT / "ci_forecasts_2013.npz")
    sel, lab = m["selected_model"], m["labels"]
    order = ["flat24h", "persist_7d", "persist_latest", "lasso", "ridge", "rf", "xgb"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.3), gridspec_kw={"width_ratios": [1, 1.55]})
    for i, k in enumerate(order):
        col = MUTED if k not in ("lasso", "ridge", "rf", "xgb") else (ORANGE if k == sel else BLUE)
        v, t = m["metrics"][k]["val"]["MAE"], m["metrics"][k]["test"]["MAE"]
        ax[0].hlines(i, min(v, t), max(v, t), color=GRID, lw=2)
        ax[0].plot(v, i, "o", ms=7, mfc="white", mec=col, mew=1.6, label="validation, Sep-Oct" if i == 6 else None)
        ax[0].plot(t, i, "o", ms=7, color=col, label="test, Nov-Dec" if i == 6 else None)
        ax[0].text(max(v, t) + 0.4, i, f"{t:.1f}", va="center", fontsize=8, color=INK2)
    ax[0].set_yticks(range(len(order)), [lab[k] for k in order]); ax[0].grid(axis="y", visible=False)
    ax[0].set(xlabel="MAE (gCO2/kWh); label = test MAE", xlim=(20, 40))
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title("a  Next-day CI forecast error", color=INK)
    d13 = pd.to_datetime(z["dates"])
    i0 = int(np.where(d13 == pd.Timestamp("2013-11-18"))[0][0])
    wk = slice(i0, i0 + 7)
    t = np.arange(7 * 48) / 48
    ax[1].plot(t, z["actual"][wk].ravel(), color=INK, lw=1.4, label="actual")
    ax[1].plot(t, z["selected"][wk].ravel(), color=ORANGE, lw=1.4, label=f"{lab[sel]}, issued noon the day before")
    ax[1].plot(t, z["persist_latest"][wk].ravel(), color=MUTED, lw=1, ls="--", label="same slot, latest day")
    ax[1].set_xticks(np.arange(7) + 0.5, [d.strftime("%a %d") for d in d13[wk]])
    ax[1].set(ylabel="gCO2/kWh", xlabel="18-24 November 2013")
    ax[1].legend(loc="upper left", fontsize=8, ncol=3)
    yl = ax[1].get_ylim(); ax[1].set_ylim(yl[0], yl[1] + 0.12 * (yl[1] - yl[0]))
    ax[1].set_title("b  One test week", color=INK)
    plt.tight_layout()
    fig.savefig(FIG / "step1_ci_forecast.png", bbox_inches="tight")


def fig_step1b():
    m = load_json(OUT / "step1b_mef.json")
    hr, h = m["hourly_2013"], np.array(m["hours_utc"]) + 0.5
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    for key, col, lab in (("adjusted", ORANGE, "marginal, storage-adjusted (main)"), ("unadjusted", BLUE, "marginal, storage as zero-carbon")):
        v, se = np.array(hr[key]["Oct-Mar"]), np.array(hr[key]["se_Oct-Mar"])
        ax[0].fill_between(h, v - 2 * se, v + 2 * se, color=col, alpha=0.15, lw=0)
        ax[0].plot(h, v, "-o", ms=3, color=col, lw=1.6, label=lab)
    ax[0].plot(h, hr["average_ci_test_by_hour"], color=INK, lw=1.6, ls="--", label="average intensity, Nov-Dec 2013")
    ax[0].axvspan(16, 20, color=SHADE, lw=0, zorder=0)
    ax[0].text(18, 760, "evening peak", ha="center", color=INK2, fontsize=8)
    ax[0].set(xlim=(0, 24), ylim=(250, 800), xlabel="hour of day (UTC)", ylabel="g CO2/kWh")
    ax[0].set_xticks(range(0, 25, 3), [f"{x:02d}:00" for x in range(0, 25, 3)])
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].set_title("a  Emission factors by hour, Oct-Mar 2013", color=INK)
    yrs = list(m["annual"])
    ax[1].plot(yrs, [m["annual"][y]["mef_unadjusted"] for y in yrs], "-o", color=BLUE, ms=5, label="marginal (all hours, unadjusted)")
    ax[1].plot(yrs, [m["annual"][y]["average_ci"] for y in yrs], "--o", color=INK, ms=5, label="average intensity")
    ax[1].set(ylim=(400, 700), ylabel="g CO2/kWh")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].set_title("b  Marginal vs average, by year", color=INK)
    plt.tight_layout()
    fig.savefig(FIG / "step1b_marginal_emissions.png", bbox_inches="tight")


def fig_step2():
    r = load_json(OUT / "step2_results.json")
    eff, sizes, bands = r["effects_main_janoct"], r["cluster_sizes"], r["bands"]
    cols = [BLUE, ORANGE, AQUA, VIOLET]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.7), gridspec_kw={"width_ratios": [1.5, 1]})
    for c in range(4):
        e = eff[str(c)]
        for k, kind in enumerate(("High", "Low")):
            x = np.arange(5) + 6 * k + (c - 1.5) * 0.18
            ci = np.array([e[f"{kind}|{b}"]["ci95"] for b in bands])
            ax[0].vlines(x, ci[:, 0], ci[:, 1], color=cols[c], lw=1)
            ax[0].plot(x, [e[f"{kind}|{b}"]["pct"] for b in bands], "o", ms=4, color=cols[c],
                       label=f"cluster {c} ({sizes[c]} households)" if k == 0 else None)
    ax[0].axhline(0, color=EDGE, lw=1)
    ax[0].set_xticks(list(range(5)) + list(range(6, 11)), [b.split(" ")[1] for b in bands] * 2, fontsize=8)
    ax[0].set_ylim(-42, 62)
    ax[0].text(2, 55, "inside High blocks", ha="center", color=INK2)
    ax[0].text(8, -36, "inside Low blocks", ha="center", color=INK2)
    ax[0].set(ylabel="effect on consumption (%)", xlabel="time of day (GMT)")
    ax[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    ax[0].set_title("a  Estimated effects inside price blocks (Jan-Oct fit, 95% bootstrap CI)", color=INK)
    V = r["heldout_validation"]["Nov-Dec (test; fitted Jan-Oct)"]
    for k, kind in enumerate(("in-block High", "in-block Low")):
        for j, c in enumerate(["0", "1", "2", "3", "all"]):
            o, x, first = V[c][kind], j + 6.5 * k, (j, k) == (0, 0)
            ax[1].vlines(x - 0.22, *o["observed_ci95"], color=INK, lw=1)
            ax[1].plot(x - 0.22, o["observed_pct"], "o", ms=5, color=INK, label="observed (95% CI)" if first else None)
            ax[1].plot(x, o["main_pct"], "D", ms=5, color=BLUE, label="response model" if first else None)
            ax[1].plot(x + 0.22, o["cvae_pct"], "s", ms=5, color=ORANGE, label="CVAE ensemble" if first else None)
    ax[1].axhline(0, color=EDGE, lw=1)
    ax[1].set_xticks(list(range(5)) + list(np.arange(5) + 6.5), ["c0", "c1", "c2", "c3", "all"] * 2, fontsize=8)
    ax[1].set(ylabel="effect on consumption (%)", ylim=(-36, 52))
    ax[1].text(2, 44, "High blocks", ha="center", color=INK2); ax[1].text(8.5, 44, "Low blocks", ha="center", color=INK2)
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].set_title("b  Unseen Nov-Dec 2013: observed vs simulated", color=INK)
    plt.tight_layout()
    fig.savefig(FIG / "step2_demand_response.png", bbox_inches="tight")


def fig_results():
    res, ci = R["results"], R["bootstrap_ci_emissions_change_pct"]
    pols = P[1:]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.2))
    y = np.arange(len(pols))[::-1]
    for yi, p in zip(y, pols):
        v, r = res[p]["emissions_change_pct"], res[p]
        ax[0].hlines(yi, *ci[p], color=COL[p], lw=1.5)
        ax[0].plot(v, yi, "o", ms=7, color=COL[p])
        ax[0].text(ci[p][1] + 0.04, yi, f"{v:+.2f}%", fontsize=8, color=INK2, va="center")
        ax[1].hlines(yi, r["bill_change_p10_pct"], r["bill_change_p90_pct"], color=COL[p], lw=5, alpha=0.35)
        ax[1].plot(r["bill_change_median_pct"], yi, "|", ms=12, mew=2, color=COL[p])
        ax[1].plot(r["bill_change_mean_pct"], yi, "D", ms=5, color=INK)
        ax[1].text(r["bill_change_p90_pct"] + 0.5, yi - 0.08, f"mean {r['bill_change_mean_pct']:+.1f}%", fontsize=8, color=INK2)
    for a in ax:
        a.axvline(0, color=EDGE, lw=1)
        a.grid(axis="y", visible=False)
    ax[0].set_yticks(y, [SHORT[p] for p in pols])
    ax[0].set(xlabel="emissions change vs historical TOU, Nov-Dec 2013 (% of footprint, marginal factors)")
    ax[0].set_title("a  Emissions (bar = 95% bootstrap)", color=INK)
    xl = ax[0].get_xlim(); ax[0].set_xlim(xl[0], xl[1] + 0.2)
    ax[1].set_yticks(y, [""] * len(pols))
    ax[1].set(xlabel="household bill vs historical TOU, Nov-Dec 2013 (%)")
    ax[1].set_title("b  Household bills (band = 10th-90th pct, | median, ◆ mean)", color=INK)
    xl = ax[1].get_xlim(); ax[1].set_xlim(xl[0], xl[1] + 0.25 * (xl[1] - xl[0]))
    plt.tight_layout()
    fig.savefig(FIG / "step3_policy_results.png", bbox_inches="tight")


def fig_schedules():
    dates = pd.to_datetime(Z["dates"])
    show = [P[0], P[2], P[6]]
    fig = plt.figure(figsize=(11.5, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9], hspace=0.6, wspace=0.22)
    cmap = ListedColormap([BLUE, SHADE, ORANGE])
    ticks = [j for j, d in enumerate(dates) if d.day in (1, 15)]
    for i, p in enumerate(show):
        ax = fig.add_subplot(gs[0, i])
        codes = np.where(Z[f"{p}|high"] > 0, 2, np.where(Z[f"{p}|low"] > 0, 0, 1)).T     # (48, days): 0 Low, 1 Normal, 2 High
        ax.imshow(codes, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest",
                  extent=(-0.5, len(dates) - 0.5, 23.5, -0.5))          # slot h covers (h-1)/2 .. h/2 h GMT
        ax.set_ylim(24, -0.5)
        ax.set_yticks(range(0, 25, 6), [f"{h:02d}:00" for h in range(0, 25, 6)])
        ax.set_xticks(ticks, [dates[j].strftime("%d %b") for j in ticks], fontsize=8)
        ax.grid(False)
        r = R["results"][p]
        ax.set_title(f"{'abc'[i]}  {SHORT[p]}", color=INK, fontsize=9.5)
        ax.text(0, -0.13, f"{r['high_slots']} High / {r['low_slots']} Low half-hours, {r['mean_price']:.2f} p/kWh",
                transform=ax.transAxes, fontsize=8, color=INK2, va="top")
        if i == 0:
            ax.set_ylabel("time of day (GMT)")
    fig.legend(handles=[Patch(color=ORANGE, label=f"High {PRICE['H']:.2f} p/kWh"),
                        Patch(facecolor=SHADE, edgecolor=EDGE, label=f"Normal {PRICE['N']:.2f}"),
                        Patch(color=BLUE, label=f"Low {PRICE['L']:.2f}")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    x = (np.arange(48) - 0.5) / 2
    ax = fig.add_subplot(gs[1, :2])
    for p, col, ls in ((P[0], INK, "-"), (P[2], ORANGE, "-"), (P[6], GREEN, "--")):
        ax.plot(x, Z[f"{p}|profile"], ls, color=col, lw=1.6, label=SHORT[p])
    ax.set(xlim=(-0.5, 24), ylabel="kWh per household per half-hour", xlabel="time of day (GMT)")
    ax.set_xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("d  Mean daily consumption profile on the test days (simulated)", color=INK)
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(x, Z["factor"].mean(0), color=ORANGE, lw=1.6, label="marginal (2013 estimate)")
    ax.plot(x, Z["factor_forecast"].mean(0), color=ORANGE, lw=1.2, ls="--", label="marginal (decision-time)")
    ax.plot(x, Z["ci"].mean(0), color=INK, lw=1.4, label="average intensity")
    ax.set(xlim=(-0.5, 24), ylim=(250, None), ylabel="gCO2/kWh", xlabel="time of day (GMT)")
    ax.set_xticks(range(0, 25, 6), [f"{h:02d}:00" for h in range(0, 25, 6)])
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("e  Emission factors, test days", color=INK)
    fig.savefig(FIG / "step3_schedules.png", bbox_inches="tight")


def fig_rolling():
    roll = list(R["rolling"].values())
    pols = [P[1], P[2], P[6], P[5]]
    x = np.arange(len(roll))
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.4), sharey=True)
    for a, tid, title in ((ax[0], "main", "a  At each window's target (pre-window historical price)"),
                          (ax[1], "matched", "b  At the 2013 tariff's own price in each window")):
        for j, (p, mk) in enumerate(zip(pols, ["s", "o", "P", "^"])):
            a.plot(x + (j - 1.5) * 0.12, [f[tid][p]["emissions_change_pct"] for f in roll], mk, ms=7, color=COL[p],
                   label=SHORT[p] if tid == "main" else None)
        a.axhline(0, color=EDGE, lw=1)
        a.set_xticks(x, [f"{f['label']}\n({f['days']['train']} training days)" for f in roll], fontsize=8)
        a.grid(axis="x", visible=False)
        a.set_title(title, color=INK)
    ax[0].set_ylabel("emissions change vs historical TOU\n(% of footprint, marginal factors)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / "step3_rolling.png", bbox_inches="tight")


def fig_sensitivity():
    rows = [(v["label"], {p: x["emissions_change_pct"] for p, x in v["results"].items()})
            for v in R["sensitivity_constraints"].values()]
    rows += [(v["label"], v["retrained_here"]) for v in R["sensitivity_worlds"].values()]
    pols = [P[1], P[2], P[6], P[3], P[5]]
    fig, ax = plt.subplots(figsize=(11.5, 0.42 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for yi, (_, vals) in zip(y, rows):
        shown = [vals[p] for p in pols if p in vals]
        ax.hlines(yi, min(shown), max(shown), color=GRID, lw=1)
        for p, mk in zip(pols, ["s", "o", "P", "D", "^"]):
            if p in vals:
                ax.plot(vals[p], yi, mk, ms=6, color=COL[p], label=SHORT[p] if yi == y[0] else None)
    ax.axvline(0, color=EDGE, lw=1)
    ax.set_yticks(y, [label for label, _ in rows], fontsize=8); ax.grid(axis="y", visible=False)
    ax.axhline(y[len(R["sensitivity_constraints"]) - 1] - 0.5, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("emissions change vs historical TOU, Nov-Dec 2013 (% of footprint, marginal factors)")
    ax.legend(fontsize=8, loc="lower left", ncol=5, bbox_to_anchor=(0, 1.0))
    plt.tight_layout()
    fig.savefig(FIG / "step3_sensitivity.png", bbox_inches="tight")


def _box(ax, x, y, w, h, text, fc="white", ec=EDGE, fs=8.5, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.1, mutation_aspect=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color, weight=weight, linespacing=1.35)


def _arrow(ax, xy0, xy1, text=None, toff=(0, 0), rad=0.0):
    ax.annotate("", xy=xy1, xytext=xy0,
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1, shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((xy0[0] + xy1[0]) / 2 + toff[0], (xy0[1] + xy1[1]) / 2 + toff[1], text, ha="center", va="center",
                fontsize=7.5, color=INK2, style="italic")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.set(xlim=(-0.01, 1.01), ylim=(0, 1)); ax.axis("off")
    cx = {"data": 0.1, "models": 0.385, "tree": 0.685, "eval": 0.915}
    for k, lab in zip(cx, ("Data", "Models", "Prescription", "Evaluation")):
        ax.text(cx[k], 0.98, lab, ha="center", fontsize=9, color=INK2, weight="bold")
    fs = 8
    _box(ax, cx["data"], 0.75, 0.2, 0.2, "NESO grid data, 2009-13\nhalf-hourly carbon intensity\nand generation mix",
         fc=SHADE, fs=fs)
    _box(ax, cx["data"], 0.28, 0.2, 0.36, "Low Carbon London, 2013\n1,032 dynamic-tariff (ToU)\nhouseholds in 4 clusters\n"
         "1,032 flat-tariff (Std)\nhouseholds as a control\nERA5 temperature", fc=SHADE, fs=fs)
    _box(ax, cx["models"], 0.84, 0.26, 0.17, "Step 1: next-day carbon forecast\nRidge, Lasso, random forest,\n"
         "XGBoost vs persistence", fs=fs)
    _box(ax, cx["models"], 0.57, 0.26, 0.17, "Step 1b: marginal emission factors\nchanges in emissions regressed\n"
         "on changes in generation", fs=fs)
    _box(ax, cx["models"], 0.25, 0.26, 0.23, "Step 2: demand-response simulator\nhow each cluster's use changes\n"
         "under a price schedule\n(checked on held-out events)", fs=fs)
    _box(ax, cx["tree"], 0.55, 0.23, 0.5, "Step 3: optimal policy tree\n\nreads features known at noon\n"
         "the day before and picks one\nof 4 daily schedules\n\nprice cap: Lagrangian penalty\n+ feasibility rule",
         fc="#fdf0ea", ec=ORANGE, fs=fs)
    _box(ax, cx["eval"], 0.68, 0.17, 0.24, "Daily price schedule\nNormal, or a High/Low\nblock, announced\n"
         "12:00 the day before", fs=fs)
    _box(ax, cx["eval"], 0.25, 0.17, 0.27, "Test Nov-Dec 2013\nemissions (marginal),\nbills, price;\n"
         "rolling windows,\nsensitivity", fs=fs)
    _arrow(ax, (0.2, 0.79), (0.255, 0.84))
    _arrow(ax, (0.2, 0.71), (0.255, 0.59))
    _arrow(ax, (0.2, 0.26), (0.255, 0.26))
    _arrow(ax, (0.515, 0.84), (0.57, 0.74), "forecast\nfeatures", (0.0, 0.115))
    _arrow(ax, (0.515, 0.57), (0.57, 0.57), "emission\nfactor", (0.0, 0.1))
    _arrow(ax, (0.515, 0.27), (0.57, 0.36), "change in\nconsumption", (0.0, -0.11))
    _arrow(ax, (0.8, 0.68), (0.83, 0.68))
    _arrow(ax, (cx["eval"], 0.56), (cx["eval"], 0.385))
    fig.savefig(FIG / "pipeline.png", bbox_inches="tight")


def fig_data():
    hh = load_households(std=False)
    low, high, _ = load_tariff_2013()
    normal = (low.sum(1) == 0) & (high.sum(1) == 0)
    cl, cols = hh["cluster"], [BLUE, ORANGE, AQUA, VIOLET]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.2), gridspec_kw={"width_ratios": [1, 1.2]})
    x = (np.arange(48) - 0.5) / 2
    for c in range(4):
        prof = hh["Y_tou"][cl == c][:, normal].mean((0, 1))
        ax[0].plot(x, prof, color=cols[c], lw=1.6, label=f"cluster {c} ({(cl == c).sum()} households)")
    ax[0].set(xlim=(-0.5, 24), ylabel="kWh per household per half-hour", xlabel="time of day (GMT)")
    ax[0].set_xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title(f"a  Mean daily profile by cluster ({normal.sum()} days without price events)", color=INK)
    month = DATES_2013.month
    nh = np.array([(high[month == m].sum(1) > 0).sum() for m in range(1, 13)])
    nl = np.array([(low[month == m].sum(1) > 0).sum() for m in range(1, 13)])
    m = np.arange(1, 13)
    ax[1].bar(m - 0.2, nh, 0.38, color=ORANGE, label=f"days with a High block ({PRICE['H']:.2f} p/kWh)")
    ax[1].bar(m + 0.2, nl, 0.38, color=BLUE, label=f"days with a Low block ({PRICE['L']:.2f} p/kWh)")
    top = 2 * max(nh.max(), nl.max()) + 1
    for k, (a, b) in SPLIT.items():
        a, b = pd.Timestamp(a).month - 0.5, pd.Timestamp(b).month + 0.5
        if k != "train":
            ax[1].axvspan(a, b, color=SHADE, lw=0, zorder=0)
        ax[1].text((a + b) / 2, top - 1.2, {"train": "training", "val": "validation", "test": "test"}[k],
                   ha="center", va="top", fontsize=8, color=INK2)
    ax[1].set(xlim=(0.5, 12.5), ylim=(0, top), ylabel="days")
    ax[1].set_xticks(m, [pd.Timestamp(2013, i, 1).strftime("%b") for i in m])
    ax[1].set_yticks(range(0, int(top), 4))
    ax[1].grid(axis="x", visible=False)
    ax[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.88))
    ax[1].set_title("b  Price events in the 2013 tariff, by month", color=INK)
    plt.tight_layout()
    fig.savefig(FIG / "data_overview.png", bbox_inches="tight")


def fig_trees():
    menu = [a["name"] for a in R["menu"]]
    fill = {"Normal": (SHADE, EDGE), "High": ("#fbe1d6", ORANGE), "Low": ("#d9e7f8", BLUE)}

    def leaf(ax, x, y, node, show_n):
        name = menu[node["action"]]
        fc, ec = fill[name.split(" ")[0]]
        txt = name.replace(" all day", "\nall day").replace("High ", "High\n").replace("Low ", "Low\n")
        if show_n:
            txt += f"\n({node['n']} days)"
        _box(ax, x, y, 0.2, 0.24, txt, fc=fc, ec=ec, fs=8.5)

    def split(ax, x, y, label, w=0.34):
        _box(ax, x, y, w, 0.17, label, fs=8.5, weight="bold")

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.4), gridspec_kw={"width_ratios": [0.75, 1.25]})
    for a in ax:
        a.set(xlim=(0, 1), ylim=(0, 1)); a.axis("off")
    t = R["tree"]
    split(ax[0], 0.5, 0.78, f"recent consumption\n$\\leq$ {t['threshold']:.3f} kWh per half-hour?", w=0.6)
    leaf(ax[0], 0.25, 0.25, t["left"], True); leaf(ax[0], 0.75, 0.25, t["right"], True)
    _arrow(ax[0], (0.4, 0.69), (0.27, 0.38), "yes", (-0.06, 0)); _arrow(ax[0], (0.6, 0.69), (0.73, 0.38), "no", (0.06, 0))
    ax[0].set_title("a  Optimal policy tree (depth 1, Nov-Dec test)", color=INK, loc="left")
    p = R["paced_tree"]
    split(ax[1], 0.5, 0.86, f"recent consumption $\\leq$ {p['threshold']:.3f} kWh per half-hour?", w=0.56)
    for side, x0 in (("left", 0.26), ("right", 0.74)):
        n = p[side]
        split(ax[1], x0, 0.55, f"remaining budget\n$\\leq$ {n['threshold']:g} p/kWh per day?", w=0.3)
        leaf(ax[1], x0 - 0.12, 0.15, n["left"], False); leaf(ax[1], x0 + 0.12, 0.15, n["right"], False)
        _arrow(ax[1], (x0 - 0.06, 0.465), (x0 - 0.11, 0.27), "yes", (-0.035, 0))
        _arrow(ax[1], (x0 + 0.06, 0.465), (x0 + 0.11, 0.27), "no", (0.035, 0))
    _arrow(ax[1], (0.4, 0.775), (0.3, 0.64), "yes", (-0.04, 0.0)); _arrow(ax[1], (0.6, 0.775), (0.7, 0.64), "no", (0.04, 0.0))
    ax[1].set_title("b  Budget-aware tree (depth 2)", color=INK, loc="left")
    plt.tight_layout()
    fig.savefig(FIG / "step3_trees.png", bbox_inches="tight")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    for f in (fig_step1, fig_step1b, fig_step2, fig_results, fig_schedules, fig_rolling, fig_sensitivity,
              fig_pipeline, fig_data, fig_trees):
        if not only or f.__name__ in only:
            f()
    print("figures written")
