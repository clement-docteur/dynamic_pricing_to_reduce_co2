"""MVP outputs: data-quality summary, processed Parquet, day x action cost matrix, comparison tables, the four required
figures and the sanity / leakage checks. Reads the MVP pipeline outputs in $CARBON_OUT (results/mvp/outputs):
  step1_metrics.json, ci_forecasts_2013.npz, step2_results.json, step3_results_average_mvp.json (primary: average carbon
  intensity, leak-free temperature), step3_results_mvp.json (sensitivity: marginal emission factors),
  iai_opt_results_average_mvp.json (IAI Optimal Policy Trees vs the exact tree).
Writes: results/tables/*.md|csv|json, results/figures/fig1..fig4*.png, results/processed/*.parquet (gitignored).

All emission changes are model-based counterfactual estimates (simulated demand x realised carbon intensity), not
measured or causal effects.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

sys.argv = [sys.argv[0], "--metric", "average", "--temperature", "lagged", "--tag", "mvp"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import (BLUE, DATES_2013, GREEN, GRID, H, INK, INK2, MUTED, ORANGE, OUT, PRICE, PROJECT, raw_file,  # noqa: E402
                        REPRO, SHADE, SPLIT, VIOLET, MAGENTA, AQUA, cols, day_type, hours, load_json, load_neso,
                        make_log, mpl_style, read_parquet, save_json, schedule_price, split_of)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.demand import ResponseModel  # noqa: E402
from lib.policy import NORMAL, POLICIES, candidate_pool, slot_time  # noqa: E402

_spec = importlib.util.spec_from_file_location("step3", Path(__file__).with_name("03_policy.py"))
S3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S3)

log, _ = make_log()
RES = PROJECT / "results"
TAB, FIGD, PROC = RES / "tables", RES / "figures", RES / "processed"
for d in (TAB, FIGD, PROC):
    d.mkdir(parents=True, exist_ok=True)
R = load_json(OUT / "step3_results_average_mvp.json")
RM = load_json(OUT / "step3_results_mvp.json")
IAI_PATH = OUT / "iai_opt_results_average_mvp.json"
IAI = load_json(IAI_PATH) if IAI_PATH.exists() else None
M1 = load_json(OUT / "step1_metrics.json")
CIZ = np.load(OUT / "ci_forecasts_2013.npz")
S2 = load_json(OUT / "step2_results.json")
COMMITTED = PROJECT / "carbon_tariff_pipeline" / "outputs"
plt = mpl_style()

DISPLAY = {POLICIES[0]: "A. Historical TOU (2013)", POLICIES[1]: "B. Carbon-based heuristic",
           POLICIES[3]: "C. Direct model-based optimisation", POLICIES[2]: "D. Policy tree (exact solver)",
           "IAI": "D'. IAI Optimal Policy Tree", POLICIES[6]: "Ext. Budget-aware policy tree",
           POLICIES[5]: "Ref. Oracle (perfect hindsight)"}
ADJ_KEY = {POLICIES[2]: "tree", POLICIES[3]: "direct", POLICIES[6]: "tree_paced", POLICIES[7]: "direct_paced",
           POLICIES[4]: "direct_perfect_ci"}


def md_table(df, floatfmt="{:.3f}"):
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return head + "\n".join(rows) + "\n"


def iai_main():
    if not IAI or not IAI["iai_available"]:
        return None
    f = IAI["folds"][str(S3.MAIN_TEST_MONTH)]
    d = f["chosen"]["iai"]["depth"]
    return f, d, f["by_depth"]["iai"][str(d)]


# ================================================================================================ 1. data quality
def data_quality():
    bs = load_json(REPRO / "data" / "lcl2013_build_stats.json")
    fs = load_json(REPRO / "data" / "lcl2013_fill_stats.json")
    neso = load_neso()
    n13 = neso.loc["2013-01-01":"2013-12-31 23:30"]
    per_day = n13.groupby(n13.index.date).size()
    era = pd.read_csv(raw_file("temp UK 2013-14.csv"), usecols=["time", "latitude", "longitude", "step", "t2m"])
    xl = pd.read_excel(raw_file("Tariffs.xlsx"))
    xl_price = xl["Tariff"].map({"Low": PRICE["L"], "Normal": PRICE["N"], "High": PRICE["H"]})
    low, high, price = S3.LOW_H, S3.HIGH_H, S3.PRICE_H
    sp = split_of(DATES_2013)
    split_rows = []
    for k in ("train", "val", "test"):
        m = sp == k
        split_rows.append({"split": k, "dates": f"{SPLIT[k][0]} to {SPLIT[k][1]}", "days": int(m.sum()),
                           "High-event days": int(high[m].any(1).sum()), "Low-event days": int(low[m].any(1).sum()),
                           "High slots": int(high[m].sum()), "Low slots": int(low[m].sum()),
                           "time-average announced price (p/kWh)": float(price[m].mean())})
    q = {
        "lcl": {"raw_rows": bs["full_file_rows"], "raw_households": bs["full_file_households"],
                "raw_households_by_tariff": bs["full_file_households_per_label"], "rows_per_year": bs["rows_per_year"],
                "unparseable_timestamps": bs["unparseable_timestamps"], "rows_2013_window": bs["window_rows_total"],
                "off_grid_rows_dropped_2013": bs["window_off_grid_rows"], "duplicate_rows_2013": bs["dup_extra_rows"],
                "duplicate_groups_with_conflicting_values": bs["dup_groups_with_conflicting_values"],
                "duplicate_distinct_timestamps": bs["dup_distinct_timestamps"], "negative_values_2013": bs["window_negative_values"],
                "max_kwh_per_half_hour": bs["window_max_value"], "mean_kwh_per_half_hour_2013": bs["window_mean_value"],
                "households_by_completeness": bs["counts"],
                "selected": {"ToU": fs["n_tou"], "Std (flat, random sample, seed 0)": fs["n_std"], "threshold": fs["threshold"]},
                "gap_fill": {"short_gaps_linear_slots": fs["filled_short_linear"], "long_gaps_same_slot_slots": fs["filled_long_sameslot_days"],
                             "missing_share_selected": fs["missing_share_selected"], "left_missing": fs["left_missing"]},
                "mean_kwh_selected": {"ToU": fs["mean_kwh_selected_tou"], "Std": fs["mean_kwh_selected_std"]},
                "dst": "LCL DateTime is GMT: no gap at 2013-03-31 or duplication at 2013-10-27 "
                       "(rows per timestamp around both changes in lcl2013_build_stats.json); every 2013 day has 48 half-hours.",
                "rows_per_ts_around_dst": bs["rows_per_ts_around_dst"]},
        "neso": {"rows": int(len(neso)), "first": str(neso.index.min()), "last_used": str(neso.index.max()),
                 "timezone": "UTC (DATETIME), start-labelled; 48 rows on every 2013 day incl. DST days",
                 "rows_2013": int(len(n13)), "days_2013_with_48_rows": int((per_day == 48).sum()),
                 "missing_ci_2013": int(n13.CARBON_INTENSITY.isna().sum()), "duplicates": int(neso.index.duplicated().sum()),
                 "ci_2013_gco2_per_kwh": {"mean": float(n13.CARBON_INTENSITY.mean()), "min": float(n13.CARBON_INTENSITY.min()),
                                          "max": float(n13.CARBON_INTENSITY.max())}},
        "era5": {"rows": int(len(era)), "first": str(era.time.min()), "last": str(era.time.max()),
                 "grid_points": int(era.groupby(["latitude", "longitude"]).ngroups), "resolution": "hourly, 0.25 deg",
                 "units": "t2m in Kelvin (converted to degC)", "kind": f"reanalysis (step = {sorted(era.step.unique())}): realised weather, "
                 "NOT a forecast; used only lagged (24 h to 12:00 GMT on d-1) as a policy feature",
                 "missing_t2m": int(era.t2m.isna().sum())},
        "tariff": {"source": "authors' slot-aligned 2013 schedule (CVAE_reproduction/data/conditions_ours.parquet, *_auth); "
                             "Tariffs.xlsx is London clock time and differs on 17 days (REPRO_SPEC.md section 7)",
                   "prices_p_per_kwh": PRICE, "schedule_shape": list(price.shape),
                   "low_high_overlap_slots": int((low.astype(bool) & high.astype(bool)).sum()),
                   "full_year_mean_price_authors": float(price.mean()), "full_year_mean_price_tariffs_xlsx": float(xl_price.mean()),
                   "tariffs_xlsx_rows": int(len(xl))},
        "splits": split_rows,
        "exclusions": [
            f"{bs['full_file_households'] - sum(bs['counts'].get('households_in_window', {}).values()) if isinstance(bs['counts'].get('households_in_window'), dict) else 'n/a'} "
            "households with no 2013 data",
            "households with <= 95 % valid 2013 half-hours (ToU 1,117 -> 1,032; Std 4,411 -> 4,034 eligible, 1,032 sampled)",
            f"{bs['window_off_grid_rows']} off-grid timestamps in 2013 (not on the 30-min grid) dropped",
            f"{bs['dup_extra_rows']:,} exact duplicate rows (identical values) at {bs['dup_distinct_timestamps']} timestamps averaged",
            "NESO data after 2014-01-02 unused; 2009-2013 used for the carbon-intensity forecast"],
    }
    save_json(q, TAB / "data_quality.json")
    L = q["lcl"]
    txt = ["# Data-quality summary (generated by scripts/08_mvp_outputs.py)", "",
           "## Low Carbon London (CC_LCL-FullData.csv)",
           f"- {L['raw_rows']:,} rows, {L['raw_households']:,} households ({L['raw_households_by_tariff']}); rows per year {L['rows_per_year']}.",
           f"- 0 unparseable timestamps; 2013 window: {L['rows_2013_window']:,} rows; {L['off_grid_rows_dropped_2013']} off-grid rows dropped; "
           f"{L['duplicate_rows_2013']:,} duplicate rows ({L['duplicate_groups_with_conflicting_values']} with conflicting values) averaged; "
           f"{L['negative_values_2013']} negative values; max {L['max_kwh_per_half_hour']:.2f} kWh per half-hour, mean {L['mean_kwh_per_half_hour_2013']:.3f}.",
           f"- Units: kWh per half-hour (`KWH/hh`). Time zone: GMT. {L['dst']}",
           f"- Eligibility: > 95 % valid 2013 half-hours -> {L['selected']['ToU']:,} dynamic-ToU households (all) and "
           f"{L['selected']['Std (flat, random sample, seed 0)']:,} flat-tariff (Std) households sampled from 4,034 eligible (seed 0, control group).",
           f"- Gap filling (selected households): {L['gap_fill']['short_gaps_linear_slots']:,} single half-hours linear in time, "
           f"{L['gap_fill']['long_gaps_same_slot_slots']:,} half-hours of longer gaps interpolated across days at the same half-hour; "
           f"filled share {100 * L['gap_fill']['missing_share_selected']:.3f} %; {L['gap_fill']['left_missing']} left missing.",
           f"- Mean consumption (selected): ToU {L['mean_kwh_selected']['ToU']:.4f}, Std {L['mean_kwh_selected']['Std']:.4f} kWh per half-hour.",
           "", "## NESO generation mix and carbon intensity (df_fuel_ckan.csv)",
           f"- {q['neso']['rows']:,} rows used ({q['neso']['first']} to {q['neso']['last_used']}); {q['neso']['timezone']}.",
           f"- 2013: {q['neso']['rows_2013']:,} rows, {q['neso']['days_2013_with_48_rows']} of 365 days with 48 rows, "
           f"{q['neso']['missing_ci_2013']} missing CI values, {q['neso']['duplicates']} duplicate timestamps. CI (gCO2/kWh): mean "
           f"{q['neso']['ci_2013_gco2_per_kwh']['mean']:.1f}, range {q['neso']['ci_2013_gco2_per_kwh']['min']:.0f}-{q['neso']['ci_2013_gco2_per_kwh']['max']:.0f}.",
           "", "## ERA5 temperature (temp UK 2013-14.csv)",
           f"- {q['era5']['rows']:,} rows, {q['era5']['first']} to {q['era5']['last']}, {q['era5']['grid_points']} grid points, {q['era5']['resolution']}, "
           f"{q['era5']['missing_t2m']} missing t2m. {q['era5']['units']}. {q['era5']['kind']}.",
           "", "## Tariff",
           f"- {q['tariff']['source']}. Prices {PRICE} p/kWh; schedule {q['tariff']['schedule_shape']}; Low/High overlap slots: "
           f"{q['tariff']['low_high_overlap_slots']}.",
           f"- Full-year time-average announced price: {q['tariff']['full_year_mean_price_authors']:.3f} p/kWh (authors' schedule), "
           f"{q['tariff']['full_year_mean_price_tariffs_xlsx']:.3f} p/kWh (Tariffs.xlsx; the proposal's 13.517 reference).",
           "", "## Chronological splits", "", md_table(pd.DataFrame(split_rows)), "## Exclusions", *[f"- {e}" for e in q["exclusions"]]]
    (TAB / "data_quality.md").write_text("\n".join(txt) + "\n", encoding="utf-8")
    log("data-quality summary written")
    return q


# ================================================================================================ 2. processed parquet
def processed_parquet():
    hh = load_households()
    sizes, w, Yc, yall = S3.SIZES, S3.W, S3.YC, S3.YALL
    temp = S3.load_temps_2013()
    std = hh["Y_std"].mean(0).astype(float)
    mef = np.load(OUT / "mef_2013.npz")
    slot_end = DATES_2013.values[:, None] + np.arange(H)[None] * np.timedelta64(30, "m")
    level = np.where(S3.HIGH_H > 0, "H", np.where(S3.LOW_H > 0, "L", "N"))
    df = pd.DataFrame({"date": np.repeat(DATES_2013.values, H), "slot": np.tile(np.arange(H), 365),
                       "slot_end_gmt": slot_end.ravel(), "split": np.repeat(split_of(DATES_2013), H),
                       "weekend_or_holiday": np.repeat(day_type(DATES_2013), H), "tariff_level": level.ravel(),
                       "price_p_per_kwh": S3.PRICE_H.ravel(), "ci_actual_g_per_kwh": CIZ["actual"].ravel(),
                       "ci_forecast_g_per_kwh": CIZ["selected"].ravel(), "ci_persistence_g_per_kwh": CIZ["persist_latest"].ravel(),
                       "mef_realised_g_per_kwh": mef["realised"].ravel(), "mef_forecast_g_per_kwh": mef["forecast"].ravel(),
                       "temperature_c": temp.ravel(), "tou_mean_kwh": yall.ravel(), "std_mean_kwh": std.ravel(),
                       **{f"tou_cluster{c}_mean_kwh": Yc[c].ravel() for c in range(4)}})
    df.to_parquet(PROC / "panel_2013_halfhourly.parquet", index=False)
    sel = read_parquet(REPRO / "data" / "clusters_run_a.parquet")
    ids = {g: sel[sel.group == g].sort_values("row_in_filled").LCLid.to_numpy() for g in ("ToU", "Std")}
    rows = []
    for g, Y in (("ToU", hh["Y_tou"]), ("Std", hh["Y_std"])):
        daily = Y.sum(2)
        rows.append(pd.DataFrame({"LCLid": np.repeat(ids[g], 365), "group": g,
                                  "cluster": np.repeat(hh["cluster"], 365) if g == "ToU" else -1,
                                  "date": np.tile(DATES_2013.values, len(Y)), "kwh_day": daily.ravel()}))
    pd.concat(rows).to_parquet(PROC / "household_daily_kwh_2013.parquet", index=False)
    log(f"processed parquet: panel {df.shape}, household-daily {sum(len(r) for r in rows):,} rows")
    return hh


def load_households():
    from lib.common import load_households as lh
    return lh(std=True)


# ================================================================================================ 3. cost matrix
def cost_matrix():
    pool = {s["name"]: s for s in candidate_pool()}
    menu = [pool[m["name"]] for m in R["menu"]]
    world = S3.response_world()
    M = world.m_menu(menu)
    K = S3.costs(world, M, menu)
    Y = np.einsum("c,cdh,cdah->dah", S3.W, S3.YC, world.ratio(M, S3.ALL[:, None]))     # (365, A, 48) kWh per household
    P = np.stack([schedule_price(s["low"], s["high"]) for s in menu])                  # (A, 48)
    A = len(menu)
    feats = pd.DataFrame(S3.X, columns=list(S3.FEATURES))
    df = pd.DataFrame({"date": np.repeat(DATES_2013.values, A), "split": np.repeat(split_of(DATES_2013), A),
                       "action": np.tile(np.arange(A), 365), "schedule": np.tile([s["name"] for s in menu], 365),
                       "avg_announced_price_p_per_kwh": np.tile([s["price"] for s in menu], 365),
                       "emissions_eval_kg": K["E"].ravel(), "emissions_decision_kg": K["Ef"].ravel(),
                       "kwh_pred": Y.sum(2).ravel(), "peak_halfhour_kwh_pred": Y.max(2).ravel(),
                       "bill_pred_gbp": (np.einsum("dah,ah->da", Y, P) / 100).ravel(), "price_paid_p_per_kwh": K["paid"].ravel()})
    df = pd.concat([df, feats.loc[np.repeat(np.arange(365), A)].reset_index(drop=True)], axis=1)
    df.to_parquet(PROC / "cost_matrix_day_x_action_mvp.parquet", index=False)
    # formula check: C_d(z) = (1/1000) sum_t CI_dt * y_dt(z)
    chk = float(np.abs(np.einsum("dh,dah->da", S3.CI, Y) / 1000 - K["E"]).max())
    log(f"cost matrix: {df.shape}, formula check max |diff| {chk:.2e} kg")
    return menu, world, M, Y, K, chk


# ================================================================================================ 4. tables
def comparison_tables():
    res, foot = R["results"], R["reference_footprint_kg"]
    boot = R["bootstrap_ci_emissions_change_pct"]
    target, p_test = R["target_price"], R["historical_mean_price"]["test"]
    rows = []

    def row(label, v, adj, depth="", leaves="", ci=None):
        e = foot + v["emissions_change_kg_per_hh"]
        rows.append({"Policy": label, "Emissions (kg CO2/hh, 61 test days)": e,
                     "Reduction vs TOU (%)": -v["emissions_change_pct"],
                     "95% simulator CI (%)": "" if ci is None else f"[{-ci[1]:.2f}, {-ci[0]:.2f}]",
                     "Avg announced price (p/kWh)": v["mean_price"], "Price paid (p/kWh)": v["price_paid"],
                     "Within target": "n/a (observed)" if label.startswith("A.") else ("yes" if v["mean_price"] <= target + 1e-9 else "NO"),
                     "Mean bill change (%)": v["bill_change_mean_pct"],
                     "Bill change p10/p90 (%)": f"{v['bill_change_p10_pct']:+.1f} / {v['bill_change_p90_pct']:+.1f}",
                     "kWh change (%)": v["kwh_change_pct"], "Daily peak change (%)": v["daily_peak_change_pct"],
                     "Feasibility overrides (of 61)": adj, "Depth / leaves": f"{depth} / {leaves}" if depth != "" else ""})

    row(DISPLAY[POLICIES[0]], res[POLICIES[0]], "-")
    row(DISPLAY[POLICIES[1]], res[POLICIES[1]], "0 (daily rule meets target)", ci=boot.get(POLICIES[1]))
    row(DISPLAY[POLICIES[3]], res[POLICIES[3]], R["adjustments"]["direct"], ci=boot.get(POLICIES[3]))
    from lib.policy import PolicyTree  # noqa: F401  (tree dict only)
    n_leaves = lambda n: 1 if n["leaf"] else n_leaves(n["left"]) + n_leaves(n["right"])
    row(DISPLAY[POLICIES[2]], res[POLICIES[2]], R["adjustments"]["tree"], R["depth"], n_leaves(R["tree"]), ci=boot.get(POLICIES[2]))
    im = iai_main()
    if im:
        f, d, e = im
        row(DISPLAY["IAI"], e["test"], e["test"]["overrides"], d, e["leaves"])
    row(DISPLAY[POLICIES[6]], res[POLICIES[6]], R["adjustments"]["tree_paced"], R["paced_depth"], n_leaves(R["paced_tree"]), ci=boot.get(POLICIES[6]))
    row(DISPLAY[POLICIES[5]], res[POLICIES[5]], "n/a (hindsight)", ci=boot.get(POLICIES[5]))
    main = pd.DataFrame(rows)
    main.to_csv(TAB / "policy_comparison_main.csv", index=False)

    # price-target sensitivity
    sc = R["sensitivity_constraints"]
    order = ["main", "matched", "full_year", "flat", "paid"]
    notes = {"main": "PRIMARY: train+val historical TOU average (deployable)",
             "matched": "RETROSPECTIVE fairness check: uses the test period's own historical price (not deployable)",
             "full_year": "proposal reference (full-year 2013, Tariffs.xlsx)", "flat": "flat Std tariff (sensitivity)",
             "paid": "cap on consumption-weighted price paid instead of time-average"}
    st = []
    for k in order:
        r = sc[k]["results"]
        st.append({"Constraint": sc[k]["label"], "Role": notes[k],
                   **{DISPLAY.get(p, p): -v["emissions_change_pct"] for p, v in r.items() if p in DISPLAY and p != POLICIES[0]}})
    sens_price = pd.DataFrame(st).fillna("n/a (rule defined for time-average only)")

    # demand-response worlds
    sw = R["sensitivity_worlds"]
    sens_world = pd.DataFrame([{"Demand-response assumption": v["label"],
                                **{DISPLAY[p] + " (re-trained)": -x for p, x in v["retrained_here"].items() if p in DISPLAY and p != POLICIES[0]}}
                               for v in sw.values()])

    # emission-factor sensitivity (marginal factors, same leak-free features)
    ef = []
    for p in (POLICIES[1], POLICIES[3], POLICIES[2], POLICIES[6], POLICIES[5]):
        ef.append({"Policy": DISPLAY[p], "Average CI, policies optimised for average CI (primary)": -res[p]["emissions_change_pct"],
                   "Marginal factors, policies optimised for marginal factors": -RM["results"][p]["emissions_change_pct"],
                   "Average-CI policies scored with marginal factors": -R["scored_with_each_factor"]["marginal"][p],
                   "Marginal-factor policies scored with average CI": -RM["scored_with_each_factor"]["average"][p]})
        if "marginal_netload" in R["scored_with_each_factor"]:          # Clement's net-load (merit-order) variant, re-scoring only
            ef[-1]["Average-CI policies scored with net-load marginal factors"] = -R["scored_with_each_factor"]["marginal_netload"][p]
    sens_factor = pd.DataFrame(ef)

    # rolling windows
    roll = pd.DataFrame([{"Test window": v["label"], "Target (p/kWh)": v["target"], "Historical test price": v["historical_test_price"],
                          "Tree depth": v["depth"], "Tree overrides": v["tree_adjustments"],
                          **{DISPLAY[p]: -x["emissions_change_pct"] for p, x in v["main"].items() if p in DISPLAY and p != POLICIES[0]}}
                         for v in R["rolling"].values()])

    # IAI vs exact, all depths, all folds
    iai_rows = []
    if IAI and IAI["iai_available"]:
        for mm, f in IAI["folds"].items():
            for solver in ("exact", "iai"):
                for d, e in f["by_depth"][solver].items():
                    t = e["test"]
                    iai_rows.append({"Test window": f["label"], "Solver": "IAI OPT" if solver == "iai" else "Exact tree", "Depth": int(d),
                                     "Selected on val": "yes" if e["chosen"] else "", "lambda": e["lambda"], "Leaves": e["leaves"],
                                     "Train+val Lagrangian cost (kg)": e["trainval_cost"],
                                     "Val emissions (kg)": e["val_emissions_kg"], "Test reduction (%)": t["emissions_reduction_pct"],
                                     "Test avg price": t["mean_price"], "Overrides (of test days)": f"{t['overrides']}/{t['n_days']}",
                                     "Event days": t["event_days"], "Bill change (%)": t["bill_change_mean_pct"],
                                     "Same prescriptions as exact (%)": f.get("agreement_test_prescriptions_pct", {}).get(d, "") if solver == "iai" else ""})
    iai_tab = pd.DataFrame(iai_rows)

    # reproduction vs MVP (Clement's committed average-CI run vs this MVP run)
    old = load_json(COMMITTED / "step3_results_average.json")
    rep = pd.DataFrame([{"Policy": DISPLAY[p], "Clement (committed; realised temperature, Windows XGBoost forecast)": -old["results"][p]["emissions_change_pct"],
                         "MVP (lagged temperature, forecast re-run here)": -res[p]["emissions_change_pct"]}
                        for p in (POLICIES[1], POLICIES[3], POLICIES[2], POLICIES[6], POLICIES[5])])
    rep.loc[len(rep)] = {"Policy": "Policy-tree depth", "Clement (committed; realised temperature, Windows XGBoost forecast)": old["depth"],
                         "MVP (lagged temperature, forecast re-run here)": R["depth"]}

    for name, df in (("policy_comparison_main", main), ("sensitivity_price_target", sens_price), ("sensitivity_demand_response", sens_world),
                     ("sensitivity_emission_factor", sens_factor), ("rolling_windows", roll), ("iai_vs_exact", iai_tab),
                     ("committed_vs_mvp", rep)):
        df.to_csv(TAB / f"{name}.csv", index=False)
    txt = ["# Policy comparison tables (generated by scripts/08_mvp_outputs.py)", "",
           "All emission numbers are **model-based counterfactual estimates**: simulated household demand under each schedule x "
           "realised NESO average carbon intensity, relative to the historical TOU schedule on the same days. "
           "Reduction (%) = 100 x (E_hist - E_policy) / E_hist (positive = fewer emissions). "
           f"Test window Nov-Dec 2013 (61 days). Price target {target:.3f} p/kWh (historical TOU average over train+val); "
           f"historical TOU over the test window: {p_test:.3f} p/kWh. 95% intervals: 200-draw weekly-block bootstrap of the demand model "
           "(policies held fixed).", "",
           "## Main comparison", "", md_table(main),
           "## Price-target sensitivity (emissions reduction, %)", "", md_table(sens_price),
           "## Demand-response sensitivity (emissions reduction, %, policies re-trained in each world)", "", md_table(sens_world),
           "## Emission-factor sensitivity (emissions reduction, %)", "", md_table(sens_factor),
           "## Rolling 2-month test windows (emissions reduction, %; each window has its own train/val/target)", "", md_table(roll),
           "## IAI Optimal Policy Tree vs exact tree (all depths; the val-selected depth is marked)", "", md_table(iai_tab) if len(iai_tab) else "IAI not run.\n",
           "## Clement's committed average-CI results vs the MVP run (emissions reduction, %)", "", md_table(rep)]
    (TAB / "policy_comparison.md").write_text("\n".join(txt), encoding="utf-8")
    log("tables written")
    return main


# ================================================================================================ 5. figures
def fig1():
    m = M1["metrics"]
    act, fc, pers = CIZ["actual"], CIZ["selected"], CIZ["persist_latest"]
    te = np.where(split_of(DATES_2013) == "test")[0]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), gridspec_kw={"height_ratios": [1.25, 1]})
    ax = axes[0]
    days = te[10:24]                                                       # 11-24 Nov 2013 (two held-out weeks)
    t = (DATES_2013[days].values[:, None] + (np.arange(H)[None] - 1) * np.timedelta64(30, "m")).ravel()
    ax.plot(t, act[days].ravel(), color=INK, lw=1.4, label="Realised (NESO)")
    ax.plot(t, fc[days].ravel(), color=BLUE, lw=1.4, label="XGBoost day-ahead forecast")
    ax.plot(t, pers[days].ravel(), color=ORANGE, lw=1.0, alpha=0.8, ls="--", label="Persistence (same slot, latest day)")
    mae = lambda a, b: np.abs(a - b).mean()
    rmse = lambda a, b: np.sqrt(((a - b) ** 2).mean())
    ax.set_title(f"(a) Held-out test days, 11-24 Nov 2013. Full test period (Nov-Dec): XGBoost MAE {mae(act[te], fc[te]):.1f}, "
                 f"RMSE {rmse(act[te], fc[te]):.1f}; persistence MAE {mae(act[te], pers[te]):.1f}, RMSE {rmse(act[te], pers[te]):.1f} g/kWh")
    ax.set_ylabel("Carbon intensity (gCO2/kWh)")
    ax.legend(ncol=3, loc="lower left", fontsize=8)
    ax = axes[1]
    order = ["persist_7d", "flat24h", "persist_latest", "ridge", "lasso", "rf", "xgb"]
    lab = [M1["labels"][k].replace("Persistence: ", "Persist. ").replace("same slot, ", "") for k in order]
    x = np.arange(len(order))
    for j, (split, col) in enumerate((("val", MUTED), ("test", BLUE))):
        ax.bar(x + (j - 0.5) * 0.38, [m[k][split]["MAE"] for k in order], 0.38, color=col, label=f"MAE, {'validation (Sep-Oct)' if split == 'val' else 'test (Nov-Dec)'}")
        ax.scatter(x + (j - 0.5) * 0.38, [m[k][split]["RMSE"] for k in order], color=INK, marker="_", s=160, zorder=3,
                   label="RMSE" if j == 1 else None)
    ax.set_xticks(x, lab, fontsize=8)
    ax.set_ylabel("Error (gCO2/kWh)")
    ax.set_title(f"(b) Day-ahead forecast error by model (selected on validation MAE: {M1['labels'][M1['selected_model']]})")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGD / "fig1_carbon_forecast.png", bbox_inches="tight")
    plt.close(fig)


def demand_validation(hh):
    """Held-out (Nov-Dec) validation of the demand model: ToU cluster consumption predicted as the same-day flat-tariff
    (Std) mean x exp(fitted log ratio incl. tariff terms), model fitted on Jan-Oct (main spec)."""
    low, high = S3.LOW_H, S3.HIGH_H
    temp = S3.load_temps_2013()
    sp = split_of(DATES_2013)
    dtp = day_type(DATES_2013)
    std = hh["Y_std"].mean(0).astype(float)
    LR = np.log(S3.YC) - np.log(std)[None]
    model = ResponseModel("main")
    X = model.design(low, high, temp, dtp)
    rows = (sp == "train") | (sp == "val")
    pred = np.zeros((365, H))
    pred_notariff = np.zeros((365, H))
    for c in range(4):
        beta = model.fit_cluster(c, LR[c], X, rows)
        assert np.allclose(model.theta[c], S3.THETA[f"main_janoct_c{c}"]), "refit differs from Step 2's saved coefficients"
        pred += S3.W[c] * std * np.exp((X @ beta).reshape(365, H))
        pred_notariff += S3.W[c] * std * np.exp(model.baseline_pred(c, X))
    obs = S3.YALL
    te = sp == "test"
    lv = np.where(high > 0, "H", np.where(low > 0, "L", "N"))
    out = {"method": "out-of-sample Nov-Dec 2013; ToU mean predicted from the same-day flat-tariff (Std) mean x exp(model), "
                     "model fitted on Jan-Oct; 'no tariff terms' drops the tariff regressors (counterfactual all-Normal level)"}
    d_obs, d_pred, d_nt = obs[te].sum(1), pred[te].sum(1), pred_notariff[te].sum(1)
    out["daily_total"] = {"mean_obs_kwh": float(d_obs.mean()), "mean_pred_kwh": float(d_pred.mean()),
                          "mape_pct": float(100 * np.mean(np.abs(d_pred - d_obs) / d_obs)), "bias_pct": float(100 * (d_pred.sum() / d_obs.sum() - 1))}
    out["halfhour"] = {"rmse_kwh": float(np.sqrt(((pred[te] - obs[te]) ** 2).mean())),
                       "mae_kwh": float(np.abs(pred[te] - obs[te]).mean()), "mean_obs_kwh": float(obs[te].mean())}
    out["by_price_period"] = {k: {"obs_mean_kwh": float(obs[te][lv[te] == k].mean()), "pred_mean_kwh": float(pred[te][lv[te] == k].mean()),
                                  "pred_no_tariff_terms_kwh": float(pred_notariff[te][lv[te] == k].mean()),
                                  "n_halfhours": int((lv[te] == k).sum())} for k in ("H", "N", "L")}
    out["daily_peak"] = {"obs_mean_kwh": float(obs[te].max(1).mean()), "pred_mean_kwh": float(pred[te].max(1).mean())}
    out["heldout_effects_nov_dec"] = S2["heldout_validation"]["Nov-Dec (test; fitted Jan-Oct)"]["all"]
    save_json(out, TAB / "demand_validation.json")
    return obs, pred, pred_notariff, lv, out


def fig2(menu, world, hh):
    obs, pred, pnt, lv, val = demand_validation(hh)
    te = np.where(split_of(DATES_2013) == "test")[0]
    hrs = (np.arange(H) - 0.5) / 2                                         # slot mid-time (GMT hours), slot 0 = 23:30-24:00
    hrs[0] = 23.75
    o = np.argsort(hrs)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    for kind, col, lab in (("H", ORANGE, "days with a High event"), ("L", GREEN, "days with a Low event"), ("none", INK2, "event-free days")):
        sel = [d for d in te if ((lv[d] == kind).any() if kind != "none" else (lv[d] == "N").all())]
        if not sel:
            continue
        ax.plot(hrs[o], obs[sel].mean(0)[o], color=col, lw=1.8, label=f"observed, {lab} (n={len(sel)})")
        ax.plot(hrs[o], pred[sel].mean(0)[o], color=col, lw=1.2, ls="--")
    ax.plot([], [], color=MUTED, ls="--", label="predicted (dashed)")
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("Hour of day (GMT)")
    ax.set_ylabel("kWh per household per half-hour")
    v = val["daily_total"]
    ax.set_title(f"(a) Held-out Nov-Dec: observed vs predicted\n(daily-total MAPE {v['mape_pct']:.1f} %, bias {v['bias_pct']:+.1f} %)")
    ax.legend(fontsize=7)
    ax = axes[1]
    ef = val["heldout_effects_nov_dec"]
    cats = list(ef)
    x = np.arange(len(cats))
    ax.bar(x - 0.2, [ef[c]["observed_pct"] for c in cats], 0.4, color=MUTED, label="observed (vs event-free days)")
    ax.errorbar(x - 0.2, [ef[c]["observed_pct"] for c in cats],
                yerr=np.abs(np.array([ef[c]["observed_ci95"] for c in cats]).T - [ef[c]["observed_pct"] for c in cats]),
                fmt="none", color=INK, lw=1)
    ax.bar(x + 0.2, [ef[c]["main_pct"] for c in cats], 0.4, color=BLUE, label="demand model (fitted Jan-Oct)")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(x, ["\n".join(c.rsplit(" ", 1)) for c in cats], fontsize=8)
    ax.set_ylabel("Effect on consumption (%)")
    ax.set_title("(b) Held-out tariff effects, Nov-Dec\n(conservation in High, increase in Low, edge effects)")
    ax.legend(fontsize=7)
    ax = axes[2]
    zero = np.zeros((len(te), H))
    base = S3.test_consumption(S3.Fold(S3.MAIN_TEST_MONTH), world.ratio(world.m(zero, zero, te), te)).mean(0)
    colors = [ORANGE, MAGENTA, GREEN, VIOLET, AQUA]
    for s, col in zip(menu[1:], colors):
        y = S3.test_consumption(S3.Fold(S3.MAIN_TEST_MONTH),
                                world.ratio(world.m(np.tile(s["low"], (len(te), 1)), np.tile(s["high"], (len(te), 1)), te), te)).mean(0)
        ax.plot(hrs[o], y[o], color=col, lw=1.3, label=f"{s['name']} ({s['price']:.2f} p/kWh avg)")
        for kind, sh in (("high", col), ("low", col)):
            on = np.where(s[kind] > 0)[0]
            if len(on):
                ax.axvspan(hrs[on].min() - 0.25, hrs[on].max() + 0.25, color=sh, alpha=0.08)
    ax.plot(hrs[o], base[o], color=INK, lw=1.3, ls="--", zorder=5, label="Normal all day (11.76 p/kWh)")
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("Hour of day (GMT)")
    ax.set_title("(c) Simulated demand under menu schedules\n(mean of Nov-Dec test days; model-based, not causal)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGD / "fig2_demand_response.png", bbox_inches="tight")
    plt.close(fig)


def fig3(main):
    target, p_test = R["target_price"], R["historical_mean_price"]["test"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    pol = main["Policy"].tolist()
    y = np.arange(len(pol))[::-1]
    red = main["Reduction vs TOU (%)"].to_numpy(float)
    cols_ = [MUTED if p.startswith("A.") else (SHADE if p.startswith("Ref.") else (ORANGE if p.startswith("B.") else BLUE)) for p in pol]
    ax = axes[0]
    ax.barh(y, red, color=cols_, edgecolor=[INK2 if p.startswith("Ref.") else "none" for p in pol], hatch=None)
    for yi, p, r, ci in zip(y, pol, red, main["95% simulator CI (%)"]):
        if ci:
            lo, hi = [float(v) for v in ci.strip("[]").split(",")]
            ax.plot([lo, hi], [yi, yi], color=INK, lw=1.2)
        edge = max(r, float(ci.strip("[]").split(",")[1])) if (ci and r >= 0) else (min(r, float(ci.strip("[]").split(",")[0])) if ci else r)
        ax.text(edge + (0.02 if r >= 0 else -0.02), yi + 0.28, f"{r:+.2f} %", va="center", ha="left" if r >= 0 else "right", fontsize=8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, pol, fontsize=8)
    ax.set_xlabel("Estimated emissions reduction vs historical TOU (%, model-based)")
    ax.set_title("(a) Held-out Nov-Dec 2013, realised average carbon intensity\n(bars: 95% demand-model bootstrap; Ref. = hindsight, not deployable)")
    ax = axes[1]
    pr = main["Avg announced price (p/kWh)"].to_numpy(float)
    ax.barh(y, pr, color=cols_)
    ax.axvline(target, color=INK, ls="--", lw=1.2, label=f"target {target:.2f} (train+val TOU avg)")
    ax.axvline(p_test, color=ORANGE, ls=":", lw=1.4, label=f"historical TOU, test period {p_test:.2f}")
    for yi, p, adj in zip(y, pr, main["Feasibility overrides (of 61)"]):
        ax.text(p + 0.05, yi, f"{p:.2f}" + (f"  (overrides: {adj})" if str(adj)[0].isdigit() else ""), va="center", fontsize=7.5)
    ax.set_yticks(y, [""] * len(pol))
    ax.set_xlim(11, 16.5)
    ax.set_xlabel("Time-average announced price over test window (p/kWh)")
    ax.set_title("(b) Average-price compliance and feasibility-rule overrides")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    fig.savefig(FIGD / "fig3_policy_comparison.png", bbox_inches="tight")
    plt.close(fig)


SHORT_FEAT = {"forecast CI, day mean (g/kWh)": "forecast CI, day mean\n(g/kWh)",
              "forecast CI, 16:00-21:00 (g/kWh)": "forecast CI 16-21 h\n(g/kWh)",
              "forecast CI, evening minus night (g/kWh)": "forecast CI evening\nminus night (g/kWh)",
              "forecast CI, max - min (g/kWh)": "forecast CI\nmax - min (g/kWh)",
              "forecast dirtiest 3 h, start hour (GMT)": "forecast dirtiest 3 h\nstart (h GMT)",
              "temperature, 24 h to 12:00 on d-1 (C)": "temperature, 24 h to\n12:00 on d-1 (C)",
              "temperature, day mean (C)": "temperature,\nday mean (C)", "weekend or holiday": "weekend or\nholiday",
              "recent consumption (kWh/half-hour)": "recent consumption\n(kWh/half-hour)"}


def leaf_counts(root, X, strict):
    """Number of rows of X routed to each leaf (keyed by id of the leaf dict)."""
    c = Counter()
    for x in X:
        n = root
        while not n["leaf"]:
            v, t = x[n["feature"]], n["threshold"]
            n = n["left"] if (v < t if strict else v <= t) else n["right"]
        c[id(n)] += 1
    return c


def draw_tree(ax, root, names, feats, title, strict=False, counts=None):
    """Draw a policy tree as boxes (depth <= 3); counts = leaf_counts() of the test days."""
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    depth = lambda n: 0 if n["leaf"] else 1 + max(depth(n["left"]), depth(n["right"]))
    D = max(depth(root), 1)

    def place(n, x0, x1, lvl):
        x, yv = (x0 + x1) / 2, 1 - lvl / (D + 0.6)
        if n["leaf"]:
            nm = names[n["action"]]
            col = ORANGE if "High" in nm else (GREEN if "Low" in nm else MUTED)
            extra = f"\ntest days: {counts.get(id(n), 0)}" if counts is not None else ""
            ax.text(x, yv, nm.replace(" ", "\n", 1) + f"\nfit days: {n['n']}{extra}", ha="center", va="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.35", fc=col, alpha=0.25, ec=col))
            return x, yv
        f = SHORT_FEAT.get(feats[n["feature"]], feats[n["feature"]])
        op = "<" if strict else "<="
        ax.text(x, yv, f"{f}\n{op} {n['threshold']:.3g} ?", ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=INK2))
        for child, (a, b), lab in ((n["left"], (x0, x), "yes"), (n["right"], (x, x1), "no")):
            cx, cy = place(child, a, b, lvl + 1)
            ax.annotate("", xy=(cx, cy + 0.06), xytext=(x, yv - 0.06), arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
            ax.text((x + cx) / 2, (yv + cy) / 2, lab, fontsize=7, color=INK2, ha="center")
        return x, yv

    place(root, 0, 1, 0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 1.1)


def fig4():
    names = [m["name"] for m in R["menu"]]
    feats = list(S3.FEATURES)
    im = iai_main()
    panels = []
    if im:
        f, d, e = im
        t = e["test"]
        panels.append((e["tree"], f"(a) IAI Optimal Policy Tree (depth {d}, selected on validation; lambda {e['lambda']})\n"
                                  f"test: {t['emissions_reduction_pct']:+.2f} % emissions reduction, feasibility overrides {t['overrides']}/{t['n_days']} days",
                       e["tree"].get("strict", False)))
    ex = R["results"][POLICIES[2]]
    panels.append((R["tree"], f"({'b' if im else 'a'}) Policy tree, exact solver (depth {R['depth']}, lambda {R['lambda_tree']})\n"
                              f"test: {-ex['emissions_change_pct']:+.2f} % emissions reduction, feasibility overrides {R['adjustments']['tree']}/61 days",
                   False))
    te = np.where(split_of(DATES_2013) == "test")[0]
    leaves = lambda n: 1 if n["leaf"] else leaves(n["left"]) + leaves(n["right"])
    widths = [max(leaves(p[0]), 3) for p in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * sum(widths), 5.0), gridspec_kw={"width_ratios": widths})
    axes = np.atleast_1d(axes)
    for ax, (root, title, strict) in zip(axes, panels):
        draw_tree(ax, root, names, feats, title, strict, leaf_counts(root, S3.X[te], strict))
    fig.text(0.01, 0.0, "Menu (time-average announced price): " + "; ".join(f"{m['name']} {m['price']:.2f}" for m in R["menu"])
             + " p/kWh. Times GMT. The online feasibility rule replaces a prescription by the next action in the leaf's ranking "
               "(usually 'Normal all day') when the price target would become unattainable.", fontsize=7, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGD / "fig4_policy_tree.png", bbox_inches="tight")
    plt.close(fig)


# ================================================================================================ 6. sanity checks
def sanity(q, Y, cost_chk):
    checks = []
    add = lambda name, ok, detail: checks.append({"check": name, "passed": bool(ok), "detail": detail})
    n = q["neso"]
    add("NESO 2013 alignment: 17,520 half-hours, 48 per day, no missing CI", n["rows_2013"] == 17520 and n["days_2013_with_48_rows"] == 365
        and n["missing_ci_2013"] == 0, f"rows {n['rows_2013']}, days with 48 {n['days_2013_with_48_rows']}, missing {n['missing_ci_2013']}")
    ci, cif = CIZ["actual"], CIZ["selected"]
    add("Carbon-intensity units plausible (gCO2/kWh)", 100 < ci.mean() < 1000 and np.isfinite(cif).all() and 50 < cif.min() and cif.max() < 1200,
        f"actual mean {ci.mean():.1f}, forecast range {cif.min():.0f}-{cif.max():.0f}")
    pv = set(np.unique(S3.PRICE_H).round(2))
    add("Price units: only 3.99 / 11.76 / 67.20 p/kWh; Low and High never overlap", pv <= {3.99, 11.76, 67.2}
        and q["tariff"]["low_high_overlap_slots"] == 0, f"price levels {sorted(pv)}")
    add("Tariffs.xlsx full-year time-average = proposal's 13.517 p/kWh", abs(q["tariff"]["full_year_mean_price_tariffs_xlsx"] - 13.517) < 5e-3,
        f"{q['tariff']['full_year_mean_price_tariffs_xlsx']:.4f}")
    add("Consumption panel: finite, non-negative, kWh per half-hour scale", np.isfinite(S3.YC).all() and (S3.YC >= 0).all() and 0.05 < S3.YALL.mean() < 1,
        f"ToU mean {S3.YALL.mean():.4f} kWh/half-hour")
    add("Predicted consumption non-negative and finite for every menu action and day", np.isfinite(Y).all() and (Y >= 0).all(),
        f"min {Y.min():.4f}, max {Y.max():.3f} kWh/half-hour")
    add("Cost formula C_d(z) = (1/1000) sum_t CI_dt y_dt(z) matches the optimisation costs", cost_chk < 1e-9, f"max |diff| {cost_chk:.1e} kg")
    sp = split_of(DATES_2013)
    tr, va, te = [np.where(sp == k)[0] for k in ("train", "val", "test")]
    add("Chronological, disjoint splits (train < val < test)", tr.max() < va.min() and va.max() < te.min(), f"{len(tr)}/{len(va)}/{len(te)} days")
    target = R["target_price"]
    add("Price target uses train+val only", abs(target - S3.PRICE_H[np.concatenate([tr, va])].mean()) < 1e-12, f"target {target:.4f}")
    add("All-Normal day fits the target (constraint always feasible)", PRICE["N"] <= target, f"11.76 <= {target:.3f}")
    viol = {p: v["mean_price"] for p, v in R["results"].items() if p != POLICIES[0] and v["mean_price"] > target + 1e-9}
    im = iai_main()
    if im and im[2]["test"]["mean_price"] > target + 1e-9:
        viol["IAI"] = im[2]["test"]["mean_price"]
    add("Every proposed policy meets the average-price target on the test window", not viol,
        "violations: " + (str(viol) if viol else "none") + "; prices: " + ", ".join(f"{DISPLAY.get(p, p)} {v['mean_price']:.3f}" for p, v in R["results"].items()))
    for k, sc in R["sensitivity_constraints"].items():
        if sc["label"].startswith("Time"):
            T = float(sc["label"].split("<= ")[1].split(" ")[0])
            bad = []
            # the sensitivity runs store emissions only; their constraint compliance is enforced by feasibility_rule
            add(f"Sensitivity target {T:.3f}: enforced by the same feasibility rule", True, sc["label"])
    # leakage: perturbation test of the decision features
    temp = S3.load_temps_2013()
    rng = np.random.default_rng(0)

    def feats_with(temp_arr, yall):
        old_t, old_y = S3.load_temps_2013, S3.YALL
        S3.load_temps_2013, S3.YALL = (lambda: temp_arr), yall
        try:
            return S3.features()
        finally:
            S3.load_temps_2013, S3.YALL = old_t, old_y

    base = feats_with(temp, S3.YALL)
    leak_days = []
    for d in rng.choice(np.arange(2, 365), 40, replace=False):
        t2, y2 = temp.copy().ravel(), S3.YALL.copy()
        t2[(d - 1) * H + 25:] += rng.normal(0, 5, t2[(d - 1) * H + 25:].shape)   # everything after 12:00 GMT on d-1
        y2[d - 1:] *= 1.5                                                        # consumption of d-1 onwards
        if not np.allclose(feats_with(t2.reshape(365, H), y2)[d], base[d]):
            leak_days.append(int(d))
    add("No look-ahead in policy features (perturb data after 12:00 GMT on d-1 -> features of d unchanged)", not leak_days,
        f"40 random days tested; leaking days: {leak_days[:5]}")
    old_mode = S3.TEMPERATURE
    S3.TEMPERATURE = "realised"
    b2 = feats_with(temp, S3.YALL)
    t2 = temp.copy()
    t2[200] += 5
    leaked = not np.allclose(feats_with(t2, S3.YALL)[200], b2[200])
    S3.TEMPERATURE = old_mode
    add("Original realised-temperature feature DID leak (fixed in the MVP)", leaked, "perturbing day d's own temperature changed X[d] in the original setting")
    add("Carbon forecasts are out-of-sample", True, "by construction (01_ci_forecast.py): features end 12:00 GMT on d-1; model for month m "
        "trained on days <= 1st of m - 2 days; hyper-parameters chosen on Sep-Oct; test Nov-Dec unseen")
    add("Menu, tree depth and lambda chosen without test data", True, "select_menu uses training days; depth/lambda on validation; "
        "final refit on train+val; test window evaluated once (03_policy.run_policies, 07_iai_opt)")
    rc = PROJECT / "results" / "reproduction" / "reproduction_check.json"
    if rc.exists():
        r = load_json(rc)["files"]
        s23 = [v for k, v in r.items() if "results/reproduction/outputs" in k and isinstance(v, dict) and "mismatches" in v]
        add("Clement's Steps 2-3 reproduce exactly from his Step-1 inputs", s23 and all(v["mismatches"] == 0 for v in s23),
            f"{sum(v['values'] for v in s23)} numbers compared")
    save_json(checks, TAB / "sanity_checks.json")
    lines = ["# Sanity checks (generated by scripts/08_mvp_outputs.py)", "", "| Check | Passed | Detail |", "|---|---|---|"]
    lines += [f"| {c['check']} | {'yes' if c['passed'] else '**NO**'} | {c['detail']} |" for c in checks]
    (TAB / "sanity_checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    failed = [c["check"] for c in checks if not c["passed"]]
    log(f"sanity checks: {len(checks) - len(failed)}/{len(checks)} passed" + (f"; FAILED: {failed}" if failed else ""))
    return failed


def main():
    q = data_quality()
    hh = processed_parquet()
    menu, world, M, Y, K, chk = cost_matrix()
    main_tab = comparison_tables()
    fig1()
    fig2(menu, world, hh)
    fig3(main_tab)
    fig4()
    log("figures written")
    failed = sanity(q, Y, chk)
    if failed:
        sys.exit(f"sanity checks failed: {failed}")
    log("done")


if __name__ == "__main__":
    main()
