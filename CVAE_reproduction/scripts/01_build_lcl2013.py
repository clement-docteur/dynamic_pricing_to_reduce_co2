"""Rebuild the 2013 LCL household data set from the raw csv (paper Sec. III).

Steps
  1. CC_LCL-FullData.csv --(one DuckDB scan)--> typed scratch parquet (all rows);
     whole-file label statistics; on-grid rows of 2013 (+ 2012-12-31 23:30 and
     2014-01-01 00:00); off-grid timestamps dropped; duplicated (LCLid, DateTime) rows
     averaged; 'Null' -> NaN.  Writes
        data/lcl2013_long.parquet       (LCLid, group, ts, kwh)
        data/lcl2013_kwh.npy            (households x 17520) float32, NaN = missing
        data/lcl2013_edges.npy          (households x 2) float32 (2012-12-31 23:30, 2014-01-01 00:00)
        data/lcl2013_households.parquet (row, LCLid, group, n_valid_2013, completeness, ...)
        data/lcl2013_build_stats.json
  2. Gap filling for the selected households (all ToU with completeness > 0.95 and a
     default_rng(0) sample of as many Std households with completeness > 0.95).  Default
     rule = the authors' rule, reverse-engineered from their all-ToU aggregate: a single
     missing half-hour -> linear interpolation in time; any longer gap -> linear
     interpolation across days at the same half-hour between the nearest preceding and
     following days with an observed value (one side at the year edges).
        data/lcl2013_kwh_filled.npy     (selected x 17520) float32, ToU rows first
        data/lcl2013_edges_filled.npy   (selected x 2)
        data/selection.parquet          (LCLid, group, row_in_filled, completeness, row_in_dense,
                                         std_draw_order, in_authors_tou_1007, in_1007_pair)
        data/lcl2013_fill_stats.json
     ``in_authors_tou_1007`` flags the 1,007 ToU households that exactly reproduce the
     authors' all-ToU aggregate (NNLS identification); ``in_1007_pair`` = those 1,007 ToU
     plus the first 1,007 Std households drawn (paper-sized 1,007 + 1,007 design).

Run:  python -P scripts/01_build_lcl2013.py            (about 5-10 min, 4 threads)
      python -P scripts/01_build_lcl2013.py --reuse-parquet   (skip the csv scan if the
                                                       scratch parquet already exists)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import lcl_data as L  # noqa: E402


def group_counts(hh: pd.DataFrame, mask: np.ndarray) -> dict:
    return {str(g): int(c) for g, c in hh.loc[mask, "group"].value_counts().sort_index().items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(L.RAW_CSV))
    ap.add_argument("--data-dir", default=str(L.DATA), help="output directory (default: REPRO/data)")
    ap.add_argument("--work", default=None, help="scratch directory for the full parquet (env LCL_WORK)")
    ap.add_argument("--reuse-parquet", action="store_true", help="reuse the scratch parquet if it exists")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--short-max", type=int, default=L.AUTHORS_SHORT_MAX,
                    help="max length (slots) of a gap filled by linear interpolation in time "
                         "(default 1 = the authors' rule; the orchestration spec suggested 4)")
    ap.add_argument("--long-rule", default=L.AUTHORS_LONG_RULE, choices=["linear_days", "mean_nearest"],
                    help="rule for longer gaps (default: the authors' linear interpolation across days)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.work:
        os.environ["LCL_WORK"] = args.work
    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    P = {k: out_dir / Path(getattr(L, k)).name for k in
         ["LONG_PARQUET", "KWH_NPY", "EDGES_NPY", "HOUSEHOLDS_PARQUET", "FILLED_NPY", "EDGES_FILLED_NPY",
          "SELECTION_PARQUET", "BUILD_STATS_JSON", "FILL_STATS_JSON"]}
    el = L.timer()
    csv = Path(args.csv)
    full_pq = L.work_dir() / f"{csv.stem}_full.parquet"

    # ---------------------------------------------------------------- 1a csv -> parquet
    if args.reuse_parquet and full_pq.exists():
        print(f"[{el()}] reusing {full_pq}", flush=True)
    else:
        print(f"[{el()}] scanning {csv} ({csv.stat().st_size / 1e9:.2f} GB) -> {full_pq}", flush=True)
        L.csv_to_parquet(csv, full_pq, threads=args.threads)
        print(f"[{el()}] parquet written ({full_pq.stat().st_size / 1e9:.2f} GB)", flush=True)

    # ---------------------------------------------------------------- 1b statistics
    st_full, labels = L.full_file_stats(full_pq, threads=args.threads)
    print(f"[{el()}] full file: {st_full['full_file_rows']:,} rows, {st_full['full_file_households']} households, "
          f"labels {st_full['full_file_households_per_label']}, both-label households: "
          f"{len(st_full['households_with_both_labels_full_file'])}, unparseable ts: "
          f"{st_full['unparseable_timestamps']}, off-grid rows: {st_full['off_grid_rows_full_file']:,} "
          f"(numeric {st_full['off_grid_rows_full_file_numeric']})", flush=True)
    print(f"           rows per year {st_full['rows_per_year']}", flush=True)

    hh_win, hid, k, v, st_win = L.window_rows(full_pq, threads=args.threads)
    print(f"[{el()}] window: {st_win['window_rows_total']:,} rows, on-grid {st_win['window_on_grid_rows']:,}, "
          f"off-grid {st_win['window_off_grid_rows']} (numeric {st_win['window_off_grid_rows_numeric']}), "
          f"Null strings {st_win['window_on_grid_null_string_rows']:,}, households {len(hh_win)}, "
          f"both labels in window: {len(st_win['window_households_with_both_labels'])}", flush=True)

    # ---------------------------------------------------------------- 1c dedup + dense
    mean_ext, cnt_ext, st_dup, dup_det = L.dedup_dense(hid, k, v, len(hh_win))
    n_raw = len(hid)
    del hid, k, v
    print(f"[{el()}] duplicates: {st_dup['dup_groups']:,} (household, ts) groups, {st_dup['dup_extra_rows']:,} "
          f"extra rows, conflicting values in {st_dup['dup_groups_with_conflicting_values']} groups "
          f"(max spread {st_dup['dup_max_abs_spread_kwh']:.3f} kWh), {st_dup['dup_households']} households, "
          f"{st_dup['dup_distinct_timestamps']} distinct timestamps; top: "
          f"{list(st_dup['dup_top_timestamps'].items())[:5]}", flush=True)
    assert int(cnt_ext.sum()) == n_raw

    hh = L.households_table(hh_win, mean_ext, cnt_ext, labels)
    n_long = L.write_long_parquet(hh_win, mean_ext, cnt_ext, P["LONG_PARQUET"])
    print(f"[{el()}] long parquet: {n_long:,} rows -> {P['LONG_PARQUET']}", flush=True)

    # verification: re-read the long parquet with DuckDB and compare with the dense arrays
    con = L.duckdb_connect(args.threads)
    chk = con.execute(f"""SELECT LCLid, count(*) AS n, count(kwh) AS nv, sum(kwh) AS s
                          FROM read_parquet('{L._sql_path(P['LONG_PARQUET'])}') GROUP BY 1 ORDER BY 1""").df()
    con.close()
    chk["LCLid"] = chk["LCLid"].astype(str)
    chk = chk.set_index("LCLid").reindex(hh["LCLid"]).fillna(0)
    assert np.array_equal(chk["n"].to_numpy(), (cnt_ext > 0).sum(axis=1)), "long parquet row counts differ"
    assert np.array_equal(chk["nv"].to_numpy(), (~np.isnan(mean_ext)).sum(axis=1)), "long parquet valid counts differ"
    assert np.allclose(chk["s"].to_numpy(), np.nansum(mean_ext, axis=1), rtol=1e-9, atol=1e-6)
    print(f"[{el()}] long parquet verified against the dense arrays (counts and sums per household)", flush=True)

    X = mean_ext[:, 1:L.N_SLOTS + 1].astype(np.float32)
    E = mean_ext[:, [0, -1]].astype(np.float32)
    assert np.array_equal((~np.isnan(X)).sum(axis=1), hh["n_valid_2013"].to_numpy())
    np.save(P["KWH_NPY"], X)
    np.save(P["EDGES_NPY"], E)
    hh.to_parquet(P["HOUSEHOLDS_PARQUET"], index=False)

    # ---------------------------------------------------------------- counts per group
    thr = args.threshold
    comp = hh["completeness"].to_numpy()
    # completeness in the authors' frame for a +1 slot mapping (LCL index 1 .. 17520)
    valid_ext = ~np.isnan(mean_ext)
    comp_shift_p1 = valid_ext[:, 2:].sum(axis=1) / L.N_SLOTS
    comp_shift_m1 = valid_ext[:, :-2].sum(axis=1) / L.N_SLOTS
    counts = {
        "households_in_window": group_counts(hh, np.ones(len(hh), bool)),
        "households_with_any_2013_row": group_counts(hh, hh["n_rows_2013"].to_numpy() > 0),
        "households_with_any_valid_2013_value": group_counts(hh, hh["n_valid_2013"].to_numpy() > 0),
        "completeness_gt_0.95 (n_valid/17520 > 0.95; main)": group_counts(hh, comp > thr),
        "completeness_ge_0.95": group_counts(hh, comp >= thr),
        "completeness_rows_gt_0.95 (Null rows counted as present)": group_counts(hh, hh["completeness_rows"].to_numpy() > thr),
        "raw_rows_incl_duplicates_gt_0.95": group_counts(hh, hh["n_raw_rows_2013"].to_numpy() / L.N_SLOTS > thr),
        "completeness_gt_0.95_on_LCL_index_1..17520 (shift +1 frame)": group_counts(hh, comp_shift_p1 > thr),
        "completeness_gt_0.95_on_LCL_index_-1..17518 (shift -1 frame)": group_counts(hh, comp_shift_m1 > thr),
        "completeness_gt_0.90": group_counts(hh, comp > 0.90),
        "completeness_gt_0.99": group_counts(hh, comp > 0.99),
        "ToU_completeness_in_(0.94,0.96]": int(((hh["group"] == "ToU") & (comp > 0.94) & (comp <= 0.96)).sum()),
        "ToU_completeness_in_(0.95,0.951]": int(((hh["group"] == "ToU") & (comp > 0.95) & (comp <= 0.951)).sum()),
        "ToU_completeness_in_(0.949,0.95]": int(((hh["group"] == "ToU") & (comp > 0.949) & (comp <= 0.95)).sum()),
    }
    for kk, vv in counts.items():
        print(f"           {kk}: {vv}", flush=True)

    build_stats = {"csv": str(csv), "full_parquet_scratch": str(full_pq), **st_full, **st_win, **st_dup,
                   "long_rows_written": n_long, "dense_shape": list(X.shape),
                   "missing_share_2013_all_households": float(np.isnan(X).mean()),
                   "counts": counts,
                   "dup_detail_by_timestamp": {str(t): {"groups": int(len(g)),
                                                        "conflicting": int((g["vmax"] > g["vmin"]).sum()),
                                                        "mixed_null": int(((g["n_valid"] > 0) & (g["n_valid"] < g["n_rows"])).sum())}
                                               for t, g in dup_det.groupby("ts")} if len(dup_det) < 200000 else "too many"}
    L.save_json(build_stats, P["BUILD_STATS_JSON"])
    del mean_ext, cnt_ext, valid_ext

    # ---------------------------------------------------------------- 2 selection + fill
    sel = L.select_households(hh, threshold=thr, seed=args.seed)
    n_tou = int((sel["group"] == "ToU").sum())
    n_std = int((sel["group"] == "Std").sum())
    print(f"[{el()}] selection: {n_tou} ToU + {n_std} Std (Std pool "
          f"{int(((hh['group'] == 'Std') & (comp > thr)).sum())}, seed {args.seed})", flush=True)
    rows = sel["row_in_dense"].to_numpy()
    Xs = X[rows]
    filled, fst = L.fill_gaps(Xs, short_max=args.short_max, long_rule=args.long_rule)
    Es = L.fill_edges(E[rows], filled)
    assert not np.isnan(filled).any(), "unfilled values left"
    assert not np.isnan(Es).any()
    obs = ~np.isnan(Xs)
    assert np.array_equal(filled[obs], Xs[obs]), "observed values were modified"

    # which of our ToU households are the authors' 1,007? (exact NNLS identification against
    # their all-ToU aggregate, see lib.lcl_data.identify_authors_subset)
    sel["in_authors_tou_1007"] = False
    sel["in_1007_pair"] = False
    try:
        _, auth, _ = L.load_authors_tou_aggregate()
        _, n_auth_sizes = L.load_authors_clusters()
        n_auth = int(n_auth_sizes.sum())
        s_best, s_scores = L.best_slot_shift(filled[:n_tou].astype(np.float64), Es[:n_tou].astype(np.float64), auth)
        Xf = L.to_authors_frame(Xs[:n_tou].astype(np.float64), E[rows][:n_tou].astype(np.float64), s_best)
        w, ident = L.identify_authors_subset(Xf, auth, n_auth)
        ident.update({"slot_shift_used": s_best, "slot_shift_mean_abs_diff": s_scores,
                      "excluded_LCLids": sel["LCLid"].to_numpy(dtype=object)[:n_tou][w > 0.5].tolist()})
        sel.loc[np.flatnonzero(w <= 0.5), "in_authors_tou_1007"] = True
        sel["in_1007_pair"] = sel["in_authors_tou_1007"] | ((sel["group"] == "Std") & (sel["std_draw_order"] >= 0)
                                                          & (sel["std_draw_order"] < n_auth))
        print(f"[{el()}] authors' ToU set: {ident['subset_size']} of our {n_tou} (NNLS weights 0/1: "
              f"{ident['n_weight_in_(0.01,0.99)'] == 0}; subset mean vs authors max |diff| "
              f"{ident['subset_mean_vs_authors_max_abs_diff_kwh_fully_observed']:.2e} kWh on "
              f"{ident['fully_observed_slots_used']} fully observed slots; shift {s_best:+d})", flush=True)
    except FileNotFoundError as exc:          # authors' files not available
        ident = {"error": str(exc)}
    np.save(P["FILLED_NPY"], filled.astype(np.float32))
    np.save(P["EDGES_FILLED_NPY"], Es.astype(np.float32))
    sel.to_parquet(P["SELECTION_PARQUET"], index=False)
    fst["authors_subset_identification"] = ident
    fst.update({
        "n_selected": int(len(sel)), "n_tou": n_tou, "n_std": n_std, "seed": args.seed, "threshold": thr,
        "missing_share_selected": float((~obs).mean()),
        "missing_share_selected_tou": float((~obs[:n_tou]).mean()),
        "missing_share_selected_std": float((~obs[n_tou:]).mean()),
        "edges_missing_before_fill": {"2012-12-31 23:30": int(np.isnan(E[rows, 0]).sum()),
                                      "2014-01-01 00:00": int(np.isnan(E[rows, 1]).sum())},
        "mean_kwh_selected_tou": float(filled[:n_tou].mean()),
        "mean_kwh_selected_std": float(filled[n_tou:].mean()),
    })
    L.save_json(fst, P["FILL_STATS_JSON"])
    print(f"[{el()}] filled: {fst['filled_short_linear']:,} short-gap slots (linear), "
          f"{fst['filled_long_sameslot_days']:,} long-gap slots (same slot, nearest days; one side only "
          f"{fst['filled_long_one_side_only']:,}), left {fst['left_missing']}; missing share selected "
          f"{fst['missing_share_selected']:.4%}", flush=True)
    print(f"[{el()}] done", flush=True)


if __name__ == "__main__":
    main()
