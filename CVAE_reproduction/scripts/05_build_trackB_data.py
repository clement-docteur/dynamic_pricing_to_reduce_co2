"""Track B datasets in the authors' schema from our run-(a) ToU clusters (scripts/04_clustering.py).

1. Relabel our 4 NMF clusters (data/clusters_run_a.parquet, ToU rows) to best match the authors'
   clusters 0-3: Hungarian assignment maximising the summed Pearson correlation between our and
   their 365 x 48 cluster-mean series (flattened).  Diagnostics: the full correlation matrix, the
   same on the series' deviations from the respective all-ToU mean, and the contingency table
   against the recovered authors' partition on the 1,007 shared households.  The match counts as
   good only if the three assignments agree and every matched raw correlation is the best in
   its row and column; otherwise the clusters are ordered by size (descending) and the JSON
   says so.
2. data/power_consump_ours.parquet (365 days x 4 clusters, rows sorted by cluster then date):
   date (datetime64), consumption_00..47 (cluster-mean kWh, authors' slot convention), cluster
   (int64), price_00..47, temp_00..47 (ERA5), low_00..47, normal_00..47, high_00..47 (Tariffs.xlsx
   'loc' version, no suffix), temp_smooth, pos_y, work_day, pca_temp_0..2, temp_smooth_n,
   n_households.
3. data/train_ours.parquet / data/test_ours.parquet: the authors' exact train / test dates
   (train_nmf / test_nmf), rows sorted by cluster then date.
4. data/train_ToU_ours.parquet / data/test_ToU_ours.parquet: the all-1,032 ToU aggregate
   (cluster = 0, n_households = 1,032), same schema and split.
5. Schema checks through lib.cvae_torch (COLS_DATA, COLS_COND, _read_table, cluster_rows, _xy)
   and lib.gam_generator.frame_arrays; results under key "trackB_data" of
   outputs/clustering_results.json.

Run:  python -P scripts/05_build_trackB_data.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))

from lib import clustering as C  # noqa: E402
from lib import lcl_data as L  # noqa: E402
from lib.tariffs import cols  # noqa: E402

DATA = REPRO / "data"
OUT = REPRO / "outputs"
RESULTS_JSON = OUT / "clustering_results.json"
H = 48
AUTHOR_ORDER = (["date"] + cols("consumption") + ["cluster"] + cols("price") + cols("temp") + cols("low")
                + cols("normal") + cols("high")
                + ["temp_smooth", "pos_y", "work_day", "pca_temp_0", "pca_temp_1", "pca_temp_2", "temp_smooth_n"])


def save_results(key, value):
    res = json.loads(RESULTS_JSON.read_text(encoding="utf-8")) if RESULTS_JSON.exists() else {}
    res[key] = value
    L.save_json(res, RESULTS_JSON)


def authors_split():
    tr = pd.read_feather(L.UP_DATA / "train_nmf.feather")
    te = pd.read_feather(L.UP_DATA / "test_nmf.feather")
    out = {}
    for name, df in (("train", tr), ("test", te)):
        df["date"] = pd.to_datetime(df["date"])
        per = [set(df.loc[df["cluster"].astype(int) == c, "date"]) for c in range(4)]
        if not all(p == per[0] for p in per):
            raise RuntimeError(f"{name} dates differ between the authors' clusters")
        out[name] = pd.DatetimeIndex(sorted(per[0]))
    if len(out["train"]) != 273 or len(out["test"]) != 92:
        raise RuntimeError("unexpected split sizes")
    if set(out["train"]) & set(out["test"]) or len(set(out["train"]) | set(out["test"])) != 365:
        raise RuntimeError("train/test dates overlap or do not cover the year")
    return out


def relabel(Y, labels, auth_lab_1007, in1007, cons_auth):
    """Mapping our label -> authors' label (or size order if the match is poor) + diagnostics."""
    from scipy.optimize import linear_sum_assignment

    ours = C.cluster_means(Y, labels)                        # (4, 365, 48)
    m_raw, C_raw = C.relabel_hungarian(ours, cons_auth)
    # deviations from the respective all-ToU means
    all_ours = Y.mean(axis=0)
    sizes_auth = np.array([325, 295, 229, 158])
    all_auth = np.tensordot(sizes_auth / sizes_auth.sum(), cons_auth, axes=1)
    m_dev, C_dev = C.relabel_hungarian(ours - all_ours, cons_auth - all_auth)
    # contingency on the shared 1,007 households
    T = C.contingency(labels[in1007], auth_lab_1007)
    r, c = linear_sum_assignment(-T)
    m_cont = {int(i): int(j) for i, j in zip(r, c)}
    mutual_best = all(C_raw[i, j] == C_raw[i].max() and C_raw[i, j] == C_raw[:, j].max() for i, j in m_raw.items())
    good = (m_raw == m_dev == m_cont) and mutual_best
    # second-best assignment gap (raw correlations)
    import itertools
    tot = sorted(((sum(C_raw[i, p[i]] for i in range(4)), p) for p in itertools.permutations(range(4))), reverse=True)
    diag = {
        "corr_raw_ours_rows_authors_cols": np.round(C_raw, 6).tolist(),
        "corr_deviation_from_all_mean": np.round(C_dev, 4).tolist(),
        "hungarian_raw": m_raw, "hungarian_deviation": m_dev, "max_overlap_contingency": m_cont,
        "matched_corr_raw": [float(C_raw[i, m_raw[i]]) for i in range(4)],
        "matched_corr_deviation": [float(C_dev[i, m_raw[i]]) for i in range(4)],
        "best_vs_second_best_assignment_sum_corr": [float(tot[0][0]), float(tot[1][0])],
        "contingency_ours_rows_authors_cols_1007": T.tolist(),
        "mutual_best": bool(mutual_best), "match_good": bool(good),
    }
    if good:
        mapping = m_raw
        diag["relabel_rule"] = "Hungarian on raw cluster-mean correlations (agrees with deviation and contingency matching)"
    else:
        _, mapping = C.relabel_by_size(labels)
        diag["relabel_rule"] = ("POOR MATCH: assignments disagree or are not mutual-best -> clusters ordered "
                                "by size, descending")
    diag["mapping_ours_to_final"] = mapping
    diag["ari_vs_authors_1007"] = C.ari(labels[in1007], auth_lab_1007)
    return mapping, diag


def build_frame(means_by_cluster, n_by_cluster, cond):
    """Rows (cluster, date) in the authors' schema from cluster means (k, 365, 48)."""
    parts = []
    normal = 1.0 - cond[cols("low")].to_numpy(np.float64) - cond[cols("high")].to_numpy(np.float64)
    for c, (m, n) in enumerate(zip(means_by_cluster, n_by_cluster)):
        data = {"date": pd.to_datetime(cond["date"]).to_numpy()}
        data.update({name: m[:, h] for h, name in enumerate(cols("consumption"))})
        data["cluster"] = np.full(len(m), c, dtype=np.int64)
        for name in (cols("price") + cols("temp") + cols("low") + cols("high")
                     + ["temp_smooth", "pos_y", "work_day", "pca_temp_0", "pca_temp_1", "pca_temp_2", "temp_smooth_n"]):
            data[name] = cond[name].to_numpy(np.float64)
        data.update({name: normal[:, h] for h, name in enumerate(cols("normal"))})
        data["n_households"] = np.full(len(m), n, dtype=np.int64)
        parts.append(pd.DataFrame(data)[AUTHOR_ORDER + ["n_households"]])
    out = pd.concat(parts, ignore_index=True)
    out["date"] = out["date"].astype("datetime64[ns]")
    return out.sort_values(["cluster", "date"], kind="stable").reset_index(drop=True)


def schema_checks(paths):
    import torch

    torch.set_num_threads(1)
    from lib import cvae_torch as CV
    from lib import gam_generator as G

    res = {}
    for name, p in paths.items():
        df = CV._read_table(str(p))
        e = {"rows": int(len(df)), "clusters": sorted(int(v) for v in df["cluster"].unique()),
             "date_dtype": str(df["date"].dtype), "cluster_dtype": str(pd.read_parquet(p)["cluster"].dtype),
             "missing_cols_COLS_DATA": [c for c in CV.COLS_DATA if c not in df.columns],
             "missing_cols_COLS_COND": [c for c in CV.COLS_COND if c not in df.columns],
             "n_cond": len(CV.COLS_COND)}
        per = []
        for c in e["clusters"]:
            sub = CV.cluster_rows(df, c)
            Yc, Cc = CV._xy(sub, CV.COLS_COND)
            a = G.frame_arrays(sub)
            per.append({"cluster": c, "Y": list(Yc.shape), "cond": list(Cc.shape),
                        "dates_ascending": bool(pd.Series(sub["date"]).is_monotonic_increasing),
                        "gam_frame_Y": list(a["Y"].shape), "gam_frame_temp": list(a["temp"].shape)})
        e["per_cluster"] = per
        res[name] = e
    return res


def main():
    hs = C.household_sets()
    dfc = pd.read_parquet(DATA / "clusters_run_a.parquet")
    sub = dfc[dfc["group"] == "ToU"].reset_index(drop=True)
    tou = hs["tou1032"]
    if not np.array_equal(sub["LCLid"].to_numpy(dtype=object), tou["LCLid"].astype(str).to_numpy(dtype=object)):
        raise RuntimeError("clusters_run_a ToU rows not aligned with tou1032")
    Y = C.load_records(sub["row_in_filled"])                          # (1032, 365, 48)
    labels = sub["nmf"].to_numpy(np.int64)
    cons_auth, sizes_auth = C.authors_cluster_means()
    rec = pd.read_parquet(DATA / "authors_clusters_recovered.parquet")
    in1007 = sub["LCLid"].isin(rec["LCLid"]).to_numpy()
    auth_lab = rec.set_index("LCLid").loc[sub.loc[in1007, "LCLid"], "cluster"].to_numpy(np.int64)
    mapping, diag = relabel(Y, labels, auth_lab, in1007, cons_auth)
    final = np.vectorize(mapping.get)(labels).astype(np.int64)
    print("relabel:", diag["relabel_rule"], "mapping", mapping)
    print("matched raw correlations", np.round(diag["matched_corr_raw"], 5), "deviation corr",
          np.round(diag["matched_corr_deviation"], 3))

    cond = C.era5_conditions("loc")
    assert (pd.to_datetime(cond["date"]).to_numpy() == L.DATES_2013.to_numpy()).all()
    means = C.cluster_means(Y, final)
    n_by = np.bincount(final, minlength=4)
    pc = build_frame(means, n_by, cond)
    split = authors_split()
    paths = {
        "power_consump_ours": DATA / "power_consump_ours.parquet",
        "train_ours": DATA / "train_ours.parquet", "test_ours": DATA / "test_ours.parquet",
        "train_ToU_ours": DATA / "train_ToU_ours.parquet", "test_ToU_ours": DATA / "test_ToU_ours.parquet",
    }
    pc.to_parquet(paths["power_consump_ours"], index=False)
    pc[pc["date"].isin(split["train"])].reset_index(drop=True).to_parquet(paths["train_ours"], index=False)
    pc[pc["date"].isin(split["test"])].reset_index(drop=True).to_parquet(paths["test_ours"], index=False)
    agg = build_frame(Y.mean(axis=0)[None], [len(Y)], cond)
    agg[agg["date"].isin(split["train"])].reset_index(drop=True).to_parquet(paths["train_ToU_ours"], index=False)
    agg[agg["date"].isin(split["test"])].reset_index(drop=True).to_parquet(paths["test_ToU_ours"], index=False)

    # ---------------------------------------------------------------- independent checks
    checks = {}
    pc2 = pd.read_parquet(paths["power_consump_ours"])
    # (i) cluster means recomputed from the member households, row by row
    errs = []
    for c in range(4):
        rows = pc2[pc2["cluster"] == c].sort_values("date")
        mem = sub.loc[final == c, "row_in_filled"].to_numpy()
        ref = C.load_records(mem).mean(axis=0)
        errs.append(float(np.abs(rows[cols("consumption")].to_numpy() - ref).max()))
    checks["consumption_vs_member_mean_max_abs"] = max(errs)
    # (ii) size-weighted cluster mean == all-1,032 aggregate
    W_ = np.stack([pc2[pc2["cluster"] == c].sort_values("date")[cols("consumption")].to_numpy() for c in range(4)])
    wmean = np.tensordot(n_by / n_by.sum(), W_, axes=1)
    tr_a = pd.read_parquet(paths["train_ToU_ours"])
    te_a = pd.read_parquet(paths["test_ToU_ours"])
    all_a = pd.concat([tr_a, te_a]).sort_values("date")[cols("consumption")].to_numpy()
    checks["size_weighted_cluster_mean_vs_ToU_aggregate_max_abs"] = float(np.abs(wmean - all_a).max())
    # (iii) conditions identical to conditions_ours ('loc' columns) and to each other across clusters
    raw = pd.read_parquet(L.DATA / "conditions_ours.parquet").sort_values("date").reset_index(drop=True)
    ok = True
    for c in range(4):
        rows = pc2[pc2["cluster"] == c].sort_values("date").reset_index(drop=True)
        for k in ("low", "high", "price"):
            ok &= np.array_equal(rows[cols(k)].to_numpy(), raw[[f"{k}_loc_{h:02d}" for h in range(H)]].to_numpy())
        ok &= np.array_equal(rows[cols("temp")].to_numpy(), raw[cols("temp")].to_numpy())
        ok &= np.allclose(rows[cols("normal")].to_numpy() + rows[cols("low")].to_numpy() + rows[cols("high")].to_numpy(), 1.0)
    checks["conditions_equal_conditions_ours_loc"] = bool(ok)
    # (iv) split: same dates as the authors', sorted by cluster then date
    tr = pd.read_parquet(paths["train_ours"])
    te = pd.read_parquet(paths["test_ours"])
    checks["train_rows"], checks["test_rows"] = int(len(tr)), int(len(te))
    checks["train_dates_equal_authors"] = bool(set(tr["date"]) == set(split["train"]))
    checks["test_dates_equal_authors"] = bool(set(te["date"]) == set(split["test"]))
    checks["sorted_cluster_then_date"] = bool(all(
        d.equals(d.sort_values(["cluster", "date"], kind="stable").reset_index(drop=True)) for d in (tr, te, pc2)))
    checks["columns_match_authors_schema_plus_n_households"] = bool(list(pc2.columns) == AUTHOR_ORDER + ["n_households"])
    up_cols = list(pd.read_feather(L.UP_DATA / "power_consump_nmf.feather").columns)
    checks["authors_power_consump_columns_identical_set"] = bool(set(up_cols) == set(AUTHOR_ORDER))
    checks["schema_via_lib"] = schema_checks(paths)
    # (v) comparison of our cluster series with the authors' matched clusters (levels)
    lev = {"ours_mean_kwh": W_.mean(axis=(1, 2)).round(4).tolist(), "n_households": n_by.tolist(),
           "authors_mean_kwh": cons_auth.mean(axis=(1, 2)).round(4).tolist(),
           "authors_sizes": sizes_auth.tolist(),
           "corr_raw_series_final_label_vs_same_authors_label": [
               float(np.corrcoef(W_[c].ravel(), cons_auth[c].ravel())[0, 1]) for c in range(4)]}
    print(json.dumps(L.to_jsonable({k: v for k, v in checks.items() if k != "schema_via_lib"}), indent=1))
    print("levels", lev)
    save_results("trackB_data", {"relabel": diag, "checks": checks, "levels": lev,
                                 "files": {k: str(v) for k, v in paths.items()}})


if __name__ == "__main__":
    main()
