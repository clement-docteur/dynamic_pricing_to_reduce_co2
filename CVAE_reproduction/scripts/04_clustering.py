"""Household clustering (Bregere & Bessa 2020, Sec. IV, App. B-C, Table 3, Fig. 2) on our data,
forensic recovery of the authors' ToU partition, and the Calinski-Harabasz definitions of Table 3.

Stages (all by default, in this order; each reads what the previous ones wrote):
  recover   NNLS recovery of the authors' 4 ToU clusters from power_consump_nmf.feather on the
            identified 1,007 households -> data/authors_clusters_recovered.parquet
  profiles  household causality profiles mu_i^h(p), sigma_i^h(p) (lib.causality, LAML, fit and
            eq.-(1) average over all 365 days):
              era5_loc : our 1,032 ToU + 1,032 Std, ERA5 temps, Tariffs.xlsx 'loc' schedule
              auth     : identified 1,007 ToU + in_1007_pair Std, the authors' temps + dummies
            -> data/causality_profiles_era5_loc.npz, data/causality_profiles_auth.npz
  cluster   runs (a) PRIMARY 1,032+1,032 ERA5/'loc'; (b) PAPER-SIZED 1,007+1,007 ERA5/'loc';
            (c) CLOSEST-TO-AUTHORS 1,007+1,007 authors' temps/'auth': M -> NMF r=5 -> k-medoids
            k=4 (+ PAM), random (20 seeds), classical features; comparison with the recovered
            partition; stability (runs, NMF inits, k-medoids variants); forensic variants
            -> data/clusters_run_{a,b,c}.parquet
  table3    Calinski-Harabasz candidate definitions on the recovered partition (which one gives
            the paper's 13,432 / 14,088 / 16,070?) and the full Table 3 for our clusterings
  fig2      average week (Sun-Sat) of cluster-mean consumption, raw and standardized, for
            random / features / NMF (run a, ToU) -> outputs/figures/fig2_clusters.png
            (+ fig2_recovered_vs_ours.png: authors' recovered partition vs our runs)
All numbers go to outputs/clustering_results.json (one key per stage).

Run:  python -P scripts/04_clustering.py [--stages recover,profiles,cluster,table3,fig2]
The upstream repo is only read (feather files by absolute path); nothing in it is executed.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")  # spec: at most 4 threads

import argparse
import json
import sys
import time
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
FIGS = OUT / "figures"
RESULTS_JSON = OUT / "clustering_results.json"
RECOVERED_PARQUET = DATA / "authors_clusters_recovered.parquet"
PROFILES = {"era5_loc": DATA / "causality_profiles_era5_loc.npz",
            "auth": DATA / "causality_profiles_auth.npz"}
RUNS = {  # run -> (profile source, ToU set, Std set, tariff schedule for the 'special' mask)
    "a": dict(profiles="era5_loc", tou="tou1032", std="std1032", tariff="loc",
              label="PRIMARY: our 1,032 ToU + 1,032 Std, ERA5 temperatures, Tariffs.xlsx 'loc'"),
    "b": dict(profiles="era5_loc", tou="tou1007", std="std1007", tariff="loc",
              label="PAPER-SIZED: identified 1,007 ToU + in_1007_pair Std, ERA5, 'loc'"),
    "c": dict(profiles="auth", tou="tou1007", std="std1007", tariff="auth",
              label="CLOSEST-TO-AUTHORS: identified 1,007 ToU (+1,007 Std), authors' temps + dummies"),
}
N_RANDOM = 20
PAPER = {
    "sizes_nmf_tou": [325, 295, 229, 158],
    "sizes_features_tou": [255, 247, 249, 256],
    "sizes_random_tou": [248, 264, 242, 253],
    "table3": {  # rows raw / std / special; columns as in the paper
        "Rd ToU": [4627, 4950, 5151], "Features ToU": [40764, 6870, 7033], "NMF ToU": [13432, 14088, 16070],
        "Rd Std": [5014, 4834, 4904], "Features Std": [5088, 4741, 4694], "NMF Std": [12971, 14934, 15460],
    },
}
SERIES = ("raw", "std", "special")
LOG_T0 = time.time()


def log(msg):
    print(f"[{time.time() - LOG_T0:7.1f}s] {msg}", flush=True)


def load_results() -> dict:
    if RESULTS_JSON.exists():
        return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    return {}


def save_results(key, value):
    res = load_results()
    res[key] = value
    L.save_json(res, RESULTS_JSON)


# ===================================================================================== recover
def stage_recover():
    hs = C.household_sets()
    t = hs["tou1007"]
    A = C.load_records(t["row_in_filled"], as_days=False)
    cons, sizes = C.authors_cluster_means()
    Ym = cons.reshape(len(cons), -1)
    log("NNLS recovery of the authors' partition (4 x NNLS on 17,520 x 1,007) ...")
    labels, W, diag = C.recover_partition(A, Ym, sizes)
    log(f"  sizes {diag['sizes_recovered']} (expected {diag['sizes_expected']}), disjoint "
        f"{diag['disjoint']}, union {diag['union_is_all']}")
    if not (diag["sizes_match"] and diag["disjoint"] and diag["union_is_all"]):
        raise RuntimeError("partition recovery failed")
    # independent checks: (i) Gram/Cholesky NNLS on even slots, held-out odd slots
    ev = np.arange(A.shape[1]) % 2 == 0
    lab_g, _, diag_g = C.recover_partition_gram(A, Ym, sizes, fit_slots=ev)
    # (ii) unconstrained least squares on all slots
    X = np.linalg.lstsq(A.T, Ym.T, rcond=None)[0].T * sizes[:, None]
    lab_ls = np.argmax(X, axis=0)
    checks = {
        "gram_nnls_even_slots_same_partition": bool(np.array_equal(lab_g, labels)),
        "gram_nnls_heldout_odd_slots": diag_g.get("heldout"),
        "lstsq_same_partition": bool(np.array_equal(lab_ls, labels)),
        "lstsq_max_abs_n_times_w_minus_1_members": float(max(np.abs(X[c][lab_ls == c] - 1).max() for c in range(4))),
        "lstsq_max_abs_n_times_w_nonmembers": float(max(np.abs(X[c][lab_ls != c]).max() for c in range(4))),
    }
    log(f"  cross-checks: {checks['gram_nnls_even_slots_same_partition']}, {checks['lstsq_same_partition']}")
    nw = (W * sizes[:, None])[labels, np.arange(len(labels))]
    out = pd.DataFrame({"LCLid": t["LCLid"].to_numpy(dtype=object), "cluster": labels.astype(np.int64),
                        "row_in_filled": t["row_in_filled"].to_numpy(np.int64), "n_c_times_weight": nw})
    out["LCLid"] = out["LCLid"].astype(str)
    out.to_parquet(RECOVERED_PARQUET, index=False)
    log(f"  wrote {RECOVERED_PARQUET}")
    save_results("authors_partition", {"nnls": diag, "cross_checks": checks,
                                       "file": str(RECOVERED_PARQUET)})


def load_recovered(hs=None):
    hs = C.household_sets() if hs is None else hs
    rec = pd.read_parquet(RECOVERED_PARQUET)
    t = hs["tou1007"]
    if not np.array_equal(rec["LCLid"].to_numpy(dtype=object), t["LCLid"].to_numpy(dtype=object)):
        raise RuntimeError("recovered partition not aligned with tou1007")
    return rec["cluster"].to_numpy(np.int64)


# ==================================================================================== profiles
def _save_profiles(path, res, sets_df, extra):
    np.savez_compressed(
        path, mu=res["mu"], sigma=res["sigma"], xi=res["xi"], g=res["g"], edf=res["edf"],
        on_grid_edge=res["on_grid_edge"], converged=res["converged"], counts=res["counts"],
        LCLid=sets_df["LCLid"].to_numpy(dtype=str), group=sets_df["group"].to_numpy(dtype=str),
        row_in_filled=sets_df["row_in_filled"].to_numpy(np.int64),
        settings=json.dumps(L.to_jsonable({**res["settings"], **extra})))


def _profile_diag(res):
    return {
        "n_households": int(res["mu"].shape[0]),
        "nan_mu": int(np.isnan(res["mu"]).sum()), "nan_sigma": int(np.isnan(res["sigma"]).sum()),
        "share_converged": float(np.mean(res["converged"])),
        "share_lambda_on_grid_edge": float(np.mean(res["on_grid_edge"])),
        "share_lambda_at_upper_edge_(linear_temperature_effect)": float(np.mean(res["g"] == np.max(res["g"]))),
        "median_edf": float(np.median(res["edf"])),
        "min_tariff_count_per_slot": {p: int(res["counts"][:, j].min()) for j, p in enumerate(("Low", "Normal", "High"))},
        "negative_mu_entries": int((res["mu"] < 0).sum()),
    }


def stage_profiles(force=False):
    hs = C.household_sets()
    diags = {}
    # --- ERA5 + 'loc' for all 2,064 households
    path = PROFILES["era5_loc"]
    if force or not path.exists():
        cond = C.era5_conditions("loc")
        assert (pd.to_datetime(cond["date"]).to_numpy() == L.DATES_2013.to_numpy()).all()
        temp, low, high = C.condition_arrays(cond)
        sets_df = pd.concat([hs["tou1032"], hs["std1032"]], ignore_index=True)
        Y = C.load_records(sets_df["row_in_filled"])
        log(f"causality profiles era5_loc: {len(sets_df)} households ...")
        res = C.causality_profiles(Y, temp, low, high, log=log)
        _save_profiles(path, res, sets_df, {"temps": "ERA5 (conditions_ours temp_hh)", "tariffs": "loc"})
        diags["era5_loc"] = _profile_diag(res)
        del Y, res
    # --- authors' temps + authors' dummies for the 1,007 + 1,007
    path = PROFILES["auth"]
    if force or not path.exists():
        af = C.authors_dummies_frame()
        assert (pd.to_datetime(af["date"]).to_numpy() == L.DATES_2013.to_numpy()).all()
        temp = C.authors_temps()
        low = af[cols("low")].to_numpy(np.float64)
        high = af[cols("high")].to_numpy(np.float64)
        sets_df = pd.concat([hs["tou1007"], hs["std1007"]], ignore_index=True)
        Y = C.load_records(sets_df["row_in_filled"])
        log(f"causality profiles auth: {len(sets_df)} households ...")
        res = C.causality_profiles(Y, temp, low, high, log=log)
        _save_profiles(path, res, sets_df, {"temps": "authors' temp_hh (power_consump_nmf)", "tariffs": "auth"})
        diags["auth"] = _profile_diag(res)
        del Y, res
    if diags:
        old = load_results().get("profiles", {})
        old.update(diags)
        save_results("profiles", old)


def load_profiles(name):
    z = np.load(PROFILES[name], allow_pickle=False)
    return {k: z[k] for k in z.files}


def profiles_for(name, set_df):
    """mu, sigma of the households of ``set_df`` (aligned by LCLid) from a profile file."""
    P = load_profiles(name)
    pos = pd.Index(P["LCLid"]).get_indexer(set_df["LCLid"].to_numpy(dtype=str))
    if (pos < 0).any():
        raise RuntimeError(f"households missing from profiles {name}")
    return P["mu"][pos], P["sigma"][pos], P


# ===================================================================================== cluster
def nmf_kmedoids(M, *, init="nndsvd", random_state=0, max_iter=2000, km="alternate", km_init="heuristic",
                 km_seed=None, tol=1e-4):
    W, Hm, info = C.nmf_features(M, init=init, random_state=random_state, max_iter=max_iter, tol=tol)
    D = C.pairwise_euclidean(W)
    if km == "alternate":
        r = C.kmedoids_alternate(D=D, init=km_init, random_state=km_seed)
    elif km == "pam":
        r = C.kmedoids_pam(D=D)
    else:
        raise ValueError(km)
    return r["labels"], W, Hm, info, r


def sizes_of(labels, k=C.K):
    return [int(v) for v in np.bincount(np.asarray(labels), minlength=k)]


def compare_to_authors(labels, auth):
    T = C.contingency(labels, auth)
    return {"ari": C.ari(labels, auth), "contingency_rows_ours_cols_authors": T.tolist(),
            "sizes_ours": sizes_of(labels), "sizes_authors": sizes_of(auth),
            "share_households_in_best_matching_cluster": float(_best_overlap(T) / T.sum())}


def _best_overlap(T):
    from scipy.optimize import linear_sum_assignment

    r, c = linear_sum_assignment(-T)
    return T[r, c].sum()


def stage_cluster():
    hs = C.household_sets()
    auth = load_recovered(hs)
    dates = L.DATES_2013
    results = {"settings": {
        "profile_matrix": "M = [mu(Low)|mu(Normal)|mu(High)] / mean_h mu(Normal); negatives clipped to 0",
        "nmf": "sklearn NMF r=5, solver cd, init nndsvd, frobenius, tol 1e-4, max_iter 2000, random_state 0, no regularisation",
        "kmedoids": "sklearn_extra 0.1/0.2 KMedoids port: method alternate, init heuristic, Euclidean, max_iter 300",
        "random": f"numpy default_rng(seed).integers(0, 4, n), seeds 0..{N_RANDOM - 1}",
        "features": ("min / max / mean of the half-hourly records in winter (Jan-Mar, Oct-Dec) and in "
                     "summer (Apr-Sep), peak_hh / off_hh = mean over the 365 days of the argmax / argmin "
                     "half-hour (0..47); each feature min-max rescaled to [0, 1]; k-medoids as above"),
    }, "runs": {}}
    W_store = {}
    for run, spec in RUNS.items():
        rr = {"label": spec["label"]}
        for grp in ("ToU", "Std"):
            set_df = hs[spec["tou"] if grp == "ToU" else spec["std"]]
            mu, sigma, _ = profiles_for(spec["profiles"], set_df)
            M, minfo = C.profile_matrix(mu)
            lab_nmf, W, Hm, ninfo, km = nmf_kmedoids(M)
            D = C.pairwise_euclidean(W)
            pam = C.kmedoids_pam(D=D)
            Y = C.load_records(set_df["row_in_filled"])
            F = C.classical_features(Y, dates, "records")
            Fs = C.rescale_features(F.to_numpy(), "minmax")
            fk = C.kmedoids_alternate(Fs)
            rnd = [C.random_labels(len(set_df), seed=s) for s in range(N_RANDOM)]
            g = {
                "n_households": int(len(set_df)), "profile_matrix": minfo, "nmf": ninfo,
                "kmedoids_alternate": {"sizes": sizes_of(lab_nmf), "cost": km["cost"], "n_iter": km["n_iter"],
                                       "converged": km["converged"], "medoids_row": [int(v) for v in km["medoids"]]},
                "kmedoids_pam": {"sizes": sizes_of(pam["labels"]), "cost": pam["cost"], "n_iter": pam["n_iter"],
                                 "ari_vs_alternate": C.ari(pam["labels"], lab_nmf)},
                "features": {"sizes": sizes_of(fk["labels"]), "cost": fk["cost"], "converged": fk["converged"],
                             "feature_summary": F.describe().loc[["mean", "std", "min", "max"]].round(4).to_dict()},
                "random_sizes_seed0": sizes_of(rnd[0]),
            }
            # feature-definition sensitivity
            sens = {}
            for variant in ("records", "profile", "daily"):
                Fv = C.classical_features(Y, dates, variant).to_numpy()
                for how in ("minmax", "standard"):
                    lv = C.kmedoids_alternate(C.rescale_features(Fv, how))["labels"]
                    sens[f"{variant}/{how}"] = {"sizes": sizes_of(lv), "ari_vs_primary": C.ari(lv, fk["labels"])}
            g["features_sensitivity"] = sens
            df = pd.DataFrame({"LCLid": set_df["LCLid"].astype(str).to_numpy(), "group": grp,
                               "row_in_filled": set_df["row_in_filled"].to_numpy(np.int64),
                               "nmf": lab_nmf.astype(np.int64), "nmf_pam": pam["labels"].astype(np.int64),
                               "features": fk["labels"].astype(np.int64)})
            for s in range(N_RANDOM):
                df[f"random_{s:02d}"] = rnd[s].astype(np.int64)
            for j in range(W.shape[1]):
                df[f"W{j}"] = W[:, j]
            df["mubar_kwh"] = mu[:, :, 1].mean(axis=1)
            for col in F.columns:
                df[f"feat_{col}"] = F[col].to_numpy()
            if grp == "ToU":
                in_auth = set_df["in_authors_tou_1007"].to_numpy(bool)
                pos = pd.Index(hs["tou1007"]["LCLid"].astype(str)).get_indexer(df["LCLid"])
                df["authors_cluster"] = np.where(pos >= 0, auth[np.maximum(pos, 0)], -1).astype(np.int64)
                sub = df[in_auth]
                g["vs_authors"] = {
                    "nmf": compare_to_authors(sub["nmf"].to_numpy(), sub["authors_cluster"].to_numpy()),
                    "nmf_pam": compare_to_authors(sub["nmf_pam"].to_numpy(), sub["authors_cluster"].to_numpy()),
                    "features": compare_to_authors(sub["features"].to_numpy(), sub["authors_cluster"].to_numpy()),
                    "random_seed0": compare_to_authors(sub["random_00"].to_numpy(), sub["authors_cluster"].to_numpy()),
                    "note": "restricted to the identified 1,007 households" if run == "a" else "all 1,007",
                }
            rr[grp] = g
            df.attrs = {}
            W_store[(run, grp)] = (df, M)
            log(f"run {run} {grp}: NMF sizes {g['kmedoids_alternate']['sizes']} (PAM {g['kmedoids_pam']['sizes']}), "
                f"features {g['features']['sizes']}"
                + (f", ARI vs authors {g['vs_authors']['nmf']['ari']:.3f}" if grp == "ToU" else ""))
        out = pd.concat([W_store[(run, "ToU")][0], W_store[(run, "Std")][0]], ignore_index=True)
        if "authors_cluster" in out.columns:
            out["authors_cluster"] = out["authors_cluster"].fillna(-1).astype(np.int64)
        out.to_parquet(DATA / f"clusters_run_{run}.parquet", index=False)
        results["runs"][run] = rr
    results["stability"] = stability(W_store, auth, hs)
    results["forensic_run_c"] = forensic_variants(W_store, auth, hs)
    results["agreg_profiles_check"] = agreg_check(auth, hs)
    save_results("clustering", results)


def _ari_on_common(df1, col1, df2, col2):
    m = df1[["LCLid", col1]].merge(df2[["LCLid", col2]], on="LCLid", suffixes=("_1", "_2"))
    c1 = col1 + ("_1" if col1 == col2 else "")
    c2 = col2 + ("_2" if col1 == col2 else "")
    return {"ari": C.ari(m[c1], m[c2]), "n_common": int(len(m))}


def stability(W_store, auth, hs):
    st = {"across_runs": {}, "nmf_inits": {}, "kmedoids_variants": {}, "random_vs_random": {}}
    for grp in ("ToU", "Std"):
        for r1, r2 in (("a", "b"), ("b", "c"), ("a", "c")):
            d1, d2 = W_store[(r1, grp)][0], W_store[(r2, grp)][0]
            st["across_runs"][f"{grp} {r1} vs {r2}"] = {
                "nmf": _ari_on_common(d1, "nmf", d2, "nmf"),
                "features": _ari_on_common(d1, "features", d2, "features"),
            }
    for run in ("a", "c"):
        for grp in ("ToU", "Std"):
            df, M = W_store[(run, grp)]
            base = df["nmf"].to_numpy()
            ent = {}
            variants = [("nndsvda", "nndsvda", 0), ("nndsvdar", "nndsvdar", 0)] + \
                       [(f"random_seed{s}", "random", s) for s in range(10)]
            for name, init, seed in variants:
                lab, W, Hm, info, km = nmf_kmedoids(M, init=init, random_state=seed)
                e = {"ari_vs_nndsvd": C.ari(lab, base), "sizes": sizes_of(lab),
                     "nmf_relative_err": info["relative_err"], "nmf_n_iter": info["n_iter"]}
                if grp == "ToU":
                    sel = df["authors_cluster"].to_numpy() >= 0
                    e["ari_vs_authors"] = C.ari(lab[sel], df["authors_cluster"].to_numpy()[sel])
                ent[name] = e
            aris = [v["ari_vs_nndsvd"] for k_, v in ent.items() if k_.startswith("random_seed")]
            ent["summary_random_inits"] = {"ari_vs_nndsvd_min": float(np.min(aris)),
                                           "ari_vs_nndsvd_median": float(np.median(aris)),
                                           "ari_vs_nndsvd_max": float(np.max(aris))}
            st["nmf_inits"][f"run {run} {grp}"] = ent
            # k-medoids variants on the primary W
            W = df[[f"W{j}" for j in range(C.R_NMF)]].to_numpy()
            D = C.pairwise_euclidean(W)
            kv = {"pam": C.kmedoids_pam(D=D)}
            for s in range(10):
                kv[f"alternate_kpp_seed{s}"] = C.kmedoids_alternate(D=D, init="k-medoids++", random_state=s)
                kv[f"alternate_random_seed{s}"] = C.kmedoids_alternate(D=D, init="random", random_state=s)
            base_cost = C.kmedoids_cost(D, C.kmedoids_alternate(D=D)["medoids"])
            ent = {"alternate_heuristic_cost": base_cost}
            for name, r in kv.items():
                e = {"ari_vs_alternate_heuristic": C.ari(r["labels"], base), "sizes": sizes_of(r["labels"]),
                     "cost": r["cost"]}
                if grp == "ToU":
                    sel = df["authors_cluster"].to_numpy() >= 0
                    e["ari_vs_authors"] = C.ari(r["labels"][sel], df["authors_cluster"].to_numpy()[sel])
                ent[name] = e
            for fam in ("alternate_kpp", "alternate_random"):
                a_ = [v["ari_vs_alternate_heuristic"] for k_, v in ent.items() if k_.startswith(fam)]
                c_ = [v["cost"] for k_, v in ent.items() if k_.startswith(fam)]
                ent[f"summary_{fam}"] = {"ari_min": float(np.min(a_)), "ari_median": float(np.median(a_)),
                                         "ari_max": float(np.max(a_)), "cost_min": float(np.min(c_)),
                                         "cost_median": float(np.median(c_))}
            st["kmedoids_variants"][f"run {run} {grp}"] = ent
    df = W_store[("a", "ToU")][0]
    rr = [C.ari(df[f"random_{i:02d}"], df[f"random_{j:02d}"]) for i in range(N_RANDOM) for j in range(i + 1, N_RANDOM)]
    st["random_vs_random"] = {"ari_mean": float(np.mean(rr)), "ari_sd": float(np.std(rr))}
    return st


def forensic_variants(W_store, auth, hs):
    """Does any reasonable variant of the pipeline on run (c) reproduce the authors' partition?"""
    from sklearn.cluster import KMeans

    df, M = W_store[("c", "ToU")]
    sel = df["authors_cluster"].to_numpy() >= 0
    a = df["authors_cluster"].to_numpy()[sel]
    t = hs["tou1007"]
    mu, sigma, _ = profiles_for("auth", t)
    out = {}

    def rec(name, lab, extra=None):
        e = {"ari_vs_authors": C.ari(np.asarray(lab)[sel], a), "sizes": sizes_of(lab)}
        if extra:
            e.update(extra)
        out[name] = e

    rec("primary (nndsvd, max_iter 2000, alternate-heuristic)", df["nmf"].to_numpy())
    for mi in (200,):
        lab, W, Hm, info, km = nmf_kmedoids(M, max_iter=mi)
        rec(f"NMF max_iter={mi} (sklearn 0.22 default)", lab, {"nmf_converged": info["converged"]})
    lab, W, Hm, info, km = nmf_kmedoids(M, km="pam")
    rec("PAM on W", lab)
    W = df[[f"W{j}" for j in range(C.R_NMF)]].to_numpy()
    # squared Euclidean distances (eq. 28 literally)
    D2 = C.pairwise_euclidean(W) ** 2
    rec("alternate-heuristic, squared Euclidean (eq. 28)", C.kmedoids_alternate(D=D2)["labels"])
    rec("k-means on W (k-means++, 10 inits, seed 0)", KMeans(4, n_init=10, random_state=0).fit_predict(W))
    # clustering M directly / unnormalised profiles
    rec("alternate-heuristic on M (no NMF)", C.kmedoids_alternate(M)["labels"])
    mu_raw = np.concatenate([mu[:, :, 0], mu[:, :, 1], mu[:, :, 2]], axis=1)
    lab, *_ = nmf_kmedoids(np.maximum(mu_raw, 0))
    rec("NMF on unnormalised mu (kWh)", lab)
    # W row-normalised (shape only)
    Wn = W / W.sum(axis=1, keepdims=True)
    rec("alternate-heuristic on row-normalised W", C.kmedoids_alternate(Wn)["labels"])
    # r variants
    for r in (4, 6):
        Wr, *_ = C.nmf_features(M, r=r)
        rec(f"NMF r={r}", C.kmedoids_alternate(Wr)["labels"])
    # GCV smoothing instead of LAML in the causality model
    af = C.authors_dummies_frame()
    Y = C.load_records(t["row_in_filled"])
    res_g = C.causality_profiles(Y, C.authors_temps(), af[cols("low")].to_numpy(float),
                                 af[cols("high")].to_numpy(float), method="gcv")
    Mg, _ = C.profile_matrix(res_g["mu"])
    lab, *_ = nmf_kmedoids(Mg)
    rec("causality smoothing GCV instead of LAML", lab,
        {"max_abs_mu_diff_vs_laml_kwh": float(np.abs(res_g["mu"] - mu).max())})
    # which smoothing criterion reproduces the authors' agreg 'mean' profiles better?
    ag = pd.read_feather(L.UP_DATA / "agreg_profiles_nmf.feather")
    remap = {0: 1, 1: 3, 2: 0, 3: 2}
    fit_cmp = {}
    for name, mu_v in (("laml", mu), ("gcv", res_g["mu"])):
        diffs, cors = [], []
        for c in range(4):
            for j, p in enumerate(("low", "normal", "high")):
                m_auth = ag[(ag["cluster"] == remap[c]) & (ag["type"] == "mean") & (ag["price"] == p)] \
                    .sort_values("instant")["kWh"].to_numpy()
                m_ours = mu_v[auth == c, :, j].mean(axis=0)
                diffs.append(np.abs(m_ours - m_auth).mean())
                cors.append(np.corrcoef(m_ours, m_auth)[0, 1])
        fit_cmp[name] = {"agreg_mean_abs_diff_kwh": float(np.mean(diffs)), "agreg_corr_median": float(np.median(cors))}
    out["smoothing_vs_agreg_mean"] = fit_cmp
    # is the authors' partition a k-medoids fixed point in our W space?
    Dw = C.pairwise_euclidean(W)
    auth_full = df["authors_cluster"].to_numpy()
    med_auth = np.array([np.flatnonzero(auth_full == c)[np.argmin(Dw[np.ix_(auth_full == c, auth_full == c)].sum(axis=1))]
                         for c in range(4)])
    assign = np.argmin(Dw[med_auth], axis=0)
    conv = C.kmedoids_alternate(D=Dw, init=med_auth)
    out["authors_partition_in_our_W"] = {
        "share_assigned_to_own_cluster_medoid": float(np.mean(assign == auth_full)),
        "kmedoids_cost_authors_medoids": C.kmedoids_cost(Dw, med_auth),
        "kmedoids_cost_ours_alternate_heuristic": C.kmedoids_cost(Dw, C.kmedoids_alternate(D=Dw)["medoids"]),
        "alternate_from_authors_medoids": {"ari_vs_authors": C.ari(conv["labels"], auth_full),
                                           "sizes": sizes_of(conv["labels"]), "cost": conv["cost"],
                                           "n_iter": conv["n_iter"]},
    }
    # raw (unsmoothed) per-tariff means instead of the causality model
    codes = np.ones((365, 48), int)
    codes[af[cols("low")].to_numpy() > 0.5] = 0
    codes[af[cols("high")].to_numpy() > 0.5] = 2
    raw_mu = np.stack([np.stack([Y[:, codes[:, h] == p, h].mean(axis=1) for h in range(48)], axis=1)
                       for p in range(3)], axis=2)
    Mr, _ = C.profile_matrix(raw_mu)
    lab, *_ = nmf_kmedoids(Mr)
    rec("raw per-tariff means (no temperature model)", lab)
    best = max((v["ari_vs_authors"], k) for k, v in out.items() if isinstance(v, dict) and "ari_vs_authors" in v)
    out["best_variant"] = {"name": best[1], "ari": best[0]}
    return out


def agreg_check(auth, hs):
    """Average of the household-level causality profiles (run c inputs) within the recovered
    authors' clusters vs the authors' agreg_profiles_nmf.feather (cluster labels remapped
    power -> agreg {0:1, 1:3, 2:0, 3:2}, phase-1 finding)."""
    t = hs["tou1007"]
    mu, sigma, _ = profiles_for("auth", t)
    a = pd.read_feather(L.UP_DATA / "agreg_profiles_nmf.feather")
    remap = {0: 1, 1: 3, 2: 0, 3: 2}
    out = {"cluster_label_map_power_to_agreg": remap}
    rows = []
    for c in range(4):
        ac = a[a["cluster"] == remap[c]]
        for j, p in enumerate(("low", "normal", "high")):
            m_auth = ac[(ac["type"] == "mean") & (ac["price"] == p)].sort_values("instant")["kWh"].to_numpy()
            v_auth = ac[(ac["type"] == "variance") & (ac["price"] == p)].sort_values("instant")["kWh"].to_numpy()
            m_ours = mu[auth == c, :, j].mean(axis=0)
            s_mean = sigma[auth == c, :, j].mean(axis=0)
            s2_mean = (sigma[auth == c, :, j] ** 2).mean(axis=0)
            rows.append(dict(cluster=c, price=p,
                             mean_corr=float(np.corrcoef(m_ours, m_auth)[0, 1]),
                             mean_max_abs_diff=float(np.abs(m_ours - m_auth).max()),
                             mean_mean_abs_diff=float(np.abs(m_ours - m_auth).mean()),
                             var_corr_with_mean_sigma=float(np.corrcoef(s_mean, v_auth)[0, 1]),
                             var_ratio_to_mean_sigma=float(np.median(v_auth / s_mean)),
                             var_corr_with_mean_sigma2=float(np.corrcoef(s2_mean, v_auth)[0, 1]),
                             var_ratio_to_mean_sigma2=float(np.median(v_auth / s2_mean))))
    df = pd.DataFrame(rows)
    out["per_cluster_price"] = df.round(5).to_dict(orient="records")
    out["summary"] = {k: float(df[k].median()) for k in df.columns if k not in ("cluster", "price")}
    return out


# ====================================================================================== table3
def table3_for(Xraw, labels_by_method, mask_low, mask_high, definitions):
    ser = C.record_series(Xraw, mask_low, mask_high)
    out = {}
    for meth, lab in labels_by_method.items():
        if isinstance(lab, list):   # random seeds -> mean, sd
            vals = [C.ch_scores(ser, l_, definitions) for l_ in lab]
            out[meth] = {s: {d: {"mean": float(np.mean([v[s][d] for v in vals])),
                                 "sd": float(np.std([v[s][d] for v in vals])),
                                 "seed0": float(vals[0][s][d])}
                             for d in definitions} for s in SERIES}
        else:
            r = C.ch_scores(ser, lab, definitions)
            out[meth] = {s: {d: float(r[s][d]) for d in definitions} for s in SERIES}
    return out


def _val(entry):
    return entry["mean"] if isinstance(entry, dict) else entry


def _ratio(a, b):
    return float(a) / float(b) if float(b) != 0.0 else float("nan")


def orderings(tab, d):
    """The paper's qualitative claims for one Table-3 dict (cols like 'NMF ToU') under definition d.
    Ratios with a zero denominator (D3 on standardized records is exactly 0) are NaN (-> null)."""
    g = lambda col, s: _val(tab[col][s][d])  # noqa: E731
    o = {
        "Features>NMF raw ToU": g("Features ToU", "raw") > g("NMF ToU", "raw"),
        "Features>NMF raw Std": g("Features Std", "raw") > g("NMF Std", "raw"),
        "NMF>Features std ToU": g("NMF ToU", "std") > g("Features ToU", "std"),
        "NMF>Features std Std": g("NMF Std", "std") > g("Features Std", "std"),
        "NMF special>std ToU": g("NMF ToU", "special") > g("NMF ToU", "std"),
        "NMF special>std Std": g("NMF Std", "special") > g("NMF Std", "std"),
        "increase special/std ToU": _ratio(g("NMF ToU", "special"), g("NMF ToU", "std")),
        "increase special/std Std": _ratio(g("NMF Std", "special"), g("NMF Std", "std")),
        "NMF vs Features ToU special (%)": 100 * (_ratio(g("NMF ToU", "special"), g("Features ToU", "special")) - 1),
        "NMF vs Features Std special (%)": 100 * (_ratio(g("NMF Std", "special"), g("Features Std", "special")) - 1),
        "NMF>Rd all": all(g(f"NMF {grp}", s) > g(f"Rd {grp}", s) for grp in ("ToU", "Std") for s in SERIES),
    }
    o["larger increase for ToU than Std"] = bool(o["increase special/std ToU"] > o["increase special/std Std"])
    return o


def paper_orderings():
    P = PAPER["table3"]
    tab = {col: {s: {"P": v} for s, v in zip(SERIES, vals)} for col, vals in P.items()}
    return orderings(tab, "P")


def stage_table3():
    hs = C.household_sets()
    auth = load_recovered(hs)
    defs = ("D1", "D2", "D3", "D4")
    res = {"definitions": {
        "D1": "sklearn calinski_harabasz_score, observations = households, Y(i) = the household's record vector",
        "D2": "paper eqs. (3)-(5) literally: (|I|-k) sum_l ||Y(C_l)-Y(I)||^2 / ((k-1) sum_l |C_l|^-1 sum_i ||Y(i)-Y(C_l)||^2)",
        "D3": "pooled 1-D records: every household x record is one observation labelled by its household's cluster",
        "D4": "observations = household-days with 48-dim daily profiles (special: non-special records set to 0)",
    }, "series": {"raw": "full-year 17,520 records (kWh)", "std": "records / household mean over the year",
                  "special": "std records at Low-or-High slots only (2,160 slots for 'auth', 2,448 for 'loc')"},
        "paper": PAPER["table3"], "paper_orderings": paper_orderings()}
    # ---- (1) recovered authors' partition, authors' data
    t = hs["tou1007"]
    X = C.load_records(t["row_in_filled"], as_days=False)
    af = C.authors_dummies_frame()
    lowA, highA = af[cols("low")].to_numpy(float), af[cols("high")].to_numpy(float)
    rnd = [C.random_labels(len(t), seed=s) for s in range(N_RANDOM)]
    rec = table3_for(X, {"NMF ToU (recovered authors' partition)": auth, "Rd ToU": rnd}, lowA, highA, defs)
    paper_nmf = dict(zip(SERIES, PAPER["table3"]["NMF ToU"]))
    match = {}
    for d in defs:
        vals = {s: rec["NMF ToU (recovered authors' partition)"][s][d] for s in SERIES}
        match[d] = {s: {"ours": vals[s], "paper": paper_nmf[s], "ratio_ours_over_paper": vals[s] / paper_nmf[s]}
                    for s in SERIES}
        match[d]["random_mean_over_seeds"] = {s: rec["Rd ToU"][s][d]["mean"] for s in SERIES}
        match[d]["nmf_over_random_mean"] = {s: _ratio(vals[s], rec["Rd ToU"][s][d]["mean"]) for s in SERIES}
        match[d]["paper_nmf_over_rd"] = {s: PAPER["table3"]["NMF ToU"][i] / PAPER["table3"]["Rd ToU"][i]
                                         for i, s in enumerate(SERIES)}
        with np.errstate(divide="ignore"):
            match[d]["max_abs_log_ratio_nmf"] = float(max(abs(np.log(vals[s] / paper_nmf[s])) for s in SERIES))
    res["recovered_partition"] = rec
    res["definition_match"] = match
    res["closest_definition_not_a_match"] = min(defs, key=lambda d: match[d]["max_abs_log_ratio_nmf"])
    res["any_definition_within_5pct_of_paper_nmf_tou"] = bool(
        any(match[d]["max_abs_log_ratio_nmf"] < np.log(1.05) for d in defs))
    # extended search over normalisations / observation units (documented in the JSON)
    res["extended_search"] = extended_search(X, auth, lowA, highA)
    log("Table 3 on the recovered partition: " + ", ".join(
        f"{d}: " + "/".join(f"{match[d][s]['ours']:.4g}" for s in SERIES) for d in defs))
    # ---- (2) our clusterings
    res["ours"] = {}
    for run, spec in RUNS.items():
        dfc = pd.read_parquet(DATA / f"clusters_run_{run}.parquet")
        if spec["tariff"] == "auth":
            lo, hi = lowA, highA
        else:
            cond = C.era5_conditions(spec["tariff"])
            lo, hi = cond[cols("low")].to_numpy(float), cond[cols("high")].to_numpy(float)
        tab = {}
        for grp in ("ToU", "Std"):
            sub = dfc[dfc["group"] == grp]
            Xg = C.load_records(sub["row_in_filled"], as_days=False)
            labs = {f"Rd {grp}": [sub[f"random_{s:02d}"].to_numpy() for s in range(N_RANDOM)],
                    f"Features {grp}": sub["features"].to_numpy(), f"NMF {grp}": sub["nmf"].to_numpy(),
                    f"NMF-PAM {grp}": sub["nmf_pam"].to_numpy()}
            tab.update(table3_for(Xg, labs, lo, hi, defs))
        res["ours"][run] = {"table": tab, "orderings": {d: orderings(tab, d) for d in defs}}
        log(f"run {run}: D1 NMF ToU {[round(tab['NMF ToU'][s]['D1'], 2) for s in SERIES]}, "
            f"Features ToU {[round(tab['Features ToU'][s]['D1'], 2) for s in SERIES]}")
    save_results("table3", res)


def extended_search(X, auth, low, high):
    """Normalisation x observation unit x special-record handling, NMF (recovered) vs random
    (mean of 10 seeds).  Reports the combinations closest to the paper's NMF ToU values."""
    from sklearn.metrics import calinski_harabasz_score

    S = len(X)
    spec = ((low > .5) | (high > .5)).ravel()
    evday = np.repeat(((low + high).sum(axis=1) > 0), 48)
    norms = {
        "raw": X, "hh_mean": X / X.mean(1, keepdims=True), "hh_max": X / X.max(1, keepdims=True),
        "hh_zscore": (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True),
        "hh_sd": X / X.std(1, keepdims=True),
        "hh_minmax": (X - X.min(1, keepdims=True)) / (X.max(1, keepdims=True) - X.min(1, keepdims=True)),
        "record_mean (column bug)": X / X.mean(0, keepdims=True),
        "record_zscore": (X - X.mean(0, keepdims=True)) / X.std(0, keepdims=True),
    }
    sels = {"all": None, "special_removed": spec, "special_zeroed": "zero", "event_days": evday}

    def defs(Z, lab):
        o = {"households_sklearn": C.ch_sklearn(Z, lab), "households_paper": C.ch_paper(Z, lab),
             "pooled_1d": C.ch_grouped(Z[:, :, None], lab)}
        if Z.shape[1] == L.N_SLOTS:
            o["household_days_48"] = C.ch_grouped(Z.reshape(S, 365, 48), lab)
            o["household_slot_365"] = C.ch_grouped(Z.reshape(S, 365, 48).transpose(0, 2, 1), lab)
            agg = np.stack([Z[lab == c].mean(0) for c in range(4)])
            o["cluster_days_48"] = float(calinski_harabasz_score(agg.reshape(-1, 48), np.repeat(np.arange(4), 365)))
        agg = np.stack([Z[lab == c].mean(0) for c in range(4)])
        o["cluster_records_1d"] = float(calinski_harabasz_score(agg.reshape(-1, 1), np.repeat(np.arange(4), agg.shape[1])))
        return o

    labs = {"NMF": auth, **{f"rd{s}": C.random_labels(S, seed=s) for s in range(10)}}
    rows = []
    for nname, Z in norms.items():
        for sname, sel in sels.items():
            if sel is None:
                Zs = Z
            elif isinstance(sel, str):
                Zs = np.where(spec[None], Z, 0.0)
            else:
                Zs = Z[:, sel]
            for lname, lab in labs.items():
                for d, v in defs(Zs, lab).items():
                    rows.append(dict(norm=nname, sel=sname, defn=d, lab=lname, ch=v))
    df = pd.DataFrame(rows)
    df["grp"] = np.where(df["lab"] == "NMF", "NMF", "rd")
    p = df.groupby(["norm", "sel", "defn", "grp"])["ch"].mean().unstack("grp").reset_index()
    with np.errstate(divide="ignore"):
        p["log_ratio_nmf_vs_paper_raw"] = np.log(p["NMF"] / 13432.0)
        p["log_ratio_rd_vs_paper_raw"] = np.log(p["rd"] / 4627.0)
    p["nmf_over_rd"] = p["NMF"] / p["rd"]
    # candidate score: how close (NMF, random) are to the paper's (NMF, Rd) pair of the same row
    paper_pairs = {"raw": (13432, 4627), "std": (14088, 4950), "special": (16070, 5151)}
    best = []
    for row_name, (pn, pr) in paper_pairs.items():
        q = p.copy()
        q["dist"] = np.sqrt(np.log(q["NMF"].clip(lower=1e-12) / pn) ** 2 + np.log(q["rd"].clip(lower=1e-12) / pr) ** 2)
        q = q.sort_values("dist").head(5)
        best.append({"paper_row": row_name, "paper_nmf": pn, "paper_rd": pr,
                     "closest": q[["norm", "sel", "defn", "NMF", "rd", "dist"]].round(3).to_dict(orient="records")})
    return {"table": p.round(4).to_dict(orient="records"), "closest_to_paper_pairs": best,
            "n_random_seeds": 10}


# ======================================================================================== fig2
def stage_fig2():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = C.household_sets()
    dates = L.DATES_2013
    dfc = pd.read_parquet(DATA / "clusters_run_a.parquet")
    sub = dfc[dfc["group"] == "ToU"].reset_index(drop=True)
    Y = C.load_records(sub["row_in_filled"])
    cons, sizes = C.authors_cluster_means()
    # order NMF clusters like the authors' (Hungarian on cluster-mean correlations; same rule as
    # scripts/05_build_trackB_data.py), features by mean level, random by label
    nmf_means = C.cluster_means(Y, sub["nmf"].to_numpy())
    mapping, Corr = C.relabel_hungarian(nmf_means, cons)
    lab_nmf = np.vectorize(mapping.get)(sub["nmf"].to_numpy())
    f_means = C.cluster_means(Y, sub["features"].to_numpy())
    order = np.argsort(f_means.mean(axis=(1, 2)))
    lab_feat = np.vectorize({int(o): i for i, o in enumerate(order)}.get)(sub["features"].to_numpy())
    panels = [("Random clustering (seed 0)", sub["random_00"].to_numpy()),
              ("Classical features", lab_feat), ("Proposed method (NMF + k-medoids)", lab_nmf)]
    # validated categorical slots 1-4 (dataviz reference palette, light mode); line style is a
    # secondary encoding because slots 3-4 are below 3:1 contrast on white
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    styles = ["-", (0, (5, 2)), (0, (6, 2, 1, 2)), (0, (1.5, 1.5))]
    x = np.arange(7 * 48) / 48.0
    fig, axes = plt.subplots(3, 2, figsize=(13, 9.5), sharex=True)
    summary = {}
    pop = C.weekly_profile(Y.mean(axis=0)[None], dates)[0]
    pop_std = pop / Y.mean()
    for r, (title, lab) in enumerate(panels):
        means = C.cluster_means(Y, lab)                      # (4, 365, 48)
        wk = C.weekly_profile(means, dates)                  # (4, 7, 48)
        n = np.bincount(lab, minlength=4)
        lev = means.mean(axis=(1, 2))
        wk_std = wk / lev[:, None, None]
        summary[title] = {"sizes": n.tolist(),
                          "cluster_mean_kwh": lev.round(4).tolist(),
                          "level_ratio_max_over_min": float(lev.max() / lev.min()),
                          "shape_rms_dev_std_weekly_vs_population": float(np.sqrt(((wk_std - pop_std) ** 2).mean())),
                          "level_rms_dev_weekly_kwh_vs_population": float(np.sqrt(((wk - pop) ** 2).mean())),
                          "weekly_peak_std": (wk.max(axis=(1, 2)) / lev).round(3).tolist()}
        for c in range(4):
            lbl = f"Cluster {c} - {n[c]} households"
            axes[r, 0].plot(x, wk[c].ravel(), color=colors[c], ls=styles[c], lw=0.9, label=lbl)
            axes[r, 1].plot(x, wk[c].ravel() / means[c].mean(), color=colors[c], ls=styles[c], lw=0.9, label=lbl)
        axes[r, 0].set_ylabel("Energy consumption (kWh)")
        axes[r, 1].set_ylabel("Standardized consumption")
        axes[r, 0].set_title(title + " - cluster mean", fontsize=10)
        axes[r, 1].set_title(title + " - divided by the cluster's mean", fontsize=10)
        for ax in axes[r]:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.30 * (hi - lo))   # headroom so the legend does not cover the curves
            ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="upper left")
            ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[-1]:
        ax.set_xticks(np.arange(7) + 0.5)
        ax.set_xticklabels(C.DOW_NAMES)
        ax.set_xlabel("Average week, 2013 (half-hourly)")
    fig.suptitle("Fig. 2 replica, run (a): our 1,032 ToU households (ERA5 temperatures, Tariffs.xlsx 'loc')",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig2_clusters.png", dpi=150)
    plt.close(fig)
    # ---- companion figure: authors' recovered partition vs our NMF clusters (runs a and c)
    auth = load_recovered(hs)
    t = hs["tou1007"]
    Yt = C.load_records(t["row_in_filled"])
    dfc_c = pd.read_parquet(DATA / "clusters_run_c.parquet")
    sub_c = dfc_c[dfc_c["group"] == "ToU"].reset_index(drop=True)
    mc = C.cluster_means(Yt, sub_c["nmf"].to_numpy())
    map_c, _ = C.relabel_hungarian(mc, cons)
    lab_c = np.vectorize(map_c.get)(sub_c["nmf"].to_numpy())
    rows = [("Authors' partition (recovered by NNLS), 1,007 ToU", Yt, auth),
            ("Our run (c): authors' temps + dummies, 1,007 ToU", Yt, lab_c),
            ("Our run (a): ERA5 + 'loc', 1,032 ToU", Y, lab_nmf)]
    fig, axes = plt.subplots(3, 2, figsize=(13, 9.5), sharex=True)
    for r, (title, Yr, lab) in enumerate(rows):
        means = C.cluster_means(Yr, lab)
        wk = C.weekly_profile(means, dates)
        n = np.bincount(lab, minlength=4)
        for c in range(4):
            lbl = f"Cluster {c} - {n[c]} households"
            axes[r, 0].plot(x, wk[c].ravel(), color=colors[c], ls=styles[c], lw=0.9, label=lbl)
            axes[r, 1].plot(x, wk[c].ravel() / means[c].mean(), color=colors[c], ls=styles[c], lw=0.9, label=lbl)
        axes[r, 0].set_ylabel("Energy consumption (kWh)")
        axes[r, 1].set_ylabel("Standardized consumption")
        axes[r, 0].set_title(title, fontsize=10)
        axes[r, 1].set_title(title + " (standardized)", fontsize=10)
        for ax in axes[r]:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.30 * (hi - lo))   # headroom so the legend does not cover the curves
            ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="upper left")
            ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[-1]:
        ax.set_xticks(np.arange(7) + 0.5)
        ax.set_xticklabels(C.DOW_NAMES)
        ax.set_xlabel("Average week, 2013 (half-hourly)")
    fig.suptitle("NMF clusters: the authors' recovered partition vs our runs (labels matched to the authors')",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS / "fig2_recovered_vs_ours.png", dpi=150)
    plt.close(fig)
    # numbers behind the figure
    wk_auth = C.weekly_profile(C.cluster_means(Yt, auth), dates)
    wk_c = C.weekly_profile(C.cluster_means(Yt, lab_c), dates)
    wk_a = C.weekly_profile(C.cluster_means(Y, lab_nmf), dates)
    summary["weekly_profile_corr_vs_authors"] = {
        "run_c": [float(np.corrcoef(wk_c[c].ravel(), wk_auth[c].ravel())[0, 1]) for c in range(4)],
        "run_a": [float(np.corrcoef(wk_a[c].ravel(), wk_auth[c].ravel())[0, 1]) for c in range(4)],
    }
    summary["run_a_nmf_label_map_to_authors"] = mapping
    summary["run_a_cluster_mean_corr_with_authors"] = np.round(Corr, 5).tolist()
    summary["files"] = [str(FIGS / "fig2_clusters.png"), str(FIGS / "fig2_recovered_vs_ours.png")]
    save_results("fig2", summary)
    log("wrote " + ", ".join(summary["files"]))


# ======================================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stages", default="recover,profiles,cluster,table3,fig2")
    ap.add_argument("--force-profiles", action="store_true", help="refit the causality profiles")
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        log(f"=== stage {s}")
        if s == "recover":
            stage_recover()
        elif s == "profiles":
            stage_profiles(force=args.force_profiles)
        elif s == "cluster":
            stage_cluster()
        elif s == "table3":
            stage_table3()
        elif s == "fig2":
            stage_fig2()
        else:
            raise SystemExit(f"unknown stage {s}")
    log("done")


if __name__ == "__main__":
    main()
