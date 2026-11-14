"""Table 3 of Bregere & Bessa (2020) under the identified Calinski-Harabasz definition.

The phase-3 verification (spec section 7, outputs/verify/v-clustering/) found that the paper's
Table 3 values come from a Calinski-Harabasz variant with NON-squared Euclidean norms and a
cluster-size-weighted between-cluster term:

    S = (|I|-k)/(k-1) * sum_l |C_l| ||Y(C_l) - Y(I)||  /  sum_l mean_{i in C_l} ||Y(i) - Y(C_l)||

with k = 4. Y(i) is household i's full-year record vector (the 17,520 filled half-hours of 2013, in
the authors' slot convention). Y(C_l) and Y(I) are the mean vectors of cluster C_l and of all
households. Three record series:
    raw      Y(i) as recorded (kWh);
    std      Y(i) / mean(Y(i)), the household's mean over the whole year (paper: "standardized");
    special  the 'std' records at the Low-or-High slots only (special-tariff mask).
On the authors' recovered NMF partition this gives 13,433 / 14,076 / 16,055 (paper 13,432 /
14,088 / 16,070). Random labels give about 5,000, as in the paper.

Computed here, for every clustering below, under that definition and, for contrast, under
eqs. 3-5 as printed (squared norms, no |C_l| weight) and sklearn's calinski_harabasz_score
(random labels give about 1 under both):
  run 'authors_recovered'  the authors' ToU NMF partition, recovered exactly by NNLS
                           (data/authors_clusters_recovered.parquet, 1,007 households), with the
                           authors' special-tariff mask (2,160 slots); random labels (20 seeds,
                           numpy default_rng(1000 + s), as the verifier) for these ToU households
                           and for the 1,007 Std households of in_1007_pair (no authors' Std
                           partition exists);
  runs 'a', 'b', 'c'       our clusterings (data/clusters_run_{a,b,c}.parquet): NMF + k-medoids,
                           its PAM variant (sensitivity), classical features, and the stored random
                           labels (seeds 0..19), for ToU and Std, with the special mask each run
                           used (a, b: Tariffs.xlsx 'loc', 2,448 slots; c: the authors' dummies).
The paper's Table 3 sits alongside (ratios ours / paper), with the paper's orderings and the two
headline numbers: NMF vs Features on ToU special-tariff records (+128%), and the NMF special/std
ratio for ToU vs Std (x1.141 vs x1.035). Also: a special-mask sensitivity (each run's special
series with the other mask), a random-label null approximation, self-checks, and a cross-check
against the verifier's JSONs (and the D1/D2 values of outputs/clustering_results.json).

Inputs (read only): data/lcl2013_kwh_filled.npy, data/selection.parquet,
data/authors_clusters_recovered.parquet, data/clusters_run_{a,b,c}.parquet,
data/conditions_ours.parquet (low_loc_hh / high_loc_hh; low_auth_hh / high_auth_hh copy),
UPSTREAM/data/power_consump_nmf.feather (the authors' low_hh / high_hh dummies, read as data by
absolute path; nothing from the upstream repository is imported or executed; if the file is gone,
the identical copy in conditions_ours is used), outputs/clustering_results.json (run metadata),
outputs/verify/v-clustering/t3_*.json (cross-check only).
Outputs: outputs/table3/table3_identified.json, outputs/table3/table3_identified.csv
(one row per run x group x method x record series).

Run:  python -P scripts/06_table3.py       (about 1 minute, single process, 1 BLAS thread)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    try:
        _n = int(os.environ.get(_v, "1"))
    except ValueError:
        _n = 1
    os.environ[_v] = str(max(1, min(_n, 4)))  # default 1 BLAS thread; spec: at most 4

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import calinski_harabasz_score  # noqa: E402

# ------------------------------------------------------------------------------------ paths
REPRO = Path(__file__).resolve().parents[1]
DATA = REPRO / "data"
OUTPUTS = REPRO / "outputs"
OUT_DIR = OUTPUTS / "table3"
OUT_JSON = OUT_DIR / "table3_identified.json"
OUT_CSV = OUT_DIR / "table3_identified.csv"
VERIFY_DIR = OUTPUTS / "verify" / "v-clustering"
CLUSTERING_JSON = OUTPUTS / "clustering_results.json"
FILLED_NPY = DATA / "lcl2013_kwh_filled.npy"
SELECTION_PARQUET = DATA / "selection.parquet"
RECOVERED_PARQUET = DATA / "authors_clusters_recovered.parquet"
CONDITIONS_PARQUET = DATA / "conditions_ours.parquet"
RUN_PARQUET = {r: DATA / f"clusters_run_{r}.parquet" for r in "abc"}
# The authors' repository is untrusted: only this .feather file is read, as data.
UPSTREAM_DATA = DATA / "authors"  # copy of the authors' repo data/*.feather (byte-identical; read as data only)
AUTHORS_FEATHER = UPSTREAM_DATA / "power_consump_nmf.feather"

# -------------------------------------------------------------------------------- constants
K = 4
N_DAYS, H = 365, 48
N_SLOTS = N_DAYS * H
SERIES = ("raw", "std", "special")
GROUPS = ("ToU", "Std")
METRICS = ("S_identified", "S_eq345_printed", "CH_sklearn")
N_RANDOM = 20
RECOVERED_SEEDS = tuple(1000 + s for s in range(N_RANDOM))   # the verifier's t3_core seeds
N_SKLEARN_SEEDS = 3          # random seeds per group and series also scored by sklearn's function
SELF_CHECK_TOL = 1e-8        # max relative disagreement tolerated by the self-checks
CROSSCHECK_TOL_PCT = 0.5

PAPER = {  # Table 3; rows: raw / standardized / special tariff - standardized
    "ToU": {"Rd": (4627, 4950, 5151), "Features": (40764, 6870, 7033), "NMF": (13432, 14088, 16070)},
    "Std": {"Rd": (5014, 4834, 4904), "Features": (5088, 4741, 4694), "NMF": (12971, 14934, 15460)},
}
PAPER_SIZES_TOU = {"NMF": [325, 295, 229, 158], "Features": [255, 247, 249, 256],
                   "Rd": [248, 264, 242, 253]}           # paper Fig. 2 / Sec. IV
VERIFIER_RECOVERED = (13432.6, 14076.2, 16055.0)        # spec section 7, v-clustering/summary.json

RUN_ORDER = ("authors_recovered", "a", "b", "c")
SHORT = {"authors_recovered": "rec", "a": "a", "b": "b", "c": "c"}
SET_TOU1007 = "identified 1,007 ToU households (selection.in_authors_tou_1007)"
SET_STD1007 = "1,007 Std households of selection.in_1007_pair"
RUNS = {
    "a": dict(mask="loc", sets={"ToU": "our 1,032 ToU households",
                                "Std": "our 1,032 Std households (random draw, selection.parquet)"}),
    "b": dict(mask="loc", sets={"ToU": SET_TOU1007, "Std": SET_STD1007}),
    "c": dict(mask="auth", sets={"ToU": SET_TOU1007, "Std": SET_STD1007}),
}
RUN_METHODS = {"NMF": "nmf", "NMF_PAM": "nmf_pam", "Features": "features"}
METHOD_DOC = {
    "NMF": "NMF (r = 5) of the causality profiles + k-medoids k = 4 ('alternate', heuristic init): "
           "column nmf",
    "NMF_PAM": "same NMF features, k-medoids by PAM (sensitivity only; compared with the paper's "
               "NMF column): column nmf_pam",
    "Features": "classical features (min / max / mean in winter and summer, peak and off half-hour; "
                "min-max rescaled) + k-medoids k = 4: column features",
    "Rd": "random labels: stored columns random_00..19 = numpy default_rng(s).integers(0, 4, n), "
          "s = 0..19",
}

SELF_CHECKS = {"sklearn_calls": 0, "max_rel_diff_sklearn_function_vs_formula": 0.0,
               "reference_checks": 0, "max_rel_diff_reference_loop_vs_vectorized": 0.0}
LOG_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - LOG_T0:6.1f}s] {msg}", flush=True)


def _val(x):
    """A single value, or the mean of a random-label summary."""
    return x["mean"] if isinstance(x, dict) else x


def _ratio(a, b):
    return None if a is None or b is None or b == 0 else float(a) / float(b)


def _pct(a, b):
    r = _ratio(a, b)
    return None if r is None else 100.0 * (r - 1.0)


# ------------------------------------------------------------------------------------ masks
def _union_mask(frame: pd.DataFrame, low_cols, high_cols) -> dict:
    lo = frame[low_cols].to_numpy(np.float64)
    hi = frame[high_cols].to_numpy(np.float64)
    for a in (lo, hi):
        if not np.isin(a, (0.0, 1.0)).all():
            raise ValueError("tariff dummies are not 0/1")
    lo_b, hi_b = lo > 0.5, hi > 0.5
    if (lo_b & hi_b).any():
        raise ValueError("a slot is both Low and High")
    m = (lo_b | hi_b)
    return {"mask": m.ravel(), "n_slots": int(m.sum()), "n_low": int(lo_b.sum()),
            "n_high": int(hi_b.sum()), "n_event_days": int(m.any(axis=1).sum())}


def load_masks() -> dict:
    """Special-tariff masks, 17,520 booleans (slot j = day j // 48, half-hour j % 48, the authors'
    slot convention, i.e. the column order of lcl2013_kwh_filled.npy):
    'auth' = the authors' low_/high_ dummies; 'loc' = Tariffs.xlsx read as London clock time and
    mapped to the authors' convention (conditions_ours low_loc_hh / high_loc_hh)."""
    dates = pd.date_range("2013-01-01", periods=N_DAYS, freq="D")
    hh = [f"{h:02d}" for h in range(H)]
    co = pd.read_parquet(CONDITIONS_PARQUET).sort_values("date").reset_index(drop=True)
    if len(co) != N_DAYS or not (pd.DatetimeIndex(co["date"]) == dates).all():
        raise ValueError("conditions_ours.parquet does not hold the 365 days of 2013")
    loc = _union_mask(co, [f"low_loc_{x}" for x in hh], [f"high_loc_{x}" for x in hh])
    loc["source"] = ("data/conditions_ours.parquet low_loc_hh / high_loc_hh (Tariffs.xlsx, London "
                     "clock time mapped to the authors' slot convention)")
    copy = _union_mask(co, [f"low_auth_{x}" for x in hh], [f"high_auth_{x}" for x in hh])
    if AUTHORS_FEATHER.exists():
        pc = pd.read_feather(AUTHORS_FEATHER)
        pc["date"] = pd.to_datetime(pc["date"])
        per_cluster = []
        for c in sorted(pc["cluster"].unique()):
            g = pc[pc["cluster"] == c].sort_values("date").reset_index(drop=True)
            if len(g) != N_DAYS or not (pd.DatetimeIndex(g["date"]) == dates).all():
                raise ValueError(f"power_consump_nmf cluster {c} does not hold the 365 days of 2013")
            per_cluster.append(_union_mask(g, [f"low_{x}" for x in hh], [f"high_{x}" for x in hh]))
        auth = per_cluster[0]
        if any(not np.array_equal(m["mask"], auth["mask"]) for m in per_cluster[1:]):
            raise ValueError("the authors' tariff dummies differ between clusters")
        if not np.array_equal(auth["mask"], copy["mask"]):
            raise ValueError("conditions_ours low_auth/high_auth differ from the authors' dummies")
        auth["source"] = ("UPSTREAM/data/power_consump_nmf.feather low_hh / high_hh (identical in "
                          "the 4 clusters and equal to the conditions_ours _auth copy)")
    else:
        auth = copy
        auth["source"] = ("data/conditions_ours.parquet low_auth_hh / high_auth_hh (copy of the "
                          "authors' dummies; the upstream file was not found)")
    return {"auth": auth, "loc": loc}


# ------------------------------------------------------------------------- labelings / data
def load_matrix(F, rows) -> np.ndarray:
    X = np.asarray(F[np.asarray(rows, dtype=np.int64)], dtype=np.float64)
    if X.shape[1] != N_SLOTS or not np.isfinite(X).all():
        raise ValueError("unexpected shape or non-finite values in the filled records")
    return X


def recovered_spec(sel: pd.DataFrame) -> dict:
    rec = pd.read_parquet(RECOVERED_PARQUET)
    lab = rec["cluster"].to_numpy(np.int64)
    rows = rec["row_in_filled"].to_numpy(np.int64)
    sizes = np.bincount(lab, minlength=K).tolist()
    if sizes != PAPER_SIZES_TOU["NMF"]:
        raise ValueError(f"recovered partition sizes {sizes} != {PAPER_SIZES_TOU['NMF']}")
    s = sel.sort_values("row_in_filled").reset_index(drop=True)
    grp = s["group"].to_numpy(dtype=object)
    ident = s.loc[s["in_authors_tou_1007"].to_numpy(bool), "row_in_filled"].to_numpy(np.int64)
    if not np.array_equal(np.sort(rows), ident):
        raise ValueError("recovered partition rows != the identified 1,007 ToU households")
    std_rows = s.loc[(grp == "Std") & s["in_1007_pair"].to_numpy(bool),
                     "row_in_filled"].to_numpy(np.int64)
    if len(std_rows) != 1007:
        raise ValueError("in_1007_pair does not hold 1,007 Std households")

    def rnd(n):
        return [np.random.default_rng(seed).integers(0, K, n) for seed in RECOVERED_SEEDS]

    rd_doc = (f"random labels numpy default_rng(1000 + s).integers(0, 4, n), s = 0..{N_RANDOM - 1} "
              "(the verifier's t3_core seeds; independent of the runs' stored seeds 0..19)")
    return {
        "label": "the authors' ToU NMF partition recovered by NNLS (exact), the authors' dummies",
        "mask": "auth",
        "groups": {
            "ToU": {"rows": rows, "household_set": SET_TOU1007,
                    "labelings": {"NMF": lab, "Rd": rnd(len(rows))},
                    "sources": {"NMF": "data/authors_clusters_recovered.parquet, column cluster "
                                       "(the authors' power_consump_nmf cluster ids)",
                                "Rd": rd_doc}},
            "Std": {"rows": std_rows, "household_set": SET_STD1007 + " (no authors' Std partition "
                                                                     "exists: random labels only)",
                    "labelings": {"Rd": rnd(len(std_rows))}, "sources": {"Rd": rd_doc}},
        },
        "notes": {"random_labels": "generated here: numpy default_rng(1000 + s), s = 0..19"},
    }


def run_metadata():
    """Run labels written by scripts/04_clustering.py (outputs/clustering_results.json)."""
    try:
        cr = json.loads(CLUSTERING_JSON.read_text(encoding="utf-8"))
        runs = cr["clustering"]["runs"]
    except (OSError, KeyError, ValueError):
        return {r: None for r in "abc"}, None
    return {r: runs.get(r, {}).get("label") for r in "abc"}, cr


def mask_from_label(label):
    if label is None:
        return None
    if "'loc'" in label:
        return "loc"
    if "dummies" in label:
        return "auth"
    return None


def run_spec(run: str, label) -> dict:
    meta_mask = mask_from_label(label)
    if meta_mask is not None and meta_mask != RUNS[run]["mask"]:
        raise ValueError(f"run {run}: metadata label implies mask {meta_mask!r}, "
                         f"expected {RUNS[run]['mask']!r}")
    df = pd.read_parquet(RUN_PARQUET[run])
    grp = df["group"].to_numpy(dtype=object)
    groups, rule_ok = {}, True
    for g in GROUPS:
        sub = df[grp == g].reset_index(drop=True)
        n = len(sub)
        labs = {m: sub[col].to_numpy(np.int64) for m, col in RUN_METHODS.items()}
        rnd = [sub[f"random_{s:02d}"].to_numpy(np.int64) for s in range(N_RANDOM)]
        rule_ok &= all(np.array_equal(lab, np.random.default_rng(s).integers(0, K, n))
                       for s, lab in enumerate(rnd))
        labs["Rd"] = rnd
        groups[g] = {"rows": sub["row_in_filled"].to_numpy(np.int64),
                     "household_set": RUNS[run]["sets"][g], "labelings": labs,
                     "sources": {m: f"data/clusters_run_{run}.parquet: {METHOD_DOC[m]}"
                                 for m in ("NMF", "NMF_PAM", "Features", "Rd")}}
    return {"label": label if label is not None else f"run {run}", "mask": RUNS[run]["mask"],
            "groups": groups,
            "notes": {"random_labels_follow_documented_rule": bool(rule_ok),
                      "mask_confirmed_by_metadata_label": meta_mask is not None}}


# -------------------------------------------------------------------- Calinski-Harabasz
class Series:
    """A record series Y (households x records) and the statistics shared by all labellings."""

    def __init__(self, Y: np.ndarray):
        self.Y = np.ascontiguousarray(Y, dtype=np.float64)
        self.row_sq = np.einsum("ij,ij->i", self.Y, self.Y)     # ||Y(i)||^2
        self.mean = self.Y.mean(axis=0)                          # Y(I)


def make_series(X: np.ndarray, mask: np.ndarray) -> dict:
    m = X.mean(axis=1, keepdims=True)
    if not (m > 0).all():
        raise ValueError("a household has a non-positive mean consumption")
    Z = X / m
    return {"raw": Series(X), "std": Series(Z), "special": Series(Z[:, mask])}


def _labels_counts(labels):
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.min() < 0 or labels.max() >= K:
        raise ValueError("labels must be integers in 0..k-1")
    counts = np.bincount(labels, minlength=K)
    if (counts == 0).any():
        raise ValueError("empty cluster")
    return labels, counts.astype(np.float64)


def _check(name: str, value: float, reference: float) -> None:
    rel = abs(value / reference - 1.0)
    SELF_CHECKS[name] = max(SELF_CHECKS[name], rel)
    if rel > SELF_CHECK_TOL:
        raise AssertionError(f"self-check {name}: {value!r} vs {reference!r} (rel {rel:.2e})")


def ch_scores(ser: Series, labels, check_sklearn: bool = False) -> dict:
    """S_identified, S_eq345_printed and CH_sklearn of one labelling, from shared sufficient
    statistics: cluster means Y(C_l) by one matrix product, squared distances to the own centroid
    by ||Y(i)||^2 - 2 Y(i).Y(C_l) + ||Y(C_l)||^2 (a second matrix product). CH_sklearn is
    sklearn's definition; with check_sklearn, sklearn.metrics.calinski_harabasz_score itself is
    also called and must agree."""
    labels, counts = _labels_counts(labels)
    Y = ser.Y
    n = len(labels)
    idx = np.arange(n)
    onehot = np.zeros((n, K))
    onehot[idx, labels] = 1.0
    C = (onehot.T @ Y) / counts[:, None]                               # Y(C_l)
    b2 = ((C - ser.mean) ** 2).sum(axis=1)                               # ||Y(C_l) - Y(I)||^2
    d2 = ser.row_sq - 2.0 * (Y @ C.T)[idx, labels] + (C * C).sum(axis=1)[labels]
    if (d2 < -1e-9 * ser.row_sq).any():
        raise FloatingPointError("negative squared distance beyond rounding")
    d2 = np.maximum(d2, 0.0)                                             # ||Y(i) - Y(C_l(i))||^2
    f = (n - K) / (K - 1)
    mean_d1 = np.bincount(labels, weights=np.sqrt(d2), minlength=K) / counts
    mean_d2 = np.bincount(labels, weights=d2, minlength=K) / counts
    out = {
        "S_identified": f * float((counts * np.sqrt(b2)).sum()) / float(mean_d1.sum()),
        "S_eq345_printed": f * float(b2.sum()) / float(mean_d2.sum()),
        "CH_sklearn": f * float((counts * b2).sum()) / float(d2.sum()),
    }
    if check_sklearn:
        SELF_CHECKS["sklearn_calls"] += 1
        _check("max_rel_diff_sklearn_function_vs_formula", out["CH_sklearn"],
               float(calinski_harabasz_score(Y, labels)))
    return out


def s_identified_reference(Y: np.ndarray, labels) -> float:
    """Literal transcription of the identified definition, cluster by cluster (self-check)."""
    labels = np.asarray(labels)
    n = len(labels)
    YI = Y.mean(axis=0)
    num = den = 0.0
    for lab in range(K):
        Yl = Y[labels == lab]
        Cl = Yl.mean(axis=0)
        num += len(Yl) * math.sqrt(float(((Cl - YI) ** 2).sum()))
        den += float(np.sqrt(((Yl - Cl) ** 2).sum(axis=1)).mean())
    return (n - K) / (K - 1) * num / den


def null_approx(ser: Series):
    """Random-label level of S_identified. With |C_l| ~ n/k, ||Y(C_l)-Y(I)|| ~ sqrt(tr(Sigma)(k-1)/n)
    and the within-cluster distances ~ the distances to Y(I), so
    S ~ (n-k)/(k-1) * sqrt(n(k-1))/k * rho,  rho = RMS / mean of ||Y(i)-Y(I)|| (>= 1).
    It grows like sqrt(n), so the identified S is not calibrated to 1 under random labels."""
    n = len(ser.Y)
    d2 = np.maximum(ser.row_sq - 2.0 * (ser.Y @ ser.mean) + float(ser.mean @ ser.mean), 0.0)
    rho = math.sqrt(float(d2.mean())) / float(np.sqrt(d2).mean())
    return (n - K) / (K - 1) * math.sqrt(n * (K - 1)) / K * rho, rho


def summarize(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=0)), "min": float(a.min()),
            "max": float(a.max())}


# ------------------------------------------------------------------------------- evaluation
def evaluate_group(X: np.ndarray, gs: dict, mask_name: str, alt_name: str, masks: dict) -> dict:
    ser = make_series(X, masks[mask_name]["mask"])
    alt = Series(ser["std"].Y[:, masks[alt_name]["mask"]])
    n_slots = {"raw": N_SLOTS, "std": N_SLOTS, "special": masks[mask_name]["n_slots"]}
    out = {"n_households": int(len(X)), "household_set": gs["household_set"],
           "special_mask": mask_name, "n_slots": n_slots, "methods": {}}
    alt_out = {"mask": alt_name, "n_slots": masks[alt_name]["n_slots"], "S_identified_special": {}}
    referenced = False   # one literal (loop) re-computation of S_identified per group

    def reference_check(lab, scores):
        for s in SERIES:
            SELF_CHECKS["reference_checks"] += 1
            _check("max_rel_diff_reference_loop_vs_vectorized", scores[s]["S_identified"],
                   s_identified_reference(ser[s].Y, lab))

    for method, lab in gs["labelings"].items():
        if method == "Rd":
            per = [{s: ch_scores(ser[s], lb, check_sklearn=j < N_SKLEARN_SEEDS) for s in SERIES}
                   for j, lb in enumerate(lab)]
            if not referenced:
                reference_check(lab[0], per[0])
                referenced = True
            entry = {"labels_source": gs["sources"][method], "n_labelings": len(lab),
                     "sizes_first_labelling": np.bincount(lab[0], minlength=K).tolist()}
            for s in SERIES:
                e = {m: summarize([d[s][m] for d in per]) for m in METRICS}
                e["S_identified"]["per_seed"] = [d[s]["S_identified"] for d in per]
                approx, rho = null_approx(ser[s])
                e["null_approx_S_identified"] = approx
                e["null_rho"] = rho
                entry[s] = e
            alt_out["S_identified_special"]["Rd"] = summarize(
                [ch_scores(alt, lb)["S_identified"] for lb in lab])
        else:
            entry = {"labels_source": gs["sources"][method], "n_labelings": 1,
                     "sizes": np.bincount(lab, minlength=K).tolist()}
            for s in SERIES:
                entry[s] = ch_scores(ser[s], lab, check_sklearn=True)
            if not referenced:
                reference_check(lab, entry)
                referenced = True
            alt_out["S_identified_special"][method] = ch_scores(alt, lab)["S_identified"]
        out["methods"][method] = entry
    out["special_alt_mask"] = alt_out
    return out


def annotate(gres: dict, group: str) -> None:
    """Paper value, ours / paper, S / Rd and where the paper value falls in our Rd distribution."""
    rd = gres["methods"].get("Rd")
    for method, entry in gres["methods"].items():
        pcol = "NMF" if method == "NMF_PAM" else method
        for i, s in enumerate(SERIES):
            e = entry[s]
            val = _val(e["S_identified"])
            p = float(PAPER[group][pcol][i])
            e["paper_column"] = f"{pcol} {group}"
            e["paper"] = p
            e["ratio_to_paper"] = val / p
            e["paper_over_paper_Rd"] = p / PAPER[group]["Rd"][i]
            if rd is not None:
                m_, sd_ = rd[s]["S_identified"]["mean"], rd[s]["S_identified"]["sd"]
                e["S_over_Rd"] = val / m_
                e["paper_z_in_Rd"] = (p - m_) / sd_ if sd_ > 0 else None


def run_getter(run_res: dict):
    def v(method, group, series):
        try:
            e = run_res["groups"][group]["methods"][method][series]
        except KeyError:
            return None
        return _val(e["S_identified"])
    return v


def paper_getter(method, group, series):
    return float(PAPER[group][method][SERIES.index(series)])


def orderings(v) -> dict:
    """The paper's qualitative claims about Table 3 (None = not computable for this run)."""
    def gt(a, b):
        return None if a is None or b is None else bool(a > b)

    o = {}
    for g in GROUPS:
        o[f"Features>NMF raw {g}"] = gt(v("Features", g, "raw"), v("NMF", g, "raw"))
    for g in GROUPS:
        o[f"NMF>Features std {g}"] = gt(v("NMF", g, "std"), v("Features", g, "std"))
    for g in GROUPS:
        o[f"NMF>Features special {g}"] = gt(v("NMF", g, "special"), v("Features", g, "special"))
    for g in GROUPS:
        o[f"NMF special>std {g}"] = gt(v("NMF", g, "special"), v("NMF", g, "std"))
    for g in GROUPS:
        c = [gt(v("NMF", g, s), v("Rd", g, s)) for s in SERIES]
        o[f"NMF>Rd {g} (all series)"] = None if None in c else all(c)
    o["NMF special/std larger for ToU than Std"] = gt(
        _ratio(v("NMF", "ToU", "special"), v("NMF", "ToU", "std")),
        _ratio(v("NMF", "Std", "special"), v("NMF", "Std", "std")))
    return o


PAPER_ORDERINGS = orderings(paper_getter)


def compare_orderings(o: dict) -> dict:
    comp = {k: o[k] == PAPER_ORDERINGS[k] for k in o if o[k] is not None}
    return {"n_comparable": len(comp), "n_agree": int(sum(comp.values())),
            "disagree": [k for k, ok in comp.items() if not ok]}


def headline_numbers(v) -> dict:
    return {
        "NMF_vs_Features_special_pct": {g: _pct(v("NMF", g, "special"), v("Features", g, "special"))
                                        for g in GROUPS},
        "NMF_vs_Features_std_pct": {g: _pct(v("NMF", g, "std"), v("Features", g, "std"))
                                    for g in GROUPS},
        "NMF_special_over_std": {g: _ratio(v("NMF", g, "special"), v("NMF", g, "std"))
                                 for g in GROUPS},
        "Rd_special_over_std": {g: _ratio(v("Rd", g, "special"), v("Rd", g, "std")) for g in GROUPS},
        "NMF_over_Rd": {g: [_ratio(v("NMF", g, s), v("Rd", g, s)) for s in SERIES] for g in GROUPS},
        "Features_over_Rd": {g: [_ratio(v("Features", g, s), v("Rd", g, s)) for s in SERIES]
                             for g in GROUPS},
    }


def evaluate_run(name: str, F, spec: dict, masks: dict) -> dict:
    mask = spec["mask"]
    alt = "loc" if mask == "auth" else "auth"
    out = {"label": spec["label"], "special_mask": mask,
           "special_mask_slots": masks[mask]["n_slots"], "notes": spec["notes"], "groups": {}}
    for g, gs in spec["groups"].items():
        X = load_matrix(F, gs["rows"])
        out["groups"][g] = evaluate_group(X, gs, mask, alt, masks)
        annotate(out["groups"][g], g)
        del X
        log(f"run {name:17s} {g}: n = {out['groups'][g]['n_households']}, "
            f"methods {', '.join(gs['labelings'])}, special mask {mask}")
    v = run_getter(out)
    out["headline"] = headline_numbers(v)
    out["orderings"] = orderings(v)
    out["orderings_vs_paper"] = compare_orderings(out["orderings"])
    return out


def cross_run_headline(runs: dict) -> dict:
    P = headline_numbers(paper_getter)
    rv = run_getter(runs["authors_recovered"])
    rec = [rv("NMF", "ToU", s) for s in SERIES]
    paper_nmf = list(PAPER["ToU"]["NMF"])
    out = {
        "recovered_partition_NMF_ToU": {
            "S_identified": rec, "paper": paper_nmf,
            "ratio_to_paper": [a / b for a, b in zip(rec, paper_nmf)],
            "rel_diff_pct": [100 * (a / b - 1) for a, b in zip(rec, paper_nmf)],
            "verifier_values": list(VERIFIER_RECOVERED),
            "matches_verifier_within_0.01pct": bool(all(abs(a / b - 1) < 1e-4
                                                        for a, b in zip(rec, VERIFIER_RECOVERED))),
        },
        "NMF_vs_Features_ToU_special_pct": {
            "paper": P["NMF_vs_Features_special_pct"]["ToU"],
            **{r: runs[r]["headline"]["NMF_vs_Features_special_pct"]["ToU"] for r in "abc"},
            "authors_recovered_NMF_vs_run_c_Features": _pct(
                rv("NMF", "ToU", "special"), run_getter(runs["c"])("Features", "ToU", "special")),
            "note": "authors_recovered_NMF_vs_run_c_Features: the authors' exact NMF partition "
                    "against our features clustering of the same 1,007 households, same mask",
        },
        "NMF_vs_Features_Std_special_pct": {
            "paper": P["NMF_vs_Features_special_pct"]["Std"],
            **{r: runs[r]["headline"]["NMF_vs_Features_special_pct"]["Std"] for r in "abc"}},
        "NMF_special_over_std": {"paper": P["NMF_special_over_std"],
                                 **{r: runs[r]["headline"]["NMF_special_over_std"] for r in RUN_ORDER}},
        "Rd_special_over_std": {"paper": P["Rd_special_over_std"],
                                **{r: runs[r]["headline"]["Rd_special_over_std"] for r in RUN_ORDER}},
        "Features_Std_over_Rd_Std": {"paper": P["Features_over_Rd"]["Std"],
                                     **{r: runs[r]["headline"]["Features_over_Rd"]["Std"]
                                        for r in "abc"}},
        "paper_Features_Std_z_in_our_Rd_Std": {
            r: [runs[r]["groups"]["Std"]["methods"]["Features"][s]["paper_z_in_Rd"] for s in SERIES]
            for r in "abc"},
        "paper_orderings": PAPER_ORDERINGS,
        "orderings_vs_paper": {r: runs[r]["orderings_vs_paper"] for r in RUN_ORDER},
    }
    rng = {}
    for m in METRICS:
        vals = [runs[r]["groups"][g]["methods"]["Rd"][s][m]["mean"]
                for r in RUN_ORDER for g in runs[r]["groups"] for s in SERIES]
        rng[m] = [min(vals), max(vals)]
    out["Rd_mean_range_over_runs_groups_series"] = rng
    out["paper_Rd_range"] = [min(min(PAPER[g]["Rd"]) for g in GROUPS),
                             max(max(PAPER[g]["Rd"]) for g in GROUPS)]
    return out


def mask_sensitivity(runs: dict, masks: dict) -> dict:
    out = {"description": "each run's special series recomputed with the other special mask "
                          "(S_identified); raw/std are unaffected"}
    for r in RUN_ORDER:
        rr = runs[r]
        alt = {g: rr["groups"][g]["special_alt_mask"]["S_identified_special"] for g in rr["groups"]}
        alt_mask = next(iter(rr["groups"].values()))["special_alt_mask"]["mask"]
        v = run_getter(rr)

        def va(method, group):
            x = alt.get(group, {}).get(method)
            return None if x is None else _val(x)

        out[r] = {
            "run_mask": rr["special_mask"], "alt_mask": alt_mask,
            "alt_mask_slots": masks[alt_mask]["n_slots"],
            "S_identified_special_alt": alt,
            "S_identified_special_run_mask": {g: {m: _val(rr["groups"][g]["methods"][m]["special"][
                "S_identified"]) for m in rr["groups"][g]["methods"]} for g in rr["groups"]},
            "NMF_vs_Features_special_pct": {
                "run_mask": rr["headline"]["NMF_vs_Features_special_pct"],
                "alt_mask": {g: _pct(va("NMF", g), va("Features", g)) for g in GROUPS}},
            "NMF_special_over_std": {
                "run_mask": rr["headline"]["NMF_special_over_std"],
                "alt_mask": {g: _ratio(va("NMF", g), v("NMF", g, "std")) for g in GROUPS}},
        }
    return out


# ------------------------------------------------------------------------------ cross-check
def cross_check(runs: dict, clustering_json) -> dict:
    comps, exact, sources = [], [], {}

    def ours(run, group, method, series, metric="S_identified", stat="mean"):
        e = runs[run]["groups"][group]["methods"][method][series][metric]
        return e[stat] if isinstance(e, dict) else e

    def add(source, key, ref, our):
        ref, our = float(ref), float(our)
        rel = 100.0 * (our - ref) / abs(ref) if ref != 0 else (0.0 if our == ref else None)
        comps.append({"source": source, "key": key, "reference": ref, "ours": our,
                      "rel_diff_pct": rel})

    def add_exact(source, key, ref, our):
        exact.append({"source": source, "key": key, "reference": ref, "ours": our,
                      "equal": bool(ref == our)})

    def load(name):
        p = VERIFY_DIR / f"{name}.json"
        if not p.exists():
            sources[name] = "missing"
            return None
        sources[name] = str(p.relative_to(REPRO)).replace("\\", "/")
        return json.loads(p.read_text(encoding="utf-8"))

    core = load("t3_core")
    if core:
        mmap = {"V1_norm_wnum_meanden": "S_identified", "D2_eq345": "S_eq345_printed",
                "D1_sklearn": "CH_sklearn"}
        cmap = {("ToU", "NMF_authors"): ("authors_recovered", "ToU", "NMF"),
                ("ToU", "Rd"): ("authors_recovered", "ToU", "Rd"),
                ("Std", "Rd"): ("authors_recovered", "Std", "Rd"),
                ("ToU", "NMF_runc"): ("c", "ToU", "NMF"),
                ("ToU", "Features_runc"): ("c", "ToU", "Features"),
                ("Std", "NMF_runc"): ("c", "Std", "NMF"),
                ("Std", "Features_runc"): ("c", "Std", "Features")}
        for (g, name), (run, grp, meth) in cmap.items():
            for s in SERIES:
                for vk, mk in mmap.items():
                    ref = core[g][name][s][vk]
                    if meth == "Rd":
                        add("t3_core", f"{g}.{name}.{s}.{vk}.mean", ref[0], ours(run, grp, meth, s, mk))
                        add("t3_core", f"{g}.{name}.{s}.{vk}.sd", ref[1],
                            ours(run, grp, meth, s, mk, "sd"))
                    else:
                        add("t3_core", f"{g}.{name}.{s}.{vk}", ref, ours(run, grp, meth, s, mk))
    detail = load("t3_v1_detail")
    if detail:
        key = "V1 f*sum(n_l*b)/sum(w/n_l)"
        for s in ("raw", "std"):
            add("t3_v1_detail", f"{s}.V1", detail[s][key], ours("authors_recovered", "ToU", "NMF", s))
        for prefix, mask_name in (("special_auth", "auth"), ("special_loc", "loc")):
            k = next((x for x in detail if x.startswith(prefix) and x[len(prefix):].isdigit()), None)
            if k is None:
                continue
            gr = runs["authors_recovered"]["groups"]["ToU"]
            if mask_name == gr["special_mask"]:
                our, slots = ours("authors_recovered", "ToU", "NMF", "special"), gr["n_slots"]["special"]
            else:
                our = gr["special_alt_mask"]["S_identified_special"]["NMF"]
                slots = gr["special_alt_mask"]["n_slots"]
            add("t3_v1_detail", f"{k}.V1", detail[k][key], our)
            add_exact("t3_v1_detail", f"{k}.n_slots", int(k[len(prefix):]), int(slots))
    v1runs = load("t3_v1_runs")
    if v1runs:
        omap = {"Features>NMF raw": "Features>NMF raw {g}", "NMF>Features std": "NMF>Features std {g}",
                "NMF special>std": "NMF special>std {g}", "NMF>Rd all": "NMF>Rd {g} (all series)"}
        for run in "abc":
            for g in GROUPS:
                d = v1runs[run][g]
                for col, meth in (("nmf", "NMF"), ("nmf_pam", "NMF_PAM"), ("features", "Features")):
                    for i, s in enumerate(SERIES):
                        add("t3_v1_runs", f"{run}.{g}.{col}.{s}", d[col][i], ours(run, g, meth, s))
                for i, s in enumerate(SERIES):
                    add("t3_v1_runs", f"{run}.{g}.Rd_mean.{s}", d["Rd_mean"][i], ours(run, g, "Rd", s))
                    add("t3_v1_runs", f"{run}.{g}.Rd_sd.{s}", d["Rd_sd"][i],
                        ours(run, g, "Rd", s, "S_identified", "sd"))
                h = runs[run]["headline"]
                add("t3_v1_runs", f"{run}.{g}.NMF_vs_Features_special_pct",
                    d["NMF_vs_Features_special_pct"], h["NMF_vs_Features_special_pct"][g])
                add("t3_v1_runs", f"{run}.{g}.NMF_special_over_std", d["NMF_special_over_std"],
                    h["NMF_special_over_std"][g])
                add_exact("t3_v1_runs", f"{run}.{g}.special_mask_slots", int(d["special_mask_slots"]),
                          int(runs[run]["special_mask_slots"]))
                add_exact("t3_v1_runs", f"{run}.{g}.n", int(d["n"]),
                          int(runs[run]["groups"][g]["n_households"]))
                for vk, ok in omap.items():
                    if vk in d.get("orderings", {}):
                        add_exact("t3_v1_runs", f"{run}.{g}.orderings.{vk}", bool(d["orderings"][vk]),
                                  runs[run]["orderings"][ok.format(g=g)])
    rdseed = load("t3_rd_seed")
    if rdseed and "float32" in rdseed:
        for i, s in enumerate(SERIES):
            add("t3_rd_seed", f"float32.{s} (verifier normalised in float32)", rdseed["float32"][i],
                ours("authors_recovered", "ToU", "NMF", s))
    # secondary: D1 (sklearn) and D2 (eqs. 3-5) of scripts/04_clustering.py
    t3 = (clustering_json or {}).get("table3") if isinstance(clustering_json, dict) else None
    if t3:
        sources["clustering_results"] = "outputs/clustering_results.json (table3, D1 / D2)"
        dmap = {"D1": "CH_sklearn", "D2": "S_eq345_printed"}
        recp = t3.get("recovered_partition", {}).get("NMF ToU (recovered authors' partition)")
        if recp:
            for s in SERIES:
                for d, mk in dmap.items():
                    add("clustering_results", f"recovered.NMF ToU.{s}.{d}", recp[s][d],
                        ours("authors_recovered", "ToU", "NMF", s, mk))
        for run in "abc":
            tab = t3.get("ours", {}).get(run, {}).get("table", {})
            for col, ent in tab.items():
                meth, g = col.rsplit(" ", 1)
                meth = meth.replace("NMF-PAM", "NMF_PAM")
                if g not in GROUPS or meth not in ("NMF", "NMF_PAM", "Features", "Rd"):
                    continue
                for s in SERIES:
                    for d, mk in dmap.items():
                        x = ent[s][d]
                        if isinstance(x, dict):
                            add("clustering_results", f"{run}.{col}.{s}.{d}.mean", x["mean"],
                                ours(run, g, meth, s, mk))
                            add("clustering_results", f"{run}.{col}.{s}.{d}.sd", x["sd"],
                                ours(run, g, meth, s, mk, "sd"))
                        else:
                            add("clustering_results", f"{run}.{col}.{s}.{d}", x,
                                ours(run, g, meth, s, mk))
    else:
        sources["clustering_results"] = "missing"
    rel = [abs(c["rel_diff_pct"]) for c in comps if c["rel_diff_pct"] is not None]
    by_source = {}
    for c in comps:
        b = by_source.setdefault(c["source"], {"n": 0, "max_abs_rel_diff_pct": 0.0})
        b["n"] += 1
        if c["rel_diff_pct"] is not None:
            b["max_abs_rel_diff_pct"] = max(b["max_abs_rel_diff_pct"], abs(c["rel_diff_pct"]))
    above = [c for c in comps if c["rel_diff_pct"] is None or abs(c["rel_diff_pct"]) > CROSSCHECK_TOL_PCT]
    return {
        "tolerance_pct": CROSSCHECK_TOL_PCT, "sources": sources,
        "n_compared": len(comps), "max_abs_rel_diff_pct": max(rel) if rel else None,
        "n_above_tolerance": len(above), "above_tolerance": above, "by_source": by_source,
        "n_exact_fields": len(exact), "n_exact_equal": int(sum(e["equal"] for e in exact)),
        "exact_mismatches": [e for e in exact if not e["equal"]],
        "comparisons": comps, "exact_fields": exact,
    }


# ----------------------------------------------------------------------------------- output
def csv_rows(runs: dict, masks: dict) -> list:
    rows = []
    for r in RUN_ORDER:
        for g, gres in runs[r]["groups"].items():
            for method, entry in gres["methods"].items():
                for s in SERIES:
                    e = entry[s]
                    is_rd = method == "Rd"
                    row = {
                        "run": r, "group": g, "method": method, "series": s,
                        "special_mask": gres["special_mask"] if s == "special" else "",
                        "n_households": gres["n_households"], "n_slots": gres["n_slots"][s],
                        "n_labelings": entry["n_labelings"],
                        "sizes": "" if is_rd else "/".join(str(x) for x in entry["sizes"]),
                        "S_identified": _val(e["S_identified"]),
                        "S_identified_sd": e["S_identified"]["sd"] if is_rd else None,
                        "paper_column": e["paper_column"], "paper": e["paper"],
                        "ratio_to_paper": e["ratio_to_paper"], "S_over_Rd": e.get("S_over_Rd"),
                        "paper_over_paper_Rd": e["paper_over_paper_Rd"],
                        "paper_z_in_Rd": e.get("paper_z_in_Rd"),
                    }
                    for m in ("S_eq345_printed", "CH_sklearn"):
                        row[m] = _val(e[m])
                        row[f"{m}_sd"] = e[m]["sd"] if is_rd else None
                    rows.append(row)
    return rows


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _fmt(x, spec):
    return "-" if x is None else format(x, spec)


def print_report(res: dict) -> None:
    runs = res["runs"]
    srcs = [("paper", None)] + [(SHORT[r], r) for r in RUN_ORDER]

    def entry(run, g, m, s):
        try:
            return runs[run]["groups"][g]["methods"][m][s]
        except KeyError:
            return None

    cols = [(g, m, w) for g in GROUPS for m, w in (("Rd", 12), ("Features", 9), ("NMF", 8))]
    head = (f"{'series':8s} {'src':6s}|" + "".join(
        f" {(('Feat' if m == 'Features' else m) + ' ' + g):>{w}s}" + (" |" if m == "NMF" and g == "ToU" else "")
        for g, m, w in cols))
    print()
    print("TABLE 3, identified CH: S = (n-k)/(k-1) * sum_l n_l ||Y(C_l)-Y(I)|| / "
          "sum_l mean_{i in C_l} ||Y(i)-Y(C_l)||")
    print("rec = the authors' recovered NMF partition (Std: random labels only); a/b/c = our runs; "
          "Rd = mean (SD) over 20 seeds")
    print(f"special mask: rec, c = the authors' dummies ({res['special_masks']['auth']['n_slots']} "
          f"slots); a, b = Tariffs.xlsx 'loc' ({res['special_masks']['loc']['n_slots']} slots)")
    for block in ("value", "ratio"):
        print()
        print("S_identified" if block == "value" else "ours / paper")
        print(head)
        print("-" * len(head))
        for i, s in enumerate(SERIES):
            for name, run in srcs:
                if block == "ratio" and run is None:
                    continue
                cells = []
                for g, m, w in cols:
                    if run is None:
                        txt = str(PAPER[g][m][i])
                    else:
                        e = entry(run, g, m, s)
                        if e is None:
                            txt = "-"
                        elif block == "ratio":
                            txt = f"{e['ratio_to_paper']:.3f}"
                        elif m == "Rd":
                            txt = f"{e['S_identified']['mean']:.0f} ({e['S_identified']['sd']:.0f})"
                        else:
                            txt = f"{e['S_identified']:.0f}"
                    cells.append(f" {txt:>{w}s}" + (" |" if m == "NMF" and g == "ToU" else ""))
                print(f"{s:8s} {name:6s}|" + "".join(cells))
    h = res["headline"]
    rec = h["recovered_partition_NMF_ToU"]
    print()
    print("Recovered partition, NMF ToU raw/std/special: "
          + " / ".join(f"{x:.1f}" for x in rec["S_identified"])
          + "  (paper " + " / ".join(str(x) for x in rec["paper"]) + "; ours/paper "
          + " / ".join(f"{x:.5f}" for x in rec["ratio_to_paper"]) + "; verifier "
          + " / ".join(f"{x:.1f}" for x in rec["verifier_values"])
          + (", match)" if rec["matches_verifier_within_0.01pct"] else ", MISMATCH)"))
    names = ["paper", "rec", "a", "b", "c"]
    print()
    print(f"{'headline':40s}" + "".join(f"{n:>9s}" for n in names))

    def line(label, vals, spec):
        print(f"{label:40s}" + "".join(f"{_fmt(x, spec):>9s}" for x in vals))

    nf = h["NMF_vs_Features_ToU_special_pct"]
    line("NMF vs Features, ToU special (%)",
         [nf["paper"], nf["authors_recovered_NMF_vs_run_c_Features"], nf["a"], nf["b"], nf["c"]], "+.1f")
    nfs = h["NMF_vs_Features_Std_special_pct"]
    line("NMF vs Features, Std special (%)", [nfs["paper"], None, nfs["a"], nfs["b"], nfs["c"]], "+.1f")
    so = h["NMF_special_over_std"]
    for g in GROUPS:
        line(f"NMF special/std, {g}", [so["paper"][g]] + [so[r][g] for r in RUN_ORDER], ".3f")
    rso = h["Rd_special_over_std"]
    for g in GROUPS:
        line(f"Rd special/std, {g}", [rso["paper"][g]] + [rso[r][g] for r in RUN_ORDER], ".3f")
    fr = h["Features_Std_over_Rd_Std"]
    line("Features Std / Rd Std, raw", [fr["paper"][0], None, fr["a"][0], fr["b"][0], fr["c"][0]], ".2f")
    line("Features Std / Rd Std, special",
         [fr["paper"][2], None, fr["a"][2], fr["b"][2], fr["c"][2]], ".2f")
    ov = h["orderings_vs_paper"]
    n_p = len(PAPER_ORDERINGS)
    print(f"{'paper orderings reproduced':40s}" + f"{f'{n_p}/{n_p}':>9s}" + "".join(
        f"{str(ov[r]['n_agree']) + '/' + str(ov[r]['n_comparable']):>9s}" for r in RUN_ORDER))
    dis = sorted({k for r in RUN_ORDER for k in ov[r]["disagree"]})
    print("  rec: authors' NMF vs our run-c Features (same households and mask); orderings not "
          "reproduced: " + (", ".join(dis) if dis else "none"))
    # contrast: eqs. 3-5 as printed and sklearn
    print()
    print("Contrast, same partitions:  eqs. 3-5 as printed (squared norms) | sklearn "
          "calinski_harabasz_score   [raw / std / special]")

    def trio(run, g, m, metric):
        e = [entry(run, g, m, s) for s in SERIES]
        return " / ".join(f"{_val(x[metric]):.2f}" for x in e)

    lines = [("authors_recovered", "ToU", "NMF"), ("authors_recovered", "ToU", "Rd"),
             ("authors_recovered", "Std", "Rd")]
    lines += [(r, "ToU", m) for r in "abc" for m in ("NMF", "Features", "Rd")]
    for run, g, m in lines:
        lab = f"{SHORT[run]} {g} {m}{' (mean)' if m == 'Rd' else ''}"
        print(f"  {lab:20s} {trio(run, g, m, 'S_eq345_printed'):>26s} | {trio(run, g, m, 'CH_sklearn'):>26s}")
    rr = h["Rd_mean_range_over_runs_groups_series"]
    print(f"  Rd means over all runs/groups/series: identified {rr['S_identified'][0]:.0f}-"
          f"{rr['S_identified'][1]:.0f} (paper {h['paper_Rd_range'][0]}-{h['paper_Rd_range'][1]}), "
          f"eqs. 3-5 {rr['S_eq345_printed'][0]:.3f}-{rr['S_eq345_printed'][1]:.3f}, "
          f"sklearn {rr['CH_sklearn'][0]:.3f}-{rr['CH_sklearn'][1]:.3f}")
    ms = res["mask_sensitivity"]
    print()
    print("Special-mask sensitivity, NMF vs Features ToU special (%), run mask -> other mask: "
          + "; ".join(f"{r} {ms[r]['run_mask']} {_fmt(ms[r]['NMF_vs_Features_special_pct']['run_mask']['ToU'], '+.1f')}"
                      f" -> {ms[r]['alt_mask']} {_fmt(ms[r]['NMF_vs_Features_special_pct']['alt_mask']['ToU'], '+.1f')}"
                      for r in "abc"))
    g = ms["authors_recovered"]
    print(f"  rec NMF ToU special: auth {g['S_identified_special_run_mask']['ToU']['NMF']:.1f}, "
          f"loc {g['S_identified_special_alt']['ToU']['NMF']:.1f}")
    cc = res["cross_check"]
    print()
    print(f"Cross-check ({', '.join(k for k, v in cc['sources'].items() if v != 'missing')}): "
          f"{cc['n_compared']} numbers, max |rel diff| {_fmt(cc['max_abs_rel_diff_pct'], '.2e')} %, "
          f"{cc['n_above_tolerance']} above {cc['tolerance_pct']} %; exact fields "
          f"{cc['n_exact_equal']}/{cc['n_exact_fields']} equal"
          + ("" if all(v != "missing" for v in cc["sources"].values()) else
             f"; missing: {', '.join(k for k, v in cc['sources'].items() if v == 'missing')}"))
    for c in cc["above_tolerance"][:10]:
        print(f"  ABOVE TOLERANCE {c['source']} {c['key']}: reference {c['reference']:.6g}, "
              f"ours {c['ours']:.6g}, {_fmt(c['rel_diff_pct'], '+.3f')} %")
    sc = res["self_checks"]
    print(f"Self-checks: sklearn's function vs formula max rel diff "
          f"{sc['max_rel_diff_sklearn_function_vs_formula']:.1e} ({sc['sklearn_calls']} calls); "
          f"loop reference vs vectorized S max rel diff "
          f"{sc['max_rel_diff_reference_loop_vs_vectorized']:.1e} ({sc['reference_checks']} checks)")


# ------------------------------------------------------------------------------------- main
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    masks = load_masks()
    log(f"masks: auth {masks['auth']['n_slots']} slots ({masks['auth']['n_low']} Low, "
        f"{masks['auth']['n_high']} High), loc {masks['loc']['n_slots']} slots")
    sel = pd.read_parquet(SELECTION_PARQUET)
    labels, clustering_json = run_metadata()
    F = np.load(FILLED_NPY, mmap_mode="r")
    if F.shape[1] != N_SLOTS:
        raise ValueError(f"{FILLED_NPY} has {F.shape[1]} columns, expected {N_SLOTS}")
    runs = {"authors_recovered": evaluate_run("authors_recovered", F, recovered_spec(sel), masks)}
    for r in "abc":
        runs[r] = evaluate_run(r, F, run_spec(r, labels[r]), masks)
    res = {
        "description": "Table 3 of Bregere & Bessa (2020) under the Calinski-Harabasz definition "
                       "identified in phase-3 verification (spec section 7), with eqs. 3-5 as "
                       "printed and sklearn's score for contrast.",
        "script": "scripts/06_table3.py",
        "definitions": {
            "S_identified": "(|I|-k)/(k-1) * sum_l |C_l| ||Y(C_l)-Y(I)|| / sum_l (1/|C_l|) sum_{i in C_l} "
                            "||Y(i)-Y(C_l)||  (Euclidean norms, NOT squared; |C_l|-weighted between term)",
            "S_eq345_printed": "(|I|-k) sum_l ||Y(C_l)-Y(I)||^2 / ((k-1) sum_l (1/|C_l|) sum_{i in C_l} "
                               "||Y(i)-Y(C_l)||^2)  (the paper's eqs. 3-5 as printed)",
            "CH_sklearn": "sklearn.metrics.calinski_harabasz_score(Y, labels) = (|I|-k) sum_l |C_l| "
                          "||Y(C_l)-Y(I)||^2 / ((k-1) sum_i ||Y(i)-Y(C_l(i))||^2); evaluated from the "
                          "shared sufficient statistics; sklearn's function itself is called on every "
                          f"single clustering and on the first {N_SKLEARN_SEEDS} random seeds of each "
                          "group and series and must agree to 1e-8 (self_checks)",
            "computation": "cluster means by a one-hot matrix product; squared distances to the own "
                           "centroid by ||Y(i)||^2 - 2 Y(i).Y(C_l) + ||Y(C_l)||^2; one literal "
                           "cluster-by-cluster recomputation of S_identified per group (self_checks)",
            "k": K,
            "Y(I), Y(C_l)": "mean record vector of all households / of cluster C_l",
            "random_labels_sd": "population SD over the 20 seeds (numpy std, ddof = 0), as in "
                                "clustering_results.json and the verifier",
            "null_approx_S_identified": "random-label approximation (n-k)/(k-1) * sqrt(n(k-1))/k * rho, "
                                        "rho = RMS / mean of ||Y(i)-Y(I)||; grows like sqrt(n)",
        },
        "series": {"raw": "the household's 17,520 filled half-hourly records of 2013 (kWh), "
                          "data/lcl2013_kwh_filled.npy, the authors' slot convention",
                   "std": "raw / the household's mean over the 17,520 records ('standardized')",
                   "special": "the std records at the Low-or-High slots of the special mask only "
                              "('special tariff - standardized'); Std households get the same "
                              "(placebo) mask"},
        "special_masks": {k: {kk: vv for kk, vv in v.items() if kk != "mask"} for k, v in masks.items()},
        "methods": METHOD_DOC,
        "paper": {"table3": {g: {m: list(v) for m, v in PAPER[g].items()} for g in GROUPS},
                  "sizes_ToU": PAPER_SIZES_TOU,
                  "headline": headline_numbers(paper_getter), "orderings": PAPER_ORDERINGS},
        "run_metadata_labels": labels,
        "runs": runs,
    }
    res["headline"] = cross_run_headline(runs)
    res["mask_sensitivity"] = mask_sensitivity(runs, masks)
    res["cross_check"] = cross_check(runs, clustering_json)
    res["self_checks"] = dict(SELF_CHECKS)
    res["self_checks"]["stored_random_labels_follow_documented_rule"] = {
        r: runs[r]["notes"]["random_labels_follow_documented_rule"] for r in "abc"}
    res["self_checks"]["run_mask_confirmed_by_metadata_label"] = {
        r: runs[r]["notes"]["mask_confirmed_by_metadata_label"] for r in "abc"}
    res["runtime_s"] = round(time.time() - LOG_T0, 1)
    write_json(res, OUT_JSON)
    pd.DataFrame(csv_rows(runs, masks)).to_csv(OUT_CSV, index=False)
    print_report(res)
    print()
    log(f"wrote {OUT_JSON.relative_to(REPRO)} and {OUT_CSV.relative_to(REPRO)}")


if __name__ == "__main__":
    main()
