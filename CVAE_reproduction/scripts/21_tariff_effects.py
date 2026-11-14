"""Tariff-scenario effects of both generators and a mechanism test of the CVAE's side/rebound effects
(paper Sec. VII-C, Figs. 6 and 8). Logic in lib/effects.py.

Usage (from PROJECT or REPRO):  python -P scripts/21_tariff_effects.py --track A|B [--no-figures] [--boot 2000]

Scenario windows (lib.effects, the single source of truth; REPRO_SPEC section 7): Low = slots 9-19 = 04:00-09:30 GMT,
High = slots 39-44 = 19:00-22:00 GMT (slot h = the half-hour ending at h x 30 min GMT). Phases 1-2 used 9-18 / 39-43;
their outputs are in outputs/archive_windows_9-18_39-43/.

Stages
  effects    outputs/<track>/{cvae,gam}_scen_{Normal,Low,High}_c<c>.npy -> per model x cluster: in-window effect
             (Low 9-19, High 39-44; kWh and % of Normal), side effects 2 and 4 slots before/after each window
             (High post4 = 45-47: only 3 slots remain in the day), rebound windows (after Low 20-31, 32-47; around
             High 24-38 before, 45-47 after), daily energy change; mean over the 200 samples, then over the 92 test
             days; 95% bootstrap CI over days
             -> outputs/<track>/effects_summary.json, outputs/figures/<track>_effects_profiles.png
  restarts   claimed effects across the 50 CVAE restarts per cluster: (a) summary of the phase-1 table
             outputs/<track>/cvae_scenario_effects_all_restarts.csv; (b) the same restarts re-run with this script's
             region definitions (outputs/<track>/models/cvae_c<c>_s<ss>.pt, same z draws)
             -> effects_summary.json["restarts"], outputs/<track>/effects_all_restarts_regions.csv,
                outputs/figures/<track>_effects_restarts.png
  mechanism  (i) Low/High blocks in the training dummies; (ii) the selected decoders regenerated with the most
             frequent real training blocks / day patterns (common random numbers) and their effects outside the
             block; (iii) data-based response around real blocks on the cluster aggregates of all days (event day
             minus matched all-Normal days of the same day type within +-14 days, same slot; lags around block start
             and end; bootstrap over event days); (iv) the CVAE's implied effect on the same event days (own
             conditions) vs the data -> outputs/<track>/mechanism_{blocks,cvae_patterns,data_eventstudy,summary}.json,
             outputs/figures/<track>_effects_mechanism.png, <track>_effects_eventstudy.png

Inputs per track: A = UPSTREAM/data/{test,train,power_consump}_nmf.feather; B = REPRO/data/{test,train,power_consump}
_ours.parquet. Missing inputs are reported and the dependent parts skipped (the script still exits 0).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction"
if REPRO not in sys.path:
    sys.path.insert(0, REPRO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import effects as ef  # noqa: E402

AUTHORS = REPRO + "/data/authors"  # copy of the authors' repo data/*.feather (byte-identical; read as data only)
FIGS = REPRO + "/outputs/figures"
MODELS = ("cvae", "gam")
H = 48
GEN_SEED = 20200801          # phase-1 CVAE generation stream: default_rng([GEN_SEED, cluster]) -> (200, 92, d) z
N_SAMPLES = 200
N_RESTARTS = 50

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
MODEL_COLOR = {"cvae": BLUE, "gam": ORANGE}
MODEL_LABEL = {"cvae": "CVAE", "gam": "GAM"}

# names derived from the scenario windows (lib.effects)
LW, HW = ef.LOW_WINDOW, ef.HIGH_WINDOW
LSPAN, HSPAN = f"{LW[0]}_{LW[-1]}", f"{HW[0]}_{HW[-1]}"            # '9_19', '39_44'
PAPER_LOW, PAPER_HIGH = f"paper_Low_{LSPAN}", f"paper_High_{HSPAN}"
HIGH_BLOCK, LOW_BLOCK = f"High_{HSPAN}", f"Low_{LSPAN}"
KEY_LREB = ef.KEY_LOW_REBOUND                                         # 'rebound_20_31'
LREB_SPAN = KEY_LREB.replace("rebound_", "")                          # '20_31'
# union of the two most frequent training High blocks that contain the paper's High window (33-44, 35-46)
COOC_SLOTS = list(range(33, 47))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def track_paths(track: str) -> dict:
    t = track.upper()
    if t == "A":
        d = AUTHORS
        return {"tag": "trackA", "out": REPRO + "/outputs/trackA", "test": d + "/test_nmf.feather",
                "train": d + "/train_nmf.feather", "power": d + "/power_consump_nmf.feather"}
    return {"tag": "trackB", "out": REPRO + "/outputs/trackB", "test": REPRO + "/data/test_ours.parquet",
            "train": REPRO + "/data/train_ours.parquet", "power": REPRO + "/data/power_consump_ours.parquet"}


def read_table(path: str):
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"])
    df["cluster"] = df["cluster"].astype(int)
    return df


def tariff_names(cond_cols=None) -> tuple[list, list]:
    if cond_cols is not None:
        il, ih = ef.tariff_col_index(cond_cols)
        return [cond_cols[j] for j in il], [cond_cols[j] for j in ih]
    return [f"low_{h:02d}" for h in range(H)], [f"high_{h:02d}" for h in range(H)]


def cluster_block(df, c: int):
    return df.loc[df["cluster"] == c].sort_values("date", kind="stable").reset_index(drop=True)


def dummies(df, low_cols, high_cols) -> tuple[np.ndarray, np.ndarray]:
    """Tariff dummies of a table (identical for every cluster: checked) in date order."""
    cl = sorted(df["cluster"].unique())
    b0 = cluster_block(df, cl[0])
    L, Hh = b0[low_cols].to_numpy(dtype=float), b0[high_cols].to_numpy(dtype=float)
    for c in cl[1:]:
        b = cluster_block(df, c)
        if not (np.array_equal(b[low_cols].to_numpy(dtype=float), L) and
                np.array_equal(b[high_cols].to_numpy(dtype=float), Hh)):
            raise ValueError(f"tariff dummies differ between clusters {cl[0]} and {c}")
    return L, Hh


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ------------------------------------------------------------------------------------------------------------------
# stage 1: scenario effects
# ------------------------------------------------------------------------------------------------------------------
def stage_effects(P, clusters, B) -> dict:
    res, missing = {}, []
    for m in MODELS:
        for c in clusters:
            paths = {s: P["out"] + f"/{m}_scen_{s}_c{c}.npy" for s in ("Normal", "Low", "High")}
            if not all(os.path.exists(p) for p in paths.values()):
                missing += [p for p in paths.values() if not os.path.exists(p)]
                continue
            A = {s: np.load(p) for s, p in paths.items()}
            for s, a in A.items():
                if a.ndim != 3 or a.shape[2] != H or not np.isfinite(a).all():
                    raise ValueError(f"{paths[s]}: bad array {a.shape}")
            e = ef.scenario_effects(A["Normal"], {"Low": A["Low"], "High": A["High"]}, B=B, seed=1000 + c)
            # CRN check: per-sample differences outside the window (GAM: exactly 0 by construction)
            e["crn_check_max_abs_sample_diff_outside_window"] = {
                k: float(np.abs((A[k] - A["Normal"])[:, :, ef.scenario_regions(k)["outside_window"]]).max())
                for k in ("Low", "High")}
            res[f"{m}_c{c}"] = e
            lo, hi = e["Low"]["regions"], e["High"]["regions"]
            log(f"[effects] {m} c{c}: Low in {lo['in_window']['pct']:+.1f}% [{lo['in_window']['pct_ci95'][0]:+.1f}, "
                f"{lo['in_window']['pct_ci95'][1]:+.1f}], post4 {lo['post4']['pct']:+.1f}%, reb{LREB_SPAN.replace('_', '-')} "
                f"{lo[KEY_LREB]['pct']:+.1f}%, reb32-47 "
                f"{lo['rebound_32_47']['pct']:+.1f}% | High in {hi['in_window']['pct']:+.1f}%, pre2 {hi['pre2']['pct']:+.1f}%, "
                f"pre4 {hi['pre4']['pct']:+.1f}%, post2 {hi['post2']['pct']:+.1f}%, post4(45-47) {hi['post4']['pct']:+.1f}%, "
                f"aft24-34 {hi['afternoon_24_34']['pct']:+.1f}% | day {lo['daily_energy']['pct']:+.1f}% / "
                f"{hi['daily_energy']['pct']:+.1f}%")
    table = []
    for key, e in res.items():
        m, c = key.split("_c")
        for scen in ("Low", "High"):
            for rname, st in e[scen]["regions"].items():
                if not isinstance(st, dict) or "kwh" not in st:
                    continue
                table.append({"model": m, "cluster": int(c), "scenario": scen, "region": rname,
                              "kwh": st["kwh"], "kwh_ci95_lo": st["kwh_ci95"][0], "kwh_ci95_hi": st["kwh_ci95"][1],
                              "pct": st["pct"], "pct_ci95_lo": st["pct_ci95"][0], "pct_ci95_hi": st["pct_ci95"][1],
                              "frac_days_negative": st["frac_units_negative"],
                              "claimed_sign": st.get("claimed_sign"), "claim_status": st.get("claim_status")})
    return {"per_model_cluster": res, "table": table, "missing_arrays": missing}


def claim_overview(eff: dict, clusters) -> dict:
    """Status of each claim per model x cluster (appears / weak / absent / contradicted)."""
    out = {}
    for (scen, rname), (sign, text) in ef.CLAIMS.items():
        row = {"text": text, "claimed_sign": sign}
        for m in MODELS:
            row[m] = {str(c): eff[f"{m}_c{c}"][scen]["regions"][rname]["claim_status"]
                      for c in clusters if f"{m}_c{c}" in eff}
            row[m + "_pct"] = {str(c): round(eff[f"{m}_c{c}"][scen]["regions"][rname]["pct"], 2)
                               for c in clusters if f"{m}_c{c}" in eff}
        out[f"{scen}:{rname}"] = row
    return out


# ------------------------------------------------------------------------------------------------------------------
# stage 2: restarts
# ------------------------------------------------------------------------------------------------------------------
PHASE1_CLAIMS = {f"{s}:{r}:pct": sign for (s, r), sign in ef.PHASE1_CLAIMED_SIGN.items()}  # phase-1 column -> sign
PHASE1_REGIONS_TEXT = "phase-1 definitions (lib.effects.phase1_regions): " + "; ".join(
    f"{s} " + ", ".join(f"{r} = slots {ef.span(v)}" for r, v in rr.items() if r != "outside_window")
    for s, rr in ef.phase1_regions().items()) + "; % of Normal over the same slots"


def _count_claims(df, cols_signs: dict, clusters, thr: float = 1.0) -> dict:
    out = {}
    for c in clusters:
        d = df.loc[df["cluster"] == c]
        if d.empty:
            continue
        rec = {"n_restarts": int(len(d))}
        joint = np.ones(len(d), bool)
        joint_thr = np.ones(len(d), bool)
        for col, sign in cols_signs.items():
            v = d[col].to_numpy(dtype=float)
            ok = np.sign(v) == sign
            okt = ok & (np.abs(v) >= thr)
            joint &= ok
            joint_thr &= okt
            rec[col] = {"n_claimed_sign": int(ok.sum()), f"n_claimed_sign_and_abs_ge_{thr:g}pct": int(okt.sum()),
                        "median_pct": float(np.median(v))}
            if "selected" in d.columns and d["selected"].any():
                rec[col]["selected_restart_pct"] = float(d.loc[d["selected"].astype(bool), col].iloc[0])
        rec["all_claims_jointly"] = int(joint.sum())
        rec[f"all_claims_jointly_abs_ge_{thr:g}pct"] = int(joint_thr.sum())
        out[str(c)] = rec
    return out


def stage_restarts(P, clusters, B) -> dict:
    out = {}
    p1 = P["out"] + "/cvae_scenario_effects_all_restarts.csv"
    if os.path.exists(p1):
        df = pd.read_csv(p1)
        missing_cols = [k for k in PHASE1_CLAIMS if k not in df.columns]
        if missing_cols:
            raise RuntimeError(f"{p1} lacks {missing_cols}: rerun the effects/robust stages of the track script with "
                               f"the current windows first")
        out["phase1_table"] = {"source": p1, "regions": PHASE1_REGIONS_TEXT,
                               "counts": _count_claims(df, PHASE1_CLAIMS, clusters)}
    else:
        out["phase1_table"] = {"source": p1, "missing": True}
    # re-run every restart with this script's regions (same z draws as phase 1)
    paths = {(c, s): P["out"] + f"/models/cvae_c{c}_s{s:02d}.pt" for c in clusters for s in range(N_RESTARTS)}
    test = read_table(P["test"])
    if test is None or not any(os.path.exists(p) for p in paths.values()):
        out["recomputed"] = {"skipped": "restart models or test table not found"}
        return out
    import torch
    from lib import cvae_torch as ct
    torch.set_num_threads(1)
    rows = []
    sel = {}
    for c in clusters:
        sp = P["out"] + f"/models/cvae_selected_c{c}.pt"
        if os.path.exists(sp):
            sel[c] = int(ct.load_model(sp).meta["seed"])
    for c in clusters:
        blk = cluster_block(test, c)
        for s in range(N_RESTARTS):
            p = paths[(c, s)]
            if not os.path.exists(p):
                continue
            model = ct.load_model(p)
            cols = model.meta["cond_cols"]
            C = blk[cols].to_numpy(dtype=np.float64)
            eps = np.random.default_rng([GEN_SEED, c]).standard_normal((N_SAMPLES, len(C), model.latent_dim))
            gen = {k: ct.generate(model, ef.pattern_condition(C, cols, *spec), N_SAMPLES, eps=eps)
                   for k, spec in (("Normal", ((), ())), ("Low", (ef.LOW_WINDOW, ())), ("High", ((), ef.HIGH_WINDOW)))}
            e = ef.scenario_effects(gen["Normal"], {"Low": gen["Low"], "High": gen["High"]}, B=B, seed=1000 + c)
            row = {"cluster": c, "seed": s, "selected": bool(sel.get(c) == s)}
            for scen in ("Low", "High"):
                for rname, st in e[scen]["regions"].items():
                    if isinstance(st, dict) and "kwh" in st:
                        row[f"{scen}:{rname}:pct"] = st["pct"]
                        row[f"{scen}:{rname}:kwh"] = st["kwh"]
                        row[f"{scen}:{rname}:status"] = st.get("claim_status")
            rows.append(row)
        log(f"[restarts] c{c}: {sum(1 for r in rows if r['cluster'] == c)} restarts re-run")
    df2 = pd.DataFrame(rows)
    df2.to_csv(P["out"] + "/effects_all_restarts_regions.csv", index=False)
    signs = {f"{scen}:{rname}:pct": sign for (scen, rname), (sign, _) in ef.CLAIMS.items()}
    counts = _count_claims(df2, signs, clusters)
    # also count CI-supported appearances ('appears' = right sign and the day-bootstrap CI excludes 0)
    for c in clusters:
        d = df2.loc[df2["cluster"] == c]
        for (scen, rname), _ in ef.CLAIMS.items():
            col = f"{scen}:{rname}:status"
            counts[str(c)][f"{scen}:{rname}:pct"]["n_appears_ci_excludes_0"] = int((d[col] == "appears").sum())
            counts[str(c)][f"{scen}:{rname}:pct"]["n_contradicted_ci_excludes_0"] = int((d[col] == "contradicted").sum())
    out["recomputed"] = {"csv": P["out"] + "/effects_all_restarts_regions.csv", "counts": counts,
                         "claims": {f"{k[0]}:{k[1]}": v[1] for k, v in ef.CLAIMS.items()}}
    return out


# ------------------------------------------------------------------------------------------------------------------
# stage 3: mechanism
# ------------------------------------------------------------------------------------------------------------------
def pick_patterns(blocks_train: dict) -> dict:
    """Paper scenarios + most frequent real within-day blocks + the most frequent real joint day patterns."""
    pats = {PAPER_HIGH: ([], list(ef.HIGH_WINDOW)), PAPER_LOW: (list(ef.LOW_WINDOW), [])}
    for kind, n_take in (("High", 5), ("Low", 5)):
        taken = 0
        for r in blocks_train[kind]["within_day_runs"]:
            s, e, cnt = r["start"], r["end"], r["count"]
            if cnt < 2 or e - s + 1 < 2:
                continue
            slots = list(range(s, e + 1))
            pats[f"real_{kind}_{s}_{e}"] = (slots, []) if kind == "Low" else ([], slots)
            taken += 1
            if taken == n_take:
                break
    k = 0
    for dp in blocks_train["day_patterns"]:
        if dp["low_runs"] and dp["high_runs"] and dp["count"] >= 2:
            lo = [h for s, e in dp["low_runs"] for h in range(s, e + 1)]
            hi = [h for s, e in dp["high_runs"] for h in range(s, e + 1)]
            name = "joint_" + "+".join(f"L{s}-{e}" for s, e in dp["low_runs"]) + "+" + \
                   "+".join(f"H{s}-{e}" for s, e in dp["high_runs"])
            pats[name] = (lo, hi)
            k += 1
            if k == 3:
                break
    return pats


def pattern_code(lo, hi) -> np.ndarray:
    code = np.zeros(H, dtype=int)
    code[list(lo)] = ef.CODE["Low"]
    code[list(hi)] = ef.CODE["High"]
    return code


def cvae_pattern_responses(P, clusters, test, pats, B) -> dict:
    import torch
    from lib import cvae_torch as ct
    torch.set_num_threads(1)
    out = {}
    for c in clusters:
        mp = P["out"] + f"/models/cvae_selected_c{c}.pt"
        if not os.path.exists(mp):
            continue
        model = ct.load_model(mp)
        cols = model.meta["cond_cols"]
        C = cluster_block(test, c)[cols].to_numpy(dtype=np.float64)
        eps = np.random.default_rng([GEN_SEED, c]).standard_normal((N_SAMPLES, len(C), model.latent_dim))
        Sn = ct.generate(model, ef.pattern_condition(C, cols), N_SAMPLES, eps=eps)
        Nm = Sn.mean(0)
        idx = ef.boot_index(len(C), B, 2000 + c)
        rc = {"selected_seed": int(model.meta["seed"]), "patterns": {}}
        # check against the saved phase-1 scenario arrays (same z draws)
        chk = {}
        for scen, (lo, hi) in (("Normal", ([], [])), ("Low", (ef.LOW_WINDOW, [])), ("High", ([], ef.HIGH_WINDOW))):
            sp = P["out"] + f"/cvae_scen_{scen}_c{c}.npy"
            if os.path.exists(sp):
                S = Sn if scen == "Normal" else ct.generate(model, ef.pattern_condition(C, cols, lo, hi), N_SAMPLES,
                                                            eps=eps)
                chk[scen] = float(np.abs(S - np.load(sp)).max())
        rc["max_abs_diff_vs_saved_scenarios"] = chk
        for name, (lo, hi) in pats.items():
            S = ct.generate(model, ef.pattern_condition(C, cols, lo, hi), N_SAMPLES, eps=eps)
            Dd = S.mean(0) - Nm
            code = pattern_code(lo, hi)
            blocks = [("Low", s, e) for s, e in ef.runs(code == 1)] + [("High", s, e) for s, e in ef.runs(code == 2)]
            rec = {"low_slots": [int(h) for h in lo], "high_slots": [int(h) for h in hi],
                   "profile_diff_kwh": Dd.mean(0).tolist(), "profile_pct": (100 * Dd.mean(0) / Nm.mean(0)).tolist(),
                   "daily_energy": ef.ratio_stats(Dd.sum(1), Nm.sum(1), idx), "blocks": {}}
            for kind, s, e in blocks:
                rec["blocks"][f"{kind}_{s}_{e}"] = ef.block_relative_stats(Dd, Nm, (s, e), code, idx)
            rc["patterns"][name] = rec
        # similarity of the paper-scenario responses to the real-pattern responses (correlation of mean profiles)
        sim = {}
        for paper_name in (PAPER_HIGH, PAPER_LOW):
            a = np.array(rc["patterns"][paper_name]["profile_diff_kwh"])
            sim[paper_name] = {k: float(np.corrcoef(a, np.array(v["profile_diff_kwh"]))[0, 1])
                               for k, v in rc["patterns"].items() if k != paper_name}
        rc["profile_correlation_with_paper_scenarios"] = sim
        # where does the paper High scenario's outside-window response sit?
        a = np.array(rc["patterns"][PAPER_HIGH]["profile_diff_kwh"])
        out_w = [h for h in range(H) if h not in ef.HIGH_WINDOW]
        in_train_hi = [h for h in out_w if h in COOC_SLOTS]
        rc["paper_High_outside_window_mass"] = {
            "sum_kwh_outside_window": float(a[out_w].sum()),
            "sum_kwh_slots_33_46_outside_window": float(a[in_train_hi].sum()),
            "sum_kwh_other_slots": float(a[[h for h in out_w if h not in in_train_hi]].sum()),
            "sum_kwh_in_window": float(a[ef.HIGH_WINDOW].sum()),
            "slots_33_46_outside_window": in_train_hi,
            "uniform_share_of_outside_slots": len(in_train_hi) / len(out_w),
            "note": f"slots 33-46 = union of the two most frequent training High blocks containing "
                    f"{ef.WINDOW_SPAN['High']} (33-44, 35-46); outside the window: "
                    f"{ef.span([h for h in in_train_hi if h < HW[0]])} and {ef.span([h for h in in_train_hi if h > HW[-1]])}"}
        out[str(c)] = rc
        ph = rc["patterns"][PAPER_HIGH]["blocks"][HIGH_BLOCK]
        log(f"[mechanism] CVAE c{c}: paper High {ef.WINDOW_SPAN['High']} in {ph['in_block']['pct']:+.1f}%, pre4 "
            f"{ph['pre4']['pct']:+.1f}%, post4 (45-47) {ph['post4']['pct']:+.1f}%; check vs saved arrays {chk}")
    return out


def data_eventstudy(P, clusters, power, low_cols, high_cols, B) -> dict:
    L, Hh = dummies(power, low_cols, high_cols)
    code = ef.tariff_codes(L, Hh)
    code_f = code.ravel()
    all_normal = code.sum(1) == 0
    out = {"n_days": int(len(L)), "n_all_normal_days": int(all_normal.sum()), "clusters": {}}
    for c in clusters:
        b = cluster_block(power, c)
        Y = b[[f"consumption_{h:02d}" for h in range(H)]].to_numpy(dtype=np.float64)
        wd = b["work_day"].to_numpy(dtype=int)
        Bm, cnt = ef.matched_baseline(Y, wd, all_normal, 14)
        delta = Y - Bm
        rc = {"matched_days_per_day": {"min": int(cnt.min()), "median": float(np.median(cnt)), "max": int(cnt.max()),
                                       "n_days_without_match": int((cnt == 0).sum())}}
        for kind in ("High", "Low"):
            rc[kind] = {
                "all_events": ef.event_study(delta.ravel(), Bm.ravel(), code_f, kind, B=B, seed=3000 + c),
                "same_day_lags_only": ef.event_study(delta.ravel(), Bm.ravel(), code_f, kind, B=B, seed=3000 + c,
                                                     same_day_only=True)}
        # day-level response profiles for the most frequent real day patterns
        prof = {}
        pats = {}
        for d in range(len(L)):
            if code[d].any():
                key = (tuple(ef.runs(L[d])), tuple(ef.runs(Hh[d])))
                pats.setdefault(key, []).append(d)
        for key, days in sorted(pats.items(), key=lambda kv: -len(kv[1])):
            if len(days) < 2:
                continue
            days = np.array(days)
            dd = delta[days]
            ok = np.isfinite(dd).all(1)
            dd, bb = dd[ok], Bm[days][ok]
            if len(dd) == 0:
                continue
            rng = np.random.default_rng(4000 + c)
            bi = rng.integers(0, len(dd), (B, len(dd)))
            bm = dd[bi].mean(1)
            name = "+".join([f"L{s}-{e}" for s, e in key[0]] + [f"H{s}-{e}" for s, e in key[1]])
            prof[name] = {"n_days": int(len(dd)), "dates_idx": days[ok].tolist(),
                          "mean_diff_kwh": dd.mean(0).tolist(),
                          "ci95_lo_kwh": np.quantile(bm, 0.025, axis=0).tolist(),
                          "ci95_hi_kwh": np.quantile(bm, 0.975, axis=0).tolist(),
                          "baseline_kwh": bb.mean(0).tolist()}
        rc["day_pattern_profiles"] = prof
        out["clusters"][str(c)] = rc
        a = rc["High"]["all_events"]["aggregates"]
        al = rc["Low"]["all_events"]["aggregates"]
        log(f"[mechanism] data c{c}: High events {rc['High']['all_events']['n_events']}: in {a['in_block']['pct']:+.1f}%, "
            f"pre4 {a['pre4']['pct']:+.1f}% {np.round(a['pre4']['pct_ci95'], 1)}, post4 {a['post4']['pct']:+.1f}% "
            f"{np.round(a['post4']['pct_ci95'], 1)} | Low events {rc['Low']['all_events']['n_events']}: in "
            f"{al['in_block']['pct']:+.1f}%, pre4 {al['pre4']['pct']:+.1f}%, post4 {al['post4']['pct']:+.1f}%, "
            f"far post {al['far_post_5_12']['pct']:+.1f}% {np.round(al['far_post_5_12']['pct_ci95'], 1)}")
    return out, (L, Hh, code)


def cvae_same_events(P, clusters, power, codes, B) -> dict:
    """The selected CVAE's implied effect on the real event days of the full year, with each day's own conditions
    (tariff pattern of that day vs the same day with an all-Normal tariff), analysed with the same event study."""
    import torch
    from lib import cvae_torch as ct
    torch.set_num_threads(1)
    L, Hh, code = codes
    code_f = code.ravel()
    ev_days = np.where(code.any(1))[0]
    out = {}
    for c in clusters:
        mp = P["out"] + f"/models/cvae_selected_c{c}.pt"
        if not os.path.exists(mp):
            continue
        model = ct.load_model(mp)
        cols = model.meta["cond_cols"]
        b = cluster_block(power, c)
        C = b[cols].to_numpy(dtype=np.float64)[ev_days]
        Cn = ef.pattern_condition(C, cols)
        eps = np.random.default_rng([GEN_SEED, c, 7]).standard_normal((N_SAMPLES, len(ev_days), model.latent_dim))
        Sr = ct.generate(model, C, N_SAMPLES, eps=eps).mean(0)
        Sn = ct.generate(model, Cn, N_SAMPLES, eps=eps).mean(0)
        delta = np.zeros(code.shape)
        base = np.full(code.shape, np.nan)
        delta[ev_days] = Sr - Sn
        base[ev_days] = Sn
        rc = {}
        for kind in ("High", "Low"):
            rc[kind] = ef.event_study(delta.ravel(), base.ravel(), code_f, kind, B=B, seed=5000 + c,
                                      same_day_only=True)
        out[str(c)] = rc
    return out


def summarize_mechanism(eff, cvae_pats, data_es, cvae_es, blocks_train, clusters) -> dict:
    s = {"definitions": {
        "pct": "effect in % of the Normal level (scenario) or of the matched baseline (data) over the same slots",
        "status": "appears = claimed sign and 95% CI excludes 0; weak = claimed sign, CI includes 0; absent = "
                  "opposite sign, CI includes 0; contradicted = opposite sign, CI excludes 0"},
         "training_context": {}, "clusters": {}}
    tr = blocks_train
    hi_runs = {(r["start"], r["end"]): r["count"] for r in tr["High"]["within_day_runs"]}
    lo_runs = {(r["start"], r["end"]): r["count"] for r in tr["Low"]["within_day_runs"]}
    h0, h1, l0, l1 = HW[0], HW[-1], LW[0], LW[-1]
    hi_cont = {(a, b): n for (a, b), n in hi_runs.items() if a <= h0 and b >= h1}
    lo_cont = {(a, b): n for (a, b), n in lo_runs.items() if a <= l0 and b >= l1}
    s["training_context"] = {
        "windows": {"High": ef.WINDOW_TEXT["High"], "Low": ef.WINDOW_TEXT["Low"]},
        f"train_high_runs_containing_{HSPAN}": {f"{a}-{b}": n for (a, b), n in hi_cont.items()},
        f"train_high_runs_exactly_{HSPAN}": hi_runs.get((h0, h1), 0),
        f"n_train_high_runs_containing_{HSPAN}": int(sum(hi_cont.values())),
        f"n_train_high_runs_containing_{HSPAN}_longer": int(sum(n for (a, b), n in hi_cont.items() if (a, b) != (h0, h1))),
        f"train_low_runs_containing_{LSPAN}": {f"{a}-{b}": n for (a, b), n in lo_cont.items()},
        f"train_low_runs_exactly_{LSPAN}": lo_runs.get((l0, l1), 0),
        f"n_train_low_runs_containing_{LSPAN}": int(sum(lo_cont.values())),
        f"n_train_low_runs_containing_{LSPAN}_longer": int(sum(n for (a, b), n in lo_cont.items() if (a, b) != (l0, l1))),
        "note": "within-day runs of the training days' tariff dummies (as the CVAE sees each day)"}

    def st(x):
        return None if x is None else {"pct": round(x["pct"], 2), "pct_ci95": [round(v, 2) for v in x["pct_ci95"]],
                                       "kwh": x["kwh"]}

    for c in clusters:
        k = str(c)
        rec = {}
        if f"cvae_c{c}" in eff:
            hr = eff[f"cvae_c{c}"]["High"]["regions"]
            lr = eff[f"cvae_c{c}"]["Low"]["regions"]
            rec["cvae_paper_scenario"] = {
                "High": {r: {"pct": round(hr[r]["pct"], 2), "pct_ci95": [round(v, 2) for v in hr[r]["pct_ci95"]],
                             "status": hr[r]["claim_status"]}
                         for r in ("in_window", "pre2", "pre4", "post2", "post4", "afternoon_24_34",
                                   "rebound_pre_24_38")},
                "Low": {r: {"pct": round(lr[r]["pct"], 2), "pct_ci95": [round(v, 2) for v in lr[r]["pct_ci95"]],
                            "status": lr[r]["claim_status"]}
                        for r in ("in_window", "pre4", "post2", "post4", KEY_LREB, "rebound_32_47")}}
        if k in cvae_pats:
            pr = cvae_pats[k]["patterns"]
            real = {}
            for name, v in pr.items():
                if not name.startswith("real_"):
                    continue
                kind = name.split("_")[1]
                bk = v["blocks"][f"{kind}_{name.split('_')[2]}_{name.split('_')[3]}"]
                real[name] = {r: st(bk.get(r)) for r in ("in_block", "pre2", "pre4", "post2", "post4", "far_pre_5_12",
                                                         "far_post_5_12")}
            rec["cvae_real_blocks_alone"] = real
            rec["paper_High_outside_window_mass"] = cvae_pats[k]["paper_High_outside_window_mass"]
            rec["profile_correlation_with_paper_scenarios"] = cvae_pats[k]["profile_correlation_with_paper_scenarios"]
        if k in data_es["clusters"]:
            de = data_es["clusters"][k]
            rec["data_all_events"] = {kind: {r: st(de[kind]["all_events"]["aggregates"].get(r))
                                             for r in ("in_block", "pre2", "pre4", "post2", "post4", "far_pre_5_12",
                                                       "far_post_5_12")} | {"n_events": de[kind]["all_events"]["n_events"]}
                                      for kind in ("High", "Low")}
            rec["data_same_day_lags"] = {kind: {r: st(de[kind]["same_day_lags_only"]["aggregates"].get(r))
                                                for r in ("in_block", "pre4", "post4", "far_pre_5_12",
                                                          "far_post_5_12")}
                                         for kind in ("High", "Low")}
        if k in cvae_es:
            rec["cvae_same_event_days"] = {kind: {r: st(cvae_es[k][kind]["aggregates"].get(r))
                                                  for r in ("in_block", "pre4", "post4", "far_pre_5_12",
                                                            "far_post_5_12")}
                                           for kind in ("High", "Low")}
        # verdict elements
        v = {}
        if "data_all_events" in rec:
            dh = rec["data_all_events"]["High"]
            v["data_High_side_effect_pre4"] = _sig(dh["pre4"])
            v["data_High_side_effect_post4"] = _sig(dh["post4"])
            dl = rec["data_all_events"]["Low"]
            v["data_Low_far_post_5_12"] = _sig(dl["far_post_5_12"])
            v["data_High_far_pre_5_12"] = _sig(dh["far_pre_5_12"])
        if "cvae_paper_scenario" in rec:
            ch = rec["cvae_paper_scenario"]["High"]
            v["cvae_scenario_High_pre4"] = ch["pre4"]["status"]
            v["cvae_scenario_High_post4"] = ch["post4"]["status"]
            cl = rec["cvae_paper_scenario"]["Low"]
            v[f"cvae_scenario_Low_{KEY_LREB}"] = cl[KEY_LREB]["status"]
            v["cvae_scenario_Low_rebound_32_47"] = cl["rebound_32_47"]["status"]
            v["cvae_scenario_High_afternoon_24_34"] = ch["afternoon_24_34"]["status"]
        if "cvae_real_blocks_alone" in rec:
            v["cvae_real_High_blocks_pre4_pct"] = {n: (x["pre4"]["pct"] if x.get("pre4") else None)
                                                   for n, x in rec["cvae_real_blocks_alone"].items() if "High" in n}
            v["cvae_real_High_blocks_post4_pct"] = {n: (x["post4"]["pct"] if x.get("post4") else None)
                                                    for n, x in rec["cvae_real_blocks_alone"].items() if "High" in n}
        rec["verdict_elements"] = v
        s["clusters"][k] = rec
    return s


def _sig(x):
    if x is None:
        return None
    lo, hi = x["pct_ci95"]
    sig = "positive" if lo > 0 else ("negative" if hi < 0 else "not significant")
    return {"pct": x["pct"], "pct_ci95": x["pct_ci95"], "significance": sig}


def verdicts(msum, eff, clusters) -> dict:
    """Explicit, reproducible criteria for the side-effect, rebound and heterogeneity claims."""
    out = {"criteria": {
        "windows": f"Low {ef.WINDOW_TEXT['Low']}, High {ef.WINDOW_TEXT['High']} (lib.effects; REPRO_SPEC section 7)",
        "cvae_side_effect_in_output": "paper High scenario: pre2 (37-38) or pre4 (35-38) 'appears' AND post2 (45-46) "
                                      "or post4 (45-47, 3 slots left in the day) 'appears' (claimed sign, "
                                      "day-bootstrap CI excludes 0)",
        "data_side_effect": "real High blocks, all events: pre4 or post4 significantly negative (CI over event days)",
        "cvae_side_effect_on_real_event_days": "CVAE implied effect on the real event days: pre4 or post4 "
                                               "significantly negative",
        "cooccurrence": f"correlation of the CVAE's mean response to High {ef.WINDOW_SPAN['High']} with its response "
                        f"to the most frequent training blocks that contain {ef.WINDOW_SPAN['High']} (33-44, 35-46) "
                        "inserted alone; share of the outside-window response located in slots 33-46",
        "cvae_low_rebound": f"paper Low scenario: {KEY_LREB} and rebound_32_47 both 'appears' (drop)",
        "cvae_high_rebound": "paper High scenario: afternoon_24_34 'appears' (rise)",
        "data_low_rebound": "real Low blocks: far_post_5_12 (5-12 half-hours after the end) significantly negative",
        "data_high_rebound": "real High blocks: far_pre_5_12 (5-12 half-hours before the start) significantly "
                             "positive"},
           "clusters": {}}
    for c in clusters:
        k = str(c)
        rec = msum["clusters"].get(k, {})
        v = {}
        ps = rec.get("cvae_paper_scenario")
        if ps:
            h, lo = ps["High"], ps["Low"]
            v["cvae_side_effect_in_output"] = bool((h["pre2"]["status"] == "appears" or h["pre4"]["status"] == "appears")
                                                   and (h["post2"]["status"] == "appears"
                                                        or h["post4"]["status"] == "appears"))
            v["cvae_low_rebound"] = {LREB_SPAN: lo[KEY_LREB]["status"], "32_47": lo["rebound_32_47"]["status"],
                                     "holds": bool(lo[KEY_LREB]["status"] == "appears"
                                                   and lo["rebound_32_47"]["status"] == "appears")}
            v["cvae_high_rebound"] = {"24_34": h["afternoon_24_34"]["status"],
                                      "24_38": h["rebound_pre_24_38"]["status"],
                                      "holds": bool(h["afternoon_24_34"]["status"] == "appears")}
        de = rec.get("data_all_events")
        if de:
            v["data_side_effect"] = bool(any(de["High"][r] and de["High"][r]["pct_ci95"][1] < 0
                                             for r in ("pre4", "post4")))
            v["data_low_rebound"] = bool(de["Low"]["far_post_5_12"] and de["Low"]["far_post_5_12"]["pct_ci95"][1] < 0)
            v["data_high_rebound"] = bool(de["High"]["far_pre_5_12"] and de["High"]["far_pre_5_12"]["pct_ci95"][0] > 0)
            v["data_in_block_pct"] = {"High": de["High"]["in_block"]["pct"], "Low": de["Low"]["in_block"]["pct"]}
        ce = rec.get("cvae_same_event_days")
        if ce:
            v["cvae_side_effect_on_real_event_days"] = bool(any(ce["High"][r] and ce["High"][r]["pct_ci95"][1] < 0
                                                                for r in ("pre4", "post4")))
        cor = rec.get("profile_correlation_with_paper_scenarios", {}).get(PAPER_HIGH, {})
        mass = rec.get("paper_High_outside_window_mass")
        if cor and mass:
            tot = mass["sum_kwh_outside_window"]
            v["cooccurrence"] = {"corr_with_High_33_44_alone": cor.get("real_High_33_44"),
                                 "corr_with_High_35_46_alone": cor.get("real_High_35_46"),
                                 "corr_with_High_21_44_alone": cor.get("real_High_21_44"),
                                 "share_of_outside_window_response_in_slots_33_46":
                                     (mass["sum_kwh_slots_33_46_outside_window"] / tot) if tot else None,
                                 "uniform_share_of_outside_slots": mass.get("uniform_share_of_outside_slots")}
        out["clusters"][k] = v
    # heterogeneity
    het = {}
    for m in MODELS:
        vals = {str(c): (eff[f"{m}_c{c}"]["Low"]["regions"]["in_window"]["pct"],
                         eff[f"{m}_c{c}"]["High"]["regions"]["in_window"]["pct"]) for c in clusters
                if f"{m}_c{c}" in eff}
        het[m] = {"in_window_pct_low_high": vals}
    het["data_in_block_pct_low_high"] = {k: (v.get("data_in_block_pct", {}).get("Low"),
                                             v.get("data_in_block_pct", {}).get("High"))
                                         for k, v in out["clusters"].items()}
    out["heterogeneity"] = het
    cl = out["clusters"]
    out["overall"] = {
        "n_clusters_cvae_side_effect_in_output": int(sum(bool(v.get("cvae_side_effect_in_output")) for v in cl.values())),
        "n_clusters_data_side_effect": int(sum(bool(v.get("data_side_effect")) for v in cl.values())),
        "n_clusters_cvae_side_effect_on_real_event_days": int(sum(bool(v.get("cvae_side_effect_on_real_event_days"))
                                                              for v in cl.values())),
        "n_clusters_cvae_low_rebound": int(sum(bool(v.get("cvae_low_rebound", {}).get("holds")) for v in cl.values())),
        "n_clusters_cvae_high_rebound": int(sum(bool(v.get("cvae_high_rebound", {}).get("holds"))
                                                for v in cl.values())),
        "n_clusters_data_low_rebound": int(sum(bool(v.get("data_low_rebound")) for v in cl.values())),
        "n_clusters_data_high_rebound": int(sum(bool(v.get("data_high_rebound")) for v in cl.values()))}
    return out


# ------------------------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "font.size": 8.5, "axes.titlesize": 9.5, "axes.linewidth": 0.6,
                         "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _shade(ax, slots, color="#d9d8d2", alpha=0.8, label=None):
    ax.axvspan(slots[0] - 0.5, slots[-1] + 0.5, color=color, alpha=alpha, linewidth=0, label=label, zorder=0)


def fig_profiles(eff, clusters, tag, sizes, path):
    plt = _plt()
    fig, axes = plt.subplots(len(clusters), 2, figsize=(11.5, 2.3 * len(clusters) + 0.6), sharex=True)
    axes = np.atleast_2d(axes)
    x = np.arange(H)
    for i, c in enumerate(clusters):
        for j, scen in enumerate(("Low", "High")):
            ax = axes[i, j]
            w = ef.WINDOWS[scen]
            _shade(ax, w, label=f"{scen} window (slots {w[0]}-{w[-1]} = {ef.WINDOW_CLOCK[scen]} GMT)")
            ax.axvspan(w[0] - 4.5, w[0] - 0.5, color="#efeee9", linewidth=0, zorder=0,
                       label="4 slots before / after (High: 3 after, end of day)")
            ax.axvspan(w[-1] + 0.5, min(w[-1] + 4.5, 47.5), color="#efeee9", linewidth=0, zorder=0)
            ax.axhline(0, color=MUTED, linewidth=0.7)
            for m in MODELS:
                e = eff.get(f"{m}_c{c}")
                if e is None:
                    continue
                pr = e[scen]["profile"]
                ax.fill_between(x, 1000 * np.array(pr["diff_kwh_ci95_lo"]), 1000 * np.array(pr["diff_kwh_ci95_hi"]),
                                color=MODEL_COLOR[m], alpha=0.18, linewidth=0)
                ax.plot(x, 1000 * np.array(pr["diff_kwh"]), color=MODEL_COLOR[m], linewidth=1.4,
                        label=f"{MODEL_LABEL[m]} mean (scenario - Normal), 95% CI over days")
            sz = f", {sizes[c]} households" if c in sizes else ""
            ax.set_title(f"Cluster {c}{sz}: {scen} scenario", loc="left")
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("Wh per household\nper half-hour")
    for ax in axes[-1]:
        ax.set_xlabel("Half-hour slot h (authors' column index; slot h ends at h/2 h GMT)")
        ax.set_xticks(range(0, 49, 6))
    h_, l_ = axes[0, 1].get_legend_handles_labels()
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    fig.legend(h0[:1] + h_, l0[:1] + l_, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975), fontsize=8)
    fig.suptitle(f"{tag}: effect of the tariff scenarios (mean of 200 samples, averaged over the 92 test days; same "
                 f"random draws in all scenarios)", y=0.995, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_mechanism(cvae_pats, data_es, clusters, tag, sizes, path):
    plt = _plt()
    fig, axes = plt.subplots(2, len(clusters), figsize=(3.4 * len(clusters), 6.2), sharex=True)
    axes = np.atleast_2d(axes)
    x = np.arange(H)
    for j, c in enumerate(clusters):
        k = str(c)
        if k not in cvae_pats:
            continue
        pr = cvae_pats[k]["patterns"]
        dp = data_es["clusters"].get(k, {}).get("day_pattern_profiles", {}) if data_es else {}
        for i, kind in enumerate(("High", "Low")):
            ax = axes[i, j]
            paper = PAPER_HIGH if kind == "High" else PAPER_LOW
            w = ef.WINDOWS[kind]

            def overlap(n):
                s_, e_ = int(n.split("_")[2]), int(n.split("_")[3])
                return len(set(range(s_, e_ + 1)) & set(w))
            real = sorted([n for n in pr if n.startswith(f"real_{kind}_")], key=lambda n: -overlap(n))[:2]
            joint = [n for n in pr if n.startswith("joint_")][:1]
            _shade(ax, w, label=f"paper {kind} window {w[0]}-{w[-1]}")
            ax.axhline(0, color=MUTED, linewidth=0.7)
            ax.plot(x, 1000 * np.array(pr[paper]["profile_diff_kwh"]), color=BLUE, linewidth=1.6,
                    label=f"CVAE, paper scenario ({kind} {w[0]}-{w[-1]} alone)")
            for n, col in zip(real, (AQUA, YELLOW)):
                s, e = n.split("_")[2:4]
                ax.plot(x, 1000 * np.array(pr[n]["profile_diff_kwh"]), color=col, linewidth=1.2,
                        label=f"CVAE, real training block {kind} {s}-{e} alone")
            for n in joint:
                ax.plot(x, 1000 * np.array(pr[n]["profile_diff_kwh"]), color=MAGENTA, linewidth=1.2,
                        label="CVAE, real joint day pattern " + n.replace("joint_", ""))
                key = n.replace("joint_", "")
                if key in dp:
                    d = dp[key]
                    ax.fill_between(x, 1000 * np.array(d["ci95_lo_kwh"]), 1000 * np.array(d["ci95_hi_kwh"]),
                                    color=INK, alpha=0.12, linewidth=0)
                    ax.plot(x, 1000 * np.array(d["mean_diff_kwh"]), color=INK, linewidth=1.3, linestyle="--",
                            label=f"data: days with that pattern minus matched Normal days (95% CI)")
            sz = f" ({sizes[c]} hh)" if c in sizes else ""
            ax.set_title(f"Cluster {c}{sz}: {kind}", loc="left")
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("Wh per household per half-hour")
            if i == 1:
                ax.set_xlabel("Half-hour slot h")
                ax.set_xticks(range(0, 49, 12))
    handles, labels = [], []
    for ax in axes.ravel():
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in labels:
                labels.append(l_)
                handles.append(h_)
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995), fontsize=7.5)
    fig.suptitle(" ", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.text(0.5, 0.835, f"{tag}: CVAE response to the paper's never-seen windows vs real training blocks and patterns, "
                         f"and the data's response on real event days", ha="center", fontsize=9.5, color=INK)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_eventstudy(data_es, cvae_es, eff, clusters, tag, sizes, path):
    plt = _plt()
    fig, axes = plt.subplots(2, len(clusters), figsize=(3.4 * len(clusters), 5.8), sharex=True)
    axes = np.atleast_2d(axes)
    ls, le = np.array(ef.LAGS_START), np.array(ef.LAGS_END)
    xs = ls                       # start-aligned lags at x = -12..5
    xe = le + 13                  # end-aligned lags shifted right (gap between the two panels of the axis)
    for j, c in enumerate(clusters):
        k = str(c)
        for i, kind in enumerate(("High", "Low")):
            ax = axes[i, j]
            ax.axhline(0, color=MUTED, linewidth=0.7)
            ax.axvspan(-0.5, 5.5, color="#d9d8d2", linewidth=0, zorder=0)
            ax.axvspan(xe[0] - 0.5, 13.5, color="#d9d8d2", linewidth=0, zorder=0)
            ax.axvline(6.5 + 0.0, color=GRID, linewidth=0.8)
            if k in data_es.get("clusters", {}):
                d = data_es["clusters"][k][kind]["same_day_lags_only"]
                for prof, xx in ((d["start_aligned"], xs), (d["end_aligned"], xe)):
                    with np.errstate(invalid="ignore", divide="ignore"):
                        base = np.array(prof["mean_kwh"]) / (np.array(prof["pct"]) / 100)
                    lo = 100 * np.array(prof["ci95_lo_kwh"]) / base
                    hi = 100 * np.array(prof["ci95_hi_kwh"]) / base
                    ax.fill_between(xx, lo, hi, color=INK, alpha=0.12, linewidth=0)
                    ax.plot(xx, prof["pct"], color=INK, linewidth=1.3, marker="o", markersize=2.5)
                ax.text(0.02, 0.03, f"n = {d['n_events']} blocks", transform=ax.transAxes, fontsize=7, color=INK2)
            if k in cvae_es:
                d = cvae_es[k][kind]
                for prof, xx in ((d["start_aligned"], xs), (d["end_aligned"], xe)):
                    ax.plot(xx, prof["pct"], color=BLUE, linewidth=1.3, marker="o", markersize=2.5)
            e = eff.get(f"cvae_c{c}")
            if e is not None:
                w = ef.WINDOWS[kind]
                pr = np.array(e[kind]["profile"]["diff_kwh"]) / np.array(e[kind]["profile"]["normal_kwh"]) * 100
                ys = [pr[w[0] + l_] if 0 <= w[0] + l_ <= w[-1] or (l_ < 0 and w[0] + l_ >= 0) else np.nan for l_ in ls]
                ye = [pr[w[-1] + l_] if (l_ <= 0 and w[-1] + l_ >= w[0]) or (l_ > 0 and w[-1] + l_ < H) else np.nan
                      for l_ in le]
                ax.plot(xs, ys, color=ORANGE, linewidth=1.1, linestyle="--")
                ax.plot(xe, ye, color=ORANGE, linewidth=1.1, linestyle="--")
            sz = f" ({sizes[c]} hh)" if c in sizes else ""
            ax.set_title(f"Cluster {c}{sz}: {kind} blocks", loc="left")
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("Effect, % of baseline")
            if i == 1:
                ticks = [-12, -8, -4, 0, 4] + list(xe[[0, 5, 9, 13, 17]])
                ax.set_xticks(ticks)
                ax.set_xticklabels(["-12", "-8", "-4", "s", "+4", "e-5", "e", "+4", "+8", "+12"], fontsize=7)
                ax.set_xlabel("lag from block start s  |  lag from block end e")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=INK, marker="o", markersize=3, linewidth=1.3,
                      label="data: event day minus matched all-Normal days (all real blocks, same-day lags), 95% CI "
                            "over event days"),
               Line2D([], [], color=BLUE, marker="o", markersize=3, linewidth=1.3,
                      label="CVAE: implied effect on the same event days (each day's own conditions and tariff)"),
               Line2D([], [], color=ORANGE, linestyle="--", linewidth=1.1,
                      label=f"CVAE: paper scenario window (High {ef.WINDOW_SPAN['High']} / Low {ef.WINDOW_SPAN['Low']}), "
                            "aligned the same way"),
               Patch(facecolor="#d9d8d2", label="inside the block")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.text(0.5, 0.875, f"{tag}: response around the start and end of real tariff blocks (effect in % of the "
                         f"baseline; lags in half-hours)", ha="center", fontsize=9.5, color=INK)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_restarts(rest, clusters, tag, path):
    rc = rest.get("recomputed", {}).get("counts")
    if not rc:
        return False
    plt = _plt()
    rows = [("Low:in_window", f"Low raises use in window {ef.WINDOW_SPAN['Low']}"),
            ("High:in_window", f"High lowers use in window {ef.WINDOW_SPAN['High']}"),
            ("High:pre2", "High side effect, 2 slots before"), ("High:pre4", "High side effect, 4 slots before"),
            ("High:post2", "High side effect, 2 slots after"),
            ("High:post4", f"High side effect, slots {ef.span(ef.scenario_regions('High')['post4'])} after"),
            (f"Low:{KEY_LREB}", f"Low rebound: drop in slots {LREB_SPAN.replace('_', '-')}"),
            ("Low:rebound_32_47", "Low rebound: drop in slots 32-47"),
            ("High:rebound_pre_24_38", "High rebound: rise in slots 24-38"),
            ("High:afternoon_24_34", "High rebound: rise in slots 24-34")]
    M = np.full((len(rows) + 1, len(clusters)), np.nan)
    Mi = np.full((len(rows) + 1, len(clusters)), np.nan)
    for i, (key, _) in enumerate(rows):
        for j, c in enumerate(clusters):
            r = rc.get(str(c), {}).get(key + ":pct")
            if r:
                M[i, j] = r["n_claimed_sign"] / rc[str(c)]["n_restarts"]
                Mi[i, j] = r["n_appears_ci_excludes_0"] / rc[str(c)]["n_restarts"]
    for j, c in enumerate(clusters):
        if str(c) in rc:
            M[-1, j] = rc[str(c)]["all_claims_jointly"] / rc[str(c)]["n_restarts"]
    rows = rows + [("joint", "all ten claims jointly (claimed sign)")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    for ax, A, title in ((axes[0], M, "claimed sign"), (axes[1], Mi, "claimed sign, CI excludes 0")):
        ax.imshow(A, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, f"{100 * A[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                            color="white" if A[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels([f"cluster {c}" for c in clusters])
        ax.set_title(f"Share of the 50 restarts: {title}", loc="left")
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[1] for r in rows])
    fig.suptitle(f"{tag}: how often do the paper's claimed CVAE effects appear across restarts? (paper scenarios, "
                 f"same random draws)", y=0.995, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


# ------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["A", "B", "a", "b"])
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    P = track_paths(args.track)
    tag, out = P["tag"], P["out"]
    t0 = time.perf_counter()
    test = read_table(P["test"])
    train = read_table(P["train"])
    power = read_table(P["power"])
    for k in ("test", "train", "power"):
        if not os.path.exists(P[k]):
            log(f"[{tag}] WARNING input not found: {P[k]} (dependent parts skipped)")
    clusters = sorted(test["cluster"].unique()) if test is not None else \
        sorted({int(f.split("_c")[-1][0]) for f in os.listdir(out) if "_scen_Normal_c" in f}) if os.path.isdir(out) else []
    clusters = [int(c) for c in clusters]
    if not clusters:
        log(f"[{tag}] nothing to do (no test table and no scenario arrays)")
        return 0
    sizes = {}
    if tag == "trackA":
        su = pd.read_feather(AUTHORS + "/sum_up_nmf.feather")
        col = [k for k in su.columns if k != "cluster"][0]
        sizes = {int(r["cluster"]): int(r[col]) for _, r in su.iterrows()}
    elif test is not None and "n_households" in test.columns:     # track B tables carry the household count per row
        sizes = {int(c): int(test.loc[test["cluster"] == c, "n_households"].iloc[0]) for c in clusters}
    os.makedirs(FIGS, exist_ok=True)

    # 1. scenario effects
    st1 = stage_effects(P, clusters, args.boot)
    eff = st1["per_model_cluster"]
    summary = {"track": tag, "definitions": {
        "scenario_windows": {k: {"slots": v, "clock_gmt": ef.WINDOW_CLOCK[k]} for k, v in ef.WINDOWS.items()},
        "scenario_windows_note": "Low 9-19 / High 39-44 endpoints inclusive (REPRO_SPEC section 7; slot h = the "
                                 "half-hour ending at h x 30 min GMT); phases 1-2 used 9-18 / 39-43 "
                                 "(outputs/archive_windows_9-18_39-43/). High post4 = 45-47 (3 slots left in the day)",
        "effect": "mean over the 200 samples of (scenario - Normal), then over the 92 test days; kWh per household "
                  "per half-hour (region mean); pct = 100 x effect / Normal mean over the same slots and days",
        "ci": f"95% percentile bootstrap over test days, B = {args.boot}",
        "regions": {s: ef.scenario_regions(s) for s in ("Low", "High")},
        "claims": {f"{k[0]}:{k[1]}": {"sign": v[0], "text": v[1]} for k, v in ef.CLAIMS.items()},
        "claim_status": "appears = claimed sign and CI excludes 0; weak = claimed sign, CI includes 0; absent = "
                        "opposite sign, CI includes 0; contradicted = opposite sign, CI excludes 0"},
               "missing_arrays": st1["missing_arrays"],
               "claims_overview": claim_overview(eff, clusters) if eff else {},
               "per_model_cluster": {k: {s: {"regions": v[s]["regions"]} for s in ("Low", "High")} |
                                     {"crn_check_max_abs_sample_diff_outside_window":
                                      v["crn_check_max_abs_sample_diff_outside_window"]}
                                     for k, v in eff.items()},
               "profiles": {k: {s: v[s]["profile"] for s in ("Low", "High")} for k, v in eff.items()}}
    pd.DataFrame(st1["table"]).to_csv(out + "/effects_regions.csv", index=False)

    # 2. restarts
    rest = stage_restarts(P, clusters, min(args.boot, 1000))
    summary["restarts"] = rest
    jdump(summary, out + "/effects_summary.json")
    log(f"[{tag}] wrote effects_summary.json, effects_regions.csv")

    # 3. mechanism
    mech_ok = test is not None and train is not None
    if mech_ok:
        model0 = out + f"/models/cvae_selected_c{clusters[0]}.pt"
        cond_cols = None
        if os.path.exists(model0):
            from lib import cvae_torch as ct
            cond_cols = ct.load_model(model0).meta["cond_cols"]
        low_cols, high_cols = tariff_names(cond_cols)
        Ltr, Htr = dummies(train, low_cols, high_cols)
        blocks = {"train": ef.block_table(Ltr, Htr), "tariff_columns": [low_cols[0], high_cols[0]]}
        if power is not None:
            Lp, Hp = dummies(power, low_cols, high_cols)
            blocks["all_days"] = ef.block_table(Lp, Hp)
        Lte, Hte = dummies(test, low_cols, high_cols)
        blocks["test"] = ef.block_table(Lte, Hte)
        blocks["train_days_covering_paper_windows"] = {
            HIGH_BLOCK: int(len(ef.days_covering(Htr, ef.HIGH_WINDOW))),
            LOW_BLOCK: int(len(ef.days_covering(Ltr, ef.LOW_WINDOW))),
            f"{HIGH_BLOCK}_days_with_Low_same_day": int(sum(Ltr[d].any() for d in ef.days_covering(Htr, ef.HIGH_WINDOW))),
            f"{LOW_BLOCK}_days_with_High_same_day": int(sum(Htr[d].any() for d in ef.days_covering(Ltr, ef.LOW_WINDOW))),
            f"{HIGH_BLOCK}_days_run_exactly_the_window": int(sum((HW[0], HW[-1]) in ef.runs(Htr[d]) for d in range(len(Htr)))),
            f"{LOW_BLOCK}_days_run_exactly_the_window": int(sum((LW[0], LW[-1]) in ef.runs(Ltr[d]) for d in range(len(Ltr))))}
        jdump(blocks, out + "/mechanism_blocks.json")
        pats = pick_patterns(blocks["train"])
        log(f"[{tag}] mechanism patterns: {list(pats)}")
        cvae_pats = cvae_pattern_responses(P, clusters, test, pats, args.boot)
        jdump({"patterns": {k: {"low_slots": v[0], "high_slots": v[1]} for k, v in pats.items()},
               "clusters": cvae_pats}, out + "/mechanism_cvae_patterns.json")
        data_es, codes, cvae_es = {"clusters": {}}, None, {}
        if power is not None:
            data_es, codes = data_eventstudy(P, clusters, power, low_cols, high_cols, args.boot)
            jdump(data_es, out + "/mechanism_data_eventstudy.json")
            cvae_es = cvae_same_events(P, clusters, power, codes, args.boot)
            jdump(cvae_es, out + "/mechanism_cvae_same_events.json")
        msum = summarize_mechanism(eff, cvae_pats, data_es, cvae_es, blocks["train"], clusters)
        msum["blocks_train_days_covering_paper_windows"] = blocks["train_days_covering_paper_windows"]
        msum["verdicts"] = verdicts(msum, eff, clusters)
        jdump(msum, out + "/mechanism_summary.json")
        log(f"[{tag}] mechanism verdict overview: {msum['verdicts']['overall']}")
        log(f"[{tag}] wrote mechanism_*.json")
    else:
        cvae_pats, data_es, cvae_es = {}, {"clusters": {}}, {}
        log(f"[{tag}] mechanism skipped (test/train table missing)")

    if not args.no_figures:
        if eff:
            fig_profiles(eff, clusters, tag, sizes, FIGS + f"/{tag}_effects_profiles.png")
        if fig_restarts(rest, clusters, tag, FIGS + f"/{tag}_effects_restarts.png"):
            pass
        if cvae_pats:
            fig_mechanism(cvae_pats, data_es, clusters, tag, sizes, FIGS + f"/{tag}_effects_mechanism.png")
        if data_es.get("clusters"):
            fig_eventstudy(data_es, cvae_es, eff, clusters, tag, sizes, FIGS + f"/{tag}_effects_eventstudy.png")
        log(f"[{tag}] figures written")
    log(f"[{tag}] done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
