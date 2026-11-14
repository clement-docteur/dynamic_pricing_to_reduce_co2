"""Step 2: fit and validate the demand-response simulators.

* Response model: per cluster, log(ToU cluster mean / Std mean) regressed on intercepts, temperature and tariff terms.
  Three specifications (lib/demand.SPECS): 'inblock', 'main' (in-block + verified edge effects) and 'full' (+ all
  post-block and day-level terms). The specification is chosen by RMSE on Sep-Oct event days (fitted on Jan-Aug).
  Final fits on Jan-Oct are the simulators used by Step 3 (checked on the unseen Nov-Dec days); weekly-block
  bootstrap (200 draws) of the main Jan-Oct fit for uncertainty.
* CVAE ensemble (sensitivity): the reproduction's Jan-Aug CVAEs (track B, 20 restarts per cluster), run in numpy.
  Loader check: the selected restart reproduces the reproduction's Nov-Dec RMSE.

Held-out check of the tariff response: on held-out days, observed effect = mean residual (log ToU/Std ratio minus the
fitted no-tariff baseline) on the event slots, minus the same slot's mean residual on held-out event-free days.

Outputs: outputs/step2_results.json, outputs/response_theta.npz (Jan-Oct coefficients per spec, bootstrap draws, and
main-spec fits on the days before each rolling test window)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import (DATES_2013, H, OUT, REPRO, cluster_aggregates, cols, day_type, load_households,  # noqa: E402
                        load_tariff_2013, load_temps_2013, make_log, read_parquet, save_json, split_of)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.demand import BANDS, SPECS, CVAEEnsemble, ResponseModel, pct_effect, tariff_regressors  # noqa: E402

log, _ = make_log()
N_BOOT = 200
ROLLING_TEST_MONTHS = (5, 7, 9, 11)          # first month of each rolling test window (Step 3)


def cvae_loader_check(cv):
    """Nov-Dec RMSE of each cluster's selected CVAE restart, numpy decoder vs the reproduction's torch result."""
    pc = read_parquet(REPRO / "data" / "power_consump_ours.parquet")
    res = json.loads((REPRO / "outputs/chrono/chrono_results.json").read_text())
    rst = pd.read_csv(REPRO / "outputs/chrono/chrono_restarts.csv")
    chk = {}
    for c in range(4):
        g = pc[pc.cluster == c].sort_values("date").reset_index(drop=True)
        te = np.where(split_of(g["date"]) == "test")[0]
        sub = rst[(rst.track == "B") & (rst.cluster == c) & (rst.variant == "no_pos_y")]
        m = cv.models[c][int(sub.loc[sub.val_mse.idxmin(), "seed"])]
        eps = np.random.default_rng([20200801, c]).standard_normal((200, len(te), 4))
        C = cv.cond_matrix(m["cols"], g[cols("low")].to_numpy(float)[te], g[cols("high")].to_numpy(float)[te], te)
        S = np.stack([cv.decode(m, e, C) for e in eps])
        Y = g.loc[te, cols("consumption")].to_numpy(float)
        chk[c] = {"numpy_rmse_median": float(np.median(np.sqrt(((S.mean(0) - Y) ** 2).mean(1)))),
                  "reproduction_rmse_median": res["tracks"]["B"][str(c)]["cvae_no_pos_y"]["rmse_median"]}
    return chk


def main():
    hh = load_households()
    low, high, _ = load_tariff_2013()
    temp = load_temps_2013()
    sp = split_of(DATES_2013)
    daytype = day_type(DATES_2013)
    sizes, w, Yc, _ = cluster_aggregates(hh)
    S = hh["Y_std"].mean(0).astype(float)
    LR = np.log(Yc) - np.log(S)[None]
    event_day = low.any(1) | high.any(1)
    log(f"clusters {sizes.tolist()}, mean kWh/half-hour {Yc.mean((1, 2)).round(3).tolist()}, Std {S.mean():.3f}")

    # ---------------- fit every spec on Jan-Aug and Jan-Oct; choose the spec on Sep-Oct event days
    fits = {"janaug": sp == "train", "janoct": (sp == "train") | (sp == "val")}
    ev_val = np.repeat(event_day & (sp == "val"), H)
    Xs, models, spec_rmse = {}, {}, {}
    for spec in SPECS:
        Xs[spec] = X = ResponseModel(spec).design(low, high, temp, daytype)
        for fk, rows in fits.items():
            models[(spec, fk)] = ResponseModel(spec)
            betas = [models[(spec, fk)].fit_cluster(c, LR[c], X, rows) for c in range(4)]
            if fk == "janaug":
                r = [np.sqrt(((LR[c].ravel() - X @ betas[c])[ev_val] ** 2).mean()) for c in range(4)]
                spec_rmse[spec] = {"per_cluster": r, "household_weighted": float(np.dot(w, r))}
    chosen = min(spec_rmse, key=lambda k: spec_rmse[k]["household_weighted"])
    log("validation RMSE of log(ToU/Std) on Sep-Oct event days (x1000): "
        + ", ".join(f"{k} {1000 * v['household_weighted']:.3f}" for k, v in spec_rmse.items()) + f" -> {chosen}")
    assert chosen == "main", "Step 3 uses the 'main' specification as its simulator"
    main_m, X = models[("main", "janoct")], Xs["main"]

    # ---------------- weekly-block bootstrap of the main Jan-Oct fit
    rng = np.random.default_rng(2013)
    pen = np.diag(main_m.penalty(X.shape[1]))
    weeks = np.arange(365) // 7
    tr_rows = np.where(fits["janoct"])[0]
    tr_weeks = np.unique(weeks[tr_rows])
    boot = {c: [] for c in range(4)}
    for _ in range(N_BOOT):
        days = np.concatenate([tr_rows[weeks[tr_rows] == x] for x in rng.choice(tr_weeks, len(tr_weeks), replace=True)])
        idx = (days[:, None] * H + np.arange(H)[None]).ravel()
        Xb = X[idx]
        XtX = Xb.T @ Xb + pen
        for c in range(4):
            boot[c].append(np.linalg.solve(XtX, Xb.T @ LR[c].ravel()[idx])[main_m.p0:])
    boot = {c: np.array(v) for c, v in boot.items()}
    # main-spec fits on the days before each rolling test window (Step 3's rolling evaluation; before11 = Jan-Oct)
    rolling = {}
    for mm in ROLLING_TEST_MONTHS:
        rm = ResponseModel("main")
        for c in range(4):
            rm.fit_cluster(c, LR[c], X, np.asarray(DATES_2013.month < mm))
            rolling[f"main_before{mm:02d}_c{c}"] = rm.theta[c]
    np.savez_compressed(OUT / "response_theta.npz",
                        **{f"{spec}_janoct_c{c}": models[(spec, "janoct")].theta[c] for spec in SPECS for c in range(4)},
                        **{f"boot_main_janoct_c{c}": v for c, v in boot.items()}, **rolling)
    eff = {}
    for c in range(4):
        pt, bs = main_m.in_block_effects(c), [main_m.in_block_effects(c, t) for t in boot[c]]
        eff[c] = {k: {"pct": pt[k], "ci95": list(np.percentile([b[k] for b in bs], [2.5, 97.5]))} for k in pt}
    log("main in-block effects (Jan-Oct), %: " + "; ".join(
        f"c{c} High-peak {eff[c]['High|peak 16-20']['pct']:+.1f} Low-night {eff[c]['Low|night 00-06']['pct']:+.1f}" for c in range(4)))

    # ---------------- CVAE ensemble + loader check
    cv = CVAEEnsemble()
    chk = cvae_loader_check(cv)
    log(f"CVAE numpy loader check (Nov-Dec RMSE, selected restart): {chk}")

    # ---------------- held-out validation of the tariff response
    noev = ~event_day
    noev3 = noev & np.roll(noev, 1) & np.roll(noev, -1)                  # no event on the day or its neighbours
    R = tariff_regressors(low, high, timeline=True)
    cats = {"in-block High": R["in_High"] > 0, "in-block Low": R["in_Low"] > 0,
            "2 h before Low": R["pre2h_Low"] > 0, "2 h before High": R["pre2h_High"] > 0}
    m_cvae = np.stack([cv.log_effect(c, low, high, np.arange(365)) for c in range(4)])
    val = {}
    for pi, (period, fk, part) in enumerate((("Sep-Oct (validation; fitted Jan-Aug)", "janaug", "val"),
                                             ("Nov-Dec (test; fitted Jan-Oct)", "janoct", "test"))):
        per, model = sp == part, models[("main", fk)]
        m_main = np.stack([model.log_effect(c, low, high, timeline=True) for c in range(4)])
        out = {}
        for ci, c in enumerate([0, 1, 2, 3, "all"]):
            cs = list(range(4)) if c == "all" else [c]
            res_c, pred = np.zeros((365, H)), {"main": np.zeros((365, H)), "cvae": np.zeros((365, H))}
            for wi, cc in zip(sizes[cs] / sizes[cs].sum(), cs):
                r = LR[cc] - model.baseline_pred(cc, X)
                ctrl = np.zeros((365, H))
                for dtp in (0, 1):
                    ctrl[daytype == dtp] = r[per & noev3 & (daytype == dtp)].mean(0)
                res_c += wi * (r - ctrl)
                pred["main"] += wi * m_main[cc]
                pred["cvae"] += wi * m_cvae[cc]
            o = {}
            for ki, (name, mask) in enumerate(cats.items()):
                mk = mask & per[:, None]
                dsel = np.where(mk.any(1))[0]
                rng_ci = np.random.default_rng([2013, pi, ci, ki])
                bs = [np.concatenate([res_c[d][mk[d]] for d in rng_ci.choice(dsel, len(dsel), replace=True)]).mean()
                      for _ in range(300)]
                o[name] = {"n_days": int(len(dsel)), "observed_pct": pct_effect(res_c[mk].mean()),
                           "observed_ci95": list(pct_effect(np.percentile(bs, [2.5, 97.5]))),
                           **{f"{k}_pct": pct_effect(v[mk].mean()) for k, v in pred.items()}}
            out[str(c)] = o
        val[period] = out
        a = out["all"]
        log(f"{period}: all-ToU " + " | ".join(f"{n}: obs {a[n]['observed_pct']:+.1f} main {a[n]['main_pct']:+.1f} "
                                               f"cvae {a[n]['cvae_pct']:+.1f}" for n in a))

    save_json({"cluster_sizes": sizes, "bands": list(BANDS), "spec_terms": SPECS, "chosen_spec": chosen,
               "spec_selection_val_rmse": spec_rmse, "effects_main_janoct": eff, "heldout_validation": val,
               "cvae_loader_check": chk}, OUT / "step2_results.json")
    log("done")


if __name__ == "__main__":
    main()
