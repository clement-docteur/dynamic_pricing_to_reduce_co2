"""Step 1: forecast next-day half-hourly grid carbon intensity (CI), NESO data 2009-2013.

Decision timing: the tariff for day d is announced at 12:00 GMT on d-1 (day-ahead), so features use NESO data up to
the half-hour ending 12:00 on d-1 (horizon 12-36 h). Target: CI of slot h of day d minus the mean CI of the last 24 h
known ("anchor"); models predict that deviation and the anchor is added back.

Splits: model selection = fit on 2009-01-09..2013-08-30, score on validation 2013-09-01..10-31.
Out-of-sample forecasts for every day of 2013 (needed as policy features): monthly expanding refits, the model for
month m is trained on all days <= (1st of m) - 2 days. Test = Nov-Dec 2013 from those refits.
Weather: ERA5 London temperature exists only for 2013, so its value is tested on a 2013-only training window
(realised temperature = a perfect day-ahead forecast, an upper bound on its value).

Outputs: outputs/ci_forecasts_2013.npz (actual, selected model, persistence), outputs/step1_metrics.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import (BANK_HOLIDAYS, DATES_2013, H, OUT, day_matrix, load_neso, load_temps_2013, make_log,  # noqa: E402
                        save_json, split_of)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Lasso, Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

log, _ = make_log()
DAYS = pd.date_range("2009-01-09", "2013-12-31", freq="D")
HH = np.timedelta64(30, "m")
LABELS = {"xgb": "XGBoost", "rf": "Random forest", "ridge": "Ridge", "lasso": "Lasso",
          "persist_latest": "Persistence: same slot, latest day", "persist_7d": "Persistence: same slot, 7 days ago",
          "flat24h": "Flat: mean of last 24 h"}


# ------------------------------------------------------------------------------------------------ features
def build_frame(neso: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    ci, gen = neso.CARBON_INTENSITY.astype(float), neso.GENERATION.astype(float)
    share = lambda cols: neso[cols].sum(axis=1) / gen
    mix = {"coal": share(["COAL"]), "gas": share(["GAS"]), "wind": share(["WIND", "WIND_EMB"]),
           "nuclear": share(["NUCLEAR"]), "imports": share(["IMPORTS"])}
    allday = pd.date_range(DAYS[0] - pd.Timedelta(days=8), DAYS[-1], freq="D")
    CI = pd.DataFrame(day_matrix(ci, allday), index=allday)
    GEN = pd.DataFrame(day_matrix(gen, allday), index=allday)
    lagged = lambda frame, k: frame.loc[DAYS - pd.Timedelta(days=k)]
    n = len(DAYS)
    cut = DAYS - pd.Timedelta(days=1) + pd.Timedelta(hours=11, minutes=30)          # last known NESO row (start label)
    last24 = (cut.values[:, None] - np.arange(H)[None, :] * HH).ravel()             # 48 rows up to the cut
    ci24 = ci.reindex(last24).to_numpy().reshape(n, H)
    day = pd.DataFrame(index=DAYS)
    day["anchor"] = ci24.mean(1)
    day["last"] = ci24[:, 0]
    day["trend3h"] = ci24[:, 0] - ci24[:, 6]
    day["daymean_2"] = lagged(CI, 2).mean(axis=1).to_numpy()
    day["daymean_7"] = lagged(CI, 7).mean(axis=1).to_numpy()
    for k, s in mix.items():
        day[f"{k}_last"] = s.reindex(cut).to_numpy()
        day[f"{k}_24h"] = s.reindex(last24).to_numpy().reshape(n, H).mean(1)
    day["gen_last"] = gen.reindex(cut).to_numpy() / 1000
    day["gen_24h"] = gen.reindex(last24).to_numpy().reshape(n, H).mean(1) / 1000
    doy = DAYS.dayofyear.to_numpy()
    day["doy_sin"], day["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    day["dow"] = DAYS.dayofweek
    day["holiday"] = DAYS.isin(BANK_HOLIDAYS).astype(float)

    lag2, lag7 = lagged(CI, 2).to_numpy(), lagged(CI, 7).to_numpy()
    lag_recent = np.where((np.arange(H) <= 24)[None, :], lagged(CI, 1).to_numpy(), lag2)   # slot h of d-1 known by 12:00 if h <= 24
    rows = pd.DataFrame({"date": np.repeat(DAYS.values, H), "slot": np.tile(np.arange(H), n)})
    for c in day.columns:
        rows[c] = np.repeat(day[c].to_numpy(), H)
    A = np.repeat(day["anchor"].to_numpy(), H)
    rows["lag_recent_dev"] = lag_recent.ravel() - A
    rows["lag2_dev"] = lag2.ravel() - A
    rows["lag7_dev"] = lag7.ravel() - A
    rows["gen_lag7"] = lagged(GEN, 7).to_numpy().ravel() / 1000
    for c in ("last", "daymean_2", "daymean_7"):
        rows[c] = rows[c] - A                                # all CI lags as deviations from the anchor
    return rows, day_matrix(ci, DAYS)


NUM = ["anchor", "last", "trend3h", "daymean_2", "daymean_7", "coal_last", "coal_24h", "gas_last", "gas_24h",
       "wind_last", "wind_24h", "nuclear_last", "nuclear_24h", "imports_last", "imports_24h", "gen_last", "gen_24h",
       "doy_sin", "doy_cos", "holiday", "lag_recent_dev", "lag2_dev", "lag7_dev", "gen_lag7"]
TEMP = ["temp_slot", "temp_day", "temp_day_change"]


# ------------------------------------------------------------------------------------------------ models
class LinearModel:
    """Ridge/Lasso on standardised numerics + slot and weekday one-hots + slot x lag interactions."""

    def __init__(self, kind, alpha, extra=()):
        self.kind, self.alpha, self.extra = kind, alpha, list(extra)

    def design(self, df):
        S = np.eye(H)[df.slot.to_numpy()]
        inter = S * df["lag_recent_dev"].to_numpy()[:, None]           # slot-specific persistence
        return df[NUM + self.extra].to_numpy(float), np.hstack([S, np.eye(7)[df.dow.to_numpy()], inter / 50.0])

    def fit(self, df, y):
        num, cat = self.design(df)
        self.sc = StandardScaler().fit(num)
        est = Ridge(alpha=self.alpha) if self.kind == "ridge" else Lasso(alpha=self.alpha, max_iter=5000)
        self.m = est.fit(np.hstack([self.sc.transform(num), cat]), y)
        return self

    def predict(self, df):
        num, cat = self.design(df)
        return self.m.predict(np.hstack([self.sc.transform(num), cat]))


class TreeModel:
    """Random forest or XGBoost on the raw feature table (slot and weekday as integers)."""

    def __init__(self, make_estimator, extra=()):
        self.make, self.extra = make_estimator, list(extra)

    def X(self, df):
        return df[NUM + ["slot", "dow"] + self.extra].to_numpy(float)

    def fit(self, df, y):
        self.m = self.make().fit(self.X(df), y)
        return self

    def predict(self, df, n_trees=None):
        return self.m.predict(self.X(df), **({"iteration_range": (0, n_trees)} if n_trees else {}))


def make_model(name, p, extra=()):
    if name in ("ridge", "lasso"):
        return LinearModel(name, p["alpha"], extra)
    if name == "rf":
        return TreeModel(lambda: RandomForestRegressor(n_estimators=200, max_samples=0.5, n_jobs=4, random_state=0, **p), extra)
    return TreeModel(lambda: XGBRegressor(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                                          n_jobs=4, random_state=0, tree_method="hist", **p), extra)


# ------------------------------------------------------------------------------------------------ metrics
def window_means(M, w=6):
    c = np.cumsum(np.pad(M, ((0, 0), (1, 0))), axis=1)
    return (c[:, w:] - c[:, :-w]) / w


def metrics(pred, actual):
    e = pred - actual
    rho = np.nanmean([spearmanr(p, a).statistic for p, a in zip(pred, actual)])
    wp, wa = window_means(pred), window_means(actual)
    r = np.arange(len(pred))
    got = wa[r, wp.argmax(1)] - wa[r, wp.argmin(1)]                  # realised gap between forecast dirtiest/cleanest 3 h
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "within_day_spearman": float(rho), "spread_captured_3h": float(got.mean() / (wa.max(1) - wa.min(1)).mean())}


# ------------------------------------------------------------------------------------------------ main
def main():
    df, Y = build_frame(load_neso())
    temps = load_temps_2013()
    n13 = len(DATES_2013)
    tday = np.full(len(DAYS), np.nan)
    tday[-n13:] = temps.mean(1)
    tday_lag2 = np.roll(tday, 2)
    tday_lag2[-n13:-n13 + 2] = np.nan
    df["temp_slot"] = np.concatenate([np.full((len(DAYS) - n13) * H, np.nan), temps.ravel()])
    df["temp_day"] = np.repeat(tday, H)
    df["temp_day_change"] = np.repeat(tday - tday_lag2, H)
    anchor = df["anchor"].to_numpy().reshape(-1, H)
    ydev = (Y - anchor).ravel()
    dd = df["date"].to_numpy()
    per_day = lambda mask: mask.reshape(-1, H)[:, 0]

    def month_masks(m, start=None):
        """Training rows (days <= 1st of month m - 2 days, optionally from `start`), forecast rows, and 2013 day rows."""
        m1, m2 = pd.Timestamp(2013, m, 1), pd.Timestamp(2013, m, 1) + pd.offsets.MonthBegin(1)
        train = dd <= np.datetime64(m1 - pd.Timedelta(days=2))
        if start:
            train &= dd >= np.datetime64(start)
        return train, (dd >= np.datetime64(m1)) & (dd < np.datetime64(m2)), (DATES_2013 >= m1) & (DATES_2013 < m2)

    tr, va = dd <= np.datetime64("2013-08-30"), (dd >= np.datetime64("2013-09-01")) & (dd <= np.datetime64("2013-10-31"))
    log(f"{len(DAYS)} days x 48 slots; train rows {tr.sum():,}, val rows {va.sum():,}")

    # ---------------- model selection on validation
    grids = {"ridge": [{"alpha": a} for a in (10, 100, 1000, 3000, 10000, 30000, 100000)],
             "lasso": [{"alpha": a} for a in (0.03, 0.1, 0.3, 1.0, 3.0)],
             "rf": [{"max_features": f, "min_samples_leaf": l} for f in (0.33, 0.66) for l in (5, 25, 100)],
             "xgb": [{"max_depth": d, "n_estimators": 1000} for d in (2, 3, 5)]}
    best, best_mae = {}, {}
    yv, av = Y[per_day(va)], anchor[per_day(va)]
    for name, grid in grids.items():
        res = []
        for p in grid:
            m = make_model(name, p).fit(df[tr], ydev[tr])
            for nt in ((100, 200, 500, 1000) if name == "xgb" else (None,)):   # xgb: score each ensemble size
                pr = (m.predict(df[va], n_trees=nt) if nt else m.predict(df[va])).reshape(-1, H) + av
                res.append(({**p, "n_estimators": nt} if nt else p, float(np.abs(pr - yv).mean())))
        best[name], best_mae[name] = min(res, key=lambda t: t[1])
        log(f"{name}: best {best[name]} val MAE {best_mae[name]:.2f}")
    sel = min(best_mae, key=best_mae.get)
    log(f"selected model (lowest validation MAE): {sel}")

    # ---------------- monthly expanding refits over 2013 (out-of-sample forecasts for every 2013 day)
    F = {k: np.full((365, H), np.nan) for k in grids}
    for m in range(1, 13):
        trm, tem, rows13 = month_masks(m)
        for name in grids:
            F[name][rows13] = make_model(name, best[name]).fit(df[trm], ydev[trm]).predict(df[tem]).reshape(-1, H) + anchor[per_day(tem)]
        log(f"refit month {m} done")
    A13, anc13 = Y[-n13:], anchor[-n13:]
    F["persist_latest"] = df["lag_recent_dev"].to_numpy().reshape(-1, H)[-n13:] + anc13
    F["persist_7d"] = df["lag7_dev"].to_numpy().reshape(-1, H)[-n13:] + anc13
    F["flat24h"] = anc13.copy()

    # ---------------- validation + test metrics
    sp = split_of(DATES_2013)
    va13, te13 = sp == "val", sp == "test"
    table = {k: {"test": metrics(F[k][te13], A13[te13]), "val": metrics(F[k][va13], A13[va13])} for k in F}
    for k, v in table.items():
        log(f"{LABELS[k]:36s} test MAE {v['test']['MAE']:.1f} RMSE {v['test']['RMSE']:.1f} rho {v['test']['within_day_spearman']:.2f} "
            f"spread {v['test']['spread_captured_3h']:.2f} | val MAE {v['val']['MAE']:.1f}")

    # ---------------- does weather help? 2013-only training window, best xgb params, with/without temperature
    wx = {}
    for label, extra in (("no weather", ()), ("with ERA5 temperature", TEMP)):
        out = np.full((365, H), np.nan)
        for m in range(3, 13):                                 # Jan-Feb: too little 2013 history
            trm, tem, rows13 = month_masks(m, start="2013-01-03")
            out[rows13] = make_model("xgb", best["xgb"], extra).fit(df[trm], ydev[trm]).predict(df[tem]).reshape(-1, H) + anchor[per_day(tem)]
        wx[label] = {"val": metrics(out[va13], A13[va13]), "test": metrics(out[te13], A13[te13])}
        log(f"weather test [{label}]: test MAE {wx[label]['test']['MAE']:.1f}, val MAE {wx[label]['val']['MAE']:.1f}")

    np.savez_compressed(OUT / "ci_forecasts_2013.npz", dates=DATES_2013.values.astype("datetime64[D]"), actual=A13,
                        selected=F[sel], persist_latest=F["persist_latest"])
    save_json({"selected_model": sel, "best_params": best, "labels": LABELS, "metrics": table, "weather_experiment": wx},
              OUT / "step1_metrics.json")
    log("done")


if __name__ == "__main__":
    main()
