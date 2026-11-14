"""Step 1b: marginal emission factors (MEF) of the GB grid, NESO data 2009-2013.

A household tariff changes demand at the margin, so its emissions effect is set by the plants that ramp, not by the
average mix. Following Hawkes (2010) and Siler-Evans et al. (2012), the MEF of a bin is the slope of the half-hourly
change in system emissions on the change in generation:  dE_t = a_b + MEF_b * dG_t  (E = CI x generation).
Bins: hour of day (UTC) x season (Oct-Mar / Apr-Sep).

Pumped storage: NESO counts its output as zero-carbon, but that energy was pumped overnight (coal at the margin in 2013).
In the main ('storage-adjusted') factors, storage output carries the overnight marginal factor divided by a 75%
round-trip efficiency; the unadjusted factors are kept as a sensitivity.

* Measured factors for 2013 (cost of a schedule on fitting days, and evaluation): estimated on calendar 2013.
* Decision-time factors (direct optimisation, carbon-only rule): monthly re-estimation on the 365 days ending 2 days
  before the month, like the CI forecast refits of Step 1.

Net-load variant (merit-order view): which plant is marginal depends on how far up the supply stack the system is, not
on the clock. Net load = generation minus nuclear, wind (transmission + embedded) and solar, i.e. the demand left for
dispatchable plants. Bins: season x net-load decile (edges = deciles of the estimation window); a half-hour gets the
factor of its own net-load bin, so two days at the same hour can get different factors. Only measured (realised)
factors are produced; a decision-time version would need a day-ahead net-load (wind) forecast. Out-of-sample check:
both binnings fitted on 2012, scored by R^2 of the predicted half-hourly emission change on 2013.

Outputs: outputs/mef_2013.npz ((365, 48) matrices: realised, realised_unadjusted, forecast, realised_netload),
         outputs/step1b_mef.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import DATES_2013, H, OUT, day_matrix, load_neso, make_log, save_json, split_of  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log, _ = make_log()
ROUND_TRIP = 0.75                      # pumped-storage round-trip efficiency
NIGHT_HOURS = range(0, 5)              # 00:00-05:00 UTC: storage is pumping (no output), coal at the margin in 2013
FUELS = ["GAS", "COAL", "STORAGE", "HYDRO", "IMPORTS", "WIND", "NUCLEAR"]
BANDS = {"night 00-05": range(0, 5), "morning 05-09": range(5, 9), "day 09-16": range(9, 16),
         "peak 16-20": range(16, 20), "evening 20-24": range(20, 24)}


def season(index):
    """1 = Oct-Mar, 0 = Apr-Sep."""
    return np.asarray(index.month.isin([10, 11, 12, 1, 2, 3])).astype(int)


def changes(neso, start, end, storage_ef=None):
    """Half-hourly changes of emissions (g/h x 1e3) and generation (MW) for rows in [start, end]."""
    w = neso.loc[pd.Timestamp(start) - pd.Timedelta(minutes=30): end]
    E = w.CARBON_INTENSITY * w.GENERATION
    if storage_ef is not None:
        E = E + w.STORAGE * storage_ef
    d = pd.DataFrame({"dE": E.diff(), "dG": w.GENERATION.diff()}).iloc[1:]
    return d[(w.index.to_series().diff() == pd.Timedelta(minutes=30)).iloc[1:]]


def slope(d):
    """OLS slope of dE on dG (with intercept) and its standard error."""
    x, y = d.dG.to_numpy(), d.dE.to_numpy()
    xc = x - x.mean()
    b = (xc * (y - y.mean())).sum() / (xc ** 2).sum()
    res = y - y.mean() - b * xc
    return b, np.sqrt((res ** 2).sum() / (len(x) - 2) / (xc ** 2).sum())


def mef_table(neso, start, end, storage_adjust=True):
    """MEF (g/kWh) and SE by season (2) x hour (24); also the storage emission factor used."""
    d = changes(neso, start, end)
    ef = None
    if storage_adjust:
        night = d[d.index.hour.isin(list(NIGHT_HOURS))]
        ef = slope(night)[0] / ROUND_TRIP
        d = changes(neso, start, end, storage_ef=ef)
    tab, se = np.zeros((2, 24)), np.zeros((2, 24))
    for (s, h), g in d.groupby([season(d.index), d.index.hour]):
        tab[s, h], se[s, h] = slope(g)
    return tab, se, ef


def net_load(neso):
    """Demand left for dispatchable plants (MW): generation minus nuclear, wind and solar."""
    return neso.GENERATION - neso.NUCLEAR - neso.WIND - neso.WIND_EMB - neso.SOLAR


def fit_bins(d, keys):
    """Per-bin OLS (intercept, slope, se) of dE on dG; keys = array of bin labels aligned with d."""
    out = {}
    for k, g in d.groupby(pd.Series(keys, index=d.index)):
        b, se = slope(g)
        out[k] = (g.dE.mean() - b * g.dG.mean(), b, se)
    return out


def netload_keys(neso, d, edges):
    """(season, net-load decile) label of each change in d; net load = mean over the two half-hours."""
    nl = net_load(neso)
    lvl = ((nl + nl.shift(1)) / 2).reindex(d.index).to_numpy()
    return list(zip(season(d.index), np.digitize(lvl, edges))), lvl


def hour_keys(d):
    return list(zip(season(d.index), d.index.hour))


def r2_out_of_sample(fit, d, keys):
    pred = np.array([fit[k][0] + fit[k][1] * x if k in fit else np.nan for k, x in zip(keys, d.dG.to_numpy())])
    y = d.dE.to_numpy()
    ok = ~np.isnan(pred)
    return float(1 - ((y[ok] - pred[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum())


def netload_mef(neso, start, end):
    """Storage-adjusted MEF by season x net-load decile on [start, end]; returns fit, edges, storage factor."""
    d0 = changes(neso, start, end)
    ef = slope(d0[d0.index.hour.isin(list(NIGHT_HOURS))])[0] / ROUND_TRIP
    d = changes(neso, start, end, storage_ef=ef)
    nl = net_load(neso).loc[start:end]
    edges = np.quantile(nl, np.linspace(0.1, 0.9, 9))
    keys, _ = netload_keys(neso, d, edges)
    return fit_bins(d, keys), edges, ef, d


def to_slots(neso_index_values, tab):
    """Map an (season, hour) table onto NESO rows, then onto the (365, 48) slot matrix."""
    idx = pd.DatetimeIndex(neso_index_values)
    return pd.Series(tab[season(idx), idx.hour], index=idx)


def main():
    neso = load_neso()
    rows13 = neso.loc["2012-12-31 23:00":"2013-12-31 23:30"].index

    # ---------------- context: annual MEF vs average CI, and which fuels meet demand changes
    annual = {}
    for y in range(2009, 2014):
        annual[y] = {"mef_unadjusted": slope(changes(neso, f"{y}-01-01", f"{y}-12-31 23:30"))[0],
                     "average_ci": float(neso.loc[str(y)].CARBON_INTENSITY.mean())}
    n13 = neso.loc["2013"]
    dd = n13[FUELS + ["GENERATION"]].diff().dropna()
    fuel_shares = {}
    for band, hrs in BANDS.items():
        x = dd[dd.index.hour.isin(list(hrs))]
        fuel_shares[band] = {f: float(np.polyfit(x.GENERATION, x[f], 1)[0]) for f in FUELS}
    log("annual MEF / average CI: " + ", ".join(f"{y} {v['mef_unadjusted']:.0f}/{v['average_ci']:.0f}" for y, v in annual.items()))

    # ---------------- measured 2013 factors (adjusted = main, unadjusted = sensitivity)
    tab, se, ef = mef_table(neso, "2013-01-01", "2013-12-31 23:30", storage_adjust=True)
    tab_u, se_u, _ = mef_table(neso, "2013-01-01", "2013-12-31 23:30", storage_adjust=False)
    realised = day_matrix(to_slots(rows13, tab), DATES_2013)
    realised_u = day_matrix(to_slots(rows13, tab_u), DATES_2013)
    log(f"storage emission factor {ef:.0f} g/kWh; 2013 hourly MEF (Oct-Mar, adjusted): {np.round(tab[1]).astype(int).tolist()}")

    # ---------------- decision-time factors: monthly re-estimation on the trailing 365 days
    forecast = np.full((365, H), np.nan)
    for m in range(1, 13):
        m1 = pd.Timestamp(2013, m, 1)
        tab_f, _, _ = mef_table(neso, m1 - pd.Timedelta(days=366), m1 - pd.Timedelta(days=2) + pd.Timedelta(hours=23, minutes=30))
        rows = (DATES_2013 >= m1) & (DATES_2013 < m1 + pd.offsets.MonthBegin(1))
        forecast[rows] = day_matrix(to_slots(rows13, tab_f), DATES_2013[rows])
    # ---------------- net-load variant (measured 2013) + out-of-sample comparison of the two binnings (fit 2012, score 2013)
    fit_nl, edges, ef_nl, d13 = netload_mef(neso, "2013-01-01", "2013-12-31 23:30")
    nl13 = net_load(neso).reindex(rows13)
    keys13 = list(zip(season(rows13), np.digitize(nl13.to_numpy(), edges)))
    realised_nl = day_matrix(pd.Series([fit_nl[k][1] for k in keys13], index=rows13), DATES_2013)
    fit12_nl, edges12, ef12, _ = netload_mef(neso, "2012-01-01", "2012-12-31 23:30")
    d13_12ef = changes(neso, "2013-01-01", "2013-12-31 23:30", storage_ef=ef12)     # 2013 changes, 2012 storage factor
    d12 = changes(neso, "2012-01-01", "2012-12-31 23:30", storage_ef=ef12)
    fit12_hr, fit12_one = fit_bins(d12, hour_keys(d12)), fit_bins(d12, [0] * len(d12))
    k13_nl, _ = netload_keys(neso, d13_12ef, edges12)
    oos = {"single slope": r2_out_of_sample(fit12_one, d13_12ef, [0] * len(d13_12ef)),
           "season x hour (main)": r2_out_of_sample(fit12_hr, d13_12ef, hour_keys(d13_12ef)),
           "season x net-load decile": r2_out_of_sample(fit12_nl, d13_12ef, k13_nl)}
    nl_table = {s_: {"decile_upper_edge_MW": list(edges) + [None],
                     "mef": [fit_nl.get((s_i, q), (np.nan, np.nan, np.nan))[1] for q in range(10)],
                     "se": [fit_nl.get((s_i, q), (np.nan, np.nan, np.nan))[2] for q in range(10)]}
                for s_, s_i in (("Oct-Mar", 1), ("Apr-Sep", 0))}
    log(f"net-load MEF Oct-Mar by decile: {np.round(nl_table['Oct-Mar']['mef']).astype(int).tolist()}")
    log(f"out-of-sample R^2 (fit 2012, score 2013): {oos}")

    sp = split_of(DATES_2013)
    te = sp == "test"
    ci13 = day_matrix(neso.CARBON_INTENSITY.astype(float), DATES_2013)
    check = {"netload_vs_hourly_test_corr": float(np.corrcoef(realised_nl[te].ravel(), realised[te].ravel())[0, 1]),
             "forecast_mae_test_g_per_kwh": float(np.abs(forecast[te] - realised[te]).mean()),
             "forecast_profile_corr_test": float(np.corrcoef(forecast[te].mean(0), realised[te].mean(0))[0, 1]),
             "mean_test": {"mef_adjusted": float(realised[te].mean()), "mef_unadjusted": float(realised_u[te].mean()),
                           "mef_netload": float(realised_nl[te].mean()),
                           "average_ci": float(ci13[te].mean())}}
    log(f"test-period means: {check['mean_test']}; forecast MAE {check['forecast_mae_test_g_per_kwh']:.0f} g/kWh")

    np.savez_compressed(OUT / "mef_2013.npz", realised=realised, realised_unadjusted=realised_u, forecast=forecast,
                        realised_netload=realised_nl)
    hours = list(range(24))
    save_json({"annual": annual, "marginal_fuel_shares_2013": fuel_shares, "storage_emission_factor": ef,
               "round_trip_efficiency": ROUND_TRIP, "hours_utc": hours,
               "hourly_2013": {"adjusted": {"Oct-Mar": tab[1], "Apr-Sep": tab[0], "se_Oct-Mar": se[1], "se_Apr-Sep": se[0]},
                               "unadjusted": {"Oct-Mar": tab_u[1], "Apr-Sep": tab_u[0], "se_Oct-Mar": se_u[1], "se_Apr-Sep": se_u[0]},
                               "average_ci_test_by_hour": [float(ci13[te][:, [(2 * h + 1) % H, (2 * h + 2) % H]].mean()) for h in hours]},
               "netload_2013": nl_table, "out_of_sample_r2_dE": oos, "checks": check}, OUT / "step1b_mef.json")
    log("done")


if __name__ == "__main__":
    main()
