"""Fit the demand-response models, save their predictions in the shared format and score them.

    python -P demand_response/run.py          (from the repository root, about 1 minute)

Models (all trained on Jan-Aug 2013 only; Sep-Oct = validation, Nov-Dec = test):
  response_model   Std-controlled tariff-response regression (the baseline, dr.ResponseModel)
  no_response      the same regression without tariff terms (what the baseline gets without any tariff response)
  cvae             the reproduction's CVAE ensemble (dr.CVAEEnsemble)
Scenarios: `historical` (the trial's actual schedule, scored against the observed consumption) and `all_normal`
(every slot at the Normal price; counterfactual, no observed values).

Outputs in demand_response/outputs/: predictions.csv.gz (shared format, see README.md) and metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dr  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = dr.HERE / "outputs"


def scores(pred: pd.DataFrame, weights) -> list[dict]:
    """Per model, split and cluster ('all' = household-weighted ToU mean): median over days of the RMSE over the
    48 slots (kWh), and the energy bias in %."""
    hist = pred[pred.scenario == "historical"].copy()
    hist["w"] = hist["cluster"].map(dict(enumerate(weights)))
    for col in ("y_true_kwh", "y_pred_kwh"):
        hist[f"w_{col}"] = hist["w"] * hist[col]
    agg = (hist.groupby(["model", "split", "date", "slot"], as_index=False)[["w_y_true_kwh", "w_y_pred_kwh"]].sum()
           .rename(columns={"w_y_true_kwh": "y_true_kwh", "w_y_pred_kwh": "y_pred_kwh"}).assign(cluster="all"))
    both = pd.concat([hist.assign(cluster=hist.cluster.astype(str)), agg])
    rows = []
    for (model, split, cluster), g in both.groupby(["model", "split", "cluster"], sort=False):
        per_day = g.assign(se=(g.y_pred_kwh - g.y_true_kwh) ** 2).groupby("date")["se"].mean() ** 0.5
        rows.append({"model": model, "split": split, "cluster": cluster, "rmse_median_kwh": per_day.median(),
                     "bias_pct": 100 * (g.y_pred_kwh.sum() / g.y_true_kwh.sum() - 1)})
    return rows


def tariff_effects(pred: pd.DataFrame) -> list[dict]:
    """Simulated effect of the historical tariff vs an all-Normal day, % of consumption, on Low / High slots."""
    keys = ["model", "cluster", "date", "slot"]
    hist = pred[pred.scenario == "historical"]
    normal = pred.loc[pred.scenario == "all_normal", keys + ["y_pred_kwh"]]
    p = hist.merge(normal, on=keys, suffixes=("", "_normal"))
    rows = []
    for (model, split, cluster, t), g in p[p.tariff != "N"].groupby(["model", "split", "cluster", "tariff"]):
        rows.append({"model": model, "split": split, "cluster": int(cluster), "tariff": t, "n_slots": len(g),
                     "effect_pct": 100 * (g.y_pred_kwh.sum() / g.y_pred_kwh_normal.sum() - 1)})
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    data = dr.load_data()
    train = data.split == "train"
    weights = data.cluster_sizes / data.cluster_sizes.sum()
    print(f"clusters: {data.cluster_sizes.tolist()} ToU households; train/val/test days: "
          f"{[int((data.split == s).sum()) for s in dr.SPLIT]}")

    models = {"response_model": dr.ResponseModel(tariff=True).fit(data, train),
              "no_response": dr.ResponseModel(tariff=False).fit(data, train),
              "cvae": dr.CVAEEnsemble(data)}
    zero = np.zeros_like(data.low)
    scenarios = {"historical": (data.low, data.high), "all_normal": (zero, zero)}
    frames = []
    for name, model in models.items():
        for scen, (low, high) in scenarios.items():
            y_pred = model.predict(data, low, high) if isinstance(model, dr.ResponseModel) else model.predict(low, high)
            frames.append(dr.predictions_frame(name, scen, data, y_pred, low, high,
                                               y_true=data.y_tou if scen == "historical" else None))
    pred = pd.concat(frames, ignore_index=True)
    dr.write_predictions(pred, OUT / "predictions.csv.gz")
    print(f"wrote {len(pred):,} rows to {OUT / 'predictions.csv.gz'}")

    metrics = {
        "cluster_sizes": data.cluster_sizes.tolist(),
        "scores": scores(pred, weights),
        "tariff_effects": tariff_effects(pred),
        "response_model_coefficients_pct": {c: models["response_model"].tariff_effects_pct(c)
                                            for c in range(dr.N_CLUSTERS)},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float), encoding="utf-8")

    sc = pd.DataFrame(metrics["scores"])
    print("\nmedian daily RMSE, kWh per household (cluster 'all' = household-weighted ToU mean):")
    print(sc.pivot_table(index=["split", "cluster"], columns="model", values="rmse_median_kwh", sort=False)
          .round(4).to_string())
    ef = pd.DataFrame(metrics["tariff_effects"])
    print("\nsimulated effect of the historical tariff vs all-Normal on Low/High slots, test days, %:")
    print(ef[ef.split == "test"].pivot_table(index=["tariff", "cluster"], columns="model", values="effect_pct")
          .round(1).to_string())


if __name__ == "__main__":
    main()
