# Demand response: baseline, CVAE findings, shared predictions

This folder holds Person B's work in one place:

1. **Demand-response baseline**: a model of how ToU households change their consumption under Low and High prices.
2. **CVAE investigation**: what the Brégère & Bessa (2020) CVAE repository does and what we found when we reproduced it.
3. **Predictions in a shared format**: one long table that every model writes and any teammate can read.

| File | What it is |
|---|---|
| `dr.py` | All the code: data loading, `ResponseModel` (baseline), `CVAEEnsemble`, prediction format helpers |
| `run.py` | Fits the models, writes `outputs/predictions.csv.gz` and `outputs/metrics.json`, prints a summary |
| `outputs/predictions.csv.gz` | Predictions of every model, 2013, all clusters (420,480 rows, 4.6 MB) |
| `outputs/metrics.json` | RMSE and bias per model / split / cluster, simulated tariff effects, baseline coefficients |

## Run

From the repository root (about 10 seconds):

```bash
python -P demand_response/run.py
```

Inputs are read from `CVAE_reproduction/` (household clusters, conditions, trained CVAE checkpoints). Only one is not
committed: `CVAE_reproduction/data/lcl2013_kwh_filled.npy` (145 MB, see the top-level README). It is needed for the
flat-tariff control mean. Dependencies: numpy, pandas, duckdb. Torch is not needed (checkpoints are read in pure Python).

## 1. Demand-response baseline (`dr.ResponseModel`)

For each of the 4 ToU household clusters c, day d and half-hour h:

    log( ToU cluster mean[c,d,h] / Std mean[d,h] ) = intercept[h, working day or not]
                                                    + temperature x time-of-day band
                                                    + tariff terms

The 1,032 flat-tariff (Std) households are the control. Weather, holidays and seasonal swings affect both groups and
cancel in the ratio, so the tariff coefficients are a difference-in-differences estimate of the response. The tariff
terms are:

* in-block High and Low effects, pooled plus a ridge-shrunk deviation per time-of-day band;
* three edge effects that the data support: a dip in the 2 h before a Low block, a carry-over in the half-hour after
  it, and pre-loading in the 2 h before a High block.

`no_response` is the same regression without tariff terms. It shows what the baseline gets right from the control
alone.

Predictions are `Std mean x exp(fitted log ratio)`. They use the same day's observed Std consumption, so this is a
**simulator of the tariff response, not a day-ahead forecast**. The main pipeline uses the same model (fitted on
Jan-Oct) as the simulator in Step 3 (`carbon_tariff_pipeline/lib/demand.py`). This folder's version reproduces its
Jan-Aug coefficients to 1e-7.

## 2. CVAE investigation

Repository: Brégère & Bessa (2020), *Simulating tariff impact in electrical energy consumption profiles with
conditional variational autoencoders*, IEEE Access 8. Their code (`codes/cvae.py`, TF2/Keras) contains only the CVAE.
Data preparation, clustering, the GAM benchmark and scoring are not in it. The full reproduction is in
`../CVAE_reproduction/` (spec and evidence: `REPRO_SPEC.md`, sections 6-7). The findings that matter for us:

* **It reproduces on a random split.** Rebuilt from the raw LCL and ERA5 data, the CVAE matches the paper's RMSE
  (about 0.02 kWh). Table 3 of the paper is also reproducible once the paper's unusual Calinski-Harabasz variant is
  identified.
* **The latent space collapses.** KL is about 1e-6 at the paper's eta = 10, and also at every eta >= 0.1. The decoder
  ignores z, so the "generator" is effectively a deterministic regression from conditions to a load curve. Samples
  have far too little spread (0.002 vs 0.02 kWh RMSE).
* **The paper's tariff side effects are not in the data.** The CVAE's High-price drop starts 1-5 h early. In the
  scenario windows, this is likely learned co-occurrence (those windows almost never occur alone in training). An
  Std-controlled difference-in-differences finds no High side effect. It does find a drop before Low blocks, and in
  cluster 3 a one-slot carry-over and pre-loading before evening High.
* **It generalises poorly in time.** Trained on Jan-Aug and tested on Nov-Dec (`scripts/50_chrono_split.py`), its
  RMSE roughly triples (0.064 vs 0.019 kWh in cluster 0). Even without the position-in-year condition, which helps
  most, it beats the naive benchmark in only 1 of 4 clusters (`outputs/chrono/chrono_results.json`). This chronological, no-position-in-year ensemble is
  `dr.CVAEEnsemble`.
* **Decision:** the baseline above is the main demand-response simulator. The CVAE is kept as a sensitivity check.

Results of `run.py` (median daily RMSE, kWh per household; "all" = household-weighted ToU mean; everything trained on
Jan-Aug):

| split | response_model | no_response | cvae |
|---|---|---|---|
| val (Sep-Oct), all | 0.0075 | 0.0080 | 0.0260 |
| test (Nov-Dec), all | 0.0093 | 0.0095 | 0.0430 |
| test, cluster 3 (most responsive) | 0.0187 | 0.0195 | 0.0450 |

Simulated effect of the historical schedule vs an all-Normal day, test days, on High / Low slots: baseline
-4.5 / -4.9 / -3.0 / -28.4 % and +4.2 / +4.5 / +3.4 / +36.9 % (clusters 0-3). For the CVAE, the High effects of
clusters 0-2 have the wrong sign (+0.4 to +4.3 %).

## 3. Shared prediction format

One row per **model x scenario x cluster x day x half-hour**, gzip-compressed CSV.

| Column | Type | Meaning |
|---|---|---|
| `model` | str | `response_model`, `no_response`, `cvae` (add your own name) |
| `scenario` | str | `historical` = the trial's actual tariff; anything else is a counterfactual schedule (here `all_normal`) |
| `cluster` | int | ToU household cluster 0-3 (sizes 285 / 299 / 299 / 149, in `metrics.json`) |
| `date` | date | 2013-01-01 ... 2013-12-31 |
| `slot` | int | half-hour 0-47; slot h of `date` is the half-hour **ending** at `date` + h x 30 min |
| `interval_end_utc` | datetime | the same end time, explicit (GMT = UTC; slot 0 ends at 00:00 of `date`) |
| `split` | str | `train` Jan-Aug, `val` Sep-Oct, `test` Nov-Dec |
| `tariff` | str | price in that slot under the row's scenario: `L` 3.99, `N` 11.76, `H` 67.20 p/kWh |
| `y_true_kwh` | float | observed mean kWh per household in the cluster (empty for counterfactual scenarios) |
| `y_pred_kwh` | float | predicted mean kWh per household |

Values are per household, so a whole-population figure is `sum over clusters of size x value`. To join with NESO
carbon intensity (which labels the start of each half-hour), use `interval_end_utc - 30 min`.

```python
import sys; sys.path.insert(0, "demand_response")
import dr
pred = dr.read_predictions("demand_response/outputs/predictions.csv.gz")
test = pred.query("scenario == 'historical' and split == 'test'")
```

To add a model, build its `(4, 365, 48)` kWh array and call
`dr.write_predictions(dr.predictions_frame(name, scenario, data, y_pred, low, high, y_true), path)`, where
`data = dr.load_data()`.
