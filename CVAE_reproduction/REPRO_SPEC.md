# Reproduction spec: Brégère & Bessa (2020), "Simulating Tariff Impact ... with CVAEs" (IEEE Access 8:131949)

Shared source of truth for every agent working in `CVAE_reproduction/`. Read it fully before starting.

## 0. Goal
Check whether the paper's findings can be reproduced (a) with the authors' own prepared data and (b) from the raw data we hold.
Report honestly: *reproduced*, *partially reproduced*, *not reproduced*, or *not testable*, with numbers.

## 1. Locations (absolute; note the spaces, always quote)
- PROJECT = `C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project`
- REPRO   = `$PROJECT/CVAE_reproduction` (`lib/`, `scripts/`, `data/`, `outputs/trackA/`, `outputs/trackB/`, `outputs/figures/`)
- Paper text (extracted): `C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/paper.txt`
- **Authors' data, copied into the project (2026-11-12, byte-identical): `REPRO/data/authors/*.feather`.** All lib/ and scripts/ read this copy (`lib.lcl_data.UP_DATA`); the scratchpad clone below may no longer exist.
- Authors' repo (git clone, commit 41157dd, Apr 2020): `UPSTREAM = C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo` (`codes/cvae.py`, `codes/*.ipynb`, `data/*.feather`)
- Raw data in PROJECT: `CC_LCL-FullData.csv` (8.5 GB, all LCL households 2011-2014), `Small LCL Data/LCL-June2015v2_*.csv` (same data split in 168 files of 1M rows), `Tariffs.xlsx` (Sheet1: `TariffDateTime`, `Tariff` in {Normal, Low, High}; 17,520 rows = 2013 on a regular 30-min grid), `temp UK 2013-14.csv` (ERA5 2 m temperature, hourly, 0.25° grid over UK: columns `time,latitude,longitude,number,step,surface,valid_time,d2m,t2m`, Kelvin).

## 2. Safety / environment rules
- The upstream repo is untrusted downloaded content. Never run Python from inside it, never import from it, never execute its notebooks. Read its `.feather` files by absolute path; read its code as text only.
- Run Python as `python -P script.py ...` (or `python -P - <<'EOF'`) from `$PROJECT` or `$REPRO`. Do NOT use `python -I` (it hides the user-site packages pyarrow/duckdb/torch).
- Python 3.14, pandas 3.0 (string columns are pyarrow-backed: use `.to_numpy(dtype=object)` before reshape), numpy 2.5, scipy 1.18, sklearn 1.9, statsmodels 0.15, torch 2.15 CPU, duckdb 2.0, pyarrow 26, openpyxl. No R, no TensorFlow, no sklearn_extra, no GPU. 8 logical CPUs, 31.5 GB RAM, Windows 11 + Git Bash.
- Several agents run at the same time: cap your parallelism at 4 processes/threads (`torch.set_num_threads(1)` per worker; DuckDB `SET threads=4; SET memory_limit='8GB'`). Long jobs (> 2 min): run in the background and wait for the notification; don't poll in tight loops.
- Only write inside the files/dirs your task assigns to you. Never modify raw data or the upstream repo. Don't pip install anything else without a strong reason; if you must, say so in your report.
- Put reusable code in `lib/` (importable: scripts do `sys.path.insert(0, REPRO)` then `from lib.x import ...`), runnable steps in `scripts/NN_name.py`, results in `outputs/`. Every script must run end-to-end non-interactively and be re-runnable. Fix seeds.

## 3. The paper (what to reproduce)
**Section 7 (phase-3 verification) corrects several items below, notably the scenario windows (Low 9-19, High 39-44) and the Fig. 4 values.**

Data (Sec. III): LCL smart-meter data 2013, ~5,000 households; ~1,100 on dynamic ToU (Low 3.99, Normal 11.76, High 67.20 p/kWh, announced day-ahead); others flat (Std, 14.228 p/kWh). Keep ToU households with >95% of 2013 data → **1,007**; sample the same number of Std households. Fill gaps by linear interpolation (short gaps) or using preceding/following days (long gaps). Temperature: London hourly NOAA observations, linearly interpolated to 30 min. Smoothed temperature (App. A): one continuous half-hourly series, s_1 = τ_1, s_k = (1-a) τ_k + a s_{k-1}, **a = 0.998**; daily τ̄_t = mean of that day's 48 smoothed values. w_t = 1 weekday / 0 weekend. κ_t increases linearly 0→1 over the year. T=365 days split randomly: **273 train / 92 test** (use the authors' exact dates, see §4).

Clustering (Sec. IV + App. B, C): per household i and half-hour h, Gaussian location-scale GAM: mean = s(τ_t^h) + ξ_L·1Low + ξ_N·1Normal + ξ_H·1High; sd = γ_L·1Low + γ_N·1Normal + γ_H·1High (mgcv gaulss). μ_i^h(p) = (1/T) Σ_t μ̂_i(x_t^h, p). Rescale each household's three 48-profiles by μ̄_i = mean_h μ_i^h(Normal); stack M (|I| × 144); NMF with **r = 5** (sklearn NMF, coordinate descent) → W (|I|×5); **k-medoids, k = 4**, Euclidean (sklearn_extra KMedoids; implement yourself). Done separately for ToU and Std (Std uses the ToU tariff schedule as a placebo "tariff"). Baselines: random labels; "classical features" (min, max, mean consumption in winter and in summer [hot = Apr-Sep], peak half-hour and trough half-hour; rescaled; k-medoids k=4).
- Paper ToU cluster sizes (NMF): **325 / 295 / 229 / 158** (classical features: 255/247/249/256; random: 248/264/242/253).
- **Table 3 (Calinski-Harabasz)**:

| record series | Rd ToU | Features ToU | NMF ToU | Rd Std | Features Std | NMF Std |
|---|---|---|---|---|---|---|
| Non-standardized | 4 627 | 40 764 | 13 432 | 5 014 | 5 088 | 12 971 |
| Standardized | 4 950 | 6 870 | 14 088 | 4 834 | 4 741 | 14 934 |
| Special tariff - standardized | 5 151 | 7 033 | 16 070 | 4 904 | 4 694 | 15 460 |

  Headline: +128% for ToU special-tariff standardized, NMF vs Features (16 070 / 7 033). Paper formula (eq. 3-5): S = (|I|-k)·Σ_ℓ‖Y(C_ℓ)-Y(I)‖² / ((k-1)·Σ_ℓ (1/|C_ℓ|)Σ_{i∈C_ℓ}‖Y(i)-Y(C_ℓ)‖²), with Y(i) = the household's full-year record vector (raw / divided by its own mean / only Low-or-High records of the normalized vector). NOTE: under that formula (and under sklearn's CH) random labels give ≈1, yet the paper reports ≈5 000 for random clusterings, so the paper's actual computation must differ. Treat the absolute values as "unknown definition"; compare orderings/ratios, and try to find a variant that reproduces the ~5 000 random baseline.

CVAE (Sec. V): conditions X_t = [w_t, 48 Low dummies, 48 High dummies, 3 temperature-PCA components rescaled to [0,1], κ_t] (paper: 100 dims; the code also adds `temp_smooth_n` → 102). Paper: latent d = 4; one hidden layer of 15 ReLU units in encoder and decoder; Glorot-uniform init; η = 10 constant; Adam lr 1e-3; consumption min-max scaled to [0,1] using train min/max of the cluster; trained 50 times, keep the decoder with the lowest test MSE. Temperature PCA: on the 48 half-hourly temps + smoothed temp of each day; "3 components explain 98% of variance". Consumption PCA: "4 components explain >80% of variance". Grid search (d 1-20, units 10-25) picked d=4, 15 units.

GAM benchmark (Sec. VI, App. E): per half-hour h: f^h = s_τ(τ_t^h) + s_τ̄(τ̄_t) + s_κ(κ_t) + α^h w_t + ξ_Low^h 1Low + ξ_High^h 1High (cubic splines, mgcv). σ^h(p) from the causality model refitted on the cluster's aggregate consumption. Standardized residuals e_t^h = (Y_t^h - f^h)/σ^h(p_t^h) on training days → Σ = their empirical correlation matrix. Samples: Y = f + diag(σ(p)) E, E ~ N(0, Σ).

Evaluation (Sec. VII): N = 200 samples per test day per generator; scores per (cluster, test day):
- RMSE = ‖mean of samples - y‖ (eq. 12 says Euclidean norm; Fig. 4 values ≈0.02 kWh suggest root-mean over the 48 half-hours. Compute both, report which matches.)
- Energy score (eq. 14): (2/N) Σ_{i≤N/2} ‖Ŷ_i - y‖ - (1/N) Σ_{i≤N/2} ‖Ŷ_i - Ŷ_{N/2+i}‖ (Euclidean, kWh).
- Variogram score p = 0.5 (eq. 16): Σ_{h,h'} (|y_h - y_h'|^0.5 - (1/N) Σ_i |Ŷ_i^h - Ŷ_i^h'|^0.5)².
- **Fig. 4 approximate medians (read off boxplots), clusters 0/1/2/3:** RMSE CVAE ≈ 0.020/0.022/0.018/0.025, GAM ≈ 0.015/0.016/0.015/0.018; ES CVAE ≈ 0.150/0.145/0.170/0.130, GAM ≈ 0.155/0.150/0.175/0.135; VG CVAE ≈ 6/7/7/10, GAM ≈ 4/4/5/5.
- Claims: GAM better on RMSE ("13% difference" in the abstract); CVAE slightly better (lower) energy score; CVAE samples have too little variance and GAM samples too much; CVAE consecutive half-hours very correlated, GAM erratic.
- Tariff impact (Sec. VII-C, Fig. 6, 8): for a test day, three condition vectors that differ only in tariff: Normal all day; Low from 04:30 to 09:30 (index convention below: slots 9-18); High from 19:30 to 22:00 (slots 39-43). Claims: both generators raise consumption under Low and lower it under High; the CVAE also shows a **side effect** (the drop under High starts a little before and lasts a little after the window) and a **rebound effect** (after a morning Low, afternoon/evening consumption falls; with an evening High, afternoon consumption rises a bit); the GAM cannot (effects confined to the window). Responsiveness differs by cluster. CVAE generalizes poorly to tariff profiles never seen (e.g., all-day High).

## 4. Authors' code and data (UPSTREAM)
`codes/cvae.py` (TF2/Keras + tfp, float64):
- Encoder: Dense(15, relu) → Dense(2d, linear) = [μ, logscale] where scale = exp(logscale) is the *standard deviation*; z = μ + exp(logscale)·ε. Decoder: Dense(15, relu) → Dense(48, linear). Inputs: encoder [y(48), cond], decoder [z(d), cond]. (`cvae_generate.ipynb`: nb_layers 2, units [15, 8] / [15, 48].)
- Loss per sample: η·KL(N(μ, diag(scale²)) ‖ N(0, I)) + Σ_h (y_h - ŷ_h)² (loss='L2'), η = kl_coeff = 10 constant.
- Training: Adam(1e-3), **batch size 1**, samples in fixed row order (no shuffle), **50 epochs**, nb_init = 50. Quirks: one Adam instance is shared across all 50 restarts and `glorot_uniform(seed=0)` is reused (in TF2 this likely gives identical initial weights each restart). Port faithfully but use a different seed per restart (the paper's intent), and document the quirk.
- `compute_mse` (selection metric): z ~ encoder posterior, 100·sqrt(mean_h (y-ŷ)²) on scaled data, averaged over test days. `compute_mse_test` uses z ~ N(0, I).
- Generation: z ~ N(0, I_4), ŷ = decoder([z, cond])·(max-min) + min.
- cols_cond order = `['temp_smooth_n','pca_temp_0','pca_temp_1','pca_temp_2','pos_y','work_day','low_00','high_00','low_01','high_01',...,'low_47','high_47']` (102 columns).
- NOT in the repo: data preparation, clustering, causality model, GAM generator, scores, figures.

`data/*.feather` (pandas.read_feather):
- `power_consump_nmf.feather` (1460 = 365 days × 4 clusters): `date, consumption_00..47` (cluster-mean kWh per half-hour), `price_00..47`, `temp_00..47` (°C), `low_00..47`, `high_00..47`, `temp_smooth`, `pos_y`, `work_day`, `pca_temp_0..2`, `temp_smooth_n`, `cluster` (0-3).
- `train_nmf.feather` (1092 = 273×4) / `test_nmf.feather` (368 = 92×4): same columns, the exact split. `train_ToU` / `test_ToU`: same for the all-ToU aggregate (= size-weighted mean of the 4 clusters, verified exactly).
- `sum_up_nmf.feather`: cluster sizes 325/295/229/158 (clusters 0/1/2/3). `agreg_profiles_nmf.feather` (1152): `instant (0-47), price (low/normal/high), type (mean/variance), cluster, kWh` = the causality-model profiles per cluster. `condition.feather`: `pca_temp_0..2, temp_smooth_n, date` (365).
- Facts verified: `pos_y` = mean over the day's half-hours of i/17520 (i = 0..17519); `work_day` = 1 Mon-Fri (holidays NOT removed; 261 work days); first test days are 2013-01-01..03.

Tariff alignment (verified): the authors' `low_/high_` dummies equal our `Tariffs.xlsx` shifted by **+1 slot in GMT months (Jan-Mar 30) and -1 slot in BST months (Mar 31-Oct 26)**, i.e. one of the two files is in local clock time and the other in GMT, with a 1-slot labeling offset. Nov-Dec also differ in content (authors: 143 event days, 1420 Low / 740 High slots; ours: 153 event days, 1660 Low / 788 High). LCL `DateTime` has no DST gap/duplication (GMT). Which alignment is physically right must be established empirically (see the data task).

Index convention: "slot h" = column `consumption_hh` / `low_hh` of the authors' files. When building our own data, map LCL timestamps to slots so that our aggregates match the authors' (determine the offset empirically), and record the mapping in `outputs/data_checks.json`.

## 5. Shared data contract
Our rebuilt data must use the authors' schema (columns above, one row per date × cluster, `date` as datetime64, `cluster` int) so that the same modeling code runs on both tracks:
- `data/conditions_ours.parquet`: 365 rows, `date, temp_00..47, temp_smooth, temp_smooth_n, pca_temp_0..2, pos_y, work_day, low_00..47, high_00..47, price_00..47` (our Tariffs.xlsx, mapped with the empirically correct alignment), plus `low_auth_00..47/high_auth_00..47` copied from the authors' data for sensitivity runs.
- `data/power_consump_ours.parquet`: like `power_consump_nmf` but from our clusters; `data/train_ours.parquet`, `data/test_ours.parquet` with the authors' exact train/test dates.

Model outputs (both tracks, both generators): for cluster c, `outputs/<track>/<model>_samples_c<c>.npy` with shape (200, 92, 48) in kWh, test days in ascending date order (`outputs/<track>/test_dates.csv`); tariff scenarios `outputs/<track>/<model>_scen_<Normal|Low|High>_c<c>.npy`, same shape (all 92 test days; only the 96 tariff dummies changed: Normal = all zero; Low = low slots 9-18 set to 1; High = high slots 39-43 set to 1; everything else from the real test-day conditions). `<model>` ∈ {cvae, gam}. Save fitted models under `outputs/<track>/models/`.

## 6. Phase-1 findings (established; build on them, do not redo)
Data (`outputs/data_checks.json`, `lib/lcl_data.py`):
- Our raw rebuild gives **1,032** ToU households with >95% 2013 data (Std 4,034). The authors' 1,007 are an exact subset (NNLS identification); flags in `data/selection.parquet`: `in_authors_tou_1007` (their ToU set) and `in_1007_pair` (those 1,007 ToU + the first 1,007 Std of our random draw). The 25 extra households have no distinguishing data feature.
- `data/lcl2013_kwh_filled.npy`: float32 (2,064 × 17,520) = 1,032 ToU rows then 1,032 Std rows, row order = `selection.row_in_filled`; gaps filled with the authors' recovered rule (1-slot gaps linear in time; longer gaps linear across days at the same half-hour). Column j = slot j of 2013 in the authors' convention: **slot h of date d = the LCL reading labelled d + h·30 min GMT, i.e. the half-hour ENDING at h·30 min** (slot 0 = 23:30-24:00 of the previous day). With the identified 1,007 households this reproduces the authors' ToU aggregate to 2.8e-8 kWh (mean abs diff).
- Tariff alignment (consumption event study): the authors' `low_/high_` dummies are physically aligned (offset 0 in both seasons). `Tariffs.xlsx` is in **London clock time, start-labelled**: correct use = `lib.tariffs.select_tariff_version(df, 'loc')` (local → GMT, +1 slot). Its *content* differs from the authors' on 17 days (03-29, 05-03, 2 days in Nov, 13 in Dec); the consumption response supports the authors' schedule on those days (11 extra xlsx event days show no response).
- Raw ToU−Std response under the right alignment: High effect −13 to −16 Wh/household/half-hour inside the window, **no pre- or post-window effect**; Low effect +14 to +18 Wh inside, a small pre-window deficit (−3 Wh) and a one-slot carry-over (+5 Wh); no significant rebound later in the day (low power).

Conditions (`outputs/conditions_checks.json`, `lib/conditions.py`, `lib/tariffs.py`, `data/conditions_ours.parquet`):
- Authors' recipes recovered exactly from their data: `temp_smooth` = App. A (a = 0.998, s_1 = τ_1); `temp_smooth_n` = min-max over 365 days; `pca_temp_0..2` = PCA (sklearn, centred, not standardized) of the 365×48 half-hourly temps ONLY, fitted on all 365 days, then ONE global min-max over the 365×3 scores; `pos_y` = daily mean of i/17519; `work_day` = Mon-Fri (no holidays). 3 PCA comps explain 98.93% (paper 98%).
- Ours: ERA5 bilinear at central London, offset 0: RMSE 1.32 °C vs the authors' NOAA temps (bias −1.09 °C, r 0.993); derived conditions r = 0.999 (temp_smooth_n), 0.997/0.969/0.915 (pca 0/1/2). `conditions_ours.parquet` holds our temps/calendar plus three tariff versions with suffixes `_auth`, `_gmt`, `_loc` (`low_loc_hh`, `high_loc_hh`, `price_loc_hh`, ...). There are no unsuffixed tariff columns yet.

Track A (authors' data):
- CVAE port `lib/cvae_torch.py` (faithful, Keras-exact Adam, selftest passes): `train_cvae`, `test_mse`, `generate`, `save_model`, `load_model`, `run_restart_job`, `COLS_COND`, `COLS_DATA`. Selected restarts (lowest test MSE): c0 seed 40, c1 32, c2 25, c3 31; models in `outputs/trackA/models/cvae_selected_c<c>.pt`. **Posterior collapse**: KL ≈ 1e-6, posterior std 1.000; samples nearly deterministic (spread ≈ 0.002 kWh vs RMSE ≈ 0.02). Preview RMSE (root-mean) 0.0191/0.0223/0.0179/0.0235 kWh (paper ≈0.020/0.022/0.018/0.025). Training not converged at 50 epochs; chronological batch-1 training gives recency bias. Scenario effects: strong and consistent only for cluster 3 (Low +50%, High −25%, drop starting ~3 h before the High window); clusters 0-2 restart-dependent, often wrong sign. The paper's scenario windows never occur alone in training (High 39-43 only inside longer runs, e.g. 33-44, 35-46), so "side effects" may be learned co-occurrence. Per-restart effects: `outputs/trackA/cvae_scenario_effects_all_restarts.csv`.
- GAM `lib/gam_generator.py` (`fit_gam_generator`, `generate`, `predict_mean`, `tariff_scenario`), `lib/causality.py` (`fit_causality`, batched, LAML/GCV), `lib/splines.py` (mgcv-like cr splines). Params `outputs/trackA/models/gam_params_c<c>.pkl`. Causality-model mean profiles match the authors' `agreg_profiles_nmf` (r ≈ 0.995-0.999) **after remapping the cluster labels: power_consump cluster → agreg cluster {0:1, 1:3, 2:0, 3:2}**; the authors' 'variance' column is not reproducible from aggregates (probably household-level). GAM RMSE 0.0142/0.0162/0.0148/0.0187 and VG 3.8/4.0/4.8/5.8 match Fig. 4; **energy score 0.074-0.095 vs the paper's 0.135-0.175 does not**. GAM sample SD / RMS test residual = 1.26-1.43 (too much variance). GAM tariff effects confined to the window (by construction).
- Sample arrays: `outputs/trackA/{cvae,gam}_samples_c<c>.npy`, `{cvae,gam}_scen_{Normal,Low,High}_c<c>.npy` (200, 92, 48) kWh; scenario arrays share random draws with each other (CRN). `outputs/trackA/test_dates.csv`.

Decision for Track B (our data): household set = our 1,032 ToU (+ 1,032 Std); temperatures = ERA5 (conditions_ours); tariffs = `Tariffs.xlsx` with the physically right alignment (`'loc'` version). Sensitivity: the identified 1,007 set and/or the authors' tariff schedule (`'auth'`).

## 7. Phase-3 verification findings (2026-11-12). These SUPERSEDE sections 3-6 where they conflict.
Evidence: `outputs/verify/<verifier>/` (v-paper, v-code, v-collapse, v-data, v-clustering, v-numbers).
- **Scenario windows (paper Figs. 6/8): Low = slots 9-19 (11 half-hours), High = slots 39-44 (6 half-hours), endpoints inclusive.** The paper's GAM panels are non-zero exactly on those slots, and our GAM's per-slot coefficients at slots 19 and 44 match the paper's to <= 0.002 kWh. The 9-18 / 39-43 of sections 3 and 5 were off by one.
  - Clock times: under the verified end-labelled convention, 9-19 = 04:00-09:30 GMT and 39-44 = 19:00-22:00 GMT. The paper's "04:30-09:30 / 19:30-22:00" reads slot h as the time h x 30 min.
  - Side regions for these windows: High pre4 = 35-38, post = 45-47 (only 3 slots remain in the day); Low pre4 = 5-8, post4 = 20-23, rebound = 20-31 and 32-47.
  - Training co-occurrence: 39-44 occurs alone on exactly 1 training day. 18 of the 19 training High runs that contain it are longer (33-44 x7, 35-46 x6, 21-44 x3, 33-47, 35-47).
- **Fig. 4 exact medians** (read from the PDF's vector paths, `outputs/verify/v-paper/fig4_exact_from_pdf_vectors.csv`), clusters 0/1/2/3:

  | Score | CVAE | GAM |
  |---|---|---|
  | RMSE | 0.0191/0.0219/0.0176/0.0244 | 0.0146/0.0176/0.0158/0.0188 |
  | ES | 0.1500/0.1441/0.1696/0.1343 | 0.1551/0.1532/0.1687/0.1346 |
  | VG | 5.94/6.62/6.79/10.54 | 3.85/4.22/5.18/6.04 |

  The RMSE gap these imply is 19.5% (relative to the CVAE) or 24.3% (relative to the GAM). The CVAE's ES is above the GAM's in cluster 2.
- **The paper's ES panel contradicts the paper's own Figs. 6-7.** On 2013-01-01, cluster 3, the CVAE samples of Fig. 6 against the observation of Fig. 7 give ES 0.256-0.260, but Fig. 4 draws nothing above 0.215. The paper's own RMSE medians and sample spreads imply ES values close to ours.
- **The paper's figures vs its text and captions:**
  - Fig. 5 shows cluster 2 (229 households), not the 158-household cluster its caption names.
  - Fig. 6 shows cluster 3, with 10 samples plus their mean.
  - The rows of Figs. 7 and 8 are clusters 0-3, top to bottom.
  - Fig. 8: the CVAE's Low does not raise use in clusters 1-2 and is about 0 in cluster 0. Its High drop starts 1-5 h early and lasts to midnight in all 4 clusters.
  - The paper's CVAE is near-deterministic too: per-slot SD 0.003-0.005 kWh.
- **Table 3 is reproducible.** It uses CH with NON-squared Euclidean norms and a |C_l|-weighted between-cluster term: S = (|I|-k)/(k-1) * sum_l |C_l| ||Y(C_l)-Y(I)|| / sum_l mean_{i in C_l} ||Y(i)-Y(C_l)||.
  - On the recovered partition: 13,433 / 14,076 / 16,055 (paper 13,432 / 14,088 / 16,070). Random labels give about 5,000.
  - Our runs: NMF vs Features, ToU special-tariff records, +87% to +110% (paper +128%).
  - The paper's Features-Std column is indistinguishable from random labels.
- **Posterior collapse:**
  - It happens at every eta >= 0.1, including the standard eta = 1. Latent units activate only at eta <= 0.03.
  - At eta = 0.01, spread/skill reaches only 0.52-0.63 (at most 0.70 over 280 models) and RMSE is 11-63% worse.
  - A second cause: generation drops the Gaussian decoder noise (variance eta/2 on the scaled data).
  - Track A train KL is 2.2-5.8e-6 nats/day. All 15 encoder ReLUs are dead in every selected model.
  - Upstream has no restart-selection code; it saves only decoders. compute_mse and compute_mse_test pick the same restart.
- **Data, from an Std-controlled difference-in-differences (`outputs/verify/v-data/`):**
  - No High side effect. Before: -0.8/-0.5/-0.3/+2.0%; after: +1.1/0.0/+0.8/+0.4%; none significant.
  - A robust drop BEFORE Low blocks: -2.0/-1.0/-2.6/-6.4%. In cluster 3 it builds over about 5 h.
  - In cluster 3, a one-slot carry-over after Low (+11.7%), and pre-loading before evening High (+8.5%, n = 9).
  - No window-specific rebound.
  - In-block effects: Low +3.7/+4.3/+1.1/+34.6%, High -4.3/-4.5/-2.5/-29.2%. Cluster 3's Low effect ranges from +27% to +45% depending on the estimator.
  - Designs without an Std control inflate the pre-High effects.
  - Tariff content: the consumption favours the authors' schedule on 15 of the 17 disputed days. The authors' own extra Low blocks on 12-29 and 12-30 show no response.
