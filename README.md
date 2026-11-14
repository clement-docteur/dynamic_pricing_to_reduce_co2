# dynamic_pricing_to_reduce_co2

Carbon-aware electricity pricing on the Low Carbon London 2013 trial (MIT 15.095).

- **Run the MVP:** `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python run_mvp.py`
  (raw data in the parent folder or `$CARBON_RAW_DIR`; add `--reproduce` to re-check Clement's original pipeline).
- **Results and methodology:** [MVP_REPORT.md](MVP_REPORT.md); tables in `results/tables/`, figures in `results/figures/`.
- **Audit of the original implementation:** [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).
- Original pipeline: `carbon_tariff_pipeline/` (see its README); CVAE reproduction: `CVAE_reproduction/`.
