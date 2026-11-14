"""Fill demand_response/data/ from CVAE_reproduction/. Run only to refresh the copies; run.py does not need it.

    python -P demand_response/prepare_data.py      (from the repository root)

Copies, unchanged:
  CVAE_reproduction/data/conditions_ours.parquet                -> data/conditions_ours.parquet
  CVAE_reproduction/data/power_consump_ours.parquet             -> data/power_consump_ours.parquet
  CVAE_reproduction/outputs/chrono/models/B_c*_no_pos_y_s*.pt   -> data/cvae_models/
Computes:
  data/std_mean_kwh.csv  mean kWh of the 1,032 flat-tariff (Std) households, 365 days x 48 half-hours, from
                         CVAE_reproduction/data/lcl2013_kwh_filled.npy (145 MB, git-ignored; built by
                         CVAE_reproduction/scripts/01_build_lcl2013.py) and clusters_run_a.parquet (which rows are Std)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dr  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPRO = dr.PROJECT / "CVAE_reproduction"


def main():
    (dr.CVAE_MODELS).mkdir(parents=True, exist_ok=True)
    for name in ("conditions_ours.parquet", "power_consump_ours.parquet"):
        shutil.copy2(REPRO / "data" / name, dr.DATA / name)
    models = sorted((REPRO / "outputs" / "chrono" / "models").glob("B_c*_no_pos_y_s*.pt"))
    assert len(models) == dr.N_CLUSTERS * 20, len(models)
    for p in models:
        shutil.copy2(p, dr.CVAE_MODELS / p.name)

    kwh = np.load(REPRO / "data" / "lcl2013_kwh_filled.npy", mmap_mode="r")      # (2064 households, 365*48)
    sel = dr.read_parquet(REPRO / "data" / "clusters_run_a.parquet")
    rows = np.sort(sel.loc[sel.group == "Std", "row_in_filled"].to_numpy())
    std = np.asarray(kwh[rows], dtype=float).mean(0).reshape(365, dr.H)
    out = pd.DataFrame(std, columns=dr.cols("kwh"))
    out.insert(0, "date", pd.date_range("2013-01-01", periods=365).strftime("%Y-%m-%d"))
    out.to_csv(dr.DATA / "std_mean_kwh.csv", index=False)
    print(f"copied 2 parquet files and {len(models)} CVAE checkpoints; Std mean from {len(rows)} households")


if __name__ == "__main__":
    main()
