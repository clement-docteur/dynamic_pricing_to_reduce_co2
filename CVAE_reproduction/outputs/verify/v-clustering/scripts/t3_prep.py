"""Prepare Table-3 inputs once (ToU 1,007 households, authors' partition): raw / std / special matrices.
Saves an .npz in the scratch folder.  Does NOT import REPRO/lib."""
import numpy as np, pandas as pd
REPRO = "C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/"
U = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo/data/"
SCR = "C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-clustering/"
rec = pd.read_parquet(REPRO + "data/authors_clusters_recovered.parquet")
F = np.load(REPRO + "data/lcl2013_kwh_filled.npy", mmap_mode="r")
X = np.asarray(F[rec.row_in_filled.to_numpy()], dtype=np.float64)
lab = rec.cluster.to_numpy()
pc = pd.read_feather(U + "power_consump_nmf.feather"); pc["d"] = pd.to_datetime(pc.date.astype(str))
c0 = pc[pc.cluster == 0].sort_values("d")
low = c0[[f"low_{h:02d}" for h in range(48)]].to_numpy().reshape(-1)
high = c0[[f"high_{h:02d}" for h in range(48)]].to_numpy().reshape(-1)
special = (low + high) > 0
# Std households: the in_1007_pair Std rows (paper-sized design)
sel = pd.read_parquet(REPRO + "data/selection.parquet")
std_rows = sel.loc[(sel.group.to_numpy(dtype=object) == "Std") & sel.in_1007_pair.to_numpy(), "row_in_filled"].to_numpy()
XS = np.asarray(F[std_rows], dtype=np.float64)
# run a features/nmf labels for ToU on the same 1,007 (for orderings)
ra = pd.read_parquet(REPRO + "data/clusters_run_c.parquet").set_index("LCLid")
ids = rec.LCLid.to_numpy(dtype=object)
feat_c = ra.loc[list(ids), "features"].to_numpy(); nmf_c = ra.loc[list(ids), "nmf"].to_numpy()
std_ids = sel.loc[(sel.group.to_numpy(dtype=object) == "Std") & sel.in_1007_pair.to_numpy(), "LCLid"].to_numpy(dtype=object)
featS = ra.loc[list(std_ids), "features"].to_numpy(); nmfS = ra.loc[list(std_ids), "nmf"].to_numpy()
np.savez(SCR + "t3_inputs.npz", X=X, lab=lab, special=special, XS=XS, feat_c=feat_c, nmf_c=nmf_c, featS=featS, nmfS=nmfS,
         dates=c0.d.dt.strftime("%Y-%m-%d").to_numpy().astype("U10"))
print(X.shape, XS.shape, special.sum(), np.bincount(lab), np.bincount(feat_c), np.bincount(nmfS), np.bincount(featS))
