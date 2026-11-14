import os
os.environ.setdefault("OMP_NUM_THREADS","2"); os.environ.setdefault("MKL_NUM_THREADS","2")
import pandas as pd, numpy as np, json
REPRO="C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/"
UP="C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo/data/"
OUT="C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-data/"
sel=pd.read_parquet(REPRO+"data/selection.parquet")
M=np.load(REPRO+"data/lcl2013_kwh_filled.npy", mmap_mode='r')
rows=np.sort(sel.loc[sel.in_authors_tou_1007,'row_in_filled'].to_numpy())
print("n 1007 flag:", len(rows), "groups:", sel.loc[sel.in_authors_tou_1007,'group'].value_counts().to_dict())
X=np.asarray(M[rows,:],dtype=np.float64)
print("NaN in 1007 rows:", np.isnan(X).sum())
agg=X.mean(0).reshape(365,48)
ccols=[f"consumption_{h:02d}" for h in range(48)]
tr=pd.read_feather(UP+"train_ToU.feather"); te=pd.read_feather(UP+"test_ToU.feather")
au=pd.concat([tr,te]).sort_values('date').reset_index(drop=True)
print("authors ToU dates:", len(au), au.date.min(), au.date.max(), "unique", au.date.nunique())
A=au[ccols].to_numpy(dtype=np.float64)
d=np.abs(agg-A)
res={"n_households":int(len(rows)),"max_abs_diff_kwh":float(d.max()),"mean_abs_diff_kwh":float(d.mean()),
     "median_abs_diff":float(np.median(d)),"corr":float(np.corrcoef(agg.ravel(),A.ravel())[0,1]),
     "mean_level_kwh":float(A.mean()),"n_cells_gt_1e-6":int((d>1e-6).sum()),"n_cells_gt_1e-4":int((d>1e-4).sum())}
# where are the largest diffs
idx=np.argsort(d.ravel())[::-1][:8]
res["top_diffs"]=[(str(au.date.iloc[i//48]), int(i%48), float(d.ravel()[i])) for i in idx]
# cluster weighted mean from power_consump_nmf
pc=pd.read_feather(UP+"power_consump_nmf.feather")
print("pc columns containing cluster:", [c for c in pc.columns if 'clus' in c])
su=pd.read_feather(UP+"sum_up_nmf.feather")
sizes=dict(zip(su.cluster,su.number_clients))
W=np.zeros((365,48))
for c,n in sizes.items():
    sub=pc[pc.cluster==c].sort_values('date')
    assert len(sub)==365
    W+=n*sub[ccols].to_numpy(dtype=np.float64)
W/=sum(sizes.values())
d2=np.abs(agg-W); d3=np.abs(W-A)
res["vs_weighted_clusters"]={"max":float(d2.max()),"mean":float(d2.mean())}
res["weighted_clusters_vs_ToU_files"]={"max":float(d3.max()),"mean":float(d3.mean())}
# alternative shifts as falsification
flat=X.mean(0)
for s in (-1,1):
    sh=np.roll(flat,s).reshape(365,48)
    dd=np.abs(sh-A)[1:-1]
    res[f"shift_{s:+d}_mean_abs"]=float(dd.mean())
# Also all 1032 ToU
rows32=np.sort(sel.loc[sel.group=='ToU','row_in_filled'].to_numpy())
a32=np.asarray(M[rows32,:],dtype=np.float64).mean(0).reshape(365,48)
res["all_1032_vs_authors"]={"max":float(np.abs(a32-A).max()),"mean":float(np.abs(a32-A).mean())}
# Cluster check: recovered partition vs authors' cluster series
ac=pd.read_parquet(REPRO+"data/authors_clusters_recovered.parquet")
print(ac.cluster.value_counts().sort_index().to_dict())
assert set(ac.row_in_filled)==set(rows)
cl={}
for c in range(4):
    r=np.sort(ac.loc[ac.cluster==c,'row_in_filled'].to_numpy())
    cm=np.asarray(M[r,:],dtype=np.float64).mean(0).reshape(365,48)
    sub=pc[pc.cluster==c].sort_values('date')[ccols].to_numpy(dtype=np.float64)
    cl[c]={"n":int(len(r)),"max":float(np.abs(cm-sub).max()),"mean":float(np.abs(cm-sub).mean())}
res["recovered_clusters_vs_authors"]=cl
print(json.dumps(res,indent=1))
json.dump(res,open(OUT+"t1_aggregate.json","w"),indent=1)
