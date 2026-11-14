import os
os.environ.setdefault("OMP_NUM_THREADS","2")
import pandas as pd, numpy as np, json
REPRO="C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/"
SMALL="C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/Small LCL Data/"
OUT="C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-data/"
sel=pd.read_parquet(REPRO+"data/selection.parquet")
ac=pd.read_parquet(REPRO+"data/authors_clusters_recovered.parquet")
M=np.load(REPRO+"data/lcl2013_kwh_filled.npy", mmap_mode='r')
Mraw=np.load(REPRO+"data/lcl2013_kwh.npy", mmap_mode='r')
print("raw npy", Mraw.shape, Mraw.dtype)
rng=pd.read_csv(OUT+"smallfile_ranges.csv",header=None,names=["i","first","last"])
cand=sel[sel.in_authors_tou_1007].merge(ac[['LCLid','cluster']],on='LCLid')
def files_of(h):
    m=rng[(rng['first']<=h)&(rng['last']>=h)]
    return tuple(int(x) for x in m.i)
cand['files']=cand.LCLid.map(files_of)
cand['nf']=cand.files.map(len)
r=np.random.default_rng(12345)
# one household from cluster 3, one from cluster 1, one with the lowest completeness (has gaps)
picks=[cand[cand.cluster==3].sample(1,random_state=1).iloc[0],
       cand[cand.cluster==1].sample(1,random_state=2).iloc[0],
       cand.sort_values('completeness').iloc[0]]
t0=pd.Timestamp("2013-01-01 00:00")
res={}
first_tod=[]
for p in picks:
    parts=[]
    for fi in p.files:
        dfi=pd.read_csv(SMALL+f"LCL-June2015v2_{fi}.csv",dtype=str); dfi.columns=["LCLid","grp","DateTime","kwh"]; parts.append(dfi)
    df=pd.concat(parts)
    # first-reading time of day for every household in this file
    g=df.groupby("LCLid")["DateTime"].min()
    first_tod+= [s[11:16] for s in g.to_numpy(dtype=object)]
    h=df[df.LCLid==p.LCLid].copy()
    grp=h.grp.unique().tolist()
    h['ts']=pd.to_datetime(h.DateTime.str.slice(0,19))
    h['v']=pd.to_numeric(h.kwh.astype(str).str.strip(),errors='coerce')
    y=h[(h.ts>=t0)&(h.ts<pd.Timestamp("2014-01-01"))].copy()
    off=(y.ts-t0)/pd.Timedelta(minutes=30)
    ongrid=(off==off.round())
    y=y[ongrid.to_numpy()]; j=((y.ts-t0)/pd.Timedelta(minutes=30)).round().astype(int).to_numpy()
    v=y.v.to_numpy(dtype=float)
    # dedupe (identical duplicates)
    dd=pd.DataFrame({'j':j,'v':v}).groupby('j').v.agg(['mean','nunique','size'])
    jj=dd.index.to_numpy(); vv=dd['mean'].to_numpy()
    ok=~np.isnan(vv)
    row=int(p.row_in_filled)
    mf=np.asarray(M[row],dtype=np.float64); mr=np.asarray(Mraw[p.row_in_dense],dtype=np.float64) if Mraw.shape[0]>p.row_in_dense else None
    out={"files_scanned":list(p.files),"group_label_in_csv":grp,"cluster":int(p.cluster),"completeness":float(p.completeness),
         "n_2013_rows":int(len(y)),"n_unique_slots":int(len(jj)),"n_dup_slots":int((dd['size']>1).sum()),
         "dups_conflicting":int((dd['nunique']>1).sum())}
    for s in (-1,0,1):
        jk=jj[ok]-s
        m=(jk>=0)&(jk<17520)
        diff=np.abs(mf[jk[m]]-vv[ok][m])
        out[f"shift{s:+d}_max_abs"]=float(diff.max()); out[f"shift{s:+d}_n_exact(<1e-6)"]=int((diff<1e-6).sum()); out[f"shift{s:+d}_n"]=int(m.sum())
    out["missing_slots_in_raw"]=int(17520-ok.sum())
    # DST days: readings per date
    cnt=y.groupby(y.ts.dt.date).size()
    out["rows_on_2013-03-31"]=int(cnt.get(pd.Timestamp("2013-03-31").date(),0)); out["rows_on_2013-10-27"]=int(cnt.get(pd.Timestamp("2013-10-27").date(),0))
    tod=y[(y.ts.dt.date==pd.Timestamp("2013-03-31").date())].ts.dt.strftime("%H:%M").tolist()
    out["dst_spring_has_01:00_and_01:30"]=("01:00" in tod) and ("01:30" in tod)
    # examples: first 3 readings of 2013 and slot mapping
    ex=y.sort_values('ts').head(3)
    out["examples"]=[(str(a),float(b),float(mf[int(round((a-t0)/pd.Timedelta(minutes=30)))])) for a,b in zip(ex.ts,ex.v)]
    # a mid-year example
    mid=y[y.ts==pd.Timestamp("2013-07-15 18:00")]
    if len(mid): out["example_2013-07-15_18:00(col=%d)"%int((pd.Timestamp('2013-07-15 18:00')-t0)/pd.Timedelta(minutes=30))]=(float(mid.v.iloc[0]),float(mf[int((pd.Timestamp('2013-07-15 18:00')-t0)/pd.Timedelta(minutes=30))]))
    res[p.LCLid]=out
res["first_reading_time_of_day_counts(households in the 3 files)"]=pd.Series(first_tod).value_counts().head(6).to_dict()
print(json.dumps(res,indent=1,default=str))
json.dump(res,open(OUT+"t1b_spotcheck.json","w"),indent=1,default=str)
