# v-data base: independent series + schedules (no import from REPRO/lib or upstream)
import os
os.environ.setdefault("OMP_NUM_THREADS","2"); os.environ.setdefault("MKL_NUM_THREADS","2")
import numpy as np, pandas as pd
REPRO="C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/"
UP="C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo/data/"
XLSX="C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/Tariffs.xlsx"
OUT="C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-data/"
KEEP="C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/outputs/verify/v-data/"
N=17520; T0=pd.Timestamp("2013-01-01 00:00")
DATES=pd.date_range("2013-01-01",periods=365,freq="D")
HOL=pd.to_datetime(["2013-01-01","2013-03-29","2013-04-01","2013-05-06","2013-05-27","2013-08-26","2013-12-25","2013-12-26"])
WORK=np.array([(d.weekday()<5) and (d not in set(HOL)) for d in DATES])
BSTDAY=np.array([(d>=pd.Timestamp("2013-03-31")) and (d<=pd.Timestamp("2013-10-26")) for d in DATES])
# GMT instant of each slot label j (authors' frame: slot j = LCL reading labelled T0 + j*30min GMT)
GMT=T0+pd.to_timedelta(np.arange(N)*30,unit="min")
BST_START=pd.Timestamp("2013-03-31 01:00"); BST_END=pd.Timestamp("2013-10-27 01:00")

def series():
    """returns dict of 17520 series: 'std','tou', 0..3 cluster means (kWh/household/half-hour)"""
    p=OUT+"series.npz"
    if os.path.exists(p):
        z=np.load(p); return {k:(int(k) if k.isdigit() else k) and z[k] for k in z.files} if False else {(int(k) if k.isdigit() else k):z[k] for k in z.files}
    sel=pd.read_parquet(REPRO+"data/selection.parquet")
    ac=pd.read_parquet(REPRO+"data/authors_clusters_recovered.parquet")
    M=np.load(REPRO+"data/lcl2013_kwh_filled.npy",mmap_mode="r")
    out={}
    out["std"]=np.asarray(M[np.sort(sel.loc[sel.group=="Std","row_in_filled"].to_numpy())],dtype=np.float64).mean(0)
    out["tou"]=np.asarray(M[np.sort(sel.loc[sel.in_authors_tou_1007,"row_in_filled"].to_numpy())],dtype=np.float64).mean(0)
    for c in range(4):
        out[str(c)]=np.asarray(M[np.sort(ac.loc[ac.cluster==c,"row_in_filled"].to_numpy())],dtype=np.float64).mean(0)
    np.savez(p,**out)
    return {(int(k) if k.isdigit() else k):v for k,v in out.items()}

def authors_sched():
    """authors' dummies in their frame: int array 17520: 0 Normal, 1 Low, 2 High"""
    pc=pd.read_feather(UP+"power_consump_nmf.feather")
    s=pc[pc.cluster==0].sort_values("date")
    L=s[[f"low_{h:02d}" for h in range(48)]].to_numpy(dtype=float).ravel()
    H=s[[f"high_{h:02d}" for h in range(48)]].to_numpy(dtype=float).ravel()
    assert not np.any((L>0)&(H>0))
    # check identical across clusters
    for c in range(1,4):
        s2=pc[pc.cluster==c].sort_values("date")
        assert np.array_equal(s2[[f"low_{h:02d}" for h in range(48)]].to_numpy(dtype=float).ravel(),L)
    return (L>0).astype(int)+2*(H>0).astype(int)

def xlsx_raw():
    T=pd.read_excel(XLSX)
    ts=pd.to_datetime(T.TariffDateTime); assert len(T)==N and (ts.diff().dropna()==pd.Timedelta(minutes=30)).all() and ts.iloc[0]==T0
    k=T.Tariff.to_numpy(dtype=object)
    return np.array([{"Normal":0,"Low":1,"High":2}[x] for x in k])

def xlsx_candidate(raw, mode, k):
    """price state at authors' slot j. mode 'gmt': xlsx row label read as GMT; 'loc': London clock time.
    k = offset in slots: state(j) = state of xlsx instant GMT(j) - k*30min."""
    g=GMT-pd.Timedelta(minutes=30*k)
    if mode=="loc":
        isb=(g>=BST_START)&(g<BST_END)
        loc=g+pd.to_timedelta(np.where(isb,60,0),unit="min")
    else:
        loc=g
    idx=((loc-T0)/pd.Timedelta(minutes=30)).to_numpy().astype(int)
    out=np.zeros(N,dtype=int); ok=(idx>=0)&(idx<N)
    out[ok]=raw[idx[ok]]
    return out

def blocks(s):
    """runs of non-Normal states in a continuous schedule: list of (start, end_exclusive, kind)"""
    b=[]; j=0
    while j<N:
        if s[j]!=0:
            k=s[j]; e=j
            while e<N and s[e]==k: e+=1
            b.append((j,e,k)); j=e
        else: j+=1
    return b

def control_days(scheds, pad=8):
    """clean days: Normal in every schedule in `scheds` over the day and +-pad slots around it"""
    bad=np.zeros(N,bool)
    for s in scheds: bad|=(s!=0)
    # dilate
    idx=np.where(bad)[0]
    dil=np.zeros(N,bool)
    for o in range(-pad,pad+1):
        ii=idx+o; ii=ii[(ii>=0)&(ii<N)]; dil[ii]=True
    return ~dil.reshape(365,48).any(1)

def counterfactual(y, ref, clean, K=10, same_regime=True):
    """CF(d,h)= ref(d,h) + mean over K nearest clean days d' (same daytype [and regime], d'!=d) of y(d',h)-ref(d',h).
    returns CF (17520,), and the list of control days per day"""
    Y=y.reshape(365,48); R=ref.reshape(365,48); D=Y-R
    CF=np.zeros((365,48)); ctrl=[]
    di=np.arange(365)
    for d in range(365):
        m=clean&(WORK==WORK[d])&(di!=d)
        if same_regime: m&=(BSTDAY==BSTDAY[d])
        cand=di[m]
        o=np.argsort(np.abs(cand-d),kind="stable")[:K]
        cs=cand[o]; ctrl.append(cs)
        CF[d]=R[d]+D[cs].mean(0)
    return CF.ravel(), ctrl
