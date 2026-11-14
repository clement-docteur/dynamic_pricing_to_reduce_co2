# Robustness: control days matched on daily mean temperature (authors' NOAA temps) within +-30 days; far-slot day-level check.
import sys; sys.path.insert(0,"C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-data/")
import vbase as V, numpy as np, pandas as pd, json
S=V.series(); A=V.authors_sched(); raw=V.xlsx_raw(); B=V.blocks(A); normal=(A==0)
cands=[V.xlsx_candidate(raw,m,k) for m in ("gmt","loc") for k in range(-3,4)]
clean=V.control_days([A]+cands,pad=8)
pc=pd.read_feather(V.UP+"power_consump_nmf.feather"); s=pc[pc.cluster==0].sort_values("date")
tday=s[[f"temp_{h:02d}" for h in range(48)]].to_numpy(dtype=float).mean(1)
day=np.repeat(np.arange(365),48); di=np.arange(365)
def cf_temp(y, ref, K=10, win=30):
    Y=y.reshape(365,48); R=ref.reshape(365,48); CF=np.zeros((365,48))
    for d in range(365):
        m=clean&(V.WORK==V.WORK[d])&(V.BSTDAY==V.BSTDAY[d])&(np.abs(di-d)<=win)&(di!=d)
        c=di[m]
        if len(c)<K:  # widen
            m=clean&(V.WORK==V.WORK[d])&(V.BSTDAY==V.BSTDAY[d])&(di!=d); c=di[m]
        o=np.argsort(np.abs(tday[c]-tday[d]),kind="stable")[:K]; cs=c[o]
        CF[d]=R[d]+(Y[cs]-R[cs]).mean(0)
    return CF.ravel()
rng=np.random.default_rng(11); NB=2000
def win_items(kind, edge, taus):
    items=[]
    for b0,b1,k in B:
        if k!=kind: continue
        if edge=="in": js=list(range(b0,b1))
        else:
            js=[]
            for t in taus:
                j=(b0+t) if edge=="start" else (b1-1+t)
                if 0<=j<V.N and not (b0<=j<b1) and normal[j]: js.append(j)
        if js: items.append((day[b0],js))
    return items
# far slots: Normal slots on event days that are >= 16 slots from any block
dist=np.full(V.N,10**6)
for b0,b1,k in B:
    idx=np.arange(V.N); dd=np.where(idx<b0,b0-idx,np.where(idx>=b1,idx-b1+1,0)); dist=np.minimum(dist,dd)
def far_items(kind):
    evdays=sorted(set(day[b0] for b0,b1,k in B if k==kind))
    items=[]
    for d in evdays:
        js=[j for j in range(d*48,(d+1)*48) if dist[j]>=16]
        if js: items.append((d,js))
    return items
def boot(items,E,CF):
    se=np.array([E[js].sum() for _,js in items]); sc=np.array([CF[js].sum() for _,js in items]); days=np.array([d for d,_ in items])
    ud,inv=np.unique(days,return_inverse=True); SE=np.bincount(inv,weights=se); SC=np.bincount(inv,weights=sc)
    idx=rng.integers(0,len(ud),size=(NB,len(ud))); bs=SE[idx].sum(1)/SC[idx].sum(1)
    return 100*SE.sum()/SC.sum(),100*np.percentile(bs,2.5),100*np.percentile(bs,97.5)
regs={"High_pre_1-4":(2,"start",range(-4,0)),"High_in":(2,"in",None),"High_post_1-4":(2,"end",range(1,5)),"High_pre_5-12":(2,"start",range(-12,-4)),
      "Low_pre_1-4":(1,"start",range(-4,0)),"Low_in":(1,"in",None),"Low_post_1-4":(1,"end",range(1,5)),"Low_post_5-12":(1,"end",range(5,13))}
out={}
for lab,refmode in (("temp-matched, Std control","std"),("temp-matched, no Std","none")):
    print("==",lab)
    for nm in ("0","1","2","3","tou","std"):
        if nm=="std" and refmode=="std": continue
        y=S[int(nm)] if nm.isdigit() else S[nm]
        ref=S["std"] if (refmode=="std") else np.zeros(V.N)
        CF=cf_temp(y,ref); E=y-CF
        line={}
        for rn,(kind,edge,taus) in regs.items(): line[rn]=boot(win_items(kind,edge,taus),E,CF)
        line["Low_day_far16"]=boot(far_items(1),E,CF); line["High_day_far16"]=boot(far_items(2),E,CF)
        out[f"{lab}|{nm}"]=line
        print(f"  {nm:4s} "+" | ".join(f"{k}:{v[0]:+.1f}[{v[1]:+.1f},{v[2]:+.1f}]{'*' if (v[1]>0 or v[2]<0) else ''}" for k,v in line.items()))
json.dump(out,open(V.OUT+"t3e_tempmatch.json","w"),indent=1,default=float)
