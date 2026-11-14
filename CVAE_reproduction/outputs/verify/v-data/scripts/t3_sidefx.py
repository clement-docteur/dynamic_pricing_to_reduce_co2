# Side/rebound effects: DiD (cluster aggregate minus Std aggregate; event days vs K nearest clean days of same daytype & DST regime)
import sys; sys.path.insert(0,"C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-data/")
import vbase as V, numpy as np, pandas as pd, json
K=int(sys.argv[1]) if len(sys.argv)>1 else 10
TAG=sys.argv[2] if len(sys.argv)>2 else "main"
REF=sys.argv[3] if len(sys.argv)>3 else "std"   # 'std' = DiD with Std control; 'none' = matched days only
S=V.series(); A=V.authors_sched(); raw=V.xlsx_raw()
cands=[V.xlsx_candidate(raw,m,k) for m in ("gmt","loc") for k in range(-3,4)]
clean=V.control_days([A]+cands,pad=8)
day=np.repeat(np.arange(365),48); hod=np.tile(np.arange(48),365)
NAMES=["0","1","2","3","tou"]
E={};CF={}
for nm in NAMES:
    y=S[int(nm)] if nm.isdigit() else S[nm]
    ref=S["std"] if REF=="std" else np.zeros(V.N)
    cf,_=V.counterfactual(y,ref,clean,K=K); CF[nm]=cf; E[nm]=y-cf
# Std placebo: Std aggregate on event days vs its own matched clean days (should be ~0 if event days are not special)
cfs,_=V.counterfactual(S["std"],np.zeros(V.N),clean,K=K); CF["std"]=cfs; E["std"]=S["std"]-cfs
NAMES=NAMES+["std"]
B=V.blocks(A)
nH=sum(1 for b in B if b[2]==2); nL=sum(1 for b in B if b[2]==1)
print(f"K={K}; clean days {clean.sum()}; authors' blocks: High {nH}, Low {nL}")
normal=(A==0)
def window_slots(b, edge, taus, need_normal=True):
    b0,b1,k=b
    out=[]
    for t in taus:
        j=(b0+t) if edge=="start" else (b1-1+t)   # end: t=0 last in-block slot, t=1 first after
        if 0<=j<V.N:
            inb=(b0<=j<b1)
            if inb or (normal[j] or not need_normal): out.append((t,j,inb))
    return out
rng=np.random.default_rng(20261112)
NB=2000
def boot_ratio(se,scf,days):
    """se, scf: per-block sums; days: event-day id per block. bootstrap over event days"""
    ud,inv=np.unique(days,return_inverse=True)
    SE=np.bincount(inv,weights=se,minlength=len(ud)); SC=np.bincount(inv,weights=scf,minlength=len(ud))
    est=SE.sum()/SC.sum()
    idx=rng.integers(0,len(ud),size=(NB,len(ud)))
    bs=SE[idx].sum(1)/SC[idx].sum(1)
    return 100*est, 100*np.percentile(bs,2.5), 100*np.percentile(bs,97.5), len(ud)
# placebo: move each block's window to a random clean day of same daytype & regime (same slots of day)
cleandays=np.where(clean)[0]
def placebo(nm, items, nrep=1000):
    """items: list of (block_start_day, [slot j]) ; returns placebo pct distribution"""
    e=E[nm]; cf=CF[nm]; vals=[]
    pools={}
    for d0,_ in items:
        key=(V.WORK[d0],V.BSTDAY[d0])
        if key not in pools: pools[key]=cleandays[(V.WORK[cleandays]==key[0])&(V.BSTDAY[cleandays]==key[1])]
    for r_ in range(nrep):
        se=0.0; sc=0.0
        for d0,js in items:
            pool=pools[(V.WORK[d0],V.BSTDAY[d0])]; d1=pool[rng.integers(len(pool))]
            jj=np.asarray(js)+(d1-d0)*48; jj=jj[(jj>=0)&(jj<V.N)]; jj=jj[normal[jj]]
            se+=e[jj].sum(); sc+=cf[jj].sum()
        vals.append(100*se/sc)
    return np.array(vals)
regions={ # name: (kind, edge, taus, in_block_only)
 "High_pre_1-4":(2,"start",range(-4,0)),
 "High_pre_5-8":(2,"start",range(-8,-4)),
 "High_pre_5-12":(2,"start",range(-12,-4)),
 "High_in":(2,"in",None),
 "High_post_1-4":(2,"end",range(1,5)),
 "High_post_5-8":(2,"end",range(5,9)),
 "Low_pre_1-4":(1,"start",range(-4,0)),
 "Low_in":(1,"in",None),
 "Low_post_1-4":(1,"end",range(1,5)),
 "Low_post_5-12":(1,"end",range(5,13)),
}
def collect(kind, edge, taus, clean_only=False, subset=None):
    items=[]
    for b in B:
        if b[2]!=kind: continue
        if subset is not None and not subset(b): continue
        if edge=="in":
            js=list(range(b[0],b[1]))
        else:
            ws=window_slots(b,edge,taus)
            js=[j for (t,j,inb) in ws if not inb]
            if clean_only and len(js)<len(list(taus)): continue
        if js: items.append((day[b[0]],js))
    return items
def est(nm, items):
    se=np.array([E[nm][js].sum() for _,js in items]); sc=np.array([CF[nm][js].sum() for _,js in items])
    days=np.array([d for d,_ in items])
    p,lo,hi,nd=boot_ratio(se,sc,days)
    wh=1000*se.sum()/sum(len(js) for _,js in items)
    return dict(pct=p,lo=lo,hi=hi,n_blocks=len(items),n_days=nd,n_slots=int(sum(len(js) for _,js in items)),Wh_per_slot=wh)
res={"K":K,"n_clean_days":int(clean.sum()),"n_high_blocks":nH,"n_low_blocks":nL,"regions":{},"regions_clean_edges":{},"profiles":{}}
for rname,(kind,edge,taus) in regions.items():
    for co in (False,True):
        if co and edge=="in": continue
        items=collect(kind,edge,taus,clean_only=co)
        for nm in NAMES:
            r=est(nm,items)
            if rname in ("High_pre_1-4","High_post_1-4","Low_post_5-12","High_pre_5-12") and not co and TAG=="main":
                pl=placebo(nm,items,nrep=600); r["placebo_sd"]=float(pl.std()); r["placebo_p_two_sided"]=float((np.abs(pl)>=abs(r["pct"])).mean())
            (res["regions_clean_edges"] if co else res["regions"]).setdefault(rname,{})[nm]=r
# per-lag profiles
for kind,kn,edges in (((2,"High",{"start":range(-12,9),"end":range(-8,13)}),(1,"Low",{"start":range(-12,13),"end":range(-4,13)})) if TAG=="main" else ()):
    for edge,taus in edges.items():
        for nm in NAMES:
            prof=[]
            for t in taus:
                items=[]
                for b in B:
                    if b[2]!=kind: continue
                    ws=window_slots(b,edge,[t])
                    if ws: items.append((day[b[0]],[ws[0][1]]))
                if items:
                    r=est(nm,items); prof.append((t,round(r["pct"],2),round(r["lo"],2),round(r["hi"],2),r["n_blocks"]))
            res["profiles"][f"{kn}_{edge}|{nm}"]=prof
json.dump(res,open(V.OUT+f"t3_sidefx_{TAG}.json","w"),indent=1,default=float)
# print summary
def fmt(r): return f"{r['pct']:+6.1f} [{r['lo']:+5.1f},{r['hi']:+5.1f}]" + (f" p={r['placebo_p_two_sided']:.2f}" if 'placebo_p_two_sided' in r else "")
for which in ("regions","regions_clean_edges"):
    print("\n==",which,"(% of counterfactual; 95% bootstrap CI over event days) clusters 0/1/2/3/all-ToU")
    for rname,d in res[which].items():
        print(f"{rname:15s} n_blk={d['0']['n_blocks']:3d} | "+" | ".join(fmt(d[nm]) for nm in NAMES))
print("\n== per-lag profiles (pct, CI) ==")
for k,v in res["profiles"].items():
    print(k, " ".join(f"{t:+d}:{p:+.1f}{'*' if (lo>0 or hi<0) else ''}" for t,p,lo,hi,n in v))
