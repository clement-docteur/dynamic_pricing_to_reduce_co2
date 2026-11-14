# Refinements: isolation masks, subgroups, scenario windows. DiD with Std control, K=10.
import sys; sys.path.insert(0,"C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-data/")
import vbase as V, numpy as np, pandas as pd, json
K=10
S=V.series(); A=V.authors_sched(); raw=V.xlsx_raw()
cands=[V.xlsx_candidate(raw,m,k) for m in ("gmt","loc") for k in range(-3,4)]
clean=V.control_days([A]+cands,pad=8)
day=np.repeat(np.arange(365),48); hod=np.tile(np.arange(48),365)
NAMES=["0","1","2","3","tou"]
E={};CF={}
for nm in NAMES:
    y=S[int(nm)] if nm.isdigit() else S[nm]
    cf,_=V.counterfactual(y,S["std"],clean,K=K); CF[nm]=cf; E[nm]=y-cf
B=V.blocks(A); normal=(A==0)
# isolation: distance (in slots) of each slot to the nearest block start (looking forward) and block end (looking backward), per block
starts=np.array([b[0] for b in B]); ends=np.array([b[1] for b in B])  # end exclusive
rng=np.random.default_rng(7); NB=2000
def boot(items,nm):
    if not items: return None
    se=np.array([E[nm][js].sum() for _,js in items]); sc=np.array([CF[nm][js].sum() for _,js in items]); days=np.array([d for d,_ in items])
    ud,inv=np.unique(days,return_inverse=True)
    SE=np.bincount(inv,weights=se); SC=np.bincount(inv,weights=sc)
    idx=rng.integers(0,len(ud),size=(NB,len(ud))); bs=SE[idx].sum(1)/SC[idx].sum(1)
    return dict(pct=100*SE.sum()/SC.sum(),lo=100*np.percentile(bs,2.5),hi=100*np.percentile(bs,97.5),n_blocks=len(items),n_days=len(ud))
def window(b, edge, taus, isolate=0):
    b0,b1,k=b; js=[]
    for t in taus:
        j=(b0+t) if edge=="start" else (b1-1+t)
        if not (0<=j<V.N) or (b0<=j<b1) or not normal[j]: continue
        if isolate:
            # exclude if within `isolate` slots BEFORE another block's start or AFTER another block's end
            other_s=starts[(starts!=b0)]; other_e=ends[(ends!=b1)]
            if np.any((other_s-j>0)&(other_s-j<=isolate)) or np.any((j-other_e>=0)&(j-other_e<isolate)): continue
        js.append(j)
    return js
def loc_hour(j):  # local clock hour of the START of the half-hour that slot j measures (slot j = half-hour ending at label j)
    g=V.GMT[j]-pd.Timedelta(minutes=30)
    isb=(g>=V.BST_START)&(g<V.BST_END)
    return (g+pd.Timedelta(hours=1 if isb else 0)).hour
res={}
def run(label, kind, edge, taus, isolate=0, subset=None):
    items=[]
    for b in B:
        if b[2]!=kind or (subset is not None and not subset(b)): continue
        js=list(range(b[0],b[1])) if edge=="in" else window(b,edge,taus,isolate)
        if js: items.append((day[b[0]],js))
    r={nm:boot(items,nm) for nm in NAMES}
    res[label]=r
    f=lambda x: "      n/a        " if x is None else f"{x['pct']:+6.1f} [{x['lo']:+5.1f},{x['hi']:+5.1f}]{'*' if (x['lo']>0 or x['hi']<0) else ' '}"
    print(f"{label:44s} n={r['0']['n_blocks'] if r['0'] else 0:3d} | "+" | ".join(f(r[nm]) for nm in NAMES))
print("% of counterfactual, 95% bootstrap CI over event days; * = CI excludes 0.  clusters 0 | 1 | 2 | 3 | all-ToU")
print("-- isolation (other blocks >= 8 slots away) --")
run("High_pre_1-4 isolated8",2,"start",range(-4,0),8)
run("High_pre_5-12 isolated8",2,"start",range(-12,-4),8)
run("High_post_1-4 isolated8",2,"end",range(1,5),8)
run("Low_pre_1-4 isolated8",1,"start",range(-4,0),8)
run("Low_pre_5-8 isolated8",1,"start",range(-8,-4),8)
run("Low_post_1-4 isolated8",1,"end",range(1,5),8)
run("Low_post_5-12 isolated8",1,"end",range(5,13),8)
print("-- subgroups: High pre 1-4 (not isolated) --")
st_h=lambda b: loc_hour(b[0])
run("High_pre_1-4 | GMT regime",2,"start",range(-4,0),0,lambda b: not V.BSTDAY[day[b[0]]])
run("High_pre_1-4 | BST regime",2,"start",range(-4,0),0,lambda b: V.BSTDAY[day[b[0]]])
run("High_pre_1-4 | workday",2,"start",range(-4,0),0,lambda b: V.WORK[day[b[0]]])
run("High_pre_1-4 | weekend/holiday",2,"start",range(-4,0),0,lambda b: not V.WORK[day[b[0]]])
run("High_pre_1-4 | evening start (16-19h local)",2,"start",range(-4,0),0,lambda b: 16<=st_h(b)<=19)
run("High_pre_1-4 | other starts",2,"start",range(-4,0),0,lambda b: not (16<=st_h(b)<=19))
run("High_pre_1-1 (tau=-1 only)",2,"start",range(-1,0),0)
run("High_pre_1-2",2,"start",range(-2,0),0)
run("High_post_1-1 (u=+1 only)",2,"end",range(1,2),0)
run("High_post_1-4 | evening end (21-24h local)",2,"end",range(1,5),0,lambda b: loc_hour(b[1]-1)>=21 or loc_hour(b[1]-1)<1)
print("-- subgroups: Low post 5-12 (rebound) --")
run("Low_post_5-12 | morning Low (start 3-8h local)",1,"end",range(5,13),0,lambda b: 3<=loc_hour(b[0])<=8)
run("Low_post_5-12 | night Low (start 21-2h local)",1,"end",range(5,13),0,lambda b: loc_hour(b[0])>=21 or loc_hour(b[0])<=2)
run("Low_post_5-12 | other Low",1,"end",range(5,13),0,lambda b: not (3<=loc_hour(b[0])<=8 or loc_hour(b[0])>=21 or loc_hour(b[0])<=2))
run("Low_post_5-12 | GMT regime",1,"end",range(5,13),0,lambda b: not V.BSTDAY[day[b[0]]])
run("Low_post_5-12 | BST regime",1,"end",range(5,13),0,lambda b: V.BSTDAY[day[b[0]]])
run("Low_post_13-24",1,"end",range(13,25),0)
print("-- subgroups: Low pre 1-4 --")
run("Low_pre_1-4 | night Low (start 21-2h local)",1,"start",range(-4,0),0,lambda b: loc_hour(b[0])>=21 or loc_hour(b[0])<=2)
run("Low_pre_1-4 | morning Low (3-8h)",1,"start",range(-4,0),0,lambda b: 3<=loc_hour(b[0])<=8)
run("Low_pre_1-4 | other Low",1,"start",range(-4,0),0,lambda b: not (3<=loc_hour(b[0])<=8 or loc_hour(b[0])>=21 or loc_hour(b[0])<=2))
# Paper scenario windows on real event days: Low slots 9-18 fully Low; High 39-43 fully High
def scen(kind, lo, hi):
    items=[]
    for d in range(365):
        js=np.arange(d*48+lo,d*48+hi+1)
        if np.all(A[js]==kind): items.append((d,list(js)))
    return items
for lab,(kind,lo,hi) in {"Scenario Low slots 9-18 (days fully Low there)":(1,9,18),"Scenario High slots 39-43 (days fully High there)":(2,39,43)}.items():
    items=scen(kind,lo,hi); r={nm:boot(items,nm) for nm in NAMES}; res[lab]=r
    print(f"{lab:44s} n={len(items):3d} | "+" | ".join(f"{r[nm]['pct']:+6.1f} [{r[nm]['lo']:+5.1f},{r[nm]['hi']:+5.1f}]" for nm in NAMES))
# start-hour distribution of blocks
bh=pd.Series([loc_hour(b[0]) for b in B if b[2]==2]).value_counts().sort_index().to_dict()
bl=pd.Series([loc_hour(b[0]) for b in B if b[2]==1]).value_counts().sort_index().to_dict()
print("High block start local hour:",bh); print("Low block start local hour:",bl)
json.dump(res,open(V.OUT+"t3b_refine.json","w"),indent=1,default=float)
