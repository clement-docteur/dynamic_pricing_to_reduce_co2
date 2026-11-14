import sys; sys.path.insert(0,"C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-data/")
import vbase as V, numpy as np, pandas as pd, json
S=V.series(); A=V.authors_sched(); raw=V.xlsx_raw()
cands={(m,k):V.xlsx_candidate(raw,m,k) for m in ("gmt","loc") for k in range(-3,4)}
clean=V.control_days([A]+list(cands.values()),pad=8)
print("clean control days:",clean.sum(), "work",(clean&V.WORK).sum(),"BST",(clean&V.BSTDAY).sum())
day=np.repeat(np.arange(365),48)
season={"GMT":~V.BSTDAY[day],"BST":V.BSTDAY[day]}
res={"n_clean_days":int(clean.sum())}
for name in ("3","tou"):
    y=S[int(name)] if name.isdigit() else S[name]
    CF,_=V.counterfactual(y,S["std"],clean,K=10)
    E=y-CF
    tab=[]
    for (m,k),s in list(cands.items())+[(("auth",kk),np.roll(A,kk)) for kk in range(-3,4)]:
        for sn,smask in season.items():
            # edges
            sj=[];
            for (b0,b1,kind) in V.blocks(s):
                for j,sgn in ((b0, +1 if kind==1 else -1),(b1, -1 if kind==1 else +1)):
                    if 1<=j<V.N and smask[j]: sj.append(sgn*(E[j]-E[j-1]))
            # R2
            X=np.stack([(s==1),(s==2)],1).astype(float)[smask]; e=E[smask]
            beta,*_=np.linalg.lstsq(X,e,rcond=None); r2=1-((e-X@beta)**2).sum()/((e-e.mean())**2).sum()
            tab.append(dict(series=name,mode=m,k=k,season=sn,n_edges=len(sj),mean_signed_jump_Wh=1000*np.mean(sj),R2=r2,beta_low_Wh=1000*beta[0],beta_high_Wh=1000*beta[1]))
    t=pd.DataFrame(tab)
    for sn in ("GMT","BST"):
        for m in ("gmt","loc","auth"):
            sub=t[(t.season==sn)&(t["mode"]==m)].sort_values("k")
            print(f"[{name}] {sn:3s} {m:4s} jump(Wh) by k=-3..3:", " ".join(f"{x:6.1f}" for x in sub.mean_signed_jump_Wh),
                  "| R2:", " ".join(f"{x:.3f}" for x in sub.R2), "| best k jump/R2:", int(sub.k.iloc[sub.mean_signed_jump_Wh.argmax()]), int(sub.k.iloc[sub.R2.argmax()]))
    res[f"table_{name}"]=t.to_dict(orient="records")
    # event-time profile around edges under loc+1 (the candidate expected right) and gmt+1
    prof={}
    for lab,s in (("loc+1",cands[("loc",1)]),("gmt+1",cands[("gmt",1)]),("gmt-1",cands[("gmt",-1)])):
        for sn,smask in season.items():
            for kind,kn in ((1,"Low"),(2,"High")):
                for edge in ("start","end"):
                    rows=[]
                    for (b0,b1,kk) in V.blocks(s):
                        if kk!=kind: continue
                        j0=b0 if edge=="start" else b1
                        if not (4<=j0<V.N-4) or not smask[j0]: continue
                        rows.append(1000*E[j0-4:j0+4])
                    if rows: prof[f"{lab}|{sn}|{kn}-{edge}"]=(len(rows),np.round(np.mean(rows,0),1).tolist())
    res[f"profiles_{name}"]=prof
    for k_,v in prof.items():
        if k_.startswith("loc+1") or name=="3": print(f"  [{name}] {k_:22s} n={v[0]:3d} tau=-4..+3 (0=first slot of new state):",v[1])
json.dump(res,open(V.OUT+"t2_alignment.json","w"),indent=1,default=float)
