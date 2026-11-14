# Why do in-block Low effects differ from phase 2 (+44.9% c3)? Compare baseline designs.
import sys; sys.path.insert(0,"C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-data/")
import vbase as V, numpy as np, pandas as pd
S=V.series(); A=V.authors_sched()
allnormal=~(A.reshape(365,48)!=0).any(1)
B=V.blocks(A); di=np.arange(365)
def cf_window(y, ref, ctrlmask, win=14, regime=False):
    Y=y.reshape(365,48); R=ref.reshape(365,48); CF=np.full((365,48),np.nan)
    for d in range(365):
        m=ctrlmask&(V.WORK==V.WORK[d])&(np.abs(di-d)<=win)&(di!=d)
        if regime: m&=(V.BSTDAY==V.BSTDAY[d])
        if m.sum()==0: continue
        CF[d]=R[d]+(Y[m]-R[m]).mean(0)
    return CF.ravel()
rows=[]
for nm in ("0","1","2","3"):
    y=S[int(nm)]
    for lab,ref,win,reg in (("pm14 allNormal(authors), no Std",np.zeros(V.N),14,False),
                            ("pm14 allNormal(authors), Std control",S["std"],14,False),
                            ("pm14 allNormal, no Std, same DST regime",np.zeros(V.N),14,True)):
        cf=cf_window(y,ref,allnormal,win,reg); e=y-cf
        out={"cluster":nm,"design":lab}
        for kind,kn in ((1,"Low"),(2,"High")):
            js=np.concatenate([np.arange(b0,b1) for b0,b1,k in B if k==kind]); ok=~np.isnan(cf[js])
            out[f"{kn}_in_ratio_of_sums"]=100*e[js][ok].sum()/cf[js][ok].sum()
            per=[100*np.nansum(e[b0:b1])/np.nansum(cf[b0:b1]) for b0,b1,k in B if k==kind and not np.all(np.isnan(cf[b0:b1]))]
            out[f"{kn}_in_mean_of_block_ratios"]=np.mean(per)
            out[f"{kn}_n_slots_missing_cf"]=int((~ok).sum())
        rows.append(out)
t=pd.DataFrame(rows); pd.set_option("display.width",250); print(t.round(2).to_string(index=False))
