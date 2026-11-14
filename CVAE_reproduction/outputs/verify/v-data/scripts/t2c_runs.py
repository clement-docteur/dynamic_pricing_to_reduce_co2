import sys; sys.path.insert(0,"C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-data/")
import vbase as V, numpy as np, pandas as pd, json
S=V.series(); A=V.authors_sched(); raw=V.xlsx_raw()
X=V.xlsx_candidate(raw,"loc",1)
cands=[V.xlsx_candidate(raw,m,k) for m in ("gmt","loc") for k in range(-3,4)]
clean=V.control_days([A]+cands,pad=8)
day=np.repeat(np.arange(365),48); hod=np.tile(np.arange(48),365); reg=V.BSTDAY[day].astype(int)
agree=(A==X)
rows=[]
for name in ("3","tou"):
    y=S[int(name)] if name.isdigit() else S[name]
    CF,_=V.counterfactual(y,S["std"],clean,K=10); r=(y-CF)/CF
    # slot-level calibration: agreed slots of same kind, same regime, slot-of-day within +-3
    def expect(state,j):
        if state==0: return 0.0
        m=agree&(A==state)&(reg==reg[j])&(np.abs(hod-hod[j])<=3)
        return r[m].mean()
    cd=np.where(clean)[0]
    # runs of disagreement with constant (A,X) pair, plus agreed event runs on the same days
    ddays=sorted(set(day[A!=X]))
    for d in ddays:
        js=np.arange(d*48,(d+1)*48)
        lab=[(int(A[j]),int(X[j])) for j in js]
        i=0
        while i<48:
            k=lab[i]; e=i
            while e<48 and lab[e]==k: e+=1
            if k!=(0,0):
                sl=js[i:e]; obs=r[sl].mean(); ea=np.mean([expect(k[0],j) for j in sl]); ex=np.mean([expect(k[1],j) for j in sl])
                pl=[r[c*48+hod[sl]].mean() for c in cd if V.BSTDAY[c]==V.BSTDAY[d] and V.WORK[c]==V.WORK[d]]
                sd=np.std(pl,ddof=1); za=(obs-ea)/sd; zx=(obs-ex)/sd
                rows.append(dict(series=name,date=str(V.DATES[d].date()),slots=f"{i}-{e-1}",auth=".LH"[k[0]],xlsx=".LH"[k[1]],
                    obs=100*obs,exp_auth=100*ea,exp_xlsx=100*ex,sd=100*sd,z_obs_vs_0=obs/sd,llr_auth_minus_xlsx=0.5*(zx**2-za**2),
                    agreed=(k[0]==k[1])))
            i=e
t=pd.DataFrame(rows)
t["verdict"]=np.where(t.agreed,"(agreed: delivered?)",np.where(t.llr_auth_minus_xlsx>2,"authors",np.where(t.llr_auth_minus_xlsx<-2,"xlsx","undecided")))
pd.set_option("display.width",250)
print(t.round(1).to_string(index=False))
t.to_csv(V.OUT+"t2c_runs.csv",index=False)
