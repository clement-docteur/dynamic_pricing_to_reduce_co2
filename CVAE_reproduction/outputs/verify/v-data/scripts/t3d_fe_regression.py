# Cross-check: two-way FE regression on (cluster - Std) with date FE + slot x daytype x month FE + slot x DST-regime FE;
# region dummies (out-of-block regions only on Normal slots); cluster-robust SE by date.
import os; os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import sys; sys.path.insert(0,"C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-data/")
import vbase as V, numpy as np, pandas as pd, json
S=V.series(); A=V.authors_sched(); B=V.blocks(A); normal=(A==0)
day=np.repeat(np.arange(365),48); hod=np.tile(np.arange(48),365)
month=V.DATES.month.to_numpy()[day]-1; wk=V.WORK[day].astype(int); reg=V.BSTDAY[day].astype(int)
regions={"High_in":(2,"in",None),"High_pre_1-4":(2,"start",range(-4,0)),"High_pre_5-12":(2,"start",range(-12,-4)),
 "High_post_1-4":(2,"end",range(1,5)),"High_post_5-8":(2,"end",range(5,9)),"Low_in":(1,"in",None),
 "Low_pre_1-4":(1,"start",range(-4,0)),"Low_pre_5-8":(1,"start",range(-8,-4)),"Low_post_1-4":(1,"end",range(1,5)),"Low_post_5-12":(1,"end",range(5,13))}
G=np.zeros((V.N,len(regions)))
for c,(nm,(kind,edge,taus)) in enumerate(regions.items()):
    for b0,b1,k in B:
        if k!=kind: continue
        if edge=="in": G[b0:b1,c]=1; continue
        for t in taus:
            j=(b0+t) if edge=="start" else (b1-1+t)
            if 0<=j<V.N and not (b0<=j<b1) and normal[j]: G[j,c]=1
# FE design (dense)
cols=[np.eye(365)[day]]
k1=hod*24+month*2+wk; F1=np.zeros((V.N,48*24)); F1[np.arange(V.N),k1]=1
k2=hod*2+reg; F2=np.zeros((V.N,96)); F2[np.arange(V.N),k2]=1
F=np.hstack(cols+[F1,F2])
F=F[:,F.sum(0)>0]
print("FE design",F.shape)
Q,R=np.linalg.qr(F)   # reduced QR; rank deficiency is harmless for projection if we drop tiny diag
keep=np.abs(np.diag(R))>1e-8*np.abs(np.diag(R)).max()
Q=Q[:,keep]
def resid(M): return M-Q@(Q.T@M)
Gt=resid(G)
out={}
for nm in ("0","1","2","3","tou"):
    y=S[int(nm)] if nm.isdigit() else S[nm]
    D=y-S["std"]; Dt=resid(D)
    XtX=Gt.T@Gt; g=np.linalg.solve(XtX,Gt.T@Dt); u=Dt-Gt@g
    # cluster-robust by date
    sc=np.zeros((365,G.shape[1]))
    np.add.at(sc,day,Gt*u[:,None])
    meat=sc.T@sc; inv=np.linalg.inv(XtX); Vc=inv@meat@inv*(365/364)
    se=np.sqrt(np.diag(Vc))
    res={}
    for c,rn in enumerate(regions):
        m=G[:,c]>0; base=y[m].mean()-g[c]
        res[rn]=dict(gamma_Wh=1000*g[c],se_Wh=1000*se[c],t=g[c]/se[c],pct=100*g[c]/base,pct_lo=100*(g[c]-1.96*se[c])/base,pct_hi=100*(g[c]+1.96*se[c])/base,n_slots=int(m.sum()))
    out[nm]=res
rows=[]
for rn in regions:
    rows.append([rn]+[f"{out[nm][rn]['pct']:+6.1f} [{out[nm][rn]['pct_lo']:+5.1f},{out[nm][rn]['pct_hi']:+5.1f}] t={out[nm][rn]['t']:+.1f}" for nm in ("0","1","2","3","tou")])
print(pd.DataFrame(rows,columns=["region","c0","c1","c2","c3","all-ToU"]).to_string(index=False))
json.dump(out,open(V.OUT+"t3d_fe_regression.json","w"),indent=1,default=float)
