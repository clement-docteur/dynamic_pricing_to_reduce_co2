import sys; sys.path.insert(0,"C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-data/")
import vbase as V, numpy as np, pandas as pd, json
S=V.series(); A=V.authors_sched(); raw=V.xlsx_raw()
X=V.xlsx_candidate(raw,"loc",1)
cands=[V.xlsx_candidate(raw,m,k) for m in ("gmt","loc") for k in range(-3,4)]
clean=V.control_days([A]+cands,pad=8)
day=np.repeat(np.arange(365),48); hod=np.tile(np.arange(48),365)
diff=np.where(A!=X)[0]
ddays=sorted(set(day[diff]))
print("differing slots:",len(diff),"differing days:",len(ddays),[str(V.DATES[d].date()) for d in ddays])
def sstr(s,d): return "".join(".LH"[v] for v in s[d*48:(d+1)*48])
out={"differing_days":[]}
for name in ("3","tou"):
    y=S[int(name)] if name.isdigit() else S[name]
    CF,ctrl=V.counterfactual(y,S["std"],clean,K=10)
    E=y-CF; r=E/CF
    # calibration of expected ratio: agreed event slots, by kind x regime x 3h block
    reg=V.BSTDAY[day].astype(int); blk=hod//6
    agree=(A==X)
    mu={}
    for kind in (1,2):
        for g in (0,1):
            for b in range(8):
                m=agree&(A==kind)&(reg==g)&(blk==b)
                mu[(kind,g,b)]=(r[m].mean() if m.sum()>=6 else np.nan, int(m.sum()))
    def expect(state,j):
        if state==0: return 0.0
        v,n=mu[(state,reg[j],blk[j])]
        if np.isnan(v):  # fallback: pooled over blocks of the regime
            m=agree&(A==state)&(reg==reg[j]); v=r[m].mean()
        return v
    cleandays=np.where(clean)[0]
    for d in ddays:
        js=np.arange(d*48,(d+1)*48); js=js[A[js]!=X[js]]
        obs=r[js].mean()
        ex_x=np.mean([expect(X[j],j) for j in js]); ex_a=np.mean([expect(A[j],j) for j in js])
        # placebo noise: same slot-of-day set on clean days of same regime & daytype
        hs=hod[js]
        pl=[r[cd*48+hs].mean() for cd in cleandays if (V.BSTDAY[cd]==V.BSTDAY[d])]
        sd=np.std(pl,ddof=1)
        zx=(obs-ex_x)/sd; za=(obs-ex_a)/sd
        rec=dict(series=name,date=str(V.DATES[d].date()),n_slots=int(len(js)),authors=sstr(A,d),xlsx=sstr(X,d),
                 obs_pct=100*obs,exp_xlsx_pct=100*ex_x,exp_auth_pct=100*ex_a,placebo_sd_pct=100*sd,z_vs_xlsx=zx,z_vs_auth=za,
                 llr_auth_minus_xlsx=0.5*(zx**2-za**2),
                 verdict=("authors" if 0.5*(zx**2-za**2)>2 else "xlsx" if 0.5*(zx**2-za**2)<-2 else "undecided"))
        out["differing_days"].append(rec)
    out[f"calibration_{name}"]={f"{['','Low','High'][k[0]]}|{'BST' if k[1] else 'GMT'}|{k[2]*3:02d}-{k[2]*3+3:02d}h":(None if np.isnan(v[0]) else round(100*v[0],1),v[1]) for k,v in mu.items()}
t=pd.DataFrame(out["differing_days"])
pd.set_option("display.width",250)
for name in ("3","tou"):
    print(f"\n== series {name}: calibration (% response on agreed slots):",out[f"calibration_{name}"])
    print(t[t.series==name][["date","n_slots","obs_pct","exp_xlsx_pct","exp_auth_pct","placebo_sd_pct","z_vs_xlsx","z_vs_auth","llr_auth_minus_xlsx","verdict"]].round(2).to_string(index=False))
print()
for d in ddays: print(str(V.DATES[d].date()),"A:",sstr(A,d)); print(" "*10,"X:",sstr(X,d))
json.dump(out,open(V.OUT+"t2b_content.json","w"),indent=1,default=float)
