import sys, json
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from pdfvec import page_objects
OUT = r"C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/outputs/verify/v-paper/"
doc, page, objs = page_objects(10)
paths = [o for o in objs if o['kind']=='path' and o['pts']]
def bb(o):
    xs=[p[1] for p in o['pts']]; ys=[p[2] for p in o['pts']]
    return min(xs),max(xs),min(ys),max(ys)
f6 = [o for o in paths if bb(o)[2]>=365 and bb(o)[3]<=474]
cal = {'CVAE': dict(xt=[69.023,114.293,159.563,204.833,250.103], yt=[394.793,412.391,429.99,447.588,465.188]),
       'GAM':  dict(xt=[300.019,344.858,389.697,434.533,479.372], yt=[394.204,411.482,428.761,446.04,463.321])}
names = {(0,0,0):'High', (211,211,211):'Low', (133,139,151):'Normal'}
res = {}
for gen, c in cal.items():
    ax, bx = np.polyfit(c['xt'], [0,6,12,18,24], 1)       # hours since Jan 01 00:00
    ay, by = np.polyfit(c['yt'], [0.1,0.2,0.3,0.4,0.5], 1)
    xl, xr = c['xt'][0]-10, c['xt'][-1]+10
    series = {}
    for o in f6:
        if len(o['pts']) != 48 or not o['dostroke']: continue
        x0,x1,_,_ = bb(o)
        if not (xl < x0 and x1 < xr): continue
        nm = names[o['stroke'][:3]]
        hrs = np.array([ax*p[1]+bx for p in o['pts']]); kwh = np.array([ay*p[2]+by for p in o['pts']])
        series.setdefault(nm, []).append((hrs, kwh))
    out = {}
    for nm, lst in series.items():
        H = np.array([l[0] for l in lst]); K = np.array([l[1] for l in lst])
        out[nm] = dict(n=len(lst), hours=H.mean(0), kwh=K)
    res[gen] = out
    h = out['Normal']['hours']
    print(f"== {gen}: n lines", {k: v['n'] for k, v in out.items()}, "x positions (h): first", np.round(h[:3],3), "last", np.round(h[-2:],3))
    mN = out['Normal']['kwh'].mean(0); mH = out['High']['kwh'].mean(0); mL = out['Low']['kwh'].mean(0)
    sdN = out['Normal']['kwh'].std(0, ddof=1)
    print(" hour  Normal  High-N   Low-N   sdNormal")
    for i in range(48):
        print(f" {h[i]:5.2f} {mN[i]:.4f} {mH[i]-mN[i]:+.4f} {mL[i]-mN[i]:+.4f}  {sdN[i]:.4f}")
json.dump({g: {nm: dict(n=v['n'], hours=v['hours'].tolist(), kwh=v['kwh'].tolist()) for nm, v in o.items()} for g, o in res.items()},
          open(OUT + "fig6_series_from_pdf_vectors.json", "w"))
