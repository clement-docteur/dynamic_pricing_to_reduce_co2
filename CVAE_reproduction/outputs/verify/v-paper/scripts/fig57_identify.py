import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from pdfvec import page_objects
OUT = r"C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/outputs/verify/v-paper/"
UP = r"C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo/data/"
def bb(o):
    xs=[p[1] for p in o['pts']]; ys=[p[2] for p in o['pts']]
    return min(xs),max(xs),min(ys),max(ys)
def panels(pageno, ylo, yhi):
    doc, page, objs = page_objects(pageno)
    paths = [o for o in objs if o['kind']=='path' and o['pts'] and bb(o)[2] >= ylo and bb(o)[3] <= yhi]
    two = [o for o in paths if len(o['pts']) == 2 and o['dostroke']]
    yaxes = [bb(o) for o in two if abs(bb(o)[0]-bb(o)[1]) < 1e-3 and bb(o)[3]-bb(o)[2] > 50]
    out = []
    for (xa, _, y0, y1) in yaxes:
        yt = sorted(bb(o)[2] for o in two if abs(bb(o)[2]-bb(o)[3]) < 1e-3 and xa-2.5 < bb(o)[0] < xa and abs(bb(o)[1]-xa) < 0.01 and y0 <= bb(o)[2] <= y1)
        xt = sorted(bb(o)[0] for o in two if abs(bb(o)[0]-bb(o)[1]) < 1e-3 and abs(bb(o)[3]-y0) < 0.01 and 1 < bb(o)[3]-bb(o)[2] < 2.5 and xa < bb(o)[0] < xa+200)
        ay, by = np.polyfit(yt, [0.1,0.2,0.3,0.4,0.5][:len(yt)], 1)
        ax, bx = np.polyfit(xt, [0,48,96,144][:len(xt)], 1)
        ser = {}
        for o in paths:
            if len(o['pts']) != 144 or not o['dostroke']: continue
            x0,x1,yy0,yy1 = bb(o)
            if not (xa < x0 < xa+30 and y0-1 < yy0 and yy1 < y1+1): continue
            col = o['stroke'][:3]; key = 'obs' if col == (0,0,0) else ('mean' if col == (169,169,169) else str(col))
            ser.setdefault(key, []).append(np.array([[ax*p[1]+bx, ay*p[2]+by] for p in o['pts']]))
        out.append(dict(xa=xa, y0=y0, nyt=len(yt), nxt=len(xt), ser=ser))
    return out
te = pd.read_feather(UP + "test_nmf.feather")
cc = [f"consumption_{h:02d}" for h in range(48)]
te['date'] = pd.to_datetime(te['date'])
d3 = te[te['date'].isin(pd.to_datetime(['2013-01-01','2013-01-02','2013-01-03']))].sort_values(['cluster','date'])
obs = {int(c): g[cc].to_numpy(float).ravel() for c, g in d3.groupby('cluster')}
print("authors' test dates first 3:", sorted(te['date'].unique())[:3])
low = [f"low_{h:02d}" for h in range(48)]; high = [f"high_{h:02d}" for h in range(48)]
print("tariff dummies on Jan 1-3 (sum low, sum high):", d3.groupby('date')[low].sum().sum(axis=1).tolist(), d3.groupby('date')[high].sum().sum(axis=1).tolist())
result = {}
for name, (pno, ylo, yhi) in {"fig5": (10, 509, 622), "fig7": (11, 274, 718)}.items():
    P = panels(pno, ylo, yhi)
    P = sorted(P, key=lambda p: (-p['y0'], p['xa']))
    for k, p in enumerate(P):
        row = k // 2 + 1; gen = 'CVAE' if k % 2 == 0 else 'GAM'
        o = p['ser']['obs'][0]; m = p['ser']['mean'][0]
        assert np.abs(o[:,0] - np.arange(144)).max() < 0.01
        dev = {c: float(np.abs(o[:,1] - obs[c]).max()) for c in obs}
        best = min(dev, key=dev.get)
        rm = [float(np.sqrt(np.mean((m[48*d:48*d+48,1]-o[48*d:48*d+48,1])**2))) for d in range(3)]
        print(f"{name} row{row} {gen}: n_obs_lines {len(p['ser']['obs'])} n_mean {len(p['ser']['mean'])}; max|obs-data| by cluster", {c: round(v,4) for c,v in dev.items()}, "-> cluster", best, "; daily RMSE(mean vs obs, from fig)", np.round(rm,4))
        result[f"{name}_row{row}_{gen}"] = dict(cluster=best, maxdev=dev, rmse_days=rm, mean=m[:,1].tolist(), obs=o[:,1].tolist())
json.dump(result, open(OUT + "fig5_fig7_row_identification.json", "w"))
