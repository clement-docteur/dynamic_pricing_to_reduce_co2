"""Exact Fig. 4 box statistics from the PDF vector paths (page 11), calibrated on the y-axis tick marks."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from pdfvec import page_objects

OUT = r"C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/outputs/verify/v-paper/"
UP = r"C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/upstream/repo/data/"
doc, page, objs = page_objects(10)
paths = [o for o in objs if o['kind'] == 'path' and o['pts']]
def bb(o):
    xs = [p[1] for p in o['pts']]; ys = [p[2] for p in o['pts']]
    return min(xs), max(xs), min(ys), max(ys)
f4 = [o for o in paths if bb(o)[2] >= 640 and bb(o)[3] <= 720]

# panel plot areas: the white rect with black axis whose y range is 662.59-696.48
panels = {'RMSE': (60.0, 195.0, [0.025, 0.050, 0.075]),
          'ES':   (238.0, 373.0, [0.100, 0.125, 0.150, 0.175, 0.200]),
          'VG':   (401.0, 536.0, [10, 20, 30])}
res = {}
for name, (xa, xb, labels) in panels.items():
    # y tick marks: short horizontal lines ending at the axis x (xa..xa+1), length ~1.48 pt
    ticks = sorted(bb(o)[2] for o in f4 if len(o['pts']) == 2 and abs(bb(o)[2]-bb(o)[3]) < 1e-3
                   and xa - 2.5 < bb(o)[0] < xa + 1 and 1.0 < bb(o)[1]-bb(o)[0] < 2.0)
    assert len(ticks) == len(labels), (name, ticks)
    a, b = np.polyfit(ticks, labels, 1)          # value = a*y + b
    resid = np.array(labels) - (a*np.array(ticks) + b)
    # x tick marks (cluster positions): short vertical lines below the x axis
    xt = sorted(bb(o)[0] for o in f4 if len(o['pts']) == 2 and abs(bb(o)[0]-bb(o)[1]) < 1e-3
                and 660.5 < bb(o)[2] < 661.5 and xa < bb(o)[0] < xb)
    boxes = [o for o in f4 if len(o['pts']) == 5 and o['fill'][:3] in [(169,169,169),(105,105,105)]
             and xa < bb(o)[0] < xb and bb(o)[2] < 700]
    lines = [o for o in f4 if len(o['pts']) == 2 and xa < bb(o)[0] < xb and bb(o)[2] < 700]
    dots = [o for o in f4 if o['fill'][:3] == (51,51,51) and len(o['pts']) == 13 and xa < bb(o)[0] < xb]
    rows = []
    for bx in boxes:
        x0, x1, y0, y1 = bb(bx); xc = (x0+x1)/2
        gen = 'CVAE' if bx['fill'][:3] == (169,169,169) else 'GAM'
        cl = int(np.argmin([abs(xc - t) for t in xt]))
        med = [bb(l) for l in lines if abs(bb(l)[0]-x0) < 0.01 and abs(bb(l)[1]-x1) < 0.01 and abs(bb(l)[2]-bb(l)[3]) < 1e-3]
        assert len(med) == 1
        wh = [bb(l) for l in lines if abs(bb(l)[0]-xc) < 0.01 and abs(bb(l)[1]-xc) < 0.01]
        assert len(wh) == 2, wh
        lo = min(w[2] for w in wh); hi = max(w[3] for w in wh)
        od = [(bb(d)[2]+bb(d)[3])/2 for d in dots if x0 <= (bb(d)[0]+bb(d)[1])/2 <= x1]
        v = lambda y: a*y + b
        rows.append(dict(panel=name, cluster=cl, gen=gen, q1=v(y0), median=v(med[0][2]), q3=v(y1),
                         whisk_lo=v(lo), whisk_hi=v(hi), n_outliers=len(od),
                         outliers=sorted(round(v(y), 4) for y in od)))
    res[name] = dict(tick_y=ticks, labels=labels, slope=a, intercept=b, calib_max_resid=float(np.abs(resid).max()),
                     cluster_x=xt, boxes=sorted(rows, key=lambda r: (r['cluster'], r['gen'])))

df = pd.DataFrame([r for p in res.values() for r in p['boxes']])
pd.set_option('display.width', 200)
print(df.drop(columns='outliers').round(5).to_string(index=False))
for name in res: print(name, 'calib max resid', res[name]['calib_max_resid'], 'value per pt', res[name]['slope'])

# authors' cluster means (kWh per household per half-hour)
pc = pd.read_feather(UP + "power_consump_nmf.feather")
te = pd.read_feather(UP + "test_nmf.feather")
cc = [f"consumption_{h:02d}" for h in range(48)]
means = {'year': pc.groupby('cluster')[cc].mean().mean(axis=1).to_dict(),
         'test': te.groupby('cluster')[cc].mean().mean(axis=1).to_dict()}
print('cluster means year', {k: round(v,4) for k,v in means['year'].items()})
print('cluster means test', {k: round(v,4) for k,v in means['test'].items()})
json.dump(dict(fig4=res, cluster_means=means), open(OUT + "fig4_exact_from_pdf_vectors.json", "w"), indent=1, default=float)
df.drop(columns='outliers').to_csv(OUT + "fig4_exact_from_pdf_vectors.csv", index=False)
