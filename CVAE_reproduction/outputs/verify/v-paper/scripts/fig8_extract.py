import sys, json
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from pdfvec import page_objects
OUT = r"C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/outputs/verify/v-paper/"
doc, page, objs = page_objects(12)
paths = [o for o in objs if o['kind']=='path' and o['pts']]
def bb(o):
    xs=[p[1] for p in o['pts']]; ys=[p[2] for p in o['pts']]
    return min(xs),max(xs),min(ys),max(ys)
rows_axis = {1: 626.155, 2: 515.715, 3: 405.276, 4: 294.837}   # x-axis y position, row 1 = top
ytick_off = np.array([307.563, 325.118, 342.673, 360.228, 377.787]) - 294.837   # tick offsets above axis (same in each row)
xt = {'CVAE': [67.566, 126.937, 186.309, 245.68], 'GAM': [293.545, 353.435, 413.328, 473.219]}
names = {(0,0,0): 'High', (211,211,211): 'Low', (133,139,151): 'Normal'}
data = {}
for row, y0 in rows_axis.items():
    ay, by = np.polyfit(y0 + ytick_off, [0.1,0.2,0.3,0.4,0.5], 1)
    for gen, ticks in xt.items():
        ax, bx = np.polyfit(ticks, [0, 48, 96, 144], 1)   # slot index since Jan 01 00:00
        for o in paths:
            if len(o['pts']) != 144 or not o['dostroke']: continue
            x0,x1,yy0,yy1 = bb(o)
            if not (y0 - 1 < yy0 and yy1 < y0 + 90 and ticks[0]-5 < x0 < ticks[1] and x1 < ticks[-1] + 5): continue
            nm = names[o['stroke'][:3]]
            s = np.array([ax*p[1]+bx for p in o['pts']]); v = np.array([ay*p[2]+by for p in o['pts']])
            data[(row, gen, nm)] = (s, v)
print("series found:", len(data))
s = data[(1,'CVAE','Normal')][0]
print("x positions in slots: first", np.round(s[:3],3), "last", np.round(s[-2:],3), "max dev from integer grid", np.abs(s-np.round(s)).max())
res = {}
np.set_printoptions(linewidth=250, precision=3, suppress=True)
for row in rows_axis:
    for gen in ['CVAE', 'GAM']:
        N = data[(row,gen,'Normal')][1].reshape(3,48); Hh = data[(row,gen,'High')][1].reshape(3,48); L = data[(row,gen,'Low')][1].reshape(3,48)
        res[f"row{row}_{gen}"] = dict(Normal=N.tolist(), High=Hh.tolist(), Low=L.tolist())
        print(f"\n=== row {row} {gen}: mean Normal level {N.mean():.4f} kWh")
        for d in range(3):
            dh = Hh[d]-N[d]; dl = L[d]-N[d]
            print(f" day{d+1} High-N (x1000):", np.round(1000*dh).astype(int))
            print(f" day{d+1} Low-N  (x1000):", np.round(1000*dl).astype(int))
json.dump(res, open(OUT + "fig8_series_from_pdf_vectors.json", "w"))
