"""Measure the envelope of the 20 sample lines per slot in Fig. 7 (and Fig. 5) after hiding the stroked mean/observation polylines."""
import sys, json, ctypes
import numpy as np
import pypdfium2 as pdfium, pypdfium2.raw as raw
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from pdfvec import PDF, getmat, matmul, apply
OUT = r"C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/outputs/verify/v-paper/"
SP = r"C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-paper/"
DPI = 2400; S = DPI / 72.0
doc = pdfium.PdfDocument(PDF)

def hide_polylines(rawparent, is_form):
    n = raw.FPDFFormObj_CountObjects(rawparent) if is_form else raw.FPDFPage_CountObjects(rawparent)
    k = 0
    for i in range(n):
        o = raw.FPDFFormObj_GetObject(rawparent, i) if is_form else raw.FPDFPage_GetObject(rawparent, i)
        t = raw.FPDFPageObj_GetType(o)
        if t == raw.FPDF_PAGEOBJ_FORM:
            k += hide_polylines(o, True)
        elif t == raw.FPDF_PAGEOBJ_PATH and raw.FPDFPath_CountSegments(o) >= 100:
            mode = ctypes.c_int(); st = ctypes.c_int()
            raw.FPDFPath_GetDrawMode(o, ctypes.byref(mode), ctypes.byref(st))
            if st.value:  # stroked polyline (mean / observation) -> transparent
                raw.FPDFPageObj_SetStrokeColor(o, 255, 255, 255, 0); k += 1
    return k

panels = {  # (page index, y-axis x, x-axis y) read from Fig. 7's own axis lines
 'fig7_c0_CVAE': (11, 58.723, 626.158), 'fig7_c1_CVAE': (11, 58.723, 515.722), 'fig7_c2_CVAE': (11, 58.723, 405.286), 'fig7_c3_CVAE': (11, 58.723, 294.85),
 'fig7_c0_GAM': (11, 289.235, 625.8), 'fig7_c1_GAM': (11, 289.235, 515.364), 'fig7_c2_GAM': (11, 289.235, 404.927), 'fig7_c3_GAM': (11, 289.235, 294.491),
}
res = {}
pages = {}
for pno in sorted(set(v[0] for v in panels.values())):
    pg = doc[pno]; nh = hide_polylines(pg.raw, False); pages[pno] = pg
    print("page", pno+1, "hidden polylines:", nh)
f7 = json.load(open(OUT + "fig5_fig7_row_identification.json"))
for name, (pno, xa, y0) in panels.items():
    pg = pages[pno]; W, H = pg.get_size()
    # calibration (same layout as Fig. 8): y ticks every 17.555 pt from y0+12.726 (0.1 kWh); x ticks
    off = np.array([12.725, 30.28, 47.835, 65.389, 82.946]) if xa < 100 else np.array([12.495, 29.732, 46.968, 64.204, 81.443])
    ytk = y0 + off;  ay, by = np.polyfit(ytk, [0.1,0.2,0.3,0.4,0.5], 1)
    xt = [67.565, 126.935, 186.304, 245.673] if xa < 100 else [297.405, 352.255, 407.11, 461.96]
    ax, bx = np.polyfit([0,48,96,144], xt, 1)     # slot -> x (pt)
    left, right, bottom, top = xa + 0.8, xt[-1] + 2, y0 + 0.8, y0 + 85.0
    img = pg.render(scale=S, crop=(left, bottom, W - right, H - top), grayscale=True).to_numpy()
    if img.ndim == 3: img = img[..., 0]
    hpx, wpx = img.shape
    env = np.full((144, 2), np.nan)
    for s in range(144):
        xpx = int(round((ax*s + bx - left) * S))
        col = img[:, max(xpx-1,0):xpx+2].min(axis=1)
        dark = np.where(col < 245)[0]
        if len(dark):
            ytop = top - dark.min()/S; ybot = top - (dark.max()+1)/S        # pt
            env[s] = [ay*ybot + by, ay*ytop + by]                           # kWh (low, high)
    width = env[:,1] - env[:,0]
    mean = np.array(f7[f"fig7_row{int(name[6])+1}_{name.split('_')[2]}"]["mean"])
    inside = np.mean((mean >= env[:,0] - 0.002) & (mean <= env[:,1] + 0.002))
    res[name] = dict(env_low=env[:,0].tolist(), env_high=env[:,1].tolist(), width=width.tolist())
    print(f"{name}: envelope width (kWh, incl. line width) median {np.nanmedian(width):.4f}  p10 {np.nanpercentile(width,10):.4f}  p90 {np.nanpercentile(width,90):.4f}  day1 median {np.nanmedian(width[:48]):.4f}; mean inside env {inside:.2f}")
json.dump(res, open(OUT + "fig7_sample_envelopes.json", "w"))
# save one check image
img = pages[11].render(scale=300/72, crop=(35, 274, 576-539, 783-718), grayscale=True).to_pil(); img.save(SP + "fig7_samples_only_check.png")
