"""Extract vector objects (paths with points/colors, text chars) from a PDF page in page coordinates."""
import ctypes, sys, json
import pypdfium2 as pdfium
import pypdfium2.raw as raw

PDF = r"C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/Simulating_Tariff_Impact_in_Electrical_Energy_Consumption_Profiles_With_Conditional_Variational_Autoencoders.pdf"

def matmul(m1, m2):
    # PDF matrices [a b c d e f]; apply m1 then m2: x' = x*a + y*c + e
    a1,b1,c1,d1,e1,f1 = m1; a2,b2,c2,d2,e2,f2 = m2
    return (a1*a2 + b1*c2, a1*b2 + b1*d2, c1*a2 + d1*c2, c1*b2 + d1*d2, e1*a2 + f1*c2 + e2, e1*b2 + f1*d2 + f2)

def apply(m, x, y):
    a,b,c,d,e,f = m
    return (x*a + y*c + e, x*b + y*d + f)

def getmat(rawobj):
    m = raw.FS_MATRIX()
    ok = raw.FPDFPageObj_GetMatrix(rawobj, ctypes.byref(m))
    return (m.a, m.b, m.c, m.d, m.e, m.f) if ok else (1,0,0,1,0,0)

def color(rawobj, fn):
    r,g,b,a = (ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint())
    ok = fn(rawobj, ctypes.byref(r), ctypes.byref(g), ctypes.byref(b), ctypes.byref(a))
    return (r.value, g.value, b.value, a.value) if ok else None

def path_info(rawobj, ctm):
    """Return list of segments (type, x, y) in page coords. The path's own matrix is included in ctm."""
    n = raw.FPDFPath_CountSegments(rawobj)
    pts = []
    for i in range(n):
        seg = raw.FPDFPath_GetPathSegment(rawobj, i)
        x = ctypes.c_float(); y = ctypes.c_float()
        raw.FPDFPathSegment_GetPoint(seg, ctypes.byref(x), ctypes.byref(y))
        t = raw.FPDFPathSegment_GetType(seg)
        close = raw.FPDFPathSegment_GetClose(seg)
        px, py = apply(ctm, x.value, y.value)
        pts.append((t, px, py, bool(close)))
    mode = ctypes.c_int(); stroke = ctypes.c_int()
    raw.FPDFPath_GetDrawMode(rawobj, ctypes.byref(mode), ctypes.byref(stroke))
    return pts, mode.value, bool(stroke.value)

def walk(rawparent, is_form, ctm, out, level=0):
    n = raw.FPDFFormObj_CountObjects(rawparent) if is_form else raw.FPDFPage_CountObjects(rawparent)
    for i in range(n):
        o = raw.FPDFFormObj_GetObject(rawparent, i) if is_form else raw.FPDFPage_GetObject(rawparent, i)
        t = raw.FPDFPageObj_GetType(o)
        m = getmat(o)
        if t == raw.FPDF_PAGEOBJ_FORM:
            # children are in form space; form matrix maps form space -> parent space
            out.append(dict(kind='form', level=level))
            walk(o, True, matmul(m, ctm), out, level+1)
        elif t == raw.FPDF_PAGEOBJ_PATH:
            # path points are in path space; path matrix maps to parent space
            pts, mode, stroke = path_info(o, matmul(m, ctm))
            lw = ctypes.c_float(); raw.FPDFPageObj_GetStrokeWidth(o, ctypes.byref(lw))
            out.append(dict(kind='path', level=level, pts=pts, fill=color(o, raw.FPDFPageObj_GetFillColor),
                            stroke=color(o, raw.FPDFPageObj_GetStrokeColor), mode=mode, dostroke=stroke, lw=lw.value))
        elif t == raw.FPDF_PAGEOBJ_TEXT:
            out.append(dict(kind='text', level=level))
        else:
            out.append(dict(kind=str(t), level=level))

def page_objects(pageno):
    doc = pdfium.PdfDocument(PDF)
    page = doc[pageno]
    out = []
    walk(page.raw, False, (1,0,0,1,0,0), out)
    return doc, page, out

def page_chars(page):
    tp = page.get_textpage()
    n = tp.count_chars()
    chars = []
    for i in range(n):
        ch = tp.get_text_range(i, 1)
        l, b, r, t = tp.get_charbox(i)
        chars.append((ch, l, b, r, t))
    return chars
