import json, numpy as np
OUT = r"C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/outputs/verify/v-paper/"
F = json.load(open(OUT + "fig4_exact_from_pdf_vectors.json")); f4 = F["fig4"]; means = F["cluster_means"]
env = json.load(open(OUT + "fig7_sample_envelopes.json"))
f8 = json.load(open(OUT + "fig8_series_from_pdf_vectors.json"))
g = lambda p, gen, k='median': np.array([[r[k] for r in f4[p]['boxes'] if r['cluster'] == c and r['gen'] == gen][0] for c in range(4)])
out = {}
# ---- 13% readings
C, G = g('RMSE','CVAE'), g('RMSE','GAM')
r = {}
r['per_cluster_rel_CVAE'] = ((C-G)/C).round(4).tolist(); r['per_cluster_rel_GAM'] = ((C-G)/G).round(4).tolist()
r['mean_of_ratios_rel_CVAE'] = float(((C-G)/C).mean()); r['mean_of_ratios_rel_GAM'] = float(((C-G)/G).mean())
r['ratio_of_sums_rel_CVAE'] = float((C.sum()-G.sum())/C.sum()); r['ratio_of_sums_rel_GAM'] = float((C.sum()-G.sum())/G.sum())
r['symmetric_(C-G)/(C+G)'] = ((C-G)/(C+G)).round(4).tolist()
for q in ['q1', 'q3']:
    Cq, Gq = g('RMSE','CVAE',q), g('RMSE','GAM',q); r[f'{q}_mean_rel_CVAE'] = float(((Cq-Gq)/Cq).mean()); r[f'{q}_mean_rel_GAM'] = float(((Cq-Gq)/Gq).mean())
sC, sG = np.array([0.020,0.022,0.018,0.025]), np.array([0.015,0.016,0.015,0.018])
r['spec_approx_ratio_of_sums'] = [float((sC.sum()-sG.sum())/sC.sum()), float((sC.sum()-sG.sum())/sG.sum())]
out['rmse_13pct'] = r
# ---- ES level vs cluster mean
yr = np.array([means['year'][str(c)] for c in range(4)]); te = np.array([means['test'][str(c) if str(c) in means['test'] else f"{c}.0"] for c in range(4)])
e = {}
for gen in ['CVAE','GAM']:
    ES = g('ES', gen)
    e[gen] = dict(es_over_mean_year=(ES/yr).round(3).tolist(), es_over_mean_test=(ES/te).round(3).tolist(), r_es_vs_mean=float(np.corrcoef(ES, yr)[0,1]),
                  r_es_vs_sqrt48rmse=float(np.corrcoef(ES, np.sqrt(48)*g('RMSE',gen))[0,1]))
allr = np.array(e['CVAE']['es_over_mean_year'] + e['GAM']['es_over_mean_year']); allt = np.array(e['CVAE']['es_over_mean_test'] + e['GAM']['es_over_mean_test'])
e['pooled_year'] = dict(min=float(allr.min()), max=float(allr.max()), mean=float(allr.mean()), cv=float(allr.std(ddof=1)/allr.mean()))
e['pooled_test'] = dict(min=float(allt.min()), max=float(allt.max()), mean=float(allt.mean()), cv=float(allt.std(ddof=1)/allt.mean()))
# implied ES from the paper's own RMSE medians and sample spread (Figs. 6-7), Gaussian norm-concentration approximation
LW = 0.0036; imp = {}
for gen in ['CVAE','GAM']:
    a = np.sqrt(48) * g('RMSE', gen)
    sd = np.array([np.median(np.clip(np.array(env[f"fig7_c{c}_{gen}"]["width"]) - LW, 0, None)) / 3.735 for c in range(4)])
    S = np.sqrt(48) * sd
    imp[gen] = dict(sd_per_slot=sd.round(4).tolist(), implied_es=(np.sqrt(a**2 + S**2) - S/np.sqrt(2)).round(4).tolist(), plotted_es=g('ES',gen).round(4).tolist(),
                    sqrt48_rmse=a.round(4).tolist())
e['implied_from_paper_rmse_and_spread'] = imp
out['es_level'] = e
# ---- orders
order = lambda v: ">".join(str(i) for i in np.argsort(-v))
out['orders'] = {f"{p}_{gen}": order(g(p, gen)) for p in ['RMSE','ES','VG'] for gen in ['CVAE','GAM']}
out['orders']['mean_consumption_year'] = order(yr); out['orders']['mean_consumption_test'] = order(te)
# ---- scenario windows and CVAE onset / recovery (Fig. 8, rows = clusters 0-3)
sc = {}
for c in range(4):
    for gen in ['CVAE','GAM']:
        d = f8[f"row{c+1}_{gen}"]; N = np.array(d['Normal']); Hh = np.array(d['High']); L = np.array(d['Low'])
        dh = (Hh - N).mean(0); dl = (L - N).mean(0); nm = N.mean(0)
        rec = dict(nonzero_high_slots=[int(i) for i in np.where(np.abs(Hh-N).max(0) > 5e-4)[0]], nonzero_low_slots=[int(i) for i in np.where(np.abs(L-N).max(0) > 5e-4)[0]])
        if gen == 'CVAE':
            for thr_name, thr in [('abs10Wh', 0.010), ('rel5pct', 0.05*nm)]:
                below = dh < -thr if np.ndim(thr) == 0 else dh < -thr
                s = 39
                while s > 0 and below[s-1]: s -= 1
                t = 44
                while t < 47 and below[t+1]: t += 1
                rec[f'high_drop_run_{thr_name}'] = [int(s), int(t)]
            rec['high_effect_pct'] = dict(pre_29_38=float(dh[29:39].sum()/nm[29:39].sum()*100), in_39_44=float(dh[39:45].sum()/nm[39:45].sum()*100),
                                          post_45_47=float(dh[45:48].sum()/nm[45:48].sum()*100), afternoon_20_28=float(dh[20:29].sum()/nm[20:29].sum()*100))
            rec['low_effect_pct'] = dict(pre_0_8=float(dl[0:9].sum()/nm[0:9].sum()*100), in_9_19=float(dl[9:20].sum()/nm[9:20].sum()*100),
                                         after_20_47=float(dl[20:48].sum()/nm[20:48].sum()*100), slots_in_window_above_normal=int((dl[9:20] > 0).sum()))
        else:
            rec['high_effect_pct_in_39_44'] = float(dh[39:45].sum()/nm[39:45].sum()*100); rec['low_effect_pct_in_9_19'] = float(dl[9:20].sum()/nm[9:20].sum()*100)
            rec['xi_high_39_44_kWh'] = dh[39:45].round(4).tolist(); rec['xi_low_9_19_kWh'] = dl[9:20].round(4).tolist()
        sc[f"c{c}_{gen}"] = rec
out['scenarios_fig8'] = sc
json.dump(out, open(OUT + "paper_readings_analysis.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:9000])
