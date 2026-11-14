# Fully independent recomputation of the authors' ToU aggregate from the raw Small LCL Data CSVs, for 4 days.
import duckdb, pandas as pd, numpy as np, json, time
REPRO="C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/CVAE_reproduction/"
SMALL="C:/Users/cleme/Desktop/MIT/Courses/Fall/Machine Learning/Project/Claude_ML_Project/Small LCL Data/"
OUT="C:/Users/cleme/AppData/Local/Temp/claude/2f83cc224499e10e/1aa3d082-c193-42de-888f-c8f5d2fbba90/scratchpad/verify/v-data/"
t=time.time()
con=duckdb.connect()
con.execute("SET threads=2; SET memory_limit='6GB'")
q=f"""
SELECT LCLid, DateTime, "KWH/hh (per half hour)" AS kwh
FROM read_csv('{SMALL}LCL-June2015v2_*.csv', header=true, all_varchar=true)
WHERE (DateTime >= '2013-02-20' AND DateTime < '2013-02-21')
   OR (DateTime >= '2013-07-15' AND DateTime < '2013-07-16')
   OR (DateTime >= '2013-03-29' AND DateTime < '2013-03-30')
   OR (DateTime >= '2013-12-31' AND DateTime < '2014-01-01 00:00:01')
"""
df=con.execute(q).df()
print("rows",len(df),"secs",time.time()-t)
df.to_parquet(OUT+"raw_4days.parquet")
