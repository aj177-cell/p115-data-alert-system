# v1/v2/v3 missing_intervals counts by venue/symbol and severity; writes results_real_v3/ only.
import pandas as pd, pyarrow.parquet as pq, glob
def load(d):
    fs=[pd.read_parquet(f,columns=["check","severity","symbol","day","venue"]) for f in glob.glob(f"{d}/flags/*.parquet")]
    x=pd.concat([f for f in fs if len(f)]); return x[x.check=="missing_intervals"] if "check" in x else x
fr={}
for v,d in (("v1","results_real"),("v2","results_real_v2"),("v3","results_real_v3")):
    x=load(d); fr[v]=x
rows=[]
for v,x in fr.items():
    g=x.groupby(["venue","symbol","severity"]).size().rename(v); rows.append(g)
t=pd.concat(rows,axis=1).fillna(0).astype(int).unstack("severity",fill_value=0)
t.columns=[f"{v}_{s}" for v,s in t.columns]; t=t[[c for v in ("v1","v2","v3") for c in (f"{v}_medium",f"{v}_high") if c in t]]
t.loc[("ALL","total"),:]=t.sum(); t.to_csv("results_real_v3/missing_by_symbol_v1_v2_v3.csv"); print(t.to_string())
