# Synthetic recall for missing_intervals options (detectors/synthetic.py injected 20 s outages), 3 seeds.
import copy,numpy as np,pandas as pd
from synthetic import generate_mbp1,inject_faults,to_databento_df
from detectors import normalize,run_all
from evaluate import match,base_config
from sens_v3 import OPTS
rows=[]
for seed in (7,8,9):
    raw=generate_mbp1(seed=seed); f,lab=inject_faults(raw,seed=seed+4)
    norm=normalize(to_databento_df(f)); clean=normalize(to_databento_df(raw))
    for o,v in OPTS.items():
        c=base_config(); c.missing.model,c.missing.min_gap_s,c.missing.alpha=v[:3]; c.missing.min_kappa=v[3] if len(v)>3 else 2.05
        fl,_=run_all(norm,c,checks=["missing_intervals"]); r=match(fl,lab).set_index("check").loc["missing_intervals"]
        cf,_=run_all(clean,c,checks=["missing_intervals"])
        rows.append(dict(seed=seed,option=o,labels=r["labels"],detected=r["detected"],precision=r["precision"],flags=r["flags"],clean_flags=len(cf)))
d=pd.DataFrame(rows); d.to_csv("results_real_v3/synthetic_missing.csv",index=False)
print(d.groupby("option",sort=False)[["labels","detected","flags","clean_flags"]].sum().to_string())
