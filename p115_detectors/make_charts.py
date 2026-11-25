import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
import glob
BASE=os.path.dirname(os.path.abspath(__file__))
R1=os.path.join(BASE,"results_real")+os.sep      # v1 aux inputs (seq hypothesis, sensitivity)
R2=os.path.join(BASE,"results_real_v2")+os.sep   # final flags, all checks except missing_intervals
R3=os.path.join(BASE,"results_real_v3")+os.sep   # final missing_intervals flags (v3 rule, min_gap_s=5.0)
C=os.path.join(R3,"charts")+os.sep
os.makedirs(C,exist_ok=True)
def load(p): return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(p))],ignore_index=True)
d2=load(R2+"flags"+os.sep+"*.parquet"); d3=load(R3+"flags"+os.sep+"*.parquet")
d=pd.concat([d2[d2.check!="missing_intervals"],d3[d3.check=="missing_intervals"]],ignore_index=True)
print("missing_intervals by severity (v3):",d[d.check=="missing_intervals"].severity.value_counts().to_dict())
print("flags by check:",d.check.value_counts().to_dict())
d["cs"]=d.check+":"+d.subtype
# 1 flags by check & venue
t=d.groupby(["cs","venue"]).size().unstack(fill_value=0)
ax=t.plot.barh(figsize=(9,5),logx=True); ax.set_xlabel("flags (log scale)"); ax.set_title("Flags by check, all 95 symbol-days"); plt.tight_layout(); plt.savefig(C+"01_flags_by_check.png",dpi=130); plt.close()
# 2 sequence gaps vs records
s=pd.read_csv(R1+"seq_hypothesis_xnas.csv")
fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(s.records,s.seq_range,label="sequence range (max-min)"); ax.scatter(s.records,s.gap_msgs,marker="x",label="sum of gap sizes")
for _,r in s.iterrows(): ax.annotate(r.symbol,(r.records,r.seq_range),fontsize=7)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("records for symbol (XNAS, 2026-10-08)"); ax.set_ylabel("sequence units"); ax.set_title("XNAS sequence is feed-wide: same range regardless of symbol activity"); ax.legend(); plt.tight_layout(); plt.savefig(C+"02_seq_range_vs_records.png",dpi=130); plt.close()
# 3 sensitivity
S=pd.read_csv(R1+"sensitivity_real.csv"); S=S[S.day=="2026-10-08"]
panels=[("extreme_prices","k"),("missing_intervals","min_gap_s"),("missing_intervals","alpha"),("timestamp_inversions","latency_min_excess_ms"),("stale_quotes","min_age_s"),("stale_quotes","min_trades")]
fig,axs=plt.subplots(2,3,figsize=(13,7))
for a,(c,p) in zip(axs.flat,panels):
    x=S[(S.check==c)&(S.param==p)]
    for sym,g in x.groupby("symbol"): a.plot(g.value,g.n_flags+1,marker="o",label=sym)
    a.set_xscale("log"); a.set_yscale("log"); a.set_title(f"{c}: {p}",fontsize=9); a.set_ylabel("flags+1")
axs[0,0].legend(fontsize=7); fig.suptitle("Threshold sensitivity, 2026-10-08 (v1 sweep; missing_intervals panels are pre-v3)"); plt.tight_layout(); plt.savefig(C+"03_sensitivity.png",dpi=130); plt.close()
# 4 extreme by ET hour, Oct14 vs other XNAS days
x=d[(d.check=="extreme_prices")&(d.venue=="XNAS")].copy(); x["h"]=x.ts_recv.dt.tz_convert("America/New_York").dt.hour
x["grp"]=np.where(x.day=="2026-10-14","2026-10-14 (CPI)","other days (mean)")
nd=x[x.grp!="2026-10-14 (CPI)"].day.nunique()
h=x.groupby(["h","grp"]).size().unstack(fill_value=0); h=h.reindex(range(int(min(h.index.min(),9)),int(max(h.index.max(),16))+1),fill_value=0); h["other days (mean)"]/=nd
ax=h.plot.bar(figsize=(9,4.5)); hl=list(h.index); ax.axvspan(hl.index(9),hl.index(16)-0.5,alpha=.1,label='RTH 09:30-16:00'); ax.set_xlabel("ET hour"); ax.set_ylabel("extreme_prices flags"); ax.set_title("XNAS extreme_prices by ET hour"); plt.tight_layout(); plt.savefig(C+"04_extreme_by_hour.png",dpi=130); plt.close()
# 5 GLBX per-day flags per million records
ri=pd.read_csv(R3+"run_index.csv"); g=d[d.venue=="GLBX"].groupby(["day","symbol","check"]).size().unstack(fill_value=0)
g=g.join(ri[ri.venue=="GLBX"].set_index(["day","symbol"]).records)
for c in ["missing_intervals","timestamp_inversions","extreme_prices"]: g[c]=g[c]/g.records*1e6
es=g.xs("ES.c.0",level="symbol").drop("2026-08-29_weekend")[["missing_intervals","timestamp_inversions","extreme_prices"]]
ax=es.plot.bar(figsize=(9,4.5),logy=True); ax.set_ylabel("flags per million records"); ax.set_title("ES.c.0: degraded window (Aug 28-31) vs normal week and Oct 14"); plt.tight_layout(); plt.savefig(C+"05_es_days_normalized.png",dpi=130); plt.close()
print(es.round(1).to_string()); print(g.records.to_string())
