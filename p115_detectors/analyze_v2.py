# v2 rerun: applies the threshold changes in detectors/config.py (v1 kept in config_v1.py).
# Writes results_real_v2/ only. Pattern of analyze.py + rerun_missing.py: <=2 workers, maxtasksperchild=1,
# skips symbol-days that already have a flag file, big symbol-days (>4M records in v1) run one at a time.
import os,sys,json,time,numpy as np,pandas as pd
from multiprocessing import Pool
import analyze as a
from config import equities_config,cme_futures_config
from detectors import run_all,sequence_discontinuities,CHECKS
from config import SequenceConfig
R2=a.Path("results_real_v2");(R2/"flags").mkdir(parents=True,exist_ok=True)
TICK={k:(0.01 if (k in a.EQ and (v is None or v!=v)) else v) for k,v in json.load(open(a.R/"ticks.json")).items()}
def cfg_for(venue,sym):
    c=equities_config() if venue=="XNAS" else cme_futures_config()
    c.spreads.tick_size=dict(TICK)
    if venue=="XNAS": c.stale.reference={"symbol":"QQQ" if sym=="SPY" else "SPY"}
    elif sym=="CL.c.0": c.stale.reference={"symbol":"ES.c.0"}
    if venue=="GLBX":
        # MDP3 sequence is a per-channel counter and these are single-instrument pulls: per-instrument
        # gaps are still mostly other instruments' messages (ES 10-05: 1.26M "gaps"), so gaps are
        # counted in run_index (glbx_instr_seq_gaps) but not flagged; regressions stay flagged.
        c.sequence.gap_mode="skip"
    return c
def one(t):
    venue,p,stp,sym,day,t0,t1=t
    out=R2/"flags"/f"{venue}_{sym}_{day}.parquet"
    if out.exists(): return None
    df=a.load(p,sym,t0,t1)
    if df.empty: return dict(venue=venue,day=day,symbol=sym,records=0)
    st=a.load(stp,sym); c=cfg_for(venue,sym)
    ref=a.load(p,c.stale.reference["symbol"],t0,t1) if c.stale.reference else None
    t_=time.time()
    fl,_=run_all(df,c,status=st,reference_df=ref)
    # info only: how many gaps the v1 publisher-level rule would still see
    sq=sequence_discontinuities(df,SequenceConfig())
    sqi=sequence_discontinuities(df,SequenceConfig(per_instrument=True)) if venue=="GLBX" else sq.iloc[:0]
    gi=sqi[sqi.subtype=="gap"]
    fl=fl.assign(venue=venue,day=day,symbol=sym); fl.to_parquet(out)
    lo=fl[fl.subtype=="latency_outlier"]
    lat_rec=int(lo.detail.str.extract(r"records=(\d+)")[0].astype(float).sum()) if len(lo) else 0
    r=dict(venue=venue,day=day,symbol=sym,records=len(df),secs=round(time.time()-t_,1),latency_records=lat_rec,
           v1rule_seq_gaps=int((sq.subtype=="gap").sum()),n_flags=len(fl),
           glbx_instr_seq_gaps=len(gi),glbx_instr_gap_share_steps=round(len(gi)/max(len(df)-1,1),4))
    json.dump(r,open(R2/"flags"/f"{venue}_{sym}_{day}.json","w"))
    return r
if __name__=="__main__":
    big=pd.read_csv(a.R/"run_index.csv").query("records>4e6")[["venue","day","symbol"]].apply(tuple,axis=1).tolist()
    T=a.tasks(); B=[t for t in T if (t[0],t[4],t[3]) in big]; S=[t for t in T if t not in B]
    print("small",len(S),"big",len(B),flush=True)
    for tasks,np_ in ((B,1),(S,int(os.environ.get("NP","2")))):
        with Pool(np_,maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(one,tasks,chunksize=1): print(r,flush=True)
    rows=[json.load(open(f)) for f in sorted((R2/"flags").glob("*.json"))]
    pd.DataFrame(rows).to_csv(R2/"run_index.csv",index=False)
