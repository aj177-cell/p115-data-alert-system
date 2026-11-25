# v3 rerun of missing_intervals only (other checks unchanged from v2), v3 config (Poisson, min_gap 5 s).
# Writes results_real_v3/ only. <=2 workers, maxtasksperchild=1, >4M-record symbol-days one at a time.
import os,json,time,pandas as pd
from multiprocessing import Pool
import analyze as a, analyze_v2 as a2
from detectors import missing_intervals,normalize
R3=a.Path("results_real_v3");(R3/"flags").mkdir(parents=True,exist_ok=True)
COLS=["ts_recv","ts_event","publisher_id","instrument_id","symbol","rtype","sequence"]
def one(t):
    venue,p,stp,sym,day,t0,t1=t
    out=R3/"flags"/f"{venue}_{sym}_{day}.parquet"
    if out.exists(): return None
    df=a.load(p,sym,t0,t1,cols=COLS)
    t_=time.time(); n=len(df)
    fl=missing_intervals(normalize(df),a2.cfg_for(venue,sym).missing) if n else pd.DataFrame()
    fl=fl.assign(venue=venue,day=day,symbol=sym); fl.to_parquet(out)
    r=dict(venue=venue,day=day,symbol=sym,records=n,secs=round(time.time()-t_,1),flags=len(fl),
           medium=int((fl.get("severity")=="medium").sum()) if n else 0,high=int((fl.get("severity")=="high").sum()) if n else 0)
    json.dump(r,open(R3/"flags"/f"{venue}_{sym}_{day}.json","w")); return r
if __name__=="__main__":
    big=pd.read_csv(a.R/"run_index.csv").query("records>4e6")[["venue","day","symbol"]].apply(tuple,axis=1).tolist()
    T=a.tasks(); B=[t for t in T if (t[0],t[4],t[3]) in big]; S=[t for t in T if t not in B]
    print("small",len(S),"big",len(B),flush=True)
    for tasks,np_ in ((B,1),(S,int(os.environ.get("NP","2")))):
        with Pool(np_,maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(one,tasks,chunksize=1): print(r,flush=True)
    rows=[json.load(open(f)) for f in sorted((R3/"flags").glob("*.json"))]
    pd.DataFrame(rows).to_csv(R3/"run_index.csv",index=False)
