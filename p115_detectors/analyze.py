import sys,json,os,time,numpy as np,pandas as pd,pyarrow.dataset as ds,pyarrow.compute as pc
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0,"detectors")
from config import equities_config,cme_futures_config
from detectors import run_all,sequence_discontinuities,CHECKS
from config import SequenceConfig
D=Path("data");R=Path("results_real");(R/"flags").mkdir(parents=True,exist_ok=True)
EQ=["AAPL","MSFT","NVDA","SPY","QQQ","CRWV","NBIS","LWAY","PLAB","ADTN"]
def load(piece,sym=None,t0=None,t1=None,cols=None):
    d=ds.dataset(D/piece/"data.parquet")
    f=None
    if sym is not None: f=pc.field("symbol")==sym
    if t0 is not None:
        g=(pc.field("ts_recv")>=pd.Timestamp(t0))&(pc.field("ts_recv")<pd.Timestamp(t1))
        f=g if f is None else f&g
    df=d.to_table(filter=f,columns=cols).to_pandas()
    if "ts_recv" in df.columns: df=df.set_index("ts_recv")
    return df
def defs(pieces):
    m={}
    for p in pieces:
        x=load(p)
        for s,v in zip(x["symbol"],x["min_price_increment"]): m[s]=float(v)
    return m
def tasks():
    T=[]
    for p,days in [("A_mbp-1_2025-11-28",["2025-11-28"]),("A_mbp-1_2026-09-16",["2026-09-16"]),
                   ("A_mbp-1_2026-10-05",["2026-10-05","2026-10-06","2026-10-07","2026-10-08","2026-10-09"]),("A_mbp-1_2026-10-14",["2026-10-14"])]:
        st=p.replace("mbp-1","status")
        for d in days:
            for s in EQ: T.append(("XNAS",p,st,s,d,pd.Timestamp(d,tz="UTC"),pd.Timestamp(d,tz="UTC")+pd.Timedelta(days=1)))
    for p,days,syms in [("B_mbp-1_wk",["2026-10-05","2026-10-06","2026-10-07","2026-10-08","2026-10-09"],["ES.c.0","CL.c.0"]),
                        ("B_mbp-1_vol",["2026-10-14"],["ES.c.0","CL.c.0"]),("B_mbp-1_deg",["2026-08-28","2026-08-31"],["ES.c.0"])]:
        st=p.replace("mbp-1","status")
        for d in days:
            for s in syms:
                a=pd.Timestamp(d,tz="UTC")-pd.Timedelta(hours=2); T.append(("GLBX",p,st,s,d,a,a+pd.Timedelta(days=1)))
    T.append(("GLBX","B_mbp-1_deg","B_status_deg","ES.c.0","2026-08-29_weekend",pd.Timestamp("2026-08-28T22:00Z"),pd.Timestamp("2026-08-30T22:00Z")))
    return T
def cfg_for(venue,sym,tick):
    c=equities_config() if venue=="XNAS" else cme_futures_config()
    c.spreads.tick_size=dict(tick)
    if venue=="XNAS": c.stale.reference={"symbol":"QQQ" if sym=="SPY" else "SPY"}
    elif sym=="CL.c.0": c.stale.reference={"symbol":"ES.c.0"}
    return c
def one(t):
    venue,p,stp,sym,day,t0,t1=t
    tick=json.load(open(R/"ticks.json"))
    df=load(p,sym,t0,t1)
    if df.empty: return dict(venue=venue,day=day,symbol=sym,records=0)
    st=load(stp,sym)
    c=cfg_for(venue,sym,tick); ref=load(p,c.stale.reference["symbol"],t0,t1) if c.stale.reference else None
    t_=time.time()
    fl,_=run_all(df,c,status=st,reference_df=ref,checks=[x for x in CHECKS if x!="sequence_discontinuities"])
    sq=sequence_discontinuities(df,SequenceConfig())
    fl=fl.assign(venue=venue,day=day,symbol=sym)
    fl.to_parquet(R/"flags"/f"{venue}_{sym}_{day}.parquet")
    g=sq[sq.subtype=="gap"]
    return dict(venue=venue,day=day,symbol=sym,records=len(df),secs=round(time.time()-t_,1),
        seq_gaps=len(g),seq_regr=int((sq.subtype=="regression").sum()),seq_gap_msgs=float(g.value.sum()) if len(g) else 0.0,
        seq_gap_median=float(g.value.median()) if len(g) else np.nan,
        maybe_bad_book=int(((df["flags"].astype(int)&4)!=0).sum()),bad_ts=int(((df["flags"].astype(int)&8)!=0).sum()))
if __name__=="__main__":
    tick=defs([p for p in os.listdir(D) if "definition" in p]); json.dump(tick,open(R/"ticks.json","w"))
    T=tasks()
    if len(sys.argv)>1: T=[t for t in T if sys.argv[1] in t[1]]
    rows=[]
    with Pool(int(os.environ.get("NP","3"))) as pool:
        for r in pool.imap_unordered(one,T): print(r,flush=True); rows.append(r)
    out=R/"run_index.csv"; new=pd.DataFrame(rows)
    if out.exists():
        old=pd.read_csv(out); k=["venue","day","symbol"]
        old=old[~old.set_index(k).index.isin(new.set_index(k).index)]; new=pd.concat([old,new])
    new.to_csv(out,index=False)
