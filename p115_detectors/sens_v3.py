# v3 sensitivity for missing_intervals: real flag counts + recall of outages injected into real data.
# One process, one symbol-day at a time, only the columns the check needs. Writes results_real_v3/ only.
import sys,zlib,json,copy,numpy as np,pandas as pd
import analyze as a, analyze_v2 as a2
from detectors import missing_intervals
from detectors import normalize
OPTS={"P2 (v2)":("poisson",2,1e-6),"P5":("poisson",5,1e-6),"OD2":("overdispersed",2,1e-6),"OD5":("overdispersed",5,1e-6),
      "OD2 a1e-4":("overdispersed",2,1e-4),"TOD2":("tod",2,1e-6),"TOD5":("tod",5,1e-6)}
OPTS.update({"P5 a1e-8":("poisson",5,1e-8),"P5 a1e-10":("poisson",5,1e-10),
          "OD5 k>=4":("overdispersed",5,1e-6,4.0),"OD5 k>=8":("overdispersed",5,1e-6,8.0),"OD5 k>=16":("overdispersed",5,1e-6,16.0)})
COLS=["ts_recv","ts_event","publisher_id","instrument_id","symbol","rtype","sequence"]
def cfg(venue,sym,o):
    c=copy.deepcopy(a2.cfg_for(venue,sym).missing); o_=OPTS[o]; c.model,c.min_gap_s,c.alpha=o_[:3]
    if len(o_)>3: c.min_kappa=o_[3]
    return c
def inject(norm,cal,rng,L,n=10):
    from detectors import SessionClock
    t=norm["ts_recv_ns"].to_numpy(); t=np.sort(t[t!=np.iinfo(np.int64).min])
    clk=SessionClock.build(cal,int(t.min()),int(t.max()))
    # start points: random records in session, >=120 s from the ends, window fully in session
    ins=clk.in_session(t)&clk.in_session(t+L*1_000_000_000)
    cand=np.flatnonzero(ins&(t>t[0]+120e9)&(t<t[-1]-120e9))
    starts=np.sort(rng.choice(t[cand],n,replace=False)); wins=[]
    keep=np.ones(len(norm),bool); tr=norm["ts_recv_ns"].to_numpy()
    for s in starts:
        if wins and s<wins[-1][1]+60e9: continue
        e=s+int(L*1e9); keep&=~((tr>=s)&(tr<e)); wins.append((s,e))
    return norm[keep].reset_index(drop=True),wins
def recall(fl,wins,L):
    if fl.empty: return 0
    ft=pd.to_datetime(fl.ts_recv,utc=True).dt.as_unit("ns").astype("int64").to_numpy(); v=fl.value.to_numpy()
    return sum(bool(((ft>=e)&(ft<=e+60e9)&(v>=0.9*L)).any()) for s,e in wins)
if __name__=="__main__":
    days=[("XNAS","2026-10-08","LWAY"),("XNAS","2026-10-14","PLAB"),("XNAS","2026-10-07","ADTN"),("XNAS","2026-10-08","NBIS"),
          ("XNAS","2026-10-14","MSFT"),("XNAS","2026-10-14","CRWV"),("XNAS","2026-10-14","AAPL"),("GLBX","2026-10-08","CL.c.0"),
          ("XNAS","2026-09-16","NVDA"),("XNAS","2026-10-14","SPY"),("XNAS","2025-11-28","SPY"),("GLBX","2026-10-14","ES.c.0")]
    out="results_real_v3/sensitivity_missing_raw.csv"
    T={(t[0],t[4],t[3]):t for t in a.tasks()}; rows=[]
    for k in days:
        venue,p,stp,sym,day,t0,t1=T[k]
        df=a.load(p,sym,t0,t1,cols=COLS); norm=normalize(df); del df
        for L in (10,20):
            rng=np.random.default_rng(zlib.crc32(f'{sym}|{day}|{L}'.encode()))
            inj,wins=inject(norm,cfg(venue,sym,next(iter(OPTS))).calendar,rng,L)
            for o in OPTS:
                c=cfg(venue,sym,o)
                if L==10:
                    fl=missing_intervals(norm,c)
                    rows.append(dict(venue=venue,day=day,symbol=sym,option=o,kind="real",records=len(norm),
                                     flags=len(fl),medium=int((fl.severity=="medium").sum()),high=int((fl.severity=="high").sum())))
                fi=missing_intervals(inj,c)
                rows.append(dict(venue=venue,day=day,symbol=sym,option=o,kind=f"inject{L}s",injected=len(wins),detected=recall(fi,wins,L)))
            print(k,L,flush=True)
        del norm
        pd.DataFrame(rows).to_csv(out,index=False)
