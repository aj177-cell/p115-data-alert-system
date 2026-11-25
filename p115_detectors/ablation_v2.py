# One-change-at-a-time ablation of the v2 extreme_prices changes on 4 symbol-days (writes results_real_v2/ablation_extreme.csv)
import copy,json,pandas as pd,sys
import analyze as a, analyze_v2 as b
from detectors import normalize,extreme_prices
from config import SessionCalendar
rows=[]
for sym,day in [("NBIS","2026-10-08"),("ADTN","2026-10-08"),("MSFT","2026-10-08"),("CRWV","2026-10-14")]:
    t=[x for x in a.tasks() if x[3]==sym and x[4]==day][0]
    n=normalize(a.load(t[1],sym,t[5],t[6])); c=b.cfg_for("XNAS",sym)
    def run(k,cal,floor_max,tick):
        cc=copy.deepcopy(c.extreme); cc.k=k; cc.calendar=cal
        if not floor_max: cc.mad_floor_log=0.0  # v1 floor = tick/p when tick known
        return len(extreme_prices(n,cc,tick_size={sym:tick} if tick else {sym:float('nan')},default_tick=None))
    v1=run(8,None,False,None)
    for name,args in [("v1 (NaN tick -> 1e-4 floor)",(8,None,True,None)),("tick 0.01, v1 floor d/p",(8,None,False,0.01)),
                      ("+floor max(d/p,1e-4)",(8,None,True,0.01)),("+RTH gate",(8,SessionCalendar(),True,0.01)),("+k=12 (=v2)",(12,SessionCalendar(),True,0.01))]:
        rows.append(dict(symbol=sym,day=day,step=name,n=run(*args)))
d=pd.DataFrame(rows); d.to_csv("results_real_v2/ablation_extreme.csv",index=False); print(d.pivot(index="step",columns="symbol",values="n"))
