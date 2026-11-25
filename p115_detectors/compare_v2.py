import glob,numpy as np,pandas as pd
def load(d):
    f=pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(f"{d}/flags/*.parquet"))],ignore_index=True)
    return f
def revert_pairs(f):
    e=f[f.check=="extreme_prices"].sort_values(["symbol","day","subtype","ts_recv"])
    same=(e.symbol.eq(e.symbol.shift()))&(e.day.eq(e.day.shift()))&(e.subtype.eq(e.subtype.shift()))
    dt=(e.ts_recv-e.ts_recv.shift()).dt.total_seconds()
    m=same&(dt<=1)&(np.sign(e.value)!=np.sign(e.value.shift()))
    return int(m.sum()),e[m|m.shift(-1,fill_value=False)]
if __name__=="__main__":
    v1=load("results_real"); v2=load("results_real_v2")
    print("v1 revert pairs",revert_pairs(v1)[0], "v2",revert_pairs(v2)[0])

def table():
    v1=load("results_real"); v2=load("results_real_v2")
    r1=pd.read_csv("results_real/run_index.csv"); r2=pd.read_csv("results_real_v2/run_index.csv")
    def cnt(f,ri,tag):
        c=f.groupby(["venue","check"]).size().rename(tag).reset_index()
        if tag=="v1":
            s=ri.groupby("venue")[["seq_gaps","seq_regr"]].sum().sum(axis=1).rename(tag).reset_index().assign(check="sequence_discontinuities")
            c=pd.concat([c,s])
        return c.set_index(["venue","check"])[tag]
    t=pd.concat([cnt(v1,r1,"v1"),cnt(v2,r2,"v2")],axis=1).fillna(0).astype(int)
    lat=pd.Series({("XNAS","latency records"):None,("GLBX","latency records"):None})
    t=t.reset_index()
    t["change_pct"]=((t.v2-t.v1)/t.v1.replace(0,np.nan)*100).round(1)
    t.to_csv("results_real_v2/before_after_by_check.csv",index=False)
    sub=pd.concat([v1.groupby(["venue","check","subtype","severity"]).size().rename("v1"),
                   v2.groupby(["venue","check","subtype","severity"]).size().rename("v2")],axis=1).fillna(0).astype(int)
    sub.reset_index().to_csv("results_real_v2/before_after_by_subtype.csv",index=False)
    v2.groupby(["venue","symbol","day","check"]).size().rename("n").reset_index().to_csv("results_real_v2/flags_by_symbol_day_check.csv",index=False)
    v2.groupby(["check","subtype","severity"]).size().rename("n").reset_index().to_csv("results_real_v2/flags_by_check.csv",index=False)
    lr=r2.groupby("venue").latency_records.sum()
    n1,_=revert_pairs(v1); n2,rp=revert_pairs(v2)
    rp.to_csv("results_real_v2/extreme_revert_pairs.csv",index=False)
    return t,sub,lr,n1,n2,v1,v2,r2
if __name__=="__main__":
    t,sub,lr,n1,n2,v1,v2,r2=table()
    print(t.to_string()); print(sub.to_string()); print(lr); print("revert",n1,n2)
