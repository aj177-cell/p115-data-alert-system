"""Stratum tables for docs/m2_representative_sample.md (reads m2/sample/results/cache and the pilot results)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m2.sample.strata import stratum_of, tags_for

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "m2" / "sample" / "results"
CACHE = OUT / "cache"
CHECKS = ["crossed_books", "extreme_prices", "timestamp_inversions", "missing_intervals", "stale_quotes",
          "unusual_spreads", "sequence_discontinuities"]
ORDER = ["normal_week", "fomc", "es_roll", "cl_expiry", "holiday", "normal_week_c1"]


def _rows(mode):
    return pd.DataFrame([json.loads(f.read_text()) for f in sorted((CACHE / mode).glob("*.json"))]) if (CACHE / mode).exists() else pd.DataFrame()


def _strat(df):
    df = df.copy()
    df["tags"] = [ "|".join(tags_for(v, s, d)) for v, s, d in zip(df["venue"], df["symbol"], df["day"])]
    df["stratum"] = [stratum_of(t.split("|")) for t in df["tags"]]
    return df


def v3_tables():
    d = _strat(_rows("v3")).fillna(0)
    cols = [f"n:{c}" for c in CHECKS]
    for c in cols:
        if c not in d:
            d[c] = 0
    d["venue_group"] = np.where(d["venue"] == "XNAS", "XNAS", np.where(d["symbol"].str.endswith(".c.1"), "GLBX c.1", "GLBX c.0"))
    g = d.groupby(["venue_group", "stratum"])
    t = g[cols].mean().round(1)
    t.insert(0, "symbol_days", g.size())
    t.columns = [c.replace("n:", "") for c in t.columns]
    t.to_csv(OUT / "v3_by_stratum.csv")
    # per-symbol ratio to the symbol's normal-week mean
    nw = d[d["stratum"] == "normal_week"].groupby("symbol")[cols].mean()
    r = []
    for _, row in d[d["source"] == "sample"].iterrows():
        sym = row["symbol"].replace(".c.1", ".c.0")
        if sym not in nw.index:
            continue
        for c in cols:
            base = nw.loc[sym, c]
            r.append(dict(stratum=row["stratum"], venue_group=row["venue_group"], symbol=row["symbol"], day=row["day"],
                          check=c[2:], count=row[c], normal_mean=base))
    r = pd.DataFrame(r)
    r.to_csv(OUT / "v3_vs_normal_by_symbol_day.csv", index=False)
    return d, t


def holiday_effect():
    d = _rows("v3")
    if "holiday" not in d:
        return pd.DataFrame()
    d = d[d["holiday"].notna() & (d["holiday"] != 0)].fillna(0)
    rows = []
    for _, r in d.iterrows():
        for c in CHECKS:
            a, b = r.get(f"n:{c}", 0), r.get(f"h:{c}", 0)
            rows.append(dict(symbol=r["symbol"], day=r["day"], kind=r["holiday"], check=c, default=int(a), holiday_calendar=int(b),
                             suppressed=int(a - b)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "holiday_effect_v3.csv", index=False)
    return out


def bayes_table():
    b = _strat(_rows("bayes")).fillna(0)
    p = pd.read_csv(ROOT / "m2/results/bayes/real_symbol_days.csv")
    p = p[p["day_type"] == "normal_week"].rename(columns={"poisson": "default:poisson", "bayes": "default:bayes"})
    p["stratum"] = "normal_week"; p["source"] = "pilot"
    allr = pd.concat([b, p[["venue", "symbol", "day", "stratum", "source", "records", "default:poisson", "default:bayes"]]], ignore_index=True)
    allr["venue_group"] = np.where(allr["venue"] == "XNAS", "XNAS", np.where(allr["symbol"].str.endswith(".c.1"), "GLBX c.1", "GLBX c.0"))
    g = allr.groupby(["venue_group", "stratum"])
    t = pd.DataFrame({"symbol_days": g.size(), "poisson_per_day": g["default:poisson"].mean().round(1),
                      "bayes_per_day": g["default:bayes"].mean().round(1)})
    t.to_csv(OUT / "bayes_by_stratum.csv")
    hol = b[b.get("holiday", 0).astype(str).isin(["early_close", "halt", "closed"])] if "holiday" in b else b.iloc[:0]
    if len(hol):
        hol[["symbol", "day", "holiday", "default:poisson", "holiday:poisson", "default:bayes", "holiday:bayes",
             "default:max_gap_s", "holiday:max_gap_s"]].to_csv(OUT / "holiday_effect_bayes.csv", index=False)
    return t, hol


def spc_table():
    s = pd.read_csv(OUT / "spc_by_symbol_day.csv")
    s = _strat(s)
    s = s[s["variant"] == "default"]
    p = pd.read_csv(ROOT / "m2/results/spc/real_signals_by_symbol_day.csv")
    p = p[(p["params"] == "phase1") & (p["day_type"] == "normal_week")].assign(stratum="normal_week")
    keys = ["kind", "chart"]
    a = s.groupby(["stratum"] + keys)["signals"].mean().unstack(keys).round(2)
    b = p.groupby(["stratum"] + keys)["signals"].mean().unstack(keys).round(2)
    t = pd.concat([b, a])
    t.to_csv(OUT / "spc_by_stratum.csv")
    return t


def halt_check():
    """Last record before the holiday halt vs the calendar (sample futures holiday days)."""
    import pyarrow.dataset as ds, pyarrow.compute as pc
    from m2.sample import holidays as H
    from m2.sample.strata import SAMPLE_ROOT, sample_tasks
    rows = []
    for venue, piece, _, _, symbol, day, t0, t1 in sample_tasks():
        kind = H.is_special(venue, day, H.root_of(symbol))
        if not kind:
            continue
        tz = "America/New_York" if venue == "XNAS" else "America/Chicago"
        t = ds.dataset(SAMPLE_ROOT / piece / "data.parquet").to_table(
            filter=(pc.field("symbol") == symbol) & (pc.field("ts_recv") >= pd.Timestamp(t0)) & (pc.field("ts_recv") < pd.Timestamp(t1)),
            columns=["ts_recv", "action"]).to_pandas()
        if "ts_recv" not in t:
            t = t.reset_index()
        loc = pd.to_datetime(t["ts_recv"], utc=True).dt.tz_convert(tz)
        same = loc[loc.dt.strftime("%Y-%m-%d") == day]
        trades = same[t.loc[same.index, "action"] == "T"]
        cal = (H.NYSE_EARLY.get(day) if venue == "XNAS" else H.CME_HALT.get(H.root_of(symbol), {}).get(day))
        rows.append(dict(venue=venue, symbol=symbol, day=day, kind=kind, calendar_close_local=cal,
                         last_trade_local=str(trades.max().time()) if len(trades) else None,
                         last_record_local_before_17=str(same[same.dt.hour < 17].max().time()) if len(same) else None))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "holiday_halt_check.csv", index=False)
    return out


def main():
    pd.set_option("display.width", 250)
    d, t = v3_tables(); print(t.to_string())
    print(holiday_effect().query("suppressed != 0").to_string(index=False))
    bt, hol = bayes_table(); print(bt.to_string()); print(hol.to_string(index=False) if len(hol) else "")
    if (OUT / "spc_by_symbol_day.csv").exists():
        print(spc_table().to_string())
    print(halt_check().to_string(index=False))
    e = _rows("engine")
    if len(e):
        e = _strat(e).fillna(0)
        cols = [c for c in e.columns if c.startswith("rule:")] + ["alerts", "crossed_no_f_last", "alerts_level_gt0_after_demote"]
        et = e.groupby("stratum")[cols].mean().round(2); et.insert(0, "symbol_days", e.groupby("stratum").size())
        et.to_csv(OUT / "engine_by_stratum.csv"); print(et.to_string())


if __name__ == "__main__":
    main()
