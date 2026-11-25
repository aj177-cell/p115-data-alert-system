"""Representative-sample extension: calendars and a cost/size estimate. No data is pulled.

    python -m m2.carryover.sample_extension            # write the plan and the estimate (offline)
    python -m m2.carryover.sample_extension --get-cost # also ask Databento metadata.get_cost (free);
                                                       # needs DATABENTO_API_KEY in the environment

``--get-cost`` calls only ``metadata.get_cost`` and ``metadata.get_record_count``, which are not
billed, and never ``timeseries.get_range`` or ``batch.submit_job``. The key is read from the
environment and never printed.

Sources (opened 2026-11-22):
  FOMC   https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
  roll   CME rules: ES expires the 3rd Friday of Mar/Jun/Sep/Dec, the CME roll date is 8 calendar days
         earlier; CL trading ends 3 business days before the 25th calendar day of the month before
         delivery (the business day before the 25th if the 25th is not one). Databento ``.c.0``
         switches on the session after expiry (docs/standards-and-conventions/symbology).
  holidays  exchange_calendars XNYS and CMES (2025-2026), checked against the NYSE list.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
EQ = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "CRWV", "NBIS", "LWAY", "PLAB", "ADTN"]

# Decision day (second day) of each FOMC meeting; statement 14:00 ET.
FOMC_DECISION_DAYS = {
    2025: ["2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10"],
    2026: ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"],
}
NYSE_CLOSED = ["2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04",
               "2025-09-01", "2025-11-27", "2025-12-25", "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
               "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"]
NYSE_EARLY_CLOSE = ["2025-07-03", "2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24"]  # 13:00 ET
CME_CLOSED = ["2025-01-01", "2025-04-18", "2025-12-25", "2026-01-01", "2026-04-03", "2026-12-25"]  # exchange_calendars CMES
CME_EARLY_CLOSE = ["2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-11-28",
                   "2025-12-24", "2026-01-19", "2026-02-16", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26",
                   "2026-11-27", "2026-12-24"]

# $ per byte (GiB pricing, verified against get_cost in /workspace/p115/pilot_cost_check.md); MBP-1 = 80 B/record
PRICE_PER_GIB = {"XNAS.ITCH": 1.20, "GLBX.MDP3": 1.80}
RECORD_BYTES = 80

# (label, day_type for the record model, day, dataset, symbols, why)
PLAN = [
    ("fomc", "cpi", "2025-12-10", "both", None, "FOMC decision (SEP meeting); 2026-09-16 in the current pull is also one"),
    ("fomc", "cpi", "2026-01-28", "both", None, "FOMC decision"),
    ("fomc", "cpi", "2026-10-28", "both", None, "FOMC decision, two weeks after the CPI day in the pull"),
    ("es_roll", "normal_week", "2025-12-11", "GLBX.MDP3", ["ES.c.0", "ES.c.1"], "CME roll date ESZ5 -> ESH6: liquidity moves to c.1"),
    ("es_roll", "normal_week", "2025-12-19", "GLBX.MDP3", ["ES.c.0"], "ESZ5 expiry (c.0 still ESZ5)"),
    ("es_roll", "normal_week", "2025-12-22", "GLBX.MDP3", ["ES.c.0"], "first session with c.0 = ESH6 (instrument_id changes)"),
    ("es_roll", "normal_week", "2026-09-10", "GLBX.MDP3", ["ES.c.0", "ES.c.1"], "CME roll date ESU6 -> ESZ6"),
    ("es_roll", "normal_week", "2026-09-18", "GLBX.MDP3", ["ES.c.0"], "ESU6 expiry"),
    ("es_roll", "normal_week", "2026-09-21", "GLBX.MDP3", ["ES.c.0"], "first session with c.0 = ESZ6"),
    ("cl_expiry", "normal_week", "2026-09-22", "GLBX.MDP3", ["CL.c.0", "CL.c.1"], "CLV6 last trade"),
    ("cl_expiry", "normal_week", "2026-09-23", "GLBX.MDP3", ["CL.c.0"], "first session with c.0 = CLX6"),
    ("cl_expiry", "normal_week", "2026-10-20", "GLBX.MDP3", ["CL.c.0", "CL.c.1"], "CLX6 last trade (c.0 in the current pull)"),
    ("cl_expiry", "normal_week", "2026-10-21", "GLBX.MDP3", ["CL.c.0"], "first session with c.0 = CLZ6"),
    ("holiday", "half_day", "2025-12-24", "both", None, "NYSE and CME early close"),
    ("holiday", "normal_week", "2025-12-26", "both", None, "thin post-holiday full session"),
    ("holiday", "half_day", "2026-01-19", "GLBX.MDP3", ["ES.c.0", "CL.c.0"], "NYSE closed (MLK), CME holiday session"),
    ("holiday", "half_day", "2026-07-03", "GLBX.MDP3", ["ES.c.0", "CL.c.0"], "NYSE closed (Independence Day obs.), CME early close"),
]


def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end, freq="C", holidays=NYSE_CLOSED)


def third_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year, month, 1)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7 + 14)


def cl_last_trade(year: int, month: int) -> pd.Timestamp:
    """Last trading day of the CL contract for delivery month (year, month)."""
    py, pm = (year, month - 1) if month > 1 else (year - 1, 12)
    b = _bdays(f"{py}-{pm:02d}-01", f"{py}-{pm:02d}-28")
    d25 = pd.Timestamp(py, pm, 25)
    anchor = b[b <= d25][-1]  # the 25th, or the business day before it
    return b[b.get_loc(anchor) - 3]


def roll_calendar(years=(2025, 2026)) -> pd.DataFrame:
    rows = []
    for y in years:
        for i, m in enumerate((3, 6, 9, 12)):
            e = third_friday(y, m)
            nxt = _bdays(str((e + pd.Timedelta(days=1)).date()), str((e + pd.Timedelta(days=10)).date()))[0]
            rows.append(dict(symbol="ES", contract=f"ES{'HMUZ'[i]}{str(y)[-1]}", expiry=str(e.date()),
                             cme_roll_date=str((e - pd.Timedelta(days=8)).date()), c0_switch_session=str(nxt.date())))
        for m in range(1, 13):
            cy, cm = (y, m + 1) if m < 12 else (y + 1, 1)
            e = cl_last_trade(cy, cm)
            nxt = _bdays(str((e + pd.Timedelta(days=1)).date()), str((e + pd.Timedelta(days=10)).date()))[0]
            rows.append(dict(symbol="CL", contract=f"CL{'FGHJKMNQUVXZ'[cm - 1]}{str(cy)[-1]}", expiry=str(e.date()),
                             cme_roll_date="", c0_switch_session=str(nxt.date())))
    return pd.DataFrame(rows)


def record_model(census: pd.DataFrame) -> pd.DataFrame:
    """Mean records per symbol-day by day type (full UTC day for XNAS, -2 h session window for GLBX)."""
    m = census.pivot_table(index="symbol", columns="day_type", values="records", aggfunc="mean")
    for s in ("ES.c.0", "CL.c.0"):
        if "half_day" not in m or pd.isna(m.loc[s].get("half_day")):
            m.loc[s, "half_day"] = 0.4 * m.loc[s, "normal_week"]  # assumption: holiday session ~40% of a normal day
    m.loc["ES.c.1"] = 0.5 * m.loc["ES.c.0"]  # assumption on roll days: half of c.0
    m.loc["CL.c.1"] = 0.5 * m.loc["CL.c.0"]
    return m


def requests() -> pd.DataFrame:
    rows = []
    for label, dtype, day, ds, syms, why in PLAN:
        if ds == "both":
            rows += [dict(label=label, day=day, model=dtype, dataset="XNAS.ITCH", symbol=s, why=why) for s in EQ]
            rows += [dict(label=label, day=day, model=dtype, dataset="GLBX.MDP3", symbol=s, why=why) for s in ("ES.c.0", "CL.c.0")]
        else:
            rows += [dict(label=label, day=day, model=dtype, dataset=ds, symbol=s, why=why) for s in syms]
    return pd.DataFrame(rows)


def estimate(census: pd.DataFrame) -> pd.DataFrame:
    m = record_model(census)
    r = requests()
    r["records_est"] = [float(m.loc[s, d]) if d in m and pd.notna(m.loc[s, d]) else float(m.loc[s, "normal_week"])
                        for s, d in zip(r["symbol"], r["model"])]
    r["bytes_est"] = r["records_est"] * RECORD_BYTES
    r["usd_est"] = r["bytes_est"] / 2**30 * r["dataset"].map(PRICE_PER_GIB)
    return r


def get_cost(req: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - needs a key and network
    import databento as db
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("DATABENTO_API_KEY is not set; the offline estimate is in sample_extension_estimate.csv")
    client = db.Historical(key)
    out = []
    for (ds, day), g in req.groupby(["dataset", "day"]):
        start = pd.Timestamp(day, tz="UTC") - (pd.Timedelta(hours=2) if ds == "GLBX.MDP3" else pd.Timedelta(0))
        kw = dict(dataset=ds, schema="mbp-1", symbols=list(g["symbol"]), start=start.isoformat(),
                  end=(start + pd.Timedelta(days=1)).isoformat(),
                  stype_in="continuous" if ds == "GLBX.MDP3" else "raw_symbol")
        out.append(dict(dataset=ds, day=day, symbols=len(g), usd=client.metadata.get_cost(**kw),
                        records=client.metadata.get_record_count(**kw)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--get-cost", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    census = pd.read_csv(OUT / "crossed_census_real.csv")
    est = estimate(census)
    est.to_csv(OUT / "sample_extension_estimate.csv", index=False)
    summ = est.groupby("label").agg(symbol_days=("symbol", "size"), records_m=("records_est", lambda x: x.sum() / 1e6),
                                    gib=("bytes_est", lambda x: x.sum() / 2**30), usd=("usd_est", "sum"))
    summ.loc["total"] = summ.sum()
    summ.round(2).to_csv(OUT / "sample_extension_summary.csv")
    (OUT / "calendars_2025_2026.json").write_text(json.dumps(dict(
        fomc_decision_days=FOMC_DECISION_DAYS, nyse_closed=NYSE_CLOSED, nyse_early_close=NYSE_EARLY_CLOSE,
        cme_closed=CME_CLOSED, cme_early_close=CME_EARLY_CLOSE), indent=1))
    roll_calendar().to_csv(OUT / "roll_calendar_2025_2026.csv", index=False)
    print(summ.round(2).to_string())
    if a.get_cost:
        get_cost(est).to_csv(OUT / "sample_extension_get_cost.csv", index=False)


if __name__ == "__main__":
    main()
