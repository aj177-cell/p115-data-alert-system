"""Stratified sample metadata: the 95 pilot symbol-days plus the representative-sample extension.

Tags are a list per symbol-day; the stored pilot results keep their original single ``day_type``
(m2.extract.DAY_TYPE) and are not edited. The labelling fix lives here only:
2026-09-16 is tagged both ``vix_high`` and ``fomc`` (FOMC decision day of the 15-16 Sep 2026 meeting).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from m2.extract import DAY_TYPE, EQ_SYMBOLS, tasks as pilot_tasks

SAMPLE_ROOT = Path("/workspace/p115/data_sample")
PILOT_ROOT = Path("/workspace/p115/data_pipeline")

FOMC = {"2025-12-10", "2026-01-28", "2026-09-16", "2026-10-28"}
ES_ROLL = {"2025-12-11": "roll_date", "2025-12-19": "expiry", "2025-12-22": "c0_switch",
           "2026-09-10": "roll_date", "2026-09-16": "roll_week", "2026-09-18": "expiry", "2026-09-21": "c0_switch"}
CL_EXPIRY = {"2026-09-22": "last_trade", "2026-09-23": "c0_switch", "2026-10-20": "last_trade", "2026-10-21": "c0_switch"}
HOLIDAY = {"2025-12-24": "early_close", "2025-12-26": "post_holiday", "2026-01-19": "nyse_closed_cme_halt",
           "2026-07-03": "nyse_closed_cme_halt", "2025-11-28": "early_close"}

XNAS_DAYS = ["2025-12-10", "2025-12-24", "2025-12-26", "2026-01-28", "2026-10-28"]
GLBX = {
    "2025-12-10": ["ES.c.0", "CL.c.0"], "2026-01-28": ["ES.c.0", "CL.c.0"], "2026-10-28": ["ES.c.0", "CL.c.0"],
    "2025-12-11": ["ES.c.0", "ES.c.1"], "2025-12-19": ["ES.c.0", "ES.c.1"], "2025-12-22": ["ES.c.0"],
    "2026-09-10": ["ES.c.0", "ES.c.1"], "2026-09-16": ["ES.c.0", "ES.c.1", "CL.c.0"],
    "2026-09-18": ["ES.c.0", "ES.c.1"], "2026-09-21": ["ES.c.0"],
    "2026-09-22": ["CL.c.0", "CL.c.1"], "2026-09-23": ["CL.c.0"], "2026-10-20": ["CL.c.0", "CL.c.1"], "2026-10-21": ["CL.c.0"],
    "2025-12-24": ["ES.c.0", "CL.c.0"], "2025-12-26": ["ES.c.0", "CL.c.0"], "2026-01-19": ["ES.c.0", "CL.c.0"],
    "2026-07-03": ["ES.c.0", "CL.c.0"], "2026-10-08": ["ES.c.1", "CL.c.1"],
}


def tags_for(venue: str, symbol: str, day: str) -> list[str]:
    t = []
    base = DAY_TYPE.get(day)
    if base:
        t.append(base)
    if day in FOMC:
        t.append("fomc")
    root = symbol.split(".")[0]
    if venue == "GLBX" and root == "ES" and day in ES_ROLL:
        t.append(f"es_roll:{ES_ROLL[day]}")
    if venue == "GLBX" and root == "CL" and day in CL_EXPIRY:
        t.append(f"cl_expiry:{CL_EXPIRY[day]}")
    if day in HOLIDAY:
        t.append(f"holiday:{HOLIDAY[day]}")
    if symbol.endswith(".c.1"):
        t.append("back_month")
    return t or ["other"]


def stratum_of(tags: list[str]) -> str:
    """One reporting stratum per symbol-day (priority: holiday > roll/expiry > FOMC > normal)."""
    if "back_month" in tags and "normal_week" in tags:
        return "normal_week_c1"
    for prefix, name in (("holiday", "holiday"), ("half_day", "holiday"), ("es_roll", "es_roll"), ("cl_expiry", "cl_expiry"),
                         ("fomc", "fomc"), ("normal_week", "normal_week"), ("cpi", "cpi"), ("vix_high", "vix_high")):
        if any(t.startswith(prefix) for t in tags):
            return name
    return tags[0]


def sample_tasks():
    """(venue, piece, status_piece, definition_piece, symbol, day, start, end) for the new symbol-days."""
    T = []
    for d in XNAS_DAYS:
        s = pd.Timestamp(d, tz="UTC")
        for sym in EQ_SYMBOLS:
            T.append(("XNAS", f"S_mbp-1_X_{d}", f"S_status_X_{d}", f"S_definition_X_{d}", sym, d, s, s + pd.Timedelta(days=1)))
    for d, syms in GLBX.items():
        s = pd.Timestamp(d, tz="UTC") - pd.Timedelta(hours=2)
        for sym in syms:
            T.append(("GLBX", f"S_mbp-1_G_{d}", f"S_status_G_{d}", f"S_definition_G_{d}", sym, d, s, s + pd.Timedelta(days=1)))
    return T


def normal_week_tasks():
    """Pilot normal-week symbol-days in the same 8-tuple layout (data under PILOT_ROOT)."""
    T = []
    for venue, piece, symbol, day, start, end in pilot_tasks():
        if DAY_TYPE.get(day) != "normal_week":
            continue
        T.append((venue, piece, piece.replace("mbp-1", "status"), piece.replace("mbp-1", "definition"), symbol, day, start, end))
    return T


def strata_table() -> pd.DataFrame:
    rows = []
    for venue, piece, symbol, day, *_ in pilot_tasks():
        tg = tags_for(venue, symbol, day)
        rows.append(dict(source="pilot", venue=venue, symbol=symbol, day=day, stored_day_type=DAY_TYPE.get(day, "other"),
                         tags="|".join(tg), stratum=stratum_of(tg), piece=piece))
    for venue, piece, _, _, symbol, day, *_ in sample_tasks():
        tg = tags_for(venue, symbol, day)
        rows.append(dict(source="sample", venue=venue, symbol=symbol, day=day, stored_day_type="",
                         tags="|".join(tg), stratum=stratum_of(tg), piece=piece))
    return pd.DataFrame(rows)
