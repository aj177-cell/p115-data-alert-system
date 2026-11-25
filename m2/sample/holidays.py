"""Opt-in holiday calendar sidecar for the v3 checks, Bayesian gaps and SPC buckets.

Nothing here changes a v3 preset. ``holiday_config(cfg, venue, root)`` returns a *copy* of a v3
DetectorConfig whose session calendars (missing, stale, extreme, spreads) know the 2025-2026
exchange holidays and early closes; ``holiday_calendar(venue, root)`` returns just the calendar.

Sources (opened 2026-11-22): NYSE closures and 13:00 ET early closes (exchange_calendars XNYS,
checked against the NYSE list); CME Globex holiday halts (CME Group holiday schedule, broker
notices of it): equity index halts 12:00 CT and energy 13:30 CT on Monday holidays and Thanksgiving,
12:15 CT (equity) / 12:45 CT (energy) on the day after Thanksgiving and Christmas Eve, 12:00 CT for
both on Independence Day and Juneteenth. Every halt used in the sample was checked against the last
record in the data (m2/sample/results/holiday_halt_check.csv).

Caveat: v3 ``SessionCalendar.early_closes`` truncates *every* window of that local date. For the CME
calendar that also removes the 17:00 CT reopen of the holiday evening (next trade date). The sample
windows end at 16:00/17:00 CT, before that reopen, so results here are unaffected; a long multi-day
window would need a per-date halt list instead (not built).
"""
from __future__ import annotations

import copy

NYSE_CLOSED = ["2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
               "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25", "2026-01-01", "2026-01-19", "2026-02-16",
               "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"]
NYSE_EARLY = {d: "13:00" for d in ["2025-07-03", "2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24"]}
CME_CLOSED = ["2025-01-01", "2025-04-18", "2025-12-25", "2026-01-01", "2026-04-03", "2026-12-25"]
_MONDAY_LIKE = ["2025-01-20", "2025-02-17", "2025-05-26", "2025-09-01", "2025-11-27",
                "2026-01-19", "2026-02-16", "2026-05-25", "2026-09-07", "2026-11-26"]
_EVE = ["2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24"]
_FULL = ["2025-06-19", "2025-07-04", "2026-06-19", "2026-07-03"]
CME_HALT = {
    "ES": {**{d: "12:00" for d in _MONDAY_LIKE}, **{d: "12:15" for d in _EVE}, **{d: "12:00" for d in _FULL}},
    "CL": {**{d: "13:30" for d in _MONDAY_LIKE}, **{d: "12:45" for d in _EVE}, **{d: "12:00" for d in _FULL}},
}


def root_of(symbol: str) -> str:
    return symbol.split(".")[0]


def holiday_calendar(venue: str, root: str | None = None, v3=None):
    if v3 is None:
        from integration.combine import load_v3
        v3 = load_v3()
    if venue == "XNAS":
        cal = copy.deepcopy(v3.config.equities_config().missing.calendar)
        cal.holidays = sorted(set(cal.holidays) | set(NYSE_CLOSED))
        cal.early_closes = {**cal.early_closes, **NYSE_EARLY}
        return cal
    cal = copy.deepcopy(v3.config.cme_calendar())
    cal.holidays = sorted(set(cal.holidays) | set(CME_CLOSED))
    cal.early_closes = {**cal.early_closes, **CME_HALT.get(root or "ES", {})}
    return cal


def holiday_config(cfg, venue: str, root: str | None = None, v3=None):
    """Copy of ``cfg`` with the holiday calendar on every calendar-aware check."""
    out = copy.deepcopy(cfg)
    cal = holiday_calendar(venue, root, v3)
    for section in ("missing", "stale", "extreme", "spreads"):
        if getattr(getattr(out, section), "calendar", None) is not None:
            getattr(out, section).calendar = copy.deepcopy(cal)
    return out


def is_special(venue: str, day: str, root: str | None = None) -> str | None:
    if venue == "XNAS":
        return "closed" if day in NYSE_CLOSED else ("early_close" if day in NYSE_EARLY else None)
    if day in CME_CLOSED:
        return "closed"
    return "halt" if day in CME_HALT.get(root or "ES", {}) else None
