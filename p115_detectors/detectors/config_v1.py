"""Tunable thresholds for the p115 market-data quality detectors.

Every number here is an *assumption* that the brief asks the team to
sensitivity-test. Each field documents what it means and why the default was
chosen. Defaults target liquid US equities on XNAS.ITCH mbp-1; use
:func:`cme_futures_config` for GLBX.MDP3.

Times are in seconds unless the name says ``_ns``. Prices are in the
instrument's price units (Databento fixed-point / 1e9).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any

# --------------------------------------------------------------------------
# Databento constants (documented in "Common fields, enums and types")
# --------------------------------------------------------------------------
F_LAST = 128          #: last record in the venue event; book is consistent only here
F_TOB = 64
F_SNAPSHOT = 32       #: record is part of a synthetic snapshot, not a live event
F_MBP = 16
F_BAD_TS_RECV = 8     #: ts_recv is inaccurate (e.g. snapshot, clock issue)
F_MAYBE_BAD_BOOK = 4  #: unrecoverable gap detected in the channel
UNDEF_PRICE = 2**63 - 1          #: INT64_MAX sentinel for "no price"
UNDEF_TIMESTAMP = 2**64 - 1      #: UINT64_MAX sentinel for "no timestamp"
FIXED_PRICE_SCALE = 1e-9         #: raw int64 price * 1e-9 = price


@dataclass
class SessionWindow:
    """One daily open window in local time. ``end='24:00'`` means midnight."""

    start: str
    end: str
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)  # Monday=0


@dataclass
class SessionCalendar:
    """Exchange session calendar used by gap and stale checks.

    Default: US equity regular trading hours 09:30-16:00 America/New_York,
    Monday-Friday. Holidays and early closes must be supplied (we do not ship
    an exchange calendar; plug in ``pandas_market_calendars`` later).
    The 2025-11-28 (day after Thanksgiving) 13:00 ET early close is included
    as an example because it is in the pilot bundle.
    """

    tz: str = "America/New_York"
    windows: list[SessionWindow] = field(default_factory=lambda: [SessionWindow("09:30", "16:00")])
    holidays: list[str] = field(default_factory=list)
    early_closes: dict[str, str] = field(default_factory=lambda: {"2025-11-28": "13:00"})


@dataclass
class CrossedBooksConfig:
    min_duration_s: float = 0.001
    """tau_c. A cross/lock must persist (until the next F_LAST book update) at
    least this long. 1 ms: below that a cross is plausibly an update-ordering
    artifact; sweep {0, 1 ms, 1 s} as in the prep doc."""
    report_locked: bool = True
    """Locked (bid == ask) is reported as its own subtype with lower severity."""
    expected_windows_local: list[tuple[str, str, str]] = field(default_factory=list)
    """Daily (tz, 'HH:MM', 'HH:MM') windows where crosses are expected and are
    downgraded to severity 'info'. Empty for equities; CME preset masks the
    daily maintenance break / pre-open."""
    expected_windows_utc: list[tuple[str, str]] = field(default_factory=list)
    """Explicit absolute (start, end) UTC windows to mask (e.g. a known auction)."""
    drop_expected: bool = False
    """If True, masked crosses are dropped instead of reported as 'info'."""


@dataclass
class ExtremePricesConfig:
    price_source: str = "both"
    """'mid' (F_LAST book mid), 'trade' (action == 'T' prints) or 'both'."""
    window: int = 500
    """W: number of prior observations in the rolling median/MAD. ~1 minute of
    a liquid name at XNAS rates; long enough for a stable MAD, short enough to
    follow intraday volatility regimes."""
    min_periods: int = 50
    """No z-score before this many prior returns (warm-up)."""
    k: float = 8.0
    """k_p: flag |z| > k. 8 is above the prep doc's {4,5,6} grid because
    tick-discrete returns are heavy-tailed; on synthetic data k=6 already
    gives clean results; sweep 4..12."""
    mad_floor_ticks: float = 1.0
    """epsilon_min = mad_floor_ticks * tick / price, so a quiet period (MAD=0)
    cannot make a one-tick move look extreme."""
    mad_floor_log: float = 1e-4
    """Fallback floor in log units (1 bp) when the tick size is unknown."""
    allow_negative: bool = False
    """Calendar spreads can trade at negative prices. When True the score uses
    price differences instead of log returns (log is undefined for p <= 0)."""
    static_bands: dict[Any, tuple[float, float]] = field(default_factory=dict)
    """Optional hard bands {symbol or instrument_id: (low, high)}, e.g. from
    definition high/low_limit_price or GLBX statistics price limits."""
    exclude_sides: tuple[str, ...] = ("N",)
    """Trade prints with side 'N' (auction/cross, no aggressor) are excluded."""


@dataclass
class TimestampConfig:
    event_tolerance_ns: int = 0
    """epsilon_e: ts_event may go back by this much before it is reported. 0
    because any regression is informative; severity is 'low' since some
    venues legitimately stamp ts_event at unsynchronized gateways."""
    clock_tolerance_ns: int = 1_000
    """epsilon_clk: ts_recv < ts_event - 1 us is a causality violation. 1 us
    is the documented PTP accuracy of Databento's capture clock."""
    latency_k: float = 10.0
    """k_L: latency L = ts_recv - ts_event is an outlier above
    median + k_L * 1.4826 * MAD per key. 10 because latency is right-skewed."""
    latency_min_excess_ns: int = 1_000_000
    """And L must exceed the median by at least 1 ms, so a tiny MAD on a
    colocated feed does not flag microsecond jitter."""
    flag_saturated_ts_in_delta: bool = True
    """ts_in_delta is int32 and clamps at +/-(2^31-1) ns (~2.15 s)."""


@dataclass
class MissingIntervalsConfig:
    calendar: SessionCalendar = field(default_factory=SessionCalendar)
    max_gap_s: float = 60.0
    """Static expected-gap threshold (in-session seconds). Used alone when
    adaptive=False and as a cap otherwise: a liquid name silent for a minute in
    RTH is suspicious regardless of rate."""
    adaptive: bool = True
    """Use the Poisson rule g > -ln(alpha) / lambda_hat (trailing rate)."""
    alpha: float = 1e-6
    """Tail probability. -ln(1e-6) = 13.8 inter-arrival means. Lower than the
    prep doc's 1e-4 because with ~1e6 gaps/day 1e-4 would yield ~100 false
    alarms per key-day even under a perfect Poisson model."""
    rate_window: int = 500
    """Number of trailing arrivals used to estimate lambda_hat."""
    min_gap_s: float = 1.0
    """Floor on the adaptive threshold: sub-second silences are never flagged."""


@dataclass
class StaleQuotesConfig:
    calendar: SessionCalendar = field(default_factory=SessionCalendar)
    min_age_s: float = 5.0
    """theta: a BBO unchanged for >= 5 s in RTH. Liquid names update many
    times a second; illiquid names will need the adaptive option."""
    adaptive_quantile: float | None = None
    """If set (e.g. 0.99), theta = max(min_age_s, multiplier * q-quantile of the
    key's own quote lifetimes)."""
    adaptive_multiplier: float = 3.0
    min_trades: int = 3
    """n0: at least this many own trades printed while the quote was frozen
    (trades normally consume size, so the BBO should have changed)."""
    reference: dict[str, Any] | None = None
    """Reference instrument, e.g. {'symbol': 'SPY'} or {'instrument_id': 15144}.
    If the reference mid moved >= ref_move_bps the quote is 'stale'."""
    ref_move_bps: float = 5.0
    """kappa expressed in basis points of the reference mid."""
    report_quiet: bool = False
    """Also emit 'quiet' (old but market did not move) intervals at severity info."""


@dataclass
class UnusualSpreadsConfig:
    tick_size: dict[Any, float] = field(default_factory=dict)
    """{symbol or instrument_id: min_price_increment}. Missing keys are
    inferred as the smallest positive difference of observed bid prices."""
    method: str = "rolling_quantile"
    """'rolling_quantile' or 'time_of_day' (robust z of log spread per bucket)."""
    window: str = "15min"
    """Trailing time window for the rolling quantile baseline."""
    quantile: float = 0.95
    multiplier: float = 3.0
    """Flag spread_ticks > max(multiplier * q95, q95 + min_excess_ticks)."""
    min_excess_ticks: float = 3.0
    tod_bucket: str = "30min"
    tz: str = "America/New_York"
    """Local zone for time-of-day buckets."""
    k: float = 6.0
    """k_s for the time-of-day robust z of log spread."""
    min_duration_s: float = 0.5
    """d: a wide spread must persist 0.5 s; one-record blips are common."""
    min_periods: int = 100


@dataclass
class SequenceConfig:
    per_instrument: bool = False
    """False: one domain per publisher_id (sequence is believed to be a venue/
    channel counter). True: per (publisher_id, instrument_id)."""
    allow_equal: bool = True
    """delta == 0 is OK: records normalized from one venue message share it."""
    min_gap: int = 1
    """Report gaps with at least this many missing messages."""


@dataclass
class DetectorConfig:
    crossed: CrossedBooksConfig = field(default_factory=CrossedBooksConfig)
    extreme: ExtremePricesConfig = field(default_factory=ExtremePricesConfig)
    timestamps: TimestampConfig = field(default_factory=TimestampConfig)
    missing: MissingIntervalsConfig = field(default_factory=MissingIntervalsConfig)
    stale: StaleQuotesConfig = field(default_factory=StaleQuotesConfig)
    spreads: UnusualSpreadsConfig = field(default_factory=UnusualSpreadsConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    report_vendor_flags: bool = True
    """Emit F_BAD_TS_RECV / F_MAYBE_BAD_BOOK records as 'vendor_flags'."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def equities_config(**overrides: Any) -> DetectorConfig:
    """Default config (XNAS.ITCH-style equities, RTH 09:30-16:00 ET)."""
    return replace(DetectorConfig(), **overrides)


def cme_calendar() -> SessionCalendar:
    """CME Globex: Sun 17:00 CT to Fri 16:00 CT with the daily 16:00-17:00 CT
    (17:00-18:00 ET) maintenance break."""
    return SessionCalendar(
        tz="America/Chicago",
        windows=[SessionWindow("00:00", "16:00", (0, 1, 2, 3, 4)),
                 SessionWindow("17:00", "24:00", (6, 0, 1, 2, 3))],
        early_closes={},
    )


def cme_futures_config() -> DetectorConfig:
    """Preset for GLBX.MDP3: CME session calendar and the daily break /
    pre-open masked for crossed books (crosses are normal there)."""
    cfg = DetectorConfig()
    cfg.missing.calendar = cme_calendar()
    cfg.stale.calendar = cme_calendar()
    cfg.crossed.expected_windows_local = [("America/Chicago", "16:00", "17:00")]
    return cfg
