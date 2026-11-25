"""Stress test: both systems on the v3 synthetic stream with labeled, injected faults.

The stream and the seven fault types come unchanged from p115_detectors/detectors/synthetic.py.
The adapter adds one more family, ``integrity`` (a zero bid or an undefined ask on a nonempty
side), because the v3 injector has no fault that the Milestone 1 integrity rule targets.

Matching follows the v3 evaluation: a finding matches a label if it is for the same publisher
(and instrument, unless the label is publisher-level) and its ts_recv lies in
[start - tol, end + tol]. Findings at severity 'info' are ignored. The engine is scored with
cooldown off, because cooldown only suppresses repeats of the same rule and scope.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from .combine import BATCH, ENGINE, F_LAST, engine_table, load_v3, run_engine, unify, v3_table

NS = 1_000_000_000
SYNTHETIC_META = {"source": "synthetic", "dataset": "XNAS.ITCH", "schema": "mbp-1"}
NO_COOLDOWN = {"cooldown_events": 0, "cooldown_seconds": 0}
ENGINE_FAMILY = {"crossed_quote": "crossed_books", "time_order": "timestamp_inversions", "integrity": "integrity"}
FAMILIES = ["crossed_books", "extreme_prices", "timestamp_inversions", "missing_intervals", "stale_quotes",
            "unusual_spreads", "sequence_discontinuities", "integrity"]


def v3_eval_config(v3):
    """Same as p115_detectors/detectors/evaluate.py::base_config()."""
    config = v3.config.DetectorConfig()
    config.stale.reference = {"symbol": "REF"}
    return config


def inject_integrity_faults(raw: pd.DataFrame, labels: pd.DataFrame, n_per_type: int = 3, seed: int = 15,
                            clearance_s: float = 3.0, warmup_s: float = 180.0, undef_price: int = 2**63 - 1):
    """Alternate ``zero_bid`` (bid 0 with bid size > 0) and ``undefined_ask`` (UNDEF_PRICE ask with
    ask size > 0) on single F_LAST records, at least ``clearance_s`` from every existing label
    and from each other. Works on the raw frame of synthetic.generate_mbp1/inject_faults."""
    rng = np.random.default_rng(seed)
    frame = raw.copy()
    times = frame["ts_recv"].to_numpy()
    free = ((frame["flags"].to_numpy() & F_LAST) != 0) & (times >= times.min() + int(warmup_s * NS))
    for label in labels.itertuples(index=False):
        free &= ~((times >= label.start.value - clearance_s * NS) & (times <= label.end.value + clearance_s * NS))
    chosen: list[int] = []
    for row in rng.permutation(np.flatnonzero(free)):
        if all(abs(int(times[row]) - int(times[other])) > 2 * clearance_s * NS for other in chosen):
            chosen.append(int(row))
        if len(chosen) == 2 * n_per_type:
            break
    added = []
    for k, row in enumerate(sorted(chosen)):
        kind = "zero_bid" if k % 2 == 0 else "undefined_ask"
        frame.loc[row, "bid_px_00" if kind == "zero_bid" else "ask_px_00"] = 0 if kind == "zero_bid" else undef_price
        stamp = pd.Timestamp(int(times[row]), unit="ns", tz="UTC")
        added.append(dict(check="integrity", subtype=kind, publisher_id=int(frame.loc[row, "publisher_id"]),
                          instrument_id=int(frame.loc[row, "instrument_id"]), start=stamp, end=stamp, magnitude=np.nan))
    labels = pd.concat([labels, pd.DataFrame(added)], ignore_index=True)
    return frame, labels.sort_values("start", kind="stable").reset_index(drop=True)


def family(source: str, rule: str) -> str | None:
    return ENGINE_FAMILY.get(rule) if source == ENGINE else rule


def match_matrix(findings: pd.DataFrame, labels: pd.DataFrame, tolerance_s: float = 1.0) -> np.ndarray:
    """hits[i, j] is True if finding i falls on label j (publisher, instrument, time window)."""
    times = findings["ts_recv"].fillna(findings["ts_event"]).dt.as_unit("ns").astype("int64").to_numpy()
    publisher = pd.to_numeric(findings["publisher_id"], errors="coerce").to_numpy(dtype="float64")
    instrument = pd.to_numeric(findings["instrument_id"], errors="coerce").to_numpy(dtype="float64")
    tol = int(tolerance_s * NS)
    hits = np.zeros((len(findings), len(labels)), bool)
    for j, label in enumerate(labels.itertuples(index=False)):
        hit = (publisher == label.publisher_id) & (times >= label.start.value - tol) & (times <= label.end.value + tol)
        if label.instrument_id is not None and not pd.isna(label.instrument_id):
            hit &= instrument == float(label.instrument_id)
        hits[:, j] = hit
    return hits


def score(unified: pd.DataFrame, labels: pd.DataFrame, tolerance_s: float = 1.0):
    """Label recall per fault family and source, and a per-rule breakdown of the findings."""
    findings = unified[unified["level"] > 0].reset_index(drop=True)
    own_family = np.array([family(s, r) for s, r in zip(findings["source"], findings["rule"])], dtype=object)
    hits = match_matrix(findings, labels, tolerance_s)
    same = hits & (own_family[:, None] == labels["check"].to_numpy()[None, :])
    covered = {ENGINE: {ENGINE_FAMILY[rule] for rule in ENGINE_FAMILY}, BATCH: set(FAMILIES) - {"integrity"}}
    rows = []
    for fam in FAMILIES:
        cols = labels["check"].to_numpy() == fam
        row = {"fault_family": fam, "labels": int(cols.sum())}
        detected_any = np.zeros(int(cols.sum()), bool)
        for source in (ENGINE, BATCH):
            mine = same[(findings["source"] == source).to_numpy()][:, cols].any(axis=0)
            row[source] = int(mine.sum()) if fam in covered[source] else None
            detected_any |= mine
        row["either"] = int(detected_any.sum())
        rows.append(row)
    detection = pd.DataFrame(rows).set_index("fault_family")
    on_label = same.any(axis=1)
    collateral = hits.any(axis=1) & ~on_label
    breakdown = (findings.assign(on_label=on_label, collateral=collateral, unlabeled=~hits.any(axis=1),
                                 no_f_last=findings["f_last"].eq(False))
                 .groupby(["source", "rule"])[["on_label", "collateral", "unlabeled", "no_f_last"]].sum()
                 .astype(int))
    breakdown.insert(0, "findings", breakdown[["on_label", "collateral", "unlabeled"]].sum(axis=1))
    return detection, breakdown


def run_both(frame: pd.DataFrame, v3, engine_config: dict | None = NO_COOLDOWN) -> pd.DataFrame:
    alerts, context = run_engine(frame, SYNTHETIC_META, engine_config)
    flags, _ = v3.detectors.run_all(frame, v3_eval_config(v3))
    return unify(engine_table(alerts, context), v3_table(flags))


def stress_test(seed: int = 7, duration_s: float = 3600.0, n_per_type: int = 5, integrity_per_type: int = 3,
                tolerance_s: float = 1.0, engine_config: dict | None = NO_COOLDOWN) -> SimpleNamespace:
    """Defaults reproduce the v3 synthetic evaluation (1 h, 2 instruments, 5 faults per type)."""
    v3 = load_v3()
    raw = v3.synthetic.generate_mbp1(duration_s=duration_s, seed=seed)
    faulted, labels = v3.synthetic.inject_faults(raw, n_per_type=n_per_type, seed=seed + 4)
    if integrity_per_type:
        faulted, labels = inject_integrity_faults(faulted, labels, integrity_per_type, seed=seed + 8,
                                                  undef_price=v3.config.UNDEF_PRICE)
    unified = run_both(v3.synthetic.to_databento_df(faulted), v3, engine_config)
    clean = run_both(v3.synthetic.to_databento_df(raw), v3, engine_config)
    detection, breakdown = score(unified, labels, tolerance_s)
    clean_counts = clean[clean["level"] > 0].groupby(["source", "rule"]).size().rename("clean_stream_findings")
    clean_no_f_last = (clean[(clean["level"] > 0) & clean["f_last"].eq(False)]
                       .groupby(["source", "rule"]).size().rename("clean_without_f_last"))
    breakdown = breakdown.join(clean_counts, how="outer").join(clean_no_f_last, how="left").fillna(0).astype(int)
    return SimpleNamespace(records=len(faulted), labels=labels, unified=unified, clean=clean,
                           detection=detection, breakdown=breakdown)
