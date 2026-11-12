"""Batch data-quality checks for pipeline Parquet.

Streaming checks (record integrity, event-time order, immediate crossed quote)
live in ``alert_system.detectors``. This package is the seven-check batch
detector plus the vendor-flag check. Thresholds are in ``config.py``.
"""

from .config import DetectorConfig, cme_futures_config, equities_config
from .detectors import CHECKS, FLAG_COLUMNS, bar_completeness, run_all, vendor_flags

__all__ = [
    "CHECKS",
    "FLAG_COLUMNS",
    "DetectorConfig",
    "bar_completeness",
    "cme_futures_config",
    "equities_config",
    "run_all",
    "vendor_flags",
]
