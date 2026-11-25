"""Shared helper: timestamps to int64 nanoseconds whatever the pandas datetime unit."""
from __future__ import annotations

import pandas as pd


def ns(values):
    x = pd.to_datetime(values, utc=True)
    if isinstance(x, pd.Series):
        return x.dt.as_unit("ns").astype("int64")
    if isinstance(x, pd.Timestamp):
        return x.as_unit("ns").value
    return x.as_unit("ns").astype("int64")
