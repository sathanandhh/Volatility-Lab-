"""Data quality checks: gaps, duplicates, dtypes, bad values."""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise an OHLCV frame: dedupe, sort, tz-strip, dtype-coerce."""
    if df is None or df.empty:
        raise ValueError("Empty price frame returned by provider.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if not required:
        raise ValueError("No OHLCV columns found in provider output.")
    out = df[required].copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Close"])
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def detect_gaps(close: pd.Series, max_ok_gap: int = 5) -> dict:
    """Return summary of missing trading days."""
    diffs = close.index.to_series().diff().dt.days
    diffs = diffs.dropna()
    if diffs.empty:
        return {"max_gap_days": 0, "n_gaps_gt_5": 0, "pct_missing": 0.0}
    return {
        "max_gap_days": int(diffs.max()),
        "n_gaps_gt_5": int((diffs > max_ok_gap).sum()),
        "pct_missing": float((diffs > max_ok_gap).mean()),
    }


def detect_duplicates(close: pd.Series) -> int:
    return int(close.index.duplicated().sum())


def detect_bad_values(returns: pd.Series) -> dict:
    """Count zero, inf, NaN — important for WTI April-2020 case."""
    return {
        "n_zero": int((returns == 0).sum()),
        "n_inf": int(np.isinf(returns).sum()),
        "n_nan": int(returns.isna().sum()),
    }
