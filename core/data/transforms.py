"""Return and resampling transforms."""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """Continuously-compounded log returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1))


def simple_returns(prices: pd.Series) -> pd.Series:
    """Arithmetic returns: P_t/P_{t-1} - 1."""
    return prices.pct_change()


def resample_close(close: pd.Series, frequency: str) -> pd.Series:
    """Resample close prices to the requested frequency.

    Daily → no resampling; Weekly → W-FRI; Monthly → ME.
    """
    rule = {"Daily": None, "Weekly": "W-FRI", "Monthly": "ME"}.get(frequency)
    if rule is None:
        return close
    return close.resample(rule).last().dropna()


def annual_factor(frequency: str) -> float:
    return {"Daily": 252.0, "Weekly": 52.0, "Monthly": 12.0}[frequency]


def prepare_returns(prices: pd.DataFrame, frequency: str = "Daily") -> tuple[pd.Series, pd.Series]:
    """Convenience wrapper: prices → (close_resampled, log_returns_pct)."""
    close = prices["Close"].copy() if "Close" in prices else prices.iloc[:, 0]
    close = resample_close(close, frequency)
    positive = close.where(close > 0)
    returns = 100.0 * np.log(positive / positive.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return close, returns
