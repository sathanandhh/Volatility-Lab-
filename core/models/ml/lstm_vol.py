"""LSTM-based volatility forecast — stub."""
from __future__ import annotations


def fit_lstm(returns, horizon: int = 10, **kwargs):
    raise NotImplementedError(
        "LSTM vol requires PyTorch. Implement when adding ML layer."
    )
