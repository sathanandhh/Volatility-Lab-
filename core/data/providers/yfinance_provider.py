"""Yahoo Finance data provider (no API key required)."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from core.data.providers.base import DataProvider


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def download(self, ticker: str, years: int | None = 20) -> pd.DataFrame:
        import yfinance as yf  # local import — heavy dep

        if years is None:
            data = yf.download(ticker, period="max", interval="1d",
                               auto_adjust=True, progress=False, threads=False)
        else:
            start = (pd.Timestamp(date.today())
                     - pd.DateOffset(years=years, days=10)).date()
            end = date.today() + timedelta(days=1)
            data = yf.download(ticker, start=start, end=end, interval="1d",
                               auto_adjust=True, progress=False, threads=False)
        if data is None or data.empty:
            raise ValueError(f"yfinance returned no data for {ticker!r}")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
