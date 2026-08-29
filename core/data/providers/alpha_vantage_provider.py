"""Alpha Vantage provider (requires API key)."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from core.data.providers.base import DataProvider


class AlphaVantageProvider(DataProvider):
    name = "alpha_vantage"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def download(self, ticker: str, years: int | None = 20) -> pd.DataFrame:
        from alpha_vantage.timeseries import TimeSeries  # local import
        ts = TimeSeries(key=self.api_key, output_format="pandas")
        data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize="full")
        data = data.rename(columns={"5. adjusted close": "Close"})
        data.index = pd.to_datetime(data.index)
        if years is not None:
            cutoff = pd.Timestamp(date.today()) - pd.DateOffset(years=years)
            data = data[data.index >= cutoff]
        return data.sort_index()
