"""Zerodha Kite provider for Indian markets."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from core.data.providers.base import DataProvider


class KiteProvider(DataProvider):
    name = "kite"

    def __init__(self, api_key: str, access_token: str) -> None:
        self.api_key = api_key
        self.access_token = access_token

    def download(self, ticker: str, years: int | None = 20) -> pd.DataFrame:  # pragma: no cover
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=self.api_key, access_token=self.access_token)
        # Map ticker → instrument_token (caller must supply NSE/BSE symbol)
        end = date.today()
        start = end - timedelta(days=365 * (years or 5))
        data = kite.historical_data(
            instrument_token=ticker,
            from_date=start, to_date=end, interval="day",
        )
        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError(f"Kite returned no data for {ticker!r}")
        df["Date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"close": "Close", "open": "Open",
                                "high": "High", "low": "Low", "volume": "Volume"})
        return df.set_index("Date").sort_index()
