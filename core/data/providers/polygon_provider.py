"""Polygon.io provider (requires API key)."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from core.data.providers.base import DataProvider


class PolygonProvider(DataProvider):
    name = "polygon"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def download(self, ticker: str, years: int | None = 20) -> pd.DataFrame:
        # Lightweight REST; uses requests if available
        import requests
        end = date.today()
        start = end - timedelta(days=365 * (years or 5))
        url = "https://api.polygon.io/v2/aggs/ticker"
        r = requests.get(
            f"{url}/{ticker}/range/1/day/{start}/{end}",
            params={"apiKey": self.api_key, "limit": 50000},
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json().get("results", [])
        if not payload:
            raise ValueError(f"Polygon returned no data for {ticker!r}")
        df = pd.DataFrame(payload)
        df["Date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"c": "Close", "o": "Open", "h": "High",
                                "l": "Low", "v": "Volume"})
        return df.set_index("Date")
