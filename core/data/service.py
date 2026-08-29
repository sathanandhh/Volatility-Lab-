"""Unified data service that abstracts over multiple market-data providers."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.config import get_settings
from core.data.providers.base import DataProvider
from core.data.transforms import prepare_returns, log_returns, resample_close
from core.data.quality import clean_ohlcv

# ── Market universes ────────────────────────────────────────────────
MARKET_UNIVERSES: dict[str, dict[str, str]] = {
    "World Indices": {
        "NIFTY 50": "^NSEI", "S&P 500": "^GSPC", "NASDAQ Composite": "^IXIC",
        "Dow Jones": "^DJI", "Russell 2000": "^RUT", "FTSE 100": "^FTSE",
        "DAX": "^GDAXI", "CAC 40": "^FCHI", "Nikkei 225": "^N225",
        "Hang Seng": "^HSI", "Shanghai Composite": "000001.SS", "S&P/ASX 200": "^AXJO",
    },
    "Currencies": {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
        "USD/CHF": "CHF=X", "AUD/USD": "AUDUSD=X", "USD/CAD": "CAD=X",
        "USD/INR": "INR=X", "US Dollar Index": "DX-Y.NYB",
    },
    "Commodities": {
        "Gold": "GC=F", "Silver": "SI=F", "WTI Crude": "CL=F",
        "Brent Crude": "BZ=F", "Copper": "HG=F", "Natural Gas": "NG=F",
    },
    "Risk & Rates": {
        "CBOE VIX": "^VIX", "US 10Y Yield": "^TNX", "US 5Y Yield": "^FVX",
        "US 13W Bill": "^IRX", "US Long Treasury ETF": "TLT",
        "Emerging Markets ETF": "EEM",
    },
    "Nifty 50 Stocks": {
        "Reliance Industries": "RELIANCE.NS", "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS", "Infosys": "INFY.NS", "TCS": "TCS.NS",
        "State Bank of India": "SBIN.NS", "Bharti Airtel": "BHARTIARTL.NS",
        "Larsen & Toubro": "LT.NS", "ITC": "ITC.NS", "Tata Motors": "TATAMOTORS.NS",
        "Adani Enterprises": "ADANIENT.NS", "Asian Paints": "ASIANPAINT.NS",
        "Bajaj Finance": "BAJFINANCE.NS", "Maruti Suzuki": "MARUTI.NS",
        "Sun Pharma": "SUNPHARMA.NS", "Tata Steel": "TATASTEEL.NS", "Wipro": "WIPRO.NS",
    },
}


def list_universes() -> dict[str, list[str]]:
    """Return {universe_name: [instrument_names]}."""
    return {u: list(instrs.keys()) for u, instrs in MARKET_UNIVERSES.items()}


def list_assets(universe: str | None = None) -> dict[str, str]:
    """Return {instrument_name: ticker} for one or all universes."""
    if universe is None:
        return {n: t for u in MARKET_UNIVERSES.values() for n, t in u.items()}
    return dict(MARKET_UNIVERSES.get(universe, {}))


class DataService:
    """High-level data access facade.

    Picks the active provider based on Settings.default_provider and
    falls back to yfinance when a primary provider is unavailable.
    """

    def __init__(self, provider: DataProvider | None = None) -> None:
        self._provider = provider or self._build_default_provider()

    # ── Factory ──────────────────────────────────────────────────────
    @classmethod
    def from_env(cls) -> "DataService":
        settings = get_settings()
        provider = cls._build_provider_by_name(settings.default_provider)
        return cls(provider=provider)

    @staticmethod
    def _build_default_provider() -> DataProvider:
        return DataService._build_provider_by_name(get_settings().default_provider)

    @staticmethod
    def _build_provider_by_name(name: str) -> DataProvider:
        from core.data.providers.yfinance_provider import YFinanceProvider
        from core.data.providers.csv_provider import CSVProvider
        name = (name or "yfinance").lower()
        if name == "yfinance":
            return YFinanceProvider()
        if name == "csv":
            return CSVProvider()
        # Optional providers — instantiated only if API keys are set
        settings = get_settings()
        if name == "alpha_vantage" and settings.alpha_vantage_api_key:
            from core.data.providers.alpha_vantage_provider import AlphaVantageProvider
            return AlphaVantageProvider(api_key=settings.alpha_vantage_api_key)
        if name == "polygon" and settings.polygon_api_key:
            from core.data.providers.polygon_provider import PolygonProvider
            return PolygonProvider(api_key=settings.polygon_api_key)
        if name == "kite" and settings.kite_api_key and settings.kite_access_token:
            from core.data.providers.kite_provider import KiteProvider
            return KiteProvider(api_key=settings.kite_api_key,
                                access_token=settings.kite_access_token)
        # Fallback
        return YFinanceProvider()

    # ── Public API ──────────────────────────────────────────────────
    def active_provider_name(self) -> str:
        return type(self._provider).__name__

    def download_prices(self, ticker: str, years: int | None = 20) -> pd.DataFrame:
        raw = self._provider.download(ticker=ticker, years=years)
        return clean_ohlcv(raw)

    def load_csv(self, path: str, date_column: str = "Date",
                 value_column: str = "Close", is_returns: bool = False) -> pd.DataFrame:
        from core.data.providers.csv_provider import CSVProvider
        csv_prov = CSVProvider() if not isinstance(self._provider, CSVProvider) else self._provider
        raw = csv_prov.load(path=path, date_column=date_column,
                            value_column=value_column, is_returns=is_returns)
        return raw

    def prepare_returns(self, prices: pd.DataFrame, frequency: str = "Daily") -> tuple[pd.Series, pd.Series]:
        """Return (close, returns) where returns are percentage log returns."""
        close = prices["Close"].copy()
        close = resample_close(close, frequency)
        positive = close.where(close > 0)
        returns = log_returns(positive) * 100.0  # percent
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        returns.name = "Return (%)"
        return close, returns
