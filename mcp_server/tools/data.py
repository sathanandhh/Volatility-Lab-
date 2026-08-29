"""Data loading tools.

Tools accept either:
  - A market ticker (resolved via the configured DataProvider, default
    yfinance)
  - A path/URL to a CSV file with Date + (Close|Returns) columns

After loading, returns are stored on the session. The session stage
advances to `data` and `preflight.run` becomes the recommended next
action.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from core.data.service import DataService, list_universes, list_assets
from core.feedback.session import SessionStore

from .session import _store  # shared in-process store


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def load_market(
        session_id: str,
        ticker: str,
        years: int = 5,
        frequency: str = "Daily",
    ) -> dict[str, Any]:
        """Load daily/weekly/monthly returns for a market ticker.

        Args:
            session_id: Active session id.
            ticker: Yahoo-Finance-style symbol (e.g. ^NSEI, RELIANCE.NS,
                EURUSD=X, GC=F, ^VIX).
            years: Lookback in calendar years (1, 3, 5, 10, 20). Use 20
                for "maximum available".
            frequency: Return frequency — "Daily", "Weekly", or "Monthly".

        Returns observation count, date range, annualized volatility,
        and recommended next actions. Stores the returns series on the
        session for downstream pre-flight and model tools.
        """
        s = _store.get(session_id)
        service = DataService.from_env()
        prices = service.download_prices(ticker, years=years)
        close, returns = service.prepare_returns(prices, frequency=frequency)
        s.returns = returns
        s.metadata = {
            "ticker": ticker.upper(),
            "frequency": frequency,
            "years": years,
            "source": service.active_provider_name(),
        }
        s.advance_stage("data")
        return {
            "n_obs": int(len(returns)),
            "start": str(returns.index[0].date()),
            "end": str(returns.index[-1].date()),
            "annualized_vol_pct": float(
                returns.std() * _annual_factor(frequency) * 100
            ),
            "metadata": s.metadata,
            "next_actions": ["preflight.run", "data.load_market", "data.load_csv"],
        }

    @mcp.tool()
    def load_csv(
        session_id: str,
        path: str,
        date_column: str = "Date",
        value_column: str = "Close",
        is_returns: bool = False,
        frequency: str = "Daily",
    ) -> dict[str, Any]:
        """Load returns or prices from a CSV file.

        Args:
            session_id: Active session id.
            path: Absolute path to CSV file (must be readable by the server process).
            date_column: Column name containing dates.
            value_column: Column name containing Close prices or Returns.
            is_returns: True if `value_column` already contains returns; False
                if prices.
            frequency: "Daily", "Weekly", or "Monthly" — used for annualisation.

        Returns observation count and date range. If `is_returns` is False
        and values are small (median abs < 0.1), they are interpreted as
        decimals and multiplied by 100.
        """
        s = _store.get(session_id)
        service = DataService.from_env()
        prices = service.load_csv(
            path,
            date_column=date_column,
            value_column=value_column,
            is_returns=is_returns,
        )
        close, returns = service.prepare_returns(prices, frequency=frequency)
        s.returns = returns
        s.metadata = {
            "ticker": "CSV_UPLOAD",
            "frequency": frequency,
            "source": f"file:{path}",
        }
        s.advance_stage("data")
        return {
            "n_obs": int(len(returns)),
            "start": str(returns.index[0].date()),
            "end": str(returns.index[-1].date()),
            "metadata": s.metadata,
            "next_actions": ["preflight.run"],
        }

    @mcp.tool()
    def list_universes() -> dict[str, list[str]]:
        """List the supported market universes and their instruments.

        Useful when the agent or user does not know which tickers are
        available. Returns a dict mapping universe name (e.g. "World
        Indices") to a list of instrument names.
        """
        return list_universes()

    @mcp.tool()
    def list_assets(universe: str | None = None) -> dict[str, str]:
        """List assets (name → ticker) for the given universe, or all assets."""
        return list_assets(universe=universe)


def _annual_factor(frequency: str) -> float:
    return {"Daily": 252, "Weekly": 52, "Monthly": 12}[frequency]
