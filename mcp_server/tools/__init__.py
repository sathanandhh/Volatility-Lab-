"""MCP tool modules for the Volatility server.

Each module exposes a `register(mcp: FastMCP) -> None` function that
registers its tools on the shared FastMCP instance. Tools share an
in-process SessionStore (defined in `session.py`) for stateful feedback.
"""
from __future__ import annotations

from . import (
    backtest, compare, data, diagnostics, feedback, models,
    optimize, preflight, report, risk, scenario, session,
)

__all__ = [
    "backtest", "compare", "data", "diagnostics", "feedback", "models",
    "optimize", "preflight", "report", "risk", "scenario", "session",
]
