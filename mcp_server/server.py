"""Volatility MCP server entry point.

Run with:
    python -m mcp_server.server                                  # stdio
    python -m mcp_server.server --transport sse --port 8765       # SSE
    uv run mcp-server-cli mcp_server.server:mcp

The server registers:
  - 18+ tools  (callable by the LLM agent)
  - 5+ resources (read-only context files)
  - 3 prompts  (workflow templates)

All tool modules delegate business logic to the framework-agnostic
`core` package. The MCP layer is a thin protocol adapter plus a stateful
session manager that drives the feedback loop.
"""
from __future__ import annotations

import argparse
import logging

from mcp.server.fastmcp import FastMCP

from mcp_server.tools import (
    backtest, compare, data, diagnostics, feedback, models,
    optimize, preflight, report, risk, scenario, session,
)
from mcp_server.resources import register_resources
from mcp_server.prompts import register_prompts

logger = logging.getLogger("volatility-mcp")

mcp = FastMCP(
    "volatility-mcp",
    dependencies=[
        "arch>=7.2,<9",
        "yfinance>=0.2.50,<2",
        "pandas>=2.2,<3",
        "numpy>=1.26,<3",
        "scipy>=1.13,<2",
        "statsmodels>=0.14,<1",
        "xlsxwriter>=3.2,<4",
        "pydantic>=2.6,<3",
        "optuna>=3.6,<5",
    ],
)


def register_all() -> None:
    """Register every tool, resource and prompt on the FastMCP instance."""
    # Tools (stateful, delegate to core.*)
    session.register(mcp)
    data.register(mcp)
    preflight.register(mcp)
    diagnostics.register(mcp)
    optimize.register(mcp)
    models.register(mcp)
    compare.register(mcp)
    risk.register(mcp)
    scenario.register(mcp)
    backtest.register(mcp)
    report.register(mcp)
    feedback.register(mcp)

    # Resources (read-only context)
    register_resources(mcp)

    # Prompts (workflow templates)
    register_prompts(mcp)


register_all()


def main() -> None:
    parser = argparse.ArgumentParser(description="Volatility MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting Volatility MCP server (transport=%s)", args.transport)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
