"""MCP prompt registration.

Prompts are pre-defined message templates that the agent can invoke to
start a guided workflow, troubleshoot a fit failure, or explain results
to a student. They are stored as Markdown files and surfaced to the LLM
via the MCP prompt protocol.
"""
from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

_PROMPTS_DIR = Path(__file__).parent


def register_prompts(mcp: FastMCP) -> None:

    @mcp.prompt()
    def guided_volatility_analysis(
        asset: str = "RELIANCE.NS", years: int = 5,
    ) -> str:
        """Start a guided end-to-end volatility analysis workflow.

        Produces a prompt that walks the agent through: data load →
        preflight → optimize → fit → compare → risk → backtest → report,
        using feedback.get_next_action at each step.
        """
        return (_PROMPTS_DIR / "guided_volatility_analysis.md").read_text(
            encoding="utf-8"
        ).format(asset=asset, years=years)

    @mcp.prompt()
    def troubleshoot_fit_failure(model: str, error: str) -> str:
        """Generate a troubleshooting prompt for a failed model fit.

        Includes the model name, error message, and a checklist of
        common causes (sample size, distribution, starting values,
        multicollinearity, non-stationarity).
        """
        return (_PROMPTS_DIR / "troubleshoot_fit_failure.md").read_text(
            encoding="utf-8"
        ).format(model=model, error=error)

    @mcp.prompt()
    def explain_results_to_student(
        audience: str = "MBA student",
    ) -> str:
        """Generate a prompt for explaining the session results to a student.

        Loads the session's markdown report and instructs the agent to
        explain it at the requested comprehension level.
        """
        return (_PROMPTS_DIR / "explain_results_to_student.md").read_text(
            encoding="utf-8"
        ).format(audience=audience)
