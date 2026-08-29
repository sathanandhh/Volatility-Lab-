"""Report generation tools: Excel, PDF, Markdown."""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from core.reporting.excel import build_excel_workbook
from core.reporting.pdf import build_pdf_report
from core.reporting.markdown import build_markdown_report
from core.feedback.session import SessionStore

from .session import _store


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def build_excel(
        session_id: str, output_path: str = "",
    ) -> dict[str, Any]:
        """Build a polished Excel workbook with all session analysis.

        Sheets: Summary, Returns, Conditional Volatility (per fitted
        model), Model Comparison, VaR Forecasts, Backtest Results,
        Diagnostics, Parameters, Methodology, Learning Guide.

        Args:
            output_path: Where to write the file. If empty, server
                picks a temp path.
        """
        s = _store.get(session_id)
        path = build_excel_workbook(s, output_path=output_path)
        s.log_decision("report.excel", "n/a", detail=path)
        return {
            "path": path,
            "next_actions": ["report.pdf", "report.markdown"],
        }

    @mcp.tool()
    def build_pdf(
        session_id: str, output_path: str = "",
    ) -> dict[str, Any]:
        """Build a regulator-ready PDF report with charts and tables.

        Sections: Executive summary, Data description, Pre-flight gate
        results, Fitted models, Diagnostics, Risk metrics, Backtest
        results, Methodology appendix.
        """
        s = _store.get(session_id)
        path = build_pdf_report(s, output_path=output_path)
        s.log_decision("report.pdf", "n/a", detail=path)
        return {"path": path, "next_actions": ["report.markdown"]}

    @mcp.tool()
    def build_markdown(session_id: str) -> dict[str, Any]:
        """Build a Markdown summary of the session — ideal for LLM consumption.

        Returns the Markdown content inline (no file path) so the agent
        can paste it into a chat or document.
        """
        s = _store.get(session_id)
        md = build_markdown_report(s)
        return {
            "content": md,
            "next_actions": ["feedback.get_next_action"],
        }
