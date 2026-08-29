"""MCP resource registration.

Resources are read-only context files served to the LLM via the MCP
resource protocol. They contain methodology, model catalogs, test
catalogs, and workflow descriptions — content the agent needs to
reason about volatility analytics but should not re-derive on each call.
"""
from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

_RESOURCES_DIR = Path(__file__).parent


def register_resources(mcp: FastMCP) -> None:

    @mcp.resource("volatility://methodology")
    def methodology() -> str:
        """Master methodology document covering the full ARCH → EGARCH workflow."""
        return (_RESOURCES_DIR / "methodology.md").read_text(encoding="utf-8")

    @mcp.resource("volatility://models/catalog")
    def model_catalog() -> str:
        """Catalog of supported volatility models with specifications."""
        return (_RESOURCES_DIR / "model_catalog.md").read_text(encoding="utf-8")

    @mcp.resource("volatility://tests/catalog")
    def test_catalog() -> str:
        """Catalog of statistical tests used in pre-flight and diagnostics."""
        return (_RESOURCES_DIR / "test_catalog.md").read_text(encoding="utf-8")

    @mcp.resource("volatility://distributions/catalog")
    def distribution_catalog() -> str:
        """Catalog of innovation distributions."""
        return (_RESOURCES_DIR / "distribution_catalog.md").read_text(encoding="utf-8")

    # Register each workflow markdown file as its own resource
    workflows_dir = _RESOURCES_DIR / "workflows"
    for wf in sorted(workflows_dir.glob("*.md")):
        _register_workflow(mcp, wf.stem, wf)


def _register_workflow(mcp: FastMCP, name: str, path: Path) -> None:
    @mcp.resource(f"volatility://workflows/{name}")
    def _workflow() -> str:
        return path.read_text(encoding="utf-8")
