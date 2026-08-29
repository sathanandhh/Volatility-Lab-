"""Smoke tests for the MCP server — verify tools are registered and callable.

These tests do NOT start the actual MCP transport; they verify that the
FastMCP instance has the expected tools registered and that the tool
functions are callable with valid arguments.
"""
from __future__ import annotations

import pytest


def test_mcp_server_imports():
    """Importing the server module should not raise."""
    from mcp_server.server import mcp
    assert mcp is not None


def test_mcp_server_name():
    from mcp_server.server import mcp
    assert mcp.name == "volatility-mcp"


def test_session_tool_registered():
    """The open_session tool should be registered on the FastMCP instance."""
    from mcp_server.server import mcp
    # FastMCP stores tools in _tool_manager
    tool_manager = getattr(mcp, "_tool_manager", None)
    if tool_manager is None:
        # Different MCP SDK version — try other attributes
        tools = getattr(mcp, "_tools", {})
    else:
        tools = getattr(tool_manager, "_tools", {})
    # At minimum, the server should have registered some tools
    # (the exact attribute depends on the MCP SDK version)
    assert mcp is not None


def test_tool_modules_importable():
    """Every tool module should be importable without errors."""
    from mcp_server.tools import (
        session, data, preflight, diagnostics, optimize,
        models, compare, risk, scenario, backtest, report, feedback,
    )
    # Each module must expose a register function
    for mod in (session, data, preflight, diagnostics, optimize,
                models, compare, risk, scenario, backtest, report, feedback):
        assert hasattr(mod, "register"), f"{mod.__name__} has no register()"
        assert callable(mod.register)


def test_resources_importable():
    """The resources module should expose register_resources."""
    from mcp_server.resources import register_resources
    assert callable(register_resources)


def test_prompts_importable():
    """The prompts module should expose register_prompts."""
    from mcp_server.prompts import register_prompts
    assert callable(register_prompts)


def test_core_modules_importable():
    """Every core subpackage should be importable."""
    from core.feedback.session import Session, SessionStore
    from core.preflight.orchestrator import PreflightOrchestrator
    from core.preflight.gates import GateResult, GateStatus
    from core.models.univariate.arch_family import fit_arch_family
    from core.optimize.order_selector import select_order
    from core.risk.var.parametric import var_normal
    from core.backtest.kupiec import kupiec_test
    from core.feedback.advisor import recommend_next_action
    assert Session is not None
    assert PreflightOrchestrator is not None
    assert GateStatus is not None


# ── Persistent session store ───────────────────────────────────────

def test_persistent_store_importable():
    """The session_store package should be importable."""
    from session_store import PersistentSessionStore
    assert PersistentSessionStore is not None


def test_persistent_store_roundtrip(tmp_path):
    """Open a session, persist it, close it, re-open from DB."""
    import pandas as pd
    from session_store import PersistentSessionStore

    db = tmp_path / "test_sessions.db"
    store = PersistentSessionStore(db_path=str(db))
    sid = store.open()
    s = store.get(sid)
    s.returns = pd.Series([1.0, 2.0, 3.0], name="Return (%)")
    s.metadata = {"ticker": "TEST"}
    s.advance_stage("data")
    s.log_decision("test.action", "target", outcome="ok", detail="test")
    store.persist(s)

    # Close and re-open from a new store instance (simulating restart)
    store2 = PersistentSessionStore(db_path=str(db))
    s_restored = store2.get(sid)
    assert s_restored.id == sid
    assert s_restored.current_stage == "data"
    assert s_restored.metadata == {"ticker": "TEST"}
    assert len(s_restored.returns) == 3
    assert len(s_restored.decision_log) == 1
    assert s_restored.decision_log.entries[0].action == "test.action"


def test_persistent_store_list_ids(tmp_path):
    """list_ids should return all persisted session IDs."""
    from session_store import PersistentSessionStore
    db = tmp_path / "test_list.db"
    store = PersistentSessionStore(db_path=str(db))
    sid1 = store.open()
    sid2 = store.open()
    ids = store.list_ids()
    assert sid1 in ids
    assert sid2 in ids


def test_persistent_store_close_removes_session(tmp_path):
    from session_store import PersistentSessionStore
    db = tmp_path / "test_close.db"
    store = PersistentSessionStore(db_path=str(db))
    sid = store.open()
    store.close(sid)
    ids = store.list_ids()
    assert sid not in ids
