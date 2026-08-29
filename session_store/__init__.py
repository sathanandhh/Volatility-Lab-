"""Persistent session storage backend for the Volatility MCP server.

By default, sessions live in-process (see core.feedback.session.SessionStore).
For multi-process deployments (SSE / streamable-http transport with multiple
workers), swap in PersistentSessionStore from this module:

    from session_store import PersistentSessionStore
    store = PersistentSessionStore(db_path="/data/volmcp.db")

The persistent store implements the same interface as the in-memory store
plus a `persist(session)` method that writes through to SQLite.
"""
from __future__ import annotations

from session_store.store import PersistentSessionStore
from session_store.schemas import SessionRow, DecisionRow

__all__ = ["PersistentSessionStore", "SessionRow", "DecisionRow"]
