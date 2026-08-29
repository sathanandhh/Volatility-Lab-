"""Session management tools.

Sessions are the backbone of the feedback loop. Every operation tool
takes a `session_id` and reads/writes state on the corresponding Session
object. The Session tracks:
  - current stage (empty → data → preflight → optimize → fit →
    compare → risk → backtest → report)
  - loaded returns
  - preflight GateResult
  - optimization results per input (order, distribution, window, refit)
  - fitted models (dict keyed by model name)
  - backtest results per model
  - decision log (append-only)

Tools consume the store via the module-level `_store` singleton. For
multi-process deployments, replace this with the Redis- or DuckDB-backed
store from `session_store.store`.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from core.feedback.session import Session, SessionStore
from core.feedback.decision_log import DecisionLogEntry

# Module-level in-process store. Replace with persistent backend in
# multi-process deployments.
_store: SessionStore = SessionStore()


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def open_session() -> dict[str, Any]:
        """Open a new analytics session.

        Returns a session_id that must be passed to every subsequent tool
        call. Sessions are stateful: preflight results, fitted models,
        diagnostics and backtests persist across calls and drive the
        feedback loop. Call `feedback.get_next_action` at any time to
        receive a recommendation for the next step.
        """
        sid = _store.open()
        return {
            "session_id": sid,
            "stage": "empty",
            "next_actions": ["data.load_market", "data.load_csv"],
            "note": "Load data before running pre-flight checks.",
        }

    @mcp.tool()
    def close_session(session_id: str) -> dict[str, str]:
        """Close and discard an analytics session.

        Frees all in-memory state. Persisted artifacts (Excel/PDF reports)
        already generated are not affected.
        """
        _store.close(session_id)
        return {"status": "closed", "session_id": session_id}

    @mcp.tool()
    def get_session_state(session_id: str) -> dict[str, Any]:
        """Inspect the current state of an analytics session.

        Returns the current stage, data metadata, fitted model names,
        optimization results, backtest results, decision log, and the
        list of recommended next actions. Useful when resuming an
        analysis or before calling feedback.get_next_action.
        """
        s = _store.get(session_id)
        return s.to_dict()

    @mcp.tool()
    def list_sessions() -> list[dict[str, Any]]:
        """List all active analytics sessions in this process."""
        return [_store.get(sid).summary() for sid in _store.list_ids()]

    @mcp.tool()
    def reset_session(session_id: str, keep_data: bool = True) -> dict[str, Any]:
        """Reset a session to its initial state.

        If `keep_data` is True (default), the loaded returns series is
        preserved so preflight can be re-run after configuration changes.
        Otherwise the session returns to the `empty` stage.
        """
        s = _store.get(session_id)
        s.reset(keep_data=keep_data)
        return s.to_dict()
