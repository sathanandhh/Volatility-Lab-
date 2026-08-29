"""Feedback loop tools: get_next_action and explain_decision.

These tools are the heart of the iterative workflow. `get_next_action`
inspects the session state and returns the recommended next tool call
(with suggested arguments) plus alternatives. `explain_decision`
replays the decision log so the agent can justify each step to a user.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from core.feedback.advisor import recommend_next_action, explain_decision
from core.feedback.recommendations import heuristic_recommendations
from core.feedback.session import SessionStore
from core.feedback.workflow_dag import valid_transitions

from .session import _store


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def get_next_action(
        session_id: str, intent: str = "auto",
    ) -> dict[str, Any]:
        """Recommend the next tool call for this session.

        Inspects the session state (current stage, fitted models,
        diagnostics, backtest results) and returns:
          - recommended_action: tool name
          - suggested_args: dict of arguments
          - rationale: one-sentence justification
          - alternatives: list of {action, args, why}
          - blocked_actions: actions that would fail given current state

        Args:
            intent: "auto" (default), "fit" (force a fit recommendation),
                "risk" (force a risk recommendation), "report" (force a
                report recommendation), or "validate" (force backtest).
        """
        s = _store.get(session_id)
        rec = recommend_next_action(s, intent=intent)
        return {
            "recommended_action": rec.action,
            "suggested_args": rec.args,
            "rationale": rec.rationale,
            "alternatives": [
                {"action": a.action, "args": a.args, "why": a.why}
                for a in rec.alternatives
            ],
            "blocked_actions": _blocked_actions(s),
            "current_stage": s.current_stage,
            "decision_count": len(s.decision_log.entries),
        }

    @mcp.tool()
    def explain_decision(
        session_id: str, decision_id: int | None = None,
    ) -> dict[str, Any]:
        """Explain one or all decisions in the session's decision log.

        If `decision_id` is given, returns that single decision.
        Otherwise returns the full chronological log. Each entry
        includes: action, target, detail, timestamp, and a human-
        readable explanation.

        Useful for generating an audit trail or justifying the analysis
        path to a user or examiner.
        """
        s = _store.get(session_id)
        if decision_id is not None:
            entry = s.decision_log.get(decision_id)
            if entry is None:
                return {"error": f"No decision with id {decision_id}"}
            return explain_decision(entry, session=s)
        return {
            "decisions": [
                explain_decision(e, session=s) for e in s.decision_log.entries
            ],
            "count": len(s.decision_log.entries),
        }

    @mcp.tool()
    def get_recommendations(session_id: str) -> list[str]:
        """Return heuristic recommendations based on current session state.

        Inspects preflight, diagnostics, and backtest results and
        returns a list of plain-language recommendations, e.g.:
          - "Normality blocked: use Student-t distribution"
          - "Sign-bias detected: try GJR-GARCH or EGARCH"
          - "Kupiec p < 0.05: underprediction — tighten refit interval"
        """
        s = _store.get(session_id)
        return heuristic_recommendations(s)

    @mcp.tool()
    def list_valid_transitions(session_id: str) -> dict[str, list[str]]:
        """List the valid next actions given the current stage.

        Returns a dict mapping each candidate action name to the
        rationale for why it is valid now (or invalid, with the reason).
        """
        s = _store.get(session_id)
        return valid_transitions(s)


def _blocked_actions(session) -> list[dict[str, str]]:
    blocked = []
    if session.returns is None:
        blocked.append({"action": "preflight.run", "reason": "No data loaded"})
        blocked.append({"action": "models.fit", "reason": "No data loaded"})
    if session.preflight_result is None and session.returns is not None:
        blocked.append({"action": "models.fit", "reason": "Preflight not run"})
    if not session.fitted_models:
        blocked.append({"action": "diagnostics.run", "reason": "No fitted model"})
        blocked.append({"action": "compare.run", "reason": "Need ≥1 fitted model"})
        blocked.append({"action": "risk.var", "reason": "No fitted model"})
    if not session.backtest_results:
        blocked.append({
            "action": "backtest.coverage",
            "reason": "No backtest yet",
        })
    return blocked
