"""Default next-action mappings for the session's _recommended_next_actions()."""
from __future__ import annotations

from core.feedback.session import Session


def default_next_actions(session: Session) -> list[str]:
    """Return the list of valid next tool calls given the session stage."""
    from core.feedback.workflow_dag import ACTION_PRECONDITIONS
    stage = session.current_stage
    valid = [action for action, prereqs in ACTION_PRECONDITIONS.items()
             if stage in prereqs]
    # Order by importance
    priority_order = [
        "data.load_market", "data.load_csv",
        "preflight.run", "preflight.explain_gate",
        "optimize.order", "optimize.distribution", "optimize.mean",
        "optimize.window", "optimize.refit", "optimize.horizon",
        "models.fit", "models.forecast",
        "diagnostics.run", "compare.run",
        "risk.var", "risk.es", "risk.basel_es",
        "scenario.apply_shock", "scenario.stressed_var",
        "backtest.rolling", "backtest.coverage",
        "report.excel", "report.pdf", "report.markdown",
        "feedback.get_next_action", "feedback.explain_decision",
    ]
    return [a for a in priority_order if a in valid]
