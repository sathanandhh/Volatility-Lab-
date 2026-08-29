"""DAG of valid action transitions in the feedback loop."""
from __future__ import annotations

from typing import Any

from core.feedback.session import Session


# Action → valid predecessor stages
ACTION_PRECONDITIONS: dict[str, list[str]] = {
    "data.load_market":        ["empty"],
    "data.load_csv":           ["empty"],
    "preflight.run":           ["data"],
    "preflight.get_gate_status": ["preflight"],
    "preflight.explain_gate":  ["preflight"],
    "optimize.order":          ["preflight", "optimize"],
    "optimize.distribution":   ["preflight", "optimize"],
    "optimize.mean":           ["preflight", "optimize"],
    "optimize.window":         ["preflight", "optimize"],
    "optimize.refit":           ["fit"],
    "optimize.horizon":        ["preflight"],
    "models.fit":              ["preflight"],
    "models.forecast":         ["fit"],
    "diagnostics.run":         ["fit"],
    "compare.run":             ["fit"],
    "risk.var":                ["fit"],
    "risk.es":                 ["fit"],
    "risk.basel_es":           ["fit"],
    "risk.portfolio_var":      ["fit"],
    "scenario.apply_shock":    ["fit"],
    "scenario.stressed_var":   ["fit"],
    "backtest.rolling":        ["fit"],
    "backtest.kupiec":         ["backtest"],
    "backtest.christoffersen": ["backtest"],
    "backtest.traffic_light":  ["backtest"],
    "backtest.dynamic_quantile":["backtest"],
    "backtest.diebold_mariano":["backtest"],
    "backtest.coverage":       ["backtest"],
    "report.excel":            ["fit"],
    "report.pdf":              ["fit"],
    "report.markdown":         ["fit"],
    "feedback.get_next_action": ["empty", "data", "preflight", "optimize",
                                  "fit", "compare", "risk", "backtest", "report"],
    "feedback.explain_decision": ["empty", "data", "preflight", "optimize",
                                   "fit", "compare", "risk", "backtest", "report"],
}


def valid_transitions(session: Session) -> dict[str, list[str]]:
    """Return {action: [valid_when, reason_if_invalid]}."""
    stage = session.current_stage
    out: dict[str, list[str]] = {}
    for action, prereqs in ACTION_PRECONDITIONS.items():
        if stage in prereqs:
            out[action] = [stage, "valid now"]
        else:
            out[action] = [stage, f"requires stage in {prereqs}"]
    return out


def is_valid_action(session: Session, action: str) -> bool:
    prereqs = ACTION_PRECONDITIONS.get(action, [])
    return session.current_stage in prereqs
