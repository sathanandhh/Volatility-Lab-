"""State machine for session stage transitions."""
from __future__ import annotations

from core.feedback.session import STAGES

# Valid forward transitions
TRANSITIONS: dict[str, list[str]] = {
    "empty":     ["data"],
    "data":      ["preflight"],
    "preflight": ["optimize", "fit"],
    "optimize":  ["optimize", "fit"],
    "fit":       ["optimize", "compare", "risk", "backtest", "fit"],
    "compare":   ["risk", "backtest", "report"],
    "risk":      ["backtest", "report", "scenario"],
    "backtest":  ["optimize", "fit", "report"],
    "report":    [],
}


def can_transition(from_stage: str, to_stage: str) -> bool:
    return to_stage in TRANSITIONS.get(from_stage, [])


def valid_next_stages(current_stage: str) -> list[str]:
    return TRANSITIONS.get(current_stage, [])
