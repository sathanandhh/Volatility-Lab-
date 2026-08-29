"""Session — the stateful workspace that drives the feedback loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from core.feedback.decision_log import DecisionLog
from core.preflight.gates import GateResult


STAGES = ["empty", "data", "preflight", "optimize", "fit",
          "compare", "risk", "backtest", "report"]


@dataclass
class Session:
    """Stateful analytics session."""
    id: str
    returns: pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    preflight_result: GateResult | None = None
    optimization_results: dict[str, Any] = field(default_factory=dict)
    fitted_models: dict[str, Any] = field(default_factory=dict)
    diagnostics_results: dict[str, Any] = field(default_factory=dict)
    backtest_results: dict[str, Any] = field(default_factory=dict)
    risk_results: dict[str, Any] = field(default_factory=dict)
    comparison_result: dict[str, Any] | None = None
    coverage_scorecard: list[dict] | None = None
    shocked_models: dict[str, Any] = field(default_factory=dict)
    last_fit_config: dict[str, Any] | None = None
    decision_log: DecisionLog = field(default_factory=DecisionLog)
    current_stage: str = "empty"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # ── State transitions ──────────────────────────────────────────
    def advance_stage(self, stage: str) -> None:
        if stage in STAGES:
            self.current_stage = stage

    def reset(self, keep_data: bool = True) -> None:
        if not keep_data:
            self.returns = None
            self.metadata = {}
        self.preflight_result = None
        self.optimization_results = {}
        self.fitted_models = {}
        self.diagnostics_results = {}
        self.backtest_results = {}
        self.risk_results = {}
        self.comparison_result = None
        self.coverage_scorecard = None
        self.shocked_models = {}
        self.last_fit_config = None
        self.current_stage = "empty" if not keep_data or self.returns is None else "data"

    def log_decision(self, action: str, target: str,
                     outcome: str = "", detail: str = "") -> None:
        self.decision_log.add(action, target, outcome, detail)

    # ── Serialisation ───────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "stage": self.current_stage,
            "metadata": self.metadata,
            "n_returns": int(len(self.returns)) if self.returns is not None else 0,
            "preflight_overall": (
                self.preflight_result.overall.value
                if self.preflight_result else None
            ),
            "optimization_kinds": list(self.optimization_results.keys()),
            "fitted_models": list(self.fitted_models.keys()),
            "diagnostics_run_on": list(self.diagnostics_results.keys()),
            "backtested_models": list(self.backtest_results.keys()),
            "risk_results_keys": list(self.risk_results.keys()),
            "comparison": self.comparison_result,
            "coverage_scorecard": self.coverage_scorecard,
            "last_fit_config": self.last_fit_config,
            "decision_count": len(self.decision_log),
            "next_actions": self._recommended_next_actions(),
            "created_at": self.created_at,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "stage": self.current_stage,
            "n_returns": int(len(self.returns)) if self.returns is not None else 0,
            "fitted_models": list(self.fitted_models.keys()),
            "decision_count": len(self.decision_log),
        }

    def _recommended_next_actions(self) -> list[str]:
        from core.feedback.defaults import default_next_actions
        return default_next_actions(self)


class SessionStore:
    """In-process session store. Replace with persistent backend for scale."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def open(self) -> str:
        import uuid
        sid = str(uuid.uuid4())
        self._sessions[sid] = Session(id=sid)
        return sid

    def get(self, sid: str) -> Session:
        if sid not in self._sessions:
            raise KeyError(f"Unknown session: {sid!r}")
        return self._sessions[sid]

    def close(self, sid: str) -> None:
        self._sessions.pop(sid, None)

    def list_ids(self) -> list[str]:
        return list(self._sessions.keys())
