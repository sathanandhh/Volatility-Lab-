"""Pydantic schemas for persisted session rows.

These are used for validation when hydrating sessions from the database
and for type-safe access in downstream code.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DecisionRow(BaseModel):
    """One row in the decision_log table."""
    session_id: str
    entry_id: int
    timestamp: str
    action: str
    target: str = ""
    outcome: str = ""
    detail: str = ""


class SessionRow(BaseModel):
    """Projection of the sessions table."""
    id: str
    created_at: str
    updated_at: str
    current_stage: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    last_fit_config: dict[str, Any] | None = None
    comparison_result: dict[str, Any] | None = None
    coverage_scorecard: list[dict[str, Any]] | None = None
    is_restored: bool = False


class FittedModelRow(BaseModel):
    """Projection of the fitted_models table."""
    session_id: str
    model_name: str
    name: str
    params: dict[str, float] = Field(default_factory=dict)
    loglikelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    n_params: int = 0
    converged: bool = False
    convergence_flag: int = 0
    next_vol: float = 0.0


class OptimizationRow(BaseModel):
    """Projection of the optimization_results table."""
    session_id: str
    kind: str
    optimal: Any = None
    optimal_score: float = 0.0
    criterion: str = ""
    sweep_table: list[dict[str, Any]] = Field(default_factory=list)
    recommendation: str = ""
