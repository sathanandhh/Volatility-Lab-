"""Base Optimizer interface and OptimizationResult dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizationResult:
    """Generic optimization result returned by all selectors."""
    kind: str
    optimal: Any  # type varies: dict for order, str for distribution, int for window
    optimal_score: float
    criterion: str
    sweep_table: list[dict]
    recommendation: str

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "optimal": self.optimal,
            "optimal_score": self.optimal_score,
            "criterion": self.criterion,
            "recommendation": self.recommendation,
        }


class BaseOptimizer:
    """Common interface for all input optimizers."""
    kind: str = "base"

    def optimize(self, *args, **kwargs) -> OptimizationResult:
        raise NotImplementedError


def _make_recommendation(criterion: str, best: Any, second_best: Any,
                         best_score: float, second_score: float) -> str:
    """Compose a plain-language recommendation comparing top-2 candidates."""
    delta = second_score - best_score if criterion in ("AIC", "BIC", "QLIKE", "RMSE", "MAE") else best_score - second_score
    return (
        f"Best: {best} ({criterion}={best_score:.2f}); "
        f"Runner-up: {second_best} ({criterion}={second_score:.2f}). "
        f"Δ={delta:.2f}. "
        + ("Simpler model preferred." if abs(delta) < 2 else "Significant improvement.")
    )
