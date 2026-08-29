"""Generic grid search over a discrete parameter space."""
from __future__ import annotations

from typing import Callable, Any

from core.optimize.optimizer import OptimizationResult


def grid_search(
    candidates: list[Any],
    objective: Callable[[Any], float],
    criterion: str = "AIC",
    minimize: bool = True,
    kind: str = "generic",
    stringify: Callable[[Any], str] | None = None,
) -> OptimizationResult:
    """Run a grid search and return the optimum with the full sweep table."""
    rows = []
    for cand in candidates:
        score = float(objective(cand))
        label = stringify(cand) if stringify else str(cand)
        rows.append({"candidate": label, "score": score})
    rows.sort(key=lambda r: r["score"], reverse=not minimize)
    best = rows[0]
    second = rows[1] if len(rows) > 1 else rows[0]
    rec = (
        f"Best: {best['candidate']} ({criterion}={best['score']:.2f}); "
        f"runner-up: {second['candidate']} ({criterion}={second['score']:.2f}). "
        f"Δ={abs(best['score'] - second['score']):.2f}."
    )
    # Recover the actual candidate object
    best_obj = candidates[[stringify(c) if stringify else str(c)
                           for c in candidates].index(best["candidate"])] if stringify else candidates[0]
    return OptimizationResult(
        kind=kind, optimal=best_obj, optimal_score=best["score"],
        criterion=criterion, sweep_table=rows, recommendation=rec,
    )
