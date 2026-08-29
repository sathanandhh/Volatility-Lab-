"""Bayesian optimization via Optuna (optional dependency)."""
from __future__ import annotations

from typing import Callable, Any

from core.optimize.optimizer import OptimizationResult


def bayesian_optimize(
    objective: Callable[[Any], float],
    search_space: dict[str, tuple],
    n_trials: int = 50,
    minimize: bool = True,
    kind: str = "bayesian",
) -> OptimizationResult:
    """Run Optuna optimization over a continuous/discrete search space.

    The `objective` callable receives a dict of suggested parameters.
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Install optuna to use Bayesian optimization.") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    direction = "minimize" if minimize else "maximize"
    study = optuna.create_study(direction=direction)

    def _trial(trial):
        params = {k: trial.suggest_uniform(k, lo, hi) if isinstance(v, tuple) and len(v) == 2
                  else trial.suggest_int(k, v[0], v[1])
                  for k, v in search_space.items()}
        return float(objective(params))

    study.optimize(_trial, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best_score = float(study.best_value)
    rows = [{"trial": t.number, "params": t.params, "value": float(t.value)}
            for t in study.trials]
    return OptimizationResult(
        kind=kind, optimal=best, optimal_score=best_score,
        criterion="objective", sweep_table=rows,
        recommendation=f"Optuna best (n={n_trials} trials): {best}",
    )
