"""Rolling-window size selection via out-of-sample QLIKE."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.optimize.optimizer import OptimizationResult


def select_window(
    returns: pd.Series, min_window: int = 10, max_window: int = 63,
    step: int = 5, metric: str = "QLIKE",
) -> OptimizationResult:
    """Sweep rolling-window sizes; pick the one with the best QLIKE."""
    r = returns.dropna()
    rows = []
    for w in range(min_window, max_window + 1, step):
        cv = r.rolling(w).std()
        # Out-of-sample evaluation: forecast day-t+1 with window ending at t
        var = cv.pow(2).shift(1)
        aligned = pd.DataFrame({"r2": r.pow(2), "var": var}).dropna()
        if aligned.empty:
            rows.append({"window": w, "score": float("inf")})
            continue
        v = aligned["var"].clip(lower=1e-10)
        if metric == "QLIKE":
            score = float(np.mean(np.log(v) + aligned["r2"] / v))
        elif metric == "RMSE":
            score = float(np.sqrt(np.mean((aligned["r2"] - v) ** 2)))
        else:
            score = float(np.mean(np.abs(aligned["r2"] - v)))
        rows.append({"window": w, "score": score})
    rows.sort(key=lambda x: x["score"])
    best = rows[0]
    second = rows[1] if len(rows) > 1 else rows[0]
    rec = (
        f"Best window: {best['window']} ({metric}={best['score']:.4f}); "
        f"runner-up: {second['window']} ({metric}={second['score']:.4f})."
    )
    return OptimizationResult(
        kind="window", optimal=int(best["window"]),
        optimal_score=float(best["score"]), criterion=metric,
        sweep_table=rows, recommendation=rec,
    )
