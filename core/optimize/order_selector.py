"""(p, q) order selection for ARCH-family models via AIC/BIC sweep."""
from __future__ import annotations

import pandas as pd

from core.optimize.optimizer import OptimizationResult
from core.models.univariate.arch_family import fit_arch_family


def select_order(
    returns: pd.Series, family: str = "GARCH",
    max_p: int = 3, max_q: int = 2,
    distribution: str = "t", criterion: str = "AIC",
) -> OptimizationResult:
    """Sweep (p, q) pairs and pick the configuration with the lowest AIC/BIC."""
    rows = []
    fits = {}
    for p in range(1, max_p + 1):
        for q in range(0, max_q + 1):
            o = 1 if family in ("GJR-GARCH", "EGARCH") else 0
            try:
                fit = fit_arch_family(returns, family=family, p=p, q=q, o=o,
                                       distribution=distribution, mean="Constant")
                score = fit.aic if criterion == "AIC" else fit.bic
                rows.append({"p": p, "q": q, "score": float(score),
                             "converged": fit.converged})
                fits[(p, q)] = fit
            except Exception as exc:
                rows.append({"p": p, "q": q, "score": float("inf"),
                             "converged": False, "error": str(exc)})
    rows.sort(key=lambda r: r["score"])
    best_row = rows[0]
    second_row = rows[1] if len(rows) > 1 else rows[0]
    best_pq = {"p": best_row["p"], "q": best_row["q"]}
    rec = (
        f"Best: ({best_row['p']},{best_row['q']}) {criterion}={best_row['score']:.2f}; "
        f"runner-up: ({second_row['p']},{second_row['q']}) {criterion}={second_row['score']:.2f}. "
        f"Δ={best_row['score']-second_row['score']:.2f}. "
        + ("Simpler preferred." if abs(best_row['score']-second_row['score']) < 2 else "")
    )
    return OptimizationResult(
        kind="order", optimal=best_pq, optimal_score=float(best_row["score"]),
        criterion=criterion, sweep_table=rows, recommendation=rec,
    )
