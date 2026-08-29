"""Distribution selection via AIC sweep."""
from __future__ import annotations

import pandas as pd

from core.optimize.optimizer import OptimizationResult
from core.models.univariate.arch_family import fit_arch_family


def select_distribution(
    returns: pd.Series, family: str = "GARCH",
    p: int = 1, q: int = 1,
    candidates: list[str] | None = None,
) -> OptimizationResult:
    """Fit each candidate distribution and pick the lowest-AIC one."""
    cands = candidates or ["normal", "t", "ged", "skewt", "jsu"]
    rows = []
    fits = {}
    for dist in cands:
        try:
            o = 1 if family in ("GJR-GARCH", "EGARCH") else 0
            fit = fit_arch_family(returns, family=family, p=p, q=q, o=o,
                                  distribution=dist, mean="Constant")
            rows.append({"distribution": dist, "aic": float(fit.aic),
                         "converged": fit.converged})
            fits[dist] = fit
        except Exception as exc:
            rows.append({"distribution": dist, "aic": float("inf"),
                         "converged": False, "error": str(exc)})
    rows.sort(key=lambda r: r["aic"])
    best = rows[0]
    second = rows[1] if len(rows) > 1 else rows[0]
    rec = (
        f"Best distribution: {best['distribution']} (AIC={best['aic']:.2f}); "
        f"runner-up: {second['distribution']} (AIC={second['aic']:.2f}). "
        f"ΔAIC={best['aic']-second['aic']:.2f}."
    )
    return OptimizationResult(
        kind="distribution", optimal=best["distribution"],
        optimal_score=float(best["aic"]), criterion="AIC",
        sweep_table=rows, recommendation=rec,
    )
