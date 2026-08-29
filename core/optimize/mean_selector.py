"""Mean specification selection via residual Ljung-Box."""
from __future__ import annotations

import pandas as pd

from core.optimize.optimizer import OptimizationResult


def select_mean(
    returns: pd.Series, candidates: list[str] | None = None,
) -> OptimizationResult:
    """Pick the mean specification with the largest Ljung-Box p-value
    on its residuals (i.e. no remaining serial correlation)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.arima.model import ARIMA
    cands = candidates or ["Constant", "AR(1)", "ARMA(1,1)"]
    r = returns.dropna()
    rows = []
    for spec in cands:
        try:
            if spec == "Constant":
                resid = r - r.mean()
            elif spec == "AR(1)":
                m = AutoReg(r, lags=1).fit()
                resid = m.resid
            elif spec == "ARMA(1,1)":
                m = ARIMA(r, order=(1, 0, 1)).fit()
                resid = m.resid
            else:
                rows.append({"spec": spec, "p_value": 0.0, "score": 0.0})
                continue
            lb = acorr_ljungbox(resid.dropna(), lags=[10], return_df=True)
            p = float(lb.loc[10, "lb_pvalue"])
            rows.append({"spec": spec, "p_value": p, "score": p})
        except Exception as exc:
            rows.append({"spec": spec, "p_value": 0.0, "score": 0.0,
                         "error": str(exc)})
    rows.sort(key=lambda r: -r["score"])  # maximize p-value
    best = rows[0]
    second = rows[1] if len(rows) > 1 else rows[0]
    rec = (
        f"Best mean: {best['spec']} (LB p={best['p_value']:.3f}); "
        f"runner-up: {second['spec']} (LB p={second['p_value']:.3f})."
    )
    return OptimizationResult(
        kind="mean", optimal=best["spec"],
        optimal_score=float(best["score"]), criterion="LB p-value",
        sweep_table=rows, recommendation=rec,
    )
