"""Refit-frequency selection via conditional-coverage p-value."""
from __future__ import annotations

import pandas as pd

from core.optimize.optimizer import OptimizationResult


def select_refit_frequency(
    returns: pd.Series, model_name: str = "GARCH",
    min_freq: int = 1, max_freq: int = 30,
    alpha: float = 0.025,
) -> OptimizationResult:
    """Sweep refit-every-N; pick the one with the highest CC p-value."""
    from core.backtest.rolling import rolling_backtest
    from core.backtest.christoffersen import conditional_coverage
    rows = []
    train = int(len(returns) * 0.8)
    for n in range(min_freq, max_freq + 1, 5):
        try:
            bt = rolling_backtest(returns, model_names=(model_name,),
                                  distribution="t",
                                  train_size=train, alpha=alpha,
                                  refit_every=n)
            hits = bt.for_model(model_name).violations
            _, _, cc_stat, cc_p = conditional_coverage(hits, alpha=alpha)
            rows.append({"refit_every": n, "cc_p": float(cc_p), "score": float(cc_p)})
        except Exception as exc:
            rows.append({"refit_every": n, "cc_p": 0.0, "score": 0.0,
                         "error": str(exc)})
    rows.sort(key=lambda r: -r["score"])  # maximize p
    best = rows[0]
    second = rows[1] if len(rows) > 1 else rows[0]
    rec = (
        f"Best refit: every {best['refit_every']} (CC p={best['cc_p']:.3f}); "
        f"runner-up: every {second['refit_every']} (CC p={second['cc_p']:.3f})."
    )
    return OptimizationResult(
        kind="refit", optimal=int(best["refit_every"]),
        optimal_score=float(best["score"]), criterion="CC p-value",
        sweep_table=rows, recommendation=rec,
    )
