"""Model comparison tools.

Compares fitted models on in-sample fit (AIC, BIC, log-likelihood) and
out-of-sample accuracy (QLIKE, volatility RMSE, MAE). The comparison
only includes models that have been fit in this session. If
`backtest.rolling` has been run, the rolling-forecast accuracy metrics
are included too.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from core.diagnostics.information_criteria import aic, bic
from core.diagnostics.accuracy import qlike, volatility_rmse, mae
from core.feedback.session import SessionStore

from .session import _store


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def compare_models(session_id: str) -> dict[str, Any]:
        """Compare all models fitted in this session.

        Produces a scorecard with columns:
          Model, AIC, BIC, Log likelihood, QLIKE, Vol RMSE, MAE,
          AIC rank, QLIKE rank.

        AIC ranks in-sample fit (lower = better, balancing complexity).
        QLIKE ranks out-of-sample forecast accuracy (lower = better).
        They may disagree — common in financial volatility because the
        best in-sample fit is not always the best forecaster.

        Returns the best-fit and best-forecast models separately so the
        agent can explain the trade-off.
        """
        s = _store.get(session_id)
        if not s.fitted_models:
            return _error(
                "No models fitted. Call models.fit first.",
                next_actions=["models.fit"],
            )
        rows = []
        for name, fit in s.fitted_models.items():
            cv = fit.conditional_volatility
            realized = s.returns
            rows.append({
                "model": name,
                "aic": float(fit.aic),
                "bic": float(fit.bic),
                "loglik": float(fit.loglikelihood),
                "qlike": float(qlike(realized, cv.pow(2))),
                "vol_rmse": float(volatility_rmse(realized, cv)),
                "mae": float(mae(realized, cv)),
            })
        # Ranks
        for metric in ("aic", "bic", "qlike", "vol_rmse", "mae"):
            sorted_rows = sorted(rows, key=lambda r: r[metric])
            for rank, r in enumerate(sorted_rows, 1):
                r[f"{metric}_rank"] = rank
        best_fit = min(rows, key=lambda r: r["aic"])["model"]
        best_forecast = min(rows, key=lambda r: r["qlike"])["model"]
        s.comparison_result = {
            "scorecard": rows,
            "best_fit": best_fit,
            "best_forecast": best_forecast,
        }
        s.advance_stage("compare")
        s.log_decision(
            "compare.run", "n/a",
            detail=f"best_fit={best_fit}, best_forecast={best_forecast}",
        )
        return {
            "scorecard": rows,
            "best_fit": best_fit,
            "best_forecast": best_forecast,
            "recommendation": (
                f"Best in-sample fit by AIC: {best_fit}. "
                f"Best out-of-sample by QLIKE: {best_forecast}. "
                "If they differ, prefer the QLIKE winner for risk forecasting."
            ),
            "next_actions": ["risk.var", "risk.es", "backtest.rolling"],
        }


def _error(msg: str, next_actions: list[str] | None = None) -> dict[str, Any]:
    return {"error": msg, "next_actions": next_actions or []}
