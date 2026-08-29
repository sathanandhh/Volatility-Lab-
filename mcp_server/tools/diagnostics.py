"""Post-fit residual diagnostics tools.

These tools run AFTER a model has been fit. They test whether the
standardized residuals behave like white noise — i.e., whether the model
has captured the volatility dynamics. Failed diagnostics should trigger
a return to `optimize.*` or `models.fit` with a different specification.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from core.diagnostics.residual import ljung_box, arch_lm_test
from core.diagnostics.information_criteria import aic, bic, hqic
from core.diagnostics.accuracy import qlike, volatility_rmse, mae
from core.diagnostics.normality import (
    jarque_bera, shapiro_wilk, anderson_darling,
)
from core.diagnostics.sign_bias import sign_bias_test
from core.diagnostics.nyblom import nyblom_stability
from core.feedback.session import SessionStore

from .session import _store


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def run_diagnostics(session_id: str, model_name: str) -> dict[str, Any]:
        """Run the full residual diagnostic suite on a fitted model.

        Tests:
          - Ljung-Box on standardized residuals (lags 10, 20)
          - Ljung-Box on squared standardized residuals
          - ARCH-LM (remaining ARCH effect)
          - Jarque-Bera, Shapiro-Wilk, Anderson-Darling normality
          - Sign-bias test (negative/positive/asymmetric)
          - Nyblom parameter stability
          - QLIKE, volatility RMSE, MAE accuracy
          - AIC, BIC, HQIC

        Returns per-test p-values and an overall recommendation. If any
        critical test fails, `optimize.*` is recommended as next action.
        """
        s = _store.get(session_id)
        if model_name not in s.fitted_models:
            available = ", ".join(s.fitted_models.keys()) or "none"
            return _error(f"Model '{model_name}' not fitted. Available: {available}")
        fit = s.fitted_models[model_name]
        std_resid = fit.std_resid
        cond_vol = fit.conditional_volatility
        realized = s.returns
        result = {
            "model": model_name,
            "ljung_box_resid_lag10": ljung_box(std_resid, lag=10),
            "ljung_box_resid_lag20": ljung_box(std_resid, lag=20),
            "ljung_box_sq_resid_lag10": ljung_box(std_resid.pow(2), lag=10),
            "arch_lm_lag10": arch_lm_test(std_resid, nlags=10),
            "normality": {
                "jarque_bera": jarque_bera(std_resid),
                "shapiro_wilk": shapiro_wilk(std_resid),
                "anderson_darling": anderson_darling(std_resid),
            },
            "sign_bias": sign_bias_test(std_resid),
            "nyblom_stability": nyblom_stability(fit),
            "accuracy": {
                "qlike": qlike(realized, cond_vol.pow(2)),
                "vol_rmse": volatility_rmse(realized, cond_vol),
                "mae": mae(realized, cond_vol),
            },
            "information_criteria": {
                "aic": aic(fit), "bic": bic(fit), "hqic": hqic(fit),
            },
        }
        s.diagnostics_results[model_name] = result
        recommendations = _recommendations(result)
        s.log_decision(
            "diagnostics.run", model_name,
            detail=f"{len(recommendations)} recommendations",
        )
        return {
            **result,
            "recommendations": recommendations,
            "next_actions": _next_actions(recommendations, list(s.fitted_models)),
        }

    @mcp.tool()
    def get_diagnostics(session_id: str, model_name: str) -> dict[str, Any]:
        """Retrieve the most recent diagnostic result for a fitted model."""
        s = _store.get(session_id)
        if model_name not in s.diagnostics_results:
            return _error(
                f"No diagnostics for '{model_name}'. Call diagnostics.run first."
            )
        return s.diagnostics_results[model_name]


def _recommendations(result: dict) -> list[str]:
    recs: list[str] = []
    if result["ljung_box_sq_resid_lag10"]["p_value"] < 0.05:
        recs.append(
            "Squared-residual LB fails: remaining ARCH effect. "
            "Increase q or try FIGARCH."
        )
    if result["arch_lm_lag10"]["p_value"] < 0.05:
        recs.append(
            "ARCH-LM rejects: model has not captured all volatility dynamics."
        )
    if result["normality"]["jarque_bera"]["p_value"] < 0.01:
        recs.append(
            "Residuals non-normal: try Student-t or skew-t distribution."
        )
    sb = result["sign_bias"]
    if sb.get("negative_p", 1) < 0.05 or sb.get("positive_p", 1) < 0.05:
        recs.append(
            "Sign-bias detected: try GJR-GARCH or EGARCH for leverage."
        )
    if not result["nyblom_stability"]["stable"]:
        recs.append(
            "Parameter instability: consider regime split or rolling refit."
        )
    if not recs:
        recs.append("Diagnostics look clean. Proceed to compare or risk.")
    return recs


def _next_actions(recs: list[str], fitted: list[str]) -> list[str]:
    if any("ARCH" in r or "LB" in r for r in recs):
        return ["optimize.order", "models.fit"]
    if any("Student-t" in r or "skew" in r for r in recs):
        return ["optimize.distribution", "models.fit"]
    if any("GJR" in r or "EGARCH" in r for r in recs):
        return ["models.fit"]
    if len(fitted) > 1:
        return ["compare.run", "risk.var"]
    return ["models.fit", "compare.run", "risk.var"]


def _error(msg: str) -> dict[str, Any]:
    return {"error": msg, "next_actions": ["models.fit"]}
