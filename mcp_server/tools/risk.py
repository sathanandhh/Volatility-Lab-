"""Risk quantification tools: VaR, Expected Shortfall, Basel ES.

Computes tail-risk metrics using the conditional volatility forecast
from a fitted model. Supports multiple tail methods (parametric Normal,
Student-t, historical simulation, filtered historical simulation) so
the agent can compare and explain methodological differences.
"""
from __future__ import annotations

import math
from typing import Any

from mcp.server.fastmcp import FastMCP

from core.risk.var.parametric import (
    var_normal, var_student_t, var_cornish_fisher,
)
from core.risk.var.historical import var_historical
from core.risk.var.filtered_hist import var_fhs
from core.risk.var.monte_carlo import var_mc
from core.risk.es.expected_shortfall import (
    es_normal, es_student_t, es_historical,
)
from core.risk.es.basel_es import basel_es_97_5
from core.risk.portfolio import portfolio_var_decomposition
from core.models.univariate.arch_family import forecast_arch_family
from core.feedback.session import SessionStore

from .session import _store


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def compute_var(
        session_id: str,
        model: str,
        confidence: float = 0.975,
        horizon: int = 1,
        portfolio_value: float = 1_000_000.0,
        method: str = "student_t",
        degrees_of_freedom: float | None = None,
    ) -> dict[str, Any]:
        """Compute Value at Risk from a fitted model's volatility forecast.

        Args:
            session_id: Active session id.
            model: Fitted model name.
            confidence: Confidence level (0.90, 0.95, 0.975, 0.99).
            horizon: Holding period in days.
            portfolio_value: Notional for monetary VaR.
            method: "normal", "student_t", "cornish_fisher",
                "historical", "fhs" (filtered historical), or
                "monte_carlo".
            degrees_of_freedom: Override for Student-t ν; if None, use
                the fitted model's ν.

        Returns VaR in both return-percent and currency, the horizon-
        scaled volatility used, and the methodological notes.
        """
        s = _store.get(session_id)
        if model not in s.fitted_models:
            return _error(f"Model '{model}' not fitted.", ["models.fit"])
        fit = s.fitted_models[model]
        f = forecast_arch_family(fit, horizon=horizon, method="analytic")
        horizon_var = sum(v for v in f.variance_path)
        sigma = math.sqrt(horizon_var)
        mu = float(fit.params.get("mu", 0.0)) * horizon
        if degrees_of_freedom is None:
            degrees_of_freedom = float(fit.params.get("nu", 8.0))
        alpha = 1 - confidence
        if method == "normal":
            var_rate = var_normal(mu, sigma, alpha)
        elif method == "student_t":
            var_rate = var_student_t(mu, sigma, alpha, degrees_of_freedom)
        elif method == "cornish_fisher":
            var_rate = var_cornish_fisher(s.returns, mu, sigma, alpha)
        elif method == "historical":
            var_rate = var_historical(s.returns, alpha, horizon=horizon)
        elif method == "fhs":
            var_rate = var_fhs(fit.std_resid, sigma, alpha)
        elif method == "monte_carlo":
            var_rate = var_mc(fit, horizon=horizon, alpha=alpha, n_sims=10_000)
        else:
            return _error(f"Unknown method '{method}'.")
        var_currency = portfolio_value * abs(var_rate)
        result = {
            "model": model,
            "method": method,
            "confidence": confidence,
            "horizon_days": horizon,
            "horizon_volatility_pct": sigma * 100,
            "var_return_pct": float(var_rate) * 100,
            "var_currency": float(var_currency),
            "portfolio_value": portfolio_value,
            "degrees_of_freedom": degrees_of_freedom if method == "student_t" else None,
        }
        s.risk_results[f"var_{model}_{method}"] = result
        s.advance_stage("risk")
        s.log_decision(
            "risk.var", model,
            detail=f"{method}, α={alpha}, VaR={var_currency:.0f}",
        )
        result["next_actions"] = ["risk.es", "backtest.rolling", "report.excel"]
        return result

    @mcp.tool()
    def compute_es(
        session_id: str,
        model: str,
        confidence: float = 0.975,
        horizon: int = 1,
        portfolio_value: float = 1_000_000.0,
        method: str = "student_t",
    ) -> dict[str, Any]:
        """Compute Expected Shortfall (CVaR) from a fitted model.

        ES is the average loss conditional on the loss exceeding VaR.
        Same parameters as compute_var. Returns ES in both return-percent
        and currency, plus the ratio ES/VaR (which quantifies tail
        thickness — ratios above 1.3 suggest heavy tails).
        """
        s = _store.get(session_id)
        if model not in s.fitted_models:
            return _error(f"Model '{model}' not fitted.", ["models.fit"])
        fit = s.fitted_models[model]
        f = forecast_arch_family(fit, horizon=horizon, method="analytic")
        sigma = math.sqrt(sum(v for v in f.variance_path))
        mu = float(fit.params.get("mu", 0.0)) * horizon
        alpha = 1 - confidence
        nu = float(fit.params.get("nu", 8.0))
        if method == "normal":
            es_rate = es_normal(mu, sigma, alpha)
            var_rate = var_normal(mu, sigma, alpha)
        elif method == "student_t":
            es_rate = es_student_t(mu, sigma, alpha, nu)
            var_rate = var_student_t(mu, sigma, alpha, nu)
        elif method == "historical":
            es_rate = es_historical(s.returns, alpha, horizon=horizon)
            var_rate = var_historical(s.returns, alpha, horizon=horizon)
        else:
            return _error(f"Unknown method '{method}' for ES.")
        es_currency = portfolio_value * abs(es_rate)
        ratio = abs(es_rate / var_rate) if var_rate else None
        result = {
            "model": model,
            "method": method,
            "confidence": confidence,
            "horizon_days": horizon,
            "es_return_pct": float(es_rate) * 100,
            "es_currency": float(es_currency),
            "es_to_var_ratio": float(ratio) if ratio else None,
            "portfolio_value": portfolio_value,
        }
        s.risk_results[f"es_{model}_{method}"] = result
        s.log_decision(
            "risk.es", model,
            detail=f"ES={es_currency:.0f}, ratio={ratio:.2f}" if ratio else "",
        )
        result["next_actions"] = ["risk.basel_es", "backtest.rolling", "report.excel"]
        return result

    @mcp.tool()
    def basel_es(
        session_id: str,
        model: str,
        portfolio_value: float = 1_000_000.0,
        base_horizon: int = 10,
    ) -> dict[str, Any]:
        """Compute a simplified Basel 97.5% Expected Shortfall.

        Implements the Basel MAR33 convention: 97.5% one-tailed ES,
        10-day base liquidity horizon. The simplified version does NOT
        include stressed calibration, risk-factor liquidity buckets,
        reduced-factor scaling, modellability tests, desk approval, or
        capital multipliers — it is a teaching benchmark only.

        Returns ES in currency, the equivalent VaR (for reference), and
        a clear disclaimer.
        """
        s = _store.get(session_id)
        if model not in s.fitted_models:
            return _error(f"Model '{model}' not fitted.", ["models.fit"])
        result = basel_es_97_5(
            s.fitted_models[model], s.returns,
            portfolio_value=portfolio_value,
            base_horizon=base_horizon,
        )
        s.risk_results[f"basel_es_{model}"] = result
        s.log_decision(
            "risk.basel_es", model,
            detail=f"ES={result['es_currency']:.0f}",
        )
        result["next_actions"] = ["backtest.rolling", "report.excel", "report.pdf"]
        result["disclaimer"] = (
            "Simplified teaching benchmark. Not a regulatory capital "
            "calculation. See Basel Framework MAR33 for full methodology."
        )
        return result

    @mcp.tool()
    def portfolio_var(
        session_id: str,
        weights: dict[str, float],
        confidence: float = 0.975,
        portfolio_value: float = 1_000_000.0,
        method: str = "normal",
    ) -> dict[str, Any]:
        """Decompose portfolio VaR into marginal, component, incremental.

        Args:
            weights: {asset_name: weight}; weights should sum to 1.0.
                Each asset must be loaded into a separate session first
                OR all assets' returns must be present in the session's
                `returns` as a DataFrame (multi-asset session).
            method: "normal" (variance-covariance), "historical", or "mc".
        """
        s = _store.get(session_id)
        result = portfolio_var_decomposition(
            s.returns, weights=weights,
            confidence=confidence,
            portfolio_value=portfolio_value,
            method=method,
        )
        s.risk_results["portfolio_var"] = result
        s.log_decision("risk.portfolio_var", "n/a", detail=f"method={method}")
        result["next_actions"] = ["report.excel", "feedback.get_next_action"]
        return result


def _error(msg: str, next_actions: list[str] | None = None) -> dict[str, Any]:
    return {"error": msg, "next_actions": next_actions or []}
