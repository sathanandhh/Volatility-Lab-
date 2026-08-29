"""Scenario shock and stressed-VaR tools.

Apply hypothetical market shocks to a fitted model and re-estimate
tail risk. Useful for stress testing and "what-if" analysis.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from core.risk.stressed import (
    stressed_var, stressed_es, list_stress_scenarios,
)
from core.risk.scenario import apply_shock
from core.feedback.session import SessionStore

from .session import _store


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def apply_market_shock(
        session_id: str,
        model: str,
        shock_pct: float,
        direction: str = "down",
    ) -> dict[str, Any]:
        """Apply a hypothetical market shock to a fitted model and re-forecast.

        Args:
            shock_pct: Magnitude of the shock in percent (e.g. 5.0 for
                a 5% move).
            direction: "down" (negative shock, typical stress) or "up".

        Returns the updated conditional variance path and volatility
        forecast for the next N periods. The original fitted model is
        not modified — a shocked copy is stored on the session.
        """
        s = _store.get(session_id)
        if model not in s.fitted_models:
            return _error(f"Model '{model}' not fitted.", ["models.fit"])
        sign = -1 if direction == "down" else 1
        shocked = apply_shock(s.fitted_models[model], shock=sign * shock_pct / 100)
        key = f"{model}_{direction}_{shock_pct}"
        s.shocked_models[key] = shocked
        s.log_decision(
            "scenario.apply_shock", model,
            detail=f"shock={sign*shock_pct}%, direction={direction}",
        )
        original_next_vol = float(s.fitted_models[model].next_vol)
        shocked_next_vol = float(shocked.next_vol)
        return {
            "model": model,
            "shock_pct": shock_pct,
            "direction": direction,
            "next_period_volatility_pct": shocked_next_vol * 100,
            "original_next_period_volatility_pct": original_next_vol * 100,
            "volatility_multiplier": (
                shocked_next_vol / original_next_vol
                if original_next_vol > 0 else None
            ),
            "next_actions": ["risk.var", "risk.es", "scenario.stressed_var"],
        }

    @mcp.tool()
    def stressed_var(
        session_id: str,
        model: str,
        confidence: float = 0.975,
        portfolio_value: float = 1_000_000.0,
        scenario_name: str = "gfc_2008",
    ) -> dict[str, Any]:
        """Compute stressed VaR using a historical crisis scenario.

        Pre-defined scenarios:
          gfc_2008, covid_2020, rates_2022, yen_carry_2024,
          oil_2014, euro_debt_2011.

        Returns the stressed VaR, the unstressed VaR (for comparison),
        and the stress multiplier.
        """
        s = _store.get(session_id)
        if model not in s.fitted_models:
            return _error(f"Model '{model}' not fitted.", ["models.fit"])
        result = stressed_var(
            s.fitted_models[model], s.returns,
            confidence=confidence,
            portfolio_value=portfolio_value,
            scenario_name=scenario_name,
        )
        s.risk_results[f"stressed_var_{model}_{scenario_name}"] = result
        s.log_decision(
            "scenario.stressed_var", model,
            detail=f"scenario={scenario_name}, "
                   f"VaR={result['stressed_var_currency']:.0f}",
        )
        result["next_actions"] = ["risk.basel_es", "report.excel", "report.pdf"]
        return result

    @mcp.tool()
    def list_stress_scenarios() -> list[dict[str, Any]]:
        """List the available pre-defined stress scenarios."""
        return list_stress_scenarios()


def _error(msg: str, next_actions: list[str] | None = None) -> dict[str, Any]:
    return {"error": msg, "next_actions": next_actions or []}
