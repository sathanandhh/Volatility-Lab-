"""Stressed VaR / ES via historical crisis scenarios."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Crisis scenarios: (name, start_date, end_date, description)
SCENARIO_LIBRARY = [
    {"name": "gfc_2008", "start": "2008-09-15", "end": "2008-12-31",
     "description": "Global Financial Crisis — Lehman collapse to year-end."},
    {"name": "covid_2020", "start": "2020-02-19", "end": "2020-04-07",
     "description": "COVID-19 crash — peak-to-trough equity drawdown."},
    {"name": "rates_2022", "start": "2022-01-03", "end": "2022-10-12",
     "description": "Global rates shock — Fed hiking cycle."},
    {"name": "yen_carry_2024", "start": "2024-07-10", "end": "2024-08-05",
     "description": "Yen carry-trade unwind."},
    {"name": "oil_2014", "start": "2014-06-19", "end": "2015-01-30",
     "description": "Oil price collapse."},
    {"name": "euro_debt_2011", "start": "2011-07-01", "end": "2011-12-19",
     "description": "European sovereign-debt crisis."},
]


def list_stress_scenarios() -> list[dict]:
    return SCENARIO_LIBRARY


def stressed_var(fit, returns: pd.Series, confidence: float = 0.975,
                 portfolio_value: float = 1_000_000.0,
                 scenario_name: str = "gfc_2008") -> dict:
    """Compute stressed VaR using a historical crisis sub-sample."""
    scenario = next((s for s in SCENARIO_LIBRARY if s["name"] == scenario_name), None)
    if scenario is None:
        raise ValueError(f"Unknown scenario: {scenario_name!r}")
    mask = (returns.index >= pd.Timestamp(scenario["start"])) & \
           (returns.index <= pd.Timestamp(scenario["end"]))
    crisis = returns.loc[mask]
    if len(crisis) < 30:
        # Fallback: use full-sample worst 60-day window
        crisis = returns.tail(60)
    alpha = 1 - confidence
    stressed_var_rate = float(crisis.quantile(alpha))
    stressed_var_currency = portfolio_value * abs(stressed_var_rate)
    # Unstressed VaR from the fit's latest conditional volatility
    sigma = fit.next_vol if hasattr(fit, "next_vol") else float(fit.conditional_volatility.iloc[-1])
    mu = float(fit.params.get("mu", 0.0))
    from scipy import stats
    unstressed_var_rate = mu + sigma * stats.norm.ppf(alpha)
    unstressed_var_currency = portfolio_value * abs(unstressed_var_rate)
    multiplier = stressed_var_currency / unstressed_var_currency if unstressed_var_currency else None
    return {
        "model": fit.name,
        "scenario": scenario_name,
        "scenario_window": f"{scenario['start']} to {scenario['end']}",
        "scenario_description": scenario["description"],
        "stressed_var_return_pct": stressed_var_rate * 100,
        "stressed_var_currency": stressed_var_currency,
        "unstressed_var_return_pct": unstressed_var_rate * 100,
        "unstressed_var_currency": unstressed_var_currency,
        "stress_multiplier": float(multiplier) if multiplier else None,
        "n_crisis_obs": int(len(crisis)),
    }


def stressed_es(fit, returns: pd.Series, confidence: float = 0.975,
                portfolio_value: float = 1_000_000.0,
                scenario_name: str = "gfc_2020") -> dict:
    """Stressed ES using a historical crisis sub-sample."""
    from core.risk.es.expected_shortfall import es_historical
    scenario = next((s for s in SCENARIO_LIBRARY if s["name"] == scenario_name), None)
    if scenario is None:
        raise ValueError(f"Unknown scenario: {scenario_name!r}")
    mask = (returns.index >= pd.Timestamp(scenario["start"])) & \
           (returns.index <= pd.Timestamp(scenario["end"]))
    crisis = returns.loc[mask]
    if len(crisis) < 30:
        crisis = returns.tail(60)
    alpha = 1 - confidence
    es_rate = es_historical(crisis, alpha)
    es_currency = portfolio_value * abs(es_rate)
    return {
        "model": fit.name,
        "scenario": scenario_name,
        "scenario_window": f"{scenario['start']} to {scenario['end']}",
        "stressed_es_return_pct": es_rate * 100,
        "stressed_es_currency": es_currency,
        "n_crisis_obs": int(len(crisis)),
    }
