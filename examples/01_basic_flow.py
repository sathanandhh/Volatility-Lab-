#!/usr/bin/env python3
"""Example 1 — Basic volatility analysis flow.

Demonstrates the simplest end-to-end workflow using the core engine
directly (no MCP transport):

    data.load → preflight.run → optimize.order → models.fit
    → models.forecast → risk.compute_var

Run:
    python examples/01_basic_flow.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.data.service import DataService
from core.preflight.orchestrator import PreflightOrchestrator
from core.preflight.gates import GateStatus
from core.optimize.order_selector import select_order
from core.models.univariate.arch_family import (
    fit_arch_family, forecast_arch_family,
)
from core.risk.var.parametric import var_student_t
from core.risk.es.expected_shortfall import es_student_t


def main() -> None:
    print("=" * 70)
    print("EXAMPLE 1 · Basic Volatility Analysis Flow")
    print("=" * 70)

    # ── Step 1: Load market data ────────────────────────────────────
    print("\n1. Loading Reliance Industries (5 years daily)...")
    service = DataService.from_env()
    prices = service.download_prices("RELIANCE.NS", years=5)
    close, returns = service.prepare_returns(prices, frequency="Daily")
    print(f"   Loaded {len(returns)} observations "
          f"({returns.index[0].date()} to {returns.index[-1].date()})")
    print(f"   Annualized volatility: {returns.std() * math.sqrt(252) * 100:.2f}%")

    # ── Step 2: Run pre-flight checks ────────────────────────────────
    print("\n2. Running pre-flight gates...")
    orch = PreflightOrchestrator()
    gate = orch.run(returns)
    print(f"   Overall status: {gate.overall.value}")
    for c in gate.checks:
        symbol = {"pass": "✓", "warn": "⚠", "block": "✗"}.get(c.status.value, "?")
        print(f"   {symbol} {c.name:30s} {c.detail}")
    if gate.overall == GateStatus.BLOCK:
        print("\n   ⛔ Preflight blocked — GARCH is not justified for this data.")
        print("   Recommendations:")
        for r in gate.recommendations:
            print(f"     → {r}")
        return

    # ── Step 3: Optimize (p, q) orders ──────────────────────────────
    print("\n3. Optimizing (p, q) by AIC...")
    order = select_order(returns, family="GARCH", max_p=3, max_q=2,
                          distribution="t")
    print(f"   Optimal: p={order.optimal['p']}, q={order.optimal['q']} "
          f"(AIC={order.optimal_score:.1f})")
    print(f"   {order.recommendation}")

    # ── Step 4: Fit GARCH ────────────────────────────────────────────
    print("\n4. Fitting GARCH(1,1) with Student-t innovations...")
    fit = fit_arch_family(
        returns, family="GARCH",
        p=order.optimal["p"], q=order.optimal["q"], o=0,
        distribution="t", mean="Constant",
    )
    print(f"   Converged: {fit.converged}")
    print(f"   AIC: {fit.aic:.1f}  BIC: {fit.bic:.1f}")
    print(f"   Parameters:")
    for k, v in fit.params.items():
        print(f"     {k:15s} = {v:.5f}")
    print(f"   Latest conditional volatility: {fit.next_vol:.4f}%")

    # ── Step 5: Forecast 10-day volatility ───────────────────────────
    print("\n5. Forecasting 10-day volatility path...")
    forecast = forecast_arch_family(fit, horizon=10, method="analytic")
    print(f"   Method: {forecast.method}")
    print(f"   Day 1 vol: {math.sqrt(forecast.variance_path[0]) * 100:.3f}%")
    print(f"   Day 10 vol: {math.sqrt(forecast.variance_path[-1]) * 100:.3f}%")
    avg_vol = math.sqrt(sum(forecast.variance_path) / len(forecast.variance_path))
    print(f"   10-day avg vol: {avg_vol * 100:.3f}%")

    # ── Step 6: Compute VaR and ES ───────────────────────────────────
    print("\n6. Computing 97.5% one-day VaR and ES...")
    alpha = 0.025
    sigma = math.sqrt(forecast.variance_path[0])
    nu = float(fit.params.get("nu", 8.0))
    portfolio = 1_000_000  # ₹10 lakh portfolio

    var_rate = var_student_t(0, sigma, alpha, nu)
    es_rate = es_student_t(0, sigma, alpha, nu)
    var_currency = portfolio * abs(var_rate)
    es_currency = portfolio * abs(es_rate)

    print(f"   σ (1-day): {sigma * 100:.3f}%")
    print(f"   ν (Student-t df): {nu:.2f}")
    print(f"   VaR (97.5%): ₹{var_currency:,.0f} ({var_rate * 100:.3f}%)")
    print(f"   ES  (97.5%): ₹{es_currency:,.0f} ({es_rate * 100:.3f}%)")
    print(f"   ES/VaR ratio: {abs(es_rate / var_rate):.2f}× "
          f"(> 1.3 indicates heavy tail)")

    print("\n" + "=" * 70)
    print("Done. This is a basic flow — see example 02 for the feedback loop.")
    print("=" * 70)


if __name__ == "__main__":
    main()
