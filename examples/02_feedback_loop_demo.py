#!/usr/bin/env python3
"""Example 2 — Full feedback-loop-driven analysis.

Demonstrates the iterative workflow that the MCP server drives:
each step checks diagnostics, and if they fail, the loop returns to
optimize or fit with a different specification.

    preflight → optimize → fit → diagnostics
        → if fail: re-optimize → re-fit → re-diagnostics
        → compare → risk → backtest
        → if fail: re-tune refit → re-backtest
        → report

Run:
    python examples/02_feedback_loop_demo.py
"""
from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np
import pandas as pd

from core.data.service import DataService
from core.feedback.session import Session
from core.feedback.advisor import recommend_next_action
from core.preflight.orchestrator import PreflightOrchestrator
from core.preflight.gates import GateStatus
from core.optimize.order_selector import select_order
from core.optimize.distribution_selector import select_distribution
from core.optimize.refit_selector import select_refit_frequency
from core.models.univariate.arch_family import (
    fit_arch_family, forecast_arch_family,
)
from core.diagnostics.residual import ljung_box
from core.diagnostics.arch_lm import arch_lm_test
from core.diagnostics.accuracy import qlike
from core.risk.var.parametric import var_student_t
from core.backtest.rolling import rolling_backtest
from core.backtest.coverage import coverage_scorecard
from core.reporting.markdown import build_markdown_report


def main() -> None:
    print("=" * 70)
    print("EXAMPLE 2 · Feedback Loop Demo")
    print("=" * 70)

    # ── Initialize session ───────────────────────────────────────────
    session = Session(id="demo-feedback-001")

    # ── Step 1: Load data ────────────────────────────────────────────
    print("\n── Step 1: Load data ─────────────────────────────────────")
    service = DataService.from_env()
    prices = service.download_prices("^NSEI", years=5)
    close, returns = service.prepare_returns(prices, frequency="Daily")
    session.returns = returns
    session.metadata = {"ticker": "^NSEI", "frequency": "Daily"}
    session.advance_stage("data")
    session.log_decision("data.load", "^NSEI", "loaded",
                         detail=f"{len(returns)} obs")
    print(f"   Loaded {len(returns)} observations")

    # ── Step 2: Advisor recommends preflight ────────────────────────
    print("\n── Step 2: Advisor recommends next action ────────────────")
    rec = recommend_next_action(session)
    print(f"   → {rec.action}: {rec.rationale}")
    assert rec.action == "preflight.run"

    # ── Step 3: Run preflight ────────────────────────────────────────
    print("\n── Step 3: Run pre-flight gates ─────────────────────────")
    orch = PreflightOrchestrator()
    gate = orch.run(returns)
    session.preflight_result = gate
    session.advance_stage("preflight")
    session.log_decision("preflight.run", "n/a", gate.overall.value,
                         detail=f"{len(gate.checks)} checks")
    print(f"   Overall: {gate.overall.value}")
    for c in gate.checks:
        sym = {"pass": "✓", "warn": "⚠", "block": "✗"}.get(c.status.value, "?")
        print(f"   {sym} {c.name}")
    if gate.overall == GateStatus.BLOCK:
        print("\n   ⛔ Blocked — cannot proceed with GARCH.")
        return

    # ── Step 4: Optimize inputs ──────────────────────────────────────
    print("\n── Step 4: Optimize (p, q) and distribution ─────────────")
    order = select_order(returns, family="GARCH", max_p=3, max_q=2,
                         distribution="t")
    session.optimization_results["order"] = order
    print(f"   Order: p={order.optimal['p']}, q={order.optimal['q']} "
          f"(AIC={order.optimal_score:.1f})")

    dist = select_distribution(returns, family="GARCH",
                               p=order.optimal["p"], q=order.optimal["q"],
                               candidates=["normal", "t"])
    session.optimization_results["distribution"] = dist
    print(f"   Distribution: {dist.optimal} (AIC={dist.optimal_score:.1f})")
    session.advance_stage("optimize")

    # ── Step 5: Fit model ─ (feedback loop starts here) ─────────────
    max_iterations = 5
    for iteration in range(1, max_iterations + 1):
        print(f"\n── Step 5: Fit GARCH (iteration {iteration}) ──────────")
        fit = fit_arch_family(
            returns, family="GARCH",
            p=order.optimal["p"], q=order.optimal["q"], o=0,
            distribution=dist.optimal, mean="Constant",
        )
        session.fitted_models[f"GARCH_iter{iteration}"] = fit
        session.advance_stage("fit")
        session.log_decision("models.fit", "GARCH", "fitted",
                             detail=f"iter={iteration}, AIC={fit.aic:.1f}")
        print(f"   Converged: {fit.converged}, AIC: {fit.aic:.1f}")

        # ── Step 6: Run diagnostics ────────────────────────────────
        print(f"   Running diagnostics...")
        lb_sq = ljung_box(fit.std_resid.pow(2), lag=10)
        arch = arch_lm_test(fit.std_resid, nlags=10)
        print(f"   LB(r²) p={lb_sq['p_value']:.4f} "
              f"({'PASS' if lb_sq['p_value'] >= 0.05 else 'FAIL'})")
        print(f"   ARCH-LM p={arch['p_value']:.4f} "
              f"({'PASS' if arch['p_value'] >= 0.05 else 'FAIL'})")

        if lb_sq["p_value"] >= 0.05 and arch["p_value"] >= 0.05:
            print("   ✓ Diagnostics passed — proceeding to risk & backtest.")
            break

        # ── Feedback: diagnostics failed → re-optimize ─────────────
        if lb_sq["p_value"] < 0.05:
            print("   ⚠ Remaining ARCH effect — increasing q...")
            new_q = min(order.optimal["q"] + 1, 3)
            order = select_order(returns, family="GARCH", max_p=3,
                                 max_q=new_q + 1, distribution=dist.optimal)
            session.optimization_results["order"] = order
            print(f"      New optimum: p={order.optimal['p']}, "
                  f"q={order.optimal['q']}")

        if arch["p_value"] < 0.05 and iteration > 1:
            print("   ⚠ ARCH-LM still failing — switching to EGARCH...")
            fit = fit_arch_family(
                returns, family="EGARCH",
                p=1, q=1, o=1,
                distribution=dist.optimal, mean="Constant",
            )
            session.fitted_models["EGARCH"] = fit
            session.log_decision("models.fit", "EGARCH", "fitted",
                                 detail=f"AIC={fit.aic:.1f}")
            arch2 = arch_lm_test(fit.std_resid, nlags=10)
            print(f"   EGARCH ARCH-LM p={arch2['p_value']:.4f}")
            if arch2["p_value"] >= 0.05:
                print("   ✓ EGARCH passed — using it.")
                fit = session.fitted_models["EGARCH"]
                break
    else:
        print(f"\n   ⚠ Max iterations ({max_iterations}) reached — "
              "proceeding with best fit.")

    # ── Step 7: Compute VaR ─────────────────────────────────────────
    print("\n── Step 7: Compute 97.5% VaR ─────────────────────────────")
    forecast = forecast_arch_family(fit, horizon=1, method="analytic")
    sigma = math.sqrt(forecast.variance_path[0])
    nu = float(fit.params.get("nu", 8.0))
    alpha = 0.025
    var_rate = var_student_t(0, sigma, alpha, nu)
    portfolio = 1_000_000
    print(f"   1-day σ: {sigma * 100:.3f}%")
    print(f"   97.5% VaR: ₹{portfolio * abs(var_rate):,.0f}")
    session.risk_results["var"] = {
        "var_return_pct": var_rate * 100,
        "var_currency": portfolio * abs(var_rate),
    }
    session.advance_stage("risk")

    # ── Step 8: Backtest ─ (feedback loop continues) ─────────────────
    print("\n── Step 8: Rolling backtest ───────────────────────────────")
    print("   Optimizing refit frequency...")
    refit = select_refit_frequency(returns, model_name="GARCH",
                                   min_freq=5, max_freq=20, alpha=alpha)
    session.optimization_results["refit"] = refit
    print(f"   Optimal refit: every {refit.optimal} obs "
          f"(CC p={refit.optimal_score:.3f})")

    print("   Running walk-forward backtest...")
    bt = rolling_backtest(
        returns, model_names=("GARCH",), distribution="t",
        train_size=int(len(returns) * 0.8), alpha=alpha,
        refit_every=refit.optimal,
    )
    session.backtest_results = {m: bt.for_model(m) for m in ("GARCH",)}
    session.advance_stage("backtest")

    # ── Step 9: Coverage scorecard ──────────────────────────────────
    print("\n── Step 9: Coverage scorecard ────────────────────────────")
    scorecard = coverage_scorecard(session.backtest_results, alpha=alpha)
    session.coverage_scorecard = scorecard
    for row in scorecard:
        print(f"   {row['model']}: {row['verdict'].upper()}")
        print(f"     Breaches: {row['breaches']}, "
              f"Kupiec p={row['kupiec_p']:.3f}, "
              f"CC p={row['conditional_coverage_p']:.3f}, "
              f"Zone: {row['traffic_light_zone']}")

        if row["verdict"] == "review":
            print("   ⚠ Backtest failed — would recommend re-tuning "
                  "in a real session.")

    # ── Step 10: Generate report ─────────────────────────────────────
    print("\n── Step 10: Generate Markdown report ────────────────────")
    session.advance_stage("report")
    md = build_markdown_report(session)
    print(f"   Report length: {len(md)} characters")
    print(f"   First 200 chars:\n   {md[:200]}...")

    # ── Step 11: Decision audit trail ───────────────────────────────
    print(f"\n── Step 11: Decision audit trail "
          f"({len(session.decision_log)} decisions) ──────")
    for e in session.decision_log.entries:
        print(f"   #{e.id} {e.action:25s} → {e.outcome}")

    # ── Final advisor recommendation ────────────────────────────────
    print("\n── Final advisor recommendation ──────────────────────────")
    final_rec = recommend_next_action(session)
    print(f"   → {final_rec.action}: {final_rec.rationale}")

    print("\n" + "=" * 70)
    print("Feedback loop demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
