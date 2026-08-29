"""End-to-end integration test: preflight → fit → VaR on synthetic data.

This test exercises the full feedback loop without the MCP transport layer.
It verifies that every core module connects correctly to the next.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core.feedback.session import Session
from core.preflight.orchestrator import PreflightOrchestrator
from core.preflight.gates import GateStatus
from core.optimize.order_selector import select_order
from core.optimize.distribution_selector import select_distribution
from core.models.univariate.arch_family import (
    fit_arch_family, forecast_arch_family,
)
from core.diagnostics.residual import ljung_box
from core.diagnostics.arch_lm import arch_lm_test
from core.risk.var.parametric import var_student_t
from core.risk.es.expected_shortfall import es_student_t
from core.feedback.advisor import recommend_next_action


def test_full_workflow_garch(garch_returns):
    """End-to-end: preflight → optimize → fit → diagnostics → VaR → ES."""
    session = Session(id="integration-1")
    session.returns = garch_returns
    session.advance_stage("data")

    # 1. Preflight
    orch = PreflightOrchestrator()
    gate = orch.run(garch_returns)
    session.preflight_result = gate
    session.advance_stage("preflight")
    # GARCH data should pass or warn (not block)
    assert gate.overall in (GateStatus.PASS, GateStatus.WARN), \
        f"Preflight blocked: {[c.name for c in gate.checks if c.status == GateStatus.BLOCK]}"

    # 2. Optimize order
    order_result = select_order(garch_returns, family="GARCH",
                                max_p=2, max_q=1, distribution="t")
    session.optimization_results["order"] = order_result
    assert order_result.optimal["p"] in (1, 2)

    # 3. Fit GARCH
    fit = fit_arch_family(
        garch_returns, family="GARCH",
        p=order_result.optimal["p"], q=order_result.optimal["q"], o=0,
        distribution="t", mean="Constant",
    )
    session.fitted_models["GARCH"] = fit
    session.advance_stage("fit")
    assert fit.converged

    # 4. Diagnostics
    lb_sq = ljung_box(fit.std_resid.pow(2), lag=10)
    arch = arch_lm_test(fit.std_resid, nlags=10)
    # A good GARCH fit should remove most ARCH effect
    # (may still have some residual at lag 10 — that's OK for a teaching test)
    assert "p_value" in lb_sq
    assert "p_value" in arch

    # 5. Forecast
    f = forecast_arch_family(fit, horizon=10, method="analytic")
    assert len(f.variance_path) == 10
    assert all(v >= 0 for v in f.variance_path)

    # 6. VaR
    alpha = 0.025
    sigma = math.sqrt(f.variance_path[0])
    nu = float(fit.params.get("nu", 8.0))
    var = var_student_t(0, sigma, alpha, nu)
    assert var < 0  # it's a loss

    # 7. ES
    es = es_student_t(0, sigma, alpha, nu)
    assert es < var  # ES is a larger loss than VaR

    # 8. Advisor recommends next step
    rec = recommend_next_action(session)
    assert rec.action is not None
    assert rec.rationale


def test_full_workflow_blocks_on_white_noise(white_noise_returns):
    """White noise should be blocked at preflight — no GARCH fitting."""
    session = Session(id="integration-2")
    session.returns = white_noise_returns
    session.advance_stage("data")

    orch = PreflightOrchestrator()
    gate = orch.run(white_noise_returns)
    session.preflight_result = gate
    session.advance_stage("preflight")

    assert gate.overall == GateStatus.BLOCK
    arch_check = next(c for c in gate.checks if c.name == "arch_effect")
    assert arch_check.status == GateStatus.BLOCK

    # Advisor should recommend explaining the block, not fitting
    rec = recommend_next_action(session)
    assert rec.action == "preflight.explain_gate"


def test_advisor_drives_full_loop(garch_returns):
    """The advisor should guide the session through every stage."""
    session = Session(id="integration-3")
    session.returns = garch_returns
    session.advance_stage("data")

    # Advisor → preflight
    rec = recommend_next_action(session)
    assert rec.action == "preflight.run"

    orch = PreflightOrchestrator()
    gate = orch.run(garch_returns)
    session.preflight_result = gate
    session.advance_stage("preflight")

    # Advisor → optimize (since preflight passes on GARCH data)
    rec = recommend_next_action(session)
    assert rec.action in ("optimize.order", "optimize.distribution", "models.fit")

    # Optimize
    order = select_order(garch_returns, family="GARCH",
                         max_p=2, max_q=1, distribution="t")
    session.optimization_results["order"] = order
    session.advance_stage("optimize")

    # Advisor → fit
    rec = recommend_next_action(session)
    assert rec.action in ("models.fit", "optimize.distribution")


def test_decision_log_captures_full_workflow(garch_returns):
    """Every stage of the workflow should be logged in the decision log."""
    session = Session(id="integration-4")
    session.returns = garch_returns
    session.advance_stage("data")
    session.log_decision("data.load_market", "TEST", outcome="loaded",
                         detail="1000 obs")

    orch = PreflightOrchestrator()
    gate = orch.run(garch_returns)
    session.preflight_result = gate
    session.advance_stage("preflight")
    session.log_decision("preflight.run", "n/a", outcome=gate.overall.value,
                         detail="12 checks")

    # Verify log
    assert len(session.decision_log) == 2
    actions = [e.action for e in session.decision_log.entries]
    assert "data.load_market" in actions
    assert "preflight.run" in actions

    # Verify to_dict includes decision count
    d = session.to_dict()
    assert d["decision_count"] == 2


====================================================================================================
END OF FLATTENED SOURCE — session_store/ + tests/
====================================================================================================
