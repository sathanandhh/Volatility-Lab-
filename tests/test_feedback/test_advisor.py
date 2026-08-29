"""Tests for the feedback-loop advisor — recommends the next tool call."""
from __future__ import annotations

import pandas as pd
import pytest

from core.feedback.session import Session
from core.feedback.advisor import recommend_next_action, explain_decision
from core.feedback.decision_log import DecisionLogEntry
from core.feedback.recommendations import heuristic_recommendations


def test_advisor_empty_session_recommends_data():
    s = Session(id="adv-1")
    rec = recommend_next_action(s)
    assert rec.action == "data.load_market"
    assert "rationale" in rec.action or rec.rationale  # has a rationale


def test_advisor_data_stage_recommends_preflight():
    s = Session(id="adv-2")
    s.returns = pd.Series([1, 2, 3])
    s.advance_stage("data")
    rec = recommend_next_action(s)
    assert rec.action == "preflight.run"


def test_advisor_preflight_pass_recommends_optimize():
    from core.preflight.gates import GateResult, GateStatus, CheckResult
    s = Session(id="adv-3")
    s.returns = pd.Series([1, 2, 3])
    s.advance_stage("preflight")
    s.preflight_result = GateResult(
        checks=[CheckResult(name="arch_effect", status=GateStatus.PASS,
                            detail="pass")],
        overall=GateStatus.PASS,
    )
    rec = recommend_next_action(s)
    assert rec.action in ("optimize.order", "optimize.distribution",
                          "models.fit")


def test_advisor_preflight_block_recommends_explain():
    from core.preflight.gates import GateResult, GateStatus, CheckResult
    s = Session(id="adv-4")
    s.returns = pd.Series([1, 2, 3])
    s.advance_stage("preflight")
    s.preflight_result = GateResult(
        checks=[CheckResult(name="arch_effect", status=GateStatus.BLOCK,
                            detail="no ARCH effect")],
        overall=GateStatus.BLOCK,
    )
    rec = recommend_next_action(s)
    assert rec.action == "preflight.explain_gate"


def test_advisor_fit_stage_with_one_model_recommends_second():
    from core.models.univariate.arch_family import FitResult
    s = Session(id="adv-5")
    s.advance_stage("fit")
    # Add one fitted model
    s.fitted_models["GARCH"] = FitResult(
        name="GARCH(1,1,0)", params={"beta[1]": 0.9},
        conditional_volatility=pd.Series([1, 2, 3]),
        std_resid=pd.Series([0.1, 0.2, 0.3]),
        loglikelihood=-100, aic=200, bic=210,
        n_params=4, converged=True, next_vol=1.5,
    )
    rec = recommend_next_action(s)
    # Should recommend either fitting a second model or diagnostics
    assert rec.action in ("models.fit", "diagnostics.run", "compare.run")


def test_advisor_backtest_failed_recommends_reoptimize():
    s = Session(id="adv-6")
    s.advance_stage("backtest")
    s.coverage_scorecard = [
        {"model": "GARCH", "verdict": "review",
         "kupiec_p": 0.01, "independence_p": 0.5,
         "conditional_coverage_p": 0.02,
         "traffic_light_zone": "yellow",
         "traffic_light_multiplier": 3.4},
    ]
    rec = recommend_next_action(s)
    assert rec.action in ("optimize.refit", "optimize.window",
                          "optimize.distribution")


def test_advisor_backtest_passed_recommends_report():
    s = Session(id="adv-7")
    s.advance_stage("backtest")
    s.coverage_scorecard = [
        {"model": "GARCH", "verdict": "pass",
         "kupiec_p": 0.5, "independence_p": 0.5,
         "conditional_coverage_p": 0.5,
         "traffic_light_zone": "green",
         "traffic_light_multiplier": 3.0},
    ]
    rec = recommend_next_action(s)
    assert rec.action == "report.excel"


def test_advisor_report_stage_recommends_explain():
    s = Session(id="adv-8")
    s.advance_stage("report")
    rec = recommend_next_action(s)
    assert rec.action == "feedback.explain_decision"


def test_advisor_returns_alternatives():
    """Recommendations should include alternatives when relevant."""
    s = Session(id="adv-9")
    rec = recommend_next_action(s)
    assert hasattr(rec, "alternatives")
    assert isinstance(rec.alternatives, list)


def test_advisor_returns_suggested_args():
    s = Session(id="adv-10")
    rec = recommend_next_action(s)
    assert hasattr(rec, "args")
    assert isinstance(rec.args, dict)


# ── explain_decision ──────────────────────────────────────────────

def test_explain_decision_returns_explanation():
    entry = DecisionLogEntry(
        id=1, timestamp="2024-01-01T00:00:00Z",
        action="preflight.run", target="n/a",
        outcome="pass", detail="12/12 checks passed",
    )
    result = explain_decision(entry)
    assert "explanation" in result
    assert "preflight.run" in result["explanation"]
    assert "pass" in result["explanation"]


# ── heuristic_recommendations ─────────────────────────────────────

def test_heuristic_recommendations_empty_session():
    s = Session(id="adv-11")
    recs = heuristic_recommendations(s)
    assert isinstance(recs, list)
    # Should at least return a "no issues" message
    assert len(recs) > 0


def test_heuristic_recommendations_with_failed_backtest():
    from core.preflight.gates import GateResult, GateStatus, CheckResult
    s = Session(id="adv-12")
    s.returns = pd.Series([1, 2, 3])
    s.advance_stage("backtest")
    s.coverage_scorecard = [
        {"model": "GARCH", "verdict": "review",
         "kupiec_p": 0.01, "independence_p": 0.5,
         "conditional_coverage_p": 0.02,
         "traffic_light_zone": "red",
         "traffic_light_multiplier": 4.0},
    ]
    recs = heuristic_recommendations(s)
    # Should mention the failed Kupiec test
    assert any("Kupiec" in r for r in recs)
    assert any("red" in r.lower() for r in recs)
