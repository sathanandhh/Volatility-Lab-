"""Backtesting tools for VaR models.

Implements a walk-forward rolling refit engine plus the standard VaR
backtest suite: Kupiec POF, Christoffersen independence, conditional
coverage, Basel Traffic Light, Engle-Manganelli Dynamic Quantile, and
Diebold-Mariano forecast comparison.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from core.backtest.rolling import rolling_backtest
from core.backtest.kupiec import kupiec_test
from core.backtest.christoffersen import (
    christoffersen_test, conditional_coverage,
)
from core.backtest.traffic_light import traffic_light_test
from core.backtest.dynamic_quantile import dynamic_quantile_test
from core.backtest.diebold_mariano import diebold_mariano_test
from core.backtest.coverage import coverage_scorecard
from core.feedback.session import SessionStore

from .session import _store


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def rolling_backtest(
        session_id: str,
        models: list[str],
        confidence: float = 0.975,
        test_fraction: float = 0.2,
        refit_every: int | None = None,
        distribution: str | None = None,
    ) -> dict[str, Any]:
        """Run a walk-forward rolling VaR backtest.

        For each model in `models`, refits the model every
        `refit_every` observations and produces one-step-ahead VaR
        forecasts on the test sample. Returns per-model: number of
        breaches, breach rate, expected breach rate, and the VaR series.

        Args:
            models: List of fitted model names to backtest.
            confidence: VaR confidence level.
            test_fraction: Fraction of the sample reserved for testing
                (0.1-0.4).
            refit_every: Refit frequency. If None, uses session's
                optimized value or defaults to 10.
            distribution: Override; if None, uses each model's fitted
                distribution.

        After this, call backtest.coverage to run the statistical
        backtests (Kupiec, Christoffersen, etc.).
        """
        s = _store.get(session_id)
        for m in models:
            if m not in s.fitted_models:
                return _error(f"Model '{m}' not fitted.", ["models.fit"])
        if refit_every is None and "refit" in s.optimization_results:
            refit_every = s.optimization_results["refit"].optimal
        if refit_every is None:
            refit_every = 10
        alpha = 1 - confidence
        bt = rolling_backtest(
            s.returns, model_names=tuple(models),
            distribution=distribution or "t",
            train_size=int(len(s.returns) * (1 - test_fraction)),
            alpha=alpha, refit_every=refit_every,
        )
        s.backtest_results = {m: bt.for_model(m) for m in models}
        s.advance_stage("backtest")
        s.log_decision(
            "backtest.rolling", ",".join(models),
            detail=f"refit={refit_every}, α={alpha}",
        )
        return {
            "models": models,
            "test_window": int(len(s.returns) * test_fraction),
            "refit_every": refit_every,
            "alpha": alpha,
            "per_model": {
                m: {
                    "breaches": int(s.backtest_results[m].violations.sum()),
                    "breach_rate_pct": float(
                        s.backtest_results[m].violations.mean() * 100
                    ),
                    "expected_breach_rate_pct": alpha * 100,
                } for m in models
            },
            "next_actions": [
                "backtest.coverage", "backtest.kupiec",
                "backtest.christoffersen", "backtest.traffic_light",
            ],
        }

    @mcp.tool()
    def backtest_kupiec(session_id: str, model: str) -> dict[str, Any]:
        """Run the Kupiec POF (Proportion of Failures) test on a backtested model.

        Tests whether the observed breach rate matches the expected rate.
        p >= 0.05 → no evidence of mis-calibration.
        """
        s = _store.get(session_id)
        if model not in s.backtest_results:
            return _error(f"Model '{model}' not backtested.", ["backtest.rolling"])
        hits = s.backtest_results[model].violations
        stat, p = kupiec_test(hits, alpha=1 - _confidence(s))
        return {
            "model": model,
            "statistic": float(stat),
            "p_value": float(p),
            "verdict": "pass" if p >= 0.05 else "review",
            "next_actions": [
                "backtest.christoffersen", "backtest.conditional_coverage",
            ],
        }

    @mcp.tool()
    def backtest_christoffersen(session_id: str, model: str) -> dict[str, Any]:
        """Run the Christoffersen independence and conditional-coverage tests."""
        s = _store.get(session_id)
        if model not in s.backtest_results:
            return _error(f"Model '{model}' not backtested.", ["backtest.rolling"])
        hits = s.backtest_results[model].violations
        ind_stat, ind_p = christoffersen_test(hits)
        cc_stat, cc_p = conditional_coverage(hits, alpha=1 - _confidence(s))
        return {
            "model": model,
            "independence": {
                "statistic": float(ind_stat),
                "p_value": float(ind_p),
            },
            "conditional_coverage": {
                "statistic": float(cc_stat),
                "p_value": float(cc_p),
            },
            "verdict": "pass" if (ind_p >= 0.05 and cc_p >= 0.05) else "review",
            "next_actions": ["backtest.traffic_light", "backtest.diebold_mariano"],
        }

    @mcp.tool()
    def backtest_traffic_light(session_id: str, model: str) -> dict[str, Any]:
        """Run the Basel Traffic Light Backtest (green/yellow/red zones)."""
        s = _store.get(session_id)
        if model not in s.backtest_results:
            return _error(f"Model '{model}' not backtested.", ["backtest.rolling"])
        hits = s.backtest_results[model].violations
        result = traffic_light_test(hits, n_obs=len(hits))
        s.log_decision(
            "backtest.traffic_light", model,
            detail=f"zone={result['zone']}",
        )
        return {
            "model": model, **result,
            "next_actions": ["report.excel", "feedback.get_next_action"],
        }

    @mcp.tool()
    def backtest_dynamic_quantile(
        session_id: str, model: str,
    ) -> dict[str, Any]:
        """Run the Engle-Manganelli Dynamic Quantile test.

        A more powerful regression-based test that checks both coverage
        and independence jointly. p >= 0.05 → model is well-calibrated.
        """
        s = _store.get(session_id)
        if model not in s.backtest_results:
            return _error(f"Model '{model}' not backtested.", ["backtest.rolling"])
        bt = s.backtest_results[model]
        result = dynamic_quantile_test(
            bt.returns, bt.var[model], alpha=1 - _confidence(s),
        )
        return {
            "model": model, **result,
            "next_actions": ["backtest.diebold_mariano", "report.excel"],
        }

    @mcp.tool()
    def backtest_diebold_mariano(
        session_id: str, model_a: str, model_b: str,
    ) -> dict[str, Any]:
        """Compare two backtested models' forecast accuracy via Diebold-Mariano.

        Tests whether the difference in forecast accuracy (QLIKE loss)
        between two models is statistically significant.
        """
        s = _store.get(session_id)
        for m in (model_a, model_b):
            if m not in s.backtest_results:
                return _error(f"Model '{m}' not backtested.", ["backtest.rolling"])
        ba, bb = s.backtest_results[model_a], s.backtest_results[model_b]
        result = diebold_mariano_test(ba.returns, ba.variance, bb.variance)
        return {
            "model_a": model_a, "model_b": model_b, **result,
            "next_actions": ["report.excel", "feedback.get_next_action"],
        }

    @mcp.tool()
    def backtest_coverage_scorecard(session_id: str) -> dict[str, Any]:
        """Run the full coverage suite on all backtested models.

        Combines Kupiec POF, Christoffersen independence, conditional
        coverage, traffic light, and dynamic quantile into a single
        scorecard with a final pass/review verdict per model.
        """
        s = _store.get(session_id)
        if not s.backtest_results:
            return _error(
                "No backtests. Call backtest.rolling first.",
                ["backtest.rolling"],
            )
        scorecard = coverage_scorecard(
            s.backtest_results, alpha=1 - _confidence(s)
        )
        s.coverage_scorecard = scorecard
        s.log_decision(
            "backtest.coverage", "n/a",
            detail=f"{len(scorecard)} models",
        )
        return {
            "scorecard": scorecard,
            "next_actions": [
                "report.excel", "report.pdf", "feedback.get_next_action",
            ],
        }


def _confidence(session) -> float:
    """Recover the confidence level from the most recent risk result."""
    for v in session.risk_results.values():
        if "confidence" in v:
            return v["confidence"]
    return 0.975


def _error(msg: str, next_actions: list[str] | None = None) -> dict[str, Any]:
    return {"error": msg, "next_actions": next_actions or []}
