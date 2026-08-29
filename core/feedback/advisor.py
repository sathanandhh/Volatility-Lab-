"""LLM-facing advisor that recommends the next tool call."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.feedback.session import Session


@dataclass
class Recommendation:
    action: str
    args: dict[str, Any]
    rationale: str
    alternatives: list = field(default_factory=list)


def recommend_next_action(session: Session, intent: str = "auto") -> Recommendation:
    """Inspect the session and return the recommended next action."""
    # Stage-based defaults
    stage = session.current_stage
    if stage == "empty":
        return Recommendation(
            action="data.load_market",
            args={"ticker": "^NSEI", "years": 5, "frequency": "Daily"},
            rationale="No data loaded yet. Start by loading market data.",
            alternatives=[
                Recommendation(action="data.load_csv", args={},
                                rationale="Alternatively, load from CS"),
            ],
        )
    if stage == "data":
        return Recommendation(
            action="preflight.run",
            args={},
            rationale="Data loaded; run pre-flight checks before any model fit.",
            alternatives=[],
        )
    if stage == "preflight":
        # If preflight blocked, recommend fixing
        if session.preflight_result and session.preflight_result.overall.value == "block":
            blocked = [c.name for c in session.preflight_result.checks
                       if c.status.value == "block"]
            return Recommendation(
                action="preflight.explain_gate",
                args={"check_name": blocked[0] if blocked else ""},
                rationale=f"Preflight blocked on {blocked}. Explain and address first.",
                alternatives=[
                    Recommendation(action="optimize.distribution", args={},
                                    rationale="If normality blocked, switch distribution."),
                ],
            )
        return Recommendation(
            action="optimize.order",
            args={},
            rationale="Preflight passed/warned. Optimize (p, q) before fitting.",
            alternatives=[
                Recommendation(action="optimize.distribution", args={},
                               rationale="Or optimize distribution first."),
                Recommendation(action="models.fit", args={"model": "GARCH"},
                               rationale="Or fit directly with defaults."),
            ],
        )
    if stage == "optimize":
        if "order" not in session.optimization_results:
            return Recommendation(
                action="optimize.order", args={},
                rationale="Optimize (p, q) by AIC before fitting.",
                alternatives=[],
            )
        if "distribution" not in session.optimization_results:
            return Recommendation(
                action="optimize.distribution", args={},
                rationale="Order optimized. Now optimize distribution.",
                alternatives=[],
            )
        return Recommendation(
            action="models.fit",
            args={"model": "GARCH"},
            rationale="Inputs optimized. Fit a model.",
            alternatives=[
                Recommendation(action="models.fit", args={"model": "EGARCH"},
                               rationale="Try EGARCH for leverage."),
            ],
        )
    if stage == "fit":
        # If only one model fit, suggest fitting a second for comparison
        if len(session.fitted_models) < 2:
            return Recommendation(
                action="models.fit",
                args={"model": "EGARCH" if "EGARCH" not in session.fitted_models else "GJR-GARCH"},
                rationale="Fit a second model for comparison.",
                alternatives=[
                    Recommendation(action="diagnostics.run", args={},
                                   rationale="Or run diagnostics on the existing fit."),
                ],
            )
        return Recommendation(
            action="compare.run", args={},
            rationale="Multiple models fit; compare them.",
            alternatives=[
                Recommendation(action="risk.var", args={},
                               rationale="Or skip straight to VaR."),
            ],
        )
    if stage == "compare":
        if session.comparison_result:
            best = session.comparison_result.get("best_forecast")
            return Recommendation(
                action="risk.var",
                args={"model": best or "GARCH", "method": "student_t"},
                rationale=f"Best forecast model is {best}. Compute VaR.",
                alternatives=[],
            )
    if stage == "risk":
        return Recommendation(
            action="backtest.rolling",
            args={"models": list(session.fitted_models.keys()),
                  "confidence": 0.975, "test_fraction": 0.2},
            rationale="Risk computed. Backtest the VaR forecasts.",
            alternatives=[],
        )
    if stage == "backtest":
        if session.coverage_scorecard is None:
            return Recommendation(
                action="backtest.coverage", args={},
                rationale="Backtest done. Run coverage suite.",
                alternatives=[],
            )
        # Check if any test failed
        any_failed = any(row.get("verdict") == "review"
                         for row in session.coverage_scorecard)
        if any_failed:
            return Recommendation(
                action="optimize.refit", args={},
                rationale="Coverage failed. Re-tune refit interval.",
                alternatives=[
                    Recommendation(action="optimize.window", args={},
                                   rationale="Or re-tune rolling window."),
                ],
            )
        return Recommendation(
            action="report.excel", args={},
            rationale="Coverage passed. Build the Excel report.",
            alternatives=[
                Recommendation(action="report.markdown", args={},
                               rationale="Or build a Markdown summary."),
            ],
        )
    if stage == "report":
        return Recommendation(
            action="feedback.explain_decision", args={},
            rationale="Reports built. Explain the decision audit trail.",
            alternatives=[],
        )
    # Fallback
    return Recommendation(
        action="feedback.get_next_action", args={},
        rationale="Unclear state — request explicit next action.",
        alternatives=[],
    )


def explain_decision(entry, session: Session = None) -> dict[str, Any]:
    """Render a decision log entry as a human-readable explanation."""
    return {
        "id": entry.id,
        "timestamp": entry.timestamp,
        "action": entry.action,
        "target": entry.target,
        "outcome": entry.outcome,
        "detail": entry.detail,
        "explanation": (
            f"At {entry.timestamp}, the analysis performed '{entry.action}' "
            f"on '{entry.target}' with outcome '{entry.outcome}'. "
            f"Detail: {entry.detail}."
        ),
    }
