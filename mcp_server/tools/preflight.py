"""Pre-flight gate tools.

Before any statistical model is fit, the data must pass pre-flight
checks. `run_preflight` executes 12 statistical checks and returns an
aggregated GateResult with status pass/warn/block plus recommendations
and next actions. `models.fit` refuses to run while any check is in
`block` status unless `force=True` is supplied (which is logged to the
decision log).
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from core.preflight.orchestrator import PreflightOrchestrator, GateResult
from core.feedback.session import SessionStore

from .session import _store


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def run_preflight(session_id: str) -> dict[str, Any]:
        """Run all pre-flight statistical checks on the loaded returns.

        Executes 12 checks:
          1.  sample_size         (min 300 GARCH / 500 EGARCH)
          2.  missing_data       (max gap, % missing)
          3.  outliers           (Hampel, rolling 4σ)
          4.  zero_infinity      (bad values — WTI April-2020 case)
          5.  stationarity       (ADF + KPSS)
          6.  structural_break   (Bai-Perron, CUSUM)
          7.  arch_effect        (Engle LM — IS GARCH EVEN NEEDED?)
          8.  volatility_clustering (Ljung-Box on r²)
          9.  normality          (Jarque-Bera, Shapiro-Wilk, A-D)
          10. mean_specification (LB on constant-mean residuals)
          11. leverage_asymmetry (sign-bias test)
          12. frequency_adequacy  (enough obs for weekly/monthly?)

        Returns a GateResult with overall status pass/warn/block, per-
        check details, recommendations, and the list of next actions
        (typically optimize.* and/or models.fit, depending on outcomes).
        """
        s = _store.get(session_id)
        if s.returns is None:
            return _error(
                "No data loaded. Call data.load_market or data.load_csv first."
            )
        orchestrator = PreflightOrchestrator()
        result: GateResult = orchestrator.run(s.returns)
        s.preflight_result = result
        s.advance_stage("preflight")
        s.log_decision(
            action="preflight.run",
            outcome=result.overall,
            detail=f"{sum(1 for c in result.checks if c.status=='pass')}/"
                   f"{len(result.checks)} checks passed",
        )
        return result.to_dict()

    @mcp.tool()
    def get_gate_status(session_id: str) -> dict[str, Any]:
        """Return the most recent pre-flight gate result without re-running it.

        Returns None if preflight has not yet been run for this session.
        """
        s = _store.get(session_id)
        if s.preflight_result is None:
            return {"status": "not_run", "next_actions": ["preflight.run"]}
        return s.preflight_result.to_dict()

    @mcp.tool()
    def explain_gate(session_id: str, check_name: str) -> dict[str, Any]:
        """Explain a single pre-flight check's result in plain language.

        Useful when the agent needs to help a user understand why a fit
        was blocked or why a warning was issued. Returns the check
        detail, the test statistic and p-value (if any), and a plain-
        language explanation.
        """
        s = _store.get(session_id)
        if s.preflight_result is None:
            return _error("Preflight has not been run. Call preflight.run first.")
        check = next(
            (c for c in s.preflight_result.checks if c.name == check_name), None
        )
        if check is None:
            return _error(
                f"No check named '{check_name}'. Available: "
                + ", ".join(c.name for c in s.preflight_result.checks)
            )
        return {
            "name": check.name,
            "status": check.status,
            "detail": check.detail,
            "statistic": check.statistic,
            "p_value": check.p_value,
            "explanation": check.explain(),
            "recommendation": check.recommendation,
        }


def _error(msg: str) -> dict[str, Any]:
    return {"error": msg, "next_actions": ["data.load_market", "data.load_csv"]}
