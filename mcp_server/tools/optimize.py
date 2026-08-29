"""Input optimization tools.

These tools remove guesswork from model specification. Instead of the
user or agent picking p, q, distribution, or window size, each tool
sweeps a search space and returns the optimum by an information or
out-of-sample criterion. The full sweep table is also returned so the
agent can justify the choice.

All optimization results are stored on the session and used by
`feedback.get_next_action` and `models.fit` defaults.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from core.optimize.order_selector import select_order
from core.optimize.distribution_selector import select_distribution
from core.optimize.mean_selector import select_mean
from core.optimize.window_selector import select_window
from core.optimize.refit_selector import select_refit_frequency
from core.optimize.horizon_selector import select_horizon
from core.feedback.session import SessionStore

from .session import _store


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def optimize_order(
        session_id: str,
        model_family: str = "GARCH",
        max_p: int = 3,
        max_q: int = 2,
        distribution: str = "t",
        criterion: str = "AIC",
    ) -> dict[str, Any]:
        """Select optimal (p, q) orders for an ARCH-family model.

        Sweeps p in [1..max_p], q in [0..max_q], fits each combination,
        and returns the configuration with the lowest AIC (or BIC if
        `criterion='BIC'`). Returns the full sweep table so the agent
        can explain why (1,1) was preferred over (2,1) — e.g. "ΔAIC < 2,
        simpler model preferred."

        Args:
            session_id: Active session id.
            model_family: "ARCH", "GARCH", "GJR-GARCH", or "EGARCH".
            max_p: Maximum ARCH order to consider (1-5).
            max_q: Maximum GARCH order to consider (0-3).
            distribution: Innovation distribution used during sweep.
            criterion: "AIC" (default) or "BIC".
        """
        s = _store.get(session_id)
        _require_returns(s)
        result = select_order(
            s.returns, family=model_family, max_p=max_p, max_q=max_q,
            distribution=distribution, criterion=criterion,
        )
        s.optimization_results["order"] = result
        s.log_decision(
            "optimize.order", result.optimal,
            detail=f"criterion={criterion}, score={result.optimal_score:.2f}",
        )
        return _format_opt_result("order", result)

    @mcp.tool()
    def optimize_distribution(
        session_id: str,
        model_family: str = "GARCH",
        p: int = 1,
        q: int = 1,
        candidates: list[str] | None = None,
    ) -> dict[str, Any]:
        """Select optimal innovation distribution.

        Fits the specified model with each candidate distribution and
        picks the one with the lowest AIC. Default candidates: Normal,
        Student-t, GED, Skew-t, JSU.

        Critical when the pre-flight normality check returns `block`:
        Normal will almost always lose to Student-t here.
        """
        s = _store.get(session_id)
        _require_returns(s)
        cands = candidates or ["normal", "t", "ged", "skewt", "jsu"]
        result = select_distribution(
            s.returns, family=model_family, p=p, q=q, candidates=cands,
        )
        s.optimization_results["distribution"] = result
        s.log_decision(
            "optimize.distribution", result.optimal,
            detail=f"AIC={result.optimal_score:.2f}",
        )
        return _format_opt_result("distribution", result)

    @mcp.tool()
    def optimize_mean(
        session_id: str,
        candidates: list[str] | None = None,
    ) -> dict[str, Any]:
        """Select optimal mean specification.

        Candidates: "Constant", "AR(1)", "AR(2)", "ARMA(1,1)". Selection
        criterion: Ljung-Box p-value > 0.05 on the residuals of the mean
        model (i.e. no remaining serial correlation in the mean).
        """
        s = _store.get(session_id)
        _require_returns(s)
        cands = candidates or ["Constant", "AR(1)", "ARMA(1,1)"]
        result = select_mean(s.returns, candidates=cands)
        s.optimization_results["mean"] = result
        s.log_decision(
            "optimize.mean", result.optimal, detail=result.optimal_score
        )
        return _format_opt_result("mean", result)

    @mcp.tool()
    def optimize_window(
        session_id: str,
        min_window: int = 10,
        max_window: int = 63,
        step: int = 5,
        metric: str = "QLIKE",
    ) -> dict[str, Any]:
        """Select optimal rolling-window size for backtesting.

        Sweeps window sizes and evaluates out-of-sample volatility
        forecast accuracy (default: QLIKE). Returns the window with
        best accuracy, plus the full sweep table.

        Only relevant if you intend to run `backtest.rolling` afterwards.
        """
        s = _store.get(session_id)
        _require_returns(s)
        result = select_window(
            s.returns, min_window=min_window, max_window=max_window,
            step=step, metric=metric,
        )
        s.optimization_results["window"] = result
        s.log_decision(
            "optimize.window", result.optimal,
            detail=f"{metric}={result.optimal_score:.4f}",
        )
        return _format_opt_result("window", result)

    @mcp.tool()
    def optimize_refit(
        session_id: str,
        model_name: str = "GARCH",
        min_freq: int = 1,
        max_freq: int = 30,
        alpha: float = 0.025,
    ) -> dict[str, Any]:
        """Select optimal refit frequency for rolling backtest.

        Sweeps refit-every-N values and picks the one that maximizes
        conditional-coverage p-value from the backtest. Lower N adapts
        faster but is slower; higher N is faster but may lag regime
        shifts.
        """
        s = _store.get(session_id)
        _require_returns(s)
        result = select_refit_frequency(
            s.returns, model_name=model_name,
            min_freq=min_freq, max_freq=max_freq, alpha=alpha,
        )
        s.optimization_results["refit"] = result
        s.log_decision(
            "optimize.refit", result.optimal,
            detail=f"CC_p={result.optimal_score:.3f}",
        )
        return _format_opt_result("refit", result)

    @mcp.tool()
    def optimize_horizon(
        session_id: str,
        target_use: str = "regulatory",
    ) -> dict[str, Any]:
        """Recommend a forecast horizon based on intended use.

        Args:
            target_use: One of "regulatory" (10-day Basel), "weekly"
                (5-day), "monthly" (21-day), "stress" (60-day), or "auto"
                (based on half-life of squared-return autocorrelation).
        """
        s = _store.get(session_id)
        _require_returns(s)
        result = select_horizon(s.returns, target_use=target_use)
        s.optimization_results["horizon"] = result
        return _format_opt_result("horizon", result)

    @mcp.tool()
    def get_optimization_summary(session_id: str) -> dict[str, Any]:
        """Return the consolidated optimization results for this session.

        Useful before calling models.fit — the fit tool will use these
        optimized inputs as defaults if the agent does not override them.
        """
        s = _store.get(session_id)
        return {k: v.summary() for k, v in s.optimization_results.items()}


def _require_returns(session) -> None:
    if session.returns is None:
        raise ValueError(
            "No data loaded. Call data.load_market or data.load_csv first."
        )


def _format_opt_result(kind: str, result) -> dict[str, Any]:
    return {
        "kind": kind,
        "optimal": result.optimal,
        "optimal_score": result.optimal_score,
        "criterion": result.criterion,
        "sweep": result.sweep_table,
        "recommendation": result.recommendation,
        "next_actions": _next_actions_for(kind),
    }


def _next_actions_for(kind: str) -> list[str]:
    return {
        "order":        ["optimize.distribution", "optimize.mean", "models.fit"],
        "distribution": ["optimize.order", "models.fit"],
        "mean":         ["optimize.order", "models.fit"],
        "window":       ["optimize.refit", "backtest.rolling"],
        "refit":        ["backtest.rolling"],
        "horizon":      ["models.forecast", "risk.var"],
    }.get(kind, ["models.fit"])
