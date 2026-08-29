"""Model fitting and forecasting tools.

The fit tool refuses to run if any pre-flight check is in `block` status
unless `force=True` is supplied (logged to the decision log). When
called without explicit arguments, it uses the session's optimized
inputs as defaults — making the feedback loop self-reinforcing.
"""
from __future__ import annotations

import math
from typing import Any

from mcp.server.fastmcp import FastMCP

from core.models.registry import list_models, get_model_spec, ModelSpec
from core.models.univariate.arch_family import (
    fit_arch_family, forecast_arch_family,
)
from core.models.univariate.ewma import fit_ewma, forecast_ewma
from core.feedback.session import SessionStore

from .session import _store


def register(mcp: FastMCP) -> None:

    @mcp.tool()
    def list_models(family: str | None = None) -> list[dict[str, Any]]:
        """List available volatility models.

        Args:
            family: Optional filter — "univariate", "multivariate",
                "stochastic", "ml", or None for all.

        Returns a list of model specs with name, family, default (p,q),
        supported distributions, and a short description.
        """
        return [m.to_dict() for m in list_models(family=family)]

    @mcp.tool()
    def fit_model(
        session_id: str,
        model: str,
        p: int | None = None,
        q: int | None = None,
        o: int | None = None,
        distribution: str | None = None,
        mean: str = "Constant",
        force: bool = False,
    ) -> dict[str, Any]:
        """Fit a volatility model to the session's returns.

        Args:
            session_id: Active session id.
            model: Model name (e.g. "ARCH", "GARCH", "GJR-GARCH",
                "EGARCH", "FIGARCH", "EWMA", "DCC-GARCH", "Realized-GARCH").
            p, q, o: ARCH/GARCH/asymmetry orders. If None, use the
                session's optimized values (from optimize.order) or
                model defaults.
            distribution: "normal", "t", "ged", "skewt", "jsu". If None,
                use the session's optimized value or default "t".
            mean: "Constant", "AR(1)", "ARMA(1,1)", or "Zero".
            force: If True, bypass pre-flight block status (logged).

        Refuses to fit if preflight has any `block` check unless
        `force=True`. Returns parameter estimates, convergence status,
        AIC/BIC, and conditional volatility series metadata.
        """
        s = _store.get(session_id)
        if s.returns is None:
            return _error(
                "No data loaded. Call data.load_market or data.load_csv first."
            )
        if s.preflight_result is None:
            return _error(
                "Preflight not run. Call preflight.run first.",
                next_actions=["preflight.run"],
            )
        blocked = [
            c.name for c in s.preflight_result.checks if c.status == "block"
        ]
        if blocked and not force:
            return {
                "status": "blocked",
                "blocked_checks": blocked,
                "message": (
                    "Preflight has blocking issues. Fix them or call "
                    "with force=True."
                ),
                "next_actions": [
                    "preflight.explain_gate", "optimize.distribution",
                    "optimize.order",
                ],
            }
        # Resolve defaults from optimization results
        opt = s.optimization_results
        if p is None and "order" in opt:
            p = opt["order"].optimal.get("p", 1)
        if q is None and "order" in opt:
            q = opt["order"].optimal.get("q", 1)
        if distribution is None and "distribution" in opt:
            distribution = opt["distribution"].optimal
        if distribution is None:
            distribution = "t"
        if p is None: p = 1
        if q is None: q = 1
        if o is None:
            o = 1 if model in ("GJR-GARCH", "EGARCH") else 0

        spec = get_model_spec(model)
        if spec is None:
            return _error(
                f"Unknown model '{model}'. Call models.list_models."
            )

        if model == "EWMA":
            fit = fit_ewma(s.returns, decay=0.94)
        else:
            fit = fit_arch_family(
                s.returns, family=spec.family, p=p, q=q, o=o,
                distribution=distribution, mean=mean,
            )
        s.fitted_models[model] = fit
        s.last_fit_config = {
            "model": model, "p": p, "q": q, "o": o,
            "distribution": distribution, "mean": mean, "force": force,
        }
        s.advance_stage("fit")
        s.log_decision(
            "models.fit", model,
            detail=f"p={p},q={q},o={o},dist={distribution},"
                   f"force={force}, converged={fit.converged}",
        )
        return {
            "model": model,
            "converged": fit.converged,
            "n_params": int(fit.n_params),
            "log_likelihood": float(fit.loglikelihood),
            "aic": float(fit.aic),
            "bic": float(fit.bic),
            "params": {k: float(v) for k, v in fit.params.items()},
            "conditional_volatility_tail": [
                {"date": str(d.date()), "vol": float(v)}
                for d, v in fit.conditional_volatility.tail(5).items()
            ],
            "next_actions": ["diagnostics.run", "compare.run", "models.forecast"],
        }

    @mcp.tool()
    def forecast(
        session_id: str,
        model: str,
        horizon: int = 10,
        method: str = "analytic",
        simulations: int = 1000,
    ) -> dict[str, Any]:
        """Produce a multi-step volatility forecast from a fitted model.

        Args:
            session_id: Active session id.
            model: Name of a previously fitted model.
            horizon: Number of periods ahead to forecast (1-60).
            method: "analytic" (closed-form, ARCH/GARCH only),
                "simulation" (required for EGARCH beyond 1 step),
                or "bootstrap" (filtered historical).
            simulations: Number of paths if method is simulation/bootstrap.

        Returns the variance path, annualized volatility path, and the
        expected average volatility over the horizon.
        """
        s = _store.get(session_id)
        if model not in s.fitted_models:
            return _error(
                f"Model '{model}' not fitted.", next_actions=["models.fit"]
            )
        fit = s.fitted_models[model]
        if model == "EWMA":
            f = forecast_ewma(fit, horizon=horizon)
        else:
            # EGARCH requires simulation beyond 1 step
            if model == "EGARCH" and horizon > 1 and method == "analytic":
                method = "simulation"
            f = forecast_arch_family(
                fit, horizon=horizon, method=method, simulations=simulations,
            )
        var_path = f.variance_path
        vol_path = [math.sqrt(v) for v in var_path]
        return {
            "model": model,
            "horizon": horizon,
            "method": method,
            "variance_path": [float(v) for v in var_path],
            "volatility_path_pct": [v * 100 for v in vol_path],
            "avg_volatility_pct": float(
                sum(vol_path) / len(vol_path) * 100
            ),
            "next_actions": ["risk.var", "risk.es", "backtest.rolling"],
        }


def _error(msg: str, next_actions: list[str] | None = None) -> dict[str, Any]:
    return {"error": msg, "next_actions": next_actions or []}
