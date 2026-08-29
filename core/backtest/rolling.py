"""Walk-forward rolling refit backtest engine."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.models.registry import get_model_spec
from core.models.univariate.arch_family import fit_arch_family
from core.models.distributions import var_quantile


@dataclass
class ModelBacktestResult:
    """Per-model backtest results."""
    model: str
    returns: pd.Series
    variance: pd.Series
    var: pd.Series
    violations: pd.Series  # bool: True if return < VaR

    def __post_init__(self) -> None:
        if not isinstance(self.violations, pd.Series):
            self.violations = pd.Series(self.violations)


@dataclass
class RollingBacktestResult:
    """Container for all per-model results."""
    results: dict[str, ModelBacktestResult] = field(default_factory=dict)

    def for_model(self, name: str) -> ModelBacktestResult:
        return self.results[name]


def rolling_backtest(
    returns: pd.Series, model_names: tuple[str, ...] = ("GARCH",),
    distribution: str = "t", train_size: int = 500,
    alpha: float = 0.025, refit_every: int = 10,
) -> RollingBacktestResult:
    """Walk-forward rolling backtest.

    For each model in `model_names`, refits the model every
    `refit_every` observations starting at `train_size`, then produces
    one-step-ahead VaR forecasts on the test sample.
    """
    r = returns.dropna()
    n = len(r)
    if train_size >= n:
        raise ValueError(f"train_size {train_size} ≥ n {n}")
    test_idx = r.index[train_size:]
    per_model: dict[str, ModelBacktestResult] = {}
    for name in model_names:
        spec = get_model_spec(name)
        if spec is None:
            raise ValueError(f"Unknown model: {name!r}")
        var_path = pd.Series(index=test_idx, dtype=float)
        var_series = pd.Series(index=test_idx, dtype=float)
        fit = None
        for j, pos in enumerate(range(train_size, n)):
            if fit is None or j % refit_every == 0:
                window = r.iloc[:pos]
                try:
                    o = spec.default_o if spec.supports_asymmetry else 0
                    fit = fit_arch_family(
                        window, family=spec.vol, p=spec.default_p, q=spec.default_q,
                        o=o, distribution=distribution, mean="Constant",
                    )
                except Exception:
                    continue
            # One-step-ahead variance forecast
            try:
                f = fit.extras["_arch_fit"].forecast(horizon=1, reindex=False)
                v = max(float(f.variance.iloc[-1, 0]), 1e-10)
            except Exception:
                continue
            mu = float(fit.params.get("mu", 0.0))
            sigma = math.sqrt(v)
            # VaR at the α quantile (loss tail)
            if distribution == "t":
                nu = float(fit.params.get("nu", 8.0))
                q = mu + sigma * var_quantile("t", alpha, nu=nu)
            else:
                q = mu + sigma * var_quantile("normal", alpha)
            dt = r.index[pos]
            var_path.loc[dt] = v
            var_series.loc[dt] = q
        # Compute violations: return < VaR (a breach)
        test_returns = r.iloc[train_size:]
        aligned_var = var_series.reindex(test_returns.index)
        violations = test_returns < aligned_var
        per_model[name] = ModelBacktestResult(
            model=name, returns=test_returns,
            variance=var_path, var=aligned_var,
            violations=violations.fillna(False),
        )
    return RollingBacktestResult(results=per_model)
