"""Tests for the arch_family wrapper (fit + forecast)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.models.univariate.arch_family import (
    fit_arch_family, forecast_arch_family, FitResult,
)
from core.models.univariate.ewma import fit_ewma, forecast_ewma


def test_fit_garch_converges(garch_returns):
    fit = fit_arch_family(garch_returns, family="GARCH", p=1, q=1, o=0,
                          distribution="t", mean="Constant")
    assert fit.converged
    assert fit.n_params > 0
    assert math.isfinite(fit.aic)
    assert math.isfinite(fit.bic)
    assert len(fit.conditional_volatility) == len(garch_returns)
    assert len(fit.std_resid) > 0


def test_fit_garch_has_beta_param(garch_returns):
    fit = fit_arch_family(garch_returns, family="GARCH", p=1, q=1, o=0,
                          distribution="t", mean="Constant")
    assert "beta[1]" in fit.params
    assert 0 <= fit.params["beta[1]]"] if "beta[1]]" in fit.params else fit.params["beta[1]"] <= 1


def test_fit_egarch(garch_returns):
    fit = fit_arch_family(garch_returns, family="EGARCH", p=1, q=1, o=1,
                          distribution="t", mean="Constant")
    assert fit.converged
    assert "gamma[1]" in fit.params


def test_fit_gjr_garch(garch_returns):
    fit = fit_arch_family(garch_returns, family="GARCH", p=1, q=1, o=1,
                          distribution="t", mean="Constant")
    assert fit.converged
    assert "gamma[1]" in fit.params


def test_forecast_garch_one_step(garch_returns):
    fit = fit_arch_family(garch_returns, family="GARCH", p=1, q=1, o=0,
                          distribution="t", mean="Constant")
    f = forecast_arch_family(fit, horizon=1, method="analytic")
    assert len(f.variance_path) == 1
    assert f.variance_path[0] > 0


def test_forecast_garch_multi_step(garch_returns):
    fit = fit_arch_family(garch_returns, family="GARCH", p=1, q=1, o=0,
                          distribution="t", mean="Constant")
    f = forecast_arch_family(fit, horizon=10, method="analytic")
    assert len(f.variance_path) == 10
    assert all(v >= 0 for v in f.variance_path)


def test_forecast_egarch_uses_simulation(garch_returns):
    """EGARCH multi-step forecast should use simulation (not analytic)."""
    fit = fit_arch_family(garch_returns, family="EGARCH", p=1, q=1, o=1,
                          distribution="t", mean="Constant")
    f = forecast_arch_family(fit, horizon=5, method="analytic")
    # The wrapper auto-switches to simulation for EGARCH h>1
    assert f.method == "simulation"
    assert len(f.variance_path) == 5


def test_fit_normal_distribution(garch_returns):
    fit = fit_arch_family(garch_returns, family="GARCH", p=1, q=1, o=0,
                          distribution="normal", mean="Constant")
    assert fit.converged
    # Normal dist should NOT have nu parameter
    assert "nu" not in fit.params


def test_fit_student_t_has_nu(garch_returns):
    fit = fit_arch_family(garch_returns, family="GARCH", p=1, q=1, o=0,
                          distribution="t", mean="Constant")
    assert "nu" in fit.params
    assert fit.params["nu"] > 2  # must be > 2 for finite variance


def test_fit_raises_on_short_series(tiny_returns):
    with pytest.raises(ValueError, match="50"):
        fit_arch_family(tiny_returns, family="GARCH", p=1, q=1, o=0,
                        distribution="t", mean="Constant")


def test_ewma_fit(garch_returns):
    fit = fit_ewma(garch_returns, decay=0.94)
    assert fit.converged
    assert fit.params["lambda"] == 0.94
    assert len(fit.conditional_volatility) > 0
    assert fit.next_vol > 0


def test_ewma_forecast_flat(garch_returns):
    """EWMA forecast is flat (no mean reversion)."""
    fit = fit_ewma(garch_returns, decay=0.94)
    f = forecast_ewma(fit, horizon=5)
    assert len(f.variance_path) == 5
    # All periods should be the same (flat)
    assert all(abs(v - f.variance_path[0]) < 1e-10 for v in f.variance_path)
