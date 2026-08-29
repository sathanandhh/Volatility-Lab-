"""Tests for residual diagnostics (Ljung-Box, ARCH-LM on residuals)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.diagnostics.residual import ljung_box
from core.diagnostics.arch_lm import arch_lm_test
from core.diagnostics.information_criteria import aic, bic
from core.diagnostics.accuracy import qlike, volatility_rmse, mae
from core.diagnostics.normality import jarque_bera, shapiro_wilk
from core.diagnostics.sign_bias import sign_bias_test


def test_ljung_box_on_white_noise():
    """White noise should have no serial correlation → high p-value."""
    rng = np.random.default_rng(42)
    s = pd.Series(rng.standard_normal(500))
    result = ljung_box(s, lag=10)
    assert result["p_value"] > 0.05


def test_ljung_box_on_autocorrelated():
    """AR(1) process should show serial correlation → low p-value."""
    rng = np.random.default_rng(42)
    n = 500
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + rng.standard_normal()
    s = pd.Series(x)
    result = ljung_box(s, lag=10)
    assert result["p_value"] < 0.05


def test_arch_lm_on_clean_residuals():
    """White-noise residuals should not show remaining ARCH effect."""
    rng = np.random.default_rng(42)
    s = pd.Series(rng.standard_normal(500))
    result = arch_lm_test(s, nlags=10)
    assert result["p_value"] > 0.05


def test_arch_lm_on_garch_residuals(garch_returns):
    """GARCH-simulated squared returns should show ARCH effect."""
    result = arch_lm_test(garch_returns, nlags=10)
    assert result["p_value"] < 0.05


def test_qlike_metric_is_finite(garch_returns):
    """QLIKE should produce a finite loss on a simple forecast."""
    cv = garch_returns.rolling(21).std()
    var = cv.pow(2).shift(1)
    result = qlike(garch_returns, var.dropna())
    assert np.isfinite(result)


def test_volatility_rmse_is_non_negative(garch_returns):
    cv = garch_returns.rolling(21).std()
    result = volatility_rmse(garch_returns, cv)
    assert result >= 0


def test_mae_is_non_negative(garch_returns):
    cv = garch_returns.rolling(21).std()
    result = mae(garch_returns, cv)
    assert result >= 0


def test_jarque_bera_on_normal_data():
    """Normal data should not reject normality."""
    rng = np.random.default_rng(42)
    s = pd.Series(rng.standard_normal(2000))
    result = jarque_bera(s)
    assert result["p_value"] > 0.01  # JB is sensitive but should pass on 2000 normal obs


def test_jarque_bera_on_garch_data(garch_returns):
    """GARCH returns are heavy-tailed → JB should reject normality."""
    result = jarque_bera(garch_returns)
    assert result["p_value"] < 0.05


def test_shapiro_wilk_on_normal_data():
    rng = np.random.default_rng(42)
    s = pd.Series(rng.standard_normal(500))
    result = shapiro_wilk(s)
    assert result["p_value"] > 0.01


def test_sign_bias_returns_negative_and_positive(garch_returns):
    result = sign_bias_test(garch_returns)
    assert "negative_p" in result
    assert "positive_p" in result
    assert 0 <= result["negative_p"] <= 1
    assert 0 <= result["positive_p"] <= 1
