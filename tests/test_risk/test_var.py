"""Tests for VaR and Expected Shortfall formulas."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.risk.var.parametric import (
    var_normal, var_student_t, var_cornish_fisher,
)
from core.risk.var.historical import var_historical
from core.risk.var.filtered_hist import var_fhs
from core.risk.es.expected_shortfall import (
    es_normal, es_student_t, es_historical,
)


# ── Parametric VaR ─────────────────────────────────────────────────

def test_var_normal_at_5pct():
    """Normal VaR at 5% should equal -1.6449σ."""
    v = var_normal(0, 1, 0.05)
    assert v < 0  # it's a loss
    assert abs(v - stats.norm.ppf(0.05)) < 1e-6


def test_var_normal_at_2_5pct():
    v = var_normal(0, 1, 0.025)
    assert abs(v - stats.norm.ppf(0.025)) < 1e-6
    assert abs(v - (-1.96)) < 0.01


def test_var_student_t_heavier_than_normal():
    """Student-t VaR should be more negative than Normal at same α."""
    v_normal = var_normal(0, 1, 0.025)
    v_t = var_student_t(0, 1, 0.025, nu=6)
    assert v_t < v_normal  # heavier tail → larger loss


def test_var_student_t_approaches_normal():
    """As ν → ∞, Student-t VaR should approach Normal VaR."""
    v_normal = var_normal(0, 1, 0.025)
    v_t_large = var_student_t(0, 1, 0.025, nu=1000)
    assert abs(v_t_large - v_normal) < 0.01


def test_var_cornish_fisher_close_to_normal_on_normal_data():
    """On normal data, CF VaR should be close to Normal VaR."""
    rng = np.random.default_rng(42)
    r = pd.Series(rng.standard_normal(5000))
    v_cf = var_cornish_fisher(r, 0, 1, 0.025)
    v_normal = var_normal(0, 1, 0.025)
    # Should be close (within 10% of the z-value)
    assert abs(v_cf - v_normal) < 0.3


# ── Historical VaR ─────────────────────────────────────────────────

def test_var_historical_on_uniform_data():
    """Historical VaR on known data should match the empirical quantile."""
    r = pd.Series(np.linspace(-10, 10, 1001))
    v = var_historical(r, alpha=0.05, horizon=1)
    assert abs(v - (-9.99)) < 0.1  # 5th percentile of uniform [-10, 10]


def test_var_historical_handles_short_series():
    r = pd.Series([1, 2, 3, 4, 5])
    v = var_historical(r, alpha=0.05, horizon=1)
    assert math.isnan(v) or v <= r.min()


# ── Filtered Historical Simulation ──────────────────────────────────

def test_var_fhs_produces_finite_result():
    rng = np.random.default_rng(42)
    std_resid = pd.Series(rng.standard_normal(500))
    v = var_fhs(std_resid, sigma=1.0, alpha=0.025)
    assert math.isfinite(v)
    assert v < 0  # loss


# ── Expected Shortfall ─────────────────────────────────────────────

def test_es_normal_positive_loss():
    """Normal ES at 2.5% should be a larger loss than VaR."""
    e = es_normal(0, 1, 0.025)
    v = var_normal(0, 1, 0.025)
    assert e < v  # ES > VaR in magnitude (more negative)


def test_es_normal_known_value():
    """Normal ES at α=0.025: E[L | L > VaR] = φ(z)/α."""
    z = stats.norm.ppf(0.025)
    expected_es = stats.norm.pdf(z) / 0.025
    e = es_normal(0, 1, 0.025)
    assert abs(e - expected_es) < 1e-6


def test_es_student_t_greater_than_normal():
    """Student-t ES should be a larger loss than Normal ES."""
    e_normal = es_normal(0, 1, 0.025)
    e_t = es_student_t(0, 1, 0.025, nu=6)
    assert e_t < e_normal  # more negative = larger loss


def test_es_historical_on_uniform_data():
    """Historical ES on uniform data should average the tail."""
    r = pd.Series(np.linspace(-10, 10, 1001))
    e = es_historical(r, alpha=0.05, horizon=1)
    # The tail beyond the 5% quantile is uniform on [-10, -9.9)
    # Mean of that tail is approximately -9.95
    assert -10.5 < e < -9.5


def test_es_to_var_ratio_indicates_tail_thickness():
    """ES/VaR ratio for Normal at 2.5% should be around 1.17."""
    v = var_normal(0, 1, 0.025)
    e = es_normal(0, 1, 0.025)
    ratio = abs(e / v)
    # For Normal: ES/VaR ≈ φ(z) / (α * |z|) ≈ 1.17 at α=2.5%
    assert 1.10 < ratio < 1.30


def test_es_to_var_ratio_higher_for_student_t():
    """Student-t ES/VaR ratio should be higher than Normal (heavier tail)."""
    v_n = var_normal(0, 1, 0.025)
    e_n = es_normal(0, 1, 0.025)
    ratio_n = abs(e_n / v_n)
    v_t = var_student_t(0, 1, 0.025, nu=5)
    e_t = es_student_t(0, 1, 0.025, nu=5)
    ratio_t = abs(e_t / v_t)
    assert ratio_t > ratio_n
