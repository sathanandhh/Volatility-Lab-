"""Tests for the Kupiec POF test and Christoffersen tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest.kupiec import kupiec_test
from core.backtest.christoffersen import (
    christoffersen_test, conditional_coverage,
)
from core.backtest.traffic_light import traffic_light_test


def _simulate_hits(n: int, p: float, seed: int = 42) -> pd.Series:
    """Simulate a breach sequence with given probability."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.random(n) < p)


# ── Kupiec POF ─────────────────────────────────────────────────────

def test_kupiec_accepts_correct_breach_rate():
    """When observed breach rate ≈ expected, Kupiec should not reject."""
    hits = _simulate_hits(1000, p=0.025, seed=42)
    stat, p = kupiec_test(hits, alpha=0.025)
    assert p > 0.05  # do not reject H0


def test_kupiec_rejects_wrong_breach_rate():
    """When breach rate is 10% but expected 2.5%, Kupiec should reject."""
    hits = _simulate_hits(1000, p=0.10, seed=42)
    stat, p = kupiec_test(hits, alpha=0.025)
    assert p < 0.05  # reject H0


def test_kupiec_statistic_non_negative():
    hits = _simulate_hits(500, p=0.05, seed=1)
    stat, p = kupiec_test(hits, alpha=0.05)
    assert stat >= 0


def test_kupiec_on_zero_breaches():
    """If expected α > 0 but zero breaches → underprediction → reject."""
    hits = pd.Series([False] * 500)
    stat, p = kupiec_test(hits, alpha=0.05)
    # With 0 breaches out of 500, Kupiec should flag (underprediction)
    assert p < 0.05


# ── Christoffersen independence ────────────────────────────────────

def test_christoffersen_independent_breaches():
    """Random (independent) breaches should not be rejected."""
    hits = _simulate_hits(1000, p=0.05, seed=7)
    stat, p = christoffersen_test(hits)
    assert p > 0.01  # should not strongly reject independence


def test_christoffersen_detects_clustering():
    """Clustered breaches (alternating runs) should be rejected."""
    # Construct a series where breaches cluster: 50 breaches, then 50 no, etc.
    hits = pd.Series([True] * 50 + [False] * 50 + [True] * 50 + [False] * 50)
    stat, p = christoffersen_test(hits)
    assert p < 0.05  # should reject independence


def test_conditional_coverage_combines_both():
    """Conditional coverage = Kupiec + Christoffersen."""
    hits = _simulate_hits(1000, p=0.05, seed=11)
    cc_stat, cc_p, kup_stat, ind_stat = conditional_coverage(hits, alpha=0.05)
    assert cc_stat >= kup_stat  # CC ≥ Kupiec
    assert cc_stat >= ind_stat  # CC ≥ independence
    assert 0 <= cc_p <= 1


# ── Basel Traffic Light ────────────────────────────────────────────

def test_traffic_light_green_zone():
    """0-4 breaches in 250 obs → green zone."""
    hits = pd.Series([False] * 248 + [True] * 2)  # 2 breaches
    result = traffic_light_test(hits, n_obs=250)
    assert result["zone"] == "green"
    assert result["capital_multiplier"] == 3.0


def test_traffic_light_yellow_zone():
    """5-9 breaches in 250 obs → yellow zone."""
    hits = pd.Series([False] * 240 + [True] * 10)
    result = traffic_light_test(hits, n_obs=250)
    assert result["zone"] == "yellow"
    assert 3.0 < result["capital_multiplier"] <= 3.85


def test_traffic_light_red_zone():
    """10+ breaches in 250 obs → red zone."""
    hits = pd.Series([False] * 235 + [True] * 15)
    result = traffic_light_test(hits, n_obs=250)
    assert result["zone"] == "red"
    assert result["capital_multiplier"] == 4.0


def test_traffic_light_scales_to_n_obs():
    """The test should scale 250-obs zones to the actual n."""
    # 2 breaches in 500 obs = 1 per 250 → green
    hits = pd.Series([False] * 498 + [True] * 2)
    result = traffic_light_test(hits, n_obs=500)
    assert result["zone"] == "green"
