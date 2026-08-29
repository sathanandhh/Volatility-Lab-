"""Pytest fixtures: synthetic GARCH data, sessions, and shared helpers.

Run with:
    pytest tests/ -v
    pytest tests/ -v -k "preflight"   # only preflight tests
    pytest tests/ -v --tb=short        # shorter tracebacks
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.feedback.session import Session


# ── Synthetic return series ─────────────────────────────────────────

def _simulate_garch(
    n: int = 1000,
    seed: int = 42,
    mu: float = 0.0,
    omega: float = 0.05,
    alpha: float = 0.10,
    beta: float = 0.85,
    nu: float = 6.0,
) -> np.ndarray:
    """Simulate GARCH(1,1) with Student-t innovations (unit variance)."""
    rng = np.random.default_rng(seed)
    var = np.full(n, omega / max(1 - alpha - beta, 1e-6))
    ret = np.zeros(n)
    for t in range(1, n):
        z = rng.standard_t(nu) / np.sqrt(max((nu - 2) / nu, 1e-6))
        ret[t] = mu + np.sqrt(var[t - 1]) * z
        var[t] = omega + alpha * ret[t - 1] ** 2 + beta * var[t - 1]
    return ret


@pytest.fixture
def garch_returns() -> pd.Series:
    """1000 obs of GARCH(1,1) percentage returns (Student-t innovations).

    Has clear ARCH effect and volatility clustering — preflight should pass.
    """
    ret = _simulate_garch(n=1000, seed=42)
    idx = pd.bdate_range("2020-01-01", periods=1000)
    return pd.Series(ret, index=idx, name="Return (%)")


@pytest.fixture
def garch_returns_short() -> pd.Series:
    """500 obs of GARCH(1,1) — enough for EGARCH but short for some checks."""
    ret = _simulate_garch(n=500, seed=99)
    idx = pd.bdate_range("2019-01-01", periods=500)
    return pd.Series(ret, index=idx, name="Return (%)")


@pytest.fixture
def white_noise_returns() -> pd.Series:
    """1000 obs of pure white noise — NO ARCH effect.

    The arch_effect preflight check should BLOCK on this series.
    """
    rng = np.random.default_rng(123)
    idx = pd.bdate_range("2020-01-01", periods=1000)
    return pd.Series(rng.standard_normal(1000), index=idx, name="Return (%)")


@pytest.fixture
def tiny_returns() -> pd.Series:
    """50 obs — too small for GARCH fitting.

    The sample_size preflight check should BLOCK on this series.
    """
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=50)
    return pd.Series(rng.standard_normal(50), index=idx, name="Return (%)")


@pytest.fixture
def break_returns() -> pd.Series:
    """1000 obs with a structural break at observation 500 (variance jumps 6×)."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal(500) * 0.5
    b = rng.standard_normal(500) * 3.0
    combined = np.concatenate([a, b])
    idx = pd.bdate_range("2020-01-01", periods=1000)
    return pd.Series(combined, index=idx, name="Return (%)")


# ── Session fixtures ───────────────────────────────────────────────

@pytest.fixture
def empty_session() -> Session:
    """A fresh session with no data loaded."""
    return Session(id="test-empty-0001")


@pytest.fixture
def data_session(garch_returns: pd.Series) -> Session:
    """A session with returns loaded but no preflight run."""
    s = Session(id="test-data-0001")
    s.returns = garch_returns
    s.metadata = {
        "ticker": "TEST",
        "frequency": "Daily",
        "years": 5,
        "source": "synthetic",
    }
    s.advance_stage("data")
    return s


# ── GARCH simulation helper exposed for direct use ─────────────────

@pytest.fixture
def simulate_garch():
    """Expose the _simulate_garch function for tests that need custom params."""
    return _simulate_garch
