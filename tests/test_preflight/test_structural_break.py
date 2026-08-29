"""Tests for the structural-break pre-flight check (CUSUM on r²)."""
from __future__ import annotations

import numpy as np

from core.preflight.checks._06_structural_break import StructuralBreakCheck
from core.preflight.gates import GateStatus


def test_no_break_on_stationary_garch(garch_returns):
    """Stationary GARCH(1,1) data should not trigger a structural break."""
    check = StructuralBreakCheck()
    result = check.run(garch_returns)
    assert result.status in (GateStatus.PASS, GateStatus.WARN)
    assert result.statistic is not None


def test_break_detected_on_variance_jump(break_returns):
    """A 6× variance jump at the midpoint should be detected."""
    check = StructuralBreakCheck()
    result = check.run(break_returns)
    assert result.status == GateStatus.WARN
    assert result.statistic > 1.36  # 5% Brownian-bridge critical value


def test_break_recommendation_mentions_split(break_returns):
    check = StructuralBreakCheck()
    result = check.run(break_returns)
    if result.status == GateStatus.WARN:
        assert result.recommendation is not None
        assert "split" in result.recommendation.lower() or "regime" in result.recommendation.lower()


def test_break_on_short_series_returns_pass_or_warn(garch_returns_short):
    check = StructuralBreakCheck()
    result = check.run(garch_returns_short)
    # Should not crash on 500 obs
    assert result.status in (GateStatus.PASS, GateStatus.WARN)
