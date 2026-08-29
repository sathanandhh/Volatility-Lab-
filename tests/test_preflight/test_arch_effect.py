"""Tests for the Engle ARCH-LM pre-flight check — the most critical gate.

If the null of no-ARCH-effect cannot be rejected, the entire GARCH
family is statistically unjustified and the MCP server blocks further
model fitting.
"""
from __future__ import annotations

from core.preflight.checks._07_arch_effect import ArchEffectCheck
from core.preflight.gates import GateStatus


def test_arch_effect_passes_on_garch_data(garch_returns):
    """GARCH-simulated data SHOULD exhibit ARCH effect → check passes."""
    check = ArchEffectCheck()
    result = check.run(garch_returns)
    assert result.status == GateStatus.PASS
    assert result.p_value is not None
    assert result.p_value < 0.05  # strongly rejects no-ARCH-effect null


def test_arch_effect_blocks_on_white_noise(white_noise_returns):
    """Pure white noise has NO ARCH effect → check BLOCKS GARCH fitting."""
    check = ArchEffectCheck()
    result = check.run(white_noise_returns)
    assert result.status == GateStatus.BLOCK
    assert result.p_value > 0.10  # fails to reject no-ARCH-effect null


def test_arch_effect_check_returns_statistic(garch_returns):
    check = ArchEffectCheck()
    result = check.run(garch_returns)
    assert result.statistic is not None
    assert result.statistic > 0  # LM statistic is non-negative


def test_arch_effect_recommendation_blocks_garch(white_noise_returns):
    check = ArchEffectCheck()
    result = check.run(white_noise_returns)
    assert result.recommendation is not None
    assert "EWMA" in result.recommendation or "constant" in result.recommendation.lower()
