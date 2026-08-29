"""Tests for the sample-size pre-flight check."""
from __future__ import annotations

from core.preflight.checks._01_sample_size import SampleSizeCheck
from core.preflight.gates import GateStatus


def test_sample_size_passes_on_garch_data(garch_returns):
    check = SampleSizeCheck()
    result = check.run(garch_returns)
    assert result.status == GateStatus.PASS
    assert "1000" in result.detail


def test_sample_size_warns_on_short_garch_data(garch_returns_short):
    check = SampleSizeCheck()
    result = check.run(garch_returns_short)
    # 500 is the EGARCH minimum — should at least pass for GARCH
    assert result.status in (GateStatus.PASS, GateStatus.WARN)


def test_sample_size_blocks_on_tiny_data(tiny_returns):
    check = SampleSizeCheck()
    result = check.run(tiny_returns)
    assert result.status == GateStatus.BLOCK
    assert "300" in result.detail or "500" in result.detail


def test_sample_size_check_name():
    assert SampleSizeCheck.name == "sample_size"
