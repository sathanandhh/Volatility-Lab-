"""Tests for the pre-flight orchestrator and GateResult aggregation."""
from __future__ import annotations

from core.preflight.orchestrator import PreflightOrchestrator
from core.preflight.gates import GateStatus


def test_orchestrator_runs_all_12_checks(garch_returns):
    orch = PreflightOrchestrator()
    result = orch.run(garch_returns)
    assert len(result.checks) == 12
    expected_names = {
        "sample_size", "missing_data", "outliers", "zero_infinity",
        "stationarity", "structural_break", "arch_effect",
        "volatility_clustering", "normality", "mean_specification",
        "leverage_asymmetry", "frequency_adequacy",
    }
    assert {c.name for c in result.checks} == expected_names


def test_orchestrator_passes_on_garch(garch_returns):
    orch = PreflightOrchestrator()
    result = orch.run(garch_returns)
    assert result.overall in (GateStatus.PASS, GateStatus.WARN)
    # ARCH effect check must pass (it's the most important)
    arch = next(c for c in result.checks if c.name == "arch_effect")
    assert arch.status == GateStatus.PASS


def test_orchestrator_blocks_on_white_noise(white_noise_returns):
    orch = PreflightOrchestrator()
    result = orch.run(white_noise_returns)
    assert result.overall == GateStatus.BLOCK
    arch = next(c for c in result.checks if c.name == "arch_effect")
    assert arch.status == GateStatus.BLOCK


def test_gate_result_to_dict_has_next_actions(garch_returns):
    orch = PreflightOrchestrator()
    result = orch.run(garch_returns)
    d = result.to_dict()
    assert "overall" in d
    assert "checks" in d
    assert "recommendations" in d
    assert "next_actions" in d
    assert isinstance(d["next_actions"], list)
    assert len(d["next_actions"]) > 0


def test_gate_result_next_actions_includes_optimize(garch_returns):
    orch = PreflightOrchestrator()
    result = orch.run(garch_returns)
    d = result.to_dict()
    assert "optimize.order" in d["next_actions"] or "models.fit" in d["next_actions"]


def test_blocked_result_recommends_fix(white_noise_returns):
    orch = PreflightOrchestrator()
    result = orch.run(white_noise_returns)
    assert len(result.recommendations) > 0
    assert any("EWMA" in r or "GARCH" in r for r in result.recommendations)
