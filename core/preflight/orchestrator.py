"""Pre-flight orchestrator: runs all 12 checks and aggregates the result."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.preflight.gates import CheckResult, GateResult, GateStatus


class PreflightOrchestrator:
    """Runs every pre-flight check and aggregates their results."""

    def __init__(self) -> None:
        from core.preflight.checks import (
            SampleSizeCheck, MissingDataCheck, OutliersCheck,
            ZeroInfinityCheck, StationarityCheck, StructuralBreakCheck,
            ArchEffectCheck, VolatilityClusteringCheck, NormalityCheck,
            MeanSpecificationCheck, LeverageAsymmetryCheck,
            FrequencyAdequacyCheck,
        )
        self.checks = [
            SampleSizeCheck(),
            MissingDataCheck(),
            OutliersCheck(),
            ZeroInfinityCheck(),
            StationarityCheck(),
            StructuralBreakCheck(),
            ArchEffectCheck(),
            VolatilityClusteringCheck(),
            NormalityCheck(),
            MeanSpecificationCheck(),
            LeverageAsymmetryCheck(),
            FrequencyAdequacyCheck(),
        ]

    def run(self, returns: pd.Series) -> GateResult:
        results: list[CheckResult] = []
        for check in self.checks:
            try:
                results.append(check.run(returns))
            except Exception as exc:  # pragma: no cover
                results.append(CheckResult(
                    name=check.name, status=GateStatus.WARN,
                    detail=f"Check errored: {exc!s}",
                ))
        overall = self._aggregate(results)
        recommendations = self._recommendations(results)
        return GateResult(checks=results, overall=overall,
                           recommendations=recommendations)

    @staticmethod
    def _aggregate(results: list[CheckResult]) -> GateStatus:
        if any(c.status == GateStatus.BLOCK for c in results):
            return GateStatus.BLOCK
        if any(c.status == GateStatus.WARN for c in results):
            return GateStatus.WARN
        return GateStatus.PASS

    @staticmethod
    def _recommendations(results: list[CheckResult]) -> list[str]:
        recs: list[str] = []
        for c in results:
            if c.recommendation:
                recs.append(c.recommendation)
        # Add structural recommendations
        normality = next((c for c in results if c.name == "normality"), None)
        if normality and normality.status == GateStatus.BLOCK:
            recs.append("Use Student-t or skew-t distribution (normality blocked).")
        arch = next((c for c in results if c.name == "arch_effect"), None)
        if arch and arch.status == GateStatus.BLOCK:
            recs.append("No ARCH effect — GARCH is unjustified; consider EWMA or constant volatility.")
        leverage = next((c for c in results if c.name == "leverage_asymmetry"), None)
        if leverage and leverage.status == GateStatus.PASS:
            recs.append("Leverage detected — prefer GJR-GARCH or EGARCH over symmetric GARCH.")
        return list(dict.fromkeys(recs))  # dedupe preserving order
