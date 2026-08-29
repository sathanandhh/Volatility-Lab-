"""Gate status types and CheckResult / GateResult dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GateStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""
    name: str
    status: GateStatus
    detail: str
    statistic: float | None = None
    p_value: float | None = None
    threshold: float | str | None = None
    recommendation: str | None = None

    def explain(self) -> str:
        """Plain-language explanation of this check's result."""
        return _EXPLANATIONS.get(self.name, self.detail).format(
            detail=self.detail,
            stat=self.statistic,
            p=self.p_value,
            threshold=self.threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "recommendation": self.recommendation,
        }


@dataclass
class GateResult:
    """Aggregated result of all pre-flight checks."""
    checks: list[CheckResult]
    overall: GateStatus
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.value,
            "checks": [c.to_dict() for c in self.checks],
            "recommendations": self.recommendations,
            "next_actions": self._next_actions(),
        }

    def _next_actions(self) -> list[str]:
        has_block = any(c.status == GateStatus.BLOCK for c in self.checks)
        if has_block:
            return ["preflight.explain_gate", "optimize.distribution",
                    "optimize.order"]
        if self.overall == GateStatus.WARN:
            return ["optimize.distribution", "optimize.order", "models.fit"]
        return ["optimize.order", "optimize.distribution", "models.fit"]


# Per-check plain-language explanations (placeholders; specialised in each check file)
_EXPLANATIONS: dict[str, str] = {
    "sample_size": "Sample size check: {detail}",
    "missing_data": "Missing-data check: {detail}",
    "outliers": "Outlier check: {detail}",
    "zero_infinity": "Bad-value check: {detail}",
    "stationarity": "Stationarity check: {detail}",
    "structural_break": "Structural-break check: {detail}",
    "arch_effect": "ARCH-effect check (Engle LM): {detail}",
    "volatility_clustering": "Volatility-clustering check (LB on r²): {detail}",
    "normality": "Normality check (Jarque-Bera): {detail}",
    "mean_specification": "Mean-specification check: {detail}",
    "leverage_asymmetry": "Leverage-asymmetry check (sign-bias): {detail}",
    "frequency_adequacy": "Frequency-adequacy check: {detail}",
}
