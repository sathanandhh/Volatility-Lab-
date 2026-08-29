"""Check 1 — Sample size adequacy."""
from __future__ import annotations

import pandas as pd

from core.preflight.checks._base import BaseCheck
from core.preflight.gates import CheckResult, GateStatus


class SampleSizeCheck(BaseCheck):
    name = "sample_size"
    MIN_GARCH = 300
    MIN_EGARCH = 500

    def run(self, returns: pd.Series) -> CheckResult:
        n = int(len(returns))
        if n < self.MIN_GARCH:
            return CheckResult(
                name=self.name, status=GateStatus.BLOCK,
                detail=f"Only {n} observations — minimum {self.MIN_GARCH} required for ARCH/GARCH.",
                threshold=self.MIN_GARCH,
                recommendation="Extend history or use higher-frequency (e.g. weekly→daily).",
            )
        if n < self.MIN_EGARCH:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail=f"{n} observations — at least {self.MIN_EGARCH} recommended for stable EGARCH.",
                threshold=self.MIN_EGARCH,
            )
        return CheckResult(
            name=self.name, status=GateStatus.PASS,
            detail=f"{n} observations ≥ {self.MIN_EGARCH}.",
            threshold=self.MIN_EGARCH,
        )
