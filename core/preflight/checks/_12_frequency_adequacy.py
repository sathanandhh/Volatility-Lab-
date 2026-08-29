"""Check 12 — Frequency adequacy (enough observations for the chosen frequency)."""
from __future__ import annotations

import pandas as pd

from core.preflight.checks._base import BaseCheck
from core.preflight.gates import CheckResult, GateStatus


class FrequencyAdequacyCheck(BaseCheck):
    name = "frequency_adequacy"
    MIN_DAILY = 250
    MIN_WEEKLY = 60
    MIN_MONTHLY = 36

    def run(self, returns: pd.Series) -> CheckResult:
        n = len(returns)
        # Infer frequency from median gap
        idx = pd.DatetimeIndex(returns.index)
        if n < 2:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail="Insufficient observations.",
            )
        median_gap = idx.to_series().diff().dt.days.median()
        if median_gap is None:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail="Could not infer frequency.",
            )
        if median_gap <= 4:  # daily-ish
            min_n = self.MIN_DAILY
            freq = "daily"
        elif median_gap <= 10:  # weekly
            min_n = self.MIN_WEEKLY
            freq = "weekly"
        else:  # monthly
            min_n = self.MIN_MONTHLY
            freq = "monthly"
        if n < min_n:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail=f"{n} {freq} observations — minimum {min_n} recommended.",
                statistic=float(n), threshold=min_n,
                recommendation="Extend history or switch to a higher frequency.",
            )
        return CheckResult(
            name=self.name, status=GateStatus.PASS,
            detail=f"{n} {freq} observations (≥ {min_n}).",
            statistic=float(n), threshold=min_n,
        )
