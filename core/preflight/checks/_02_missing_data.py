"""Check 2 — Missing data and gaps."""
from __future__ import annotations

import pandas as pd

from core.preflight.checks._base import BaseCheck
from core.preflight.gates import CheckResult, GateStatus


class MissingDataCheck(BaseCheck):
    name = "missing_data"
    MAX_OK_GAP_DAYS = 10
    MAX_PCT_MISSING = 0.05

    def run(self, returns: pd.Series) -> CheckResult:
        n = len(returns)
        if n == 0:
            return CheckResult(
                name=self.name, status=GateStatus.BLOCK,
                detail="Empty series.",
            )
        # Index is daily-ish; compute consecutive-day gaps
        idx = pd.DatetimeIndex(returns.index)
        diffs = idx.to_series().diff().dt.days.dropna()
        if diffs.empty:
            return CheckResult(
                name=self.name, status=GateStatus.PASS,
                detail="Single observation; no gaps to assess.",
            )
        max_gap = int(diffs.max())
        n_big = int((diffs > self.MAX_OK_GAP_DAYS).sum())
        pct = float(n_big / len(diffs))
        if max_gap > 30 or pct > self.MAX_PCT_MISSING:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail=f"Max gap {max_gap} days; {n_big} gaps > {self.MAX_OK_GAP_DAYS}d ({pct:.1%}).",
                statistic=max_gap,
                threshold=self.MAX_OK_GAP_DAYS,
                recommendation="Inspect data source; consider imputation or shorter history.",
            )
        return CheckResult(
            name=self.name, status=GateStatus.PASS,
            detail=f"Max gap {max_gap} days; {n_big} gaps > {self.MAX_OK_GAP_DAYS}d.",
            statistic=max_gap,
            threshold=self.MAX_OK_GAP_DAYS,
        )
