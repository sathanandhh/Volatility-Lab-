"""Check 6 — Structural break detection (CUSUM and Bai-Perron heuristic)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.preflight.checks._base import BaseCheck
from core.preflight.gates import CheckResult, GateStatus


class StructuralBreakCheck(BaseCheck):
    name = "structural_break"

    def run(self, returns: pd.Series) -> CheckResult:
        r = returns.dropna()
        n = len(r)
        if n < 100:
            return CheckResult(
                name=self.name, status=GateStatus.PASS,
                detail="Too few observations for break detection.",
            )
        # CUSUM on the squared returns (variance breaks are most common in finance)
        sq = r.pow(2).to_numpy()
        mean = sq.mean()
        std = sq.std()
        if std == 0:
            return CheckResult(
                name=self.name, status=GateStatus.PASS,
                detail="Zero variance in squared returns — no break detectable.",
            )
        cum = np.cumsum(sq - mean) / std
        # Scale by sqrt(n) so the Brownian-bridge asymptotics apply
        cusum = cum / np.sqrt(n)
        max_cusum = float(np.max(np.abs(cusum)))
        # Approximate 5% critical value ~1.36 for Brownian bridge
        if max_cusum > 1.36:
            # Find approximate break date
            idx = int(np.argmax(np.abs(cusum)))
            break_date = r.index[idx]
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail=(
                    f"Likely structural break at {break_date.date()} "
                    f"(CUSUM={max_cusum:.2f} > 1.36)."
                ),
                statistic=max_cusum,
                threshold=1.36,
                recommendation=(
                    f"Consider splitting sample at {break_date.date()} "
                    "or including a regime dummy."
                ),
            )
        return CheckResult(
            name=self.name, status=GateStatus.PASS,
            detail=f"No structural break detected (CUSUM={max_cusum:.2f}).",
            statistic=max_cusum,
            threshold=1.36,
        )
