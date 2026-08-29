"""Check 3 — Outlier detection via Hampel filter / rolling MAD."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.preflight.checks._base import BaseCheck
from core.preflight.gates import CheckResult, GateStatus


class OutliersCheck(BaseCheck):
    name = "outliers"
    WINDOW = 21
    THRESHOLD = 5.0  # in MAD units

    def run(self, returns: pd.Series) -> CheckResult:
        r = returns.dropna()
        if len(r) < self.WINDOW:
            return CheckResult(
                name=self.name, status=GateStatus.PASS,
                detail="Too few observations for rolling MAD outlier scan.",
            )
        med = r.rolling(self.WINDOW, center=True).median()
        mad = (r - med).abs().rolling(self.WINDOW, center=True).median()
        scaled = mad * 1.4826  # to be σ-comparable
        scaled = scaled.replace(0, np.nan)
        z = (r - med).abs() / scaled
        outliers = z[z > self.THRESHOLD].dropna()
        n_out = int(len(outliers))
        pct = n_out / len(r)
        if n_out > 20 or pct > 0.02:
            status = GateStatus.WARN
            rec = "Winsorize or use Student-t innovations; investigate large moves."
        else:
            status = GateStatus.PASS
            rec = None
        return CheckResult(
            name=self.name, status=status,
            detail=f"{n_out} potential outliers > {self.THRESHOLD} MAD ({pct:.2%}).",
            statistic=n_out,
            threshold=self.THRESHOLD,
            recommendation=rec,
        )
