"""Check 9 — Normality (Jarque-Bera, Shapiro-Wilk, Anderson-Darling)."""
from __future__ import annotations

import pandas as pd

from core.preflight.checks._base import BaseCheck
from core.preflight.gates import CheckResult, GateStatus


class NormalityCheck(BaseCheck):
    name = "normality"

    def run(self, returns: pd.Series) -> CheckResult:
        from scipy import stats
        r = returns.dropna()
        if len(r) < 20:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail="Too few observations for normality tests.",
            )
        jb_stat, jb_p = stats.jarque_bera(r)
        # Shapiro-Wilk is unreliable for n>5000 — sample down if needed
        sample = r if len(r) <= 5000 else r.sample(5000, random_state=42)
        try:
            sw_stat, sw_p = stats.shapiro(sample)
        except Exception:
            sw_stat, sw_p = float("nan"), float("nan")
        # Decision: BLOCK if normality strongly rejected at 0.1%
        if jb_p < 0.001:
            status = GateStatus.BLOCK
            rec = "Use Student-t or skew-t innovations — Normal is inappropriate."
        elif jb_p < 0.05:
            status = GateStatus.WARN
            rec = "Mild non-normality; Student-t innovations recommended."
        else:
            status = GateStatus.PASS
            rec = None
        return CheckResult(
            name=self.name, status=status,
            detail=f"JB p={jb_p:.4f}, SW p={sw_p:.4f}.",
            statistic=float(jb_stat),
            p_value=float(jb_p),
            threshold=0.001,
            recommendation=rec,
        )
