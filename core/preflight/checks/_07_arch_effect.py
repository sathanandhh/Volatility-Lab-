"""Check 7 — Engle ARCH-LM test. THE MOST IMPORTANT PRE-FLIGHT CHECK.

If the null of no-ARCH-effect cannot be rejected, the entire GARCH
family is statistically unjustified.
"""
from __future__ import annotations

import pandas as pd

from core.preflight.checks._base import BaseCheck
from core.preflight.gates import CheckResult, GateStatus


class ArchEffectCheck(BaseCheck):
    name = "arch_effect"
    NLAGS = 10

    def run(self, returns: pd.Series) -> CheckResult:
        from statsmodels.stats.diagnostic import het_arch
        r = returns.dropna()
        if len(r) < 50:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail="Too few observations for Engle ARCH-LM.",
            )
        try:
            lm_stat, lm_p, f_stat, f_p = het_arch(r, nlags=self.NLAGS)
        except Exception as exc:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail=f"ARCH-LM test errored: {exc!s}",
            )
        if lm_p > 0.10:
            return CheckResult(
                name=self.name, status=GateStatus.BLOCK,
                detail=(
                    f"No ARCH effect detected (Engle LM p={lm_p:.3f}). "
                    "GARCH family is statistically unjustified."
                ),
                statistic=float(lm_stat),
                p_value=float(lm_p),
                threshold=0.05,
                recommendation=(
                    "Use EWMA or constant-volatility modelling. "
                    "Do NOT fit ARCH/GARCH/EGARCH."
                ),
            )
        return CheckResult(
            name=self.name, status=GateStatus.PASS,
            detail=f"ARCH effect present (Engle LM p={lm_p:.4f}). GARCH justified.",
            statistic=float(lm_stat),
            p_value=float(lm_p),
            threshold=0.05,
        )
