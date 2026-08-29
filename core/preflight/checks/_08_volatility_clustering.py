"""Check 8 — Ljung-Box on squared returns."""
from __future__ import annotations

import pandas as pd

from core.preflight.checks._base import BaseCheck
from core.preflight.gates import CheckResult, GateStatus


class VolatilityClusteringCheck(BaseCheck):
    name = "volatility_clustering"
    LAG = 10

    def run(self, returns: pd.Series) -> CheckResult:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        r = returns.dropna()
        if len(r) < 30:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail="Too few observations for Ljung-Box on r².",
            )
        sq = r.pow(2)
        lb = acorr_ljungbox(sq, lags=[self.LAG], return_df=True)
        stat = float(lb.loc[self.LAG, "lb_stat"])
        p = float(lb.loc[self.LAG, "lb_pvalue"])
        if p > 0.10:
            return CheckResult(
                name=self.name, status=GateStatus.BLOCK,
                detail=f"No clustering (LB r² p={p:.3f} at lag {self.LAG}).",
                statistic=stat, p_value=p, threshold=0.05,
                recommendation="No volatility clustering — GARCH will not add value.",
            )
        return CheckResult(
            name=self.name, status=GateStatus.PASS,
            detail=f"Clustering detected (LB r² p={p:.4f} at lag {self.LAG}).",
            statistic=stat, p_value=p, threshold=0.05,
        )
