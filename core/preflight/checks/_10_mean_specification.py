"""Check 10 — Mean specification (LB on constant-mean residuals)."""
from __future__ import annotations

import pandas as pd

from core.preflight.checks._base import BaseCheck
from core.preflight.gates import CheckResult, GateStatus


class MeanSpecificationCheck(BaseCheck):
    name = "mean_specification"
    LAG = 10

    def run(self, returns: pd.Series) -> CheckResult:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        r = returns.dropna()
        if len(r) < 30:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail="Too few observations.",
            )
        # Residuals from constant-mean model
        resid = r - r.mean()
        lb = acorr_ljungbox(resid, lags=[self.LAG], return_df=True)
        p = float(lb.loc[self.LAG, "lb_pvalue"])
        if p < 0.05:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail=f"Constant mean leaves serial correlation (LB p={p:.3f}).",
                statistic=float(lb.loc[self.LAG, "lb_stat"]),
                p_value=p,
                threshold=0.05,
                recommendation="Add AR(1) or ARMA(1,1) mean specification.",
            )
        return CheckResult(
            name=self.name, status=GateStatus.PASS,
            detail=f"Constant mean adequate (LB p={p:.3f}).",
            p_value=p, threshold=0.05,
        )
