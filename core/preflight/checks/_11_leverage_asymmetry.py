"""Check 11 — Leverage / sign-bias test (Engle-Ng 1993)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.preflight.checks._base import BaseCheck
from core.preflight.gates import CheckResult, GateStatus


class LeverageAsymmetryCheck(BaseCheck):
    name = "leverage_asymmetry"

    def run(self, returns: pd.Series) -> CheckResult:
        r = returns.dropna()
        if len(r) < 30:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail="Too few observations.",
            )
        # Sign-bias: regress r² on indicator(r<0)
        neg = (r < 0).astype(float).to_numpy()
        y = r.pow(2).to_numpy()
        X = np.column_stack([np.ones_like(neg), neg])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            n = len(y)
            k = X.shape[1]
            sigma2 = resid @ resid / (n - k)
            cov = sigma2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            t_neg = beta[1] / se[1] if se[1] > 0 else 0.0
            from scipy import stats as ss
            p_neg = 2 * (1 - ss.t.cdf(abs(t_neg), df=n - k))
        except Exception:
            t_neg, p_neg = float("nan"), float("nan")
        # Positive sign-bias test using indicator(r>=0) is similar; we report the negative one
        if p_neg < 0.05:
            return CheckResult(
                name=self.name, status=GateStatus.PASS,
                detail=f"Sign-bias detected (t={t_neg:.2f}, p={p_neg:.4f}).",
                statistic=float(t_neg), p_value=float(p_neg), threshold=0.05,
                recommendation="Negative shocks carry different vol response — try GJR-GARCH or EGARCH.",
            )
        return CheckResult(
            name=self.name, status=GateStatus.PASS,
            detail=f"No strong sign-bias (t={t_neg:.2f}, p={p_neg:.3f}).",
            statistic=float(t_neg), p_value=float(p_neg), threshold=0.05,
        )
