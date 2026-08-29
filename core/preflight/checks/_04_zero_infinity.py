"""Check 4 — Bad values: zeros, infinities, NaNs in returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.preflight.checks._base import BaseCheck
from core.preflight.gates import CheckResult, GateStatus


class ZeroInfinityCheck(BaseCheck):
    name = "zero_infinity"

    def run(self, returns: pd.Series) -> CheckResult:
        n_inf = int(np.isinf(returns).sum())
        n_nan = int(returns.isna().sum())
        # Zero log-returns are common in illiquid series; only flag if excessive
        n_zero = int((returns == 0).sum())
        pct_zero = n_zero / max(len(returns), 1)
        if n_inf > 0:
            return CheckResult(
                name=self.name, status=GateStatus.BLOCK,
                detail=f"{n_inf} infinite values — log returns undefined (WTI April-2020 case?).",
                statistic=n_inf,
                recommendation="Filter non-positive prices before computing returns.",
            )
        if pct_zero > 0.10:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail=f"{n_zero} zero returns ({pct_zero:.1%}) — series may be illiquid.",
                statistic=n_zero,
            )
        return CheckResult(
            name=self.name, status=GateStatus.PASS,
            detail=f"{n_inf} inf, {n_nan} NaN, {n_zero} zero ({pct_zero:.1%}).",
            statistic=n_inf,
        )
