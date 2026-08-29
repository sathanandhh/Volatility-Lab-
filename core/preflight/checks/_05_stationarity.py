"""Check 5 — Stationarity via ADF and KPSS tests."""
from __future__ import annotations

import pandas as pd

from core.preflight.checks._base import BaseCheck
from core.preflight.gates import CheckResult, GateStatus


class StationarityCheck(BaseCheck):
    name = "stationarity"

    def run(self, returns: pd.Series) -> CheckResult:
        from statsmodels.tsa.stattools import adfuller, kpss
        r = returns.dropna()
        if len(r) < 30:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail="Too few observations for ADF/KPSS.",
            )
        # ADF: H0 = unit root (non-stationary)
        adf_stat, adf_p, *_ = adfuller(r, autolag="AIC")
        # KPSS: H0 = stationary
        try:
            kpss_stat, kpss_p, *_ = kpss(r, regression="c", nlags="auto")
        except Exception:
            kpss_stat, kpss_p = float("nan"), float("nan")
        # Returns are expected to be stationary; ADF p<0.05 = pass, KPSS p>0.05 = pass
        if adf_p > 0.10 and kpss_p < 0.05:
            return CheckResult(
                name=self.name, status=GateStatus.WARN,
                detail=f"ADF p={adf_p:.3f} (non-stationary), KPSS p={kpss_p:.3f} (non-stationary).",
                statistic=float(adf_stat),
                p_value=float(adf_p),
                threshold=0.05,
                recommendation="Difference the series or use AR/ARMA in mean.",
            )
        return CheckResult(
            name=self.name, status=GateStatus.PASS,
            detail=f"ADF p={adf_p:.3f}, KPSS p={kpss_p:.3f}.",
            statistic=float(adf_stat),
            p_value=float(adf_p),
            threshold=0.05,
        )
