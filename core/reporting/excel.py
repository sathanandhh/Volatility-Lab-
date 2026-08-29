"""Excel workbook generation via xlsxwriter."""
from __future__ import annotations

import io
from typing import Any

import pandas as pd

from core.config import get_settings


def build_excel_workbook(session, output_path: str = "") -> str:
    """Build a polished Excel workbook with all session analysis."""
    import os
    settings = get_settings()
    out_dir = output_path or os.path.join(settings.report_output_dir, "")
    os.makedirs(out_dir, exist_ok=True) if out_dir else None
    path = output_path or os.path.join(
        settings.report_output_dir,
        f"volatility_report_{session.id[:8]}.xlsx",
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        # Summary sheet
        summary = pd.DataFrame([
            {"field": "session_id", "value": session.id},
            {"field": "stage", "value": session.current_stage},
            {"field": "n_returns", "value": int(len(session.returns)) if session.returns is not None else 0},
            {"field": "fitted_models", "value": ", ".join(session.fitted_models.keys()) or "none"},
            {"field": "decisions", "value": len(session.decision_log)},
        ])
        summary.to_excel(writer, sheet_name="Summary", index=False)
        # Returns
        if session.returns is not None:
            pd.DataFrame({"Return (%)": session.returns}).to_excel(
                writer, sheet_name="Returns")
        # Conditional volatility per model
        for name, fit in session.fitted_models.items():
            cv = fit.conditional_volatility
            pd.DataFrame({"Conditional Volatility (%)": cv}).to_excel(
                writer, sheet_name=f"{name[:25]} CV", index=False)
        # Parameters per model
        for name, fit in session.fitted_models.items():
            params = pd.DataFrame({
                "Estimate": fit.params,
                "Std Error": getattr(fit, "std_err",
                                     pd.Series(dtype=float)) if hasattr(fit, "std_err") else None,
            })
            params.to_excel(writer, sheet_name=f"{name[:25]} Params")
        # Backtest
        if session.coverage_scorecard:
            pd.DataFrame(session.coverage_scorecard).to_excel(
                writer, sheet_name="Coverage", index=False)
        # Decision log
        if session.decision_log.entries:
            pd.DataFrame(session.decision_log.to_list()).to_excel(
                writer, sheet_name="Decision Log", index=False)
    with open(path, "wb") as f:
        f.write(out.getvalue())
    return path
