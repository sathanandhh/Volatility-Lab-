"""PDF report generation (stub — uses ReportLab or WeasyPrint if available)."""
from __future__ import annotations

import os
from typing import Any

from core.config import get_settings


def build_pdf_report(session, output_path: str = "") -> str:
    """Build a PDF report. Falls back to a Markdown dump if no PDF engine."""
    settings = get_settings()
    path = output_path or os.path.join(
        settings.report_output_dir,
        f"volatility_report_{session.id[:8]}.pdf",
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        # Fallback: write a Markdown file with .pdf extension
        from core.reporting.markdown import build_markdown_report
        md = build_markdown_report(session)
        with open(path + ".md", "w", encoding="utf-8") as f:
            f.write(md)
        return path + ".md"
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Volatility Analysis Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Session: {session.id}", styles["Normal"]),
        Paragraph(f"Stage: {session.current_stage}", styles["Normal"]),
        Spacer(1, 12),
    ]
    if session.metadata:
        story.append(Paragraph("Metadata", styles["Heading2"]))
        for k, v in session.metadata.items():
            story.append(Paragraph(f"{k}: {v}", styles["Normal"]))
    if session.preflight_result:
        story.append(Paragraph("Pre-flight Gates", styles["Heading2"]))
        story.append(Paragraph(
            f"Overall: {session.preflight_result.overall.value}",
            styles["Normal"],
        ))
        for c in session.preflight_result.checks:
            story.append(Paragraph(
                f"{c.name}: {c.status.value} — {c.detail}",
                styles["Normal"],
            ))
    if session.fitted_models:
        story.append(Paragraph("Fitted Models", styles["Heading2"]))
        for name, fit in session.fitted_models.items():
            story.append(Paragraph(
                f"{name}: AIC={fit.aic:.1f}, BIC={fit.bic:.1f}, "
                f"converged={fit.converged}",
                styles["Normal"],
            ))
    if session.coverage_scorecard:
        story.append(Paragraph("Backtest Coverage", styles["Heading2"]))
        for row in session.coverage_scorecard:
            story.append(Paragraph(
                f"{row.get('model')}: verdict={row.get('verdict')}, "
                f"zone={row.get('traffic_light_zone')}",
                styles["Normal"],
            ))
    doc.build(story)
    return path
