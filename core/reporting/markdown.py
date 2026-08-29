"""Markdown report generation — ideal for LLM consumption."""
from __future__ import annotations

from typing import Any


def build_markdown_report(session) -> str:
    """Build a comprehensive Markdown summary of the session."""
    lines: list[str] = []
    lines.append("# Volatility Analysis — Session Report")
    lines.append("")
    lines.append(f"**Session ID:** `{session.id}`  ")
    lines.append(f"**Stage:** {session.current_stage}  ")
    lines.append(f"**Decisions logged:** {len(session.decision_log)}  ")
    if session.metadata:
        lines.append(f"**Asset:** {session.metadata.get('ticker', 'n/a')}  ")
        lines.append(f"**Frequency:** {session.metadata.get('frequency', 'n/a')}  ")
        lines.append(f"**Source:** {session.metadata.get('source', 'n/a')}  ")
    if session.returns is not None:
        lines.append(f"**Observations:** {len(session.returns)}  ")
    lines.append("")
    # Preflight
    if session.preflight_result:
        lines.append("## Pre-flight Gates")
        lines.append(f"**Overall status:** `{session.preflight_result.overall.value}`")
        lines.append("")
        lines.append("| Check | Status | Detail |")
        lines.append("|---|---|---|")
        for c in session.preflight_result.checks:
            lines.append(f"| {c.name} | {c.status.value} | {c.detail} |")
        lines.append("")
        if session.preflight_result.recommendations:
            lines.append("**Recommendations:**")
            for r in session.preflight_result.recommendations:
                lines.append(f"- {r}")
            lines.append("")
    # Optimization
    if session.optimization_results:
        lines.append("## Input Optimization")
        for kind, opt in session.optimization_results.items():
            summary = opt.summary() if hasattr(opt, "summary") else {"optimal": opt}
            lines.append(f"**{kind}:** {summary.get('optimal')} "
                         f"({summary.get('criterion', 'n/a')}={summary.get('optimal_score', 'n/a')})")
        lines.append("")
    # Fitted models
    if session.fitted_models:
        lines.append("## Fitted Models")
        lines.append("| Model | AIC | BIC | Converged |")
        lines.append("|---|---|---|---|")
        for name, fit in session.fitted_models.items():
            lines.append(f"| {name} | {fit.aic:.1f} | {fit.bic:.1f} | {fit.converged} |")
        lines.append("")
        # Parameters
        lines.append("### Parameters")
        for name, fit in session.fitted_models.items():
            lines.append(f"**{name}:**")
            lines.append("")
            for k, v in fit.params.items():
                lines.append(f"- `{k}` = {v:.5f}")
            lines.append("")
    # Comparison
    if session.comparison_result:
        lines.append("## Model Comparison")
        lines.append(f"**Best in-sample fit (AIC):** {session.comparison_result.get('best_fit')}")
        lines.append(f"**Best out-of-sample (QLIKE):** {session.comparison_result.get('best_forecast')}")
        lines.append("")
        if session.comparison_result.get("scorecard"):
            lines.append("| Model | AIC | BIC | QLIKE | Vol RMSE |")
            lines.append("|---|---|---|---|---|")
            for row in session.comparison_result["scorecard"]:
                lines.append(f"| {row['model']} | {row['aic']:.1f} | "
                             f"{row['bic']:.1f} | {row['qlike']:.4f} | "
                             f"{row['vol_rmse']:.4f} |")
            lines.append("")
    # Risk results
    if session.risk_results:
        lines.append("## Risk Metrics")
        for key, val in session.risk_results.items():
            lines.append(f"**{key}:**")
            if isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        lines.append(f"- {k}: {v:,.2f}")
                    else:
                        lines.append(f"- {k}: {v}")
            lines.append("")
    # Backtest coverage
    if session.coverage_scorecard:
        lines.append("## Backtest Coverage")
        lines.append("| Model | Breaches | Kupiec p | CC p | Zone | Verdict |")
        lines.append("|---|---|---|---|---|---|")
        for row in session.coverage_scorecard:
            lines.append(
                f"| {row.get('model')} | {row.get('breaches')} | "
                f"{row.get('kupiec_p'):.3f} | "
                f"{row.get('conditional_coverage_p'):.3f} | "
                f"{row.get('traffic_light_zone')} | "
                f"{row.get('verdict')} |"
                if row.get('kupiec_p') is not None and row.get('conditional_coverage_p') is not None
                else f"| {row.get('model')} | {row.get('breaches')} | n/a | n/a | {row.get('traffic_light_zone')} | {row.get('verdict')} |"
            )
        lines.append("")
    # Decision log
    if session.decision_log.entries:
        lines.append("## Decision Audit Trail")
        lines.append("| # | Timestamp | Action | Target | Outcome |")
        lines.append("|---|---|---|---|---|")
        for e in session.decision_log.entries:
            lines.append(f"| {e.id} | {e.timestamp} | {e.action} | "
                         f"{e.target} | {e.outcome} |")
        lines.append("")
    lines.append("---")
    lines.append("*Generated by the Volatility MCP server. Educational use only.*")
    return "\n".join(lines)
