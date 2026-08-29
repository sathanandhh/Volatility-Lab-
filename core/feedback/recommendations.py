"""Heuristic recommendation engine based on session state."""
from __future__ import annotations

from core.feedback.session import Session


def heuristic_recommendations(session: Session) -> list[str]:
    """Inspect the session and produce a list of plain-language recommendations."""
    recs: list[str] = []
    # Preflight-based
    if session.preflight_result:
        for c in session.preflight_result.checks:
            if c.status.value == "block" and c.recommendation:
                recs.append(f"{c.name}: {c.recommendation}")
            elif c.status.value == "warn" and c.recommendation:
                recs.append(f"{c.name} (warn): {c.recommendation}")
    # Diagnostics-based
    for model, diag in session.diagnostics_results.items():
        for r in diag.get("recommendations", []):
            recs.append(f"[{model}] {r}")
    # Backtest-based
    if session.coverage_scorecard:
        for row in session.coverage_scorecard:
            if row.get("verdict") == "review":
                if row.get("kupiec_p") is not None and row["kupiec_p"] < 0.05:
                    recs.append(
                        f"[{row['model']}] Kupiec p<0.05: under/over-prediction — "
                        "re-tune distribution or window."
                    )
                if (row.get("independence_p") is not None
                        and row["independence_p"] < 0.05):
                    recs.append(
                        f"[{row['model']}] Christoffersen p<0.05: breach clustering — "
                        "shorten refit interval."
                    )
                if (row.get("traffic_light_zone") in ("yellow", "red")):
                    recs.append(
                        f"[{row['model']}] Traffic light {row['traffic_light_zone']} — "
                        f"capital multiplier {row.get('traffic_light_multiplier')}."
                    )
    if not recs:
        recs.append("No issues detected. Proceed to next stage.")
    return list(dict.fromkeys(recs))  # dedupe preserving order
