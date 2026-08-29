"""Combined coverage scorecard — runs every backtest test for every model."""
from __future__ import annotations

import pandas as pd

from core.backtest.kupiec import kupiec_test
from core.backtest.christoffersen import conditional_coverage, christoffersen_test
from core.backtest.traffic_light import traffic_light_test
from core.backtest.dynamic_quantile import dynamic_quantile_test


def coverage_scorecard(backtest_results: dict, alpha: float = 0.025) -> list[dict]:
    """Run the full coverage suite on all backtested models.

    Args:
        backtest_results: {model_name: ModelBacktestResult}
        alpha: VaR α (loss tail probability)

    Returns a list of per-model scorecards.
    """
    rows = []
    for name, bt in backtest_results.items():
        hits = bt.violations
        kup_stat, kup_p = kupiec_test(hits, alpha)
        ind_stat, ind_p = christoffersen_test(hits)
        cc_stat, cc_p, _, _ = conditional_coverage(hits, alpha)
        tl = traffic_light_test(hits)
        dq = dynamic_quantile_test(bt.returns, bt.var[name] if name in bt.var.columns
                                    else bt.var, alpha)
        rows.append({
            "model": name,
            "breaches": int(hits.sum()),
            "observed_breach_pct": float(hits.mean() * 100),
            "expected_breach_pct": alpha * 100,
            "kupiec_p": float(kup_p) if not pd.isna(kup_p) else None,
            "independence_p": float(ind_p) if not pd.isna(ind_p) else None,
            "conditional_coverage_p": float(cc_p) if not pd.isna(cc_p) else None,
            "traffic_light_zone": tl["zone"],
            "traffic_light_multiplier": tl["capital_multiplier"],
            "dynamic_quantile_p": dq.get("p_value"),
            "verdict": (
                "pass" if (kup_p >= 0.05 and cc_p >= 0.05
                          and tl["zone"] in ("green", "yellow"))
                else "review"
            ),
        })
    return rows
