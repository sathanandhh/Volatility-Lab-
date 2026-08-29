#!/usr/bin/env python3
"""Example 3 — Input optimization walkthrough.

Shows every optimizer in action and explains why each optimum was
chosen by referencing the AIC/QLIKE/LB sweep tables.

    optimize.order → optimize.distribution → optimize.mean
    → optimize.window → optimize.refit → optimize.horizon

Run:
    python examples/03_optimization_walkthrough.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.data.service import DataService
from core.optimize.order_selector import select_order
from core.optimize.distribution_selector import select_distribution
from core.optimize.mean_selector import select_mean
from core.optimize.window_selector import select_window
from core.optimize.refit_selector import select_refit_frequency
from core.optimize.horizon_selector import select_horizon
from core.optimize.heuristics import get_heuristic


def main() -> None:
    print("=" * 70)
    print("EXAMPLE 3 · Input Optimization Walkthrough")
    print("=" * 70)

    # Load data
    service = DataService.from_env()
    prices = service.download_prices("RELIANCE.NS", years=5)
    close, returns = service.prepare_returns(prices, frequency="Daily")
    print(f"\nLoaded {len(returns)} daily returns "
          f"({returns.index[0].date()} to {returns.index[-1].date()})")

    # ── 1. Order (p, q) selection ────────────────────────────────────
    print("\n" + "─" * 70)
    print("1. ORDER SELECTION — sweep (p, q) by AIC")
    print("─" * 70)
    print(f"\n   Heuristic: {get_heuristic('aic_delta')}")
    order = select_order(returns, family="GARCH", max_p=3, max_q=2,
                         distribution="t", criterion="AIC")
    print(f"\n   Optimal: p={order.optimal['p']}, q={order.optimal['q']} "
          f"(AIC={order.optimal_score:.1f})")
    print(f"   {order.recommendation}")
    print("\n   Full sweep table:")
    print(f"   {'p':>3s} {'q':>3s} {'AIC':>12s} {'Converged':>10s}")
    for row in order.sweep_table:
        print(f"   {row['p']:>3d} {row['q']:>3d} "
              f"{row['score']:>12.1f} {str(row.get('converged', '?')):>10s}")

    # ── 2. Distribution selection ────────────────────────────────────
    print("\n" + "─" * 70)
    print("2. DISTRIBUTION SELECTION — sweep by AIC")
    print("─" * 70)
    print(f"\n   Heuristic: {get_heuristic('distribution_when_normality_blocked')}")
    dist = select_distribution(returns, family="GARCH", p=1, q=1,
                               candidates=["normal", "t", "ged"])
    print(f"\n   Optimal: {dist.optimal} (AIC={dist.optimal_score:.1f})")
    print(f"   {dist.recommendation}")
    print("\n   Full sweep table:")
    print(f"   {'Distribution':>15s} {'AIC':>12s} {'Converged':>10s}")
    for row in dist.sweep_table:
        print(f"   {row['distribution']:>15s} {row['aic']:>12.1f} "
              f"{str(row.get('converged', '?')):>10s}")

    # ── 3. Mean specification ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("3. MEAN SPECIFICATION — sweep by residual Ljung-Box p")
    print("─" * 70)
    mean = select_mean(returns, candidates=["Constant", "AR(1)", "ARMA(1,1)"])
    print(f"\n   Optimal: {mean.optimal} (LB p={mean.optimal_score:.3f})")
    print(f"   {mean.recommendation}")
    print("\n   Full sweep table:")
    print(f"   {'Spec':>15s} {'LB p-value':>12s}")
    for row in mean.sweep_table:
        print(f"   {row['spec']:>15s} {row['p_value']:>12.3f}")

    # ── 4. Rolling window selection ──────────────────────────────────
    print("\n" + "─" * 70)
    print("4. ROLLING WINDOW — sweep by out-of-sample QLIKE")
    print("─" * 70)
    print(f"\n   Heuristic: {get_heuristic('qlike_preference')}")
    window = select_window(returns, min_window=10, max_window=63,
                           step=5, metric="QLIKE")
    print(f"\n   Optimal: {window.optimal} days (QLIKE={window.optimal_score:.4f})")
    print(f"   {window.recommendation}")
    print("\n   Full sweep table:")
    print(f"   {'Window':>8s} {'QLIKE':>12s}")
    for row in window.sweep_table:
        print(f"   {row['window']:>8d} {row['score']:>12.4f}")

    # ── 5. Refit frequency ───────────────────────────────────────────
    print("\n" + "─" * 70)
    print("5. REFIT FREQUENCY — sweep by conditional-coverage p-value")
    print("─" * 70)
    print(f"\n   Heuristic: {get_heuristic('refit_tradeoff')}")
    refit = select_refit_frequency(returns, model_name="GARCH",
                                   min_freq=5, max_freq=25, alpha=0.025)
    print(f"\n   Optimal: every {refit.optimal} obs (CC p={refit.optimal_score:.3f})")
    print(f"   {refit.recommendation}")
    print("\n   Full sweep table:")
    print(f"   {'Refit':>8s} {'CC p':>10s}")
    for row in refit.sweep_table:
        print(f"   {row['refit_every']:>8d} {row['cc_p']:>10.3f}")

    # ── 6. Horizon recommendation ────────────────────────────────────
    print("\n" + "─" * 70)
    print("6. FORECAST HORIZON — by use case")
    print("─" * 70)
    for use_case in ["regulatory", "weekly", "monthly", "stress", "auto"]:
        h = select_horizon(returns, target_use=use_case)
        print(f"   {use_case:>15s}: {h.optimal} periods — {h.recommendation}")

    print("\n" + "=" * 70)
    print("Optimization walkthrough complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
