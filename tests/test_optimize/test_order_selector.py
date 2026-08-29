"""Tests for the (p, q) order selector."""
from __future__ import annotations

from core.optimize.order_selector import select_order
from core.optimize.distribution_selector import select_distribution
from core.optimize.window_selector import select_window
from core.optimize.horizon_selector import select_horizon


def test_select_order_returns_optimum(garch_returns):
    result = select_order(garch_returns, family="GARCH",
                          max_p=2, max_q=1, distribution="t")
    assert result.kind == "order"
    assert "p" in result.optimal
    assert "q" in result.optimal
    assert result.optimal["p"] in (1, 2)
    assert result.optimal["q"] in (0, 1)


def test_select_order_sweep_table_populated(garch_returns):
    result = select_order(garch_returns, family="GARCH",
                          max_p=2, max_q=1, distribution="t")
    assert len(result.sweep_table) > 0
    assert all("score" in row for row in result.sweep_table)


def test_select_order_recommendation_present(garch_returns):
    result = select_order(garch_returns, family="GARCH",
                          max_p=2, max_q=1, distribution="t")
    assert result.recommendation
    assert "Best" in result.recommendation


def test_select_distribution_returns_string(garch_returns):
    result = select_distribution(garch_returns, family="GARCH",
                                 p=1, q=1,
                                 candidates=["normal", "t"])
    assert result.kind == "distribution"
    assert result.optimal in ("normal", "t")


def test_select_distribution_t_beats_normal(garch_returns):
    """Student-t should have lower AIC than Normal on GARCH data."""
    result = select_distribution(garch_returns, family="GARCH",
                                 p=1, q=1,
                                 candidates=["normal", "t"])
    # On heavy-tailed GARCH data, t should win
    assert result.optimal == "t"


def test_select_window_returns_int(garch_returns):
    result = select_window(garch_returns, min_window=10, max_window=30,
                           step=5, metric="QLIKE")
    assert result.kind == "window"
    assert isinstance(result.optimal, int)
    assert 10 <= result.optimal <= 30


def test_select_horizon_regulatory():
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(42)
    r = pd.Series(rng.standard_normal(500))
    result = select_horizon(r, target_use="regulatory")
    assert result.optimal == 10


def test_select_horizon_weekly():
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(42)
    r = pd.Series(rng.standard_normal(500))
    result = select_horizon(r, target_use="weekly")
    assert result.optimal == 5
