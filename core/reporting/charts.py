"""Plotly chart helpers for inline display and static export."""
from __future__ import annotations

from typing import Any

import pandas as pd


def returns_and_volatility_chart(returns: pd.Series, cond_vol: pd.Series,
                                  title: str = "Returns and Conditional Volatility") -> Any:
    """Build a 2-row Plotly subplots figure: returns + conditional volatility."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.55],
                        subplot_titles=("Log returns (%)", "Conditional volatility (%)"))
    fig.add_trace(go.Scatter(x=returns.index, y=returns, name="Return",
                             line=dict(color="#6f7973", width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol, name="σ",
                             line=dict(color="#195f46", width=1.6)),
                  row=2, col=1)
    fig.update_layout(height=520, title=title, showlegend=False)
    return fig


def var_backtest_chart(returns: pd.Series, var: pd.Series, hits: pd.Series,
                        title: str = "VaR Backtest") -> Any:
    """Plot returns vs VaR with breach markers."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns.index, y=returns, name="Return",
                             line=dict(color="#6f7973", width=1)))
    fig.add_trace(go.Scatter(x=var.index, y=var, name="VaR",
                             line=dict(color="#b85042", width=1.6)))
    hit_idx = hits.index[hits]
    if not hit_idx.empty:
        fig.add_trace(go.Scatter(x=hit_idx, y=returns.loc[hit_idx],
                                 name="Breaches", mode="markers",
                                 marker=dict(color="#b85042", size=7, symbol="x")))
    fig.update_layout(height=420, title=title)
    return fig


def to_static_png(fig, width: int = 900, height: int = 420) -> bytes:
    """Render a Plotly figure to PNG bytes (for PDF embedding)."""
    return fig.to_image(format="png", width=width, height=height)


====================================================================================================
END OF FLATTENED SOURCE — core/
====================================================================================================
