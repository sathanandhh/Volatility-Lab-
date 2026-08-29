"""Basel Traffic-Light Backtest (green/yellow/red zones)."""
from __future__ import annotations

import pandas as pd


# Basel zones for 250 daily observations at 99% VaR
ZONES_250 = {
    "green":  (0, 4),   # 0-4 breaches
    "yellow": (5, 9),  # 5-9 breaches
    "red":    (10, 999),
}


def traffic_light_test(hits: pd.Series, n_obs: int | None = None) -> dict:
    """Basel Traffic Light Backtest.

    Returns the zone and the implied capital multiplier.
    """
    n = int(n_obs if n_obs is not None else len(hits))
    x = int(hits.sum())
    # Scale 250-obs zones to actual n
    breaches_per_250 = x * 250 / max(n, 1)
    if breaches_per_250 <= 4:
        zone = "green"; multiplier = 3.0
    elif breaches_per_250 <= 9:
        zone = "yellow"
        # Multiplier increases with breaches (3.4 - 3.9 in Basel)
        if breaches_per_250 <= 5: multiplier = 3.40
        elif breaches_per_250 <= 6: multiplier = 3.50
        elif breaches_per_250 <= 7: multiplier = 3.65
        elif breaches_per_250 <= 8: multiplier = 3.75
        else: multiplier = 3.85
    else:
        zone = "red"; multiplier = 4.0
    return {
        "zone": zone,
        "breaches": x,
        "n_obs": n,
        "breaches_per_250": float(breaches_per_250),
        "capital_multiplier": multiplier,
    }
