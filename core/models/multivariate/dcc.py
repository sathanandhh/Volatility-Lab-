"""DCC-GARCH (Engle 2002) — stub interface."""
from __future__ import annotations

import pandas as pd


def fit_dcc(returns_df: pd.DataFrame) -> dict:
    """Stub — fit DCC-GARCH on a multi-asset returns frame."""
    raise NotImplementedError(
        "DCC-GARCH requires the `mgarch` or `dcc` package. "
        "Implement when adding multivariate support."
    )
