"""CSV file provider — supports Date + Close or Date + Returns columns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.data.providers.base import DataProvider


class CSVProvider(DataProvider):
    name = "csv"

    def download(self, ticker: str, years: int | None = 20) -> pd.DataFrame:  # pragma: no cover
        # CSV provider is invoked via load() directly
        raise NotImplementedError("Use CSVProvider.load(path=...) instead.")

    def load(self, path: str, date_column: str = "Date",
             value_column: str = "Close", is_returns: bool = False) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        lower = {c.lower(): c for c in df.columns}
        date_col = lower.get(date_column.lower(), date_column)
        val_col = lower.get(value_column.lower(), value_column)
        if date_col not in df.columns or val_col not in df.columns:
            raise ValueError(
                f"CSV must have '{date_column}' and '{value_column}' columns. "
                f"Found: {list(df.columns)}"
            )
        idx = pd.to_datetime(df[date_col], errors="coerce")
        vals = pd.to_numeric(df[val_col], errors="coerce")
        out = pd.DataFrame({"Close": vals.to_numpy()}, index=idx).dropna().sort_index()
        out = out[~out.index.duplicated(keep="last")]
        if is_returns:
            # Reconstruct a synthetic price path so downstream log-returns work
            out["Close"] = 100 * np.exp(out["Close"].cumsum() / 100)
        return out
