"""Abstract base class for data providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class DataProvider(ABC):
    """Provider-agnostic interface for downloading historical prices."""

    name: str = "base"

    @abstractmethod
    def download(self, ticker: str, years: int | None = 20) -> pd.DataFrame:
        """Return a DataFrame with at least a Close column.

        Args:
            ticker: Provider-native symbol.
            years: Lookback in calendar years. None → maximum available.
        """
        raise NotImplementedError
