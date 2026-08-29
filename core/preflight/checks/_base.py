"""Base class for all pre-flight checks."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from core.preflight.gates import CheckResult


class BaseCheck(ABC):
    """Abstract pre-flight check.

    Subclasses must set `name` (snake_case) and implement `run()`.
    """
    name: str = "base"

    @abstractmethod
    def run(self, returns: pd.Series) -> CheckResult:
        """Run the check on the percentage-log-returns series."""
        raise NotImplementedError
