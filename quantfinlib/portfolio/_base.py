"""Base class for portfolio optimization models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class PortfolioOptimize(ABC):
    """Base class for portfolio optimization models."""

    cov_matrix: pd.DataFrame
    expected_returns: Optional[pd.DataFrame] = None
    _optimized_weights: pd.Series = field(init=False)

    @abstractmethod
    def optimize(self) -> None:
        """Abstract method for portfolio optimization."""

    @property
    def weights(self) -> pd.Series:
        """Returns the asset weights."""
        return self._optimized_weights
