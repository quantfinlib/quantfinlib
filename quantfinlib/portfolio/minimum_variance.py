"""Minimum Variance Portfolio"""

from dataclasses import dataclass, field

import cvxpy as cp
import pandas as pd

from quantfinlib.portfolio._base import PortfolioOptimize


@dataclass
class MinimumVariance(PortfolioOptimize):
    """Minimum variance portfolio optimization.
    
    Attributes
    ----------
    cov_matrix : pd.DataFrame
        The covariance matrix of asset returns.
    upper_bound_constraints : dict
        A dictionary of upper bound constraints.
    lower_bound_constraints : dict
        A dictionary of lower bound constraints.
    weights : pd.Series
        The optimized asset weights. Only available after calling the `optimize` method.

    Methods
    -------
    optimize()
        Optimize the minimum variance portfolio with the given constraints.
    
    Properties
    ----------
    weights : pd.Series
        The optimized asset weights.
    """

    cov_matrix: pd.DataFrame
    upper_bound_constraints: dict[str, float] = field(default_factory=dict)
    lower_bound_constraints: dict[str, float] = field(default_factory=dict)
    _optimized_weights: pd.Series = field(init=False)

    def optimize(self):
        """Optimize the portfolio."""
        if self.cov_matrix.shape[0] != self.cov_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square")
        n = len(self.cov_matrix)
        w = cp.Variable(n)
        risk = cp.quad_form(w, self.cov_matrix.values)
        objective = cp.Minimize(risk)
        constraints = _get_constraints(
            w=w,
            upper_bound_constraints=self.upper_bound_constraints,
            lower_bound_constraints=self.lower_bound_constraints,
            asset_names=self.cov_matrix.index,
        )
        problem = cp.Problem(objective, constraints)
        problem.solve()
        self._optimized_weights = pd.Series(w.value, index=self.cov_matrix.index)

    @property
    def weights(self) -> pd.Series:
        """Get the asset weights."""
        return self._optimized_weights


def _get_constraints(
    w: cp.Variable,
    upper_bound_constraints: dict[str, float],
    lower_bound_constraints: dict[str, float],
    asset_names: list[str],
) -> list[cp.Constraint]:
    """Get constraints for the optimization problem.

    Parameters
    ----------
    w : cp.Variable
        The variable to optimize.
    upper_bound_constraints : dict[str, float]
        A dictionary of upper bound constraints.
    lower_bound_constraints : dict[str, float]
        A dictionary of lower bound constraints.
    asset_names : list[str]
        A list of asset names.

    Returns
    -------
    constraints : list[cp.Constraint]
        A list of constraints for the optimization problem.
    """
    constraints = [cp.sum(w) == 1, w >= 0]
    asset_map = {asset: i for i, asset in enumerate(asset_names)}
    for asset, upper_bound in upper_bound_constraints.items():
        if asset not in asset_map:
            raise ValueError(f"Asset '{asset}' not found in asset list")
        constraints.append(w[asset_map[asset]] <= upper_bound)
    for asset, lower_bound in lower_bound_constraints.items():
        if asset not in asset_map:
            raise ValueError(f"Asset '{asset}' not found in asset list")
        constraints.append(w[asset_map[asset]] >= lower_bound)
    return constraints
