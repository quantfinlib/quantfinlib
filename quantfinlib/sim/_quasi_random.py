"""
File: quantfinlib/sim/_quasibm.py

Description:
    Simulation of a semi-random process exhibiting a dependency on a technical indicator.

Author:    Nathan de Vries
Copyright: (c) 2024 Nathan de Vries
License:   MIT License
"""

__all__ = ["QuasiRandom"]


from typing import Callable, Optional, Union, Protocol

import numpy as np

from quantfinlib.sim._base import SimBase
from quantfinlib.sim._bm import BrownianMotion


class BaseModel(Protocol):
    def _path_sample_np(
        self,
        x0: Union[float, np.ndarray],
        dt: float,
        num_steps: int,
        num_paths: int,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        pass


class QuasiRandom(SimBase):

    def __init__(
        self,
        tech_ind_func: Callable[[np.ndarray], np.ndarray],
        f_signal_vol: Union[float, np.ndarray] = 0.1,
        base_model: BaseModel = BrownianMotion(),
    ):
        r"""Initializes the QuasiBrownianMotion instance for simulating a semi-random process.
        First a random path is generated using Brownian motion, then the technical indicator is
        calculated, scaled, lagged by one step, and added to the path.

        Parameters
        ----------
        tech_ind_func : Callable[[np.ndarray], np.ndarray]
            A function that takes a 1D numpy array of prices and returns a 1D numpy array of
            technical indicator values.
        f_signal_vol : float or array, optional
            The factor by which to scale the signal volatility compared to the Brownian motion
            volatility (default is 0.1).
        base_model : BaseModel
            The base simulation model to use (default is BrownianMotion).
        """
        super().__init__()
        self.tech_ind_func = tech_ind_func
        self.f_signal_vol = f_signal_vol
        self.base_model = base_model

    def __repr__(self) -> str:
        """Return a string representation of the QuasiBrownianMotion instance."""
        return f"QuasiRandom(tech_ind_func={self.tech_ind_func}, " \
            f"f_signal_vol={self.f_signal_vol}, base_model={self.base_model})"

    def _fit_np(self, x: np.ndarray, dt: float):
        raise NotImplementedError

    def _path_sample_np(
        self,
        x0: Union[float, np.ndarray],
        dt: float,
        num_steps: int,
        num_paths: int,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        # Generate Brownian motion paths
        bm_ans = self.base_model._path_sample_np(
            x0=x0, dt=dt, num_steps=num_steps, num_paths=num_paths, random_state=random_state
        )

        # Generate the technical indicator signal
        indicator = np.zeros_like(bm_ans)
        for i in range(bm_ans.shape[1]):
            indicator[1:, i] = np.nan_to_num(self.tech_ind_func(bm_ans[:-1, i]))

        # Scale the signal volatility
        signal_vol = self.f_signal_vol * np.tile(self.base_model.vol, num_paths)
        indicator = indicator * signal_vol * np.sqrt(dt) / (np.std(indicator[1:-1], axis=0, ddof=2))
        return bm_ans + np.cumsum(indicator, axis=0)
