"""
File: quantfinlib/sim/_quasibm.py

Description:
    Simulation of a semi-random process exhibiting a dependency on a technical indicator.

Author:    Nathan de Vries
Copyright: (c) 2024 Nathan de Vries
License:   MIT License
"""

__all__ = ["QuasiBrownianMotion"]


from typing import Callable, Optional, Union

import numpy as np

from quantfinlib.sim._bm import BrownianMotion


class QuasiBrownianMotion(BrownianMotion):

    def __init__(
        self,
        tech_ind_func: Callable[[np.ndarray], np.ndarray],
        drift: Union[float, np.ndarray] = 0.0,
        vol: Union[float, np.ndarray] = 0.1,
        f_signal_vol: Union[float, np.ndarray] = 0.1,
        cor: Optional[np.ndarray] = None
    ):
        r"""Initializes the QuasiBrownianMotion instance for simulating a semi-random process.
        First a random path is generated using Brownian motion, then the technical indicator is
        calculated, scaled, lagged by one step, and added to the path.

        Parameters
        ----------
        tech_ind_func : Callable[[np.ndarray], np.ndarray]
            A function that takes a 1D numpy array of prices and returns a 1D numpy array of
            technical indicator values.
        drift : float or array, optional
            The annualized drift rate (default is 0.0).
        vol : float or array, optional
            The annualized volatility (default is 0.1).
        f_signal_vol : float or array, optional
            The factor by which to scale the signal volatility compared to the Brownian motion
            volatility (default is 0.1).
        cor : optional
            Correlation matrix for multivariate model (default is None, uncorrelated).

        """
        super().__init__(drift=drift, vol=vol, cor=cor)
        self.tech_ind_func = tech_ind_func
        self.f_signal_vol = f_signal_vol

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
        bm_ans = super()._path_sample_np(
            x0=x0,
            dt=dt,
            num_steps=num_steps,
            num_paths=num_paths,
            random_state=random_state
        )

        # Generate the technical indicator signal
        indicator = np.zeros_like(bm_ans)
        for i in range(bm_ans.shape[1]):
            indicator[1:, i] = np.nan_to_num(self.tech_ind_func(bm_ans[:-1, i]))

        # Scale the signal volatility
        signal_vol = self.f_signal_vol * self.vol
        indicator = indicator * signal_vol * np.sqrt(dt) / (np.std(indicator, axis=0))
        return bm_ans + np.cumsum(indicator, axis=0)
