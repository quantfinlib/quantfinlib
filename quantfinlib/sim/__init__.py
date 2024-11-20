"""
Simulation models.
=================

This module contains routines for generating random price scenarions.


.. currentmodule:: quantfinlib.sim

.. autosummary::
   :toctree: _autosummary

   BrownianMotion          Simple Brownian Model with Normal changes.
   GeometricBrownianMotion Geometric Brownian Motion with Normal returns.
   OrnsteinUhlenbeck       Mean reverting noise model.
"""

from ._bm import BrownianMotion
from ._gbm import GeometricBrownianMotion
from ._quasibm import QuasiBrownianMotion
from ._ou import OrnsteinUhlenbeck

__all__ = [BrownianMotion, GeometricBrownianMotion, OrnsteinUhlenbeck, QuasiBrownianMotion]

"""
todo:
HistoricalSampling(..., sync, replace)
RandomizePhases
AR
Heston
Garch
DCCGarch
FractalBrownianMotion
MarkovChain
"""
