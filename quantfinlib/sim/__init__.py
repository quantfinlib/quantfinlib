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
   QuasiRandom             Semi-random process with dependency on a technical indicator.
"""

from ._bm import BrownianMotion
from ._gbm import GeometricBrownianMotion
from ._quasi_random import QuasiRandom
from ._ou import OrnsteinUhlenbeck

__all__ = [BrownianMotion, GeometricBrownianMotion, OrnsteinUhlenbeck, QuasiRandom]

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
