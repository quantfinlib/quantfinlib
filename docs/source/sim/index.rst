sim
===

The **sim** submodule provides tools for simulating various stochastic differential equations (SDEs) 
and models. You can use this model to simulate stocks, futures, exchange rates, 
interest rates, as well as multivariate simulations for correlated and cointegrated portfolios. 

Each model within this submodule has a **fit** and **path_sample** member functions, 
for the calibration to historical data and the generation of future sample paths.

.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    usage

.. toctree::
    :maxdepth: 1
    :caption: Simulation Models

    bm
    gbm
    ou
