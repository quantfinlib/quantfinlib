# QuantFinLib

<h1 align='center'>
<img src="./_static/quantfinlib_logo.png"  alt="QuantFinLib Logo" width="60"/>
</h1><br>


[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/quantfinlib/quantfinlib/blob/main/LICENSE)

QuantFinLib is a comprehensive Python library designed for quantitative finance. It offers a wide range
of tools and algorithms that cater to various domains within quantitative finance, including machine learning,
asset management, portfolio optimization, time series transformations, indicators, labeling, feature engineering,
stochastic simulation, randomization tests, and backtesting. This library aims to provide robust and efficient
solutions for financial data analysis, modeling, and trading strategy development.

## Time Series Simulation

Historical Sampling
: Returns are randomly sampled from a historical time-series. The generated random simulation has 
the same return distribution as the historical return distribution, but any autocorrelation or dependencies 
over time is broken. This simulation method is used as a benchmark for generating time series that have
no structural patterns, only random patterns.

Brownian Motion
: Classical random price model without memory. Price changes are Normal distributed, the price can potentially
go below zero.

| Model              | Properties |
| :---------------- | :------ | 
| Historical Sampling   | Returns have the same distributions as in the past  |
| Brownian Motion   |  Classical random price model without memory   |
| Geometric Brownian Motion  |  Classical random price model without memory, and where prices are postive  |
| Ornstein Uhlenbeck  |  Mean reverting model   | 
| Garch  |  Dynamical volatility  | 
| Dynamic Conditional Correlation GARCH | Dynamical volatility and correlations |
| Markov Chains |  Discrete state transitioning model   | 
| Phase Randomisation  | Randomization that preserves return distributions and autocorrelations   |
| Fractal Brownian Motion | Brownian motion with trends or mean reversion | 


