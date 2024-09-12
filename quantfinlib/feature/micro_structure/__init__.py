"""Micro-structure features module.

This module contains functions to calculate micro-structure features 
based on the book: Advances in Financial Machine Learning by Marcos Lopez de Prado.

The implemented features are:

 First generation: Price sequences
    The tick rule
    The roll model
    The high-low volatility estimator
    The Corwin-Schulz bid-ask spread model

Second generation: Strategic trade models
    Kyle's lambda
    Amihud's lambda
    Hasbrouck's lambda

Third generation: Sequential trade models
    Probability of information-based trading
    Volume-synchronized probability of informed trading

    
References
----------
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3270269
"""
