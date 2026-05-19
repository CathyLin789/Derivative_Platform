"""
src.pricers package
-------------------
Pricing methods for derivative contracts, organised one method per file.

This __init__.py re-exports every pricer class so that external
callers can write:

    from src.pricers import BlackScholesPricer, BinomialTreePricer, MonteCarloPricer

without needing to know which file each class lives in.
"""

from .base_pricers import Pricer
from .black_scholes import BlackScholesPricer
from .binomial import BinomialTreePricer
from .monte_carlo import MonteCarloPricer

__all__ = [
    "Pricer",
    "BlackScholesPricer",
    "BinomialTreePricer",
    "MonteCarloPricer",
]
