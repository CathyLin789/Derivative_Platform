"""
derivatives/european.py
-----------------------
European-style options: exercisable only at maturity.

Priced analytically via the Black-Scholes formula
(see src/pricers/black_scholes.py).
"""

from .base_derivatives import Derivative


class EuropeanCall(Derivative):
    """
    European call option.
    Payoff at maturity: max(S_T - K, 0)
    Exercise: only at T (no early exercise).
    """
    option_type = "call"
    is_american = False
    is_path_dependent = False


class EuropeanPut(Derivative):
    """
    European put option.
    Payoff at maturity: max(K - S_T, 0)
    Exercise: only at T (no early exercise).
    """
    option_type = "put"
    is_american = False
    is_path_dependent = False
