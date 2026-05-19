"""
derivatives/american.py
-----------------------
American-style options: exercisable at any time up to maturity.

Priced numerically via the Cox-Ross-Rubinstein binomial tree
(see src/pricers/binomial.py). At each node, the binomial pricer
compares the continuation value against the immediate-exercise
payoff and takes the maximum.
"""

from .base import Derivative


class AmericanCall(Derivative):
    """
    American call option.
    Payoff if exercised at time t: max(S_t - K, 0)
    Exercise: any time t in [0, T].

    Without dividends, early exercise of an American call is never
    optimal (so AmericanCall == EuropeanCall in our model). With
    dividends it can be — but dividends are out of scope for this prototype.
    """
    option_type = "call"
    is_american = True
    is_path_dependent = False


class AmericanPut(Derivative):
    """
    American put option.
    Payoff if exercised at time t: max(K - S_t, 0)
    Exercise: any time t in [0, T].

    Early exercise of an American put CAN be optimal even without
    dividends, because exercising lets the holder earn interest on
    the strike K from time t to T. Hence AmericanPut >= EuropeanPut.
    This inequality is used as a validation check in the notebook.
    """
    option_type = "put"
    is_american = True
    is_path_dependent = False
