"""
derivatives/barrier.py
----------------------
Barrier options: path-dependent contracts whose payoff depends on
whether the underlying price crosses a specified barrier level
at any point during the option's life.

Four standard types:
    - Up-and-out:   knocks out (becomes worthless) if S ever rises above B
    - Up-and-in:    knocks in  (activates)         if S ever rises above B
    - Down-and-out: knocks out                     if S ever falls below B
    - Down-and-in:  knocks in                      if S ever falls below B

Priced via crude Monte Carlo (see src/pricers/monte_carlo.py) by simulating
price paths under the risk-neutral measure and checking the barrier
condition path-by-path.

STATUS: stubs only. The barrier MC pricer and payoff_path logic
will be implemented as a separate workstream after the folder
refactor is verified.
"""

from .base import Derivative


class BarrierCall(Derivative):
    """
    Barrier call option (stub).

    Will require additional attributes:
        - barrier : float
            The barrier level B.
        - barrier_type : str
            One of "up-and-out", "up-and-in", "down-and-out", "down-and-in".

    Payoff (e.g. up-and-out): max(S_T - K, 0) if max(S_t) < B else 0.
    """
    option_type = "call"
    is_american = False
    is_path_dependent = True

    # __init__ and payoff_path will be implemented in the barrier workstream.


class BarrierPut(Derivative):
    """
    Barrier put option (stub).

    See BarrierCall docstring for design.
    Payoff (e.g. down-and-out): max(K - S_T, 0) if min(S_t) > B else 0.
    """
    option_type = "put"
    is_american = False
    is_path_dependent = True

    # __init__ and payoff_path will be implemented in the barrier workstream.
