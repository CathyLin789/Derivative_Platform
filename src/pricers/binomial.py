"""
pricers/binomial.py
-------------------
Cox-Ross-Rubinstein binomial tree pricer for European and American options.
"""

import numpy as np

from .base import Pricer


class BinomialTreePricer(Pricer):
    """
    Cox-Ross-Rubinstein (CRR) binomial tree pricer.

    Supports:
        European and American calls and puts.

    Does NOT support:
        Path-dependent options (Barrier, etc.) -- the tree only tracks
        terminal prices at each node, not the full price path.

    Method (no dividends)
    ---------------------
    Time to maturity T is divided into N steps of length dt = T/N.
    At each step the stock can move up by factor u = exp(sigma*sqrt(dt))
    or down by factor d = 1/u. The risk-neutral up probability is
        p = (exp(r * dt) - d) / (u - d)

    Pricing proceeds by:
        1. Building the tree of terminal stock prices at step N
        2. Computing payoffs at each terminal node
        3. Walking backwards: at each node, the option value is the
           discounted expected value of its two child nodes
        4. For American options, at each interior node we also compare
           to the immediate-exercise payoff and take the maximum

    The price converges to Black-Scholes as N -> infinity for European
    options.
    """

    name = "Binomial Tree"

    def __init__(self, N=200):
        """
        Parameters
        ----------
        N : int
            Number of time steps in the tree. Larger N -> more accurate
            but slower. Default 200 is a standard choice that balances
            accuracy and speed.
        """
        if not isinstance(N, (int, np.integer)) or N < 1:
            raise ValueError("N must be a positive integer")
        self.N = int(N)

    def _check_compatible(self, contract):
        if contract.is_path_dependent:
            raise ValueError(
                f"{self.name} cannot price path-dependent options "
                f"({type(contract).__name__}). Use MonteCarloPricer instead."
            )

    def price(self, contract):
        """
        Return the CRR binomial tree price of a European or American option.
        """
        self._check_compatible(contract)

        S0 = contract.S0
        K = contract.K
        T = contract.T
        sigma = contract.sigma
        r = contract.yield_curve.get_zero_rate(T)
        N = self.N

        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        disc = np.exp(-r * dt)             # per-step discount factor
        p = (np.exp(r * dt) - d) / (u - d)  # risk-neutral up probability

        # Sanity check: p must be a valid probability for the tree to be
        # arbitrage-free. If it isn't, sigma is too small or dt too large.
        if not (0 < p < 1):
            raise ValueError(
                f"Risk-neutral probability p={p:.4f} is outside (0, 1). "
                f"Try increasing N or sigma."
            )

        # Step 1: terminal stock prices at maturity (N+1 nodes)
        # Node j has had j up-moves and (N-j) down-moves
        j = np.arange(N + 1)
        S_terminal = S0 * (u ** j) * (d ** (N - j))

        # Step 2: terminal payoffs
        values = contract.payoff(S_terminal)

        # Step 3: walk backwards through the tree
        for step in range(N - 1, -1, -1):
            # Discounted expected continuation value at each node of this step
            values = disc * (p * values[1:] + (1 - p) * values[:-1])

            if contract.is_american:
                # Stock prices at this step (step+1 nodes)
                j = np.arange(step + 1)
                S_step = S0 * (u ** j) * (d ** (step - j))
                exercise = contract.payoff(S_step)
                # Take max of continuation and immediate exercise
                values = np.maximum(values, exercise)

        return float(values[0])
