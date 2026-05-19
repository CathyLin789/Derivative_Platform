"""
derivatives/base.py
-------------------
Abstract base class for all derivative contracts.

Design principle (separation of concerns):
    Contracts describe WHAT is being valued (parameters, payoff structure).
    Pricers (see src/pricers/) describe HOW it is valued (numerical method).

The same contract can therefore be priced by multiple pricers, which makes
cross-validation (e.g. Monte Carlo against Black-Scholes) natural and is
the basis for the validation notebook.
"""

import numpy as np


class Derivative:
    """
    Base class for derivative contracts.

    Holds the shared parameters every option needs and declares the
    interface that pricers rely on:
        - payoff(S_terminal):   terminal payoff (path-independent options)
        - payoff_path(S_path):  payoff given a full price path (path-dependent options)

    Concrete subclasses must set:
        option_type       : "call" or "put"
        is_american       : True if the option can be exercised early
        is_path_dependent : True if payoff depends on the whole path, not just S_T
    """

    # Defaults overridden by subclasses
    option_type = None
    is_american = False
    is_path_dependent = False

    def __init__(self, S0, K, T, sigma, yield_curve):
        """
        Parameters
        ----------
        S0 : float
            Current price of the underlying asset.
        K : float
            Strike (exercise) price.
        T : float
            Time to maturity in years.
        sigma : float
            Annualised volatility of the underlying (decimal, e.g. 0.20 = 20%).
        yield_curve : YieldCurve
            Yield curve object providing get_zero_rate(T) and get_discount_factor(T).
        """
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.sigma = float(sigma)
        self.yield_curve = yield_curve

        self._validate()

    def _validate(self):
        """Basic sanity checks on contract parameters."""
        if self.S0 <= 0:
            raise ValueError("S0 must be positive")
        if self.K <= 0:
            raise ValueError("K must be positive")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

    # ------------------------------------------------------------------
    # Payoff interface (used by pricers)
    # ------------------------------------------------------------------
    def payoff(self, S_terminal):
        """
        Terminal payoff for a path-independent option.

        Parameters
        ----------
        S_terminal : float or np.ndarray
            Underlying price(s) at maturity.

        Returns
        -------
        float or np.ndarray
            Payoff value(s).
        """
        S_terminal = np.asarray(S_terminal)
        if self.option_type == "call":
            return np.maximum(S_terminal - self.K, 0.0)
        elif self.option_type == "put":
            return np.maximum(self.K - S_terminal, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def payoff_path(self, S_path):
        """
        Payoff for a path-dependent option, given the full price path.

        Default implementation uses the terminal payoff (i.e. ignores path
        history). Path-dependent subclasses (Barrier, etc.) override this.

        Parameters
        ----------
        S_path : np.ndarray
            Array of shape (n_paths, n_steps+1) containing simulated price paths.

        Returns
        -------
        np.ndarray
            One payoff per path, shape (n_paths,).
        """
        S_terminal = S_path[:, -1]
        return self.payoff(S_terminal)

    def __repr__(self):
        family = type(self).__name__  # e.g. "EuropeanCall", "AmericanPut"
        return (
            f"{family}(S0={self.S0}, K={self.K}, T={self.T}, "
            f"sigma={self.sigma})"
        )
