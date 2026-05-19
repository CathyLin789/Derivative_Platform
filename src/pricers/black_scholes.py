"""
pricers/black_scholes.py
------------------------
Black-Scholes closed-form pricer for European options.
"""

import numpy as np
from scipy.stats import norm

from .base_pricers import Pricer


class BlackScholesPricer(Pricer):
    """
    Black-Scholes closed-form pricer.

    Supports:
        European calls and puts.

    Does NOT support:
        American options (early exercise has no closed-form solution).
        Path-dependent options (Barrier, etc.).

    Formulas (no dividends):
        d1 = [ln(S0/K) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

        Call: C = S0 * N(d1) - K * exp(-r*T) * N(d2)
        Put:  P = K * exp(-r*T) * N(-d2) - S0 * N(-d1)

    where:
        r     = zero rate at maturity T (from the yield curve)
        N(.)  = standard normal CDF
    """

    name = "Black-Scholes"

    def _check_compatible(self, contract):
        if contract.is_american:
            raise ValueError(
                f"{self.name} cannot price American options "
                f"(early exercise has no closed-form solution). "
                f"Use BinomialTreePricer instead."
            )
        if contract.is_path_dependent:
            raise ValueError(
                f"{self.name} cannot price path-dependent options "
                f"({type(contract).__name__}). Use MonteCarloPricer instead."
            )

    def price(self, contract):
        """
        Return the Black-Scholes price of a European option.

        Parameters
        ----------
        contract : Derivative
            A European call or put.

        Returns
        -------
        float
            Option price.
        """
        self._check_compatible(contract)

        S0 = contract.S0
        K = contract.K
        T = contract.T
        sigma = contract.sigma
        r = contract.yield_curve.get_zero_rate(T)

        # Edge case: sigma = 0 -> option payoff is deterministic
        # (the discounted intrinsic value of the forward price)
        if sigma == 0:
            forward = S0 * np.exp(r * T)
            disc = np.exp(-r * T)
            if contract.option_type == "call":
                return disc * max(forward - K, 0.0)
            else:
                return disc * max(K - forward, 0.0)

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        if contract.option_type == "call":
            return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif contract.option_type == "put":
            return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        else:
            raise ValueError(f"Unknown option_type: {contract.option_type}")
