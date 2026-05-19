"""
pricers/monte_carlo.py
----------------------
Monte Carlo pricer for European and path-dependent options
(used to validate Black-Scholes for Europeans and to price Barriers).
"""

import numpy as np
from scipy.stats import norm

from .base import Pricer


class MonteCarloPricer(Pricer):
    """
    Monte Carlo pricer using geometric Brownian motion under the
    risk-neutral measure.

    Supports:
        European calls and puts (used to validate against Black-Scholes).
        Barrier (path-dependent) calls and puts.
        In general, any path-independent or path-dependent contract whose
        payoff can be expressed as a function of the price path.

    Does NOT support:
        American options. Pricing American options with Monte Carlo
        requires the Longstaff-Schwartz regression algorithm, which is
        out of scope for this prototype. Use BinomialTreePricer instead
        (exact and faster for low-dimensional problems).

    Method (no dividends)
    ---------------------
    Under the risk-neutral measure, the underlying follows geometric
    Brownian motion:
        dS = r * S * dt + sigma * S * dW

    Discretised log-Euler scheme:
        S(t + dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z),
        Z ~ N(0, 1)

    The Ito correction term -0.5*sigma^2 ensures the simulated process
    has the correct mean E[S(t)] = S(0) * exp(r*t).

    Pricing proceeds by:
        1. Simulate n_paths price paths of n_steps each
        2. Compute the payoff for each path (using contract.payoff_path)
        3. Discount payoffs to present value at the zero rate
        4. Price estimate = mean of discounted payoffs
        5. Standard error = std(discounted payoffs) / sqrt(n_paths)

    The standard error decays at rate 1/sqrt(n_paths) -- halving the
    error requires quadrupling paths.
    """

    name = "Monte Carlo"

    def __init__(self, n_paths=50_000, n_steps=None, seed=None):
        """
        Parameters
        ----------
        n_paths : int
            Number of simulated price paths. More paths -> lower standard
            error but slower. Default 50,000 is a balance.
        n_steps : int or None
            Number of time steps per path. If None, use n_steps=1 for
            path-independent options (only terminal price matters) and
            n_steps=252 (one per trading day per year) for path-dependent.
            Larger values give finer path resolution at higher cost.
        seed : int or None
            Random seed for reproducibility. If None, results vary run-to-run.
        """
        if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
            raise ValueError("n_paths must be a positive integer")
        if n_steps is not None and (
            not isinstance(n_steps, (int, np.integer)) or n_steps < 1
        ):
            raise ValueError("n_steps must be a positive integer or None")

        self.n_paths = int(n_paths)
        self.n_steps = None if n_steps is None else int(n_steps)
        self.seed = seed

        # Populated after each .price() call -- accessible for diagnostics
        self.last_std_error = None
        self.last_payoffs = None

    def _check_compatible(self, contract):
        if contract.is_american:
            raise ValueError(
                f"{self.name} cannot price American options with the naive "
                f"algorithm (would require Longstaff-Schwartz regression). "
                f"Use BinomialTreePricer instead."
            )

    def _simulate_paths(self, contract):
        """
        Simulate price paths under the risk-neutral measure.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). Column 0 is S0; column n_steps is S_T.
        """
        S0 = contract.S0
        T = contract.T
        sigma = contract.sigma
        r = contract.yield_curve.get_zero_rate(T)

        # Auto-choose n_steps based on whether path matters
        if self.n_steps is None:
            n_steps = 252 if contract.is_path_dependent else 1
        else:
            n_steps = self.n_steps

        dt = T / n_steps
        drift = (r - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)

        rng = np.random.default_rng(self.seed)

        # Simulate log-returns, then build paths cumulatively
        Z = rng.standard_normal((self.n_paths, n_steps))
        log_returns = drift + diffusion * Z

        # Cumulative log-returns -> price path
        log_S = np.log(S0) + np.cumsum(log_returns, axis=1)
        S_paths = np.concatenate(
            [np.full((self.n_paths, 1), S0), np.exp(log_S)], axis=1
        )

        return S_paths

    def price(self, contract):
        """
        Return the Monte Carlo price estimate.

        Also stores:
            self.last_std_error : float
                Standard error of the price estimate.
            self.last_payoffs : np.ndarray
                Discounted payoffs from each simulated path (for diagnostics).
        """
        self._check_compatible(contract)

        T = contract.T
        r = contract.yield_curve.get_zero_rate(T)
        discount = np.exp(-r * T)

        S_paths = self._simulate_paths(contract)
        payoffs = contract.payoff_path(S_paths)
        discounted_payoffs = discount * payoffs

        price = float(np.mean(discounted_payoffs))
        std_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(self.n_paths))

        self.last_std_error = std_error
        self.last_payoffs = discounted_payoffs

        return price

    def last_confidence_interval(self, confidence=0.95):
        """
        Return the (lower, upper) confidence interval bounds for the most
        recent price estimate.

        Parameters
        ----------
        confidence : float
            Confidence level. Default 0.95 -> ~1.96 standard errors.

        Returns
        -------
        (float, float)
            Lower and upper CI bounds, centred on the last price estimate.
        """
        if self.last_std_error is None:
            raise RuntimeError(
                "No price computed yet. Call .price(contract) first."
            )
        z = norm.ppf(0.5 + confidence / 2)
        center = float(np.mean(self.last_payoffs))
        half = z * self.last_std_error
        return (center - half, center + half)
