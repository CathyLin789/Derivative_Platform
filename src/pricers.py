"""
pricers.py
----------
Pricing methods for derivative contracts.

Design principle (separation of concerns, see derivatives.py):
    Pricers describe HOW a contract is valued (numerical method).
    Contracts describe WHAT is being valued (parameters, payoff).

Each pricer is responsible for:
    1. Checking that the contract is compatible (e.g. BS rejects Americans)
    2. Computing a price from the contract's parameters and the yield curve
"""

import numpy as np
from scipy.stats import norm


class Pricer:
    """
    Base class for all pricers.

    Subclasses must implement price(contract) and may override
    _check_compatible(contract) to declare which contract types they support.
    """

    name = "Pricer"  # override in subclasses

    def price(self, contract):
        raise NotImplementedError("Subclasses must implement price()")

    def _check_compatible(self, contract):
        """Override in subclasses to restrict supported contract types."""
        pass


class BlackScholesPricer(Pricer):
    """
    Black-Scholes-Merton closed-form pricer.

    Supports:
        European calls and puts, with continuous dividend yield q.

    Does NOT support:
        American options (early exercise has no closed-form solution)
        Path-dependent options (Asian, barrier, etc.)

    Formulas (with continuous dividend yield q):
        d1 = [ln(S0/K) + (r - q + 0.5*sigma^2)*T] / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

        Call: C = S0 * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)
        Put:  P = K * exp(-r*T) * N(-d2) - S0 * exp(-q*T) * N(-d1)

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
        Return the Black-Scholes-Merton price of a European option.

        Parameters
        ----------
        contract : Derivative
            A European call or put (with optional dividend yield q).

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
        q = contract.q
        r = contract.yield_curve.get_zero_rate(T)

        # Edge case: sigma = 0 -> option payoff becomes deterministic
        # (computed as the discounted intrinsic value of the forward price)
        if sigma == 0:
            forward = S0 * np.exp((r - q) * T)
            disc = np.exp(-r * T)
            if contract.option_type == "call":
                return disc * max(forward - K, 0.0)
            else:
                return disc * max(K - forward, 0.0)

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        if contract.option_type == "call":
            return (
                S0 * np.exp(-q * T) * norm.cdf(d1)
                - K * np.exp(-r * T) * norm.cdf(d2)
            )
        elif contract.option_type == "put":
            return (
                K * np.exp(-r * T) * norm.cdf(-d2)
                - S0 * np.exp(-q * T) * norm.cdf(-d1)
            )
        else:
            raise ValueError(f"Unknown option_type: {contract.option_type}")


class BinomialTreePricer(Pricer):
    """
    Cox-Ross-Rubinstein (CRR) binomial tree pricer.

    Supports:
        European and American calls and puts (with continuous dividend yield q).

    Does NOT support:
        Path-dependent options (Asian, barrier, etc.) -- the tree only
        tracks terminal prices at each node, not the full price path.

    Method
    ------
    Time to maturity T is divided into N steps of length dt = T/N.
    At each step the stock can move up by factor u = exp(sigma*sqrt(dt))
    or down by factor d = 1/u. The risk-neutral up probability is
        p = (exp((r - q) * dt) - d) / (u - d)

    Pricing proceeds by:
        1. Building the tree of terminal stock prices at step N
        2. Computing payoffs at each terminal node
        3. Walking backwards: at each node, the option value is the
           discounted expected value of its two child nodes
        4. For American options, at each interior node we also compare
           to the immediate-exercise payoff and take the maximum

    The price converges to Black-Scholes as N -> infinity (for European
    options). The Stage 5 notebook plots this convergence as a validation
    check.
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
        q = contract.q
        r = contract.yield_curve.get_zero_rate(T)
        N = self.N

        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        disc = np.exp(-r * dt)             # per-step discount factor
        p = (np.exp((r - q) * dt) - d) / (u - d)   # risk-neutral up probability

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
    
class MonteCarloPricer(Pricer):
    """
    Monte Carlo pricer using geometric Brownian motion under the
    risk-neutral measure.

    Supports:
        European calls and puts (used to validate against Black-Scholes)
        Asian (path-dependent average-price) calls and puts
        In general, any path-independent or path-dependent contract whose
        payoff can be expressed as a function of the price path.

    Does NOT support:
        American options. Pricing American options with Monte Carlo
        requires the Longstaff-Schwartz regression algorithm, which is
        out of scope for this prototype. Use BinomialTreePricer instead
        (exact and faster for low-dimensional problems).

    Method
    ------
    Under the risk-neutral measure, the underlying follows geometric
    Brownian motion:
        dS = (r - q) * S * dt + sigma * S * dW

    Discretised log-Euler scheme:
        S(t + dt) = S(t) * exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z),
        Z ~ N(0, 1)

    The Ito correction term -0.5*sigma^2 ensures the simulated process
    has the correct mean E[S(t)] = S(0) * exp((r-q)*t).

    Pricing proceeds by:
        1. Simulate n_paths price paths of n_steps each
        2. Compute the payoff for each path (using contract.payoff_path)
        3. Discount payoffs to present value at the zero rate
        4. Price estimate = mean of discounted payoffs
        5. Standard error = std(discounted payoffs) / sqrt(n_paths)

    The standard error decays at rate 1/sqrt(n_paths) -- halving the
    error requires quadrupling paths. This is plotted as a validation
    check in the Stage 5 notebook.
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
        q = contract.q
        r = contract.yield_curve.get_zero_rate(T)

        # Auto-choose n_steps based on whether path matters
        if self.n_steps is None:
            n_steps = 252 if contract.is_path_dependent else 1
        else:
            n_steps = self.n_steps

        dt = T / n_steps
        drift = (r - q - 0.5 * sigma ** 2) * dt
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
        # Two-sided z-score for the given confidence level
        z = norm.ppf(0.5 + confidence / 2)
        # The mean of last_payoffs is the price we returned
        center = float(np.mean(self.last_payoffs))
        half = z * self.last_std_error
        return (center - half, center + half)
