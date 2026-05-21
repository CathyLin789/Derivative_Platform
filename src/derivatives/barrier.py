"""
derivatives/barrier.py
----------------------
Barrier options: path-dependent contracts whose payoff depends on
whether the underlying price crosses a specified barrier level
at any point during the option's life.

Four standard subtypes:
    - up-and-out:    knocks out  (becomes worthless) if S ever rises above B
    - up-and-in:     knocks in   (activates)         if S ever rises above B
    - down-and-out:  knocks out                       if S ever falls below B
    - down-and-in:   knocks in                        if S ever falls below B

Priced via crude Monte Carlo (see src/pricers/monte_carlo.py) by simulating
price paths under the risk-neutral measure and checking the barrier
condition path-by-path. This is the "crude MC" approach: no variance
reduction, no antithetic sampling, no quasi-Monte Carlo.

Design note: subtype is encoded as a parameter (barrier_type), not a
subclass. All four behaviours share the same payoff structure — only
the activation condition differs. One class per direction (Call vs Put)
with a parameter to control activation keeps the contract hierarchy flat.
"""

import numpy as np

from .base_derivatives import Derivative


# All valid barrier subtype strings, exposed for validation and reference
VALID_BARRIER_TYPES = ("up-and-out", "up-and-in", "down-and-out", "down-and-in")


class _BarrierMixin:
    """
    Shared barrier mechanics for BarrierCall and BarrierPut.

    Holds the barrier-level B and subtype string, validates them at
    construction time, and provides the path-conditional payoff logic.
    """

    is_path_dependent = True

    def __init__(self, S0, K, T, sigma, yield_curve, barrier, barrier_type):
        super().__init__(S0, K, T, sigma, yield_curve)
        self.barrier = float(barrier)
        self.barrier_type = barrier_type
        self._validate_barrier()

    def _validate_barrier(self):
        """Sanity checks specific to barrier options."""
        if self.barrier <= 0:
            raise ValueError(f"barrier must be positive, got {self.barrier}")
        if self.barrier_type not in VALID_BARRIER_TYPES:
            raise ValueError(
                f"barrier_type must be one of {VALID_BARRIER_TYPES}, "
                f"got {self.barrier_type!r}"
            )
        # Logical sanity: 'up' barriers should be above current spot;
        # 'down' barriers below. Warn but don't raise: someone may want
        # an unusual setup deliberately.
        if self.barrier_type.startswith("up") and self.barrier <= self.S0:
            pass  # tolerated; unusual setup
        if self.barrier_type.startswith("down") and self.barrier >= self.S0:
            pass  # tolerated; unusual setup

    def payoff_path(self, S_path):
        """
        Compute terminal payoff conditional on the barrier event.

        Parameters
        ----------
        S_path : np.ndarray
            Shape (n_paths, n_steps + 1). Column 0 is S0; column -1 is S_T.

        Returns
        -------
        np.ndarray
            Shape (n_paths,). Payoff per simulated path.

        Logic
        -----
        For "up" barriers we check the path maximum against B.
        For "down" barriers we check the path minimum against B.

        For "knock-out" subtypes: payoff is the terminal payoff ONLY IF
        the barrier was never breached. If breached, payoff = 0.

        For "knock-in" subtypes: payoff is the terminal payoff ONLY IF
        the barrier WAS breached. If never breached, payoff = 0.
        """
        # Standard terminal payoff: max(S_T - K, 0) for calls, max(K - S_T, 0) for puts
        terminal_payoff = self.payoff(S_path[:, -1])

        if self.barrier_type.startswith("up"):
            # Did any path step exceed the barrier?
            barrier_breached = (S_path.max(axis=1) >= self.barrier)
        else:
            # Did any path step fall below the barrier?
            barrier_breached = (S_path.min(axis=1) <= self.barrier)

        if self.barrier_type.endswith("out"):
            # Knock-out: payoff if NOT breached
            return np.where(barrier_breached, 0.0, terminal_payoff)
        else:
            # Knock-in: payoff only if breached
            return np.where(barrier_breached, terminal_payoff, 0.0)

    def __repr__(self):
        family = type(self).__name__
        return (
            f"{family}(S0={self.S0}, K={self.K}, T={self.T}, "
            f"sigma={self.sigma}, barrier={self.barrier}, "
            f"barrier_type={self.barrier_type!r})"
        )


class BarrierCall(_BarrierMixin, Derivative):
    """
    Barrier call option. Path-dependent.

    Parameters
    ----------
    S0, K, T, sigma, yield_curve : as in Derivative base class
    barrier : float
        The barrier level B.
    barrier_type : str
        One of "up-and-out", "up-and-in", "down-and-out", "down-and-in".

    Pricing
    -------
    Priced via Monte Carlo only (set is_path_dependent=True). No closed-form
    or binomial-tree implementation in this prototype.

    Example payoffs
    ---------------
    up-and-out call:    max(S_T - K, 0) if max(S_path) < B  else 0
    up-and-in call:     max(S_T - K, 0) if max(S_path) >= B else 0
    down-and-out call:  max(S_T - K, 0) if min(S_path) > B  else 0
    down-and-in call:   max(S_T - K, 0) if min(S_path) <= B else 0
    """

    option_type = "call"
    is_american = False


class BarrierPut(_BarrierMixin, Derivative):
    """
    Barrier put option. Path-dependent.

    See BarrierCall for parameter and pricing details.

    Example payoffs
    ---------------
    down-and-in put:    max(K - S_T, 0) if min(S_path) <= B else 0  ← TAIL HEDGE
    down-and-out put:   max(K - S_T, 0) if min(S_path) > B  else 0
    up-and-in put:      max(K - S_T, 0) if max(S_path) >= B else 0
    up-and-out put:     max(K - S_T, 0) if max(S_path) < B  else 0

    The down-and-in put is the canonical "cheap disaster insurance" trade:
    only pays out in a sharp drop, so the premium is small.
    """

    option_type = "put"
    is_american = False
