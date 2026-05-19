"""
pricers/base.py
---------------
Abstract base class for all pricers.

Design principle (separation of concerns, see src/derivatives/base.py):
    Pricers describe HOW a contract is valued (numerical method).
    Contracts describe WHAT is being valued (parameters, payoff).

Each pricer is responsible for:
    1. Checking that the contract is compatible
       (e.g. BS rejects American and path-dependent options)
    2. Computing a price from the contract's parameters and the yield curve
"""


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
