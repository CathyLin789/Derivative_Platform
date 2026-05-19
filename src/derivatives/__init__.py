"""
src.derivatives package
-----------------------
Derivative contract definitions, organised one family per file.

This __init__.py re-exports every contract class so that external
callers can write:

    from src.derivatives import EuropeanCall, AmericanPut, BarrierCall

without needing to know which file each class lives in.
"""

from .base_derivatives import Derivative
from .european import EuropeanCall, EuropeanPut
from .american import AmericanCall, AmericanPut
from .barrier import BarrierCall, BarrierPut

__all__ = [
    "Derivative",
    "EuropeanCall",
    "EuropeanPut",
    "AmericanCall",
    "AmericanPut",
    "BarrierCall",
    "BarrierPut",
]
