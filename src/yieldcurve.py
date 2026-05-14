import numpy as np
import matplotlib.pyplot as plt


class YieldCurve:
    """
    YieldCurve represents a term structure of zero rate and 
    provides discount factors for valuation.
    
    This class is infrastructure: all interest rate logic 
    should live here and be reused by other models
    """
    def __init__(self, maturities, zero_rates, compounding = "continuous"):
        """
        Maturities: the time at which the principle is returned to the bond purchaser

        Zero Rate: the price-implied discount rate on a zero-coupon bond

        Compounding (annual):
        """
        self.maturities = np.array(maturities, dtype=float)
        self.zero_rates = np.array(zero_rates, dtype=float)
        self.compounding = compounding
       
        if not(self.compounding == "continuous" or self.compounding == "annual"):
            raise ValueError("Unsupported compounding type")
        
        if len(self.maturities) != len(self.zero_rates):
            raise ValueError("Must have the same number of maturities and zero rates")
        
        order = np.argsort(self.maturities)
        self.maturities = self.maturities[order]
        self.zero_rates = self.zero_rates[order]

    def get_zero_rate(self, T):
        """
        Return the interpolated zro rate for maturity T (in years)
        """
        T = float(T)
        return float(np.interp(T, self.maturities, self.zero_rates))
    
    def get_discount_factor(self, T):
        """
        Return the discount factor using yield curve
        """
        z = self.get_zero_rate(T)

        if self.compounding == "continuous":
            return np.exp(-z * T)
        
        elif self.compounding == "annual":
            return 1.0 / (1.0 + z) ** T
        
    def plot(self, max_maturity=None):
        """
        plot the zero-rate yield curve
        """
        if max_maturity is None:
            T_grid = self.maturities
        else:
            T_grid = np.linspace(
                self.maturities.min(),
                max_maturity,
                100
            )
        z_grid = [self.get_zero_rate(T) for T in T_grid]

        plt.figure()
        plt.plot(T_grid, z_grid)
        plt.xlabel("Maturity (Years)")
        plt.ylabel("Zero Rate (%)")
        plt.title("Zero Rate Yield Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

