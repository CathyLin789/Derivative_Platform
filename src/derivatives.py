import numpy as np
from scipy.stats import norm

class Derivative:
    """
    Base class for derivative contracts
    
    Defines the shared strcuture for all derivatives
    
    Does not implement any pricing logic
    """

    def __init__(self, S0, K, T, sigma, yield_curve):
        """
        Define all parameters. e.g.
        S0: float
            Current price of the underlying asset
        K:
        T:
        sigma:
        yield_curve:
        """

        self.S0 = S0
        self.K = K
        self.T = T
        self.sigma = sigma
        self.yield_curve = yield_curve

    def price(self):
        """
        Compute the price of the derviative
        
        Must be implemented within subclass (different pricing methods)
        """
        raise NotImplementedError(
            "Pricing logic must be implemented in the subclass"
        )

class EuropeanCall(Derivative):
    """
    European call option priced using Black-Scholes
    """
    def price(self):
        """
        Return the Black-Scholes price of a European call option
        """
        # get zero rate for option maturity
        r = self.yield_curve.get_zero_rate(self.T)

        #d1 and d2

        d1 = (
            np.log(self.S0 / self.K)
            + (r + 0.5 * self.sigma ** 2) * self.T
            ) /(self.sigma * np.sqrt(self.T))
        
        d2 = d1 - self.sigma * np.sqrt(self.T)

        # Black-Scholes call price
        call_price = (
            self.S0 * norm.cdf(d1)
            - self.K
            * np.exp(-r * self.T)
            * norm.cdf(d2)
        )

        return call_price



class EuropeanPut(Derivative):

    def price(self):
        """
        Return the Black-Scholes price of a European put option
        """
        # get zero rate for option maturity
        r = self.yield_curve.get_zero_rate(self.T)

        #d1 and d2

        d1 = (
            np.log(self.S0 / self.K)
            + (r + 0.5 * self.sigma ** 2) * self.T
            ) /(self.sigma * np.sqrt(self.T))
        
        d2 = d1 - self.sigma * np.sqrt(self.T)

        # Black-Scholes call price
        put_price = (
            self.K
            * np.exp(-r * self.T)
            * norm.cdf(-d2)
            -self.S0 * norm.cdf(-d1)
        )

        return put_price