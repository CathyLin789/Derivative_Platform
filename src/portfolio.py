"""
portfolio.py
------------
Portfolio construction and risk analysis layer.

Sits on top of derivatives/, pricers/, and yieldcurve.py to provide:
    - EquityPosition   : a long/short equity holding
    - OptionPosition   : a long/short derivative position with an attached pricer
    - Portfolio        : a collection of positions with aggregate valuation and Greeks
    - RiskEngine       : VaR (historical and parametric), scenario analysis,
                         and delta-linearisation vs full-revaluation P&L curves
 
Design principle (consistent with the rest of the platform):
    This module does NOT hard-code any specific stocks or options.
    Positions are constructed externally (e.g. in the notebook) and passed in.
 
Greeks design note:
    All Greeks are computed via central finite difference on the contract's
    underlying inputs. This is *model-consistent*: it works for any pricer
    (Black-Scholes, binomial tree, Monte Carlo) without requiring closed-form
    derivatives. Equity positions return zero for gamma and vega — equity is
    linear in spot and has no volatility exposure — which keeps Portfolio
    aggregation polymorphic (no isinstance branches in the aggregation loop).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm



# ======================================================================
# Position classes
# ======================================================================

class EquityPosition:
    """
    A long equity holding in a single stock.
    """

    def __init__(self, ticker, quantity, spot_price):
        if spot_price <= 0:
            raise ValueError(f"spot_price must be positive, got {spot_price}")
        self.ticker     = ticker
        self.quantity   = float(quantity)
        self.spot_price = float(spot_price)
        self.label      = ticker

    def price(self):
        """Unit price (spot)."""
        return self.spot_price
        
    def value(self):
        """Total market value of the equity position (AUD)."""
        return self.quantity * self.spot_price

    def delta(self):
        """Unit delta = 1.0 for equity. Position delta = quantity * 1.0."""
        return 1.0
 
    def gamma(self):
        """Equity is linear in spot — zero gamma."""
        return 0.0
 
    def vega(self):
        """Equity has no volatility exposure — zero vega."""
        return 0.0
 
    def theta(self):
        """Equity does not decay with time — zero theta."""
        return 0.0
 
    def delta_dollars(self):
        """
        Delta-dollars = position's first-order $ sensitivity to a 100%
        proportional move in its underlying.
 
        For equity: delta_dollars = quantity * spot * 1.0 = market value.
        """
        return self.quantity * self.spot_price * self.delta()
 
    def reprice(self, new_spot=None, new_yield_curve=None):
        """Return a new EquityPosition with shocked spot price."""
        spot = new_spot if new_spot is not None else self.spot_price
        return EquityPosition(
            ticker=self.ticker,
            quantity=self.quantity,
            spot_price=spot,
        )
 
    def __repr__(self):
        return (
            f"EquityPosition(ticker={self.ticker!r}, "
            f"quantity={self.quantity}, spot_price={self.spot_price:.2f})"
        )


class OptionPosition:
    """
    A long derivative position with an attached pricer.
 
    Greeks (delta, gamma, vega) are all computed via central finite difference
    on the contract's inputs — model-consistent and works for any pricer.
 
    Note on conventions:
        - `delta()`, `gamma()`, `vega()` return *position-level* values
          (already multiplied by quantity).
        - This matches the existing delta() convention used elsewhere in the
          codebase; do not pass these into formulas that re-multiply by qty.
    """
    # Relative bump sizes for finite-difference Greeks
    _DELTA_BUMP = 1e-3   # 0.1% of spot
    _GAMMA_BUMP = 1e-3   # 0.1% of spot (same as delta for consistency)
    _VEGA_BUMP  = 1e-4   # absolute σ bump (1bp of vol)
    _THETA_DT   = 1.0 / 252.0   # one trading day, in years (theta horizon)
 
    def __init__(self, contract, pricer, quantity, underlying_ticker, label=None):
        self.contract           = contract
        self.pricer             = pricer
        self.quantity           = float(quantity)
        self.underlying_ticker  = underlying_ticker
        self.label              = label or repr(contract)
 
    def price(self):
        """Unit price (single contract)."""
        return self.pricer.price(self.contract)
 
    def value(self):
        """Total value of the option position (AUD)."""
        return self.quantity * self.pricer.price(self.contract)
 
    def delta(self):
        """
        Position delta, central finite difference on S0.
 
        delta_per_contract = (price(S0+dS) - price(S0-dS)) / (2*dS)
        position_delta     = quantity * delta_per_contract
        """
        S0 = self.contract.S0
        dS = S0 * self._DELTA_BUMP
        price_up = self.pricer.price(_clone_contract_with(self.contract, S0=S0 + dS))
        price_dn = self.pricer.price(_clone_contract_with(self.contract, S0=S0 - dS))
        delta_unit = (price_up - price_dn) / (2.0 * dS)
        return self.quantity * delta_unit
 
    def gamma(self):
        """
        Position gamma, central finite difference on S0 (second derivative).
 
        gamma_per_contract = (price(S0+dS) - 2*price(S0) + price(S0-dS)) / dS^2
        position_gamma     = quantity * gamma_per_contract
        """
        S0 = self.contract.S0
        dS = S0 * self._GAMMA_BUMP
        p_up = self.pricer.price(_clone_contract_with(self.contract, S0=S0 + dS))
        p_0  = self.pricer.price(self.contract)
        p_dn = self.pricer.price(_clone_contract_with(self.contract, S0=S0 - dS))
        gamma_unit = (p_up - 2.0 * p_0 + p_dn) / (dS ** 2)
        return self.quantity * gamma_unit
 
    def vega(self):
        """
        Position vega, central finite difference on volatility.
 
        Reported per 1.00 unit of σ (i.e. per +100% vol change). For a
        per-1%-vol interpretation, divide by 100.
        """
        sigma = self.contract.sigma
        dv = self._VEGA_BUMP
        p_up = self.pricer.price(_clone_contract_with(self.contract, sigma=sigma + dv))
        p_dn = self.pricer.price(_clone_contract_with(self.contract, sigma=sigma - dv))
        vega_unit = (p_up - p_dn) / (2.0 * dv)
        return self.quantity * vega_unit
 
    def theta(self):
        """
        Position theta, one-sided finite difference on time to maturity.

        Reported as the value change over one trading day (1/252 of a year):
            theta_per_contract = price(T - dt) - price(T)
            position_theta     = quantity * theta_per_contract

        A one-sided (backward) difference is used rather than central because
        time only moves forward — we cannot price a contract with more time
        than it has. For a long-optionality position theta is negative: the
        position loses value as maturity approaches, all else equal.
        """
        T = self.contract.T
        dt = self._THETA_DT
        # Guard: never bump past expiry (would give T <= 0 and break pricers)
        if T <= dt:
            return 0.0
        p_now  = self.pricer.price(self.contract)
        p_next = self.pricer.price(_clone_contract_with(self.contract, T=T - dt))
        theta_unit = p_next - p_now
        return self.quantity * theta_unit
 
    def delta_dollars(self):
        """
        Delta-dollars = position's first-order $ sensitivity to a 100%
        proportional move in its underlying.
 
        For an option: delta_dollars = position_delta * S0.
        (position_delta is already quantity-weighted by self.delta().)
        """
        return self.delta() * self.contract.S0
 
    def reprice(self, new_spot=None, new_yield_curve=None):
        """Return a new OptionPosition with shocked inputs."""
        new_S0 = new_spot if new_spot is not None else self.contract.S0
        new_yc = new_yield_curve if new_yield_curve is not None else self.contract.yield_curve
 
        return OptionPosition(
            contract=_clone_contract_with(self.contract, S0=new_S0, yield_curve=new_yc),
            pricer=self.pricer,
            quantity=self.quantity,
            underlying_ticker=self.underlying_ticker,
            label=self.label,
        )
 
    def __repr__(self):
        return (
            f"OptionPosition(label={self.label!r}, "
            f"quantity={self.quantity}, pricer={self.pricer.name!r})"
        )


# ======================================================================
# Portfolio
# ======================================================================

class Portfolio:
    """
    A collection of equity and option positions.

    Parameters
    ----------
    positions : list
        List of EquityPosition and/or OptionPosition objects.

    Methods
        value() / total_value()      -> float : total mark-to-market value
        delta() / portfolio_delta()  -> float : aggregate position delta
        portfolio_gamma()            -> float : aggregate position gamma
        portfolio_vega()             -> float : aggregate position vega
        delta_dollars()              -> float : total delta-dollar exposure
                                                (slope for linear P&L approx)
        add_position()                       : add a position
        position_table()             -> DataFrame : per-position breakdown
                                                    (Value, Delta, Gamma, Vega)
        summary_table()              -> DataFrame : alias for position_table()
    """

    def __init__(self, positions=None):
        self.positions = []
        if positions:
            for p in positions:
                self.add_position(p, quantity=p.quantity,
                                  label=getattr(p, 'label', None))
 
    def add_position(self, instrument, quantity=None, label=None):
        """
        Add a position to the portfolio
        """
        # If instrument is already a position object, quantity lives on it
        if quantity is None:
            quantity = getattr(instrument, 'quantity', 1.0)
        self.positions.append({
            "instrument": instrument,
            "quantity":   float(quantity),
            "label":      label or getattr(instrument, 'label', None),
        })

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
 
    def value(self):
        """Total mark-to-market portfolio value."""
        return sum(p["instrument"].value() for p in self.positions)
 
    def total_value(self):
        """Alias for value()."""
        return self.value()
 
    def delta(self):
        """Aggregate portfolio delta (sum of position deltas)."""
        return sum(p["instrument"].delta() for p in self.positions)
 
    def portfolio_delta(self):
        """Alias for delta()."""
        return self.delta()
 
    def portfolio_gamma(self):
        """Aggregate portfolio gamma (sum of position gammas)."""
        return sum(p["instrument"].gamma() for p in self.positions)
 
    def portfolio_vega(self):
        """Aggregate portfolio vega (sum of position vegas)."""
        return sum(p["instrument"].vega() for p in self.positions)
 
    def portfolio_theta(self):
        """Aggregate portfolio theta (sum of position thetas), per trading day."""
        return sum(p["instrument"].theta() for p in self.positions)
 
    def delta_dollars(self):
        """
        Aggregate delta-dollar exposure.
 
        This is the *slope* of the portfolio's linear P&L approximation
        under a uniform proportional spot shock. For a uniform shock x:
            PnL_linear ≈ delta_dollars * x
        """
        return sum(p["instrument"].delta_dollars() for p in self.positions)
 
    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
 
    def position_table(self):
        """
        Per-position breakdown including all first- and second-order Greeks.
 
        Columns:
            Position, Quantity,
            Unit Value, Position Value,
            Unit Delta, Position Delta,
            Unit Gamma, Position Gamma,
            Unit Vega,  Position Vega
 
        A TOTAL row is appended summing Position Value, Position Delta,
        Position Gamma, and Position Vega.
        """
        rows = []
        for position in self.positions:
            instrument = position["instrument"]
            quantity   = position["quantity"]
 
            if position["label"] is not None:
                name = position["label"]
            elif hasattr(instrument, "ticker"):
                name = instrument.ticker
            else:
                name = instrument.__class__.__name__
 
            unit_value = instrument.price()
            # Unit Greeks = position Greek / quantity (Greeks here are
            # position-level by convention; we divide back out for "per-unit").
            position_delta = instrument.delta()
            position_gamma = instrument.gamma()
            position_vega  = instrument.vega()
 
            unit_delta = position_delta / quantity if quantity else 0.0
            unit_gamma = position_gamma / quantity if quantity else 0.0
            unit_vega  = position_vega  / quantity if quantity else 0.0
 
            rows.append({
                "Position":       name,
                "Quantity":       quantity,
                "Unit Value":     unit_value,
                "Position Value": quantity * unit_value,
                "Unit Delta":     unit_delta,
                "Position Delta": position_delta,
                "Unit Gamma":     unit_gamma,
                "Position Gamma": position_gamma,
                "Unit Vega":      unit_vega,
                "Position Vega":  position_vega,
            })
 
        df = pd.DataFrame(rows)
        if len(df) > 0:
            total_row = pd.DataFrame([{
                "Position":       "TOTAL",
                "Quantity":       np.nan,
                "Unit Value":     np.nan,
                "Position Value": df["Position Value"].sum(),
                "Unit Delta":     np.nan,
                "Position Delta": df["Position Delta"].sum(),
                "Unit Gamma":     np.nan,
                "Position Gamma": df["Position Gamma"].sum(),
                "Unit Vega":      np.nan,
                "Position Vega":  df["Position Vega"].sum(),
            }])
            df = pd.concat([df, total_row], ignore_index=True)
 
        return df
 
    def summary_table(self):
        """Alias for position_table() — used by the trading desk notebook."""
        return self.position_table()
 
 
# ======================================================================
# Risk Engine
# ======================================================================
 
class RiskEngine:
    """
    Portfolio risk metrics: historical VaR, parametric VaR, scenario analysis,
    and delta-linearisation vs full-revaluation P&L curves.
 
    Parameters
    ----------
    portfolio : Portfolio
    returns_df : pd.DataFrame
        Historical log-returns indexed by date, with one column per ticker
        present in the portfolio.
    """
 
    def __init__(self, portfolio, returns_df):
        self.portfolio = portfolio
        self.returns_df = returns_df.copy()
 
    # ------------------------------------------------------------------
    # VaR methods
    # ------------------------------------------------------------------
 
    def historical_var(self, confidence=0.95, horizon=1):
        """
        Historical simulation VaR.
        Returns VaR in dollars at the given confidence level and horizon.
        """
        port_returns = self._portfolio_returns()
        scaled_returns = port_returns * np.sqrt(horizon)
        portfolio_value = self.portfolio.total_value()
        var = -np.quantile(scaled_returns, 1 - confidence) * abs(portfolio_value)
        return float(var)
 
    def parametric_var(self, confidence=0.95, horizon=1):
        """
        Parametric (delta-normal) VaR. Assumes normally distributed returns.
        """
        port_returns = self._portfolio_returns()
        sigma = port_returns.std()
        z = norm.ppf(confidence)
        portfolio_value = self.portfolio.total_value()
        var = z * sigma * np.sqrt(horizon) * abs(portfolio_value)
        return float(var)
 
    def var_summary(self, confidence=0.95, horizon=1):
        """DataFrame comparing historical and parametric VaR."""
        hist     = self.historical_var(confidence, horizon)
        para     = self.parametric_var(confidence, horizon)
        port_val = self.portfolio.value()
        return pd.DataFrame({
            "Method":         ["Historical", "Parametric"],
            "Confidence":     [f"{confidence:.0%}", f"{confidence:.0%}"],
            "Horizon (days)": [horizon, horizon],
            "VaR ($)":        [round(hist, 2), round(para, 2)],
            "VaR (% NAV)":    [f"{hist/port_val:.2%}", f"{para/port_val:.2%}"],
        })
 
    # ------------------------------------------------------------------
    # Scenario analysis
    # ------------------------------------------------------------------
 
    def scenario_analysis(self, spot_shocks=None, rate_shocks=None):
        """
        Reprice under a grid of spot and rate shocks.
        Returns DataFrame of P&L relative to base value (full revaluation).
        """
        if spot_shocks is None:
            spot_shocks = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
        if rate_shocks is None:
            rate_shocks = [-0.01, -0.005, 0.0, 0.005, 0.01]
 
        base_value = self.portfolio.total_value()
        results = {}
 
        for rate_shock in rate_shocks:
            col_pnl = []
            for spot_shock in spot_shocks:
                shocked_value = self._reprice_portfolio(spot_shock, rate_shock)
                pnl = shocked_value - base_value
                col_pnl.append(round(pnl, 2))
            col_label = f"Rate {rate_shock * 10000:+.0f}bps"
            results[col_label] = col_pnl
 
        index_labels = [f"Spot {s * 100:+.0f}%" for s in spot_shocks]
        return pd.DataFrame(results, index=index_labels)
 
    def scenario_pnl_table(self, spot_shocks=None, rate_shocks=None):
        """
        Return both a dollar P&L table and a percentage P&L table.
        """
        base_value = self.portfolio.total_value()
        dollar_pnl = self.scenario_analysis(spot_shocks, rate_shocks)
        pct_pnl = (dollar_pnl / abs(base_value) * 100).round(2)
        pct_pnl = pct_pnl.map(lambda x: f"{x:+.2f}%")
        return dollar_pnl, pct_pnl
 
    # ------------------------------------------------------------------
    # Delta-linearisation vs full revaluation
    # ------------------------------------------------------------------
 
    def delta_linearisation_curve(self, spot_moves=None):
        """
        Compare delta-linear vs full-revaluation P&L across a range of
        uniform proportional spot moves.
 
        Parameters
        ----------
        spot_moves : array-like, optional
            Array of proportional spot moves (e.g. -0.20 to +0.20).
            Defaults to 81 points spanning ±20%.
 
        Returns
        -------
        pd.DataFrame with columns:
            spot_move    : the proportional shock (-0.20 ... +0.20)
            delta_pnl    : linear approximation, slope = portfolio.delta_dollars()
            full_pnl     : full revaluation P&L under the same shock
            convexity_gap: full_pnl - delta_pnl (the curvature term)
        """
        if spot_moves is None:
            spot_moves = np.linspace(-0.20, 0.20, 81)
        spot_moves = np.asarray(spot_moves, dtype=float)
 
        base_value = self.portfolio.total_value()
        slope      = self.portfolio.delta_dollars()
 
        delta_pnl = slope * spot_moves
        full_pnl = np.array([
            self._reprice_portfolio(spot_shock=float(x), rate_shock=0.0) - base_value
            for x in spot_moves
        ])
 
        return pd.DataFrame({
            "spot_move":     spot_moves,
            "delta_pnl":     delta_pnl,
            "full_pnl":      full_pnl,
            "convexity_gap": full_pnl - delta_pnl,
        })
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _portfolio_returns(self):
        total_value  = self.portfolio.value()
        port_returns = pd.Series(np.zeros(len(self.returns_df)),
                                index=self.returns_df.index)
 
        for position in self.portfolio.positions:
            instrument = position["instrument"]
            quantity   = position["quantity"]
 
            if isinstance(instrument, EquityPosition):
                ticker = instrument.ticker
                if ticker not in self.returns_df.columns:
                    raise ValueError(f"Ticker {ticker!r} not in returns_df. "
                                    f"Available: {list(self.returns_df.columns)}")
                weight = (quantity * instrument.price()) / total_value
                port_returns += weight * self.returns_df[ticker]
 
            elif isinstance(instrument, OptionPosition):
                ticker = instrument.underlying_ticker
                if ticker not in self.returns_df.columns:
                    raise ValueError(f"Ticker {ticker!r} not in returns_df. "
                                    f"Available: {list(self.returns_df.columns)}")
                # instrument.delta() already includes quantity
                dollar_delta = instrument.delta() * instrument.contract.S0
                weight       = dollar_delta / total_value
                port_returns += weight * self.returns_df[ticker]
 
        return port_returns
 
    def _reprice_portfolio(self, spot_shock, rate_shock):
        total = 0.0
        for position in self.portfolio.positions:
            instrument = position["instrument"]
            quantity   = position["quantity"]
 
            if isinstance(instrument, EquityPosition):
                new_spot = instrument.spot_price * (1.0 + spot_shock)
                total   += quantity * instrument.reprice(new_spot=new_spot).price()
 
            elif isinstance(instrument, OptionPosition):
                new_spot = instrument.contract.S0 * (1.0 + spot_shock)
                new_yc   = _shift_yield_curve(instrument.contract.yield_curve, rate_shock)
                total   += quantity * instrument.reprice(new_spot=new_spot,
                                                        new_yield_curve=new_yc).price()
        return total
 
 
# ======================================================================
# Internal utility functions
# ======================================================================
 
def _clone_contract_with(contract, **overrides):
    """
    Return a new contract of the same type as `contract`, with any
    parameters in `overrides` replaced.
 
    Used by OptionPosition.delta()/gamma()/vega()/reprice() to create
    shocked versions of a contract without modifying the original.
 
    Carries forward base Derivative parameters plus any contract-specific
    parameters (e.g. barrier, barrier_type for BarrierCall/BarrierPut)
    so that subclasses with extended __init__ signatures clone correctly.
    """
    # Base parameters every Derivative subclass accepts
    params = {
        "S0":          contract.S0,
        "K":           contract.K,
        "T":           contract.T,
        "sigma":       contract.sigma,
        "yield_curve": contract.yield_curve,
    }
    # Barrier-specific parameters, present only on BarrierCall/BarrierPut
    if hasattr(contract, "barrier"):
        params["barrier"]      = contract.barrier
        params["barrier_type"] = contract.barrier_type
    params.update(overrides)
    return type(contract)(**params)
 
 
def _shift_yield_curve(yield_curve, shift):
    """New YieldCurve with all zero rates shifted by `shift` (decimal)."""
    from src.yieldcurve import YieldCurve
    shifted_rates = yield_curve.zero_rates + shift
    return YieldCurve(
        maturities=yield_curve.maturities.copy(),
        zero_rates=shifted_rates,
        compounding=yield_curve.compounding,
    )

def build_desk_portfolio(spot, sigma, yc, bs, bt, mc):
    """
    Construct the trading desk's book and return an assembled Portfolio.

    Parameters
    ----------
    spot  : dict  Ticker -> latest spot price  e.g. {"NAB.AX": 38.50, ...}
    sigma : dict  Ticker -> annualised vol      e.g. {"NAB.AX": 0.21, ...}
    yc    : YieldCurve
    bs    : BlackScholesPricer
    bt    : BinomialTreePricer
    mc    : MonteCarloPricer

    Returns
    -------
    Portfolio
    """
    from src.derivatives import EuropeanCall, EuropeanPut, AmericanPut, BarrierPut

    nab_equity = EquityPosition(ticker="NAB.AX", quantity=200, spot_price=spot["NAB.AX"])
    bhp_equity = EquityPosition(ticker="BHP.AX", quantity=150, spot_price=spot["BHP.AX"])

    nab_protective_put = OptionPosition(
        contract=AmericanPut(
            S0=spot["NAB.AX"], K=spot["NAB.AX"] * 0.95,
            T=0.5, sigma=sigma["NAB.AX"], yield_curve=yc,
        ),
        pricer=bt, quantity=3, underlying_ticker="NAB.AX",
        label="NAB American Put 6m 5%-OTM (protective, ASX-style)",
    )

    csl_call = OptionPosition(
        contract=EuropeanCall(
            S0=spot["CSL.AX"], K=spot["CSL.AX"],
            T=0.25, sigma=sigma["CSL.AX"], yield_curve=yc,
        ),
        pricer=bs, quantity=5, underlying_ticker="CSL.AX",
        label="CSL Call 3m ATM (tactical convexity)",
    )

    wow_straddle_call = OptionPosition(
        contract=EuropeanCall(
            S0=spot["WOW.AX"], K=spot["WOW.AX"],
            T=0.25, sigma=sigma["WOW.AX"], yield_curve=yc,
        ),
        pricer=bs, quantity=4, underlying_ticker="WOW.AX",
        label="WOW Call 3m ATM (straddle leg)",
    )

    wow_straddle_put = OptionPosition(
        contract=EuropeanPut(
            S0=spot["WOW.AX"], K=spot["WOW.AX"],
            T=0.25, sigma=sigma["WOW.AX"], yield_curve=yc,
        ),
        pricer=bs, quantity=4, underlying_ticker="WOW.AX",
        label="WOW Put 3m ATM (straddle leg)",
    )

    bhp_barrier_put = OptionPosition(
        contract=BarrierPut(
            S0=spot["BHP.AX"], K=spot["BHP.AX"] * 0.95,
            T=1.0, sigma=sigma["BHP.AX"], yield_curve=yc,
            barrier=spot["BHP.AX"] * 0.80,
            barrier_type="down-and-in",
        ),
        pricer=mc, quantity=50, underlying_ticker="BHP.AX",
        label="BHP Barrier Put 1y K=95%, B=80% (DI tail hedge)",
    )

    return Portfolio([
        nab_equity,
        bhp_equity,
        nab_protective_put,
        csl_call,
        wow_straddle_call,
        wow_straddle_put,
        bhp_barrier_put,
    ])

def plot_delta_linearisation(portfolio, engine, ax=None):
    """
    Figure 5.1 — Delta linearisation vs full revaluation P&L curve.

    Parameters
    ----------
    portfolio : Portfolio
    engine    : RiskEngine
    ax        : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    import matplotlib.ticker as mticker

    curve = engine.delta_linearisation_curve(spot_moves=np.linspace(-0.20, 0.20, 81))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = None

    ax.plot(curve["spot_move"] * 100, curve["full_pnl"],
            color="steelblue", linewidth=2, label="Full revaluation")
    ax.plot(curve["spot_move"] * 100, curve["delta_pnl"],
            color="darkorange", linewidth=2, linestyle="--", label="Delta linearisation")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Uniform Spot Move (%)")
    ax.set_ylabel("Portfolio P&L ($)")
    ax.set_title("Figure 5.1 — Delta Linearisation vs Full Revaluation")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    if own_fig:
        plt.tight_layout()
        plt.show()

    return fig, ax

def build_var_table(engine):
    """
    Compute Historical, Parametric, and Monte Carlo VaR across confidence
    levels and horizons, returning a summary DataFrame (Table 6.1).

    Parameters
    ----------
    engine : RiskEngine

    Returns
    -------
    var_table                                     : pd.DataFrame
    (hist_95, hist_99, para_95, mc_95, es_95)     : tuple of key scalar metrics
    """
    def _bootstrap_var_ci(returns, confidence, horizon, n_resamples=1000, seed=42):
        rng = np.random.default_rng(seed)
        n = len(returns)
        boot_vars = np.empty(n_resamples)
        for i in range(n_resamples):
            sample = rng.choice(returns, size=n, replace=True)
            boot_vars[i] = -np.quantile(sample * np.sqrt(horizon), 1 - confidence)
        return np.quantile(boot_vars, 0.025), np.quantile(boot_vars, 0.975)

    def _monte_carlo_var(confidence, horizon, n_paths=20_000, seed=42):
        rng = np.random.default_rng(seed)
        port_rets = engine._portfolio_returns()
        mu, sigma = port_rets.mean(), port_rets.std()
        pnl = rng.normal(mu, sigma, n_paths) * np.sqrt(horizon) * abs(engine.portfolio.total_value())
        return float(-np.quantile(pnl, 1 - confidence))

    def _expected_shortfall(confidence, horizon):
        port_rets = engine._portfolio_returns() * np.sqrt(horizon)
        pnl = port_rets * abs(engine.portfolio.total_value())
        return float(-pnl[pnl <= np.quantile(pnl, 1 - confidence)].mean())

    n_obs     = len(engine.returns_df)
    pv        = engine.portfolio.total_value()
    port_rets = engine._portfolio_returns()
    rows      = []

    for confidence in [0.95, 0.99]:
        for horizon in [1, 10]:
            hist = engine.historical_var(confidence=confidence, horizon=horizon)
            para = engine.parametric_var(confidence=confidence, horizon=horizon)
            mc   = _monte_carlo_var(confidence, horizon)
            es   = _expected_shortfall(confidence, horizon)
            ci_low, ci_high = _bootstrap_var_ci(port_rets.values, confidence, horizon)
            ci_low_d, ci_high_d = ci_low * abs(pv), ci_high * abs(pv)

            rows.append({"Method": "Historical",  "Alpha": f"{int(confidence*100)}%",
                         "Horizon (days)": horizon, "VaR ($)": round(hist, 2),
                         "ES ($)": round(es, 2), "CI Low ($)": round(ci_low_d, 2),
                         "CI High ($)": round(ci_high_d, 2), "n": n_obs})
            rows.append({"Method": "Parametric",  "Alpha": f"{int(confidence*100)}%",
                         "Horizon (days)": horizon, "VaR ($)": round(para, 2),
                         "ES ($)": "-", "CI Low ($)": "-", "CI High ($)": "-", "n": n_obs})
            rows.append({"Method": "Monte Carlo", "Alpha": f"{int(confidence*100)}%",
                         "Horizon (days)": horizon, "VaR ($)": round(mc, 2),
                         "ES ($)": "-", "CI Low ($)": "-", "CI High ($)": "-", "n": 20_000})

    key_metrics = (
        engine.historical_var(confidence=0.95, horizon=1),
        engine.historical_var(confidence=0.99, horizon=1),
        engine.parametric_var(confidence=0.95, horizon=1),
        _monte_carlo_var(0.95, 1),
        _expected_shortfall(0.95, 1),
    )

    return pd.DataFrame(rows), key_metrics

def plot_pnl_distribution(engine, hist_var_95, hist_var_99, para_var_95, ax=None):
    """
    Figure 6.1 — Portfolio P&L distribution with VaR lines and Gaussian overlay.

    Parameters
    ----------
    engine      : RiskEngine
    hist_var_95 : float   Historical VaR at 95% (dollar)
    hist_var_99 : float   Historical VaR at 99% (dollar)
    para_var_95 : float   Parametric VaR at 95% (dollar)
    ax          : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    from scipy.stats import norm
    import matplotlib.ticker as mticker

    port_rets = engine._portfolio_returns()
    daily_pnl = port_rets * abs(engine.portfolio.total_value())

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = None

    counts, edges, _ = ax.hist(daily_pnl, bins=60, color="steelblue", alpha=0.6,
                               edgecolor="white", label="Empirical daily P&L")

    mu, sigma  = daily_pnl.mean(), daily_pnl.std()
    bin_width  = edges[1] - edges[0]
    x          = np.linspace(daily_pnl.min(), daily_pnl.max(), 400)
    normal_curve = norm.pdf(x, mu, sigma) * len(daily_pnl) * bin_width
    ax.plot(x, normal_curve, color="black", linewidth=2,
            label="Fitted Normal (parametric assumption)")

    ax.axvline(-hist_var_95, color="darkorange", linewidth=2, linestyle="--",
               label=f"Hist VaR 95% = ${hist_var_95:,.0f}")
    ax.axvline(-hist_var_99, color="red",        linewidth=2, linestyle="--",
               label=f"Hist VaR 99% = ${hist_var_99:,.0f}")
    ax.axvline(-para_var_95, color="purple",     linewidth=2, linestyle=":",
               label=f"Param VaR 95% = ${para_var_95:,.0f}")

    ax.set_xlabel("Daily P&L ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Figure 6.1 — Portfolio P&L Distribution: Empirical vs Gaussian")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return fig, ax

def build_scenario_table(engine):
    """
    Table 7.1 — Named stress scenarios with full revaluation.

    Parameters
    ----------
    engine : RiskEngine

    Returns
    -------
    pd.DataFrame
    """
    named_scenarios = [
        {"Name": "Base Case",               "Spot Shock":  0.00, "Rate Shock (bps)":   0},
        {"Name": "Bull Market (+10%)",      "Spot Shock":  0.10, "Rate Shock (bps)":   0},
        {"Name": "Bear Market (-10%)",      "Spot Shock": -0.10, "Rate Shock (bps)":   0},
        {"Name": "Crash (-20%)",            "Spot Shock": -0.20, "Rate Shock (bps)":   0},
        {"Name": "Rate Hike +100bps",       "Spot Shock":  0.00, "Rate Shock (bps)": 100},
        {"Name": "Rate Cut -100bps",        "Spot Shock":  0.00, "Rate Shock (bps)":-100},
        {"Name": "Stagflation (-5%,+50bps)","Spot Shock": -0.05, "Rate Shock (bps)":  50},
        {"Name": "Risk-off (-15%,-50bps)",  "Spot Shock": -0.15, "Rate Shock (bps)": -50},
    ]

    base_val = engine.portfolio.total_value()
    rows = []
    for sc in named_scenarios:
        new_val = engine._reprice_portfolio(
            spot_shock=sc["Spot Shock"],
            rate_shock=sc["Rate Shock (bps)"] / 10_000,
        )
        pnl = new_val - base_val
        rows.append({
            "Scenario":            sc["Name"],
            "Spot Shock":          f"{sc['Spot Shock']:+.0%}",
            "Rate Shock (bps)":    f"{sc['Rate Shock (bps)']:+d}",
            "Portfolio Value ($)": round(new_val, 2),
            "P&L ($)":             round(pnl, 2),
            "P&L (%)":             f"{pnl/abs(base_val):+.2%}",
        })

    return pd.DataFrame(rows)

def plot_pnl_surface(engine, ax=None):
    """
    Figure 7.1 — Portfolio P&L surface (full revaluation) across a
    spot x rate shock grid.

    Parameters
    ----------
    engine : RiskEngine
    ax     : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    spot_shocks = np.linspace(-0.30, 0.30, 25)
    rate_shocks = np.linspace(-0.02, 0.02, 9)
    base_val    = engine.portfolio.total_value()

    pnl_matrix = np.zeros((len(spot_shocks), len(rate_shocks)))
    for i, ss in enumerate(spot_shocks):
        for j, rs in enumerate(rate_shocks):
            pnl_matrix[i, j] = engine._reprice_portfolio(ss, rs) - base_val

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = None

    X, Y   = np.meshgrid(rate_shocks * 10_000, spot_shocks * 100)
    levels = np.linspace(pnl_matrix.min(), pnl_matrix.max(), 20)
    cf     = ax.contourf(X, Y, pnl_matrix, levels=levels, cmap="RdYlGn")
    cs     = ax.contour(X, Y, pnl_matrix, levels=[0], colors="black",
                        linewidths=1.5, linestyles="--")
    plt.colorbar(cf, ax=ax, label="P&L ($)")
    ax.clabel(cs, fmt="Break-even", fontsize=9)
    ax.set_xlabel("Rate Shock (bps)")
    ax.set_ylabel("Spot Shock (%)")
    ax.set_title("Figure 7.1 — Portfolio P&L Surface (Full Revaluation)")

    if own_fig:
        plt.tight_layout()
        plt.show()

    return fig, ax

def build_risk_dashboard(portfolio, engine, mc_v95, es_95, hist_var_95, para_var_95):
    """
    Table 9.1 — Head Desk Risk Dashboard.

    Parameters
    ----------
    portfolio   : Portfolio
    engine      : RiskEngine
    mc_v95      : float   Monte Carlo VaR 95% 1d (from build_var_table)
    es_95       : float   Expected Shortfall 95% 1d (from build_var_table)
    hist_var_95 : float   Historical VaR 95% 1d (from build_var_table)
    para_var_95 : float   Parametric VaR 95% 1d (from build_var_table)

    Returns
    -------
    pd.DataFrame
    """
    rows = [
        ("Portfolio Value",            f"${portfolio.total_value():,.2f}"),
        ("Portfolio Delta",            f"{portfolio.portfolio_delta():.4f}"),
        ("Portfolio Gamma",            f"{portfolio.portfolio_gamma():.4f}"),
        ("Portfolio Vega",             f"{portfolio.portfolio_vega():.4f}"),
        ("Portfolio Theta (per day)",  f"{portfolio.portfolio_theta():.4f}"),
        ("Historical VaR 95% 1d",     f"${hist_var_95:,.2f}"),
        ("Parametric VaR 95% 1d",     f"${para_var_95:,.2f}"),
        ("Monte Carlo VaR 95% 1d",    f"${mc_v95:,.2f}"),
        ("Expected Shortfall 95% 1d", f"${es_95:,.2f}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])

def build_dollar_delta_table(portfolio):
    """
    Table 9.2 — Dollar delta exposure aggregated by underlier.

    Parameters
    ----------
    portfolio : Portfolio

    Returns
    -------
    pd.DataFrame
    """
    dollar_delta = {}
    for p in portfolio.positions:
        inst   = p["instrument"]
        ticker = inst.ticker if isinstance(inst, EquityPosition) else inst.underlying_ticker
        dollar_delta[ticker] = dollar_delta.get(ticker, 0.0) + inst.delta_dollars()

    return pd.DataFrame(
        [(k, f"${v:,.2f}") for k, v in dollar_delta.items()],
        columns=["Underlier", "Dollar Delta ($)"],
    )