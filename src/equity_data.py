"""
Equity data loader for the FINM3422 A3 risk and derivatives platform.

Provides:
    Single-ticker layer
    -------------------
    * load_equity_data(ticker, start_date, end_date, cache_path):
        Load historical daily closes for an equity ticker. Caches locally
        so the system is reproducible without repeated network calls.

    * estimate_historical_volatility(prices):
        Annualised volatility computed from log returns of a price series.

    Multi-ticker layer
    ------------------
    * load_equity_panel(tickers, start_date, end_date, cache_dir):
        Load close prices for several tickers at once, returning one
        DataFrame indexed by date with one column per ticker. Caches
        each ticker as a separate CSV via load_equity_data().

    * compute_log_returns(prices_df):
        Convert a price panel into daily log returns.

    * summary_table(prices_df, returns_df):
        Per-ticker summary: spot price, annualised vol, observation count.
        This is the first of the two tables the Week 12 tutor specified.

    * correlation_matrix(returns_df):
        Pairwise correlation between tickers' daily returns.
        This is the second of the two tables the Week 12 tutor specified.

Design principles (matching yieldcurve.py):

1. **Data ingestion as a function, not a hard link.** Tickers and file
   paths are parameters. Switching to a new ticker or a new vendor only
   requires changing the function arguments or adding a new loader.

2. **Cache-first, network-second.** If `cache_path` exists, we load from
   CSV. Otherwise we fetch from Yahoo Finance via yfinance and save.
   The platform runs offline once data is cached, which guarantees
   reproducibility across team members and markers.

3. **Sanity checks before returning.** No NaN prices, no negative prices,
   ascending unique dates, returned range covers the request. Bad data
   raises a clear ValueError before it pollutes the pricing engine.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# Single-ticker layer
# =============================================================================

def load_equity_data(ticker, start_date, end_date, cache_path=None, refresh=False):
    """
    Load historical daily closing prices for an equity ticker.

    Uses a local CSV cache so the system runs offline once data is fetched.
    Default behaviour: if the cache exists, read it; otherwise fetch from
    Yahoo Finance via yfinance and write the cache. Pass `refresh=True` to
    force a fresh fetch even when a cache exists - the cache is then
    overwritten with the new data.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol, e.g. 'BHP.AX', 'CBA.AX', 'AAPL'.
    start_date, end_date : str (YYYY-MM-DD) or pandas-compatible date
        Inclusive date range to request from the data source. Note:
        Yahoo's end_date is exclusive by convention, so internally we
        add one day; the returned data is inclusive of end_date if a
        trading day.
    cache_path : str or Path, optional
        Local CSV path. If the file exists and refresh=False, read from
        it directly. If the file does not exist (or refresh=True), fetch
        from yfinance and save here. If None, no caching - always fetches
        fresh and does not save.
    refresh : bool, default False
        If True, ignore any existing cache and re-fetch from Yahoo Finance,
        overwriting the cache file. Used to bring the snapshot up to date
        with more recent market data. Default False keeps results
        reproducible across runs.

    Returns
    -------
    pd.DataFrame
        Indexed by date (DatetimeIndex), columns include at least 'Close'.

    Raises
    ------
    ValueError
        If returned data fails any sanity check (empty, NaN, negative,
        non-monotonic dates).
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    if end_date <= start_date:
        raise ValueError(
            f"end_date ({end_date.date()}) must be after start_date ({start_date.date()})"
        )

    cache_path = Path(cache_path) if cache_path is not None else None

    # ---- Try cache first (unless refresh forces a fresh fetch) ----
    if cache_path is not None and cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        source = f"cache ({cache_path})"
    else:
        # ---- Fetch from Yahoo Finance via yfinance ----
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required to fetch equity data. "
                "Install with: pip install yfinance"
            ) from e

        # yfinance treats end_date as exclusive, so add one day to make
        # the user-facing API inclusive (consistent with start_date).
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,  # adjusts for splits and dividends - one consistent series
        )

        # yfinance returns a MultiIndex when given a single ticker in
        # recent versions; flatten so 'Close' is a simple column.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path)

        source = (
            f"yfinance ({ticker}, refresh -> cache overwritten)"
            if refresh
            else f"yfinance ({ticker})"
        )

    # ---- Sanity checks ----
    _validate_price_data(df, ticker, start_date, end_date, source)

    return df


def _validate_price_data(df, ticker, start_date, end_date, source):
    """Sanity-check loaded price data; raise ValueError on any issue."""
    if df is None or len(df) == 0:
        raise ValueError(
            f"No data returned for {ticker} from {source}. "
            f"Check the ticker symbol and date range."
        )
    if "Close" not in df.columns:
        raise ValueError(
            f"'Close' column missing from {source} data for {ticker}. "
            f"Got columns: {list(df.columns)}"
        )
    if df["Close"].isna().any():
        raise ValueError(
            f"NaN values found in 'Close' column for {ticker}. "
            f"This usually indicates a corrupted cache or a data-vendor issue."
        )
    if (df["Close"] <= 0).any():
        raise ValueError(
            f"Non-positive Close prices found for {ticker} - data is corrupted."
        )
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"Dates not in ascending order for {ticker}.")
    if df.index.duplicated().any():
        raise ValueError(f"Duplicate dates in {ticker} data.")

    # Warn (not raise) if returned range is significantly shorter than requested.
    actual_days = (df.index.max() - df.index.min()).days
    requested_days = (end_date - start_date).days
    if actual_days < 0.7 * requested_days:
        print(
            f"  Warning: returned date range ({actual_days} days) is much shorter "
            f"than requested ({requested_days} days). Check ticker / dates."
        )


def estimate_historical_volatility(prices, trading_days_per_year=252):
    """
    Estimate annualised volatility from a series of daily closing prices.

    Follows the standard methodology from the supplementary reader (§5.7.1):

        r_t        = log(P_t / P_{t-1})          daily log returns
        sigma_d    = std(r_t)                    daily volatility
        sigma_ann  = sigma_d * sqrt(252)         annualised

    Parameters
    ----------
    prices : pd.Series or array-like
        Daily closing prices, in chronological order.
    trading_days_per_year : int, default 252
        Convention for annualising. 252 is standard for US/AU equities;
        260 is sometimes used for FX, 365 for crypto.

    Returns
    -------
    float
        Annualised volatility as a decimal (e.g. 0.245 = 24.5%).

    Raises
    ------
    ValueError
        If fewer than 30 returns are available, or if any return is NaN.
    """
    prices = pd.Series(prices).dropna()
    if len(prices) < 31:
        raise ValueError(
            f"Need at least 31 prices for volatility estimation, got {len(prices)}. "
            f"~30 observations is the minimum credible sample size for std."
        )
    if (prices <= 0).any():
        raise ValueError("Prices must all be positive for log-return computation.")

    log_returns = np.log(prices / prices.shift(1)).dropna()
    if log_returns.isna().any():
        raise ValueError("NaN values in computed log returns - check input prices.")

    sigma_daily = log_returns.std(ddof=1)
    sigma_annual = sigma_daily * np.sqrt(trading_days_per_year)

    return float(sigma_annual)


# =============================================================================
# Multi-ticker layer
# =============================================================================

def load_equity_panel(
    tickers,
    start_date,
    end_date,
    cache_dir=None,
    refresh=False,
):
    """
    Load close prices for several tickers and combine into one DataFrame.

    Calls load_equity_data() once per ticker (each cached separately under
    cache_dir/<TICKER>.csv), then merges the 'Close' columns into a single
    panel indexed by date.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance ticker symbols, e.g. ['AAPL', 'MSFT', 'XOM'].
    start_date, end_date : str or pandas-compatible date
        Inclusive date range, same convention as load_equity_data().
    cache_dir : str or Path, default "data/equities"
        Directory where per-ticker CSVs are cached.
    refresh : bool, default False
        If True, re-fetch all tickers and overwrite their caches.

    Returns
    -------
    pd.DataFrame
        Indexed by date, one column per ticker. Only dates where every
        ticker has data are kept (inner join), so the panel is rectangular
        and ready for log-return computation.

    Raises
    ------
    ValueError
        If `tickers` is empty, or if any ticker's load fails its sanity
        checks (delegated to load_equity_data).
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list.")

    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "data" / "equities"
    cache_dir = Path(cache_dir)

    cache_dir = Path(cache_dir)

    closes = {}
    for ticker in tickers:
        cache_path = cache_dir / f"{ticker}.csv"
        df = load_equity_data(
            ticker, start_date, end_date,
            cache_path=cache_path,
            refresh=refresh,
        )
        closes[ticker] = df["Close"]

    # Inner join on the date index so every row is fully populated.
    # This avoids accidentally feeding NaN into the return calculation.
    panel = pd.concat(closes, axis=1, join="inner")
    panel.index.name = "Date"

    if len(panel) == 0:
        raise ValueError(
            "Equity panel has zero rows after aligning ticker dates. "
            "Check that the requested date range overlaps for all tickers."
        )

    return panel


def compute_log_returns(prices_df):
    """
    Compute daily log returns from a price panel.

    r_t = log(P_t / P_{t-1})

    Log returns are used (not simple returns) because they are time-additive
    -- the sum of daily log returns equals the multi-day log return -- and
    they are the natural quantity in geometric Brownian motion, which is the
    asset-price model used by the pricing engine.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Daily closing prices, one column per ticker, indexed by date.

    Returns
    -------
    pd.DataFrame
        Daily log returns, same shape as prices_df minus the first row
        (which has no prior price to difference against).
    """
    if (prices_df <= 0).any().any():
        raise ValueError(
            "All prices must be positive to compute log returns. "
            "Got a non-positive value somewhere in prices_df."
        )

    log_returns = np.log(prices_df / prices_df.shift(1)).dropna(how="any")
    return log_returns


def summary_table(prices_df, returns_df, trading_days_per_year=252):
    """
    Per-ticker summary statistics.

    Builds the first of the two tables the Week 12 tutor specified:
    each row is a ticker, columns are spot, annualised vol, observation count.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Daily closing prices, one column per ticker. Used for spot price.
    returns_df : pd.DataFrame
        Daily log returns, one column per ticker. Used for vol and n_obs.
    trading_days_per_year : int, default 252
        Annualisation convention.

    Returns
    -------
    pd.DataFrame
        Indexed by ticker, columns ['spot', 'ann_vol', 'n_obs'].
        spot is the latest close, ann_vol is decimal (e.g. 0.245 = 24.5%),
        n_obs is the number of return observations.
    """
    rows = {}
    for ticker in prices_df.columns:
        rows[ticker] = {
            "spot": float(prices_df[ticker].iloc[-1]),
            "ann_vol": float(returns_df[ticker].std(ddof=1) * np.sqrt(trading_days_per_year)),
            "n_obs": int(returns_df[ticker].count()),
        }
    return pd.DataFrame(rows).T[["spot", "ann_vol", "n_obs"]]


def correlation_matrix(returns_df):
    """
    Pairwise correlation between tickers' daily log returns.

    This is the second of the two tables the Week 12 tutor specified.
    Used by the portfolio risk layer to model diversification and to
    construct parametric VaR (which depends on the covariance matrix).

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily log returns, one column per ticker.

    Returns
    -------
    pd.DataFrame
        Symmetric (n_tickers x n_tickers) correlation matrix, diagonal = 1.
    """
    return returns_df.corr()