import numpy as np
import matplotlib.pyplot as plt

"""
data.py
-------
Data ingestion functions for the platform.

Design principle:
    Data sources are passed in as paths/parameters, not hard-coded
    inside class definitions. This keeps pricing logic decoupled
    from data sourcing, and lets the system "drop in" new data
    files without code changes elsewhere.

Currently implemented:
    load_rba_yield_curve   - parse RBA F2 CSV into yield curve inputs
    validate_yield_data    - sanity-check yield curve data before use
    list_available_dates   - inspect what historical dates the file has

Equity price ingestion (yfinance) will be added in a later iteration
once portfolio underlyings are decided.
"""

import numpy as np
import pandas as pd


# Maturity (in years) of each RBA nominal Government Bond series.
# The indexed-bond column is intentionally excluded -- it is a real-yield
# (inflation-linked) curve, not a nominal one, and the pricing engine
# expects nominal zero rates.
_RBA_NOMINAL_MATURITIES = {
    "Australian Government 2 year bond": 2.0,
    "Australian Government 3 year bond": 3.0,
    "Australian Government 5 year bond": 5.0,
    "Australian Government 10 year bond": 10.0,
}

# Row index (0-based) where the dated time series begins in the RBA F2 CSV.
# Rows 0-10 are metadata (title, description, frequency, type, units,
# blank rows, source, publication date, series IDs). The actual data
# starts at row 11 of the CSV.
_RBA_METADATA_ROWS = 11


def load_rba_yield_curve(path, date=None):
    """
    Load a nominal Australian Government bond yield curve from the
    RBA F2 daily CSV.

    The RBA file has 11 metadata rows at the top, then daily observations
    of bond yields at fixed maturities (2y, 3y, 5y, 10y) plus an indexed
    (real-yield) series which we exclude. Yields are quoted in per cent
    per annum; this function converts them to decimals.

    Parameters
    ----------
    path : str
        Path to the RBA F2 CSV file.
    date : str, pandas.Timestamp, or None
        Snapshot date to extract.
            - None (default): the most recent date with complete data
            - "YYYY-MM-DD" string or Timestamp: that specific date
        If the requested date is missing or has any blank maturity,
        a ValueError is raised.

    Returns
    -------
    maturities : np.ndarray
        Maturities in years, sorted ascending. e.g. [2.0, 3.0, 5.0, 10.0]
    zero_rates : np.ndarray
        Decimal yields aligned to `maturities`. e.g. [0.047, 0.047, ...]
    snapshot_date : pandas.Timestamp
        The actual date the data was sourced from.

    Examples
    --------
    >>> maturities, rates, date = load_rba_yield_curve("data/raw/f2-data.csv")
    >>> yc = YieldCurve(maturities, rates)
    """
    df = _read_rba_csv(path)

    # Restrict to nominal bond columns only, in maturity order
    nominal_cols = [c for c in _RBA_NOMINAL_MATURITIES if c in df.columns]
    if not nominal_cols:
        raise ValueError(
            f"None of the expected RBA nominal bond columns were found in "
            f"{path}. Got columns: {list(df.columns)}"
        )
    df = df[nominal_cols].copy()

    # Pick the snapshot row
    if date is None:
        # Most recent date with all maturities populated
        valid_rows = df.dropna(how="any")
        if valid_rows.empty:
            raise ValueError(
                "No date in the file has data for every nominal maturity."
            )
        snapshot_date = valid_rows.index[-1]
        row = valid_rows.iloc[-1]
    else:
        snapshot_date = pd.Timestamp(date)
        if snapshot_date not in df.index:
            raise ValueError(
                f"Date {snapshot_date.date()} not found in the RBA file. "
                f"Available range: {df.index.min().date()} to {df.index.max().date()}."
            )
        row = df.loc[snapshot_date]
        if row.isna().any():
            missing = row[row.isna()].index.tolist()
            raise ValueError(
                f"Date {snapshot_date.date()} is missing values for: {missing}"
            )

    # Build aligned arrays of maturities and (decimal) zero rates
    maturities = np.array(
        [_RBA_NOMINAL_MATURITIES[c] for c in nominal_cols], dtype=float
    )
    zero_rates = np.array(row.values, dtype=float) / 100.0   # per-cent -> decimal

    # Sort by maturity (RBA columns are already in order, but be defensive)
    order = np.argsort(maturities)
    maturities = maturities[order]
    zero_rates = zero_rates[order]

    return maturities, zero_rates, snapshot_date


def validate_yield_data(maturities, zero_rates,
                        min_rate=-0.02, max_rate=0.25,
                        max_jump=0.05):
    """
    Sanity-check a yield curve before passing it into the pricing engine.

    Raises ValueError if any of the following are violated:
        - Arrays must be 1D and the same length, with at least 2 points
        - No NaN or infinite values
        - Maturities strictly increasing and positive
        - Rates within plausible bounds [min_rate, max_rate]
        - Adjacent maturities do not jump by more than `max_jump` in rate
          (catches data-entry errors and bad rows)

    Parameters
    ----------
    maturities : array-like
        Maturities in years.
    zero_rates : array-like
        Decimal zero rates (NOT percent).
    min_rate, max_rate : float
        Lower and upper plausible bounds for any single rate. Defaults
        permit very low negative rates (post-COVID Europe) but reject
        rates above 25% (would indicate per-cent vs decimal confusion).
    max_jump : float
        Maximum absolute change in rate between adjacent maturities.
        Default 5% catches gross outliers without flagging normal curve shape.

    Returns
    -------
    None
        Function is silent on success; raises on failure.
    """
    maturities = np.asarray(maturities, dtype=float)
    zero_rates = np.asarray(zero_rates, dtype=float)

    if maturities.ndim != 1 or zero_rates.ndim != 1:
        raise ValueError("maturities and zero_rates must be 1D arrays")

    if maturities.shape != zero_rates.shape:
        raise ValueError(
            f"Length mismatch: {len(maturities)} maturities vs {len(zero_rates)} rates"
        )

    if len(maturities) < 2:
        raise ValueError("Need at least 2 points to define a yield curve")

    if np.any(~np.isfinite(maturities)) or np.any(~np.isfinite(zero_rates)):
        raise ValueError("NaN or infinite values present in yield curve data")

    if np.any(maturities <= 0):
        raise ValueError("All maturities must be strictly positive")

    if not np.all(np.diff(maturities) > 0):
        raise ValueError("Maturities must be strictly increasing")

    if np.any(zero_rates < min_rate) or np.any(zero_rates > max_rate):
        bad = zero_rates[(zero_rates < min_rate) | (zero_rates > max_rate)]
        raise ValueError(
            f"Zero rate(s) outside plausible range [{min_rate:.0%}, {max_rate:.0%}]: "
            f"{bad}. Did you forget to convert per-cent to decimal?"
        )

    jumps = np.abs(np.diff(zero_rates))
    if np.any(jumps > max_jump):
        raise ValueError(
            f"Adjacent rates jump by more than {max_jump:.0%} "
            f"(max observed: {jumps.max():.2%}). This usually indicates "
            f"data entry error or a corrupted row."
        )


def list_available_dates(path):
    """
    Return all dates that appear in the RBA F2 CSV, sorted ascending.

    Useful for sanity-checking what historical snapshots are available
    before requesting one with `date=...` in load_rba_yield_curve().
    """
    df = _read_rba_csv(path)
    return df.index.sort_values()


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _read_rba_csv(path):
    """
    Parse the RBA F2 CSV into a DataFrame indexed by date.

    Handles:
        - BOM character at the start of the file (utf-8-sig encoding)
        - 11 rows of metadata above the data
        - Date format DD-Mon-YYYY (e.g. 20-May-2013)
        - Trailing empty rows at the end of the file
        - Blank (NaN) values in early rows for short maturities

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex of observation dates
        Columns: one per bond series (e.g. "Australian Government 2 year bond")
        Values: float yields in per-cent-per-annum (not yet converted to decimal)
    """
    # Row 1 of the CSV (after skipping the title) holds the column headers
    # under "Title,...". We need those as our column names.
    headers = pd.read_csv(
        path, encoding="utf-8-sig", skiprows=1, nrows=1, header=None
    ).iloc[0].tolist()
    # Column 0 is the "Title" label; columns 1+ are the bond series names
    column_names = ["Date"] + headers[1:]

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        skiprows=_RBA_METADATA_ROWS,
        header=None,
        names=column_names,
    )

    # Drop trailing all-NaN rows
    df = df.dropna(how="all")

    # Parse the date column (DD-Mon-YYYY format)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")

    # Coerce yield columns to numeric (any non-numeric becomes NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

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

