import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


"""
src/yieldcurve.py
-----------------
Yield curve construction and data ingestion.
 
Public API:
    load_rba_yield_curve  -- parse RBA F17 CSV → (maturities, zero_rates, date)
    validate_yield_data   -- sanity-check arrays before use
    list_available_dates  -- available dates in the CSV
    spot_check_table      -- zero rates + discount factors at key maturities
    plot_yield_curve      -- zero-rate + discount factor chart
    YieldCurve            -- interpolating yield curve class
"""

# Row index (0-based) where the dated time series begins in the RBA F17 CSV.
# Rows 0-10 are metadata (title, description, frequency, type, units,
# blank rows, source, publication date, series IDs). The actual data
# starts at row 11 of the CSV.
_RBA_METADATA_ROWS = 11


def load_rba_yield_curve (path, date=None):
    """Parse RBA F17 CSV → (maturities, zero_rates, snap_date). date=None returns most recent row."""
    df = _read_rba_csv(path)

    mat_map = {col: float(m.group(1))
               for col in df.columns
               if (m := re.search(r"([\d.]+)\s*yr", col))}
 
    if not mat_map:
        raise ValueError(f"No maturity columns found in {path}.")
 
    df = df[list(mat_map.keys())].copy()

    # Pick the snapshot row
    if date is None:
        valid = df.dropna(how="any")
        if valid.empty:
            raise ValueError("No complete row found.")
        snap_date, row = valid.index[-1], valid.iloc[-1]
    else:
        snap_date = pd.Timestamp(date)
        if snap_date not in df.index:
            raise ValueError(f"{snap_date.date()} not in file ({df.index.min().date()} – {df.index.max().date()}).")
        row = df.loc[snap_date]
        if row.isna().any():
            raise ValueError(f"{snap_date.date()} has missing values.")
        
    maturities = np.array([mat_map[c] for c in mat_map], dtype=float)
    zero_rates = np.array(row[list(mat_map.keys())].values, dtype=float) / 100.0
    order = np.argsort(maturities)
    return maturities[order], zero_rates[order], snap_date


def validate_yield_data(maturities, zero_rates,
                        min_rate=-0.05, max_rate=0.25,
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

    if np.any(maturities < 0):
        raise ValueError("Maturities must be non-negative.")

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
    """Return all dates in the CSV sorted ascending."""
    return _read_rba_csv(path).index.sort_values()


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _read_rba_csv(path):
    """
    Parse the RBA F17 CSV into a DataFrame indexed by date.

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

