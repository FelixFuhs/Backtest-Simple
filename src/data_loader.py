import pandas as pd
import yfinance as yf
from typing import List
from pathlib import Path

def get_prices(tickers: List[str], start: str, end: str, cache: bool = True) -> pd.DataFrame:
    """Download daily OHLCV from yfinance for the given tickers.

    Returns a MultiIndex DataFrame:
    - index = date (UTC, daily)
    - columns = ticker x ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    - Uses yfinance with auto_adjust=False to get 'Adj Close' (see note below)
    - If cache=True, saves data to 'data/cache/{ticker}.parquet'

    Note: The prompt specified 'auto_adjust=True' but also an 'Adj Close' column.
    To provide 'Adj Close' along with 'Close', auto_adjust=False is used.
    If auto_adjust=True were used, 'Close' would be the adjusted price,
    and 'Adj Close' column would not be available.
    """
    cache_dir = Path('data/cache')
    
    # Desired column order and selection
    ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    all_ticker_data = {}

    # Convert start/end strings to datetime objects for comparison and slicing
    # These will be timezone-naive by default, which is fine for yfinance date ranges
    # and for comparing with cached data if it's also stored naively.
    dt_start = pd.to_datetime(start)
    dt_end = pd.to_datetime(end)

    for ticker in tickers:
        ticker_df = None
        parquet_file_path = cache_dir / f"{ticker}.parquet"

        if cache and parquet_file_path.exists():
            try:
                cached_data = pd.read_parquet(parquet_file_path)
                if not isinstance(cached_data.index, pd.DatetimeIndex):
                    cached_data.index = pd.to_datetime(cached_data.index)
                
                # Ensure cached index is naive for comparison, assuming it was stored naive
                if cached_data.index.tz is not None:
                    cached_data.index = cached_data.index.tz_localize(None)

                # Check if the cached data covers the requested date range
                if not cached_data.empty and \
                   cached_data.index.min() <= dt_start and \
                   cached_data.index.max() >= dt_end:
                    
                    # Slice the cached data to the requested range
                    s_df = cached_data.loc[dt_start:dt_end].copy()
                    if not s_df.empty: # Ensure slice is not empty
                         ticker_df = s_df
                    # If slice is empty (e.g. requested range had no trading days within a larger cached range),
                    # ticker_df remains None, will trigger download to confirm empty range from source.
                    # However, yf.download on an empty range also yields empty, so this logic path is okay.
                    # Let's assign it if the slice operation itself is valid.
                    # If dt_start:dt_end is a valid slice of cached_data, even if empty, it's a "hit".
                    ticker_df = s_df # Assign slice directly

            except Exception:
                # print(f"Could not read or process cache {parquet_file_path}: {e}")
                ticker_df = None # Ensure download is triggered if cache is problematic

        if ticker_df is None: # Cache miss, or cache exists but doesn't cover the range, or failed to load
            try:
                downloaded_data = yf.download(
                    ticker, 
                    start=start, # Use original string for yf
                    end=end,     # Use original string for yf
                    auto_adjust=False, 
                    progress=False,
                    actions=False # Avoids 'Dividends' and 'Stock Splits' columns
                )
                if not downloaded_data.empty:
                    # Select/Reorder to desired columns. Fills with NaN if a column is missing.
                    ticker_df_processed = downloaded_data.reindex(columns=ohlcv_columns)
                    
                    # Filter again to ensure exact date range, as yf might sometimes include extra points
                    # if start/end are not precise trading days.
                    # Ensure index is datetime before filtering by dt_start/dt_end
                    if not isinstance(ticker_df_processed.index, pd.DatetimeIndex):
                        ticker_df_processed.index = pd.to_datetime(ticker_df_processed.index)
                    
                    ticker_df = ticker_df_processed.loc[dt_start:dt_end].copy()

                    if cache and not ticker_df.empty: # Save the newly downloaded (and correctly ranged) data
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        # Save the dataframe that corresponds to the requested start/end
                        ticker_df.to_parquet(parquet_file_path) 
                    elif cache and ticker_df.empty: # If download results in empty for range, still "cache" this empty result
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        ticker_df.to_parquet(parquet_file_path) # Cache the empty dataframe for this range
            except Exception:
                # print(f"Could not download data for {ticker}: {e}")
                # ticker_df remains None or empty
                pass # ticker_df will be None or empty, correctly skipped later
        
        if ticker_df is not None and not ticker_df.empty:
            # Ensure columns are exactly as specified and in order,
            # handling cases where yf might return fewer columns (e.g., new listings)
            all_ticker_data[ticker] = ticker_df.reindex(columns=ohlcv_columns)

    if not all_ticker_data:
        raise ValueError("No data returned for the given tickers and date range.")

    final_df = pd.concat(all_ticker_data, axis=1, names=['Ticker', 'Measure'])
    
    # Ensure the index is DatetimeIndex.
    if not isinstance(final_df.index, pd.DatetimeIndex):
        final_df.index = pd.to_datetime(final_df.index)

    # Localize to UTC as per requirement.
    # yfinance daily data index is typically timezone-naive.
    if final_df.index.tz is None:
        final_df.index = final_df.index.tz_localize('UTC')
    else:
        final_df.index = final_df.index.tz_convert('UTC')
    
    # Sort columns for consistent output: Ticker (alphabetical), then Measure (by ohlcv_columns order)
    final_df = final_df.sort_index(axis=1, level=0) 
    # The measure level will maintain its order from ohlcv_columns if concat works as expected.
    # To be absolutely sure about measure order within each ticker:
    # final_df = final_df.reindex(columns=pd.MultiIndex.from_product([final_df.columns.levels[0], ohlcv_columns], names=['Ticker', 'Measure']))
    # This reindex is more robust for column order but can be complex if tickers vary.
    # The current concat and column selection in the loop should maintain order.

    return final_df


def load_risk_free(path: str = "risk-free.csv") -> pd.Series:
    """Load a CSV with daily risk-free rates.

    - Parses the first column as datetime index (YYYY-MM-DD)
    - Renames value column to 'RF'
    - Ensures daily frequency, forward-fills missing dates
    - Returns a Series with name 'RF'
    """
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=[0])
    except FileNotFoundError:
        raise FileNotFoundError(f"Risk-free rate file not found at {Path(path).resolve()}")
    except Exception as e:
        raise ValueError(f"Error reading or parsing CSV at {path}: {e}")

    if df.empty:
        raise ValueError(f"Risk-free rate file at {path} is empty or could not be parsed correctly.")
        
    if df.shape[1] == 0:
         raise ValueError(f"Risk-free rate file at {path} has no value columns after parsing Date column.")
    
    # Take the first data column and ensure its name is 'RF'
    series = df.iloc[:, 0].copy() # Use .copy() to avoid SettingWithCopyWarning on rename
    series.name = 'RF'
    
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    # Ensure daily frequency, forward-filling missing dates
    series = series.asfreq('D', method='ffill')

    # Localize index to UTC for consistency (optional, but good practice)
    if series.index.tz is None:
        series.index = series.index.tz_localize('UTC')
    else:
        series.index = series.index.tz_convert('UTC')
        
    return series
