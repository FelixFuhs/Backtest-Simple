import pandas as pd
import numpy as np # numpy is implicitly used by pandas, but good for np.nan if needed explicitly

def sma_crossover_signal(price: pd.Series, short: int = 50, long: int = 200) -> pd.Series:
    """Return trading position (+1 long / 0 flat) based on SMA crossover.

    - price: Series of daily adjusted close prices (already cleaned)
    - Compute short and long SMAs
    - If SMA_short > SMA_long → position = 1 (long)
    - Else → position = 0
    - Shift signal by 1 day to avoid look-ahead bias (executes next day)
    - Return a Series indexed like price, with values 0 or 1
    """

    if not isinstance(price, pd.Series):
        raise TypeError("Input 'price' must be a pandas Series.")
    if price.empty:
        return pd.Series(dtype=int) # Return empty series of int type if input is empty

    if short <= 0 or long <= 0:
        raise ValueError("SMA window periods 'short' and 'long' must be positive integers.")
    if short >= long:
        # This is not strictly an error, but a warning might be useful for typical crossover interpretation.
        # However, the function will still compute based on the numbers given.
        # For now, no warning as per "clean implementation" note.
        pass

    # Calculate short and long Simple Moving Averages
    # .rolling().mean() will produce NaNs for initial periods where window is not filled
    sma_short = price.rolling(window=short).mean()
    sma_long = price.rolling(window=long).mean()

    # Initialize positions to 0 (flat)
    # The index will be the same as the input price Series
    position = pd.Series(0, index=price.index)

    # Set position to 1 (long) where short SMA is greater than long SMA
    # Pandas handles comparisons involving NaNs by evaluating them to False.
    # So, during initial periods where sma_short or sma_long are NaN,
    # the condition (sma_short > sma_long) will be False, and position remains 0.
    position[sma_short > sma_long] = 1

    # Shift the signal by 1 day to avoid look-ahead bias
    # This means the decision for day D is based on data up to D-1's close.
    # The shift introduces a NaN at the beginning of the series.
    final_signal = position.shift(1)

    # Fill the NaN introduced by the shift (at the first data point) with 0 (flat)
    # Also, ensure the output Series contains only integers (0 or 1).
    final_signal = final_signal.fillna(0).astype(int)

    return final_signal
