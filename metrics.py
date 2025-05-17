import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from typing import Optional # For older Python versions if | syntax is an issue; assuming Python 3.10+ for |

# Assuming BacktestResult is in src/backtester.py and src is importable
# If metrics.py is in the same directory as backtester.py within a package 'src',
# then from .backtester import BacktestResult would be used.
# Sticking to the prompt's direct import style.
from src.backtester import BacktestResult


def summarize(bt: BacktestResult) -> pd.Series:
    """Compute basic strategy performance metrics.

    Returns a Series with:
    - CAGR (compound annual growth rate) in %
    - Sharpe ratio (√252, assumes risk-free = 0, uses log returns)
    - Max drawdown (positive percentage)
    - Win rate (% of days with positive arithmetic return)
    - Turnover (% of trading days with position change)
    """
    nav = bt.equity_curve
    trades = bt.trades
    metrics = {}

    # Handle cases with insufficient data for most metrics
    if nav.empty or len(nav) < 2:
        metrics['CAGR (%)'] = np.nan
        metrics['Sharpe Ratio'] = np.nan
        metrics['Max Drawdown (%)'] = np.nan
        metrics['Win Rate (%)'] = np.nan
        if trades.empty:
            metrics['Turnover (%)'] = np.nan
        else:
            # Turnover might still be calculable if trades series has entries
            num_trade_events = trades.abs().sum()
            metrics['Turnover (%)'] = (num_trade_events / len(trades)) * 100 if len(trades)
