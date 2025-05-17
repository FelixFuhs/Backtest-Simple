import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from typing import Optional # For older Python versions if | syntax is an issue; assuming Python 3.10+ for |

# Assuming BacktestResult is in src/backtester.py and src is importable
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
    # trades from BacktestResult is positions.diff().fillna(0).astype(int)
    # It has the same length as nav.
    trades = bt.trades 
    
    metrics = {}

    # Initialize all metrics to NaN. They will be updated if calculable.
    metrics['CAGR (%)'] = np.nan
    metrics['Sharpe Ratio'] = np.nan
    metrics['Max Drawdown (%)'] = np.nan
    metrics['Win Rate (%)'] = np.nan
    metrics['Turnover (%)'] = np.nan

    # If NAV is empty or too short for meaningful calculations, return NaNs
    if nav.empty or len(nav) < 2:
        # Turnover might be 0 if nav is short but trades exist (e.g. len(nav)=1, trades=[0])
        # If trades is also empty or its length is 0, turnover remains NaN.
        if not trades.empty and len(trades) > 0: # trades should have same len as nav
             num_trade_events = trades.abs().sum()
             metrics['Turnover (%)'] = (num_trade_events / len(trades)) * 100
        else: # handles trades.empty or len(trades) == 0
            metrics['Turnover (%)'] = np.nan # Explicitly NaN if trades cannot yield turnover
        return pd.Series(metrics, name="Performance Metrics")

    # --- CAGR ---
    start_date = nav.index[0]
    end_date = nav.index[-1]
    num_years = (end_date - start_date).days / 365.25
    
    start_nav = nav.iloc[0]
    end_nav = nav.iloc[-1]

    if pd.notna(start_nav) and pd.notna(end_nav) and start_nav != 0:
        if num_years > 1e-6: # Check for a meaningful period to annualize
            cagr_val = (end_nav / start_nav)**(1 / num_years) - 1
            metrics['CAGR (%)'] = cagr_val * 100
        # else: CAGR remains np.nan if period is too short for annualization

    # --- Sharpe Ratio ---
    log_rets = np.log(nav / nav.shift(1)).dropna()
    if len(log_rets) >= 2: # Need at least 2 log returns for std()
        mean_log_ret = log_rets.mean()
        std_log_ret = log_rets.std()
        if std_log_ret > 1e-9: # Avoid division by zero if returns are flat
            metrics['Sharpe Ratio'] = (np.sqrt(252) * mean_log_ret) / std_log_ret
        elif mean_log_ret == 0 and std_log_ret <= 1e-9: # Perfectly flat returns
            metrics['Sharpe Ratio'] = 0.0
        # else: Sharpe remains np.nan (e.g., if mean_log_ret != 0 but std_log_ret is ~0)
    # else: Sharpe remains np.nan if not enough returns

    # --- Max Drawdown ---
    # Ensure nav is not all NaNs and contains valid numbers for cummax
    if not nav.isna().all() and nav.count() > 0: # count() gives number of non-NA observations
        cumulative_max_nav = nav.cummax()
        # Filter out periods where cumulative_max_nav might be zero or NaN if NAV had issues
        # (though Backtester NAV starts at 1.0 and should be positive)
        valid_mask = (cumulative_max_nav > 1e-9) & pd.notna(cumulative_max_nav) & pd.notna(nav)
        if valid_mask.any():
            drawdown = pd.Series(np.nan, index=nav.index) # Initialize with NaNs
            drawdown[valid_mask] = nav[valid_mask] / cumulative_max_nav[valid_mask] - 1
            max_drawdown_val = drawdown.min() # min() ignores NaNs by default
            if pd.notna(max_drawdown_val):
                metrics['Max Drawdown (%)'] = abs(max_drawdown_val) * 100
    # else: Max Drawdown remains np.nan

    # --- Win Rate ---
    daily_arith_rets = nav.pct_change().dropna()
    if not daily_arith_rets.empty:
        positive_return_days = (daily_arith_rets > 0).sum()
        total_return_days = len(daily_arith_rets)
        if total_return_days > 0:
            metrics['Win Rate (%)'] = (positive_return_days / total_return_days) * 100
    # else: Win Rate remains np.nan

    # --- Turnover ---
    # trades series has same length as nav and is positions.diff().fillna(0).astype(int)
    # len(trades) is the total number of days in the backtest period.
    if not trades.empty and len(trades) > 0: # This condition should hold if nav was not empty
        num_trade_events = trades.abs().sum() # Counts buy (1) or sell (-1) days
        metrics['Turnover (%)'] = (num_trade_events / len(trades)) * 100
    # else: Turnover remains np.nan (already covered by initial check if nav is empty)
    
    return pd.Series(metrics, name="Performance Metrics")


def plot_equity(bt: BacktestResult, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot equity curve over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if bt.equity_curve.empty:
        ax.text(0.5, 0.5, "No equity data to plot", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Equity Curve")
        return ax

    bt.equity_curve.plot(ax=ax, legend=False)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, linestyle='--', alpha=0.7)
    return ax


def plot_drawdown(bt: BacktestResult, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot drawdowns from peak."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    nav = bt.equity_curve
    if nav.empty or nav.count() < 1: # nav.count() for number of non-NA values
        ax.text(0.5, 0.5, "No drawdown data to plot", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Drawdown from Peak")
        return ax

    cumulative_max_nav = nav.cummax()
    drawdown_series = pd.Series(index=nav.index, dtype=float)
    
    # Calculate drawdown only where cumulative_max_nav is positive and not NaN
    valid_mask = (cumulative_max_nav > 1e-9) & pd.notna(cumulative_max_nav) & pd.notna(nav)
    
    if valid_mask.any():
        drawdown_series[valid_mask] = nav[valid_mask] / cumulative_max_nav[valid_mask] - 1
        # Fill non-valid parts with 0 (no drawdown) or leave as NaN if preferred
        drawdown_series.fillna(0, inplace=True) 

    drawdown_series.plot(ax=ax, kind='area', color='red', alpha=0.3, legend=False)
    ax.plot(drawdown_series.index, drawdown_series, color='red', lw=0.5) # Add line for clarity
    
    ax.axhline(0, color='grey', linestyle='--')
    ax.set_title("Drawdown from Peak")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, linestyle='--', alpha=0.7)
    return ax