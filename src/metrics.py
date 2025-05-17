import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from typing import Optional 

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

    metrics['CAGR (%)'] = np.nan
    metrics['Sharpe Ratio'] = np.nan
    metrics['Max Drawdown (%)'] = np.nan
    metrics['Win Rate (%)'] = np.nan
    metrics['Turnover (%)'] = np.nan

    if nav.empty or len(nav) < 2:
        if not trades.empty and len(trades) > 0:
             num_trade_events = trades.abs().sum()
             metrics['Turnover (%)'] = (num_trade_events / len(trades)) * 100
        else:
            metrics['Turnover (%)'] = np.nan
        return pd.Series(metrics, name="Performance Metrics")

    # --- CAGR ---
    start_date = nav.index[0]
    end_date = nav.index[-1]
    num_years = (end_date - start_date).days / 365.25
    
    start_nav = nav.iloc[0]
    end_nav = nav.iloc[-1]

    if pd.notna(start_nav) and pd.notna(end_nav) and start_nav != 0:
        if num_years > 1e-6: 
            cagr_val = (end_nav / start_nav)**(1 / num_years) - 1
            metrics['CAGR (%)'] = cagr_val * 100

    # --- Sharpe Ratio ---
    log_rets = np.log(nav / nav.shift(1)).dropna()
    if len(log_rets) >= 2: 
        mean_log_ret = log_rets.mean()
        std_log_ret = log_rets.std()
        if std_log_ret > 1e-9: 
            metrics['Sharpe Ratio'] = (np.sqrt(252) * mean_log_ret) / std_log_ret
        elif abs(mean_log_ret) < 1e-9 and abs(std_log_ret) < 1e-9 : # Essentially zero returns and zero vol
            metrics['Sharpe Ratio'] = 0.0
    
    # --- Max Drawdown ---
    if not nav.isna().all() and nav.count() > 0:
        cumulative_max_nav = nav.cummax()
        valid_mask = (cumulative_max_nav > 1e-9) & pd.notna(cumulative_max_nav) & pd.notna(nav)
        if valid_mask.any():
            drawdown = pd.Series(np.nan, index=nav.index) 
            drawdown[valid_mask] = nav[valid_mask] / cumulative_max_nav[valid_mask] - 1
            max_drawdown_val = drawdown.min() 
            if pd.notna(max_drawdown_val):
                metrics['Max Drawdown (%)'] = abs(max_drawdown_val) * 100

    # --- Win Rate ---
    daily_arith_rets = nav.pct_change().dropna() # Using fill_method=None by default in modern pandas
    if not daily_arith_rets.empty:
        positive_return_days = (daily_arith_rets > 0).sum()
        total_return_days = len(daily_arith_rets)
        if total_return_days > 0:
            metrics['Win Rate (%)'] = (positive_return_days / total_return_days) * 100

    # --- Turnover ---
    if not trades.empty and len(trades) > 0: 
        num_trade_events = trades.abs().sum() 
        metrics['Turnover (%)'] = (num_trade_events / len(trades)) * 100
    
    return pd.Series(metrics, name="Performance Metrics")


def plot_equity(bt: BacktestResult, ax: Optional[plt.Axes] = None) -> plt.Axes:
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
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    nav = bt.equity_curve
    if nav.empty or nav.count() < 1: 
        ax.text(0.5, 0.5, "No drawdown data to plot", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Drawdown from Peak")
        return ax

    cumulative_max_nav = nav.cummax()
    drawdown_series = pd.Series(index=nav.index, dtype=float)
    
    valid_mask = (cumulative_max_nav > 1e-9) & pd.notna(cumulative_max_nav) & pd.notna(nav)
    
    if valid_mask.any():
        drawdown_series[valid_mask] = nav[valid_mask] / cumulative_max_nav[valid_mask] - 1
        drawdown_series.fillna(0, inplace=True) 

    drawdown_series.plot(ax=ax, kind='area', color='red', alpha=0.3, legend=False)
    ax.plot(drawdown_series.index, drawdown_series, color='red', lw=0.5) 
    
    ax.axhline(0, color='grey', linestyle='--')
    ax.set_title("Drawdown from Peak")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, linestyle='--', alpha=0.7)
    return ax