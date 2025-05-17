from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class BacktestResult:
    equity_curve: pd.Series  # daily NAV starting at 1.0
    positions: pd.Series     # long/flat position (0 or 1)
    trades: pd.Series        # +1 (buy), -1 (sell), 0 (hold)

class Backtester:
    def __init__(self, prices: pd.Series, positions: pd.Series, rf: pd.Series | None = None, cost_bps: float = 10.0):
        """Simulate a strategy over time.

        - prices: daily adjusted close prices
        - positions: Series of {0,1} long/flat signals (aligned to prices)
        - rf: optional risk-free rate series (annualized percentages, e.g., 3.0 for 3%)
        - cost_bps: round-trip trading cost in basis points (default 10bps)
        """
        if not isinstance(prices, pd.Series) or not isinstance(positions, pd.Series):
            raise TypeError("prices and positions must be pandas Series.")
        if rf is not None and not isinstance(rf, pd.Series):
            raise TypeError("rf must be a pandas Series if provided.")

        self.original_prices = prices.copy()
        self.original_positions = positions.copy()
        self.original_rf = rf.copy() if rf is not None else None
        self.cost_bps = cost_bps

        # Align data to common index
        self._align_data()

    def _align_data(self):
        """Aligns prices, positions, and rf to their common intersecting index."""
        common_idx = self.original_prices.index.intersection(self.original_positions.index)
        if self.original_rf is not None:
            common_idx = common_idx.intersection(self.original_rf.index)
        
        self.prices = self.original_prices.loc[common_idx].copy()
        self.positions = self.original_positions.loc[common_idx].copy()
        if self.original_rf is not None:
            self.rf = self.original_rf.loc[common_idx].copy()
        else:
            self.rf = None 

    def run(self) -> BacktestResult:
        """Run backtest, return equity curve, trades, and final positions."""
        if self.prices.empty:
            empty_idx = pd.DatetimeIndex([])
            return BacktestResult(
                equity_curve=pd.Series(dtype=float, index=empty_idx),
                positions=pd.Series(dtype=int, index=empty_idx),
                trades=pd.Series(dtype=int, index=empty_idx)
            )

        # 1. Calculate price returns (first element will be NaN)
        # Explicitly set fill_method=None to adopt modern behavior and silence warning about 'pad'
        asset_returns = self.prices.pct_change(fill_method=None)

        # 2. Determine trades (+1 for buy, -1 for sell, 0 for hold)
        trades_ts = self.positions.diff() # First element is NaN

        # 3. Calculate transaction cost percentage
        one_way_cost_rate = (self.cost_bps / 2.0) / 10000.0 
        transaction_costs_pct = abs(trades_ts) * one_way_cost_rate
        transaction_costs_pct = transaction_costs_pct.fillna(0.0)

        # 4. Calculate daily gross returns from asset and risk-free rate
        active_asset_returns = self.positions * asset_returns

        rf_component_returns = pd.Series(0.0, index=self.prices.index)
        if self.rf is not None:
            daily_rf_rate = (self.rf / 100.0) / 365.0
            rf_component_returns = (1 - self.positions) * daily_rf_rate
        
        gross_daily_returns = active_asset_returns.fillna(0.0) + rf_component_returns.fillna(0.0)

        # 5. Calculate net daily returns (after transaction costs)
        net_daily_returns = gross_daily_returns - transaction_costs_pct

        # 6. Calculate equity curve (NAV)
        equity_curve = pd.Series(index=self.prices.index, dtype=float)
        if not self.prices.empty: # Should always be true if we passed the initial empty check
            equity_curve.iloc[0] = 1.0
            for t in range(1, len(self.prices)):
                equity_curve.iloc[t] = equity_curve.iloc[t-1] * (1 + net_daily_returns.iloc[t])
            
            # Use .ffill() instead of .fillna(method='ffill')
            equity_curve = equity_curve.ffill() 
        
        final_trades = trades_ts.fillna(0).astype(int)
        final_positions = self.positions.astype(int)

        return BacktestResult(
            equity_curve=equity_curve,
            positions=final_positions,
            trades=final_trades
        )