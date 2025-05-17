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
            self.rf = None # Ensure self.rf is explicitly None if not used

        # It's assumed strategy.py provides clean 0/1 integer positions.
        # Add a clip/round for robustness if inputs could be noisy, though not specified.
        # self.positions = self.positions.clip(0, 1).round().astype(int)


    def run(self) -> BacktestResult:
        """Run backtest, return equity curve, trades, and final positions."""
        if self.prices.empty:
            # Handle case with no overlapping data
            empty_idx = pd.DatetimeIndex([])
            return BacktestResult(
                equity_curve=pd.Series(dtype=float, index=empty_idx),
                positions=pd.Series(dtype=int, index=empty_idx),
                trades=pd.Series(dtype=int, index=empty_idx)
            )

        # 1. Calculate price returns (first element will be NaN)
        asset_returns = self.prices.pct_change()

        # 2. Determine trades (+1 for buy, -1 for sell, 0 for hold)
        # positions.diff() gives NaN for the first element.
        trades_ts = self.positions.diff()

        # 3. Calculate transaction cost percentage
        # Assuming cost_bps is round-trip, so one-way cost is cost_bps / 2.0
        one_way_cost_rate = (self.cost_bps / 2.0) / 10000.0 
        # abs(trades_ts) is 1 on trade, 0 on hold. NaN for first element.
        transaction_costs_pct = abs(trades_ts) * one_way_cost_rate
        # No transaction cost for the very first day's "non-trade" (where trades_ts is NaN)
        transaction_costs_pct = transaction_costs_pct.fillna(0.0)

        # 4. Calculate daily gross returns from asset and risk-free rate
        # self.positions[t] is the position held during day t, earning asset_returns[t]
        # (strategy.py ensures self.positions is already shifted for this purpose)
        active_asset_returns = self.positions * asset_returns

        # Risk-free returns for idle capital (when position is 0)
        rf_component_returns = pd.Series(0.0, index=self.prices.index)
        if self.rf is not None:
            # Assuming self.rf contains annualized percentages (e.g., 3.0 for 3%)
            # De-annualize to daily rate. Using 365 for calendar days.
            daily_rf_rate = (self.rf / 100.0) / 365.0
            # Apply RF return only when position is 0
            rf_component_returns = (1 - self.positions) * daily_rf_rate
        
        # Combined gross daily returns
        # .fillna(0.0) for asset_returns.iloc[0] which is NaN and for any NaNs in rf_component_returns
        gross_daily_returns = active_asset_returns.fillna(0.0) + rf_component_returns.fillna(0.0)

        # 5. Calculate net daily returns (after transaction costs)
        net_daily_returns = gross_daily_returns - transaction_costs_pct

        # 6. Calculate equity curve (NAV)
        equity_curve = pd.Series(index=self.prices.index, dtype=float)
        if not self.prices.empty:
            equity_curve.iloc[0] = 1.0
            for t in range(1, len(self.prices)):
                equity_curve.iloc[t] = equity_curve.iloc[t-1] * (1 + net_daily_returns.iloc[t])
            # In case of any all-NaN slices leading to issues, ffill NAV.
            # Given fillna(0) on returns components, this should mostly handle stable NAV during no-activity.
            equity_curve = equity_curve.fillna(method='ffill') 
        else: # Should be caught by the initial self.prices.empty check
             equity_curve = pd.Series(dtype=float)


        # Prepare BacktestResult outputs
        # trades_ts has NaN for first row, fill with 0 and cast to int
        final_trades = trades_ts.fillna(0).astype(int)

        # self.positions is already aligned and should be int (0 or 1) from strategy module
        final_positions = self.positions.astype(int)

        return BacktestResult(
            equity_curve=equity_curve,
            positions=final_positions,
            trades=final_trades
        )
