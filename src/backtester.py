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
        if not isinstance(prices, pd.Series) or not isinstance(positions, pd.Series):
            raise TypeError("prices and positions must be pandas Series.")
        if rf is not None and not isinstance(rf, pd.Series):
            raise TypeError("rf must be a pandas Series if provided.")

        self.original_prices = prices.copy()
        self.original_positions = positions.copy()
        self.original_rf = rf.copy() if rf is not None else None
        self.cost_bps = cost_bps

        self._align_data()

    def _align_data(self):
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
        if self.prices.empty:
            empty_idx = pd.DatetimeIndex([])
            return BacktestResult(
                equity_curve=pd.Series(dtype=float, index=empty_idx),
                positions=pd.Series(dtype=int, index=empty_idx),
                trades=pd.Series(dtype=int, index=empty_idx)
            )

        asset_returns = self.prices.pct_change(fill_method=None) # Address FutureWarning

        trades_ts = self.positions.diff() 

        one_way_cost_rate = (self.cost_bps / 2.0) / 10000.0 
        transaction_costs_pct = abs(trades_ts) * one_way_cost_rate
        transaction_costs_pct = transaction_costs_pct.fillna(0.0)

        active_asset_returns = self.positions * asset_returns

        rf_component_returns = pd.Series(0.0, index=self.prices.index)
        if self.rf is not None:
            daily_rf_rate = (self.rf / 100.0) / 365.0
            rf_component_returns = (1 - self.positions) * daily_rf_rate
        
        gross_daily_returns = active_asset_returns.fillna(0.0) + rf_component_returns.fillna(0.0)
        net_daily_returns = gross_daily_returns - transaction_costs_pct

        equity_curve = pd.Series(index=self.prices.index, dtype=float)
        if not self.prices.empty:
            equity_curve.iloc[0] = 1.0
            for t in range(1, len(self.prices)):
                equity_curve.iloc[t] = equity_curve.iloc[t-1] * (1 + net_daily_returns.iloc[t])
            equity_curve = equity_curve.ffill() # Address FutureWarning
        
        final_trades = trades_ts.fillna(0).astype(int)
        final_positions = self.positions.astype(int)

        return BacktestResult(
            equity_curve=equity_curve,
            positions=final_positions,
            trades=final_trades
        )