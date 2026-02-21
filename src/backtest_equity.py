import pandas as pd
import numpy as np
from typing import Dict, List


BARS_PER_DAY = 26
TRADING_DAYS_PER_YEAR = 252
ANNUALIZATION = np.sqrt(BARS_PER_DAY * TRADING_DAYS_PER_YEAR)


class EquityBacktest:

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital

    def run(self, prices: pd.Series, positions: pd.Series, trades: List[Dict]) -> Dict:
        returns = prices.pct_change().fillna(0)
        strategy_returns = positions.shift(1).fillna(0) * returns
        equity = (1 + strategy_returns).cumprod() * self.initial_capital

        total_return = (equity.iloc[-1] / self.initial_capital - 1) * 100

        sr = strategy_returns.dropna()
        sharpe = sr.mean() / sr.std() * ANNUALIZATION if sr.std() > 0 else 0.0

        cumulative = (1 + sr).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() * 100

        trade_results = []
        for t in trades:
            entry_px = prices.iloc[t["entry_idx"]]
            exit_px = prices.iloc[t["exit_idx"]]
            if t["direction"] == 1:
                pnl_pct = (exit_px / entry_px - 1) * 100
            else:
                pnl_pct = (entry_px / exit_px - 1) * 100
            trade_results.append(pnl_pct)

        wins = [r for r in trade_results if r > 0]
        losses = [r for r in trade_results if r <= 0]
        win_rate = len(wins) / len(trade_results) * 100 if trade_results else 0.0

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "num_trades": len(trades),
            "win_rate": win_rate,
            "equity_curve": equity,
        }
