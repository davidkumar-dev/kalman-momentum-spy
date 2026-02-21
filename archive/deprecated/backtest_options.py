import pandas as pd
import numpy as np
from typing import Dict, List
from src.options_pricing import BlackScholes

BARS_PER_DAY = 26
TRADING_DAYS_PER_YEAR = 252
ANNUALIZATION = np.sqrt(BARS_PER_DAY * TRADING_DAYS_PER_YEAR)


class OptionsBacktest:

    def __init__(self,
                 initial_capital: float = 10000,
                 days_to_expiry: int = 30,
                 num_contracts: int = 1,
                 fixed_iv: float = 0.20):
        self.initial_capital = initial_capital
        self.dte = days_to_expiry
        self.contracts = num_contracts
        self.iv = fixed_iv
        self.bs = BlackScholes()

    def _option_value(self, option_type: str, S: float, K: float, bars_held: int) -> float:
        days_elapsed = bars_held / BARS_PER_DAY
        T = max((self.dte - days_elapsed) / TRADING_DAYS_PER_YEAR, 0.001)
        if option_type == "call":
            per_share = self.bs.call(S, K, T, self.iv)
        else:
            per_share = self.bs.put(S, K, T, self.iv)
        return per_share * 100 * self.contracts

    def run(self, prices: pd.Series, trades: List[Dict]) -> Dict:
        n = len(prices)
        px = prices.values

        cash = self.initial_capital
        equity_curve = np.full(n, np.nan)

        active = None
        trade_idx = 0
        trade_results = []

        sorted_trades = sorted(trades, key=lambda t: t["entry_idx"])

        for i in range(n):

            if active is not None:
                bars_held = i - active["entry_idx"]
                mark = self._option_value(active["type"], px[i], active["strike"], bars_held)
                equity_curve[i] = cash + mark

                if i >= active["exit_idx"]:
                    exit_value = self._option_value(
                        active["type"], px[i], active["strike"], bars_held
                    )
                    pnl = exit_value - active["entry_cost"]
                    cash += exit_value

                    trade_results.append({
                        "entry_idx": active["entry_idx"],
                        "exit_idx": i,
                        "type": active["type"],
                        "strike": active["strike"],
                        "entry_cost": active["entry_cost"],
                        "exit_value": exit_value,
                        "pnl": pnl,
                        "pnl_pct": pnl / active["entry_cost"] * 100,
                    })
                    active = None
            else:
                equity_curve[i] = cash

            if active is None and trade_idx < len(sorted_trades):
                t = sorted_trades[trade_idx]
                if i == t["entry_idx"]:
                    option_type = "call" if t["direction"] == 1 else "put"
                    strike = px[i]
                    T_entry = self.dte / TRADING_DAYS_PER_YEAR
                    if option_type == "call":
                        price_per = self.bs.call(px[i], strike, T_entry, self.iv)
                    else:
                        price_per = self.bs.put(px[i], strike, T_entry, self.iv)
                    entry_cost = price_per * 100 * self.contracts

                    if entry_cost <= cash:
                        cash -= entry_cost
                        active = {
                            "type": option_type,
                            "entry_idx": t["entry_idx"],
                            "exit_idx": t["exit_idx"],
                            "strike": strike,
                            "entry_cost": entry_cost,
                        }
                        mark = self._option_value(option_type, px[i], strike, 0)
                        equity_curve[i] = cash + mark
                    trade_idx += 1

        eq = pd.Series(equity_curve, index=prices.index)
        eq.ffill(inplace=True)
        eq.bfill(inplace=True)

        ret = eq.pct_change().dropna()
        total_return = (eq.iloc[-1] / self.initial_capital - 1) * 100
        sharpe = ret.mean() / ret.std() * ANNUALIZATION if ret.std() > 0 else 0.0

        cumulative = (1 + ret).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() * 100 if len(drawdown) > 0 else 0.0

        wins = [t for t in trade_results if t["pnl"] > 0]
        losses = [t for t in trade_results if t["pnl"] <= 0]
        win_rate = len(wins) / len(trade_results) * 100 if trade_results else 0.0

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "num_trades": len(trade_results),
            "final_capital": eq.iloc[-1],
            "win_rate": win_rate,
            "equity_curve": eq,
            "trades": trade_results,
        }
