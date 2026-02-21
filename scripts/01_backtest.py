import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_loader import DataLoader
from src.signals import ConnorSignals, KalmanSignals
from src.position_manager import PositionManager
from src.backtest_equity import EquityBacktest
from src.performance import print_results_table, buy_and_hold_return


def run_strategy(name, prices, positions, trades, eq_bt):
    eq = eq_bt.run(prices, positions, trades)
    return {
        "name": name,
        "total_return": eq["total_return"],
        "sharpe": eq["sharpe"],
        "max_drawdown": eq["max_drawdown"],
        "num_trades": eq["num_trades"],
        "win_rate": eq["win_rate"],
    }


def long_short_breakdown(label, trades, prices):
    longs = [t for t in trades if t["direction"] == 1]
    shorts = [t for t in trades if t["direction"] == -1]

    def trade_pnls(trade_list):
        results = []
        for t in trade_list:
            entry_px = prices.iloc[t["entry_idx"]]
            exit_px = prices.iloc[t["exit_idx"]]
            if t["direction"] == 1:
                results.append((exit_px / entry_px - 1) * 100)
            else:
                results.append((entry_px / exit_px - 1) * 100)
        return results

    long_pnls = trade_pnls(longs)
    short_pnls = trade_pnls(shorts)

    def summarize(pnls, side):
        if not pnls:
            return
        wins = [p for p in pnls if p > 0]
        total = sum(pnls)
        wr = len(wins) / len(pnls) * 100
        avg = np.mean(pnls)
        print(f"      {side}: {len(pnls)} trades | sum: {total:+.2f}% | avg: {avg:+.3f}% | WR: {wr:.1f}%")

    print(f"    {label}:")
    if long_pnls:
        summarize(long_pnls, "LONG ")
    if short_pnls:
        summarize(short_pnls, "SHORT")
    if not long_pnls and not short_pnls:
        print(f"      0 trades")


def run_period(period_name, prices, connor_signals, kalman_signals, pm, eq_bt):
    bh = buy_and_hold_return(prices)
    print(f"\n  Buy-and-hold ({period_name}): {bh:.2f}%")

    results = []
    trade_map = {}

    print(f"    Connor signals: {connor_signals['long_entry'].sum()} long, {connor_signals['short_entry'].sum()} short")
    print(f"    Kalman signals: {kalman_signals['long_entry'].sum()} long, {kalman_signals['short_entry'].sum()} short")

    pos_c, trades_c = pm.connor_exit(connor_signals, hold_period=211)
    results.append(run_strategy("Connor (actual)", prices, pos_c, trades_c, eq_bt))
    trade_map["Connor (actual)"] = trades_c

    pos_cf, trades_cf = pm.fixed_hold(connor_signals, hold_period=211)
    results.append(run_strategy("Connor (fixed hold only)", prices, pos_cf, trades_cf, eq_bt))
    trade_map["Connor (fixed hold)"] = trades_cf

    pos_conv, trades_conv = pm.adaptive_strategy(
        kalman_signals, prices,
        long_only=True, conviction_weighted=True,
    )
    results.append(run_strategy("Kalman Conviction 100%", prices, pos_conv, trades_conv, eq_bt))
    trade_map["Kalman Conviction"] = trades_conv

    results.append(run_strategy("Kalman Conviction 50%", prices, pos_conv * 0.50, trades_conv, eq_bt))

    bh_row = {
        "name": "Buy & Hold",
        "total_return": bh,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "num_trades": 0,
        "win_rate": 0.0,
    }
    results.append(bh_row)

    print_results_table(results, f"{period_name.upper()}")

    print(f"\n  Trade stats ({period_name}):")
    for label, trades in trade_map.items():
        if trades:
            holds = [t["exit_idx"] - t["entry_idx"] for t in trades]
            avg_b = np.mean(holds)
            print(f"    {label:<30} {len(trades):>4} trades | avg hold: {avg_b:.0f} bars ({avg_b/26:.1f}d)")
        else:
            print(f"    {label:<30}    0 trades")

    print(f"\n  Long/Short ({period_name}):")
    for label, trades in trade_map.items():
        long_short_breakdown(label, trades, prices)


def main():
    print("=" * 70)
    print("  CONNOR (ACTUAL) vs KALMAN CONVICTION")
    print("  Fair comparison with correct Connor implementation")
    print("=" * 70)

    loader = DataLoader()
    df = loader.load_spy_data()
    train, val, test = loader.split_data(df)

    pm = PositionManager()
    eq_bt = EquityBacktest(initial_capital=10000)

    for period_name, data in [("train", train), ("val", val), ("test", test)]:
        print("\n" + "=" * 70)
        print(f"  {period_name.upper()}")
        print("=" * 70)
        connor_sig = ConnorSignals().generate(data["close"])
        kalman_sig = KalmanSignals().generate(data["close"])
        run_period(period_name, data["close"], connor_sig, kalman_sig, pm, eq_bt)


if __name__ == "__main__":
    main()
