import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.signals import KalmanSignals
from src.monte_carlo_utils import (
    get_strategy_results, random_entry_benchmark,
    bootstrap_confidence, path_shuffle,
    plot_random_entry, plot_bootstrap,
    plot_path_shuffle, plot_summary_dashboard
)


def load_bloomberg(filepath):
    df = pd.read_csv(filepath, skiprows=3)
    df.columns = ["datetime", "open", "close", "high", "low"]
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
    df = df.dropna(subset=["datetime", "close"])
    df = df.set_index("datetime")
    df = df[["open", "high", "low", "close"]]
    df = df.sort_index()

    last_bar = df.index.map(lambda x: x.time())
    df = df[last_bar != pd.Timestamp("16:00:00").time()]

    print(f"  Loaded {len(df)} bars")
    print(f"  From: {df.index[0]}")
    print(f"  To:   {df.index[-1]}")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df


def main():
    print("=" * 70)
    print("  BLOOMBERG OUT-OF-SAMPLE VALIDATION (2025-2026)")
    print("  Data the strategy has NEVER seen")
    print("  3+ years after training period ended")
    print("=" * 70)

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
    os.makedirs(fig_dir, exist_ok=True)

    df = load_bloomberg(os.path.join(data_dir, "SPY_DATA_2025-2026.csv"))
    prices = df["close"]

    bh = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    print(f"  Buy & Hold: {bh:.2f}%\n")

    kalman_signals = KalmanSignals().generate(prices)
    trades, trade_rets, trade_weights, eq, pos, strat_bar_rets = get_strategy_results(prices, kalman_signals)

    actual_sharpe = eq["sharpe"]
    actual_return = eq["total_return"]
    actual_wr = eq["win_rate"]
    n_trades = len(trades)

    print(f"\n  Strategy Results:")
    print(f"    Trades:     {n_trades}")
    print(f"    Return:     {actual_return:.2f}%")
    print(f"    Sharpe:     {actual_sharpe:.2f}")
    print(f"    Win Rate:   {actual_wr:.1f}%")
    print(f"    Max DD:     {eq['max_drawdown']:.2f}%")

    if n_trades < 5:
        print(f"\n  WARNING: Only {n_trades} trades. Not enough for Monte Carlo.")
        return

    pname = "Bloomberg OOS (2025-2026)"

    print(f"\n  Running Monte Carlo...")
    rand_sharpes, percentile = random_entry_benchmark(prices, trades, actual_sharpe, trade_weights)
    boot_rets, boot_wrs, boot_pfs, prob_pos_boot = bootstrap_confidence(trade_rets, trade_weights)
    max_dds, min_eqs, eq_paths, final_eq = path_shuffle(trade_rets, trade_weights)

    ci_return = np.percentile(boot_rets, [2.5, 97.5])
    ci_wr = np.percentile(boot_wrs, [2.5, 97.5])
    ci_pf = np.percentile(boot_pfs, [2.5, 97.5])
    ci_dd = np.percentile(max_dds, [5, 95])
    median_dd = np.median(max_dds)
    median_min = np.median(min_eqs)

    print(f"\n  Generating charts...")

    plot_random_entry(rand_sharpes, actual_sharpe, percentile, pname, fig_dir)
    os.rename(f"{fig_dir}/mc_random_entry.png", f"{fig_dir}/mc_random_bloomberg.png")

    plot_bootstrap(boot_rets, boot_wrs, boot_pfs, actual_return, actual_wr, fig_dir)
    os.rename(f"{fig_dir}/mc_bootstrap.png", f"{fig_dir}/mc_bootstrap_bloomberg.png")

    plot_path_shuffle(max_dds, min_eqs, eq_paths, final_eq, fig_dir=fig_dir)
    os.rename(f"{fig_dir}/mc_path_shuffle.png", f"{fig_dir}/mc_paths_bloomberg.png")

    plot_summary_dashboard(
        actual_sharpe, actual_return, actual_wr, n_trades,
        percentile, ci_return, ci_wr, ci_pf,
        prob_pos_boot, median_dd, ci_dd,
        median_min, final_eq, pname, fig_dir)
    os.rename(f"{fig_dir}/mc_summary.png", f"{fig_dir}/mc_summary_bloomberg.png")

    p_val = (100 - percentile) / 100
    print(f"\n{'=' * 70}")
    print(f"  BLOOMBERG OOS RESULTS")
    print(f"{'=' * 70}")
    print(f"  Period:          Aug 2025 - Feb 2026")
    print(f"  Gap from train:  3+ years")
    print(f"  Trades:          {n_trades}")
    print(f"  Return:          {actual_return:.2f}% (B&H: {bh:.2f}%)")
    print(f"  Sharpe:          {actual_sharpe:.2f}")
    print(f"  Win Rate:        {actual_wr:.1f}%")
    print(f"  vs Random:       {percentile:.1f}th percentile (p = {p_val:.4f})")
    print(f"  P(positive):     {prob_pos_boot:.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
