import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_loader import DataLoader
from src.signals import KalmanSignals
from src.monte_carlo_utils import (
    get_strategy_results, random_entry_benchmark,
    bootstrap_confidence, path_shuffle,
    plot_random_entry, plot_bootstrap,
    plot_path_shuffle, plot_summary_dashboard
)


def main():
    print("=" * 70)
    print("  MONTE CARLO — COMBINED OUT-OF-SAMPLE (2018-2022)")
    print("  Val + Test pooled (never tuned on either)")
    print("=" * 70)

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    loader = DataLoader()
    df = loader.load_spy_data()
    train, val, test = loader.split_data(df)

    combined = pd.concat([val, test])
    prices = combined["close"]

    print(f"\n  Combined OOS: {len(combined)} bars")
    print(f"  {combined.index[0]} to {combined.index[-1]}")

    kalman_signals = KalmanSignals().generate(prices)
    trades, trade_rets, trade_weights, eq, pos, strat_bar_rets = get_strategy_results(prices, kalman_signals)

    actual_sharpe = eq["sharpe"]
    actual_return = eq["total_return"]
    actual_wr = eq["win_rate"]
    n_trades = len(trades)
    bh = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

    print(f"\n  Strategy: {n_trades} trades | {actual_return:.2f}% return | {actual_sharpe:.2f} Sharpe | {actual_wr:.1f}% WR")
    print(f"  Buy & Hold: {bh:.2f}%")

    pname = "Combined OOS (2018-2022)"

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
    os.rename(f"{fig_dir}/mc_random_entry.png", f"{fig_dir}/mc_random_combined.png")

    plot_bootstrap(boot_rets, boot_wrs, boot_pfs, actual_return, actual_wr, fig_dir)
    os.rename(f"{fig_dir}/mc_bootstrap.png", f"{fig_dir}/mc_bootstrap_combined.png")

    plot_path_shuffle(max_dds, min_eqs, eq_paths, final_eq, fig_dir=fig_dir)
    os.rename(f"{fig_dir}/mc_path_shuffle.png", f"{fig_dir}/mc_paths_combined.png")

    plot_summary_dashboard(
        actual_sharpe, actual_return, actual_wr, n_trades,
        percentile, ci_return, ci_wr, ci_pf,
        prob_pos_boot, median_dd, ci_dd,
        median_min, final_eq, pname, fig_dir)
    os.rename(f"{fig_dir}/mc_summary.png", f"{fig_dir}/mc_summary_combined.png")

    p_val = (100 - percentile) / 100
    print(f"\n{'=' * 70}")
    print(f"  COMBINED OOS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Trades:          {n_trades}")
    print(f"  Return:          {actual_return:.2f}% (B&H: {bh:.2f}%)")
    print(f"  Sharpe:          {actual_sharpe:.2f}")
    print(f"  vs Random:       {percentile:.1f}th percentile (p = {p_val:.4f})")
    print(f"  P(positive):     {prob_pos_boot:.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
