import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_loader import DataLoader
from src.signals import KalmanSignals
from src.monte_carlo_utils import (
    apply_style, get_strategy_results, random_entry_benchmark,
    bootstrap_confidence, path_shuffle,
    plot_random_entry, plot_bootstrap,
    plot_path_shuffle, plot_summary_dashboard
)


def main():
    print("=" * 70)
    print("  MONTE CARLO VALIDATION — PER PERIOD")
    print("=" * 70)

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    loader = DataLoader()
    df = loader.load_spy_data()
    train, val, test = loader.split_data(df)

    all_results = {}

    for period_name, data in [("Train (2014-2017)", train), ("Val (2018-2020)", val), ("Test (2021-2022)", test)]:
        print(f"\n{'=' * 70}")
        print(f"  {period_name}")
        print(f"{'=' * 70}")

        prices = data["close"]
        kalman_signals = KalmanSignals().generate(prices)

        trades, trade_rets, trade_weights, eq, pos, strat_bar_rets = get_strategy_results(prices, kalman_signals)
        actual_sharpe = eq["sharpe"]
        actual_return = eq["total_return"]
        actual_wr = eq["win_rate"]
        n_trades = len(trades)

        print(f"\n  Strategy: {n_trades} trades | {actual_return:.2f}% return | {actual_sharpe:.2f} Sharpe | {actual_wr:.1f}% WR")

        rand_sharpes, percentile = random_entry_benchmark(prices, trades, actual_sharpe, trade_weights)
        boot_rets, boot_wrs, boot_pfs, prob_pos_boot = bootstrap_confidence(trade_rets, trade_weights)
        max_dds, min_eqs, eq_paths, final_eq = path_shuffle(trade_rets, trade_weights)

        all_results[period_name] = {
            "actual_sharpe": actual_sharpe, "actual_return": actual_return,
            "actual_wr": actual_wr, "n_trades": n_trades, "percentile": percentile,
            "ci_return": np.percentile(boot_rets, [2.5, 97.5]),
            "ci_wr": np.percentile(boot_wrs, [2.5, 97.5]),
            "ci_pf": np.percentile(boot_pfs, [2.5, 97.5]),
            "prob_pos_boot": prob_pos_boot,
            "median_dd": np.median(max_dds),
            "ci_dd": np.percentile(max_dds, [5, 95]),
            "median_min": np.median(min_eqs),
            "final_eq": final_eq,
            "rand_sharpes": rand_sharpes, "boot_rets": boot_rets,
            "boot_wrs": boot_wrs, "boot_pfs": boot_pfs,
            "max_dds": max_dds, "min_eqs": min_eqs, "eq_paths": eq_paths,
        }

    print(f"\n{'=' * 70}")
    print(f"  GENERATING CHARTS")
    print(f"{'=' * 70}")

    for pname, r in all_results.items():
        tag = pname.split("(")[0].strip().lower()
        print(f"\n  Charts for {pname}...")

        plot_random_entry(r["rand_sharpes"], r["actual_sharpe"], r["percentile"], pname, fig_dir)
        os.rename(f"{fig_dir}/mc_random_entry.png", f"{fig_dir}/mc_random_{tag}.png")

        plot_bootstrap(r["boot_rets"], r["boot_wrs"], r["boot_pfs"],
                       r["actual_return"], r["actual_wr"], fig_dir)
        os.rename(f"{fig_dir}/mc_bootstrap.png", f"{fig_dir}/mc_bootstrap_{tag}.png")

        plot_path_shuffle(r["max_dds"], r["min_eqs"], r["eq_paths"], r["final_eq"], fig_dir=fig_dir)
        os.rename(f"{fig_dir}/mc_path_shuffle.png", f"{fig_dir}/mc_paths_{tag}.png")

        plot_summary_dashboard(
            r["actual_sharpe"], r["actual_return"], r["actual_wr"], r["n_trades"],
            r["percentile"], r["ci_return"], r["ci_wr"], r["ci_pf"],
            r["prob_pos_boot"], r["median_dd"], r["ci_dd"],
            r["median_min"], r["final_eq"], pname, fig_dir)
        os.rename(f"{fig_dir}/mc_summary.png", f"{fig_dir}/mc_summary_{tag}.png")

    print(f"\n{'=' * 70}")
    print(f"  CROSS-PERIOD COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Period':<25} {'Sharpe':>8} {'Return':>10} {'Trades':>8} {'%ile':>8} {'p-val':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for name, r in all_results.items():
        p_val = (100 - r["percentile"]) / 100
        print(f"  {name:<25} {r['actual_sharpe']:>8.2f} {r['actual_return']:>9.2f}% {r['n_trades']:>8} {r['percentile']:>7.1f}% {p_val:>8.4f}")

    print(f"\n  12 charts saved to figures/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
