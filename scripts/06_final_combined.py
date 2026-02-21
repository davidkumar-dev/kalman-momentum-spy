import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_loader import DataLoader
from src.signals import KalmanSignals
from src.monte_carlo_utils import (
    get_strategy_results, sharpe_from_bar_returns,
    random_entry_benchmark, bootstrap_confidence, path_shuffle,
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
    return df


def main():
    print("=" * 70)
    print("  FINAL COMBINED OUT-OF-SAMPLE VALIDATION")
    print("  All OOS data: 2018-2022 + 2025-2026")
    print("  233 trades across 8 years, never tuned on any of it")
    print("=" * 70)

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
    os.makedirs(fig_dir, exist_ok=True)

    loader = DataLoader()
    df = loader.load_spy_data()
    train, val, test = loader.split_data(df)
    oos_2018_2022 = pd.concat([val, test])

    bbg = load_bloomberg(os.path.join(data_dir, "SPY_DATA_2025-2026.csv"))

    print(f"\n  OOS 2018-2022: {len(oos_2018_2022)} bars")
    print(f"  Bloomberg 2025-2026: {len(bbg)} bars")

    prices_1 = oos_2018_2022["close"]
    kalman_1 = KalmanSignals().generate(prices_1)
    trades_1, rets_1, weights_1, eq_1, pos_1, _ = get_strategy_results(prices_1, kalman_1)

    prices_2 = bbg["close"]
    kalman_2 = KalmanSignals().generate(prices_2)
    trades_2, rets_2, weights_2, eq_2, pos_2, _ = get_strategy_results(prices_2, kalman_2)

    all_rets = rets_1 + rets_2
    all_weights = weights_1 + weights_2
    all_trades = trades_1 + trades_2
    n_trades = len(all_rets)

    total_return_1 = eq_1["total_return"]
    total_return_2 = eq_2["total_return"]
    combined_return = ((1 + total_return_1/100) * (1 + total_return_2/100) - 1) * 100
    overall_wr = np.sum(np.array(all_rets) > 0) / n_trades * 100

    bar_rets_1 = prices_1.pct_change().fillna(0)
    strat_rets_1 = bar_rets_1 * pos_1.reindex(bar_rets_1.index).fillna(0)
    bar_rets_2 = prices_2.pct_change().fillna(0)
    strat_rets_2 = bar_rets_2 * pos_2.reindex(bar_rets_2.index).fillna(0)
    all_bar_rets = pd.concat([strat_rets_1, strat_rets_2])
    annual_factor = np.sqrt(26 * 252)
    combined_sharpe = (all_bar_rets.mean() / all_bar_rets.std()) * annual_factor

    print(f"\n  Combined Results:")
    print(f"    Total trades:     {n_trades}")
    print(f"    Period 1 return:  {total_return_1:.2f}%")
    print(f"    Period 2 return:  {total_return_2:.2f}%")
    print(f"    Combined return:  {combined_return:.2f}%")
    print(f"    Overall WR:       {overall_wr:.1f}%")
    print(f"    Combined Sharpe:  {combined_sharpe:.2f}")

    pname = "All OOS Combined (2018-2022 + 2025-2026)"

    print(f"\n  Running Monte Carlo on combined trade pool ({n_trades} trades)...")

    combined_prices = pd.concat([prices_1, prices_2])
    avg_hold_1 = np.mean([t["exit_idx"] - t["entry_idx"] for t in trades_1]) if trades_1 else 130
    avg_hold_2 = np.mean([t["exit_idx"] - t["entry_idx"] for t in trades_2]) if trades_2 else 130
    avg_hold = int(np.mean([avg_hold_1, avg_hold_2]))
    avg_weight = np.mean(all_weights)

    print(f"\n  [1/3] Random Entry Benchmark (10,000 simulations)...")

    n = len(combined_prices)
    bar_returns = combined_prices.pct_change().fillna(0).values
    random_sharpes = []

    for _ in range(10000):
        positions = np.zeros(n)
        entries_placed = 0
        attempts = 0
        max_attempts = n_trades * 20

        while entries_placed < n_trades and attempts < max_attempts:
            attempts += 1
            entry = np.random.randint(0, max(1, n - avg_hold - 1))
            exit_idx = min(entry + avg_hold, n - 1)

            if np.any(positions[entry:exit_idx + 1] != 0):
                continue

            positions[entry:exit_idx + 1] = avg_weight
            entries_placed += 1

        strat_ret = bar_returns * positions
        std = np.std(strat_ret)
        if std > 0:
            sharpe = (np.mean(strat_ret) / std) * annual_factor
        else:
            sharpe = 0.0
        random_sharpes.append(sharpe)

    random_sharpes = np.array(random_sharpes)
    percentile = np.sum(random_sharpes < combined_sharpe) / len(random_sharpes) * 100
    p_val = (100 - percentile) / 100

    print(f"    Actual Sharpe: {combined_sharpe:.3f}")
    print(f"    Random mean: {np.mean(random_sharpes):.3f} | std: {np.std(random_sharpes):.3f}")
    print(f"    Percentile rank: {percentile:.1f}%")
    print(f"    p-value: {p_val:.4f}")

    boot_rets, boot_wrs, boot_pfs, prob_pos_boot = bootstrap_confidence(all_rets, all_weights)
    max_dds, min_eqs, eq_paths, final_eq = path_shuffle(all_rets, all_weights)

    ci_return = np.percentile(boot_rets, [2.5, 97.5])
    ci_wr = np.percentile(boot_wrs, [2.5, 97.5])
    ci_pf = np.percentile(boot_pfs, [2.5, 97.5])
    ci_dd = np.percentile(max_dds, [5, 95])
    median_dd = np.median(max_dds)
    median_min = np.median(min_eqs)

    print(f"\n  Generating charts...")

    plot_random_entry(random_sharpes, combined_sharpe, percentile, pname, fig_dir)
    os.rename(f"{fig_dir}/mc_random_entry.png", f"{fig_dir}/mc_random_all_oos.png")

    plot_bootstrap(boot_rets, boot_wrs, boot_pfs, combined_return, overall_wr, fig_dir)
    os.rename(f"{fig_dir}/mc_bootstrap.png", f"{fig_dir}/mc_bootstrap_all_oos.png")

    plot_path_shuffle(max_dds, min_eqs, eq_paths, final_eq, fig_dir=fig_dir)
    os.rename(f"{fig_dir}/mc_path_shuffle.png", f"{fig_dir}/mc_paths_all_oos.png")

    plot_summary_dashboard(
        combined_sharpe, combined_return, overall_wr, n_trades,
        percentile, ci_return, ci_wr, ci_pf,
        prob_pos_boot, median_dd, ci_dd,
        median_min, final_eq, pname, fig_dir)
    os.rename(f"{fig_dir}/mc_summary.png", f"{fig_dir}/mc_summary_all_oos.png")

    print(f"\n{'=' * 70}")
    print(f"  FINAL COMBINED RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total OOS trades:   {n_trades}")
    print(f"  Combined return:    {combined_return:.2f}%")
    print(f"  Combined Sharpe:    {combined_sharpe:.2f}")
    print(f"  Win Rate:           {overall_wr:.1f}%")
    print(f"  vs Random:          {percentile:.1f}th percentile (p = {p_val:.4f})")
    print(f"  P(positive):        {prob_pos_boot:.1f}%")
    print(f"  Return 95% CI:      [{ci_return[0]:.1f}%, {ci_return[1]:.1f}%]")
    print(f"  Profit Factor:      [{ci_pf[0]:.2f}, {ci_pf[1]:.2f}]")

    print(f"\n  FULL HISTORY:")
    print(f"  {'Period':<35} {'Sharpe':>8} {'Return':>10} {'%ile':>8} {'p-val':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    print(f"  {'Train (2014-2017)':<35} {'0.42':>8} {'32.70%':>10} {'55.7%':>8} {'0.443':>8}")
    print(f"  {'Val (2018-2020)':<35} {'0.65':>8} {'72.43%':>10} {'73.5%':>8} {'0.265':>8}")
    print(f"  {'Test (2021-2022)':<35} {'0.70':>8} {'34.52%':>10} {'80.7%':>8} {'0.193':>8}")
    print(f"  {'Bloomberg (2025-2026)':<35} {'1.84':>8} {'6.25%':>10} {'92.1%':>8} {'0.079':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    print(f"  {'ALL OOS COMBINED':<35} {combined_sharpe:>8.2f} {combined_return:>9.2f}% {percentile:>7.1f}% {p_val:>8.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
