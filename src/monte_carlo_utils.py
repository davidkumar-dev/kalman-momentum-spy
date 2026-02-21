import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy import stats as sp_stats

from src.signals import KalmanSignals
from src.position_manager import PositionManager
from src.backtest_equity import EquityBacktest


PALETTE = {
    "bg": "#0f1117",
    "panel": "#1a1d29",
    "grid": "#2a2d3a",
    "text": "#e0e0e0",
    "accent": "#4fc3f7",
    "green": "#66bb6a",
    "red": "#ef5350",
    "gold": "#ffd54f",
    "purple": "#ab47bc",
    "orange": "#ffa726",
}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["panel"],
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["text"],
        "text.color": PALETTE["text"],
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 11,
    })


def get_strategy_results(prices, kalman_signals):
    pm = PositionManager()
    eq_bt = EquityBacktest(initial_capital=10000)

    pos, trades = pm.adaptive_strategy(
        kalman_signals, prices,
        long_only=True, conviction_weighted=True,
    )

    eq = eq_bt.run(prices, pos, trades)

    trade_returns = []
    trade_weights = []
    for t in trades:
        entry_px = prices.iloc[t["entry_idx"]]
        exit_px = prices.iloc[t["exit_idx"]]
        if t["direction"] == 1:
            trade_returns.append((exit_px / entry_px - 1))
        else:
            trade_returns.append((entry_px / exit_px - 1))

        entry_bar = t["entry_idx"]
        w = abs(pos.iloc[entry_bar]) if entry_bar < len(pos) else 1.0
        trade_weights.append(w)

    bar_returns = prices.pct_change().fillna(0)
    strat_bar_rets = bar_returns * pos.reindex(bar_returns.index).fillna(0)

    return trades, trade_returns, trade_weights, eq, pos, strat_bar_rets


def sharpe_from_bar_returns(bar_returns, annual_factor=None):
    if annual_factor is None:
        annual_factor = np.sqrt(26 * 252)
    vals = bar_returns.values if hasattr(bar_returns, "values") else np.array(bar_returns)
    if np.std(vals) == 0:
        return 0.0
    return (np.mean(vals) / np.std(vals)) * annual_factor


def random_entry_benchmark(prices, actual_trades, actual_sharpe, actual_weights, n_sims=10000):
    print(f"\n  [1/3] Random Entry Benchmark ({n_sims:,} simulations)...")

    n = len(prices)
    bar_returns = prices.pct_change().fillna(0).values
    num_trades = len(actual_trades)
    holds = [t["exit_idx"] - t["entry_idx"] for t in actual_trades]
    avg_hold = int(np.mean(holds))
    avg_weight = np.mean(actual_weights)

    random_sharpes = []

    for _ in range(n_sims):
        positions = np.zeros(n)
        entries_placed = 0
        attempts = 0
        max_attempts = num_trades * 20

        while entries_placed < num_trades and attempts < max_attempts:
            attempts += 1
            entry = np.random.randint(0, max(1, n - avg_hold - 1))
            exit_idx = min(entry + avg_hold, n - 1)

            if np.any(positions[entry:exit_idx + 1] != 0):
                continue

            positions[entry:exit_idx + 1] = avg_weight
            entries_placed += 1

        strat_ret = bar_returns * positions
        sharpe = sharpe_from_bar_returns(strat_ret)
        random_sharpes.append(sharpe)

    random_sharpes = np.array(random_sharpes)
    percentile = np.sum(random_sharpes < actual_sharpe) / len(random_sharpes) * 100
    p_value = (100 - percentile) / 100

    print(f"    Actual Sharpe: {actual_sharpe:.3f}")
    print(f"    Random mean: {np.mean(random_sharpes):.3f} | std: {np.std(random_sharpes):.3f}")
    print(f"    Percentile rank: {percentile:.1f}%")
    print(f"    p-value: {p_value:.4f}")

    return random_sharpes, percentile


def bootstrap_confidence(trade_returns, trade_weights, n_sims=10000):
    print(f"\n  [2/3] Bootstrap Confidence Intervals ({n_sims:,} simulations)...")

    trade_arr = np.array(trade_returns)
    weight_arr = np.array(trade_weights)
    n_trades = len(trade_arr)

    boot_returns = []
    boot_win_rates = []
    boot_profit_factors = []

    for _ in range(n_sims):
        idx = np.random.choice(n_trades, size=n_trades, replace=True)
        sample = trade_arr[idx]
        weights = weight_arr[idx]

        weighted_returns = sample * weights
        total_return = np.sum(weighted_returns) * 100
        win_rate = np.sum(sample > 0) / n_trades * 100

        gross_profit = np.sum(weighted_returns[weighted_returns > 0])
        gross_loss = abs(np.sum(weighted_returns[weighted_returns < 0]))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        boot_returns.append(total_return)
        boot_win_rates.append(win_rate)
        boot_profit_factors.append(min(pf, 10))

    ci_return = np.percentile(boot_returns, [2.5, 97.5])
    ci_wr = np.percentile(boot_win_rates, [2.5, 97.5])
    ci_pf = np.percentile(boot_profit_factors, [2.5, 97.5])
    prob_positive = np.sum(np.array(boot_returns) > 0) / n_sims * 100

    print(f"    Return 95% CI: [{ci_return[0]:.1f}%, {ci_return[1]:.1f}%]")
    print(f"    Win Rate 95% CI: [{ci_wr[0]:.1f}%, {ci_wr[1]:.1f}%]")
    print(f"    Profit Factor 95% CI: [{ci_pf[0]:.2f}, {ci_pf[1]:.2f}]")
    print(f"    P(positive return): {prob_positive:.1f}%")

    return boot_returns, boot_win_rates, boot_profit_factors, prob_positive


def path_shuffle(trade_returns, trade_weights, initial_capital=10000, n_sims=10000):
    print(f"\n  [3/3] Path-Dependent Simulation ({n_sims:,} simulations)...")

    trade_arr = np.array(trade_returns)
    weight_arr = np.array(trade_weights)
    n_trades = len(trade_arr)

    max_drawdowns = []
    min_equities = []
    equity_paths = []

    n_store = min(300, n_sims)
    store_every = max(1, n_sims // n_store)

    for sim in range(n_sims):
        perm = np.random.permutation(n_trades)
        shuffled_returns = trade_arr[perm]
        shuffled_weights = weight_arr[perm]

        equity = initial_capital
        peak = equity
        max_dd = 0.0
        min_eq = equity
        path = [equity]

        for j in range(n_trades):
            pnl = equity * shuffled_weights[j] * shuffled_returns[j]
            equity += pnl
            equity = max(equity, 0.01)

            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak
            if dd < max_dd:
                max_dd = dd
            if equity < min_eq:
                min_eq = equity

            path.append(equity)

        max_drawdowns.append(max_dd * 100)
        min_equities.append(min_eq)

        if sim % store_every == 0:
            equity_paths.append(path)

    max_drawdowns = np.array(max_drawdowns)
    min_equities = np.array(min_equities)
    final_equity = equity_paths[0][-1] if equity_paths else initial_capital

    median_dd = np.median(max_drawdowns)
    worst_dd = np.min(max_drawdowns)
    median_min = np.median(min_equities)
    ci_dd = np.percentile(max_drawdowns, [5, 95])
    ci_min = np.percentile(min_equities, [5, 95])

    print(f"    Final equity (all paths): ${final_equity:,.0f} (order-invariant)")
    print(f"    Median max drawdown: {median_dd:.1f}% | worst: {worst_dd:.1f}%")
    print(f"    DD 90% CI: [{ci_dd[0]:.1f}%, {ci_dd[1]:.1f}%]")
    print(f"    Median min equity: ${median_min:,.0f} | worst: ${np.min(min_equities):,.0f}")
    print(f"    Min equity 90% CI: [${ci_min[0]:,.0f}, ${ci_min[1]:,.0f}]")

    return max_drawdowns, min_equities, equity_paths, final_equity


def plot_random_entry(random_sharpes, actual_sharpe, percentile, period_name, fig_dir="figures"):
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(random_sharpes, bins=80, density=True,
            color=PALETTE["accent"], alpha=0.5,
            edgecolor=PALETTE["accent"], linewidth=0.5)

    kde_x = np.linspace(np.percentile(random_sharpes, 0.5),
                        np.percentile(random_sharpes, 99.5), 300)
    kde = sp_stats.gaussian_kde(random_sharpes)
    ax.plot(kde_x, kde(kde_x), color=PALETTE["accent"], linewidth=2.5, alpha=0.9)

    ax.axvline(actual_sharpe, color=PALETTE["gold"], linewidth=2.5, linestyle="--", zorder=5)

    ymax = ax.get_ylim()[1]
    ax.annotate(
        f"Kalman Strategy\nSharpe = {actual_sharpe:.2f}\n({percentile:.1f}th percentile)",
        xy=(actual_sharpe, ymax * 0.7),
        xytext=(actual_sharpe + 0.35, ymax * 0.85),
        fontsize=12, fontweight="bold", color=PALETTE["gold"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["gold"], lw=2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["bg"],
                  edgecolor=PALETTE["gold"], alpha=0.9))

    p_value = (100 - percentile) / 100
    if p_value < 0.01:
        sig_text = f"SIGNIFICANT (p = {p_value:.4f})"
        sig_color = PALETTE["green"]
    elif p_value < 0.05:
        sig_text = f"SIGNIFICANT (p = {p_value:.3f})"
        sig_color = PALETTE["green"]
    elif p_value < 0.10:
        sig_text = f"MARGINAL (p = {p_value:.3f})"
        sig_color = PALETTE["orange"]
    else:
        sig_text = f"NOT SIGNIFICANT (p = {p_value:.3f})"
        sig_color = PALETTE["red"]

    ax.text(0.02, 0.95, sig_text, transform=ax.transAxes, fontsize=11,
            fontweight="bold", color=sig_color, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["bg"],
                      edgecolor=sig_color, alpha=0.9))

    ax.text(0.02, 0.82,
            f"n = 10,000 random entries\nSame trade count, hold, weight\nNon-overlapping enforced",
            transform=ax.transAxes, fontsize=9, color=PALETTE["text"], alpha=0.6, va="top")

    ax.set_xlabel("Sharpe Ratio", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"Random Entry Benchmark — {period_name}",
                 fontsize=15, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = f"{fig_dir}/mc_random_entry.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"    Saved: {path}")
    plt.close()
    return path


def plot_bootstrap(boot_returns, boot_win_rates, boot_profit_factors,
                   actual_return, actual_wr, fig_dir="figures"):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    datasets = [
        (boot_returns, actual_return, "Total Return (%)", PALETTE["purple"]),
        (boot_win_rates, actual_wr, "Win Rate (%)", PALETTE["accent"]),
        (boot_profit_factors, None, "Profit Factor", PALETTE["green"]),
    ]

    for ax, (data, actual, xlabel, color) in zip(axes, datasets):
        ci = np.percentile(data, [2.5, 97.5])
        n_out, bins, patches = ax.hist(data, bins=60, density=True,
                                       color=color, alpha=0.5,
                                       edgecolor=color, linewidth=0.5)

        for patch, b in zip(patches, bins):
            center = b + (bins[1] - bins[0]) / 2
            if ci[0] <= center <= ci[1]:
                patch.set_alpha(0.7)
            else:
                patch.set_facecolor(PALETTE["grid"])
                patch.set_alpha(0.3)

        if actual is not None:
            ax.axvline(actual, color=PALETTE["gold"], linewidth=2.5, linestyle="--")

        ax.axvline(ci[0], color=PALETTE["text"], linewidth=1, linestyle=":", alpha=0.5)
        ax.axvline(ci[1], color=PALETTE["text"], linewidth=1, linestyle=":", alpha=0.5)
        ax.axvspan(ci[0], ci[1], alpha=0.05, color=color)

        ax.text(0.5, 0.95, f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]",
                transform=ax.transAxes, fontsize=10, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["bg"],
                          edgecolor=color, alpha=0.9))

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Bootstrap Confidence Intervals (10,000 resamples of trade returns)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{fig_dir}/mc_bootstrap.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"    Saved: {path}")
    plt.close()
    return path


def plot_path_shuffle(max_drawdowns, min_equities, equity_paths, final_equity,
                      initial_capital=10000, fig_dir="figures"):
    apply_style()
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(min_equities, bins=60, density=True,
             color=PALETTE["orange"], alpha=0.5,
             edgecolor=PALETTE["orange"], linewidth=0.5)
    median_min = np.median(min_equities)
    ax1.axvline(median_min, color=PALETTE["gold"], linewidth=2, linestyle="--")
    ax1.axvline(initial_capital, color=PALETTE["text"], linewidth=1, linestyle="--", alpha=0.4)
    ymax1 = ax1.get_ylim()[1]
    ax1.text(median_min, ymax1 * 0.85, f"  Median: ${median_min:,.0f}",
             fontsize=10, color=PALETTE["gold"], fontweight="bold")
    worst_min = np.min(min_equities)
    ax1.text(0.02, 0.95, f"Worst case: ${worst_min:,.0f}",
             transform=ax1.transAxes, fontsize=9, color=PALETTE["red"], va="top",
             bbox=dict(boxstyle="round,pad=0.2", facecolor=PALETTE["bg"],
                       edgecolor=PALETTE["red"], alpha=0.8))
    ax1.set_title("Minimum Equity Touched\n(worst moment during each path)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Equity ($)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(max_drawdowns, bins=60, density=True,
             color=PALETTE["red"], alpha=0.5,
             edgecolor=PALETTE["red"], linewidth=0.5)
    median_dd = np.median(max_drawdowns)
    ax2.axvline(median_dd, color=PALETTE["gold"], linewidth=2, linestyle="--")
    ymax2 = ax2.get_ylim()[1]
    ax2.text(median_dd, ymax2 * 0.85, f"  Median: {median_dd:.1f}%",
             fontsize=10, color=PALETTE["gold"], fontweight="bold")
    ci_dd = np.percentile(max_drawdowns, [5, 95])
    ax2.text(0.02, 0.95, f"90% CI: [{ci_dd[0]:.1f}%, {ci_dd[1]:.1f}%]",
             transform=ax2.transAxes, fontsize=9, color=PALETTE["text"], va="top",
             bbox=dict(boxstyle="round,pad=0.2", facecolor=PALETTE["bg"],
                       edgecolor=PALETTE["red"], alpha=0.8))
    ax2.set_title("Max Drawdown Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Max Drawdown (%)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(gs[1, :])
    if equity_paths:
        for path in equity_paths:
            min_eq = min(path)
            color = PALETTE["green"] if min_eq >= initial_capital * 0.7 else PALETTE["red"]
            ax3.plot(range(len(path)), path, color=color, alpha=0.06, linewidth=0.8)

        paths_arr = np.array(equity_paths)
        median_path = np.median(paths_arr, axis=0)
        p5_path = np.percentile(paths_arr, 5, axis=0)
        p95_path = np.percentile(paths_arr, 95, axis=0)

        x = range(len(median_path))
        ax3.plot(x, median_path, color=PALETTE["gold"], linewidth=2.5, label="Median path", zorder=5)
        ax3.fill_between(x, p5_path, p95_path, alpha=0.15, color=PALETTE["accent"], label="90% CI band")
        ax3.axhline(initial_capital, color=PALETTE["text"], linewidth=1, linestyle="--", alpha=0.3)
        ax3.axhline(final_equity, color=PALETTE["accent"], linewidth=1.5, linestyle=":",
                     alpha=0.7, label=f"Final: ${final_equity:,.0f}")
        ax3.legend(loc="upper left", fontsize=10)

    ax3.set_title("Equity Paths — Random Trade Orderings", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Trade Number", fontsize=11)
    ax3.set_ylabel("Equity ($)", fontsize=11)
    ax3.grid(True, alpha=0.2)

    fig.suptitle("Path-Dependent Analysis — How Much Does Trade Ordering Matter?\n"
                 "Final equity is identical (multiplication is commutative), but the journey differs",
                 fontsize=14, fontweight="bold", y=1.02)
    path = f"{fig_dir}/mc_path_shuffle.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"    Saved: {path}")
    plt.close()
    return path


def plot_summary_dashboard(actual_sharpe, actual_return, actual_wr, n_trades,
                           percentile, ci_return, ci_wr, ci_pf,
                           prob_positive_boot, median_dd, ci_dd,
                           median_min, final_equity, period_name, fig_dir="figures"):
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    ax.text(0.5, 0.96, "MONTE CARLO VALIDATION SUMMARY",
            transform=ax.transAxes, fontsize=22, fontweight="bold",
            ha="center", va="top", color=PALETTE["gold"])

    ax.text(0.5, 0.90,
            f"Kalman Long-Only Conviction Strategy — {period_name} — {n_trades} trades",
            transform=ax.transAxes, fontsize=13, ha="center", va="top",
            color=PALETTE["text"], alpha=0.7)

    p_value = (100 - percentile) / 100

    sections = [
        ("TEST 1: SIGNAL QUALITY (vs Random Entries)", PALETTE["accent"], [
            ("Strategy Sharpe", f"{actual_sharpe:.2f}"),
            ("Percentile vs Random", f"{percentile:.1f}%"),
            ("p-value", f"{p_value:.4f}"),
        ]),
        ("TEST 2: RETURN CONFIDENCE (Bootstrap)", PALETTE["purple"], [
            ("Return 95% CI", f"[{ci_return[0]:.1f}%, {ci_return[1]:.1f}%]"),
            ("Win Rate 95% CI", f"[{ci_wr[0]:.1f}%, {ci_wr[1]:.1f}%]"),
            ("Profit Factor 95% CI", f"[{ci_pf[0]:.2f}, {ci_pf[1]:.2f}]"),
            ("P(positive return)", f"{prob_positive_boot:.1f}%"),
        ]),
        ("TEST 3: PATH ROBUSTNESS (Trade Ordering)", PALETTE["green"], [
            ("Final Equity (all paths)", f"${final_equity:,.0f}"),
            ("Median Max Drawdown", f"{median_dd:.1f}%"),
            ("DD 90% CI", f"[{ci_dd[0]:.1f}%, {ci_dd[1]:.1f}%]"),
            ("Median Min Equity", f"${median_min:,.0f}"),
        ]),
    ]

    y = 0.82
    for title, color, items in sections:
        ax.text(0.06, y, title, transform=ax.transAxes,
                fontsize=12, fontweight="bold", color=color)
        y -= 0.05
        for label, value in items:
            ax.text(0.10, y, label, transform=ax.transAxes,
                    fontsize=11, color=PALETTE["text"])
            ax.text(0.60, y, value, transform=ax.transAxes,
                    fontsize=13, fontweight="bold", color=color, family="monospace")
            y -= 0.045
        y -= 0.025

    if p_value < 0.05 and prob_positive_boot > 80:
        verdict = "STRONG STATISTICAL EVIDENCE OF EDGE"
        vc = PALETTE["green"]
    elif p_value < 0.10 and prob_positive_boot > 70:
        verdict = "MODERATE EVIDENCE — EDGE LIKELY BUT NEEDS MORE DATA"
        vc = PALETTE["gold"]
    elif prob_positive_boot > 70:
        verdict = "POSITIVE EDGE DETECTED — SIGNIFICANCE REQUIRES MORE TRADES"
        vc = PALETTE["orange"]
    else:
        verdict = "INSUFFICIENT EVIDENCE — CANNOT CONFIRM EDGE"
        vc = PALETTE["red"]

    box = FancyBboxPatch((0.05, 0.01), 0.9, 0.07,
                          boxstyle="round,pad=0.01",
                          facecolor=vc, alpha=0.12,
                          edgecolor=vc, linewidth=2,
                          transform=ax.transAxes)
    ax.add_patch(box)
    ax.text(0.5, 0.045, verdict, transform=ax.transAxes,
            fontsize=15, fontweight="bold", ha="center", va="center", color=vc)

    path = f"{fig_dir}/mc_summary.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"    Saved: {path}")
    plt.close()
    return path
