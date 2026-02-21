import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_loader import DataLoader
from src.signals import KalmanSignals
from src.position_manager import PositionManager
from src.backtest_equity import EquityBacktest
from src.monte_carlo_utils import PALETTE, apply_style


MIN_HOLDS = [70, 90, 110, 130, 150, 170, 200]
PROCESS_NOISES = [0.003, 0.005, 0.01, 0.02, 0.05, 0.1]
VEL_LOOKBACKS = [200, 350, 500, 650, 800]


def run_config(prices, process_noise, min_hold, vel_lookback=500):
    kalman = KalmanSignals(process_noise=process_noise)
    signals = kalman.generate(prices)

    pm = PositionManager()
    pos, trades = pm.adaptive_strategy(
        signals, prices,
        min_hold=min_hold, max_hold=520,
        long_only=True, conviction_weighted=True,
        vel_lookback=vel_lookback,
    )

    eq_bt = EquityBacktest(initial_capital=10000)
    eq = eq_bt.run(prices, pos, trades)

    return eq["sharpe"], eq["total_return"], eq["max_drawdown"], eq["num_trades"], eq["win_rate"]


def plot_heatmap(grid, row_labels, col_labels, row_name, col_name,
                 metric_name, filepath, highlight_row=None, highlight_col=None):
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    masked = np.ma.masked_invalid(grid)
    vmin = np.nanpercentile(grid, 5)
    vmax = np.nanpercentile(grid, 95)

    im = ax.imshow(masked, cmap=plt.cm.RdYlGn, aspect="auto", vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric_name, fontsize=12)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["text"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels([str(c) for c in col_labels], fontsize=11)
    ax.set_yticklabels([str(r) for r in row_labels], fontsize=11)
    ax.set_xlabel(col_name, fontsize=13)
    ax.set_ylabel(row_name, fontsize=13)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = grid[i, j]
            if np.isnan(val):
                continue

            brightness = (val - vmin) / (vmax - vmin + 1e-8)
            text_color = "black" if brightness > 0.5 else "white"

            is_base = (highlight_row is not None and highlight_col is not None
                       and row_labels[i] == highlight_row and col_labels[j] == highlight_col)

            if is_base:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=PALETTE["gold"],
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=PALETTE["bg"],
                                  edgecolor=PALETTE["gold"], linewidth=2, alpha=0.9))
            else:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, color=text_color)

    ax.set_title(f"Parameter Sensitivity — {metric_name}\n"
                 f"Gold box = base configuration (never changed)",
                 fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    print(f"    Saved: {filepath}")
    plt.close()


def main():
    print("=" * 70)
    print("  PARAMETER SENSITIVITY ANALYSIS")
    print("  Proving the strategy isn't fragile")
    print("=" * 70)

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    loader = DataLoader()
    df = loader.load_spy_data()
    train, val, test = loader.split_data(df)

    combined = pd.concat([val, test])
    prices = combined["close"]
    bh = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

    print(f"\n  Combined OOS: {len(combined)} bars (2018-2022)")
    print(f"  Buy & Hold: {bh:.2f}%")
    print(f"\n  Base config: min_hold=130, process_noise=0.01, vel_lookback=500")

    print(f"\n  [1/3] min_hold x process_noise grid...")
    sharpe_grid = np.full((len(MIN_HOLDS), len(PROCESS_NOISES)), np.nan)
    return_grid = np.full((len(MIN_HOLDS), len(PROCESS_NOISES)), np.nan)

    for i, mh in enumerate(MIN_HOLDS):
        for j, pn in enumerate(PROCESS_NOISES):
            sharpe, ret, dd, ntrades, wr = run_config(prices, pn, mh)
            sharpe_grid[i, j] = sharpe
            return_grid[i, j] = ret
            print(f"    mh={mh:>4}, pn={pn:.3f} -> Sharpe={sharpe:.2f}, Ret={ret:.1f}%, Trades={ntrades}")

    plot_heatmap(sharpe_grid, MIN_HOLDS, PROCESS_NOISES,
                 "min_hold", "process_noise", "Sharpe Ratio",
                 f"{fig_dir}/sensitivity_sharpe.png", highlight_row=130, highlight_col=0.01)

    plot_heatmap(return_grid, MIN_HOLDS, PROCESS_NOISES,
                 "min_hold", "process_noise", "Total Return (%)",
                 f"{fig_dir}/sensitivity_return.png", highlight_row=130, highlight_col=0.01)

    print(f"\n  [2/3] vel_lookback sensitivity (min_hold=130, pn=0.01)...")
    vel_results = []
    for vl in VEL_LOOKBACKS:
        sharpe, ret, dd, ntrades, wr = run_config(prices, 0.01, 130, vel_lookback=vl)
        vel_results.append((vl, sharpe, ret, dd, ntrades, wr))
        print(f"    vel_lookback={vl:>4} -> Sharpe={sharpe:.2f}, Ret={ret:.1f}%, Trades={ntrades}")

    print(f"\n  [3/3] Generating 1D sensitivity plot...")
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    base_pn = 0.01
    base_mh = 130
    pn_idx = PROCESS_NOISES.index(base_pn)
    mh_idx = MIN_HOLDS.index(base_mh)

    mh_sharpes = sharpe_grid[:, pn_idx]
    ax = axes[0]
    ax.plot(MIN_HOLDS, mh_sharpes, "o-", color="#4fc3f7", linewidth=2, markersize=8)
    ax.axvline(130, color="#ffd54f", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("min_hold", fontsize=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_title("min_hold Sensitivity\n(process_noise=0.01)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    pn_sharpes = sharpe_grid[mh_idx, :]
    ax = axes[1]
    ax.plot(PROCESS_NOISES, pn_sharpes, "o-", color="#ab47bc", linewidth=2, markersize=8)
    ax.axvline(0.01, color="#ffd54f", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("process_noise", fontsize=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_title("process_noise Sensitivity\n(min_hold=130)", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    vl_vals = [v[0] for v in vel_results]
    vl_sharpes = [v[1] for v in vel_results]
    ax.plot(vl_vals, vl_sharpes, "o-", color="#66bb6a", linewidth=2, markersize=8)
    ax.axvline(500, color="#ffd54f", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("vel_lookback", fontsize=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_title("vel_lookback Sensitivity\n(base config)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    fig.suptitle("1D Parameter Sensitivity — Gold line = chosen value",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/sensitivity_1d.png", dpi=200, bbox_inches="tight")
    print(f"    Saved: {fig_dir}/sensitivity_1d.png")
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Sharpe range across all configs: {np.nanmin(sharpe_grid):.2f} to {np.nanmax(sharpe_grid):.2f}")
    print(f"  Base config Sharpe: {sharpe_grid[mh_idx, pn_idx]:.2f}")
    pct_positive = np.sum(sharpe_grid > 0) / sharpe_grid.size * 100
    print(f"  Configs with positive Sharpe: {pct_positive:.0f}%")

    best_idx = np.unravel_index(np.nanargmax(sharpe_grid), sharpe_grid.shape)
    print(f"  Best config: min_hold={MIN_HOLDS[best_idx[0]]}, pn={PROCESS_NOISES[best_idx[1]]} -> Sharpe={sharpe_grid[best_idx]:.2f}")
    print(f"  (We use 130/0.01 regardless — no post-hoc optimization)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
