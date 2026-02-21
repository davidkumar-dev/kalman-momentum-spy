from typing import Dict, List


def print_results_table(results: List[Dict], title: str = "RESULTS"):
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    print(f"  {'Strategy':<30} {'Return':>9} {'Sharpe':>8} {'MaxDD':>9} {'Trades':>7} {'WinRate':>8}")
    print(f"  {'-' * 30} {'-' * 9} {'-' * 8} {'-' * 9} {'-' * 7} {'-' * 8}")

    for r in results:
        print(
            f"  {r['name']:<30} "
            f"{r['total_return']:>8.2f}% "
            f"{r['sharpe']:>8.2f} "
            f"{r['max_drawdown']:>8.2f}% "
            f"{r['num_trades']:>7d} "
            f"{r['win_rate']:>7.1f}%"
        )

    print(f"{'=' * 90}")


def buy_and_hold_return(prices) -> float:
    return (prices.iloc[-1] / prices.iloc[0] - 1) * 100
