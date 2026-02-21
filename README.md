# Kalman Momentum - SPY Trading Signal

A long-only trading signal for SPY using 15-minute intraday data. The strategy uses a Kalman filter to smooth raw price data and pull out velocity and acceleration. When both are positive (price is going up and speeding up) it enters long. Position size depends on how strong velocity is compared to recent bars. Stronger momentum, bigger position. Trades hold for a minimum of 130 bars to avoid getting chopped up by noise, and exit when acceleration flips negative (trend is slowing down) or at a 520-bar cap.

Trained on 2014-2017 data. Tested on everything after, 2018-2022 and 2025-2026. There's a 3-year gap because my original dataset only went to 2022, and the Bloomberg terminal at my university only gives about 6 months of intraday history. So the most recent data I could pull was Aug 2025 onward. Turned out to be a good stress test since the signal had to work across completely different market conditions with no changes or retuning.

## Results

| Period | Sharpe | Return | p-value |
|---|---|---|---|
| Train (2014-2017) | 0.42 | +32.7% | - |
| Test (2018-2022) | 0.77 | +181.6% | 0.078 |
| Bloomberg (2025-2026) | 1.84 | +6.3% | 0.079 |
| **All Out-of-Sample** | **1.13** | **+199.2%** | **0.0002** |

233 out-of-sample trades across 8 years. p-value from Monte Carlo random entry benchmark with 10,000 simulations of random entries using the same trade count, hold period, and position size. None of them beat the strategy. Parameter sensitivity tested across 42 configurations, all produced positive Sharpe.

## Project Structure

    src/                              Core library
      signals.py                      Kalman signal generator
      position_manager.py             Entry, exit, and position sizing
      backtest_equity.py              Equity curve and trade tracking
      data_loader.py                  Data loading and train/test splits
      performance.py                  Metrics display
      monte_carlo_utils.py            Monte Carlo functions and chart generation

    scripts/                          Run in order to reproduce results
      01_backtest.py                  Strategy backtest across all periods
      02_monte_carlo.py               Per-period Monte Carlo validation
      03_monte_carlo_combined.py      Combined 2018-2022 Monte Carlo
      04_param_sensitivity.py         Parameter robustness grid
      05_bloomberg_validation.py      Bloomberg 2025-2026 out-of-sample test
      06_final_combined.py            All out-of-sample combined (p=0.0002)

    data/raw/                         CSV data files (not in repo)
    figures/                          Generated charts
    archive/                          Old notebooks and deprecated code

## Data

Two CSV files needed in data/raw/:
- SPY_15min.csv - 56,274 bars from 2014-2022
- SPY_DATA_2025-2026.csv - 3,468 bars from Aug 2025 to Feb 2026 (Bloomberg)

## Setup

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Reproduce

    python scripts/01_backtest.py
    python scripts/02_monte_carlo.py
    python scripts/03_monte_carlo_combined.py
    python scripts/04_param_sensitivity.py
    python scripts/05_bloomberg_validation.py
    python scripts/06_final_combined.py

Scripts 02, 03, 05, and 06 take a few minutes each (Monte Carlo simulations).