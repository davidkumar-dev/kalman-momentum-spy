import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


class PositionManager:

    def fixed_hold(self, signals: pd.DataFrame, hold_period: int = 211) -> Tuple[pd.Series, List[Dict]]:
        n = len(signals)
        positions = np.zeros(n)
        trades = []

        current_dir = 0
        entry_idx = -1
        bars_held = 0

        for i in range(n):
            if current_dir != 0:
                bars_held += 1
                if bars_held >= hold_period:
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": current_dir,
                    })
                    current_dir = 0
                    bars_held = 0

            if current_dir == 0:
                if signals["long_entry"].iloc[i] == 1:
                    current_dir = 1
                    entry_idx = i
                    bars_held = 0
                elif signals["short_entry"].iloc[i] == 1:
                    current_dir = -1
                    entry_idx = i
                    bars_held = 0

            positions[i] = current_dir

        if current_dir != 0:
            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": n - 1,
                "direction": current_dir,
            })

        return pd.Series(positions, index=signals.index), trades

    def connor_exit(self, signals: pd.DataFrame, hold_period: int = 211) -> Tuple[pd.Series, List[Dict]]:
        n = len(signals)
        positions = np.zeros(n)
        trades = []

        current_dir = 0
        entry_idx = -1
        bars_held = 0

        for i in range(n):
            if current_dir != 0:
                bars_held += 1

                opposing_signal = False
                if current_dir == 1 and signals["short_entry"].iloc[i] == 1:
                    opposing_signal = True
                elif current_dir == -1 and signals["long_entry"].iloc[i] == 1:
                    opposing_signal = True

                expired = bars_held >= hold_period

                if opposing_signal or expired:
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": current_dir,
                    })
                    current_dir = 0
                    bars_held = 0

                    if opposing_signal:
                        if signals["long_entry"].iloc[i] == 1:
                            current_dir = 1
                            entry_idx = i
                            bars_held = 0
                        elif signals["short_entry"].iloc[i] == 1:
                            current_dir = -1
                            entry_idx = i
                            bars_held = 0

            if current_dir == 0:
                if signals["long_entry"].iloc[i] == 1:
                    current_dir = 1
                    entry_idx = i
                    bars_held = 0
                elif signals["short_entry"].iloc[i] == 1:
                    current_dir = -1
                    entry_idx = i
                    bars_held = 0

            positions[i] = current_dir

        if current_dir != 0:
            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": n - 1,
                "direction": current_dir,
            })

        return pd.Series(positions, index=signals.index), trades

    def adaptive_strategy(self, signals: pd.DataFrame, prices: pd.Series,
                          min_hold: int = 130, max_hold: int = 520,
                          long_only: bool = True,
                          conviction_weighted: bool = False,
                          vel_lookback: int = 500) -> Tuple[pd.Series, List[Dict]]:

        n = len(signals)
        positions = np.zeros(n)
        trades = []
        accel = signals["acceleration"].values
        vel = signals["velocity"].values

        vel_rank = np.zeros(n)
        if conviction_weighted:
            abs_vel = np.abs(vel)
            for i in range(vel_lookback, n):
                window = abs_vel[i - vel_lookback:i]
                valid = window[~np.isnan(window)]
                if len(valid) > 0:
                    pctile = np.sum(valid < abs_vel[i]) / len(valid)
                    vel_rank[i] = pctile
                else:
                    vel_rank[i] = 0.5

        current_dir = 0
        entry_idx = -1
        bars_held = 0
        current_weight = 1.0

        for i in range(n):
            if current_dir != 0:
                bars_held += 1

                force_exit = bars_held >= max_hold
                derivative_exit = False

                if bars_held >= min_hold:
                    if current_dir == 1 and accel[i] < 0:
                        derivative_exit = True
                    elif current_dir == -1 and accel[i] > 0:
                        derivative_exit = True

                if force_exit or derivative_exit:
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": current_dir,
                    })
                    current_dir = 0
                    bars_held = 0

            if current_dir == 0:
                take_long = signals["long_entry"].iloc[i] == 1
                take_short = (not long_only) and signals["short_entry"].iloc[i] == 1

                if take_long:
                    current_dir = 1
                    entry_idx = i
                    bars_held = 0
                    if conviction_weighted:
                        current_weight = 0.2 + 0.8 * vel_rank[i]
                    else:
                        current_weight = 1.0
                elif take_short:
                    current_dir = -1
                    entry_idx = i
                    bars_held = 0
                    if conviction_weighted:
                        current_weight = 0.2 + 0.8 * vel_rank[i]
                    else:
                        current_weight = 1.0

            if current_dir != 0:
                positions[i] = current_dir * current_weight
            else:
                positions[i] = 0

        if current_dir != 0:
            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": n - 1,
                "direction": current_dir,
            })

        return pd.Series(positions, index=signals.index), trades
