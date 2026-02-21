import pandas as pd
import numpy as np
from pykalman import KalmanFilter


class ConnorSignals:

    def __init__(self, ma_period: int = 200, ma_long_period: int = 400, lookback: int = 30):
        self.ma_period = ma_period
        self.ma_long_period = ma_long_period
        self.lookback = lookback

    def generate(self, prices: pd.Series) -> pd.DataFrame:
        ma200 = prices.rolling(window=self.ma_period).mean()
        ma400 = prices.rolling(window=self.ma_long_period).mean()

        n = len(prices)
        slope_raw = np.zeros(n)
        convexity_raw = np.zeros(n)
        slope_400_raw = np.zeros(n)

        ma200_vals = ma200.values
        ma400_vals = ma400.values
        px = prices.values

        start = self.ma_long_period + self.lookback

        for i in range(start, n):
            ma_window = ma200_vals[i - self.lookback:i + 1]
            if np.any(np.isnan(ma_window)):
                continue

            start_ma = ma_window[0]
            end_ma = ma_window[-1]
            slope = (end_ma - start_ma) / self.lookback

            t = np.arange(self.lookback + 1)
            secant = start_ma + slope * t
            area = np.trapezoid(secant - ma_window, t)

            price = px[i]
            if price == 0:
                continue

            normalized_slope = (slope / price) * 100 * self.lookback
            normalized_area = (area / price) * self.lookback

            slope_raw[i] = 100 * np.tanh(normalized_slope)
            convexity_raw[i] = 100 * np.tanh(normalized_area)

            if not np.isnan(ma400_vals[i]) and not np.isnan(ma400_vals[i - self.lookback]):
                slope_400_raw[i] = (ma400_vals[i] - ma400_vals[i - self.lookback]) / self.lookback

        signals = pd.DataFrame(index=prices.index)
        signals["ma200"] = ma200
        signals["ma400"] = ma400
        signals["slope"] = slope_raw
        signals["convexity"] = convexity_raw
        signals["convexity_ma5"] = pd.Series(convexity_raw, index=prices.index).rolling(window=5).mean()
        signals["slope_400"] = slope_400_raw
        signals["price"] = px

        slope = signals["slope"].values
        slope_prev = np.roll(slope, 1)
        slope_prev[0] = 0
        conv_ma5 = signals["convexity_ma5"].values
        s400 = signals["slope_400"].values

        long_entry = np.zeros(n, dtype=int)
        short_entry = np.zeros(n, dtype=int)

        for i in range(1, n):
            if (slope[i] > 0 and slope_prev[i] <= 0
                    and conv_ma5[i] > 0
                    and px[i] > ma200_vals[i]
                    and s400[i] > 0):
                long_entry[i] = 1

            if (slope[i] < 0 and slope_prev[i] >= 0
                    and conv_ma5[i] < 0
                    and px[i] < ma200_vals[i]
                    and s400[i] < 0):
                short_entry[i] = 1

        signals["long_entry"] = long_entry
        signals["short_entry"] = short_entry

        return signals


class MA200Signals:

    def __init__(self, ma_period: int = 200, slope_lookback: int = 30):
        self.ma_period = ma_period
        self.slope_lookback = slope_lookback

    def generate(self, prices: pd.Series) -> pd.DataFrame:
        ma = prices.rolling(window=self.ma_period).mean()

        slope = (ma - ma.shift(self.slope_lookback)) / self.slope_lookback

        convexity = pd.Series(np.nan, index=prices.index)
        ma_vals = ma.values
        for i in range(self.slope_lookback, len(ma_vals)):
            window = ma_vals[i - self.slope_lookback : i + 1]
            if np.any(np.isnan(window)):
                continue
            secant = np.linspace(window[0], window[-1], len(window))
            convexity.iloc[i] = np.trapezoid(window - secant)

        long_cond = (slope > 0) & (convexity > 0)
        short_cond = (slope < 0) & (convexity < 0)

        signals = pd.DataFrame(index=prices.index)
        signals["ma"] = ma
        signals["slope"] = slope
        signals["convexity"] = convexity
        signals["long_entry"] = (long_cond & ~long_cond.shift(1).fillna(False)).astype(int)
        signals["short_entry"] = (short_cond & ~short_cond.shift(1).fillna(False)).astype(int)

        return signals


class KalmanSignals:

    def __init__(self, process_noise: float = 0.01, observation_noise: float = 1.0):
        self.process_noise = process_noise
        self.observation_noise = observation_noise

    def generate(self, prices: pd.Series) -> pd.DataFrame:
        kf = KalmanFilter(
            transition_matrices=[[1, 1, 0.5], [0, 1, 1], [0, 0, 1]],
            observation_matrices=[[1, 0, 0]],
            initial_state_mean=[prices.iloc[0], 0, 0],
            transition_covariance=self.process_noise * np.eye(3),
            observation_covariance=[[self.observation_noise]],
        )

        state_means, _ = kf.filter(prices.values)

        signals = pd.DataFrame(index=prices.index)
        signals["filtered_price"] = state_means[:, 0]
        signals["velocity"] = state_means[:, 1]
        signals["acceleration"] = state_means[:, 2]

        long_cond = (signals["velocity"] > 0) & (signals["acceleration"] > 0)
        short_cond = (signals["velocity"] < 0) & (signals["acceleration"] < 0)

        signals["long_entry"] = (long_cond & ~long_cond.shift(1).fillna(False)).astype(int)
        signals["short_entry"] = (short_cond & ~short_cond.shift(1).fillna(False)).astype(int)

        return signals
