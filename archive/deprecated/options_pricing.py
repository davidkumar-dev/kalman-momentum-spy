import numpy as np
from scipy.stats import norm


class BlackScholes:

    def __init__(self, risk_free_rate: float = 0.02):
        self.r = risk_free_rate

    def call(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0:
            return max(S - K, 0.0)
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)

    def put(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0:
            return max(K - S, 0.0)
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
