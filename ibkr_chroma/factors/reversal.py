"""
Short-Term Flow & Microstructure Reversal Factors.

Features:
- Weekly Mean-Reversion
- Bollinger Band/Distance to Rolling Mean z-score
"""

import numpy as np
import pandas as pd


def short_term_reversal(closes: pd.DataFrame, window:int = 5) -> pd.DataFrame:
    """
    Score = -1 * (P_t/P_{t-5} - 1.0)
    High score -> Stock sold off heavily last week, expect bounce.
    """
    ret = closes.pct_change(periods=window)
    return -ret.replace([np.inf,-np.inf], np.nan)


def price_to_bollinger_band(closes: pd.DataFrame, window:int = 20) -> pd.DataFrame:
    """
    Score = -1 * (P_t - mean/std)
    """
    rolling_mean = closes.rolling(window=window).mean()
    rolling_std = closes.rolling(window=window).std()
    z_score = (closes - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return -z_score.replace([np.inf, -np.inf], np.nan)