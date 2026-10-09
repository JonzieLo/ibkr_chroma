"""
Volatility, Risk, Liquidity Factors.

Features:
- Realized Volatility
- Idiosyncratic Volatility
- Illiquidlity Premium (Amihud Ratio)
"""

import numpy as np
import pandas as pd

def realized_vol(closes: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """
    Annualized Realized Volatility, default over 3 months.
    """
    returns = closes.pct_change()
    ann_vol = returns.rolling(window=window).std() * np.sqrt(252)
    return ann_vol.replace([np.inf, -np.inf], np.nan)


def idiosyncrate_vol(
    closes: pd.DataFrame,
    benchmark_closes: pd.Series,
    window: int = 63
) -> pd.DataFrame:
    """
    Standard deviation of residuals over CAPM regression
    """
    stock_returns = closes.pct_change()
    bench_returns = benchmark_closes.pct_change()

    rolling_cov = stock_returns.rolling(window).corr(bench_returns) * (
        stock_returns.rolling(window).std().mul(bench_returns.rolling(window).std(), axis=0)
    )
    rolling_var = bench_returns.rolling(window).var()
    betas = rolling_cov.div(rolling_var, axis=0)

    stock_var = stock_returns.rolling(window).var()
    bench_var_term = betas.pow(2).mul(rolling_var, axis=0)
    residual_var = (stock_var - bench_var_term).clip(lower=1e-12)
    ivol = (residual_var * 252) ** 0.5
    return ivol.replace([np.inf, -np.inf], np.nan)


def amihud_ratio(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    Measures daily price impact per dollar value: |Return|/(P_t * Volume)
    """
    returns = closes.pct_change().abs()
    dollar_vol = closes * volumes
    daily_impact = returns/dollar_vol.replace(0.0, np.nan)
    return daily_impact.rolling(window).mean().replace([np.inf,-np.inf],np.nan)