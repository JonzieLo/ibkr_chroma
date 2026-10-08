"""
Cross-Sectional Modmentum Factors.

Features:
- Momentum
- Residual Momentum
- 52-week High Ratio
"""

import numpy as np
import pandas as pd


def momentum(closes: pd.DataFrame, lookback: int = 252, skip: int = 0) -> pd.DataFrame:
    """
    Score = P(t)/P(t-252) - 1.0
    Skip 21 days if skipping past month
    """
    past_price = closes.shift(skip)
    base_price = closes.shift(lookback)
    mom = (past_price/base_price) - 1.0
    return mom.replace([np.inf, -np.inf], np.nan)


def high_ratio_52w(closes: pd.DataFrame, lookback = 252) ->pd.DataFrame:
    rolling_peak = closes.rolling(window=lookback,min_periods=int(lookback * 0.8)).max()
    ratio = closes/rolling_peak
    return ratio.replace([np.inf, -np.inf],np.nan)


def momentum_residual(
    closes: pd.DataFrame,
    benchmark_closes: pd.Series,
    lookback: int = 252,
    skip: int = 0
) -> pd.DataFrame:
    """
    Momentum ex. market beta. Regresses stock returns against benchmark (SPY/QQQ/HSI), extracts cumulative residual return.
    """
    stock_returns = closes.pct_change()
    bench_returns = benchmark_closes.pct_change()

    aligned_bench = bench_returns.reindex(stock_returns.index)
    rolling_cov = stock_returns.rolling(lookback).cov(aligned_bench)
    rolling_var = stock_returns.rolling(lookback).var()
    betas = rolling_cov.div(rolling_var, axis=0)

    cum_stock = (closes.shift(skip)/closes.shift(lookback)) - 1.0
    cum_bench = (aligned_bench.shift(skip)/aligned_bench.shift(lookback)) - 1.0

    # Residual = R_stock - beta * R_bench
    res_mom = cum_stock.sub(betas.mul(cum_bench, axis=0))
    return res_mom.replace([np.inf, -np.inf], np.nan)