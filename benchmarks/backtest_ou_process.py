"""
CHROMA: Econometric Validation of the Ornstein-Uhlenbeck (O-U) Model.
Tests:
1. Augmented Dickey-Fuller (ADF) Stationarity Check (p-values).
2. Empirical Half-Life Distribution (tau = ln(2)/kappa).
3. S-Score Monotonicity: Conditional 5-day forward residual returns.
"""
from __future__ import annotations

import argparse
import math
import numpy as np
import pandas as pd
from scipy import stats

from ibkr_chroma.data.pipeline import DataPipeline
from ibkr_chroma.data.storage import MarketDataStorage
from ibkr_chroma.data.universe import UniverseManager
from ibkr_chroma.statarb.ou_process import OUProcessModel
from ibkr_chroma.statarb.pca_factors import PCAFactorModel


def test_stationarity_quick_adf(series: pd.Series) -> Tuple[float, float]:
    """
    Simplified Augmented Dickey-Fuller regression:
    delta_X(t) = alpha + gamma * X(t-1) + epsilon
    t-statistic on gamma tests for unit root.
    """
    clean = series.dropna().values
    if len(clean) < 30:
        return 1.0
    dx = np.diff(clean)
    x_prev = clean[:-1]
    
    # Regression: dx = alpha + gamma * x_prev
    A = np.column_stack([np.ones(len(x_prev)), x_prev])
    try:
        params, _, _, _ = np.linalg.lstsq(A, dx, rcond=None)
        gamma = params[1]
        sigma_sq = np.sum((dx - A @ params) ** 2) / (len(dx) - 2)
        var_gamma = sigma_sq / np.sum((x_prev - np.mean(x_prev)) ** 2)
        t_stat = gamma / np.sqrt(var_gamma)
        
        # Approximate MacKinnon 5% critical value (-2.86)
        if t_stat < -3.43:
            p_val = 0.01
        elif t_stat < -2.86:
            p_val = 0.05
        elif t_stat < -2.57:
            p_val = 0.10
        else:
            p_val = 0.50
        return t_stat, p_val
    except Exception:
        return 0.0, 1.0


def main():
    parser = argparse.ArgumentParser(description="Run Ornstein-Uhlenbeck Econometric Validation.")
    parser.add_argument("--index", default="SP500", choices=["SP500", "QQQ", "HSI"], help="Target index")
    parser.add_argument("--top-n", type=int, default=50, help="Universe size (default: 50)")
    parser.add_argument("--lookback", type=int, default=126, help="Calibration lookback window (default: 126)")
    parser.add_argument("--days", type=int, default=504, help="Total days if synthetic fallback is used")
    args = parser.parse_args()

    um = UniverseManager()
    symbols = um.get_symbols(indices=[args.index], top_n=args.top_n)
    storage = MarketDataStorage()
    pipeline = DataPipeline(storage=storage)

    stock_prices, _ = pipeline.build_feature_panels(symbols, bar_size="1_day")

    if not stock_prices.empty and len(stock_prices.columns) >= 20:
        print(f"Loaded {len(stock_prices.columns)} stocks from local Parquet cache ({len(stock_prices)} trading days).")
    else:
        print(f"Generating synthetic panel for diagnostic demonstration...")
        from benchmarks.backtest_statarb_pca import generate_benchmark_data
        stock_prices, _ = generate_benchmark_data(symbols, days=args.days)

    returns = stock_prices.pct_change().dropna(how="all")

    print("\n" + "=" * 85)
    print(f"=== ORNSTEIN-UHLENBECK PROCESS ECONOMETRIC VALIDATION ({args.index}) ===")
    print(f"Sample: {stock_prices.shape[1]} Stocks across {len(stock_prices)} Trading Days | Window: {args.lookback}d")
    print("=" * 85)

    # 1. Fit PCA Factor Model to extract residuals
    pca = PCAFactorModel(use_rmt=True)
    pca_res = pca.fit_transform(returns.iloc[-args.lookback:])
    residuals = pca_res.residuals
    cum_residuals = residuals.cumsum()

    # 2. Econometric Tests: ADF Stationarity & Half-Life
    ou_engine = OUProcessModel(min_half_life=2.0, max_half_life=35.0)
    valid_ou = {}
    stationary_count = 0

    print("\n--- 1. STATIONARITY & O-U PARAMETERS (LATEST ESTIMATION SLICE) ---")
    print(f"{'Ticker':<10} {'ADF Stat':<11} {'ADF p-val':<11} {'Half-Life':<12} {'Speed (kappa)':<15} {'Stationary?':<12}")
    print("-" * 75)

    for col in cum_residuals.columns:
        t_stat, p_val = test_stationarity_quick_adf(cum_residuals[col])
        param = ou_engine.calibrate_single_stock(cum_residuals[col])
        is_stat = p_val <= 0.05
        if is_stat:
            stationary_count += 1

        if param:
            valid_ou[col] = param
            status = "YES (p<=0.05)" if is_stat else "NO (Unit Root)"
            print(f"{col:<10} {t_stat:<11.2f} {p_val:<11.2f} {param.half_life_days:<12.1f}d {param.kappa:<15.4f} {status:<12}")

    pct_stat = (stationary_count / stock_prices.shape[1]) * 100.0
    print("-" * 75)
    print(f"Percentage of Universe with Stationary Residuals (ADF p<=0.05): {pct_stat:.1f}%")
    print(f"Valid Mean-Reverting Candidates (2d <= tau <= 35d):           {len(valid_ou)} / {stock_prices.shape[1]} stocks")

    # 3. S-Score Signal Monotonicity Test
    print("\n--- 2. S-SCORE SIGNAL MONOTONICITY: FORWARD 5-DAY RESIDUAL REVERSION ---")
    s_scores = ou_engine.calibrate_universe(residuals)

    if len(cum_residuals) > 10:
        # Measure forward change in cumulative residual: X(t+5) - X(t)
        fwd_delta_x = cum_residuals.shift(-5) - cum_residuals
        records = []
        
        # Test on historical slice where forward return is known
        eval_idx = -6
        for ticker, param in s_scores.items():
            s_val = param.centered_s_score
            delta_val = fwd_delta_x[ticker].iloc[eval_idx]
            if not np.isnan(delta_val):
                records.append({"s_score": s_val, "fwd_5d_reversion": delta_val})

        if records:
            df_mono = pd.DataFrame(records)
            df_mono["s_bucket"] = pd.qcut(
                df_mono["s_score"], 
                q=4, 
                labels=["Deep Cheap (s<-1.0)", "Mild Cheap", "Mild Rich", "Deep Rich (s>+1.0)"]
            )
            mean_rev = df_mono.groupby("s_bucket", observed=False)["fwd_5d_reversion"].mean()
            print(mean_rev.to_string())
            print("\nEconometric Rule: Deep Cheap residuals MUST be positive; Deep Rich residuals MUST be negative.")

    # 4. Standalone O-U Reversion Backtest
    print("\n--- 3. STANDALONE O-U RESIDUAL ARBITRAGE SIMULATION ---")
    strategy_returns = []
    prev_positions = pd.Series(0.0, index=returns.columns)

    for t_idx in range(args.lookback, len(returns)):
        curr_date = returns.index[t_idx]
        window_ret = returns.iloc[t_idx - args.lookback : t_idx]
        curr_prices = stock_prices.iloc[t_idx]

        pca_step = pca.fit_transform(window_ret)
        s_step = ou_engine.calibrate_universe(pca_step.residuals)
        target_w = ou_engine.update_positions(curr_date, curr_prices, s_step)
        target_w = target_w.reindex(returns.columns).fillna(0.0)

        # Residual P&L = Sum(w_{t-1} * Residual_t)
        day_res = pca_step.residuals.iloc[-1]
        pnl = float((prev_positions * day_res).sum())
        strategy_returns.append(pnl)
        prev_positions = target_w

    pnl_series = pd.Series(strategy_returns, index=returns.index[args.lookback:])
    ann_ret = pnl_series.mean() * 252.0
    ann_vol = pnl_series.std() * math.sqrt(252.0)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    print(f"Annualized Residual Alpha:  {ann_ret:+,.2%}")
    print(f"Residual Volatility:        {ann_vol:,.2%}")
    print(f"Residual Sharpe Ratio:      {sharpe:+,.2f}")
    print(f"Total Completed Trades:     {len(ou_engine.closed_trades)}")
    print("=" * 85)


if __name__ == "__main__":
    main()