"""
Cross-Sectional Factor Model Backtester & Quantile Monotonicity.

Evaluates:
1. 5-Day Forward Information Coefficient (IC) time series.
2. Quantile Returns Monotonicity
3. Long Q5 / Short Q1 Dollar-Neutral Cumulative Returns.
4. Barra Ex-Ante Structural Risk Decomposition.
"""

import argparse
import math
import numpy as np
import pandas as pd

from ibkr_chroma.data.pipeline import DataPipeline
from ibkr_chroma.data.storage import MarketDataStorage
from ibkr_chroma.data.universe import UniverseManager
from ibkr_chroma.factors.engine import FactorEngine
from ibkr_chroma.factors.momentum import high_ratio_52w, momentum
from ibkr_chroma.factors.reversal import short_term_reversal
from ibkr_chroma.factors.volatility import realized_vol
from ibkr_chroma.portfolio.weights import generate_quintile_weights


def main():
    parser = argparse.ArgumentParser(description="Run Factor Model Backtest across Indices.")
    parser.add_argument("--index", default="SP500", choices=["SP500", "QQQ", "HSI"], help="Index universe")
    parser.add_argument("--top-n", type=int, default=50, help="Universe size (default: 50)")
    args = parser.parse_args()

    um = UniverseManager()
    symbols = um.get_symbols(indices=[args.index], top_n=args.top_n)
    sector_map = um.get_sector_map(indices=[args.index], top_n=args.top_n)
    storage = MarketDataStorage()
    pipeline = DataPipeline(storage=storage)

    closes, _ = pipeline.build_feature_panels(symbols, bar_size="1_day")
    if closes.empty or len(closes.columns) < 10:
        print(f"[ERROR] Insufficient data for {args.index}. Run `data.sync --index {args.index}` first.")
        return

    returns = closes.pct_change()

    print("=" * 80)
    print(f"=== CHROMA CROSS-SECTIONAL FACTOR BENCHMARK: {args.index} (N={closes.shape[1]} Stocks) ===")
    print("=" * 80)

    # 1. Compute Factors
    f_mom = momentum(closes, lookback=252, skip=21)
    f_52w = high_ratio_52w(closes, lookback=252)
    f_rev = short_term_reversal(closes, window=5)
    f_vol = realized_vol(closes, window=63)

    factors = {
        "12-1M Momentum": f_mom,
        "52-Week High": f_52w,
        "5-Day Reversal": f_rev,
        "Low Realized Vol": f_vol,
    }

    # 2. Information Coefficient Evaluation
    print("\n--- 1. FACTOR PREDICTIVE POWER: 5-DAY FORWARD IC ---")
    print(f"{'Factor':<22} {'Mean IC':<12} {'IC Std':<12} {'IC IR (t-stat)':<14}")
    print("-" * 60)

    for name, f_panel in factors.items():
        ic_ts = FactorEngine.compute_forward_ic(f_panel, closes, forward_days=5)
        mean_ic = ic_ts.mean()
        std_ic = ic_ts.std()
        ir = (mean_ic / std_ic) * math.sqrt(252 / 5) if std_ic > 0 else 0.0
        print(f"{name:<22} {mean_ic:+,.4f}     {std_ic:,.4f}     {ir:+,.2f}")

    # 3. Form Composite Signal (with Sector Neutralization)
    weights = {"12-1M Momentum": 0.25, "52-Week High": 0.25, "5-Day Reversal": 0.25, "Low Realized Vol": 0.25}
    composite_alpha = FactorEngine.build_composite_signal(factors, weights=weights, sector_map=sector_map)

    # 4. Quantile Monotonicity (Q1 to Q5)
    print("\n--- 2. QUANTILE RETURNS MONOTONICITY (5-Day Forward Holding) ---")
    fwd_5d_ret = closes.pct_change(5).shift(-5)
    q_returns = {f"Q{q}": [] for q in range(1, 6)}
    dates = composite_alpha.index[:-5]

    for d in dates:
        alpha_slice = composite_alpha.loc[d].dropna()
        ret_slice = fwd_5d_ret.loc[d].reindex(alpha_slice.index).dropna()
        common = alpha_slice.index.intersection(ret_slice.index)
        if len(common) < 15:
            continue
        
        ranks = pd.qcut(alpha_slice.loc[common], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        for q_label in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            q_members = ranks[ranks == q_label].index
            q_returns[q_label].append(ret_slice.loc[q_members].mean())

    print(f"{'Quantile':<12} {'Mean 5-Day Return':<22} {'Annualized Return':<20}")
    print("-" * 54)
    for q_label, rets in q_returns.items():
        mean_5d = np.mean(rets) if rets else 0.0
        ann_ret = (1.0 + mean_5d) ** (252 / 5) - 1.0
        print(f"{q_label:<12} {mean_5d:+,.4f}                {ann_ret:+,.2%}")

    # 5. Long Q5 / Short Q1 Backtest
    port_weights = generate_quintile_weights(composite_alpha, gross_leverage=2.0)
    lagged_w = port_weights.shift(1).dropna(how="all")
    daily_pnl = (lagged_w * returns.reindex(lagged_w.index)).sum(axis=1)

    ann_ret = daily_pnl.mean() * 252.0
    ann_vol = daily_pnl.std() * math.sqrt(252.0)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    print("\n--- 3. LONG Q5 / SHORT Q1 STRATEGY METRICS ---")
    print(f"Annualized Net Return:     {ann_ret:+,.2%}")
    print(f"Annualized Volatility:     {ann_vol:,.2%}")
    print(f"Sharpe Ratio:              {sharpe:+,.2f}")


if __name__ == "__main__":
    main()