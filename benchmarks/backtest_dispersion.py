"""
Historical Equity Index Dispersion P&L Backtest.
Simulates daily Delta-Hedged Straddle Cash Flows: Gamma Gains vs Theta Decay.
"""

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd

from ibkr_chroma.data.pipeline import DataPipeline
from ibkr_chroma.data.storage import MarketDataStorage
from ibkr_chroma.data.universe import UniverseManager
from ibkr_chroma.dispersion.correlation import DispersionCorrelationModel
from ibkr_chroma.dispersion.pricing import BSMDividendModel, BinomialModel


def load_real_iv_panels(symbols: list[str]) -> pd.DataFrame:
    iv_dict = {}
    for s in symbols:
        p = Path(f"data/parquet/{s.upper()}_iv_1_day.parquet")
        if p.exists():
            df = pd.read_parquet(p)
            iv_dict[s] = df["close"]  # Real ATM IV
    if not iv_dict:
        return pd.DataFrame()
    return pd.DataFrame(iv_dict).sort_index().ffill().dropna(how="all")


def main():
    parser = argparse.ArgumentParser(description="Run Historical Dispersion P&L Backtest.")
    parser.add_argument("--index", default="HSI", choices=["SP500", "QQQ", "HSI"], help="Index universe")
    parser.add_argument("--top-n", type=int, default=10, help="Constituent basket size")
    parser.add_argument("--min-spread", type=float, default=0.08, help="Min correlation spread to enter")
    args = parser.parse_args()

    index_ticker = "SPY" if args.index == "SP500" else ("QQQ" if args.index == "QQQ" else "2800")
    um = UniverseManager()
    symbols = um.get_symbols(indices=[args.index], top_n=args.top_n)

    storage = MarketDataStorage()
    pipeline = DataPipeline(storage=storage)

    closes, _ = pipeline.build_feature_panels(symbols + [index_ticker], bar_size="1_day")
    real_ivs = load_real_iv_panels(symbols + [index_ticker])

    if real_ivs.empty or index_ticker not in real_ivs.columns:
        print(f"[ERROR] Real option IV bars missing for {args.index}. Run `data.sync_options --index {args.index}` first.")
        return

    # Align dates
    common_dates = closes.index.intersection(real_ivs.index)
    closes = closes.loc[common_dates]
    real_ivs = real_ivs.loc[common_dates]
    returns = closes.pct_change().dropna(how="all")


    print(f"=== {args.index} DISPERSION P&L BACKTEST ===")
    print(f"Index Asset: {index_ticker} | Basket: Top {len(symbols)} Stocks | Entry Threshold: +{args.min_spread:.1%} Spread")

    caps_file = Path("data/universe/market_caps.parquet")
    if caps_file.exists():
        caps_df = pd.read_parquet(caps_file)
        raw_weights = caps_df["market_cap"].reindex(symbols).fillna(1.0)
        weights = raw_weights / raw_weights.sum()
    else:
        weights = pd.Series(1.0 / len(symbols), index=symbols)
    daily_pnl_history = []
    trade_events = 0
    tau = 30.0 / 365.0
    dt = 1.0 / 252.0

    print("Simulating rolling delta-hedged straddle gamma/theta cash flows...")
    for t_idx in range(63, len(returns) - 1):
        curr_date = returns.index[t_idx]
        idx_iv = float(real_ivs[index_ticker].iloc[t_idx])
        const_ivs = real_ivs[symbols].iloc[t_idx]

        # Check Correlation Spread Signal
        ret_window = returns[symbols].iloc[t_idx - 63 : t_idx]
        metrics = DispersionCorrelationModel.evaluate_correlation_opportunity(
            index_iv=idx_iv,
            constituent_ivs=const_ivs,
            constituent_weights=weights,
            constituent_returns=ret_window,
            window=63,
        )

        if metrics.correlation_spread >= args.min_spread:
            trade_events += 1
            # Next-day realized price moves
            next_idx_ret = returns[index_ticker].iloc[t_idx + 1]
            next_const_rets = returns[symbols].iloc[t_idx + 1]

            # Index Leg: Short Straddle (Loss from Gamma Move, Gain from Theta Decay)
            # Gamma ~= 2 * N(d1) / (S * sigma * sqrt(tau)) ~= 0.80 / (S * sigma * sqrt(tau))
            idx_gamma_dollar = 0.80 / (idx_iv * math.sqrt(tau))
            idx_theta_dollar = 0.40 * idx_iv / math.sqrt(tau) * dt
            index_pnl = - (0.5 * idx_gamma_dollar * (next_idx_ret ** 2)) + idx_theta_dollar

            # Constituent Basket: Long Straddles (Gain from Gamma Move, Loss from Theta Decay)
            basket_pnl = 0.0
            for s in symbols:
                s_iv = float(const_ivs[s])
                s_ret = float(next_const_rets[s])
                s_gamma_dollar = 0.80 / (s_iv * math.sqrt(tau))
                s_theta_dollar = 0.40 * s_iv / math.sqrt(tau) * dt
                basket_pnl += weights[s] * ((0.5 * s_gamma_dollar * (s_ret ** 2)) - s_theta_dollar)

            net_day_pnl = basket_pnl + index_pnl
            daily_pnl_history.append({"date": curr_date, "pnl": net_day_pnl})

    pnl_df = pd.DataFrame(daily_pnl_history).set_index("date")
    ret_series = pnl_df["pnl"]

    mean_ann = ret_series.mean() * 252.0
    vol_ann = ret_series.std() * math.sqrt(252.0)
    sharpe = mean_ann / vol_ann if vol_ann > 0 else 0.0
    cum_pnl = ret_series.cumsum()
    max_dd = (cum_pnl - cum_pnl.cummax()).min()

    print("\n--- DISPERSION P&L BACKTEST RESULTS (NET OF GAMMA/THETA DRAG) ---")
    print(f"Annualized Return Proxy:    {mean_ann:+,.2%}")
    print(f"Annualized Volatility:      {vol_ann:,.2%}")
    print(f"Dispersion Sharpe Ratio:    {sharpe:+,.2f}")
    print(f"Maximum Drawdown:           {max_dd:,.2%}")
    print(f"Active Trading Days:        {len(ret_series)} / {len(returns) - 64}")


if __name__ == "__main__":
    main()