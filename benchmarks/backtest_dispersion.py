"""
CHROMA: Equity Index Dispersion Trading Benchmark & Correlation Gap Scanner.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from ibkr_chroma.data.pipeline import DataPipeline
from ibkr_chroma.data.storage import MarketDataStorage
from ibkr_chroma.data.universe import UniverseManager
from ibkr_chroma.dispersion.correlation import DispersionCorrelationModel
from ibkr_chroma.dispersion.engine import DispersionTradeEngine


def main():
    parser = argparse.ArgumentParser(description="Run Equity Index Dispersion Correlation Scanner.")
    parser.add_argument("--index", default="SP500", choices=["SP500", "SPY", "QQQ", "HSI"], help="Target index")
    parser.add_argument("--top-n", type=int, default=10, help="Number of constituent basket stocks (default: 10)")
    parser.add_argument("--window", type=int, default=63, help="Rolling correlation window in days")
    args = parser.parse_args()

    if args.index in ["SP500", "SPY"]:
        target_universe = "SP500"
        index_ticker = "SPY"
    elif args.index == "QQQ":
        target_universe = "QQQ"
        index_ticker = "QQQ"
    else:  # HSI
        target_universe = "HSI"
        index_ticker = "2800"

    um = UniverseManager()
    storage = MarketDataStorage()
    pipeline = DataPipeline(storage=storage)

    # Identify True Market Leaders (Mega-Caps)
    if args.index in ["SP500", "SPY"]:
        target_universe = "SP500"
        index_ticker = "SPY"
        priority_leaders = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK B", "LLY", "AVGO", "JPM", "TSLA"]
    elif args.index == "QQQ":
        target_universe = "QQQ"
        index_ticker = "QQQ"
        priority_leaders = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "COST", "ASML", "AMD"]
    else:  # HSI
        target_universe = "HSI"
        index_ticker = "2800"
        priority_leaders = ["700", "9988", "5", "939", "1299", "1398", "941", "388", "1810", "3988"]

    available_symbols = [s for s in priority_leaders if storage.has_data(s)]
    if len(available_symbols) < 5:
        all_symbols = um.get_symbols(indices=[target_universe])
        available_symbols = [s for s in all_symbols if storage.has_data(s)][:args.top_n]

    symbols = available_symbols[:args.top_n]
    closes, _ = pipeline.build_feature_panels(symbols + [index_ticker], bar_size="1_day")

    if closes.empty or index_ticker not in closes.columns:
        print(f"Loading local data... Generating demonstration panel for {args.index} dispersion...")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq="B")
        np.random.seed(42)
        idx_ret = np.random.normal(0.0004, 0.012, 252)
        closes = pd.DataFrame(index=dates)
        closes[index_ticker] = 450.0 * np.exp(np.cumsum(idx_ret))
        for s in symbols:
            idio = np.random.normal(0.0, 0.018, 252)
            closes[s] = 150.0 * np.exp(np.cumsum(0.8 * idx_ret + idio))

    returns = closes.pct_change().dropna(how="all")
    constituent_returns = returns[symbols]

    # Constituent weights (approximated by market-cap/equal weight for top constituents)
    weights = pd.Series(1.0 / len(symbols), index=symbols)

    # Historical Implied Volatility Proxy (1-month realized as proxy)
    constituent_ivs = constituent_returns.rolling(21).std().iloc[-1] * np.sqrt(252)
    index_iv = float(returns[index_ticker].rolling(21).std().iloc[-1] * np.sqrt(252))

    # Market premium: index implied vol typically trades rich by +2.5 to 4.0 vol points
    index_iv_market = index_iv + 0.035

    print("=" * 85)
    print(f"=== CHROMA: {args.index} EQUITY INDEX DISPERSION & CORRELATION ARBITRAGE ===")
    print(f"Index Asset: {index_ticker} | Constituent Basket: Top {len(symbols)} Stocks | Window: {args.window}d")
    print("=" * 85)

    # Calculate CBOE Implied vs Realized Correlation
    metrics = DispersionCorrelationModel.evaluate_correlation_opportunity(
        index_iv=index_iv_market,
        constituent_ivs=constituent_ivs,
        constituent_weights=weights,
        constituent_returns=constituent_returns,
        window=args.window,
    )

    print("\n--- 1. CORRELATION RISK PREMIUM METRICS ---")
    print(f"Index Implied Volatility:        {metrics.index_implied_vol:.1%}")
    print(f"Weighted Constituent Volatility:  {metrics.weighted_constituent_vol:.1%}")
    print(f"CBOE Implied Correlation (ρ_imp): {metrics.implied_correlation:.1%}")
    print(f"Realized Correlation (ρ_real):    {metrics.realized_correlation:.1%}")
    print(f"Correlation Spread (Alpha):       {metrics.correlation_spread:+.1%} (ρ_imp - ρ_real)")

    # Generate Vega-Neutral Trade Structure
    engine = DispersionTradeEngine(target_index_contracts=10.0, max_constituent_count=args.top_n)
    latest_spots = closes.iloc[-1]

    structure = engine.build_dispersion_structure(
        index_symbol=index_ticker,
        index_spot=float(latest_spots[index_ticker]),
        index_iv=metrics.index_implied_vol,
        tau_days=30.0,
        constituent_spots=latest_spots[symbols],
        constituent_ivs=constituent_ivs,
        constituent_weights=weights,
        correlation_spread=metrics.correlation_spread,
    )

    print("\n--- 2. EXECUTABLE DISPERSION STRUCTURE (30-Day Expiry) ---")
    idx_leg = structure.index_leg
    print(f"SHORT INDEX LEG: {idx_leg.side} {idx_leg.contracts:.1f}x {idx_leg.symbol} Straddles @ ${idx_leg.straddle_price:,.2f} | Vega: -${idx_leg.dollar_vega:,.2f}/vol")
    print(f"Initial Required Margin: ${structure.initial_margin_usd:,.2f}")
    print("\nLONG CONSTITUENT BASKET (VEGA-MATCHED):")
    print(f"{'Ticker':<10} {'Spot Px':<10} {'Contracts':<11} {'Straddle Px':<13} {'Dollar Vega ($/vol)':<20}")
    print("-" * 68)

    for leg in structure.constituent_legs:
        print(f"{leg.symbol:<10} ${leg.spot_price:<9.2f} {leg.contracts:<11.2f} ${leg.straddle_price:<12.2f} +${leg.dollar_vega:<19.2f}")

    print("-" * 68)
    print(f"Net Portfolio Dollar Vega: ${structure.net_dollar_vega:+,.2f}/vol (Vega Neutral)")
    print(f"Strategy Edge: When constituent stocks diverge and correlation collapses, long basket prints while short index decays.")


if __name__ == "__main__":
    main()