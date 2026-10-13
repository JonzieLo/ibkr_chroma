"""
Live Paper Execustion, Portfolio Rebalance CLI.
Pulls Account NAV & positions, generates latest Stat-Arb target portfolio, places paper orders.
"""

import argparse
import asyncio
import pandas as pd
from ib_async import IB

from ibkr_chroma.data.pipeline import DataPipeline
from ibkr_chroma.data.storage import MarketDataStorage
from ibkr_chroma.data.universe import UniverseManager
from ibkr_chroma.execution.position_sync import PositionSynchronizer
from ibkr_chroma.execution.router import ExecutionRouter
from ibkr_chroma.statarb.ou_process import OUProcessModel
from ibkr_chroma.statarb.pca_factors import SectorETFFactorModel


async def main():
    parser = argparse.ArgumentParser(description="Live Paper Rebalancing Orchestrator.")
    parser.add_argument("--port", type=int, default=4002, help="IB Gateway Port (4002 paper, 7497 TWS)")
    parser.add_argument("--client-id", type=int, default=20, help="Client ID")
    parser.add_argument("--top-n", type=int, default=50, help="Universe size")
    parser.add_argument("--dry-run", action="store_true", help="Review orders without submitting")
    args = parser.parse_args()

    ib = IB()
    print(f"Connecting to IB Gateway on 127.0.0.1:{args.port}...")
    try:
        await ib.connectAsync("127.0.0.1", args.port, clientId=args.client_id)
        print("Connected successfully.")
    except Exception as e:
        print(f"[ERROR] Could not connect to IB Gateway: {e}")
        return

    try:
        sync = PositionSynchronizer(ib)
        account = await sync.get_account_snapshot()
        print(f"Account ID:           {account.account_id}")
        print(f"Net Liquidation Value: ${account.net_liquidation_usd:,.2f}")
        print(f"Cash Available:        ${account.total_cash_usd:,.2f}")
        print(f"Open Positions Count:  {len(account.current_positions)}")

        um = UniverseManager()
        symbols = um.get_symbols(indices=["SP500"], top_n=args.top_n)
        storage = MarketDataStorage()
        pipeline = DataPipeline(storage=storage)

        stock_prices, _ = pipeline.build_feature_panels(symbols, bar_size="1_day")
        etf_returns = pipeline.build_etf_panel(bar_size="1_day")

        if stock_prices.empty or etf_returns.empty:
            print("[ERROR] Local Parquet data missing. Run data.sync first.")
            return

        stock_returns = stock_prices.pct_change().dropna(how="all")
        common = stock_returns.index.intersection(etf_returns.index)
        stock_returns = stock_returns.loc[common]
        etf_returns = etf_returns.loc[common]

        # Fit Sector ETF Model to get residuals
        etf_model = SectorETFFactorModel(l1_ratio=0.5)
        _, residuals = etf_model.fit_residuals(stock_returns.iloc[-126:], etf_returns.iloc[-126:])

        # Fit O-U Model to get target weights
        ou_engine = OUProcessModel(s_open=1.25, s_close=0.50, s_stop=3.50)
        s_scores = ou_engine.calibrate_universe(residuals)
        latest_date = stock_prices.index[-1]
        latest_prices = stock_prices.iloc[-1]
        target_weights = ou_engine.update_positions(latest_date, latest_prices, s_scores)

        # Generate Orders
        router = ExecutionRouter(ib, dry_run=args.dry_run, max_order_usd=50_000.0)
        orders = router.generate_rebalance_orders(
            target_weights=target_weights,
            current_weights=account.current_weights,
            current_prices=latest_prices,
            account_nav_usd=account.net_liquidation_usd,
        )

        mode_str = "Dry-run (Simulated Review)" if args.dry_run else "Paper Submission"
        print(f"Rebalance orders: {mode_str}")
        if not orders:
            print("Portfolio is already balanced. Zero orders needed.")
        else:
            await router.execute_orders(orders)

    finally:
        ib.disconnect()
        print("\nDisconnected from IB Gateway.")


if __name__ == "__main__":
    asyncio.run(main())