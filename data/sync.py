import argparse
import asyncio

from ibkr_chroma.data.client import IBKRClient
from ibkr_chroma.data.pipeline import DataPipeline
from ibkr_chroma.data.universe import UniverseManager


async def main():
    parser = argparse.ArgumentParser(description="Sync IBKR historical market data to local Parquet.")
    parser.add_argument("--index", default="SP500", choices=["SP500", "QQQ", "HSI"], help="Index universe to sync")
    parser.add_argument("--top-n", type=int, default=50, help="Number of constituents to sync (default: 50)")
    parser.add_argument("--duration", default="2 Y", help="Lookback duration (e.g. '1 Y', '2 Y')")
    parser.add_argument("--port", type=int, default=7497, help="TWS API port (default: 7497)")
    parser.add_argument("--client-id", type=int, default=10, help="API Client ID")
    args = parser.parse_args()

    um = UniverseManager()
    universe = um.get_universe(indices=[args.index], top_n=args.top_n)

    if not universe:
        print(f"No constituents found for index: {args.index}")
        return

    print(f"Connecting to TWS on port {args.port}...")
    client = IBKRClient(port=args.port, client_id=args.client_id)
    await client.connect()

    try:
        pipeline = DataPipeline(client=client)
        # Sync stocks + sector ETFs (for US)
        sync_etfs = (args.index in ["SP500", "QQQ"])
        await pipeline.sync_universe(universe, duration=args.duration, bar_size="1 day", sync_etfs=sync_etfs)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())