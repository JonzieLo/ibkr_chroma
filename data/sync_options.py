import argparse
import asyncio
from pathlib import Path
import pandas as pd
from ib_async import Stock, Index

from ibkr_chroma.data.client import IBKRClient
from ibkr_chroma.data.storage import MarketDataStorage
from ibkr_chroma.data.universe import UniverseManager, EquityMetadata


# Index and Top Constituent Mapping
INDEX_ROUTING = {
    "SP500": {"symbol": "SPY",  "sec_type": "STK", "exchange": "SMART", "currency": "USD"},
    "QQQ":   {"symbol": "QQQ",  "sec_type": "STK", "exchange": "SMART", "currency": "USD"},
    "HSI":   {"symbol": "2800", "sec_type": "STK", "exchange": "SEHK",  "currency": "HKD"},
}

class OptionDataSynchronizer:
    def __init__(self, client: IBKRClient, storage: MarketDataStorage = None):
        self.client = client
        self.storage = storage or MarketDataStorage()

    async def fetch_iv_bars(
        self,
        symbol: str,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
        duration: str = "1 Y",
    ) -> pd.DataFrame:
        """Pulls real daily implied volatility bars calculated by IBKR's volatility server."""
        if sec_type == "IND":
            contract = Index(symbol, exchange, currency)
        else:
            contract = Stock(symbol, exchange, currency)

        await self.client.ib.qualifyContractsAsync(contract)
        bars = await self.client.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting="1 day",
            whatToShow="OPTION_IMPLIED_VOLATILITY",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame([{
            "date": b.date,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,  # Real ATM Implied Volatility (annualized decimal e.g. 0.25)
            "volume": b.volume,
        } for b in bars])

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    async def sync_dispersion_iv(self, index_name: str, top_n: int = 10, duration: str = "1 Y"):
        specs = INDEX_ROUTING.get(index_name.upper())
        if not specs:
            raise ValueError(f"Unknown index {index_name}")
        um = UniverseManager()
        leaders = um.get_top_constituents(index=index_name, top_n=top_n, by="market_cap")
        
        # Index + Top N Market Cap Leaders
        index_meta = EquityMetadata(
            symbol=specs["symbol"],
            name=f"{index_name} Index Benchmark",
            sector="Index",
            sub_industry="ETF",
            index=index_name,
            currency=specs["currency"],
            exchange=specs["exchange"],
        )
        all_targets = [index_meta] + leaders

        total = len(all_targets)
        print(f"=== Option IV Sync ({index_name.upper()}) ===")
        print(f"Index Asset: {specs['symbol']} | Dynamically Selected Top {len(leaders)} Leaders by Market Cap:")
        print(" -> " + ", ".join([f"{m.symbol}" for m in leaders]))

        for idx, item in enumerate(all_targets, 1):
            iv_file = Path(f"data/parquet/{item.symbol.upper()}_iv_1_day.parquet")
            if not iv_file.exists():
                print(f"[{idx}/{total}] Fetching real IV for {item.symbol} ({item.exchange}/{item.currency})...")
                try:
                    df = await self.fetch_iv_bars(
                        symbol=item.symbol,
                        sec_type="STK",
                        exchange=item.exchange,
                        currency=item.currency,
                        duration=duration,
                    )
                    if not df.empty:
                        df.to_parquet(iv_file, compression="snappy")
                        print(f"  Saved {len(df)} IV bars for {item.symbol} (Latest IV: {df['close'].iloc[-1]:.1%})")
                    else:
                        print(f"  [WARN] No IV bars returned for {item.symbol}")
                except Exception as e:
                    print(f"  [ERROR] Failed to fetch IV for {item.symbol}: {e}")

                await asyncio.sleep(0.5)
            else:
                print(f"[{idx}/{total}] Cached: {item.symbol} IV already exists on disk.")
        print("\nSync complete.")


async def main():
    parser = argparse.ArgumentParser(description="Sync Real Option Implied Volatility Bars.")
    parser.add_argument("--index", default="HSI", choices=["SP500", "QQQ", "HSI"], help="Index universe")
    parser.add_argument("--top-n", type=int, default=500, help="Number of constituents to sync (default: 50)")
    parser.add_argument("--port", type=int, default=7497, help="TWS API port")
    parser.add_argument("--duration", default="2 Y", help="Lookback duration")
    args = parser.parse_args()

    client = IBKRClient(port=args.port, client_id=30)
    await client.connect()
    try:
        syncer = OptionDataSynchronizer(client)
        await syncer.sync_dispersion_iv(args.index, top_n=args.top_n, duration=args.duration)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())