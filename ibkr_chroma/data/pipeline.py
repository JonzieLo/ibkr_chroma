import asyncio
from typing import List, Union, Optional
import pandas as pd

from ibkr_chroma.data.client import IBKRClient
from ibkr_chroma.data.storage import MarketDataStorage
from ibkr_chroma.data.universe import UniverseManager, EquityMetadata

CORE_SECTOR_ETFS = [
    EquityMetadata("SPY",   "SPDR S&P 500 ETF",             "Index",                    "Index",    "SP500", "USD", "SMART"),
    EquityMetadata("XLK",   "Technology Select Sector",     "Information Technology",   "ETF",      "SP500", "USD", "SMART"),
    EquityMetadata("XLF",   "Financial Select Sector",      "Financials",               "ETF",      "SP500", "USD", "SMART"),
    EquityMetadata("XLV",   "Health Care Select Sector",    "Health Care",              "ETF",      "SP500", "USD", "SMART"),
    EquityMetadata("XLE",   "Energy Select Sector",         "Energy",                   "ETF",      "SP500", "USD", "SMART"),
    EquityMetadata("XLI",   "Industrial Select Sector",     "Industrials",              "ETF",      "SP500", "USD", "SMART"),
    EquityMetadata("XLC",   "Communication Services",       "Communication Services",   "ETF",      "SP500", "USD", "SMART"),
    EquityMetadata("XLY",   "Consumer Discretionary",       "Consumer Discretionary",   "ETF",      "SP500", "USD", "SMART"),
    EquityMetadata("XLP",   "Consumer Staples",             "Consumer Staples",         "ETF",      "SP500", "USD", "SMART"),
    EquityMetadata("XLU",   "Utilities Select Sector",      "Utilities",                "ETF",      "SP500", "USD", "SMART"),
    EquityMetadata("XLB",   "Materials Select Sector",      "Materials",                "ETF",      "SP500", "USD", "SMART"),
    EquityMetadata("XLRE",  "Real Estate Select Sector",    "Real Estate",              "ETF",      "SP500", "USD", "SMART"),
]


class DataPipeline:
    def __init__(self, client: Optional[IBKRClient] = None, storage: MarketDataStorage = None):
        self.client = client
        self.storage = storage or MarketDataStorage()

    async def sync_universe(
        self,
        universe: Union[List[EquityMetadata], List[str]],
        duration: str = "2 Y",
        bar_size: str = "1 day",
        sync_etfs: bool = True,
    ) -> None:
        targets: List[EquityMetadata] = []
        for item in universe:
            if isinstance(item, str):
                targets.append(EquityMetadata(item, item, "Equity", "N/A", "SP500", "USD", "SMART"))
            else:
                targets.append(item)

        # Include Sector ETFs for US indices
        if sync_etfs and any(t.index in ["SP500", "QQQ"] for t in targets):
            for etf in CORE_SECTOR_ETFS:
                if not any(t.symbol == etf.symbol for t in targets):
                    targets.append(etf)

        total = len(targets)
        print(f"Syncing {total} instruments to Parquet storage...")

        for idx, item in enumerate(targets, 1):
            if not self.storage.has_data(item.symbol, bar_size):
                print(f"[{idx}/{total}] Fetching {item.symbol} ({item.exchange}/{item.currency}) from IBKR...")
                try:
                    df = await self.client.fetch_daily_bars(
                        symbol=item.symbol,
                        exchange=item.exchange,
                        currency=item.currency,
                        duration=duration,
                        bar_size=bar_size,
                    )
                    if df is not None and not df.empty:
                        self.storage.save_bars(item.symbol, df, bar_size=bar_size)
                    else:
                        print(f"  [WARN] No bars returned for {item.symbol}")
                except Exception as e:
                    print(f"  [ERROR] Failed to fetch {item.symbol}: {e}")
                await asyncio.sleep(0.5)
            else:
                print(f"[{idx}/{total}] Cached: {item.symbol} already exists on disk.")
        print("Done.")


    def build_feature_panels(self, symbols: List[str], bar_size: str = "1 day") -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Builds aligned Dataframes:
        - closes_panel: Rows = Dates, Columns = Symbols
        - volumes_panel: Rows = Dates, Columns = Symbols
        """
        closes = {}
        volumes = {}
        for s in symbols:
            df = self.storage.load_bars(s, bar_size=bar_size)
            if not df.empty:
                closes[s] = df["close"]
                volumes[s] = df["volume"]

        closes_df = pd.DataFrame(closes).sort_index().ffill().dropna(how="all")
        volumes_df = pd.DataFrame(volumes).sort_index().ffill().dropna(how="all")
        return closes_df, volumes_df


    def build_etf_panel(self, bar_size: str = "1_day") -> pd.DataFrame:
        etf_symbols = [etf.symbol for etf in CORE_SECTOR_ETFS]
        etf_closes = {}
        for s in etf_symbols:
            df = self.storage.load_bars(s, bar_size=bar_size)
            if not df.empty and "close" in df.columns:
                etf_closes[s] = df["close"]

        if not etf_closes:
            return pd.DataFrame()

        closes_df = pd.DataFrame(etf_closes).sort_index().ffill().dropna(how="all")
        return closes_df.pct_change().dropna(how="all")