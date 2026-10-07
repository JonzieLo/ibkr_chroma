import asyncio
from typing import List
import pandas as pd

from ibkr_chroma.data.client import IBKRClient
from ibkr_chroma.data.storage import MarketDataStorage
from ibkr_chroma.data.universe import get_universe_symbols


class DataPipeline:
    def __init__(self, client: IBKRClient, storage: MarketDataStorage = None):
        self.client = client
        self.storage = storage or MarketDataStorage()

    async def sync_universe(self, symbols: List[str], duration: str = "2 Y", bar_size: str = "1 day") -> None:
        for symbol in symbols:
            if not self.storage.has_data(symbol, bar_size):
                print(f"Fetching {symbol} from IBKR ({duration})...")
                df = await self.client.fetch_daily_bars(symbol, duration=duration, bar_size=bar_size)
                self.storage.save_bars(symbol, df, bar_size=bar_size)
                await asyncio.sleep(0.5)

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

        closes_df = pd.DataFrame(closes).sort_index().ffill().dropna()
        volumes_df = pd.DataFrame(volumes).sort_index().ffill().dropna()
        return closes_df, volumes_df