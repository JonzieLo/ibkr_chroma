from pathlib import Path
import pandas as pd


class MarketDataStorage:
    """
    Local Parquet Historical Market Data Cache
    """
    def __init__(self, data_dir: str = "data/parquet"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, symbol: str, bar_size: str) -> Path:
        clean_size = bar_size.replace(" ", "_").lower()
        return self.data_dir / f"{symbol.upper()}_{clean_size}.parquet"

    def has_data(self, symbol: str, bar_size: str = "1_day") -> bool:
        return self._file_path(symbol, bar_size).exists()

    def save_bars(self, symbol: str, df: pd.DataFrame, bar_size: str = "1_day") -> None:
        if df.empty:
            return
        clean_df = df.copy()
        if "date" in clean_df.columns:
            clean_df["date"] = pd.to_datetime(clean_df["date"])
            clean_df.set_index("date", inplace=True)
        path = self._file_path(symbol, bar_size)
        clean_df.to_parquet(path, compression="snappy")

    def load_bars(self, symbol: str, bar_size: str = "1_day") -> pd.DataFrame:
        path = self._file_path(symbol, bar_size)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)