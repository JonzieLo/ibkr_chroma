import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class EquityMetadata:
    symbol: str
    name: str
    sector: str          # GICS Sector
    sub_industry: str    # GICS Sub-Industry
    index: str           # SP500, QQQ, etc.
    currency: str
    exchange: str


class UniverseManager:
    def __init__(self, cache_dir: str = "data/universe"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "multi_index_constituents.parquet"

    @staticmethod
    def normalize_ibkr_symbol(symbol: str, index: str) -> str:
        s = str(symbol).strip()
        if index == "HKEX":
            digits = re.sub(r"\D", "", s)
            if digits:
                return str(int(digits))
            return s
        return s.replace(".", " ")

    @staticmethod
    def fetch_sp500() -> pd.DataFrame:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        raw_df = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})[0]
        
        df = pd.DataFrame()
        df["symbol"] = raw_df["Symbol"].apply(lambda s: UniverseManager.normalize_ibkr_symbol(s, "SP500"))
        df["name"] = raw_df["Security"]
        df["sector"] = raw_df["GICS Sector"]
        df["sub_industry"] = raw_df["GICS Sub-Industry"]
        df["index"] = "SP500"
        df["currency"] = "USD"
        df["exchange"] = "SMART"
        return df
    
    @staticmethod
    def fetch_nasdaq100() -> pd.DataFrame:
        url = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
        tables = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})
        
        raw_df = None
        for t in tables:
            cols_lower = [str(c).lower() for c in t.columns]
            if any("ticker" in c for c in cols_lower) and any("company" in c for c in cols_lower):
                raw_df = t
                break
        
        if raw_df is None:
            raw_df = tables[0]

        ticker_col = next(c for c in raw_df.columns if "ticker" in str(c).lower())
        company_col = next(c for c in raw_df.columns if "company" in str(c).lower())
        sector_col = next((c for c in raw_df.columns if "sector" in str(c).lower() or "industry" in str(c).lower()), None)
        sub_col = next((c for c in raw_df.columns if "subsector" in str(c).lower() or "sub-industry" in str(c).lower()), None)

        df = pd.DataFrame()
        df["symbol"] = raw_df[ticker_col].apply(lambda s: UniverseManager.normalize_ibkr_symbol(s, "QQQ"))
        df["name"] = raw_df[company_col]
        df["sector"] = raw_df[sector_col] if sector_col else "Technology"
        df["sub_industry"] = raw_df[sub_col] if sub_col else "N/A"
        df["index"] = "QQQ"
        df["currency"] = "USD"
        df["exchange"] = "SMART"
        return df

    @staticmethod
    def fetch_hkex_hsi() -> pd.DataFrame:
        url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
        tables = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})
        
        raw_df = None
        for t in tables:
            cols_lower = [str(c).lower() for c in t.columns]
            if any("ticker" in c or "code" in c for c in cols_lower) and any("name" in c or "company" in c for c in cols_lower):
                raw_df = t
                break

        if raw_df is None:
            raw_df = tables[1]

        ticker_col = next(c for c in raw_df.columns if "ticker" in str(c).lower() or "code" in str(c).lower())
        name_col = next(c for c in raw_df.columns if "name" in str(c).lower() or "company" in str(c).lower())
        sector_col = next((c for c in raw_df.columns if "sub-index" in str(c).lower() or "sector" in str(c).lower()), None)

        df = pd.DataFrame()
        df["symbol"] = raw_df[ticker_col].apply(lambda s: UniverseManager.normalize_ibkr_symbol(s, "HKEX"))
        df["name"] = raw_df[name_col]
        df["sector"] = raw_df[sector_col] if sector_col else "Hang Seng Equity"
        df["sub_industry"] = "N/A"
        df["index"] = "HSI"
        df["currency"] = "HKD"
        df["exchange"] = "SEHK"
        return df

    def fetch_all_constituents(self, force_refresh: bool = False) -> pd.DataFrame:
        now = datetime.now(timezone.utc)

        if self._cache_file.exists() and not force_refresh:
            try:
                df = pd.read_parquet(self._cache_file)
                cache_age_days = (now - pd.to_datetime(df["cached_at"].iloc[0])).days
                if cache_age_days < 30:
                    return df
            except Exception:
                pass

        print("Fetching live constituents for SP500, NASDAQ-100, and HKEX...")
        sp500_df = self.fetch_sp500()
        qqq_df = self.fetch_nasdaq100()
        hkex_df = self.fetch_hkex_hsi()

        combined_df = pd.concat([sp500_df, qqq_df, hkex_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["symbol", "index"])
        combined_df["cached_at"] = now.isoformat()

        combined_df.to_parquet(self._cache_file, index=False)
        print(f"Cached {len(combined_df)} total constituents to {self._cache_file}")
        return combined_df

    def get_universe(
        self,
        indices: Optional[List[str]] = None,
        sectors: Optional[List[str]] = None,
        top_n: Optional[int] = None,
    ) -> List[EquityMetadata]:
        df = self.fetch_all_constituents()

        if indices:
            df = df[df["index"].isin(indices)]
        if sectors:
            df = df[df["sector"].isin(sectors)]
        if top_n:
            df = df.iloc[:top_n]

        return [
            EquityMetadata(
                symbol=row["symbol"],
                name=row["name"],
                sector=row["sector"],
                sub_industry=row["sub_industry"],
                index=row["index"],
                currency=row["currency"],
                exchange=row["exchange"]
            )
            for _, row in df.iterrows()
        ]

    def get_symbols(self, indices: Optional[List[str]] = None, top_n: Optional[int] = None) -> List[str]:
        return [meta.symbol for meta in self.get_universe(indices=indices, top_n=top_n)]

    def get_sector_map(self, indices: Optional[List[str]] = None, top_n: Optional[int] = None) -> Dict[str, str]:
        universe = self.get_universe(indices=indices, top_n=top_n)
        return {meta.symbol: meta.sector for meta in universe}