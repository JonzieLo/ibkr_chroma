from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import yfinance as yf
from ibkr_chroma.data.universe import UniverseManager


def fetch_single_cap(y_sym: str, ibkr_sym: str) -> Tuple[str, float]:
    """Worker function to fetch market cap with multiple property fallbacks."""
    try:
        t = yf.Ticker(y_sym)
        cap = 0.0
        try:
            cap = float(t.fast_info["market_cap"])
        except Exception:
            pass
        if not cap:
            try:
                cap = float(t.fast_info["marketCap"])
            except Exception:
                pass
        if not cap:
            try:
                cap = float(t.info.get("marketCap") or 0.0)
            except Exception:
                pass

        return ibkr_sym, cap
    except Exception as e:
        return ibkr_sym, 0.0

def sync_market_caps(cache_path: str = "data/universe/market_caps.parquet", max_workers: int = 8):
    um = UniverseManager()
    all_metadata = um.get_universe()

    # Build ticker mappings (IBKR -> Yahoo)
    us_yahoo_map = {m.symbol.replace(" ", "-"): m.symbol for m in all_metadata if m.currency == "USD"}
    hk_yahoo_map = {f"{int(m.symbol):04d}.HK": m.symbol for m in all_metadata if m.currency == "HKD"}

    combined_map = {**us_yahoo_map, **hk_yahoo_map}
    total = len(combined_map)
    caps: Dict[str, float] = {}

    print(f"Fetching market caps for {total} instruments using {max_workers} parallel workers...")

    # Parallel bulk execution
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_single_cap, y_sym, ibkr_sym): ibkr_sym
            for y_sym, ibkr_sym in combined_map.items()
        }

        for future in as_completed(futures):
            completed += 1
            ibkr_sym, cap_val = future.result()
            if cap_val > 0:
                caps[ibkr_sym] = cap_val

            # Live progress indicator
            pct = (completed / total) * 100.0
            print(f"[{completed}/{total}] ({pct:.1f}%) Processed {ibkr_sym:<6} | Cap: ${cap_val/1e9:,.1f}B", end="\r")

    print("\n" + "=" * 65)
    df_caps = pd.DataFrame(list(caps.items()), columns=["symbol", "market_cap"]).set_index("symbol")
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    df_caps.to_parquet(cache_path)

    print(f"Successfully saved {len(df_caps)} market caps to {cache_path}!")
    print("\n--- TOP 10 STOCKS BY MARKET CAP ---")
    print(df_caps.sort_values(by="market_cap", ascending=False).head(10))


if __name__ == "__main__":
    sync_market_caps()