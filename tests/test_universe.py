from ibkr_chroma.data.universe import UniverseManager

um = UniverseManager()

full_universe = um.get_universe()
print(f"\nTotal Multi-Index Constituents Loaded: {len(full_universe)}")

indices = [m.index for m in full_universe]
import pandas as pd
print("\nIndex Breakdown:")
print(pd.Series(indices).value_counts())
hkex_symbols = um.get_symbols(indices=["HKEX"], top_n=10)
print("\nTop 10 HKEX Tickers:", hkex_symbols)
qqq_symbols = um.get_symbols(indices=["QQQ"], top_n=10)
print("\nTop 10 NASDAQ-100 Tickers:", qqq_symbols)