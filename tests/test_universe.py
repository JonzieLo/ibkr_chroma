from ibkr_chroma.data.universe import UniverseManager

um = UniverseManager()

full_universe = um.get_universe()
print(f"\nTotal Multi-Index Constituents Loaded: {len(full_universe)}")

indices = [m.index for m in full_universe]
import pandas as pd
print("\nIndex Breakdown:")
print(pd.Series(indices).value_counts())
sp_symbols = um.get_symbols(indices=["SP500"], top_n=10)
print("\nTop 10 S&P 500 Tickers:", sp_symbols)
hkex_symbols = um.get_symbols(indices=["HSI"], top_n=10)
print("\nTop 10 HSI Tickers:", hkex_symbols)
qqq_symbols = um.get_symbols(indices=["QQQ"], top_n=10)
print("\nTop 10 NASDAQ-100 Tickers:", qqq_symbols)