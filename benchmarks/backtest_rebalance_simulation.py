"""
CHROMA: Event-Driven Realistic Portfolio Rebalance Simulation.
Simulates the exact production behavior of run_portfolio_rebalance.py through history:
- Discrete integer share orders (no fractional shares).
- Minimum trade size filters ($200 threshold).
- Continuous cash tracking, short borrow fees, and 5 bps execution slippage.
- Stateful O-U trade holding across days (take profit, stop-loss, time-stop).
"""
from __future__ import annotations

import argparse
import math
from typing import Dict, List
import numpy as np
import pandas as pd

from ibkr_chroma.data.pipeline import DataPipeline
from ibkr_chroma.data.storage import MarketDataStorage
from ibkr_chroma.data.universe import UniverseManager
from ibkr_chroma.statarb.ou_process import OUProcessModel
from ibkr_chroma.statarb.pca_factors import SectorETFFactorModel, PCAFactorModel


class SimulatedBrokerAccount:
    """Simulates an IBKR margin account with discrete share tracking and cash reconciliation."""
    def __init__(self, initial_cash: float = 100_000.0, slippage_bps: float = 5.0, borrow_rate_ann: float = 0.015):
        self.cash: float = initial_cash
        self.positions: Dict[str, int] = {}  # Symbol -> Integer shares
        self.slippage_rate: float = slippage_bps / 10_000.0
        self.daily_borrow_rate: float = borrow_rate_ann / 252.0
        self.total_commissions_paid: float = 0.0
        self.total_borrow_paid: float = 0.0

    def compute_nav(self, current_prices: pd.Series) -> float:
        stock_value = sum(self.positions.get(sym, 0) * current_prices.get(sym, 0.0) for sym in self.positions)
        return self.cash + stock_value

    def get_weights(self, current_prices: pd.Series) -> pd.Series:
        nav = self.compute_nav(current_prices)
        if nav <= 0:
            return pd.Series(0.0, index=current_prices.index)
        return pd.Series({sym: (self.positions.get(sym, 0) * current_prices.get(sym, 0.0)) / nav for sym in current_prices.index})

    def execute_order(self, symbol: str, target_shares: int, current_price: float):
        current_shares = self.positions.get(symbol, 0)
        delta_shares = target_shares - current_shares

        if delta_shares == 0:
            return

        order_usd = abs(delta_shares) * current_price
        slippage_cost = order_usd * self.slippage_rate
        self.total_commissions_paid += slippage_cost

        # Cash accounting: Buying deducts cash; Selling adds cash
        trade_cash_flow = - (delta_shares * current_price) - slippage_cost
        self.cash += trade_cash_flow

        if target_shares == 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = target_shares

    def deduct_daily_borrow_fee(self, current_prices: pd.Series):
        """Deducts daily margin borrowing cost on short positions."""
        short_liability = sum(abs(qty) * current_prices.get(sym, 0.0) for sym, qty in self.positions.items() if qty < 0)
        daily_fee = short_liability * self.daily_borrow_rate
        self.cash -= daily_fee
        self.total_borrow_paid += daily_fee


def run_realistic_simulation(
    index_name: str,
    top_n: int = 50,
    lookback: int = 126,
    initial_cash: float = 100_000.0,
    slippage_bps: float = 5.0,
    borrow_rate_ann: float = 0.015,
):
    um = UniverseManager()
    symbols = um.get_symbols(indices=[index_name], top_n=top_n)
    storage = MarketDataStorage()
    pipeline = DataPipeline(storage=storage)

    stock_prices, _ = pipeline.build_feature_panels(symbols, bar_size="1_day")
    if stock_prices.empty or len(stock_prices.columns) < 10:
        print(f"[ERROR] Insufficient Parquet bars for {index_name}. Run data.sync first.")
        return

    stock_returns = stock_prices.pct_change().dropna(how="all")

    # Load ETF panel if US, otherwise use PCA
    use_etf = index_name in ["SP500", "QQQ"]
    if use_etf:
        etf_returns = pipeline.build_etf_panel(bar_size="1_day")
        common = stock_returns.index.intersection(etf_returns.index)
        stock_returns = stock_returns.loc[common]
        stock_prices = stock_prices.loc[common]
        etf_returns = etf_returns.loc[common]
        model = SectorETFFactorModel(l1_ratio=0.5)
    else:
        model = PCAFactorModel(use_rmt=True)

    broker = SimulatedBrokerAccount(initial_cash=initial_cash, slippage_bps=slippage_bps, borrow_rate_ann=borrow_rate_ann)
    ou_engine = OUProcessModel(s_open=1.25, s_close=0.50, s_stop=3.50, max_holding_days=35)

    dates = stock_returns.index[lookback:]
    history = []

    print("=" * 85)
    print(f"=== CHROMA: REALISTIC EVENT-DRIVEN REBALANCE SIMULATION ({index_name}) ===")
    print(f"Starting Capital: ${initial_cash:,.2f} | Execution: Discrete Integer Shares | Lookback: {lookback}d")
    print("=" * 85)

    for t_idx in range(lookback, len(stock_returns)):
        curr_date = stock_returns.index[t_idx]
        window_stocks = stock_returns.iloc[t_idx - lookback : t_idx]
        curr_prices = stock_prices.iloc[t_idx]

        # 1. Fit factor model to extract residuals
        if use_etf:
            window_etfs = etf_returns.iloc[t_idx - lookback : t_idx]
            _, residuals = model.fit_residuals(window_stocks, window_etfs)
        else:
            pca_res = model.fit_transform(window_stocks)
            residuals = pca_res.residuals

        # 2. Fit O-U Model and determine target weights
        s_scores = ou_engine.calibrate_universe(residuals)
        target_weights = ou_engine.update_positions(curr_date, curr_prices, s_scores)

        # 3. Router logic: calculate discrete integer share orders
        account_nav = broker.compute_nav(curr_prices)
        current_weights = broker.get_weights(curr_prices)

        for sym in curr_prices.index:
            w_target = float(target_weights.get(sym, 0.0))
            w_curr = float(current_weights.get(sym, 0.0))
            px = float(curr_prices.get(sym, 0.0))

            if px <= 0.0:
                continue

            delta_usd = (w_target - w_curr) * account_nav

            # Router Rule: Skip micro-trades under $200
            if abs(delta_usd) < 200.0:
                continue

            target_usd = w_target * account_nav
            # Discrete integer share rounding (no fractions!)
            target_shares = int(round(target_usd / px))

            # Router Rule: Max $50k ceiling per position
            if abs(target_shares * px) > 50_000.0:
                target_shares = int(np.sign(target_shares) * (50_000.0 / px))

            broker.execute_order(sym, target_shares, px)

        # 4. Overnight borrow fee deduction on short stock liabilities
        broker.deduct_daily_borrow_fee(curr_prices)

        # 5. End-of-day Mark to Market
        eod_nav = broker.compute_nav(curr_prices)
        long_val = sum(qty * curr_prices[s] for s, qty in broker.positions.items() if qty > 0)
        short_val = sum(qty * curr_prices[s] for s, qty in broker.positions.items() if qty < 0)

        history.append({
            "date": curr_date,
            "nav": eod_nav,
            "cash": broker.cash,
            "long_value": long_val,
            "short_value": short_val,
            "active_positions": len(broker.positions),
        })

    df_res = pd.DataFrame(history).set_index("date")
    returns_series = df_res["nav"].pct_change().dropna()

    ann_ret = returns_series.mean() * 252.0
    ann_vol = returns_series.std() * math.sqrt(252.0)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum_ret = (df_res["nav"].iloc[-1] - initial_cash) / initial_cash
    max_dd = ((df_res["nav"] - df_res["nav"].cummax()) / df_res["nav"].cummax()).min()

    trades = ou_engine.closed_trades
    n_trades = len(trades)
    win_rate = (sum(t.pnl_pct > 0 for t in trades) / n_trades) if n_trades > 0 else 0.0
    avg_hold = float(np.mean([t.days_held for t in trades])) if n_trades > 0 else 0.0

    print("\n--- REALISTIC PRODUCTION BACKTEST PERFORMANCE ---")
    print(f"Ending Account NAV:           ${df_res['nav'].iloc[-1]:,.2f} (from ${initial_cash:,.2f})")
    print(f"Cumulative Net Return:        {cum_ret:+,.2%}")
    print(f"Annualized Net Return:        {ann_ret:+,.2%}")
    print(f"Annualized Volatility:        {ann_vol:,.2%}")
    print(f"Sharpe Ratio (Net of Drag):   {sharpe:+,.2f}")
    print(f"Maximum Portfolio Drawdown:   {max_dd:,.2%}")
    print("-" * 65)
    print(f"Total Slippage Incurred:      ${broker.total_commissions_paid:,.2f}")
    print(f"Total Short Borrow Paid:      ${broker.total_borrow_paid:,.2f}")
    print(f"Total Completed Roundtrips:   {n_trades}")
    print(f"Win Rate on Closed Trades:    {win_rate:+,.2%}")
    print(f"Average Trade Holding Period: {avg_hold:.1f} days")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Run Production Rebalancer Simulation.")
    parser.add_argument("--index", default="SP500", choices=["SP500", "QQQ", "HSI"], help="Target index")
    parser.add_argument("--top-n", type=int, default=50, help="Universe size (default: 50)")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Initial cash balance")
    parser.add_argument("--lookback", type=int, default=126, help="Calibration lookback window")
    args = parser.parse_args()

    run_realistic_simulation(
        index_name=args.index,
        top_n=args.top_n,
        lookback=args.lookback,
        initial_cash=args.capital,
    )


if __name__ == "__main__":
    main()