import argparse
import math
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from ibkr_chroma.data.pipeline import DataPipeline
from ibkr_chroma.data.storage import MarketDataStorage
from ibkr_chroma.data.universe import UniverseManager
from ibkr_chroma.statarb.ou_process import OUProcessModel
from ibkr_chroma.statarb.pca_factors import PCAFactorModel, SectorETFFactorModel


def generate_benchmark_data(symbols: list[str], days: int = 504) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic stock and sector ETF panels for verification."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="B")
    np.random.seed(42)

    # Simulate 4 Sector ETFs + SPY
    etf_names = ["SPY", "XLK", "XLF", "XLV", "XLE"]
    etf_rets = np.random.normal(0.0004, 0.012, (days, len(etf_names)))
    etf_df = pd.DataFrame(etf_rets, index=dates, columns=etf_names)

    # Simulate Stocks driven by ETF exposures + mean-reverting idiosyncratic noise
    stock_data = {}
    for i, s in enumerate(symbols):
        # Sparse loading on 1-2 ETFs
        betas = np.zeros(len(etf_names))
        betas[0] = np.random.uniform(0.6, 1.2)  # SPY beta
        primary_sector = (i % (len(etf_names) - 1)) + 1
        betas[primary_sector] = np.random.uniform(0.5, 1.1)

        # O-U residual simulation
        # dX = -kappa * X dt + sigma * dW
        kappa = np.random.uniform(0.05, 0.15)  # Reverts in 5 to 15 days
        sigma = 0.018
        x = np.zeros(days)
        for t in range(1, days):
            x[t] = x[t - 1] - kappa * x[t - 1] + sigma * np.random.normal()

        # Stock return = Beta * ETF returns + delta X
        delta_x = np.diff(x, prepend=0.0)
        systemic_ret = etf_rets @ betas
        total_ret = systemic_ret + delta_x
        stock_data[s] = 100.0 * np.exp(np.cumsum(total_ret))

    return pd.DataFrame(stock_data, index=dates), etf_df


def run_single_simulation(
    mode: str,
    stock_returns: pd.DataFrame,
    stock_prices: pd.DataFrame,
    etf_returns: Optional[pd.DataFrame] = None,
    lookback: int = 126,
    slippage_bps: float = 5.0,
    borrow_rate_ann: float = 0.015,
) -> Dict[str, any]:
    """Runs a rolling stat-arb backtest for either PCA or ETF model."""
    pca_model = PCAFactorModel(use_rmt=True)
    etf_model = SectorETFFactorModel(l1_ratio=0.5)
    ou_engine = OUProcessModel(s_open=1.25, s_close=0.50, s_stop=3.50, max_holding_days=35)

    dates = stock_returns.index[lookback:]
    portfolio_weights_list = []

    daily_borrow_cost = borrow_rate_ann / 252.0
    slippage_rate = slippage_bps / 10_000.0

    prev_weights = pd.Series(0.0, index=stock_returns.columns)
    strategy_daily_returns = []

    for t_idx in range(lookback, len(stock_returns)):
        curr_date = stock_returns.index[t_idx]
        window_stocks = stock_returns.iloc[t_idx - lookback : t_idx]
        curr_prices = stock_prices.iloc[t_idx]

        # Extract Residuals
        if mode == "PCA_RMT":
            pca_res = pca_model.fit_transform(window_stocks)
            residuals = pca_res.residuals
        else:  # ETF_ELASTICNET
            window_etfs = etf_returns.iloc[t_idx - lookback : t_idx]
            _, residuals = etf_model.fit_residuals(window_stocks, window_etfs)

        s_scores = ou_engine.calibrate_universe(residuals)
        target_weights = ou_engine.update_positions(curr_date, curr_prices, s_scores)
        target_weights = target_weights.reindex(stock_returns.columns).fillna(0.0)

        # Calculate P&L: Asset return + Slippage Drag + Short Borrow Drag
        day_stock_returns = stock_returns.iloc[t_idx]
        gross_return = float((prev_weights * day_stock_returns).sum())

        # Turnover slippage
        turnover = float(np.sum(np.abs(target_weights - prev_weights)))
        slippage_drag = turnover * slippage_rate

        # Short borrow cost on negative weights
        short_exposure = float(np.sum(np.abs(prev_weights[prev_weights < 0])))
        borrow_drag = short_exposure * daily_borrow_cost

        net_return = gross_return - slippage_drag - borrow_drag
        strategy_daily_returns.append(net_return)

        prev_weights = target_weights

    ret_series = pd.Series(strategy_daily_returns, index=dates)

    # Performance Analytics
    mean_ann = ret_series.mean() * 252.0
    vol_ann = ret_series.std() * math.sqrt(252.0)
    sharpe = mean_ann / vol_ann if vol_ann > 0 else 0.0

    cum_pnl = (1.0 + ret_series).cumprod()
    max_dd = ((cum_pnl - cum_pnl.cummax()) / cum_pnl.cummax()).min()

    trades = ou_engine.closed_trades
    n_trades = len(trades)
    win_rate = (np.sum([t.pnl_pct > 0 for t in trades]) / n_trades) if n_trades > 0 else 0.0
    avg_hold = float(np.mean([t.days_held for t in trades])) if n_trades > 0 else 0.0
    stops_hit = np.sum([t.exit_reason == "STOP_LOSS" for t in trades])

    return {
        "mode": mode,
        "sharpe": sharpe,
        "annualized_return": mean_ann,
        "annualized_vol": vol_ann,
        "max_drawdown": max_dd,
        "cumulative_net": float(cum_pnl.iloc[-1] - 1.0),
        "total_trades": n_trades,
        "win_rate": win_rate,
        "avg_holding_days": avg_hold,
        "stop_loss_rate": (stops_hit / n_trades) if n_trades > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Avellaneda & Lee Stat-Arb Backtest.")
    parser.add_argument("--index", default="SP500", choices=["SP500", "QQQ", "HSI"], help="Target index")
    parser.add_argument("--top-n", type=int, default=50, help="Universe size (default: 50)")
    parser.add_argument("--days", type=int, default=504, help="Historical trading days (default: 504 = 2 Years)")
    parser.add_argument("--lookback", type=int, default=126, help="Rolling estimation window (default: 126)")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="One-way slippage in bps (default: 5.0)")
    args = parser.parse_args()

    if args.slippage_bps is None:
        slippage_bps = 15.0 if args.index == "HSI" else 5.0  # Accounts for HK 0.10% stamp duty
    else:
        slippage_bps = args.slippage_bps

    um = UniverseManager()
    symbols = um.get_symbols(indices=["SP500"], top_n=args.top_n)
    storage = MarketDataStorage()
    pipeline = DataPipeline(storage=storage)

    stock_prices, _ = pipeline.build_feature_panels(symbols, bar_size="1_day")
    if stock_prices.empty or len(stock_prices.columns) < 10:
        print(f"[ERROR] Insufficient Parquet data for {args.index}. Run `data.sync --index {args.index}` first.")
        return

    stock_returns = stock_prices.pct_change().dropna(how="all")
    if args.index in ["SP500", "QQQ"]:
        etf_returns = pipeline.build_etf_panel(bar_size="1_day")
        if not etf_returns.empty:
            common = stock_returns.index.intersection(etf_returns.index)
            stock_returns = stock_returns.loc[common]
            stock_prices = stock_prices.loc[common]
            etf_returns = etf_returns.loc[common]
        else:
            etf_returns = None
    else:
        etf_returns = None  # US sector ETFs do not trade during Hong Kong market hours

    print("=== AVELLANEDA & LEE (2010): PCA vs. SECTOR ETF STAT-ARB BENCHMARK ===")
    print(f"Universe: {len(stock_prices.columns)} Stocks | Lookback: {args.lookback}d | Slippage: {args.slippage_bps} bps | Borrow: 1.5%/yr")

    # Run Mode A: Statistical PCA with Marchenko-Pastur RMT
    print("\n[1/2] Replaying Mode A: PCA with Marchenko-Pastur RMT Factor Selection...")
    res_pca = run_single_simulation(
        "PCA_RMT", stock_returns, stock_prices, lookback=args.lookback, slippage_bps=args.slippage_bps
    )

    # Run Mode B: Sector ETF ElasticNet Sparse Model (Section 3)
    if etf_returns is not None:
        print("[2/2] Replaying Mode B: Sector ETF ElasticNet Sparse Factor Model...")
        res_etf = run_single_simulation(
            "ETF_ELASTICNET", stock_returns, stock_prices, etf_returns=etf_returns, lookback=args.lookback, slippage_bps=args.slippage_bps
        )
    else:
        res_etf = None

    # Print Comparative Tearsheet
    print("--- STAT-ARB RESEARCH COMPARISON: PCA vs. SECTOR ETF ---")
    print(f"{'Performance Metric':<30} {'Mode A: PCA (RMT)':<24} {'Mode B: Sector ETF (Lasso)':<24}")

    def _fmt_pct(v): return f"{v:+.2%}"
    def _fmt_f(v): return f"{v:+.2f}"

    metrics = [
        ("Annualized Net Return", res_pca["annualized_return"], res_etf["annualized_return"] if res_etf else 0.0, _fmt_pct),
        ("Annualized Volatility", res_pca["annualized_vol"], res_etf["annualized_vol"] if res_etf else 0.0, _fmt_pct),
        ("Sharpe Ratio (Net of Fees)", res_pca["sharpe"], res_etf["sharpe"] if res_etf else 0.0, _fmt_f),
        ("Maximum Drawdown", res_pca["max_drawdown"], res_etf["max_drawdown"] if res_etf else 0.0, _fmt_pct),
        ("Cumulative Net P&L", res_pca["cumulative_net"], res_etf["cumulative_net"] if res_etf else 0.0, _fmt_pct),
        ("Total Closed Trades", res_pca["total_trades"], res_etf["total_trades"] if res_etf else 0, lambda v: f"{int(v)}"),
        ("Trade Win Rate", res_pca["win_rate"], res_etf["win_rate"] if res_etf else 0.0, _fmt_pct),
        ("Avg Holding Period", res_pca["avg_holding_days"], res_etf["avg_holding_days"] if res_etf else 0.0, lambda v: f"{v:.1f} days"),
        ("Stop-Loss Trigger Rate", res_pca["stop_loss_rate"], res_etf["stop_loss_rate"] if res_etf else 0.0, _fmt_pct),
    ]

    for label, v_pca, v_etf, formatter in metrics:
        s_pca = formatter(v_pca)
        s_etf = formatter(v_etf) if res_etf else "N/A"
        print(f"{label:<30} {s_pca:<24} {s_etf:<24}")


if __name__ == "__main__":
    main()