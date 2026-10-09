import numpy as np
import pandas as pd

from ibkr_chroma.data.pipeline import DataPipeline
from ibkr_chroma.data.storage import MarketDataStorage
from ibkr_chroma.data.universe import UniverseManager
from ibkr_chroma.factors.engine import FactorEngine
from ibkr_chroma.factors.momentum import momentum
from ibkr_chroma.factors.volatility import realized_vol
from ibkr_chroma.portfolio.optimizer import BarraPortfolioOptimizer
from ibkr_chroma.portfolio.risk_barra import BarraRiskModel
from ibkr_chroma.portfolio.risk_slides import RiskSlidesEngine


def main():
    um = UniverseManager()
    symbols = um.get_symbols(indices=["SP500"], top_n=50)
    sector_map = um.get_sector_map(indices=["SP500"], top_n=50)
    storage = MarketDataStorage()
    pipeline = DataPipeline(storage=storage)

    # Load real S&P 500 cached data + SPY Benchmark
    closes, _ = pipeline.build_feature_panels(symbols + ["SPY"], bar_size="1_day")
    returns = closes.pct_change().dropna(how="all")

    spy_returns = returns["SPY"]
    stock_returns = returns.drop(columns=["SPY"])
    stock_symbols = [s for s in symbols if s in stock_returns.columns]
    stock_returns = stock_returns[stock_symbols]

    # Compute Historical 63-Day Market Betas
    rolling_cov = stock_returns.rolling(63).cov(spy_returns)
    rolling_var = spy_returns.rolling(63).var()
    real_betas = rolling_cov.iloc[-1].div(rolling_var.iloc[-1]).fillna(1.0)

    # Compute Composite Alpha
    f_mom = momentum(stock_returns, lookback=252, skip=21)
    f_vol = realized_vol(stock_returns, window=63)
    composite_alpha = FactorEngine.build_composite_signal(
        {"Momentum": f_mom, "LowVol": f_vol}, sector_map=sector_map
    )
    latest_alpha = composite_alpha.iloc[-1].dropna()

    # Calibrate Barra Risk Model
    active_stocks = latest_alpha.index
    barra = BarraRiskModel(half_life_days=63)
    caps = pd.Series(np.random.uniform(20e9, 2e12, len(active_stocks)), index=active_stocks)
    sector_series = pd.Series(sector_map).reindex(active_stocks)
    style_dict = {
        "Momentum": f_mom.iloc[-1].reindex(active_stocks),
        "LowVol": f_vol.iloc[-1].reindex(active_stocks),
    }

    X = barra.build_exposure_matrix(style_dict, sector_series, caps)
    for d in stock_returns.index[-60:]:
        d_ret = stock_returns.loc[d].reindex(active_stocks)
        f_t, u_t = barra.cross_sectional_wls_regression(d_ret, X, caps)
        barra.update_historical_step(f_t, u_t)

    F_cov, spec_var = barra.compute_covariance_matrices()

    # Run Barra Optimization
    optimizer = BarraPortfolioOptimizer(
        risk_aversion=2.0,
        gross_leverage=2.0,
        max_position_weight=0.06,
        max_style_exposure=0.15,
    )
    opt_result = optimizer.optimize(
        alpha_scores=latest_alpha,
        exposure_matrix=X,
        factor_covariance=F_cov,
        specific_variances=spec_var,
        betas=real_betas.reindex(active_stocks),
    )

    # Barra Risk Forecast Decomposition
    risk_forecast = barra.forecast_portfolio_risk(opt_result.weights, X)
    factor_pct = (risk_forecast.factor_risk_annualized ** 2) / (risk_forecast.total_risk_annualized ** 2)
    spec_pct = (risk_forecast.specific_risk_annualized ** 2) / (risk_forecast.total_risk_annualized ** 2)

    print("=== BARRA-OPTIMIZED LONG/SHORT EQUITY BOOK ($10M NAV) ===")
    print("=" * 80)
    print(f"Status:                      {opt_result.status_message}")
    print(f"Net Dollar Exposure:         ${opt_result.net_exposure:+,.2f} (Exact Dollar Neutral)")
    portfolio_beta = float(np.dot(opt_result.weights.values, real_betas.reindex(active_stocks).values))
    print(f"Net Portfolio Market Beta:   {portfolio_beta:+.4f} (Strict Beta Neutral)")
    print(f"Active Position Count:       {int(np.sum(opt_result.weights != 0))} / {len(active_stocks)}")
    print("-" * 80)
    print(f"Total Forecast Risk:         {risk_forecast.total_risk_annualized:.2%}")
    print(f"  ├── Systematic Factor Risk:{risk_forecast.factor_risk_annualized:.2%} ({factor_pct:.1%} of variance)")
    print(f"  └── Idiosyncratic Risk:    {risk_forecast.specific_risk_annualized:.2%} ({spec_pct:.1%} of variance)")

    print("\n--- NET FACTOR LOADINGS (w^T * X) ---")
    print(risk_forecast.factor_exposures.round(4).to_string())

    slides = RiskSlidesEngine()
    spot_slides_df = slides.generate_macro_spot_slides(
        opt_result.weights, real_betas.reindex(active_stocks), portfolio_nav_usd=10_000_000.0
    )
    print("=== 1. MACRO BENCHMARK SHOCKS (SPOT SLIDES) ===")
    print(spot_slides_df.to_string(index=False))

    factor_stds = np.sqrt(np.diag(F_cov))
    factor_slides_df = slides.generate_factor_shock_slides(
        risk_forecast.factor_exposures, pd.Series(factor_stds, index=F_cov.columns), portfolio_nav_usd=10_000_000.0
    )
    print("=== 2. BARRA FACTOR CRISIS SCENARIOS (WHAT ACTUALLY KILLS QUANTS) ===")
    print(factor_slides_df.to_string(index=False))


if __name__ == "__main__":
    main()