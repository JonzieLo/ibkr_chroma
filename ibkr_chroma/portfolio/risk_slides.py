"""
Portfolio Scenario Stress Testing & Risk Slides Engine
Scenarios:
1. Macro Market Shocks (Spot +/-10% with conditional volatility shifts).
2. Pure Barra Factor Shocks (Momentum -3sigma, Volatility +2sigma, etc.).
3. Historical Crisis Scenarios (2007 Quant Quake, 2020 COVID Shock, 2021 Short Squeeze).
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class RiskSlidesEngine:
    def __init__(
        self,
        spot_shocks: Optional[List[float]] = None,
        vol_correlation: float = -0.60,
    ):
        self.spot_shocks = spot_shocks or [-0.10, -0.05, -0.02, -0.01, 0.0, +0.01, +0.02, +0.05, +0.10]
        self.vol_correlation = vol_correlation

    def generate_macro_spot_slides(
        self,
        weights: pd.Series,
        betas: pd.Series,
        portfolio_nav_usd: float = 10_000_000.0,
    ) -> pd.DataFrame:
        """Evaluates portfolio P&L under broad market benchmark moves."""
        aligned_w = weights.dropna()
        aligned_b = betas.reindex(aligned_w.index).fillna(1.0)

        net_beta = float(np.dot(aligned_w.values, aligned_b.values))
        gross_exp = float(aligned_w.abs().sum())

        records = []
        for shock in self.spot_shocks:
            exp_ret_pct = net_beta * shock
            exp_pnl_usd = exp_ret_pct * portfolio_nav_usd
            cond_vol_pts = self.vol_correlation * shock * 100.0

            records.append({
                "Scenario Shock": f"{shock:+.1%}",
                "Portfolio Beta": f"{net_beta:+.4f}",
                "Expected PnL (%)": f"{exp_ret_pct:+.2%}",
                "Expected PnL ($)": f"${exp_pnl_usd:+,.2f}",
                "Cond. Vol Shift": f"{cond_vol_pts:+.1f} pts",
            })

        return pd.DataFrame(records)

    def generate_factor_shock_slides(
        self,
        portfolio_factor_exposures: pd.Series,
        factor_std_annualized: pd.Series,  # Daily sigma = annualized / sqrt(252)
        portfolio_nav_usd: float = 10_000_000.0,
    ) -> pd.DataFrame:
        """
        Simulate market shocks by empirical factor volatility (3-sigma shocks).
        """
        daily_factor_stds = factor_std_annualized / np.sqrt(252.0)

        # Scenarios defined in statistical sigmas (-3 sigma crash, +2 sigma rally)
        scenarios = [
            ("Quant Quake (Momentum Crash)", "Momentum", -3.0, "3-sigma momentum unwind"),
            ("Vol Spike / Low-Vol Inversion", "LowVol", -2.5, "2.5-sigma low-vol anomaly break"),
            ("Value Rotation / Growth Dump", "Momentum", -2.0, "2-sigma rotational reversal into laggards"),
            ("Mega-Cap Momentum Rally", "Momentum", +2.5, "2.5-sigma narrow breadth rally"),
        ]

        records = []
        for name, factor_name, sigma_multiplier, desc in scenarios:
            if factor_name in portfolio_factor_exposures.index and factor_name in daily_factor_stds.index:
                loading = portfolio_factor_exposures[factor_name]
                f_vol = daily_factor_stds[factor_name]
                daily_shock = sigma_multiplier * f_vol
                exp_ret_pct = loading * daily_shock
                exp_pnl_usd = exp_ret_pct * portfolio_nav_usd

                records.append({
                    "Crisis Scenario": name,
                    "Target Factor": factor_name,
                    "Loading (x_p)": f"{loading:+.3f}",
                    "Empirical Shock": f"{daily_shock:+.2%}",
                    "Expected PnL ($)": f"${exp_pnl_usd:+,.2f}",
                    "Description": desc,
                })

        return pd.DataFrame(records)