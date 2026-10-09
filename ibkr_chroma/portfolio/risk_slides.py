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
        portfolio_factor_exposures: pd.Series,  # x_p = w^T * X
        factor_std_annualized: pd.Series,       # sqrt(diag(F_cov))
        portfolio_nav_usd: float = 10_000_000.0,
    ) -> pd.DataFrame:
        """
        Stress-tests specific Barra Style Factor crashes:
        PnL = Portfolio_Loading * Factor_Shock * NAV
        """
        # Define institutional stress scenarios
        scenarios = [
            ("Quant Quake (Momentum Crash)", "Momentum", -0.08, "Momentum plunges -8% (Aug 2007 unwind)"),
            ("Vol Spike / Flight to Quality", "LowVol", -0.05, "Low-Vol anomaly breaks as vol surges"),
            ("Value Rotation / Growth Dump", "Momentum", -0.04, "Rotational reversal into value laggards"),
            ("Tech / Mega-Cap Rally", "Momentum", +0.05, "Narrow breadth mega-cap momentum surge"),
        ]

        records = []
        for name, factor_name, daily_shock, desc in scenarios:
            if factor_name in portfolio_factor_exposures.index:
                loading = portfolio_factor_exposures[factor_name]
                exp_ret_pct = loading * daily_shock
                exp_pnl_usd = exp_ret_pct * portfolio_nav_usd

                records.append({
                    "Crisis Scenario": name,
                    "Target Factor": factor_name,
                    "Loading (x_p)": f"{loading:+.3f}",
                    "Factor Shock": f"{daily_shock:+.1%}",
                    "Expected PnL ($)": f"${exp_pnl_usd:+,.2f}",
                    "Description": desc,
                })

        return pd.DataFrame(records)