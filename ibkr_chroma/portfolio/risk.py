"""
Portfolio Risk Models & Scenario Stress Testing (Risk Slides).
"""
import numpy as np
import pandas as pd

def calculate_equity_risk_slides(
    weights: pd.Series, 
    betas: pd.Series, 
    spot_shocks: list[float] = [-0.05, -0.02, -0.01, 0.0, +0.01, +0.02, +0.05]
) -> pd.DataFrame:
    """
    Simulates portfolio P&L under discrete market scenario shocks.
    Portfolio P&L = Sum(w_i * beta_i * market_shock)
    """
    portfolio_beta = np.dot(weights, betas)
    slides = []
    for shock in spot_shocks:
        expected_pnl_pct = portfolio_beta * shock
        slides.append({
            "Market_Shock": f"{shock:+.1%}",
            "Portfolio_Beta": portfolio_beta,
            "Expected_PnL_Pct": f"{expected_pnl_pct:+.2%}",
        })
    return pd.DataFrame(slides)