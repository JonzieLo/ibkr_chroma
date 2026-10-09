"""
Portfolio Construction & Optimization Weights.

Generates:
- Long/Short Quintile Dollar-Neutral Weights (Q5 Long, Q1 Short)
- Beta-Neutral Sizing
- Gross Leverage Constraints
"""

import numpy as np
import pandas as pd


def generate_quintile_weights(
    composite_alpha: pd.DataFrame,
    gross_leverage: float = 2.0,  # 100% Long, 100% Short
) -> pd.DataFrame:
    """
    Constructs Long/Short dollar-neutral weights:
    - Long top 20%, Short bottom 20%
    - Sum(weights) = 0.0 (Dollar Neutral)
    - Sum(|weights|) = gross_leverage
    """
    weights_panel = pd.DataFrame(index=composite_alpha.index, columns=composite_alpha.columns, data=0.0)

    for date, row in composite_alpha.iterrows():
        valid = row.dropna()
        if len(valid) < 10:
            continue

        q80 = valid.quantile(0.80)
        q20 = valid.quantile(0.20)

        longs = valid[valid >= q80].index
        shorts = valid[valid <= q20].index

        if len(longs) > 0 and len(shorts) > 0:
            # Allocate half of gross leverage to longs, half to shorts
            w_long = (gross_leverage / 2.0) / len(longs)
            w_short = -(gross_leverage / 2.0) / len(shorts)

            weights_panel.loc[date, longs] = w_long
            weights_panel.loc[date, shorts] = w_short

    return weights_panel


def neutralize_portfolio_beta(
    weights: pd.Series,
    betas: pd.Series,
) -> pd.Series:
    valid = weights.dropna()
    aligned_betas = betas.reindex(valid.index).fillna(1.0)

    long_mask = valid > 0
    short_mask = valid < 0

    long_beta = np.sum(valid[long_mask] * aligned_betas[long_mask])
    short_beta = np.sum(valid[short_mask] * aligned_betas[short_mask])

    adjusted = valid.copy()
    if long_beta > 0 and abs(short_beta) > 0:
        ratio = long_beta / abs(short_beta)
        if ratio > 1.0:
            adjusted[long_mask] /= ratio
        else:
            adjusted[short_mask] *= ratio

    return adjusted