"""
Factor Normalization, Sector-Neutralization Engine
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

class FactorEngine:
    @staticmethod
    def winsorize_cross_section(
        panel: pd.DataFrame,
        limit: float = 0.02
    ) -> pd.DataFrame:
        """Clips values at percentage limit"""
        q_low = panel.quantile(limit, axis=1)
        q_high = panel.quantile(1-limit, axis=1)
        return panel.clip(lower=q_low, upper=q_high, axis=0)

    @staticmethod
    def zscore_cross_section(
        panel: pd.DataFrame,
        cap_weights: Optional[pd.DataFrame] = None  # Market Cap Weights
    ) -> pd.DataFrame:
        if cap_weights is not None:
            norm_weights = cap_weights.reindex_like(panel).div(cap_weights.sum(axis=1),axis=0)
            p_mean = (panel*norm_weights).sum(axis=1)
            p_std = panel.std(axis=1).replace(0.0,np.nan)
            return panel.sub(p_mean,axis=0).div(p_std,axis=0)

        p_mean = panel.mean(axis=1)
        p_std = panel.std(axis=1).replace(0.0,np.nan)
        return panel.sub(p_mean,axis=0).div(p_std,axis=0)

    @classmethod
    def sector_neutralize(
        cls,
        zscores: pd.DataFrame,
        sector_map: Dict[str, str]
    ) -> pd.DataFrame:
        """De-means factor scores within each GICS sector on each date."""
        sectors = pd.Series(sector_map).reindex(zscores.columns)

        stacked = zscores.stack().rename("score").to_frame()
        stacked["sector"] = stacked.index.get_level_values(1).map(sectors)
        sector_means = stacked.groupby([stacked.index.get_level_values(0), "sector"])["score"].transform("mean")
        stacked["neutral_score"] = stacked["score"] - sector_means

        neutral_panel = stacked["neutral_score"].unstack()
        return cls.zscore_cross_section(neutral_panel)

    @classmethod
    def build_composite_signal(
        cls,
        factor_dict: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
        sector_map: Optional[Dict[str,str]] = None,
        cap_weights: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        processed_factors = []
        names = list(factor_dict.keys())
        if weights is None:
            weights = {name: 1.0 / len(names) for name in names}

        for name in names:
            raw = factor_dict[name]
            winsorized = cls.winsorize_cross_section(raw)
            zscored = cls.zscore_cross_section(winsorized, cap_weights=cap_weights)
            if sector_map:
                neutral = cls.sector_neutralize(zscored, sector_map)
            else:
                neutral = zscored

            weighted = neutral * weights[name]
            processed_factors.append(weighted)

        composite = sum(processed_factors)
        return cls.zscore_cross_section(composite, cap_weights=cap_weights)

    @staticmethod
    def compute_forward_ic(
        factor_panel: pd.DataFrame,
        closes: pd.DataFrame,
        forward_days: int = 5
    ) -> pd.Series:
        """
        Spearman Rank Correlation between Factor Score at t and forward returns from t to t + forward_days.
        """
        forward_returns = closes.pct_change(forward_days).shift(-forward_days)
        
        ranked_factors = factor_panel.rank(axis=1)
        ranked_returns = forward_returns.rank(axis=1)

        def _calc_row_corr(date):
            f_row = ranked_factors.loc[date].dropna()
            r_row = ranked_returns.loc[date].dropna()
            common = f_row.index.intersection(r_row.index)
            if len(common) < 15:
                return np.nan
            return f_row.loc[common].corr(r_row.loc[common])

        dates = factor_panel.index[:-forward_days]
        ic_series = pd.Series({d: _calc_row_corr(d) for d in dates})
        return ic_series.dropna()