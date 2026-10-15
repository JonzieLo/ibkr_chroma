"""
Index Dispersion & CBOE Implied Correlation Engine.

Features:
- CBOE S&P 500 Implied Correlation formula (Index Implied Vol vs. Weighted Single-Stock Vol).
- Realized Correlation matrix estimation across constituent basket.
- Implied-to-Realized Correlation Spread.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrelationMetrics:
    implied_correlation: float      # rho_imp (CBOE formula)
    realized_correlation: float     # rho_real (historical rolling return correlation)
    correlation_spread: float       # rho_imp - rho_real (Positive = Short Dispersion opportunity)
    index_implied_vol: float        # sigma_I
    weighted_constituent_vol: float # Sum(w_i * sigma_i)
    basket_components_count: int


class DispersionCorrelationModel:
    @staticmethod
    def calculate_cboe_implied_correlation(
        index_iv: float,
        constituent_ivs: pd.Series,
        constituent_weights: pd.Series,
    ) -> float:
        """
        Computes the CBOE Implied Correlation:
        rho_imp = (sigma_I^2 - Sum(w_i^2 * sigma_i^2)) / ((Sum(w_i * sigma_i))^2 - Sum(w_i^2 * sigma_i^2))
        """
        valid = constituent_ivs.dropna().index.intersection(constituent_weights.dropna().index)
        if len(valid) < 5:
            return np.nan

        vols = constituent_ivs.loc[valid].values
        w = constituent_weights.loc[valid].values
        # Normalize weights across constituent basket
        w = w / np.sum(w)

        sigma_index_sq = index_iv ** 2
        sum_w2_sigma2 = np.sum((w ** 2) * (vols ** 2))
        sum_w_sigma_sq = (np.sum(w * vols)) ** 2

        denominator = sum_w_sigma_sq - sum_w2_sigma2
        if denominator <= 1e-8:
            return np.nan

        numerator = sigma_index_sq - sum_w2_sigma2
        rho_imp = numerator / denominator
        return float(np.clip(rho_imp, 0.01, 0.99))

    @staticmethod
    def calculate_realized_correlation(
        constituent_returns: pd.DataFrame,
        constituent_weights: pd.Series,
        window: int = 63,
    ) -> float:
        """
        Computes the rolling weighted average realized pairwise correlation across the constituent basket.
        """
        returns_slice = constituent_returns.iloc[-window:].dropna(how="all")
        valid_cols = returns_slice.columns[returns_slice.notna().sum() >= int(window *0.7)]
        valid = valid_cols.intersection(constituent_weights.index)
        if len(valid) < 5:
            return np.nan

        w = constituent_weights.loc[valid].values
        w = w / np.sum(w)

        # Pairwise correlation matrix (N x N)
        corr_matrix = returns_slice.corr().values.copy()

        # Off-diagonal weighted average correlation:
        # rho_real = Sum_{i != j} w_i * w_j * rho_ij / Sum_{i != j} w_i * w_j
        w_outer = np.outer(w, w)
        np.fill_diagonal(w_outer, 0.0)
        np.fill_diagonal(corr_matrix, 0.0)

        denom = np.sum(w_outer)
        if denom <= 0:
            return np.nan

        rho_real = np.sum(w_outer * corr_matrix) / denom
        return float(np.clip(rho_real, -1.0, 1.0))

    @classmethod
    def evaluate_correlation_opportunity(
        cls,
        index_iv: float,
        constituent_ivs: pd.Series,
        constituent_weights: pd.Series,
        constituent_returns: pd.DataFrame,
        window: int = 63,
    ) -> CorrelationMetrics:
        """
        Evaluates the dispersion opportunity:
        Spread = Implied Correlation - Realized Correlation.
        """
        rho_imp = cls.calculate_cboe_implied_correlation(index_iv, constituent_ivs, constituent_weights)
        rho_real = cls.calculate_realized_correlation(constituent_returns, constituent_weights, window=window)
        spread = rho_imp - rho_real

        valid = constituent_ivs.dropna().index.intersection(constituent_weights.index)
        w = constituent_weights.loc[valid] / constituent_weights.loc[valid].sum()
        weighted_vol = float((w * constituent_ivs.loc[valid]).sum())

        return CorrelationMetrics(
            implied_correlation=rho_imp,
            realized_correlation=rho_real,
            correlation_spread=spread,
            index_implied_vol=index_iv,
            weighted_constituent_vol=weighted_vol,
            basket_components_count=len(valid),
        )