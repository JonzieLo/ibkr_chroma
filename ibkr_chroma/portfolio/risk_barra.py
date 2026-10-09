"""
Structural BARRA Multi-Factor Risk Model.

Implements:
1. Descriptor standardization
2. Factor Exposure Matrix X construction 
3. Cross-Sectional Weighted Least Squares (WLS) Regression.
4. EWMA Factor Covariance Matrix.
5. Specific Risk Estimation (Delta).
6. Full Portfolio Risk Decomposition (Systematic vs. Specific Variance).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarraRiskForecast:
    total_risk_annualized: float
    factor_risk_annualized: float
    specific_risk_annualized: float
    factor_exposures: pd.Series        # Portfolio loading on each factor (w^T * X)
    factor_pnl_attribution: pd.Series  # Factor return contributions


class BarraRiskModel:
    def __init__(self, half_life_days: int = 63):
        self.half_life = half_life_days
        self.lambda_decay = 0.5 ** (1.0 / half_life_days)
        
        # State matrices
        self.factor_covariance: Optional[pd.DataFrame] = None
        self.specific_variances: Optional[pd.Series] = None
        self.historical_factor_returns: List[pd.Series] = []
        self.historical_specific_returns: List[pd.Series] = []

    @staticmethod
    def standardize_descriptor(values: pd.Series, cap_weights: pd.Series, limit: float = 3.0) -> pd.Series:
        valid = values.dropna()
        if len(valid) < 5:
            return values

        w = cap_weights.reindex(valid.index).fillna(0.0)
        if w.sum() > 0:
            w = w / w.sum()
            mean = valid.dot(w)
        else:
            mean = valid.mean()

        std = valid.std()
        if std == 0 or np.isnan(std):
            return values * 0.0

        z = (values - mean) / std
        return z.clip(lower=-limit, upper=limit)

    @classmethod
    def build_exposure_matrix(
        cls,
        style_factors: Dict[str, pd.Series],
        sector_series: pd.Series,
        market_caps: pd.Series,
    ) -> pd.DataFrame:
        """
        Constructs the N x K Factor Exposure Matrix:
        Columns 0-J: GICS Sector Dummies (Binary 1 or 0)
        Columns J-K: Standardized Risk Indices
        """
        tickers = sector_series.dropna().index
        cap_weights = market_caps.reindex(tickers).fillna(1.0)
        cap_weights = cap_weights / cap_weights.sum()

        industry_dummies = pd.get_dummies(sector_series.reindex(tickers), dtype=float)
        standardized_indices = pd.DataFrame(index=tickers)
        for name, series in style_factors.items():
            s_aligned = series.reindex(tickers)
            standardized_indices[name] = cls.standardize_descriptor(s_aligned, cap_weights)

        X = pd.concat([industry_dummies, standardized_indices], axis=1).fillna(0.0)
        return X

    @staticmethod
    def cross_sectional_wls_regression(
        returns: pd.Series,
        X: pd.DataFrame,
        market_caps: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Runs Cross-Sectional Weighted Least Squares (WLS)
        """
        aligned_tickers = returns.reindex(X.index).dropna().index
        y = returns.reindex(aligned_tickers).to_numpy(dtype=np.float64)
        X_mat = X.reindex(aligned_tickers).to_numpy(dtype=np.float64)
        caps = market_caps.reindex(aligned_tickers).to_numpy(dtype=np.float64)

        weights = np.sqrt(np.maximum(1e-6, caps))

        # WLS Solution: (X^T W X)^(-1) X^T W y
        X_weighted = X_mat * weights[:, np.newaxis]
        XtWX = X_mat.T @ X_weighted
        XtWy = X_weighted.T @ y

        # Solve system with Ridge regularization
        reg = 1e-6 * np.eye(XtWX.shape[0])
        f_hat = np.linalg.solve(XtWX + reg, XtWy)

        u_hat = y - (X_mat @ f_hat)

        factor_returns = pd.Series(f_hat, index=X.columns)
        specific_returns = pd.Series(u_hat, index=aligned_tickers)
        return factor_returns, specific_returns

    def update_historical_step(self, factor_returns: pd.Series, specific_returns: pd.Series) -> None:
        self.historical_factor_returns.append(factor_returns)
        self.historical_specific_returns.append(specific_returns)

    def compute_covariance_matrices(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Estimates the Factor Covariance Matrix F and Specific Variances Delta using Exponentially Weighted Moving Average (EWMA).
        """
        if len(self.historical_factor_returns) < 10:
            raise ValueError("Need at least 10 historical periods to compute Barra covariance.")

        F_df = pd.DataFrame(self.historical_factor_returns)
        U_df = pd.DataFrame(self.historical_specific_returns).fillna(0.0)

        T = len(F_df)
        weights = np.array([self.lambda_decay ** (T - 1 - t) for t in range(T)])
        weights = weights / weights.sum()

        # Weighted Factor Covariance Matrix
        mean_f = np.average(F_df.values, axis=0, weights=weights)
        centered_f = F_df.values - mean_f
        # Annualized Covariance
        F_cov = (centered_f.T @ np.diag(weights) @ centered_f) * 252.0
        self.factor_covariance = pd.DataFrame(F_cov, index=F_df.columns, columns=F_df.columns)

        # Annualized Specific Variances
        u_sq = U_df.values ** 2
        spec_var = (weights @ u_sq) * 252.0
        self.specific_variances = pd.Series(spec_var, index=U_df.columns)

        return self.factor_covariance, self.specific_variances

    def forecast_portfolio_risk(
        self,
        portfolio_weights: pd.Series,
        X: pd.DataFrame,
    ) -> BarraRiskForecast:
        """
        Calculates Ex-Ante Annualized Portfolio Risk Decomposition:
        Total Variance = Factor Variance + Specific Variance
        """
        if self.factor_covariance is None or self.specific_variances is None:
            self.compute_covariance_matrices()

        assert self.factor_covariance is not None
        assert self.specific_variances is not None

        tickers = portfolio_weights.index
        w = portfolio_weights.to_numpy(dtype=np.float64)

        # Portfolio factor exposure: x_p = X^T * w (K x 1)
        X_aligned = X.reindex(tickers).fillna(0.0).to_numpy(dtype=np.float64)
        x_p = X_aligned.T @ w  # Factor loading vector

        F_mat = self.factor_covariance.to_numpy(dtype=np.float64)
        factor_variance = float(x_p.T @ F_mat @ x_p)

        # Specific variance: w^T * Delta * w
        delta = self.specific_variances.reindex(tickers).fillna(self.specific_variances.median()).to_numpy(dtype=np.float64)
        specific_variance = float(np.sum((w ** 2) * delta))

        total_variance = factor_variance + specific_variance
        total_risk = np.sqrt(max(0.0, total_variance))
        factor_risk = np.sqrt(max(0.0, factor_variance))
        specific_risk = np.sqrt(max(0.0, specific_variance))

        factor_loadings = pd.Series(x_p, index=X.columns)

        return BarraRiskForecast(
            total_risk_annualized=total_risk,
            factor_risk_annualized=factor_risk,
            specific_risk_annualized=specific_risk,
            factor_exposures=factor_loadings,
            factor_pnl_attribution=factor_loadings * self.historical_factor_returns[-1],
        )