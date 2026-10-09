"""
Factor Decomposition Engine. Referring to 'Statistical Arbitrage in the US Equities Market (Avellaneda & Lee 2010)'

Features:
- Factor Selection via Random Matrix Theory
- Volatility-scaled Eigenportfolio weights: Q_i^(k) = v_ik / sigma_i
    - PCA Factor Decomposition          (Section 2: Avellaneda & Lee)
    - Sector ETF ElasticNet/RidgeCV Sparse Factor Model    (Section 3: Avellaneda & Lee)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV

@dataclass(frozen=True)
class PCAResults:
    factor_returns: pd.DataFrame            # Time series of Factor Returns
    betas: pd.DataFrame                     # Factor exposures per stock
    residuals: pd.DataFrame                 # Idiosyncratic residual returns
    n_factors: int                          # Dynamic/Fixed Factor Count
    explained_variance_ratio: np.ndarray
    eigenvalues: np.ndarray
    marchenko_pastur_limit: float           # Upper noise boundary from Random Matrix Theory


class PCAFactorModel:
    def __init__(
        self,
        n_componenets: Optional[int] = None,
        min_factors: int = 3,
        max_factors: int = 15,
        use_rmt: bool = True
    ):
        self.n_components = n_componenets
        self.min_factors = min_factors
        self.max_factors = max_factors
        self.use_rmt = use_rmt

    
    @staticmethod
    def marchenko_pastur_threshold(
        T: int,
        N: int,
        sigma_sq: float = 1.0
    ):
        q = float(T) / float(N)
        lambda_max = sigma_sq * (1 + np.sqrt(1.0/q)) ** 2
        return float(lambda_max)


    def fit_transform(
        self,
        returns: pd.DataFrame
    ):
        clean_returns = returns.ffill().dropna(axis=1, thresh=int(len(returns) * 0.80)).dropna(axis=0)
        T,N = clean_returns.shape

        if N < self.min_factors or T < N:
            raise ValueError(f"Insufficient dimensions for PCA: T = {T}, N = {N}")
        
        # Standardize returns
        means = clean_returns.mean(axis=0)
        stds = clean_returns.std(axis=0).replace(0.0,np.nan)
        Y = (clean_returns - means)/stds

        # Eigen-decomposition of sample correlation matrix:
        # C = (1/T)*Y^T*Y
        corr = (Y.T @ Y)/(T - 1.0)
        eigenvalues, eigenvectors = np.linalg.eigh(corr.values)

        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:,sort_idx]

        # Number of Factors (Marchenko-Pastur RMT or Fixed K)
        mp_limit = self.marchenko_pastur_threshold(T,N)
        if self.use_rmt and self.n_components is None:
            rmt_count = int(np.sum(eigenvalues > mp_limit))
            K = min(self.max_factors, max(self.min_factors, rmt_count))
        else:
            K = self.n_components if self.n_components else 5
        
        top_eigenvalues = eigenvalues[:K]
        top_eigenvectors = eigenvectors[:,:K]
        explained_var = top_eigenvalues / np.sum(eigenvalues) # Q_i = v_i/ std_i

        weights_k = top_eigenvectors / stds.values[:,None]
        weights_k = weights_k / np.sum(abs(weights_k), axis=0, keepdims=True) # Sum(|w_i|) = 1.0

        # Factor Returns
        # F_k(t) = Sum(w_i^(k)) * R_it
        factor_returns_mat = clean_returns.values @ weights_k
        factor_names = [f"PCA_{k+1}" for k in range(K)]
        factor_returns_df = pd.DataFrame(factor_returns_mat, index=clean_returns.index,columns=factor_names)

        # Regress each stock on factor returns to extract betas & residuals
        # R_i = alpha_i + Sum_k beta_ik * F_k + epsilon_i
        F_mat = np.column_stack([np.ones(T), factor_returns_mat])
        betas_all = np.linalg.solve(F_mat.T @ F_mat, F_mat.T @ clean_returns.values)

        betas_df = pd.DataFrame(betas_all[1:, :].T, index=clean_returns.columns, columns=factor_names)
        predicted = F_mat @ betas_all
        residuals_df = pd.DataFrame(clean_returns.values - predicted, index=clean_returns.index, columns=clean_returns.columns)
        
        return PCAResults(
            factor_returns=factor_returns_df,
            betas=betas_df,
            residuals=residuals_df,
            n_factors=K,
            explained_variance_ratio=explained_var,
            eigenvalues=eigenvalues,
            marchenko_pastur_limit=mp_limit,
        )


class SectorETFFactorModel:
    """
    Implements Section 3 of Avellaneda & Lee (2010): The ETF Approach.
    Uses ElasticNet / Ridge regularized regression of stocks on sector ETFs to avoid multicollinearity and enforce economic sparsity.
    """
    def __init__(self, l1_ratio: float = 0.5):
        self.l1_ratio = l1_ratio

    def fit_residuals(
        self,
        stock_returns: pd.DataFrame,
        etf_returns: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Regresses each stock return on liquid Sector ETFs using ElasticNet.
        R_stock = alpha + beta_SPY * R_SPY + Sum beta_sector * R_sector + epsilon

        Returns:
        - betas: Sparse factor exposures
        - residuals: Idiosyncratic residual returns
        """
        aligned_etfs = etf_returns.reindex(stock_returns.index).dropna()
        aligned_stocks = stock_returns.reindex(aligned_etfs.index).dropna(axis=1)

        residuals_dict = {}
        betas_dict = {}

        for col in aligned_stocks.columns:
            y = aligned_stocks[col].values
            X = aligned_etfs.values

            model = ElasticNetCV(l1_ratio=self.l1_ratio, cv=3, random_state=42)
            model.fit(X, y)

            pred = model.predict(X)
            residuals_dict[col] = y - pred
            betas_dict[col] = model.coef_

        residuals_df = pd.DataFrame(residuals_dict, index=aligned_etfs.index)
        betas_df = pd.DataFrame(betas_dict, index=aligned_etfs.columns).T

        return betas_df, residuals_df