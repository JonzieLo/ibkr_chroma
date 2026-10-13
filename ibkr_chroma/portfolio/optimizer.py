"""
Barra-Constrained Mean-Variance Portfolio Optimizer.

Solves the Convex Optimization Problem:
    max_w   w^T * alpha - (lambda_risk / 2) * w^T * Sigma_Barra * w - penalty * ||w - w_prev||_1
    s.t.    Sum(w) = 0                       (Dollar Neutrality)
            X_sector^T * w = 0               (Sector Neutrality)
            w^T * beta = 0                   (Beta Neutrality)
            Sum(|w|) <= gross_leverage       (Leverage Constraint)
            w_min <= w_i <= w_max            (Single-Stock Position Limits)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    expected_alpha: float
    forecast_risk_annualized: float
    net_exposure: float
    gross_leverage: float
    success: bool
    status_message: str


class BarraPortfolioOptimizer:
    def __init__(
        self,
        risk_aversion: float = 2.0,
        gross_leverage: float = 2.0,
        max_position_weight: float = 0.06,
        max_style_exposure: float = 0.15,  # Max allowable net loading on any style factor
        turnover_penalty: float = 0.0005,
    ):
        self.risk_aversion = risk_aversion
        self.gross_leverage = gross_leverage
        self.max_position_weight = max_position_weight
        self.max_style_exposure = max_style_exposure
        self.turnover_penalty = turnover_penalty

    def optimize(
        self,
        alpha_scores: pd.Series,
        exposure_matrix: pd.DataFrame,
        factor_covariance: pd.DataFrame,
        specific_variances: pd.Series,
        betas: Optional[pd.Series] = None,
        current_weights: Optional[pd.Series] = None,
    ) -> OptimizationResult:
        tickers = alpha_scores.dropna().index
        aligned_alpha = alpha_scores.reindex(tickers).values
        N = len(tickers)

        if N < 10:
            raise ValueError(f"Universe too small for optimization (N={N})")

        X = exposure_matrix.reindex(tickers).fillna(0.0).values
        F = factor_covariance.values
        delta = specific_variances.reindex(tickers).fillna(specific_variances.median()).values
        Sigma = (X @ F @ X.T + np.diag(delta)) / 252.0

        w_prev = current_weights.reindex(tickers).fillna(0.0).values if current_weights is not None else np.zeros(N)
        b_vec = betas.reindex(tickers).fillna(1.0).values if betas is not None else np.ones(N)

        # Row 0: Dollar Neutrality (Sum(w) = 0)
        # Row 1: Beta Neutrality (Sum(w * beta) = 0)
        # Rows 2..J: Sector Neutrality (Sum(w_sector) = 0)
        style_col_names = ["Momentum", "LowVol", "FiftyTwoWeekHigh", "Reversal"]
        sector_indices = [
            i for i, col in enumerate(exposure_matrix.columns) if col not in style_col_names
        ]

        eq_rows = [np.ones(N), b_vec]
        for s_idx in sector_indices:
            eq_rows.append(X[:, s_idx])

        A_eq = np.array(eq_rows)  # Shape: (M_eq x N)
        M_eq = A_eq.shape[0]

        q_high = np.quantile(aligned_alpha, 0.80)
        q_low = np.quantile(aligned_alpha, 0.20)
        w_raw = np.zeros(N)
        w_raw[aligned_alpha >= q_high] = +1.0
        w_raw[aligned_alpha <= q_low] = -1.0

        # Project w_raw onto  nullspace of A_eq
        # w_proj = w - A^T (A A^T)^(-1) A w
        try:
            proj_kernel = np.linalg.solve(A_eq @ A_eq.T + 1e-8 * np.eye(M_eq), A_eq @ w_raw)
            w0 = w_raw - A_eq.T @ proj_kernel
            # Rescale to target gross leverage
            if np.sum(np.abs(w0)) > 0:
                w0 = w0 * (self.gross_leverage / np.sum(np.abs(w0)))
        except Exception:
            w0 = np.zeros(N)

        gamma_turnover = self.turnover_penalty if current_weights is not None else 0.0

        def objective(w: np.ndarray) -> float:
            port_alpha = np.dot(w, aligned_alpha)
            port_var = w.T @ Sigma @ w
            turnover_sq = 0.5 * gamma_turnover * np.sum((w - w_prev) ** 2)
            return float(-port_alpha + (self.risk_aversion / 2.0) * port_var + turnover_sq)

        def objective_grad(w: np.ndarray) -> np.ndarray:
            return -aligned_alpha + self.risk_aversion * (Sigma @ w) + gamma_turnover * (w - w_prev)


        constraints = [
            {"type": "eq", "fun": lambda w: A_eq @ w, "jac": lambda w: A_eq},               # Equality: A_eq @ w = 0
            {"type": "ineq", "fun": lambda w: self.gross_leverage - np.sum(np.abs(w))},     # Leverage: Sum(|w|) <= 2.0
        ]

        # Sector Neutrality constraints (Industry columns)
        for col_name in style_col_names:
            if col_name in exposure_matrix.columns:
                c_idx = list(exposure_matrix.columns).index(col_name)
                style_col = X[:, c_idx]
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, c=style_col: self.max_style_exposure - np.dot(w, c),
                    "jac": lambda w, c=style_col: -c,
                })
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, c=style_col: self.max_style_exposure + np.dot(w, c),
                    "jac": lambda w, c=style_col: c,
                })

        bounds = [(-self.max_position_weight, self.max_position_weight) for _ in range(N)]

        # Initial guess
        w0 = np.zeros(N)
        q_high = np.quantile(aligned_alpha, 0.80)
        q_low = np.quantile(aligned_alpha, 0.20)
        w0[aligned_alpha >= q_high] = (self.gross_leverage / 2.0) / max(1, np.sum(aligned_alpha >= q_high))
        w0[aligned_alpha <= q_low] = -(self.gross_leverage / 2.0) / max(1, np.sum(aligned_alpha <= q_low))

        res = minimize(
            objective,
            w0,
            method="SLSQP",
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2500, "ftol": 1e-5},
        )

        w_opt = pd.Series(res.x, index=tickers)
        w_opt[w_opt.abs() < 1e-5] = 0.0

        ann_port_var = float(res.x.T @ (X @ F @ X.T + np.diag(delta)) @ res.x)
        ann_risk = np.sqrt(max(0.0, ann_port_var))

        return OptimizationResult(
            weights=w_opt,
            expected_alpha=float(np.dot(res.x, aligned_alpha)),
            forecast_risk_annualized=ann_risk,
            net_exposure=float(w_opt.sum()),
            gross_leverage=float(w_opt.abs().sum()),
            success=bool(res.success),
            status_message=str(res.message),
        )