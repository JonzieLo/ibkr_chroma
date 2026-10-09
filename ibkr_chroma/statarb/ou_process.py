"""
Ornstein-Uhlenbeck Engine Referring to 'Statistical Arbitrage in the US Equities Market (Avellaneda & Lee 2010)'

Features:
- AR(1) discretization of cotinuous Ornstein-Uhlenback SDE:
    X_t = a + b * X_{t-1} + zeta_t
    - kappa = -ln(b)/dt
    - half_life = ln(2)/kappa
    - sigma_eq = sqrt(Var(zeta) / (1 - b^2))
- Centered s*-score caluclation: s_i* = s_i - Mean(s)
- Stateful Position Lifecycle: Tracking open trades, holding duration, stop-losses.
"""


from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class OUParams:
    kappa: float            # Mean-reversion speed per day
    half_life_days: float   # Days to halve dislocation
    mu: float               # Equilibrium residual mean
    sigma_eq: float         # Equilibrium standard deviation
    raw_s_score: float
    centered_s_score: float # Centered s*-score (Eq. 17)


@dataclass
class ActiveStatArbPosition:
    ticker: str
    entry_date: pd.Timestamp
    side: int               # Long = +1, Short = -1
    entry_s_score: float
    entry_price: float
    current_shares: float
    days_held: int = 0


@dataclass(frozen=True)
class ClosedStatArbTrade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    side: int
    entry_s_score: float
    exit_s_score: float
    entry_price: float
    exit_price: float
    days_held: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str        # 'TARGET_REVERTED', 'STOP_LOSS', 'HORIZON_EXCEEDED'


class OUProcessModel:
    def __init__(
        self,
        dt: float = 1.0,             # Daily time step
        min_half_life: float = 2.0,  # Discard if reverts faster than 2 days (microstructure noise)
        max_half_life: float = 35.0, # Discard if half-life > 35 days (non-stationary drift)
        s_open: float = 1.25,        # Entry threshold (|s| >= 1.25)
        s_close: float = 0.50,       # Take-profit target (|s| <= 0.50)
        s_stop: float = 3.50,        # Stop-loss hurdle (|s| >= 3.50)
        max_holding_days: int = 45,  # Time stop
    ):
        self.dt = dt
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.s_open = s_open
        self.s_close = s_close
        self.s_stop = s_stop
        self.max_holding_days = max_holding_days

        # Stateful trade container
        self.open_positions: Dict[str, ActiveStatArbPosition] = {}
        self.closed_trades: List[ClosedStatArbTrade] = []

    def calibrate_single_stock(
        self,
        cumulative_residuals: pd.Series
    ) -> Optional[OUParams]:
        """
        Calibrates AR(1) least squares to cumulative residual series X(t) = Sum(epsilon).
        """
        clean_x = cumulative_residuals.dropna()
        if len(clean_x) < 40:
            return None

        x_t = clean_x.values[1:]
        x_prev = clean_x.values[:-1]

        # AR(1) linear regression
        # x_t = a + b * x_{t-1} + zeta
        A = np.column_stack([np.ones(len(x_prev)), x_prev])
        try:
            params, residuals, _, _ = np.linalg.lstsq(A, x_t, rcond=None)
            a, b = params[0], params[1]
        except Exception:
            return None

        if not (0.001 < b < 0.999):
            return None
        kappa = -np.log(b) / self.dt
        half_life = np.log(2.0) / kappa

        if half_life < self.min_half_life or half_life > self.max_half_life:
            return None

        # Long-run equilibrium parameters
        mu = a / (1.0 - b)
        residuals_zeta = x_t - (a + b * x_prev)
        sigma_zeta = float(np.std(residuals_zeta, ddof=2))
        
        # Equilibrium variance
        # sigma_eq^2 = sigma_zeta^2 / (1 - b^2)
        denom = max(1e-12, 1.0 - b**2)
        sigma_eq = sigma_zeta / np.sqrt(denom)

        if sigma_eq <= 0.0 or np.isnan(sigma_eq):
            return None

        current_x = clean_x.iloc[-1]
        raw_s = (current_x - mu) / sigma_eq

        return OUParams(
            kappa=float(kappa),
            half_life_days=float(half_life),
            mu=float(mu),
            sigma_eq=float(sigma_eq),
            raw_s_score=float(raw_s),
            centered_s_score=float(raw_s),  # Passthrough
        )

    def calibrate_universe(self, residual_returns: pd.DataFrame) -> Dict[str, OUParams]:
        """
        Calibrates O-U models across all stocks and centers s-scores:
        s_i* = s_i - Mean(s)
        """
        cum_residuals = residual_returns.cumsum()
        raw_results: Dict[str, OUParams] = {}

        for col in cum_residuals.columns:
            res = self.calibrate_single_stock(cum_residuals[col])
            if res is not None:
                raw_results[col] = res

        if not raw_results:
            return {}

        # Center s-scores across universe (Avellaneda & Lee Eq. 17)
        mean_raw_s = float(np.mean([p.raw_s_score for p in raw_results.values()]))

        centered_results: Dict[str, OUParams] = {}
        for ticker, p in raw_results.items():
            centered_s = p.raw_s_score - mean_raw_s
            centered_results[ticker] = OUParams(
                kappa=p.kappa,
                half_life_days=p.half_life_days,
                mu=p.mu,
                sigma_eq=p.sigma_eq,
                raw_s_score=p.raw_s_score,
                centered_s_score=centered_s,
            )

        return centered_results

    def update_positions(
        self,
        current_date: pd.Timestamp,
        current_prices: pd.Series,
        s_scores: Dict[str, OUParams],
        capital_per_position: float = 10_000.0,
    ) -> pd.Series:
        """
        Stateful Trade Manager:
        1. Checks exit conditions for open trades.
        2. Enters new positions for stocks breaching entry hurdles.
        3. Returns target dollar weights vector.
        """
        # Evaluate Open Positions
        for ticker in list(self.open_positions.keys()):
            pos = self.open_positions[ticker]
            pos.days_held += 1

            px = current_prices.get(ticker, np.nan)
            if np.isnan(px) or px <= 0:
                continue

            # Current s-score
            current_ou = s_scores.get(ticker)
            current_s = current_ou.centered_s_score if current_ou else 0.0

            should_close = False
            exit_reason = ""

            # Rule A: Take profit
            if pos.side == 1 and current_s >= -self.s_close:
                should_close = True
                exit_reason = "TARGET_REVERTED"
            elif pos.side == -1 and current_s <= self.s_close:
                should_close = True
                exit_reason = "TARGET_REVERTED"

            # Rule B: Stop loss
            elif abs(current_s) >= self.s_stop:
                should_close = True
                exit_reason = "STOP_LOSS"

            # Rule C: Time stop
            elif pos.days_held >= self.max_holding_days:
                should_close = True
                exit_reason = "HORIZON_EXCEEDED"

            if should_close:
                pnl_pct = pos.side * (px - pos.entry_price) / pos.entry_price
                pnl_usd = pos.current_shares * pos.side * (px - pos.entry_price)

                self.closed_trades.append(
                    ClosedStatArbTrade(
                        ticker=pos.ticker,
                        entry_date=pos.entry_date,
                        exit_date=current_date,
                        side=pos.side,
                        entry_s_score=pos.entry_s_score,
                        exit_s_score=current_s,
                        entry_price=pos.entry_price,
                        exit_price=px,
                        days_held=pos.days_held,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                    )
                )
                del self.open_positions[ticker]

        # Screen for new positions
        for ticker, ou in s_scores.items():
            if ticker in self.open_positions:
                continue  # Already holding

            px = current_prices.get(ticker, np.nan)
            if np.isnan(px) or px <= 0:
                continue

            s = ou.centered_s_score

            # Open Long: Underpriced residual
            if s <= -self.s_open:
                shares = capital_per_position / px
                self.open_positions[ticker] = ActiveStatArbPosition(
                    ticker=ticker,
                    entry_date=current_date,
                    side=1,
                    entry_s_score=s,
                    entry_price=px,
                    current_shares=shares,
                )
            # Open Short: Overpriced residual
            elif s >= self.s_open:
                shares = capital_per_position / px
                self.open_positions[ticker] = ActiveStatArbPosition(
                    ticker=ticker,
                    entry_date=current_date,
                    side=-1,
                    entry_s_score=s,
                    entry_price=px,
                    current_shares=shares,
                )

        # Convert Active Positions to Weights
        weights = pd.Series(0.0, index=current_prices.index)
        long_tickers = [t for t, p in self.open_positions.items() if p.side == 1 and t in weights.index]
        short_tickers = [t for t, p in self.open_positions.items() if p.side == -1 and t in weights.index]

        if len(long_tickers) > 0 and len(short_tickers) > 0:
            weights.loc[long_tickers] = 1.0 / len(long_tickers)
            weights.loc[short_tickers] = -1.0 / len(short_tickers)

        return weights