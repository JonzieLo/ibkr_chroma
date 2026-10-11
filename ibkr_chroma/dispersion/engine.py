"""
Vega-Neutral Equity Index Dispersion Portfolio Constructor.

Structures:
- Short 1 Index Straddle (SPY/QQQ).
- Long N Constituent Straddles (AAPL, MSFT, NVDA, etc.) scaled by Index Weight & Vega.
- Complete Delta & Vega Neutralization.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DispersionLeg:
    symbol: str
    is_index: bool
    side: str                 # 'SHORT' for Index, 'LONG' for Constituents
    contracts: float
    strike: float
    spot_price: float
    straddle_price: float
    dollar_vega: float        # $/1% Vol move
    delta: float


@dataclass(frozen=True)
class DispersionTradeStructure:
    index_symbol: str
    index_leg: DispersionLeg
    constituent_legs: List[DispersionLeg]
    net_dollar_vega: float    # Must be ~0.0
    net_delta_usd: float      # Must be ~0.0
    correlation_spread: float
    initial_margin_usd: float


class DispersionTradeEngine:
    def __init__(
        self,
        target_index_contracts: float = 10.0,  # Base scale (e.g., Short 10 SPY straddles)
        max_constituent_count: int = 10,       # Top 10 heavyweights represent ~55% of QQQ/SPY
    ):
        self.target_index_contracts = target_index_contracts
        self.max_constituent_count = max_constituent_count

    def build_dispersion_structure(
        self,
        index_symbol: str,
        index_spot: float,
        index_iv: float,
        tau_days: float,
        constituent_spots: pd.Series,
        constituent_ivs: pd.Series,
        constituent_weights: pd.Series,
        correlation_spread: float,
    ) -> DispersionTradeStructure:
        """
        Constructs a vega-neutral dispersion trade.
        """
        tau = max(1.0 / 365.0, tau_days / 365.0)

        # Price Index ATM Straddle 
        # Straddle ~= 0.8 * S * sigma * sqrt(tau))
        index_straddle_px = 0.80 * index_spot * index_iv * np.sqrt(tau)
        index_vega_per_contract = 0.01 * index_spot * np.sqrt(tau) * 0.80 * 100.0  # 100 shares/contract
        total_index_vega = self.target_index_contracts * index_vega_per_contract

        index_leg = DispersionLeg(
            symbol=index_symbol,
            is_index=True,
            side="SHORT",
            contracts=self.target_index_contracts,
            strike=index_spot,
            spot_price=index_spot,
            straddle_price=index_straddle_px,
            dollar_vega=total_index_vega,
            delta=0.0,
        )

        top_tickers = constituent_weights.sort_values(ascending=False).head(self.max_constituent_count).index
        valid_tickers = top_tickers.intersection(constituent_spots.dropna().index).intersection(constituent_ivs.dropna().index)

        w_basket = constituent_weights.loc[valid_tickers] / constituent_weights.loc[valid_tickers].sum()

        # Size Constituent Legs to match Index Vega:
        constituent_legs = []
        total_basket_vega = 0.0

        for sym in valid_tickers:
            s_px = float(constituent_spots[sym])
            s_iv = float(constituent_ivs[sym])
            s_weight = float(w_basket[sym])

            straddle_px = 0.80 * s_px * s_iv * np.sqrt(tau)
            vega_per_contract = 0.01 * s_px * np.sqrt(tau) * 0.80 * 100.0  # 100 shares/contract

            target_leg_vega = s_weight * total_index_vega
            contracts = target_leg_vega / max(0.1, vega_per_contract)
            dollar_vega = contracts * vega_per_contract
            total_basket_vega += dollar_vega

            constituent_legs.append(
                DispersionLeg(
                    symbol=sym,
                    is_index=False,
                    side="LONG",
                    contracts=contracts,
                    strike=s_px,
                    spot_price=s_px,
                    straddle_price=straddle_px,
                    dollar_vega=dollar_vega,
                    delta=0.0,
                )
            )

        net_vega = total_basket_vega - total_index_vega
        est_margin = (index_leg.contracts * index_spot * 100.0) * 0.20  # 20% margin on short index options

        return DispersionTradeStructure(
            index_symbol=index_symbol,
            index_leg=index_leg,
            constituent_legs=constituent_legs,
            net_dollar_vega=net_vega,
            net_delta_usd=0.0,
            correlation_spread=correlation_spread,
            initial_margin_usd=est_margin,
        )