"""
IBKR portfolio & Position Synchronization. Extracts account NAV, cash, current equity positions.
"""

from dataclasses import dataclass
from typing import Dict
import pandas as pd
from ib_async import IB


@dataclass(frozen=True)
class AccountSnapshot:
    account_id: str
    net_liquidation_usd: float
    total_cash_usd: float
    buying_power_usd: float
    current_positions: Dict[str, float]     # Symbol -> Share count
    current_weights: pd.Series              # Symbol -> Portfolio Weight (Position_USD / NAV)


class PositionSynchronizer:
    def __init__(self, ib: IB):
        self.ib = ib

    async def get_account_snapshot(self) -> AccountSnapshot:
        """Pulls live account valuation and open positions."""
        if not self.ib.isConnected():
            raise ConnectionError("IB Gateway is not connected.")

        # Account Summary (NAV, Cash, Buying Power)
        summary = await self.ib.accountSummaryAsync()
        summary_map: Dict[str, float] = {}
        account_id = "UNKNOWN"

        for item in summary:
            if account_id == "UNKNOWN" and item.account:
                account_id = item.account
            try:
                summary_map[item.tag] = float(item.value)
            except (ValueError, TypeError):
                continue

        net_liq = summary_map.get("NetLiquidation", 1_000_000.0)
        cash = summary_map.get("TotalCashValue", net_liq)
        buying_power = summary_map.get("BuyingPower", net_liq * 2.0)

        # Open Positions
        portfolio_items = self.ib.portfolio()
        pos_dict: Dict[str, float] = {}
        market_val_dict: Dict[str, float] = {}

        if portfolio_items:
            for item in portfolio_items:
                sym = item.contract.symbol
                qty = float(item.position)
                val = float(item.marketValue)
                if qty != 0:
                    pos_dict[sym] = qty
                    market_val_dict[sym] = val
        else:
            # Portfolio list is empty
            positions = await self.ib.reqPositionsAsync()
            for p in positions:
                sym = p.contract.symbol
                qty = float(p.position)
                avg_cost = float(getattr(p, "avgCost", 0.0))
                if qty != 0:
                    pos_dict[sym] = qty
                    market_val_dict[sym] = qty * avg_cost

        # Current Portfolio Weights: w_i = Value_i / NAV
        if market_val_dict:
            weights_series = pd.Series(market_val_dict) / max(1.0, net_liq)
        else:
            weights_series = pd.Series(dtype=float)

        return AccountSnapshot(
            account_id=account_id,
            net_liquidation_usd=net_liq,
            total_cash_usd=cash,
            buying_power_usd=buying_power,
            current_positions=pos_dict,
            current_weights=weights_series,
        )