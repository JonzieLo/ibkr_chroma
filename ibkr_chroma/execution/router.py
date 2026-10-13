"""
Execustion Router & Order Reconciliation.
Computes order Targets Shares, Current Shares.
"""

from dataclasses import dataclass
from typing import List
import pandas as pd
from ib_async import IB, MarketOrder, Order, Stock, Trade


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    exchange: str
    currency: str
    side: str                               # 'BUY' or 'SELL'
    shares: int
    target_weight: float
    current_weight: float
    est_dollar_value: float


class ExecutionRouter:
    def __init__(
        self,
        ib: IB,
        dry_run: bool = True,
        max_order_usd: float = 100_000.0,   # Max safety ceiling per order
    ):
        self.ib = ib
        self.dry_run = dry_run
        self.max_order_usd = max_order_usd

    def generate_rebalance_orders(
        self,
        target_weights: pd.Series,          # Desired portfolio weights from model
        current_weights: pd.Series,         # Real-time holdings from position_sync
        current_prices: pd.Series,          # Live stock prices
        account_nav_usd: float,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> List[OrderIntent]:
        """Calculates share orders to transition portfolio to target weights."""
        all_symbols = target_weights.index.union(current_weights.index)
        orders: List[OrderIntent] = []

        for sym in all_symbols:
            w_target = float(target_weights.get(sym, 0.0))
            w_curr = float(current_weights.get(sym, 0.0))
            px = float(current_prices.get(sym, 0.0))

            if px <= 0.0:
                continue

            # Desired dollar value and current dollar value
            target_usd = w_target * account_nav_usd
            current_usd = w_curr * account_nav_usd
            delta_usd = target_usd - current_usd

            # Safety check: avoid micro-trades (< $200)
            if abs(delta_usd) < 200.0:
                continue

            delta_shares = int(round(delta_usd / px))
            if delta_shares == 0:
                continue

            side = "BUY" if delta_shares > 0 else "SELL"
            abs_shares = abs(delta_shares)
            order_usd = abs_shares * px

            if order_usd > self.max_order_usd:
                abs_shares = int(self.max_order_usd / px)

            orders.append(
                OrderIntent(
                    symbol=sym,
                    exchange=exchange,
                    currency=currency,
                    side=side,
                    shares=abs_shares,
                    target_weight=w_target,
                    current_weight=w_curr,
                    est_dollar_value=order_usd,
                )
            )

        return orders

    async def execute_orders(self, order_intents: List[OrderIntent]) -> List[Trade]:
        """Submits orders to IB Gateway (logs them if dry_run=True)."""
        trades: List[Trade] = []

        for intent in order_intents:
            contract = Stock(intent.symbol, intent.exchange, intent.currency)
            order = MarketOrder(intent.side, intent.shares)

            if self.dry_run:
                print(f"[DRY-RUN] {intent.side:<4} {intent.shares:>5} shares of {intent.symbol:<5} (~${intent.est_dollar_value:,.2f})")
            else:
                await self.ib.qualifyContractsAsync(contract)
                trade = self.ib.placeOrder(contract, order)
                print(f"[SUBMITTED] {intent.side:<4} {intent.shares:>5} shares of {intent.symbol:<5} | OrderId: {trade.order.orderId}")
                trades.append(trade)

        return trades