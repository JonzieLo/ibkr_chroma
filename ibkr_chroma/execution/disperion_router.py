"""
Live Options Dispersion Execution Router & Delta Hedger.
Generates executable IBKR Option Straddle orders and underlying delta hedges.
"""

from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
from ib_async import IB, Stock

from ibkr_chroma.dispersion.engine import DispersionTradeEngine
from ibkr_chroma.dispersion.pricing import BSMDividendModel, BinomialModel


@dataclass(frozen=True)
class OptionOrderIntent:
    symbol: str
    sec_type: str
    expiry: str
    strike: float
    right: str          # 'C' or 'P'
    action: str         # 'BUY' or 'SELL'
    quantity: int
    est_price: float
    dollar_vega: float
    delta: float


class DispersionExecutionRouter:
    def __init__(self, ib: IB, dry_run: bool = True):
        self.ib = ib
        self.dry_run = dry_run
        self.tree = BinomialModel(steps=100)

    async def get_nearest_option_chain(self, symbol: str, exchange: str = "SMART", target_days: int = 30) -> Tuple[str, List[float]]:
        """Queries IBKR for valid expiries and strikes near target_days out."""
        stock = Stock(symbol, exchange, "USD" if exchange != "SEHK" else "HKD")
        await self.ib.qualifyContractsAsync(stock)
        chains = await self.ib.reqSecDefOptParamsAsync(stock.symbol, "", stock.secType, stock.conId)
        
        if not chains:
            raise ValueError(f"No option chains found for {symbol}")

        chain = next((c for c in chains if c.exchange == exchange), chains[0])
        # Find expiry closest to target_days
        now = pd.Timestamp.now()
        expiries = sorted(chain.expirations)
        best_expiry = min(expiries, key=lambda exp: abs((pd.to_datetime(exp) - now).days - target_days))
        return best_expiry, sorted(chain.strikes)

    def generate_orders(
        self,
        index_symbol: str,
        index_spot: float,
        index_iv: float,
        constituent_spots: pd.Series,
        constituent_ivs: pd.Series,
        weights: pd.Series,
        index_expiry: str,
        index_strike: float,
        rate: float = 0.045,
        div_yield: float = 0.035,
        exchange: str = "SEHK",
        currency: str = "HKD",
    ) -> List[OptionOrderIntent]:
        """Calculates integer option straddle orders and prices."""
        tau = 30.0 / 365.0
        engine = DispersionTradeEngine(target_index_contracts=10.0, max_constituent_count=len(constituent_spots))
        
        # Build theoretical structure
        structure = engine.build_dispersion_structure(
            index_symbol=index_symbol,
            index_spot=index_spot,
            index_iv=index_iv,
            tau_days=30.0,
            constituent_spots=constituent_spots,
            constituent_ivs=constituent_ivs,
            constituent_weights=weights,
            correlation_spread=0.135,
        )

        # Scale to whole integer contracts (no decimals!)
        plan = DispersionTradeEngine.calculate_integer_dispersion_capital(structure)
        orders: List[OptionOrderIntent] = []

        # 1. Short Index Straddle (Sell Call + Sell Put using BSMDividendModel)
        idx_call_px = BSMDividendModel.price(index_spot, index_strike, tau, index_iv, rate, div_yield, is_call=True)
        idx_put_px = BSMDividendModel.price(index_spot, index_strike, tau, index_iv, rate, div_yield, is_call=False)
        idx_qty = plan["scaled_index_contracts"]

        orders.append(OptionOrderIntent(index_symbol, "OPT", index_expiry, index_strike, "C", "SELL", idx_qty, idx_call_px, -plan["short_index_margin_usd"] * 0.01, -0.5 * idx_qty))
        orders.append(OptionOrderIntent(index_symbol, "OPT", index_expiry, index_strike, "P", "SELL", idx_qty, idx_put_px, -plan["short_index_margin_usd"] * 0.01, +0.5 * idx_qty))

        # 2. Long Constituent Straddles (Buy Call + Buy Put using AmericanBinomialModel)
        for leg in plan["constituent_legs"]:
            sym = leg["symbol"]
            s_px = float(constituent_spots[sym])
            s_iv = float(constituent_ivs[sym])
            qty = leg["int_contracts"]
            
            c_px = self.tree.price(s_px, s_px, tau, s_iv, rate, div_yield=0.02, is_call=True)
            p_px = self.tree.price(s_px, s_px, tau, s_iv, rate, div_yield=0.02, is_call=False)
            
            orders.append(OptionOrderIntent(sym, "OPT", index_expiry, s_px, "C", "BUY", qty, c_px, leg["dollar_vega"] * 0.5, +0.5 * qty))
            orders.append(OptionOrderIntent(sym, "OPT", index_expiry, s_px, "P", "BUY", qty, p_px, leg["dollar_vega"] * 0.5, -0.5 * qty))

        return orders