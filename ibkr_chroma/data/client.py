from __future__ import annotations

import asyncio
from ib_async import IB, Stock, util


class IBKRClient:
    """Asynchronous IBKR TWS/Gateway Connector via ib_async."""
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()

    async def connect(self) -> None:
        if not self.ib.isConnected():
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            print(f"Connected to IBKR on {self.host}:{self.port} (Client ID: {self.client_id})")

    async def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
            print("Disconnected from IBKR.")

    async def fetch_daily_bars(self, symbol: str, duration: str = "1 Y", bar_size: str = "1 day"):
        contract = Stock(symbol, "SMART", "USD")
        await self.ib.qualifyContractsAsync(contract)
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        return util.df(bars)


async def main():
    client = IBKRClient(port=7497, client_id=10)
    try:
        await client.connect()
        df = await client.fetch_daily_bars("AAPL", duration="30 D")
        print("\nSuccessfully pulled historical bars from IBKR:")
        print(df[["date", "open", "high", "low", "close", "volume"]].tail())
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())