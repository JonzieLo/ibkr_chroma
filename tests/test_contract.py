import asyncio
from ib_async import IB, Contract, Stock


async def inspect_stock(ib: IB, symbol: str, exchange: str = "SMART", currency: str = "USD"):
    contract = Stock(symbol, exchange, currency)
    details_list = await ib.reqContractDetailsAsync(contract)
    
    if not details_list:
        print(f"[ERROR] Could not resolve contract for {symbol} ({exchange}/{currency})")
        return

    details = details_list[0]
    c = details.contract
    print(f"\n--- Contract: {c.symbol} ({details.longName}) ---")
    print(f"  Exchange:         {c.exchange} (Primary: {c.primaryExchange})")
    print(f"  Currency:         {c.currency}")
    print(f"  SecType:          {c.secType}")
    print(f"  Min Tick:         {details.minTick}")
    print(f"  Trading Hours:    {details.timeZoneId}")


async def main():
    ib = IB()
    await ib.connectAsync("127.0.0.1", 7497, clientId=99)
    try:
        await inspect_stock(ib, "AAPL", exchange="SMART", currency="USD")
        await inspect_stock(ib, "700", exchange="SEHK", currency="HKD")
    finally:
        ib.disconnect()

if __name__ == "__main__":
    asyncio.run(main())