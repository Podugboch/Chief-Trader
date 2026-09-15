"""Multi-Exchange Unified Client.

Leverages CCXT to provide uniform connectivity across 100+ cryptocurrency exchanges,
including Binance, Bybit, OKX, Kraken, Coinbase, KuCoin, Bitget, Gate.io, and HTX.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import ccxt
import pandas as pd

from bot.execution.order import Order, OrderSide, OrderType, OrderStatus

logger = logging.getLogger("bot.execution.multi_exchange")

# Supported popular exchanges mapping
POPULAR_EXCHANGES = {
    "binance": "Binance",
    "bybit": "Bybit",
    "okx": "OKX",
    "kraken": "Kraken",
    "coinbase": "Coinbase Advanced",
    "kucoin": "KuCoin",
    "bitget": "Bitget",
    "gateio": "Gate.io",
    "mexc": "MEXC",
    "htx": "HTX (Huobi)",
}


class MultiExchangeClient:
    """Unified trading client supporting any CCXT exchange."""

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        passphrase: Optional[str] = None,
        use_testnet: bool = False,
        timeout: int = 15,
    ):
        self.exchange_id = exchange_id.lower().replace(" ", "").replace("-", "")
        if self.exchange_id == "gate":
            self.exchange_id = "gateio"
        elif self.exchange_id == "coinbasepro":
            self.exchange_id = "coinbase"

        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.use_testnet = use_testnet
        self.timeout = timeout * 1000  # ccxt uses milliseconds

        if not hasattr(ccxt, self.exchange_id):
            raise ValueError(f"Exchange '{exchange_id}' is not supported by CCXT.")

        exchange_class = getattr(ccxt, self.exchange_id)
        config: Dict[str, Any] = {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "timeout": self.timeout,
            "enableRateLimit": True,
        }

        # Passphrase for exchanges that require it (OKX, KuCoin, Bitget)
        if self.passphrase:
            config["password"] = self.passphrase

        self.exchange: ccxt.Exchange = exchange_class(config)

        # Set sandbox / testnet if requested and supported
        if self.use_testnet:
            try:
                self.exchange.set_sandbox_mode(True)
                logger.info("Sandbox mode activated for %s", self.exchange_id)
            except Exception as e:
                logger.warning("Sandbox mode not supported or failed for %s: %s", self.exchange_id, e)

    def ping(self) -> bool:
        """Test exchange connectivity."""
        try:
            if hasattr(self.exchange, "fetch_time"):
                self.exchange.fetch_time()
            elif hasattr(self.exchange, "fetch_status"):
                self.exchange.fetch_status()
            else:
                self.exchange.load_markets()
            return True
        except Exception as e:
            logger.warning("Ping failed for %s: %s", self.exchange_id, e)
            return False

    def get_account_balances(self) -> Dict[str, float]:
        """Fetch spot account free balances for all non-zero assets."""
        if not self.api_key or not self.api_secret:
            raise ValueError(f"API key and secret required to fetch balances from {self.exchange_id}")

        balance_data = self.exchange.fetch_balance()
        free_balances = balance_data.get("free", {})
        non_zero: Dict[str, float] = {}

        for asset, amount in free_balances.items():
            amt = float(amount or 0.0)
            if amt > 0.0:
                non_zero[asset] = amt

        return non_zero

    def _normalize_symbol(self, symbol: str) -> str:
        """Convert symbol e.g. BTCUSDT to CCXT standard BTC/USDT if needed."""
        sym = symbol.upper()
        if "/" not in sym:
            # Common quote assets
            for q in ["USDT", "USDC", "USD", "EUR", "BTC", "ETH"]:
                if sym.endswith(q) and len(sym) > len(q):
                    base = sym[: -len(q)]
                    return f"{base}/{q}"
        return sym

    def fetch_ticker_price(self, symbol: str) -> float:
        """Fetch latest price for a symbol."""
        ccxt_sym = self._normalize_symbol(symbol)
        ticker = self.exchange.fetch_ticker(ccxt_sym)
        return float(ticker.get("last") or ticker.get("close") or 0.0)

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles."""
        ccxt_sym = self._normalize_symbol(symbol)
        raw_ohlcv = self.exchange.fetch_ohlcv(ccxt_sym, timeframe=timeframe, limit=limit)
        if not raw_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(raw_ohlcv, columns=["open_time", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.set_index("timestamp").sort_index()
        df["trades"] = 0
        return df[["open", "high", "low", "close", "volume", "trades"]]

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Place order on exchange."""
        ccxt_sym = self._normalize_symbol(symbol)
        ccxt_side = side.value.lower()  # "buy" or "sell"
        ccxt_type = order_type.value.lower()  # "market" or "limit"

        params = {}
        if stop_price is not None:
            params["stopPrice"] = stop_price

        res = self.exchange.create_order(
            symbol=ccxt_sym,
            type=ccxt_type,
            side=ccxt_side,
            amount=quantity,
            price=price,
            params=params,
        )

        status_str = res.get("status", "").lower()
        if status_str in ("closed", "filled"):
            status = OrderStatus.FILLED
        elif status_str in ("canceled", "cancelled"):
            status = OrderStatus.CANCELED
        else:
            status = OrderStatus.NEW

        exec_px = float(res.get("average") or res.get("price") or price or 0.0)
        fee_paid = 0.0
        fee_info = res.get("fee") or {}
        if fee_info:
            fee_paid = float(fee_info.get("cost") or 0.0)

        return Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=float(res.get("amount", quantity)),
            price=price,
            order_id=str(res.get("id")),
            client_order_id=str(res.get("clientOrderId", "")),
            status=status,
            executed_price=exec_px,
            executed_quantity=float(res.get("filled", 0.0)),
            fee_paid=fee_paid,
        )

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order."""
        ccxt_sym = self._normalize_symbol(symbol)
        try:
            self.exchange.cancel_order(order_id, symbol=ccxt_sym)
            return True
        except Exception as e:
            logger.error("Failed to cancel order %s on %s: %s", order_id, self.exchange_id, e)
            return False
