"""Binance Exchange API Client.

Handles HMAC-SHA256 authentication, timestamp synchronization, order routing,
and rate limit tracking for both Binance Spot Testnet and Live exchange.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Dict, Any, Optional
from urllib.parse import urlencode
import requests

from bot.execution.order import Order, OrderSide, OrderType, OrderStatus

logger = logging.getLogger("bot.execution.binance")


class BinanceClient:
    """Production-grade client for Binance Spot API."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        use_testnet: bool = True,
        timeout: int = 10,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet
        self.timeout = timeout

        if use_testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"

        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})
        self.session.headers.update({"User-Agent": "CryptoTradingBot/1.0"})

    def _sign_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adds timestamp and HMAC-SHA256 signature to query parameters."""
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000

        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        params["signature"] = signature
        return params

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        if signed:
            if not self.api_key or not self.api_secret:
                raise ValueError("Authenticated endpoint requires API Key and API Secret.")
            params = self._sign_params(params)

        headers = {}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key

        try:
            resp = self.session.request(
                method=method,
                url=url,
                params=params if method.upper() in ["GET", "DELETE"] else None,
                data=params if method.upper() in ["POST", "PUT"] else None,
                headers=headers,
                timeout=self.timeout,
            )
            # Track rate limit weight
            used_weight = resp.headers.get("x-mbx-used-weight-1m")
            if used_weight and int(used_weight) > 1000:
                logger.warning("Binance rate limit warning: weight %s/1200", used_weight)

            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as http_err:
            try:
                err_data = resp.json()
                logger.error("Binance API HTTP %d error: %s", resp.status_code, err_data)
                raise RuntimeError(f"Binance API Error [{err_data.get('code')}]: {err_data.get('msg')}") from http_err
            except Exception:
                raise RuntimeError(f"Binance API HTTP error: {http_err}") from http_err
        except Exception as e:
            logger.error("Binance network error on %s: %s", endpoint, e)
            raise

    def ping(self) -> bool:
        """Test exchange connectivity."""
        try:
            self._request("GET", "/api/v3/ping")
            return True
        except Exception:
            return False

    def get_account_balances(self) -> Dict[str, float]:
        """Fetch spot account free balances for all non-zero assets."""
        data = self._request("GET", "/api/v3/account", signed=True)
        balances: Dict[str, float] = {}
        for b in data.get("balances", []):
            free_amt = float(b.get("free", 0.0))
            if free_amt > 0.0:
                balances[b["asset"]] = free_amt
        return balances

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Create order on Binance."""
        clean_symbol = symbol.upper().replace("/", "")
        params: Dict[str, Any] = {
            "symbol": clean_symbol,
            "side": side.value,
            "type": order_type.value,
            "quantity": f"{quantity:.6f}".rstrip("0").rstrip("."),
        }

        if order_type == OrderType.LIMIT:
            if price is None:
                raise ValueError("LIMIT order requires price")
            params["price"] = f"{price:.2f}"
            params["timeInForce"] = "GTC"
        elif order_type == OrderType.STOP_LOSS_LIMIT:
            if price is None or stop_price is None:
                raise ValueError("STOP_LOSS_LIMIT order requires price and stop_price")
            params["price"] = f"{price:.2f}"
            params["stopPrice"] = f"{stop_price:.2f}"
            params["timeInForce"] = "GTC"

        res = self._request("POST", "/api/v3/order", params=params, signed=True)

        order_status = OrderStatus.NEW
        if res.get("status") == "FILLED":
            order_status = OrderStatus.FILLED

        # Calculate executed price from fills if available
        fills = res.get("fills", [])
        executed_px = float(res.get("price", 0.0))
        fee_paid = 0.0
        fee_asset = "USDT"

        if fills:
            total_qty = sum(float(f["qty"]) for f in fills)
            if total_qty > 0:
                executed_px = sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty
            fee_paid = sum(float(f["commission"]) for f in fills)
            fee_asset = fills[0].get("commissionAsset", "USDT")

        return Order(
            symbol=clean_symbol,
            side=side,
            order_type=order_type,
            quantity=float(res.get("origQty", quantity)),
            price=price,
            order_id=str(res.get("orderId")),
            client_order_id=res.get("clientOrderId", ""),
            status=order_status,
            executed_price=executed_px,
            executed_quantity=float(res.get("executedQty", 0.0)),
            fee_paid=fee_paid,
            fee_asset=fee_asset,
        )

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order."""
        clean_symbol = symbol.upper().replace("/", "")
        params = {"symbol": clean_symbol, "orderId": int(order_id)}
        res = self._request("DELETE", "/api/v3/order", params=params, signed=True)
        return res.get("status") == "CANCELED"
