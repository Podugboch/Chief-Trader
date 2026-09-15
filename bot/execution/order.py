"""Order and Execution Models.

Strongly typed models for order management, execution tracking, and fees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"


class OrderStatus(str, Enum):
    NEW = "NEW"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    client_order_id: str = field(default_factory=lambda: f"bot_{str(uuid.uuid4())[:8]}")
    status: OrderStatus = OrderStatus.NEW
    executed_price: float = 0.0
    executed_quantity: float = 0.0
    fee_paid: float = 0.0
    fee_asset: str = "USDT"
    slippage_usd: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: Optional[datetime] = None

    @property
    def notional_value(self) -> float:
        px = self.executed_price if self.executed_price > 0 else (self.price or 0.0)
        return self.quantity * px


@dataclass
class Position:
    symbol: str
    units: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    highest_price_seen: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    strategy_name: str = "Unknown"
    break_even_activated: bool = False

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.units

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100.0

    @property
    def notional_value(self) -> float:
        return self.units * self.current_price
