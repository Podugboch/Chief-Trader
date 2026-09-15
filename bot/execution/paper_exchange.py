"""High-Fidelity Paper Trading Exchange.

Simulates exchange matching engine locally with realistic order fills,
explicit maker/taker fees, and slippage modeling for zero-risk testing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid

from bot.execution.order import Order, OrderSide, OrderType, OrderStatus, Position

logger = logging.getLogger("bot.execution.paper_exchange")


class PaperExchange:
    """Simulated matching engine with fee & slippage modeling."""

    def __init__(
        self,
        initial_cash_usd: float = 10_000.0,
        fee_rate: float = 0.001,  # 0.10% standard exchange fee
        slippage_bps: float = 5.0,  # 5 bps = 0.05% slippage
    ):
        self.initial_cash = initial_cash_usd
        self.cash = initial_cash_usd
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_bps / 10_000.0

        # Asset balances: e.g. {"BTC": 0.5, "ETH": 2.0}
        self.balances: Dict[str, float] = {}
        # Open positions: symbol -> Position
        self.positions: Dict[str, Position] = {}
        # Completed trades log (round trips)
        self.trade_history: List[Dict] = []
        # All executed orders log (both BUY and SELL)
        self.order_history: List[Dict] = []
        # Total fees paid across all trades
        self.total_fees_paid: float = 0.0
        self.total_slippage_lost: float = 0.0

    def get_portfolio_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate total mark-to-market portfolio value."""
        total = self.cash
        for sym, pos in self.positions.items():
            mark = current_prices.get(sym, pos.current_price)
            pos.current_price = mark
            total += pos.units * mark
        return total

    def execute_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_market_price: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        strategy_name: str = "Strategy",
    ) -> Order:
        """Execute a market order with fees and realistic slippage."""
        if current_market_price <= 0:
            raise ValueError(f"Invalid market price: {current_market_price}")

        # Compute slippage
        if side == OrderSide.BUY:
            fill_price = current_market_price * (1.0 + self.slippage_rate)
            slippage_usd = (fill_price - current_market_price) * quantity
        else:
            fill_price = current_market_price * (1.0 - self.slippage_rate)
            slippage_usd = (current_market_price - fill_price) * quantity

        notional = fill_price * quantity
        fee = notional * self.fee_rate

        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            executed_price=round(fill_price, 4),
            executed_quantity=quantity,
            fee_paid=round(fee, 4),
            slippage_usd=round(slippage_usd, 4),
            status=OrderStatus.FILLED,
            filled_at=datetime.now(timezone.utc),
        )

        self.total_fees_paid += fee
        self.total_slippage_lost += slippage_usd

        # Process Portfolio Updates
        if side == OrderSide.BUY:
            total_cost = notional + fee
            if total_cost > self.cash:
                # Insufficient funds
                order.status = OrderStatus.REJECTED
                logger.warning("Rejected BUY order: Required $%.2f > Cash $%.2f", total_cost, self.cash)
                return order

            self.cash -= total_cost
            base_asset = symbol.replace("USDT", "").replace("USD", "")
            self.balances[base_asset] = self.balances.get(base_asset, 0.0) + quantity

            # Create or update Position
            if symbol in self.positions:
                pos = self.positions[symbol]
                new_qty = pos.units + quantity
                # Weighted average entry price
                new_entry = ((pos.units * pos.entry_price) + notional) / new_qty
                pos.units = new_qty
                pos.entry_price = new_entry
                pos.current_price = fill_price
                pos.stop_loss = stop_loss if stop_loss > 0 else pos.stop_loss
                pos.take_profit = take_profit if take_profit > 0 else pos.take_profit
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    units=quantity,
                    entry_price=fill_price,
                    current_price=fill_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    highest_price_seen=fill_price,
                    strategy_name=strategy_name,
                )

            logger.info("BUY FILLED: %s %.4f @ $%.2f (Fee: $%.3f, Slippage: $%.3f)", symbol, quantity, fill_price, fee, slippage_usd)
            self.order_history.append({
                "order_id": order.order_id,
                "timestamp": order.filled_at.isoformat(),
                "symbol": symbol,
                "side": "BUY",
                "quantity": quantity,
                "price": fill_price,
                "notional": notional,
                "fee": fee,
                "pnl_usd": 0.0,
                "pnl_pct": 0.0,
                "strategy": strategy_name,
            })

        elif side == OrderSide.SELL:
            proceeds = notional - fee
            self.cash += proceeds

            base_asset = symbol.replace("USDT", "").replace("USD", "")
            current_held = self.balances.get(base_asset, 0.0)
            self.balances[base_asset] = max(0.0, current_held - quantity)

            # Close or reduce position
            pnl_usd = 0.0
            pnl_pct = 0.0
            if symbol in self.positions:
                pos = self.positions[symbol]
                entry_cost = pos.entry_price * quantity
                # Net PnL after entry & exit fees
                pnl_usd = notional - entry_cost - fee - (entry_cost * self.fee_rate)
                pnl_pct = ((fill_price - pos.entry_price) / pos.entry_price) * 100.0

                if quantity >= pos.units - 1e-6:
                    del self.positions[symbol]
                else:
                    pos.units -= quantity

            sell_record = {
                "order_id": order.order_id,
                "timestamp": order.filled_at.isoformat(),
                "symbol": symbol,
                "side": "SELL",
                "quantity": quantity,
                "price": fill_price,
                "notional": notional,
                "fee": fee,
                "pnl_usd": round(pnl_usd, 2),
                "pnl_pct": round(pnl_pct, 2),
                "strategy": strategy_name,
            }
            self.trade_history.append(sell_record)
            self.order_history.append(sell_record)

            logger.info(
                "SELL FILLED: %s %.4f @ $%.2f | PnL: $%.2f (%.2f%%) (Fee: $%.3f)",
                symbol, quantity, fill_price, pnl_usd, pnl_pct, fee
            )

        return order
