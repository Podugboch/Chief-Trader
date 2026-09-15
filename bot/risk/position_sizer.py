"""Risk-First Position Sizing.

Calculates optimal order size based on volatility (ATR risk budgeting),
total account equity, and strict portfolio exposure caps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("bot.risk.position_sizer")


@dataclass
class SizingResult:
    allowed: bool
    units: float  # e.g., BTC quantity
    notional_usd: float  # units * entry_price
    risk_usd: float  # Max dollar amount risked if stopped out
    reason: str


class PositionSizer:
    """Calculates position size using ATR risk budgeting and portfolio caps."""

    def __init__(
        self,
        max_risk_pct: float = 0.015,  # 1.5% equity risked per trade
        max_position_equity_pct: float = 0.35,  # Max 35% of total capital in one asset
        min_order_usd: float = 10.0,  # Binance minimum order size is typically $10 USDT
    ):
        self.max_risk_pct = max_risk_pct
        self.max_position_equity_pct = max_position_equity_pct
        self.min_order_usd = min_order_usd

    def calculate_size(
        self,
        account_equity: float,
        available_cash: float,
        entry_price: float,
        stop_loss_price: float,
        current_open_positions_count: int,
        max_open_positions: int = 3,
    ) -> SizingResult:
        """Compute position size with multiple safety constraints."""
        if current_open_positions_count >= max_open_positions:
            return SizingResult(
                allowed=False,
                units=0.0,
                notional_usd=0.0,
                risk_usd=0.0,
                reason=f"Max concurrent positions reached ({current_open_positions_count}/{max_open_positions})",
            )

        if entry_price <= 0 or stop_loss_price <= 0:
            return SizingResult(
                allowed=False,
                units=0.0,
                notional_usd=0.0,
                risk_usd=0.0,
                reason="Invalid entry or stop loss price",
            )

        stop_distance = abs(entry_price - stop_loss_price)
        stop_distance_pct = stop_distance / entry_price

        # If stop distance is unrealistically small (< 0.2%) or huge (> 15%)
        if stop_distance_pct < 0.002:
            stop_distance = entry_price * 0.01  # default 1% floor

        # Dollar risk budget (e.g. 1.5% of total account equity)
        dollar_risk_budget = account_equity * self.max_risk_pct

        # Position units derived from dollar risk:
        # risk = units * (entry - stop) => units = risk / stop_distance
        target_units = dollar_risk_budget / stop_distance
        target_notional = target_units * entry_price

        # Cap 1: Single asset exposure cap (e.g. max 35% of total equity)
        max_notional_cap = account_equity * self.max_position_equity_pct
        if target_notional > max_notional_cap:
            target_notional = max_notional_cap
            target_units = target_notional / entry_price
            logger.info("Position size capped by single-asset exposure limit: $%.2f", max_notional_cap)

        # Cap 2: Available cash buffer (must never exceed available liquid cash)
        # Retain a safety buffer for fees and slippage (98% of cash)
        cash_limit = available_cash * 0.98
        if target_notional > cash_limit:
            target_notional = cash_limit
            target_units = target_notional / entry_price
            logger.info("Position size capped by available liquid cash: $%.2f", cash_limit)

        # Check minimum order size
        if target_notional < self.min_order_usd:
            return SizingResult(
                allowed=False,
                units=0.0,
                notional_usd=0.0,
                risk_usd=0.0,
                reason=f"Calculated size (${target_notional:.2f}) below minimum order value (${self.min_order_usd:.2f})",
            )

        actual_risk_usd = target_units * stop_distance

        return SizingResult(
            allowed=True,
            units=round(target_units, 6),
            notional_usd=round(target_notional, 2),
            risk_usd=round(actual_risk_usd, 2),
            reason="Size successfully calculated within risk limits",
        )
