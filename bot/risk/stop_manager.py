"""Dynamic Stop-Loss and Take-Profit Manager.

Manages trailing stops, break-even ratcheting, and profit targets
to protect gains and truncate downside risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StopLevels:
    entry_price: float
    current_stop: float
    take_profit: float
    highest_price_seen: float
    atr_multiplier: float
    break_even_activated: bool = False


class StopManager:
    """Manages active stops and trails."""

    def __init__(self, atr_multiplier: float = 2.0, break_even_atr_trigger: float = 1.2):
        self.atr_multiplier = atr_multiplier
        self.break_even_atr_trigger = break_even_atr_trigger

    def update_stop_level(
        self,
        stop_state: StopLevels,
        current_price: float,
        current_atr: float,
    ) -> Tuple[StopLevels, Optional[str]]:
        """Update trailing stop based on latest candle price and volatility.

        Returns updated StopLevels and exit reason if triggered (None if trade continues).
        """
        # 1. Update highest price seen
        if current_price > stop_state.highest_price_seen:
            stop_state.highest_price_seen = current_price

        profit_dist = current_price - stop_state.entry_price

        # 2. Break-Even check: if profit >= break_even_atr_trigger * ATR, move stop to entry price + slight fee buffer
        if not stop_state.break_even_activated and profit_dist >= (self.break_even_atr_trigger * current_atr):
            stop_state.current_stop = max(stop_state.current_stop, stop_state.entry_price * 1.001)
            stop_state.break_even_activated = True

        # 3. Dynamic Trailing Stop: Stop only ratchets upward for longs!
        dynamic_trail = stop_state.highest_price_seen - (stop_state.atr_multiplier * current_atr)
        if dynamic_trail > stop_state.current_stop:
            stop_state.current_stop = dynamic_trail

        # 4. Check Exit Conditions
        # Stop-loss hit
        if current_price <= stop_state.current_stop:
            return stop_state, "STOP_LOSS_HIT"

        # Take-profit hit
        if stop_state.take_profit > 0 and current_price >= stop_state.take_profit:
            return stop_state, "TAKE_PROFIT_HIT"

        return stop_state, None
