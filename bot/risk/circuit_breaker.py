"""Circuit Breakers & Emergency Halts.

Protects against black swan volatility shocks, max portfolio drawdowns,
daily loss limits, and repeated API failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Optional

logger = logging.getLogger("bot.risk.circuit_breaker")


@dataclass
class CircuitBreakerStatus:
    is_tripped: bool
    reason: Optional[str]
    cooldown_remaining_bars: int
    daily_drawdown_pct: float
    total_drawdown_pct: float
    high_water_mark: float


class CircuitBreaker:
    """Manages trading pauses and emergency circuit breakers."""

    def __init__(
        self,
        max_daily_drawdown_pct: float = 0.05,  # 5% max loss in single day
        max_total_drawdown_pct: float = 0.15,  # 15% max drawdown from peak
        volatility_atr_multiplier: float = 3.0,  # 3x ATR spike
        cooldown_bars: int = 4,
    ):
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_total_drawdown_pct = max_total_drawdown_pct
        self.volatility_atr_multiplier = volatility_atr_multiplier
        self.cooldown_bars = cooldown_bars

        # State tracking
        self.high_water_mark: float = 0.0
        self.daily_start_equity: float = 0.0
        self.current_day: Optional[date] = None
        self.cooldown_remaining: int = 0
        self.consecutive_api_errors: int = 0
        self.max_consecutive_api_errors: int = 5
        self.is_hard_stopped: bool = False
        self.trip_reason: Optional[str] = None
        self.last_status: Optional[CircuitBreakerStatus] = None

    def update_equity(self, current_equity: float, current_time: Optional[datetime] = None) -> CircuitBreakerStatus:
        """Update portfolio equity and evaluate drawdown limits."""
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        today = current_time.date()

        # Day rollover check
        if self.current_day != today:
            self.current_day = today
            self.daily_start_equity = current_equity

        # Initialize HWM
        if self.high_water_mark <= 0.0 or current_equity > self.high_water_mark:
            self.high_water_mark = current_equity

        # Drawdown calculations
        daily_dd = (
            (self.daily_start_equity - current_equity) / self.daily_start_equity
            if self.daily_start_equity > 0
            else 0.0
        )
        total_dd = (
            (self.high_water_mark - current_equity) / self.high_water_mark
            if self.high_water_mark > 0
            else 0.0
        )

        # 1. Total portfolio max drawdown check
        if total_dd >= self.max_total_drawdown_pct:
            self.is_hard_stopped = True
            self.trip_reason = (
                f"MAX DRAWDOWN BREACHED: Portfolio fell {total_dd*100:.2f}% from peak "
                f"(limit: {self.max_total_drawdown_pct*100:.1f}%). Halting trading to protect capital."
            )
            logger.critical(self.trip_reason)

        # 2. Daily loss limit check
        elif daily_dd >= self.max_daily_drawdown_pct:
            self.trip_reason = (
                f"DAILY LOSS LIMIT REACHED: Daily loss is {daily_dd*100:.2f}% "
                f"(limit: {self.max_daily_drawdown_pct*100:.1f}%). Trading paused for the day."
            )
            logger.warning(self.trip_reason)

        is_tripped = self.is_hard_stopped or (daily_dd >= self.max_daily_drawdown_pct) or (self.cooldown_remaining > 0)

        status = CircuitBreakerStatus(
            is_tripped=is_tripped,
            reason=self.trip_reason if is_tripped else None,
            cooldown_remaining_bars=self.cooldown_remaining,
            daily_drawdown_pct=daily_dd,
            total_drawdown_pct=total_dd,
            high_water_mark=self.high_water_mark,
        )
        self.last_status = status
        return status

    @property
    def status(self) -> CircuitBreakerStatus:
        """Get current circuit breaker status."""
        if self.last_status is not None:
            return self.last_status
        is_tripped = self.is_hard_stopped or (self.cooldown_remaining > 0)
        return CircuitBreakerStatus(
            is_tripped=is_tripped,
            reason=self.trip_reason if is_tripped else None,
            cooldown_remaining_bars=self.cooldown_remaining,
            daily_drawdown_pct=0.0,
            total_drawdown_pct=0.0,
            high_water_mark=self.high_water_mark,
        )

    def check_volatility_spike(self, current_atr: float, baseline_atr: float) -> bool:
        """Trigger temporary cooldown if ATR violently spikes."""
        if baseline_atr > 0 and (current_atr >= self.volatility_atr_multiplier * baseline_atr):
            self.cooldown_remaining = self.cooldown_bars
            self.trip_reason = (
                f"VOLATILITY SPIKE DETECTED: ATR ({current_atr:.2f}) > "
                f"{self.volatility_atr_multiplier}x baseline ({baseline_atr:.2f}). "
                f"Pausing new entries for {self.cooldown_bars} bars."
            )
            logger.warning(self.trip_reason)
            return True
        return False

    def record_api_success(self) -> None:
        self.consecutive_api_errors = 0

    def record_api_error(self, err_msg: str) -> bool:
        """Record API failure. Trips breaker if error count exceeds threshold."""
        self.consecutive_api_errors += 1
        logger.error("API failure count: %d/%d (%s)", self.consecutive_api_errors, self.max_consecutive_api_errors, err_msg)
        if self.consecutive_api_errors >= self.max_consecutive_api_errors:
            self.trip_reason = f"EXCHANGE API FAILURE: {self.consecutive_api_errors} consecutive errors. Pausing."
            return True
        return False

    def step_bar(self) -> None:
        """Call on each new bar to decrement temporary cooldowns."""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining == 0 and not self.is_hard_stopped:
                self.trip_reason = None
                logger.info("Volatility cooldown elapsed. Trading resumed.")
