"""Tests for Risk Management Modules."""

import pytest
from datetime import datetime, timezone

from bot.risk.position_sizer import PositionSizer
from bot.risk.stop_manager import StopManager, StopLevels
from bot.risk.circuit_breaker import CircuitBreaker


def test_position_sizer_standard_calculation():
    sizer = PositionSizer(max_risk_pct=0.01, max_position_equity_pct=0.30)
    # Account: $10,000 equity, $10,000 cash
    # Risk = 1% of 10,000 = $100
    # Entry: $1,000, Stop: $950 -> distance = $50
    # Target units = 100 / 50 = 2.0 units ($2,000 notional = 20% equity < 30% cap)
    res = sizer.calculate_size(
        account_equity=10_000.0,
        available_cash=10_000.0,
        entry_price=1000.0,
        stop_loss_price=950.0,
        current_open_positions_count=0,
        max_open_positions=3,
    )

    assert res.allowed is True
    assert res.units == 2.0
    assert res.notional_usd == 2000.0
    assert res.risk_usd == 100.0


def test_position_sizer_exposure_cap():
    # If stop loss is very tight, sizing would be huge, so exposure cap kicks in
    sizer = PositionSizer(max_risk_pct=0.05, max_position_equity_pct=0.20)
    # 20% cap on 10,000 equity = max $2,000 notional
    res = sizer.calculate_size(
        account_equity=10_000.0,
        available_cash=10_000.0,
        entry_price=100.0,
        stop_loss_price=99.9,  # tiny stop
        current_open_positions_count=0,
    )

    assert res.allowed is True
    assert res.notional_usd <= 2000.0  # capped by single asset exposure


def test_position_sizer_max_positions_limit():
    sizer = PositionSizer()
    res = sizer.calculate_size(
        account_equity=10_000.0,
        available_cash=5000.0,
        entry_price=100.0,
        stop_loss_price=90.0,
        current_open_positions_count=3,
        max_open_positions=3,
    )
    assert res.allowed is False
    assert "Max concurrent positions reached" in res.reason


def test_stop_manager_trailing_and_breakeven():
    sm = StopManager(atr_multiplier=2.0, break_even_atr_trigger=1.0)
    # Entry at 100, initial stop at 90, TP at 130
    stop_state = StopLevels(
        entry_price=100.0,
        current_stop=90.0,
        take_profit=130.0,
        highest_price_seen=100.0,
        atr_multiplier=2.0,
    )

    atr = 5.0
    # Price rises to 106 (+6 > 1.0 * 5.0 ATR trigger) -> break even activated
    updated, exit_reason = sm.update_stop_level(stop_state, current_price=106.0, current_atr=atr)
    assert exit_reason is None
    assert updated.break_even_activated is True
    assert updated.current_stop >= 100.0  # ratcheted to break even!

    # Price rises further to 120 (highest price 120 - 2*5 = 110 stop)
    updated, exit_reason = sm.update_stop_level(updated, current_price=120.0, current_atr=atr)
    assert exit_reason is None
    assert updated.current_stop == 110.0

    # Price hits take profit at 130
    updated, exit_reason = sm.update_stop_level(updated, current_price=131.0, current_atr=atr)
    assert exit_reason == "TAKE_PROFIT_HIT"


def test_circuit_breaker_drawdown_protection():
    cb = CircuitBreaker(
        max_daily_drawdown_pct=0.05,  # 5%
        max_total_drawdown_pct=0.15,  # 15%
    )

    # Initial equity 10,000
    st1 = cb.update_equity(10_000.0)
    assert st1.is_tripped is False

    # Equity rises to 12,000 (new HWM)
    st2 = cb.update_equity(12_000.0)
    assert st2.is_tripped is False
    assert cb.high_water_mark == 12_000.0

    # Equity drops to 10,000 (drop from peak is (12000 - 10000)/12000 = 16.67% > 15%)
    st3 = cb.update_equity(10_000.0)
    assert st3.is_tripped is True
    assert "MAX DRAWDOWN BREACHED" in st3.reason
