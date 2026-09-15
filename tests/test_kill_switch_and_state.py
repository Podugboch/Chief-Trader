"""Tests for Kill Switch and State Persistence."""

import os
from pathlib import Path
import pytest

from bot.engine.kill_switch import KillSwitch
from bot.engine.state import BotState
from bot.execution.order import Position


def test_kill_switch_trigger_and_reset(tmp_path):
    flag_file = tmp_path / "kill_switch.flag"
    ks = KillSwitch(flag_path=flag_file)

    assert not ks.is_triggered()
    assert not flag_file.exists()

    # Trigger
    ks.trigger("Test emergency stop")
    assert ks.is_triggered()
    assert flag_file.exists()
    assert "Test emergency stop" in ks.reason

    # Reset
    ks.reset()
    assert not ks.is_triggered()
    assert not flag_file.exists()


def test_state_save_and_load(tmp_path):
    state_dir = tmp_path / "state"
    state = BotState(state_dir=state_dir)

    pos = Position(
        symbol="BTCUSDT",
        units=0.25,
        entry_price=60000.0,
        current_price=61000.0,
        stop_loss=58000.0,
        take_profit=65000.0,
        highest_price_seen=61500.0,
        strategy_name="TrendMomentum",
    )

    state.save(
        cash=7500.0,
        balances={"BTC": 0.25},
        positions={"BTCUSDT": pos},
        high_water_mark=11000.0,
        trade_history=[{"symbol": "BTCUSDT", "pnl_usd": 150.0}],
    )

    loaded = state.load()
    assert loaded is not None
    assert loaded["cash"] == 7500.0
    assert loaded["high_water_mark"] == 11000.0

    restored_positions = loaded["reconstructed_positions"]
    assert "BTCUSDT" in restored_positions
    r_pos = restored_positions["BTCUSDT"]
    assert r_pos.units == 0.25
    assert r_pos.entry_price == 60000.0
    assert r_pos.stop_loss == 58000.0
    assert r_pos.take_profit == 65000.0
    assert r_pos.highest_price_seen == 61500.0
