"""Tests for Paper Exchange Simulator."""

import pytest
from bot.execution.paper_exchange import PaperExchange
from bot.execution.order import OrderSide, OrderStatus


def test_paper_exchange_buy_and_sell_cycle():
    # Start with 10,000 cash, 0.1% fee, 10 bps slippage (0.10%)
    exchange = PaperExchange(
        initial_cash_usd=10_000.0,
        fee_rate=0.001,
        slippage_bps=10.0,
    )

    # 1. Execute BUY of 1 BTC at mark price $50,000
    # Slippage = +0.10% -> fill at 50,050
    # Fee = 50,050 * 0.001 = 50.05
    # Total deduction = 50,050 + 50.05 = 50,100.05
    buy_order = exchange.execute_market_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=0.1,
        current_market_price=50_000.0,
        stop_loss=48_000.0,
        take_profit=55_000.0,
    )

    assert buy_order.status == OrderStatus.FILLED
    assert buy_order.executed_price > 50_000.0  # slipped upward on buy
    assert "BTCUSDT" in exchange.positions
    assert exchange.positions["BTCUSDT"].units == 0.1
    assert exchange.cash < 10_000.0 - 5000.0

    # 2. Execute SELL at higher price $55,000
    sell_order = exchange.execute_market_order(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        quantity=0.1,
        current_market_price=55_000.0,
    )

    assert sell_order.status == OrderStatus.FILLED
    assert "BTCUSDT" not in exchange.positions  # Fully closed
    assert len(exchange.trade_history) == 1

    last_trade = exchange.trade_history[0]
    # Trade was profitable even after paying entry fee, exit fee, and slippage
    assert last_trade["pnl_usd"] > 0
    assert exchange.total_fees_paid > 0
    assert exchange.total_slippage_lost > 0


def test_paper_exchange_insufficient_funds():
    exchange = PaperExchange(initial_cash_usd=100.0)
    # Attempt to buy $50,000 worth of BTC with only $100 cash
    order = exchange.execute_market_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=1.0,
        current_market_price=50_000.0,
    )
    assert order.status == OrderStatus.REJECTED
    assert len(exchange.positions) == 0
