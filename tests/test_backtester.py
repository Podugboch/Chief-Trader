"""Tests for Backtesting Engine."""

import numpy as np
import pandas as pd
import pytest

from bot.strategies.trend_momentum import TrendMomentumStrategy
from bot.backtesting.engine import BacktestEngine
from bot.config import RiskConfig


def test_backtest_engine_execution():
    n = 150
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    # Sine curve with upward drift to create trade opportunities
    t = np.linspace(0, 10, n)
    close = 50000 + (t * 500) + (1000 * np.sin(t))
    high = close + 100
    low = close - 100
    open_p = close + 10
    volume = np.ones(n) * 500

    df = pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume}, index=dates)

    strat = TrendMomentumStrategy()
    risk_cfg = RiskConfig(
        max_risk_per_trade_pct=0.015,
        atr_stop_multiplier=1.5,
        atr_take_profit_multiplier=3.0,
        fee_rate=0.001,
        slippage_bps=5.0,
    )
    engine = BacktestEngine(strategy=strat, initial_cash=10_000.0, risk_config=risk_cfg)

    res = engine.run(df, symbol="BTCUSDT", timeframe="1h")

    assert res.total_bars_evaluated == len(df) - 50
    assert len(res.equity_curve) == res.total_bars_evaluated
    assert res.metrics.initial_equity == 10_000.0
    assert not np.isnan(res.metrics.total_return_pct)
    assert not np.isnan(res.metrics.max_drawdown_pct)
    assert res.metrics.max_drawdown_pct >= 0.0
