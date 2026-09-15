"""Tests for Trading Strategies."""

import numpy as np
import pandas as pd
import pytest

from bot.strategies.trend_momentum import TrendMomentumStrategy
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.regime_adaptive import RegimeAdaptiveStrategy
from bot.strategies.base import SignalAction


@pytest.fixture
def trending_dataframe():
    n = 100
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    # Steady upward trend with volume expansion
    close = np.linspace(40000, 60000, n)
    high = close + 100
    low = close - 100
    open_p = close - 50
    volume = np.linspace(100, 1000, n)  # expanding volume
    return pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def test_trend_momentum_strategy_signals(trending_dataframe):
    strat = TrendMomentumStrategy()
    df_prepared = strat.prepare_indicators(trending_dataframe)

    assert "ema_fast" in df_prepared.columns
    assert "macd" in df_prepared.columns
    assert "supertrend_dir" in df_prepared.columns

    signal = strat.generate_signal(df_prepared, "BTCUSDT")
    assert signal.symbol == "BTCUSDT"
    assert signal.action in (SignalAction.BUY, SignalAction.HOLD, SignalAction.CLOSE)

    if signal.action == SignalAction.BUY:
        assert signal.stop_loss is not None
        assert signal.take_profit is not None
        assert signal.stop_loss < signal.price < signal.take_profit


def test_mean_reversion_strategy_signals(trending_dataframe):
    strat = MeanReversionStrategy()
    df_prepared = strat.prepare_indicators(trending_dataframe)

    assert "bb_lower" in df_prepared.columns
    assert "rsi" in df_prepared.columns

    signal = strat.generate_signal(df_prepared, "BTCUSDT")
    assert signal.action in (SignalAction.BUY, SignalAction.HOLD, SignalAction.CLOSE)


def test_adaptive_strategy(trending_dataframe):
    adaptive = RegimeAdaptiveStrategy()
    sig = adaptive.generate_signal(trending_dataframe, "BTCUSDT")
    assert sig.symbol == "BTCUSDT"
    assert "regime" in sig.metadata
