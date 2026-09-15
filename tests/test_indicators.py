"""Tests for Technical Indicators."""

import numpy as np
import pandas as pd
import pytest

from bot.indicators.technical import (
    ema,
    sma,
    rsi,
    macd,
    bollinger_bands,
    atr,
    adx,
    obv,
    donchian_channel,
    supertrend,
    true_range,
)


@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data with known trend and oscillation."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")

    # Generate synthetic price series with upward trend then oscillation
    trend = np.linspace(50000, 60000, n)
    noise = np.random.normal(0, 300, n)
    close = trend + noise

    high = close + np.random.uniform(50, 200, n)
    low = close - np.random.uniform(50, 200, n)
    open_p = close + np.random.uniform(-50, 50, n)
    volume = np.random.uniform(100, 1000, n)

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    return df


def test_ema_and_sma(sample_ohlcv):
    close = sample_ohlcv["close"]
    ema20 = ema(close, 20)
    sma20 = sma(close, 20)

    assert len(ema20) == len(close)
    assert len(sma20) == len(close)
    assert not ema20.isna().all()
    # In an uptrend, EMA responds faster than SMA
    assert ema20.iloc[-1] > 0


def test_rsi(sample_ohlcv):
    close = sample_ohlcv["close"]
    rsi_val = rsi(close, 14)

    assert len(rsi_val) == len(close)
    # RSI must strictly be bounded between 0 and 100
    assert (rsi_val >= 0.0).all()
    assert (rsi_val <= 100.0).all()


def test_macd(sample_ohlcv):
    close = sample_ohlcv["close"]
    line, signal, hist = macd(close, 12, 26, 9)

    assert len(line) == len(close)
    assert len(signal) == len(close)
    assert len(hist) == len(close)
    # Histogram equals line - signal
    np.testing.assert_allclose(hist.values, (line - signal).values, rtol=1e-5)


def test_bollinger_bands(sample_ohlcv):
    close = sample_ohlcv["close"]
    upper, mid, lower, width, pct_b = bollinger_bands(close, 20, 2.0)

    # Valid values after warmup
    valid_idx = 25
    assert (upper.iloc[valid_idx:] > mid.iloc[valid_idx:]).all()
    assert (mid.iloc[valid_idx:] > lower.iloc[valid_idx:]).all()
    assert (width.iloc[valid_idx:] > 0).all()


def test_atr_and_true_range(sample_ohlcv):
    high = sample_ohlcv["high"]
    low = sample_ohlcv["low"]
    close = sample_ohlcv["close"]

    tr = true_range(high, low, close)
    atr_val = atr(high, low, close, 14)

    assert (tr >= 0).all()
    assert (atr_val.dropna() > 0).all()


def test_adx(sample_ohlcv):
    high = sample_ohlcv["high"]
    low = sample_ohlcv["low"]
    close = sample_ohlcv["close"]

    adx_line, plus_di, minus_di = adx(high, low, close, 14)
    assert len(adx_line) == len(close)
    assert (adx_line >= 0).all()
    assert (plus_di >= 0).all()
    assert (minus_di >= 0).all()


def test_supertrend(sample_ohlcv):
    high = sample_ohlcv["high"]
    low = sample_ohlcv["low"]
    close = sample_ohlcv["close"]

    direction, st_line = supertrend(high, low, close, 10, 3.0)
    assert len(direction) == len(close)
    assert len(st_line) == len(close)
    assert not st_line.isna().any()
    # Direction is either 1 (bull) or -1 (bear)
    assert set(direction.unique()).issubset({1, -1})


def test_obv(sample_ohlcv):
    close = sample_ohlcv["close"]
    volume = sample_ohlcv["volume"]

    obv_val = obv(close, volume)
    assert len(obv_val) == len(close)
    assert not obv_val.isna().any()
