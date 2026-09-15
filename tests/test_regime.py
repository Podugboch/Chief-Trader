"""Tests for Market Regime Classifier."""

import numpy as np
import pandas as pd
import pytest

from bot.analysis.regime import MarketRegimeClassifier, MarketRegime, StrategyMode


def create_bull_market(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    # Steady upward progression
    close = np.linspace(50000, 70000, n)
    high = close + 100
    low = close - 100
    open_p = close - 50
    volume = np.ones(n) * 500
    return pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def create_ranging_market(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    # Low-volatility horizontal consolidation around 60000
    noise = np.random.normal(0, 10, n)
    close = 60000.0 + noise
    high = close + 15.0
    low = close - 15.0
    open_p = close + np.random.normal(0, 5, n)
    volume = np.ones(n) * 200
    return pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def test_classify_bull_market():
    df = create_bull_market(100)
    classifier = MarketRegimeClassifier()
    res = classifier.classify(df)

    assert res.regime in (MarketRegime.STRONG_BULL, MarketRegime.WEAK_BULL)
    assert res.strategy_mode == StrategyMode.TREND_FOLLOWING
    assert res.trend_bias == "BULLISH"


def test_classify_ranging_market():
    df = create_ranging_market(100)
    classifier = MarketRegimeClassifier()
    res = classifier.classify(df)

    assert res.regime in (MarketRegime.RANGING_LOW_VOL, MarketRegime.RANGING_NORMAL)
    assert res.strategy_mode == StrategyMode.MEAN_REVERSION


def test_classify_insufficient_data():
    dates = pd.date_range("2026-01-01", periods=20, freq="1h")
    df = pd.DataFrame(
        {"open": [100]*20, "high": [105]*20, "low": [95]*20, "close": [100]*20, "volume": [10]*20},
        index=dates,
    )
    classifier = MarketRegimeClassifier()
    res = classifier.classify(df)
    assert res.strategy_mode == StrategyMode.STAND_ASIDE
