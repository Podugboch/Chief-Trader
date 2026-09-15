"""Market Regime Classifier.

Analyzes market dynamics (trend strength, volatility clusters, momentum)
to categorize market conditions without overfitting. Enables the bot to
adapt strategies (e.g. Trend Following vs Mean Reversion vs Stand Aside).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from bot.indicators.technical import adx, atr, bollinger_bands, ema, macd


class MarketRegime(str, Enum):
    STRONG_BULL = "STRONG_BULL"
    WEAK_BULL = "WEAK_BULL"
    STRONG_BEAR = "STRONG_BEAR"
    WEAK_BEAR = "WEAK_BEAR"
    RANGING_LOW_VOL = "RANGING_LOW_VOL"
    RANGING_NORMAL = "RANGING_NORMAL"
    HIGH_VOLATILITY_CHOP = "HIGH_VOLATILITY_CHOP"  # Danger zone


class StrategyMode(str, Enum):
    TREND_FOLLOWING = "TREND_FOLLOWING"
    MEAN_REVERSION = "MEAN_REVERSION"
    STAND_ASIDE = "STAND_ASIDE"


@dataclass
class RegimeAnalysisResult:
    regime: MarketRegime
    strategy_mode: StrategyMode
    adx_value: float
    atr_value: float
    atr_percent: float  # ATR / Close
    volatility_zscore: float
    trend_bias: str  # "BULLISH", "BEARISH", "NEUTRAL"
    summary: str


class MarketRegimeClassifier:
    """Classifies market conditions based on multi-indicator statistical criteria."""

    def __init__(
        self,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        high_vol_atr_mult: float = 2.5,
    ):
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.high_vol_atr_mult = high_vol_atr_mult

    def classify(self, df: pd.DataFrame) -> RegimeAnalysisResult:
        """Classify the latest regime from an OHLCV dataframe.

        DataFrame must contain columns: ['open', 'high', 'low', 'close', 'volume'].
        """
        if len(df) < 50:
            return RegimeAnalysisResult(
                regime=MarketRegime.RANGING_NORMAL,
                strategy_mode=StrategyMode.STAND_ASIDE,
                adx_value=0.0,
                atr_value=0.0,
                atr_percent=0.0,
                volatility_zscore=0.0,
                trend_bias="NEUTRAL",
                summary="Insufficient data points for robust regime classification (< 50 bars)",
            )

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Calculate Indicators
        adx_line, plus_di, minus_di = adx(high, low, close, period=14)
        atr_val = atr(high, low, close, period=14)
        ema_20 = ema(close, 20)
        ema_50 = ema(close, 50)
        ema_200 = ema(close, min(200, len(df) - 1))
        macd_line, sig_line, hist = macd(close)

        curr_close = close.iloc[-1]
        curr_adx = adx_line.iloc[-1]
        curr_plus_di = plus_di.iloc[-1]
        curr_minus_di = minus_di.iloc[-1]
        curr_atr = atr_val.iloc[-1]
        curr_ema20 = ema_20.iloc[-1]
        curr_ema50 = ema_50.iloc[-1]
        curr_ema200 = ema_200.iloc[-1]
        curr_hist = hist.iloc[-1]

        # Volatility assessment: compute ATR z-score against 50-period rolling ATR
        atr_rolling_mean = atr_val.rolling(50).mean().iloc[-1]
        atr_rolling_std = atr_val.rolling(50).std().iloc[-1]
        vol_zscore = (
            (curr_atr - atr_rolling_mean) / atr_rolling_std
            if atr_rolling_std and not np.isnan(atr_rolling_std) and atr_rolling_std > 0
            else 0.0
        )
        atr_pct = (curr_atr / curr_close) * 100.0

        # Check for Extreme / Danger Volatility Spike
        if vol_zscore > self.high_vol_atr_mult or (curr_atr > 2.5 * atr_rolling_mean):
            return RegimeAnalysisResult(
                regime=MarketRegime.HIGH_VOLATILITY_CHOP,
                strategy_mode=StrategyMode.STAND_ASIDE,
                adx_value=float(curr_adx),
                atr_value=float(curr_atr),
                atr_percent=float(atr_pct),
                volatility_zscore=float(vol_zscore),
                trend_bias="NEUTRAL",
                summary="High volatility shock detected. Circuit breaker advises STAND_ASIDE.",
            )

        # Bullish vs Bearish Alignment
        bullish_alignment = (curr_close > curr_ema20) and (curr_ema20 > curr_ema50)
        bearish_alignment = (curr_close < curr_ema20) and (curr_ema20 < curr_ema50)

        # 1. Trending Regimes (ADX >= 25)
        if curr_adx >= self.adx_trend_threshold:
            if curr_plus_di > curr_minus_di and bullish_alignment:
                regime = MarketRegime.STRONG_BULL
                mode = StrategyMode.TREND_FOLLOWING
                bias = "BULLISH"
                summary = f"Strong Bull Trend (ADX {curr_adx:.1f}, +DI > -DI, Price > EMA20 > EMA50)"
            elif curr_minus_di > curr_plus_di and bearish_alignment:
                regime = MarketRegime.STRONG_BEAR
                mode = StrategyMode.TREND_FOLLOWING
                bias = "BEARISH"
                summary = f"Strong Bear Trend (ADX {curr_adx:.1f}, -DI > +DI, Price < EMA20 < EMA50)"
            elif curr_plus_di > curr_minus_di:
                regime = MarketRegime.WEAK_BULL
                mode = StrategyMode.TREND_FOLLOWING
                bias = "BULLISH"
                summary = f"Moderate Bullish momentum (ADX {curr_adx:.1f})"
            else:
                regime = MarketRegime.WEAK_BEAR
                mode = StrategyMode.TREND_FOLLOWING
                bias = "BEARISH"
                summary = f"Moderate Bearish momentum (ADX {curr_adx:.1f})"

        # 2. Ranging / Consolidation Regimes (ADX < 25)
        else:
            if curr_adx < self.adx_range_threshold:
                regime = MarketRegime.RANGING_LOW_VOL
                mode = StrategyMode.MEAN_REVERSION
                bias = "NEUTRAL"
                summary = f"Low Volatility Range (ADX {curr_adx:.1f} < 20). Optimal for Mean Reversion."
            else:
                regime = MarketRegime.RANGING_NORMAL
                mode = StrategyMode.MEAN_REVERSION
                bias = "NEUTRAL"
                summary = f"Normal Range Consolidation (ADX {curr_adx:.1f})."

        return RegimeAnalysisResult(
            regime=regime,
            strategy_mode=mode,
            adx_value=float(curr_adx),
            atr_value=float(curr_atr),
            atr_percent=float(atr_pct),
            volatility_zscore=float(vol_zscore),
            trend_bias=bias,
            summary=summary,
        )
