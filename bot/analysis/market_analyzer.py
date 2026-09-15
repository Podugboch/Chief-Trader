"""Top-Notch Market Analyzer.

Provides comprehensive multi-factor technical analysis, combining
trend, momentum, volatility, volume, and support/resistance levels
into a synthesized score and actionable report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

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
)
from bot.analysis.regime import MarketRegimeClassifier, RegimeAnalysisResult


@dataclass
class SupportResistanceLevels:
    resistance_2: float
    resistance_1: float
    pivot: float
    support_1: float
    support_2: float
    recent_swing_high: float
    recent_swing_low: float


@dataclass
class MarketAnalysisReport:
    symbol: str
    timeframe: str
    latest_price: float
    regime: RegimeAnalysisResult
    composite_score: float  # -100.0 (extreme bearish) to +100.0 (extreme bullish)
    verdict: str  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    indicators: Dict[str, Any]
    levels: SupportResistanceLevels
    key_observations: List[str]


class MarketAnalyzer:
    """Comprehensive multi-factor market analyzer."""

    def __init__(self):
        self.regime_classifier = MarketRegimeClassifier()

    def analyze(self, df: pd.DataFrame, symbol: str = "BTCUSDT", timeframe: str = "1h") -> MarketAnalysisReport:
        """Run full spectrum technical analysis on an OHLCV DataFrame."""
        if len(df) < 50:
            raise ValueError(f"Need at least 50 bars for analysis, got {len(df)}")

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        curr_price = float(close.iloc[-1])

        # 1. Trend Indicators
        ema9 = ema(close, 9)
        ema21 = ema(close, 21)
        ema50 = ema(close, 50)
        ema200 = ema(close, min(200, len(df) - 1))
        st_dir, st_val = supertrend(high, low, close)

        # 2. Momentum Indicators
        rsi_vals = rsi(close, 14)
        curr_rsi = float(rsi_vals.iloc[-1])
        macd_line, sig_line, macd_hist = macd(close, 12, 26, 9)

        # 3. Volatility Indicators
        bb_up, bb_mid, bb_low, bb_width, bb_pct_b = bollinger_bands(close, 20, 2.0)
        atr_vals = atr(high, low, close, 14)
        curr_atr = float(atr_vals.iloc[-1])

        # 4. Volume Indicators
        vol_sma20 = sma(volume, 20)
        curr_vol = float(volume.iloc[-1])
        avg_vol = float(vol_sma20.iloc[-1]) if not np.isnan(vol_sma20.iloc[-1]) and vol_sma20.iloc[-1] > 0 else 1.0
        vol_ratio = curr_vol / avg_vol
        obv_vals = obv(close, volume)
        obv_ema = ema(obv_vals, 20)

        # 5. Support & Resistance (Fibonacci & Classical Pivot Points from last bar)
        prev_h = float(high.iloc[-2])
        prev_l = float(low.iloc[-2])
        prev_c = float(close.iloc[-2])
        pivot = (prev_h + prev_l + prev_c) / 3.0
        r1 = (2.0 * pivot) - prev_l
        s1 = (2.0 * pivot) - prev_h
        r2 = pivot + (prev_h - prev_l)
        s2 = pivot - (prev_h - prev_l)
        swing_high = float(high.tail(20).max())
        swing_low = float(low.tail(20).min())

        levels = SupportResistanceLevels(
            resistance_2=r2,
            resistance_1=r1,
            pivot=pivot,
            support_1=s1,
            support_2=s2,
            recent_swing_high=swing_high,
            recent_swing_low=swing_low,
        )

        # 6. Regime Classification
        regime_res = self.regime_classifier.classify(df)

        # 7. Multi-Factor Composite Scoring (-100 to +100)
        score = 0.0
        observations: List[str] = []

        # (a) EMA Trend Alignment (weight: up to 30 points)
        c_ema9 = float(ema9.iloc[-1])
        c_ema21 = float(ema21.iloc[-1])
        c_ema50 = float(ema50.iloc[-1])
        c_ema200 = float(ema200.iloc[-1])

        if curr_price > c_ema9 > c_ema21 > c_ema50:
            score += 30.0
            observations.append("Bullish moving average ribbon alignment (Price > EMA9 > EMA21 > EMA50)")
        elif curr_price < c_ema9 < c_ema21 < c_ema50:
            score -= 30.0
            observations.append("Bearish moving average ribbon alignment (Price < EMA9 < EMA21 < EMA50)")
        elif curr_price > c_ema50:
            score += 15.0
            observations.append("Price trading above medium-term baseline (EMA50)")
        else:
            score -= 15.0
            observations.append("Price trading below medium-term baseline (EMA50)")

        # (b) MACD Momentum (weight: up to 25 points)
        curr_hist = float(macd_hist.iloc[-1])
        prev_hist = float(macd_hist.iloc[-2])
        if curr_hist > 0:
            if curr_hist > prev_hist:
                score += 25.0
                observations.append(f"MACD Histogram positive and expanding (+{curr_hist:.2f})")
            else:
                score += 15.0
                observations.append("MACD Histogram positive but momentum decelerating")
        else:
            if curr_hist < prev_hist:
                score -= 25.0
                observations.append(f"MACD Histogram negative and expanding ({curr_hist:.2f})")
            else:
                score -= 15.0
                observations.append("MACD Histogram negative with contracting bearish momentum")

        # (c) RSI Momentum (weight: up to 20 points)
        if 50.0 <= curr_rsi <= 65.0:
            score += 20.0
            observations.append(f"RSI({curr_rsi:.1f}) in healthy bullish expansion zone")
        elif curr_rsi > 70.0:
            score += 5.0
            observations.append(f"RSI({curr_rsi:.1f}) is Overbought (>70) — caution for pullback")
        elif 35.0 <= curr_rsi < 50.0:
            score -= 15.0
            observations.append(f"RSI({curr_rsi:.1f}) in bearish territory")
        elif curr_rsi < 30.0:
            score -= 5.0
            observations.append(f"RSI({curr_rsi:.1f}) is Oversold (<30) — potential mean reversion bounce")

        # (d) Supertrend Confirmation (weight: up to 15 points)
        curr_st_dir = int(st_dir.iloc[-1])
        if curr_st_dir == 1:
            score += 15.0
            observations.append(f"Supertrend is BULLISH (Support trail at {st_val.iloc[-1]:.2f})")
        else:
            score -= 15.0
            observations.append(f"Supertrend is BEARISH (Resistance trail at {st_val.iloc[-1]:.2f})")

        # (e) Volume Confirmation (weight: up to 10 points)
        curr_obv = float(obv_vals.iloc[-1])
        curr_obv_ema = float(obv_ema.iloc[-1])
        if vol_ratio >= 1.5:
            if score > 0:
                score += 10.0
                observations.append(f"High volume expansion ({vol_ratio:.2f}x avg) confirming upside move")
            else:
                score -= 10.0
                observations.append(f"High volume expansion ({vol_ratio:.2f}x avg) confirming downside pressure")
        if curr_obv > curr_obv_ema:
            observations.append("On-Balance Volume (OBV) above its 20-period trendline (accumulation)")
        else:
            observations.append("On-Balance Volume (OBV) below its 20-period trendline (distribution)")

        # Clamp composite score between -100 and +100
        composite_score = max(-100.0, min(100.0, score))

        # Determine Verdict
        if composite_score >= 50.0:
            verdict = "STRONG_BUY"
        elif composite_score >= 20.0:
            verdict = "BUY"
        elif composite_score <= -50.0:
            verdict = "STRONG_SELL"
        elif composite_score <= -20.0:
            verdict = "SELL"
        else:
            verdict = "NEUTRAL"

        indicators_summary = {
            "price": curr_price,
            "ema9": c_ema9,
            "ema21": c_ema21,
            "ema50": c_ema50,
            "ema200": c_ema200,
            "rsi": curr_rsi,
            "macd": float(macd_line.iloc[-1]),
            "macd_signal": float(sig_line.iloc[-1]),
            "macd_hist": curr_hist,
            "bollinger_upper": float(bb_up.iloc[-1]),
            "bollinger_middle": float(bb_mid.iloc[-1]),
            "bollinger_lower": float(bb_low.iloc[-1]),
            "bollinger_bandwidth": float(bb_width.iloc[-1]),
            "bollinger_percent_b": float(bb_pct_b.iloc[-1]),
            "atr": curr_atr,
            "atr_percent": (curr_atr / curr_price) * 100.0,
            "adx": regime_res.adx_value,
            "supertrend_direction": "BULLISH" if curr_st_dir == 1 else "BEARISH",
            "supertrend_value": float(st_val.iloc[-1]),
            "volume_ratio": vol_ratio,
        }

        return MarketAnalysisReport(
            symbol=symbol,
            timeframe=timeframe,
            latest_price=curr_price,
            regime=regime_res,
            composite_score=composite_score,
            verdict=verdict,
            indicators=indicators_summary,
            levels=levels,
            key_observations=observations,
        )
