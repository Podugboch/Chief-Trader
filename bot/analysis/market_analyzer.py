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
    vwap,
    rolling_vwap,
    stochastic_oscillator,
    fibonacci_retracement,
    FibonacciLevels,
    detect_divergence,
    detect_dow_market_structure,
    DowMarketStructure,
    detect_wyckoff_phase,
    WyckoffAnalysis,
    detect_chart_patterns,
    ChartPatternResult,
    detect_elliott_wave_context,
    ElliottWaveContext,
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
    fib_0: float = 0.0
    fib_236: float = 0.0
    fib_382: float = 0.0
    fib_500: float = 0.0
    fib_618: float = 0.0
    fib_786: float = 0.0
    fib_100: float = 0.0
    nearest_fib_name: str = ""
    is_near_golden_pocket: bool = False


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
    dow_structure: Optional[DowMarketStructure] = None
    wyckoff: Optional[WyckoffAnalysis] = None
    patterns: Optional[ChartPatternResult] = None
    elliott_wave: Optional[ElliottWaveContext] = None
    fibonacci: Optional[FibonacciLevels] = None


class MarketAnalyzer:
    """Comprehensive multi-factor market analyzer with institutional TA suite."""

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

        # 1. Moving Averages & Trend
        ema9 = ema(close, 9)
        ema21 = ema(close, 21)
        ema50 = ema(close, 50)
        ema200 = ema(close, min(200, len(df) - 1))
        st_dir, st_val = supertrend(high, low, close)
        vwap_vals = vwap(high, low, close, volume)
        curr_vwap = float(vwap_vals.iloc[-1])

        # 2. Oscillators & Momentum
        rsi_vals = rsi(close, 14)
        curr_rsi = float(rsi_vals.iloc[-1])
        macd_line, sig_line, macd_hist = macd(close, 12, 26, 9)
        curr_hist = float(macd_hist.iloc[-1])
        prev_hist = float(macd_hist.iloc[-2])
        stoch_k, stoch_d = stochastic_oscillator(high, low, close, 14, 3, 3)
        curr_stoch_k = float(stoch_k.iloc[-1])
        curr_stoch_d = float(stoch_d.iloc[-1])
        prev_stoch_k = float(stoch_k.iloc[-2])
        prev_stoch_d = float(stoch_d.iloc[-2])

        # 3. Divergence Detection
        rsi_bull_div, rsi_bear_div = detect_divergence(close, rsi_vals, lookback=30)
        macd_bull_div, macd_bear_div = detect_divergence(close, macd_hist, lookback=30)

        # 4. Volatility Indicators
        bb_up, bb_mid, bb_low, bb_width, bb_pct_b = bollinger_bands(close, 20, 2.0)
        atr_vals = atr(high, low, close, 14)
        curr_atr = float(atr_vals.iloc[-1])

        # 5. Volume Indicators
        vol_sma20 = sma(volume, 20)
        curr_vol = float(volume.iloc[-1])
        avg_vol = float(vol_sma20.iloc[-1]) if not np.isnan(vol_sma20.iloc[-1]) and vol_sma20.iloc[-1] > 0 else 1.0
        vol_ratio = curr_vol / avg_vol
        obv_vals = obv(close, volume)
        obv_ema = ema(obv_vals, 20)
        curr_obv = float(obv_vals.iloc[-1])
        curr_obv_ema = float(obv_ema.iloc[-1])

        # 6. Fibonacci & S/R Pivots
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

        fib = fibonacci_retracement(high, low, curr_price, lookback=50)

        levels = SupportResistanceLevels(
            resistance_2=r2,
            resistance_1=r1,
            pivot=pivot,
            support_1=s1,
            support_2=s2,
            recent_swing_high=swing_high,
            recent_swing_low=swing_low,
            fib_0=fib.fib_0,
            fib_236=fib.fib_236,
            fib_382=fib.fib_382,
            fib_500=fib.fib_500,
            fib_618=fib.fib_618,
            fib_786=fib.fib_786,
            fib_100=fib.fib_100,
            nearest_fib_name=str(fib.nearest_level_name),
            is_near_golden_pocket=bool(fib.is_near_golden_pocket),
        )

        # 7. Institutional Frameworks: Dow, Wyckoff, Chart Patterns, Elliott Wave
        dow = detect_dow_market_structure(high, low, close, lookback=40)
        wyckoff = detect_wyckoff_phase(close, high, low, volume, lookback=50)
        patterns = detect_chart_patterns(high, low, close, lookback=35)
        elliott = detect_elliott_wave_context(close, high, low, lookback=50)
        regime_res = self.regime_classifier.classify(df)

        # 8. Rigorous Institutional Composite Scoring (-100 to +100)
        score = 0.0
        observations: List[str] = []

        c_ema9 = float(ema9.iloc[-1])
        c_ema21 = float(ema21.iloc[-1])
        c_ema50 = float(ema50.iloc[-1])
        c_ema200 = float(ema200.iloc[-1])

        # (a) Moving Average Ribbon & 200 EMA Benchmark
        if curr_price > c_ema9 > c_ema21 > c_ema50:
            score += 20.0
            observations.append("Bullish moving average ribbon alignment (Price > EMA9 > EMA21 > EMA50)")
        elif curr_price < c_ema9 < c_ema21 < c_ema50:
            score -= 20.0
            observations.append("Bearish moving average ribbon alignment (Price < EMA9 < EMA21 < EMA50)")

        if curr_price > c_ema200:
            score += 10.0
            observations.append(f"Price (${curr_price:,.2f}) trading above macro 200 EMA (${c_ema200:,.2f})")
        else:
            score -= 10.0
            observations.append(f"Price (${curr_price:,.2f}) trading below macro 200 EMA (${c_ema200:,.2f}) — Macro Bear Bias")

        # (b) Institutional VWAP Alignment
        if curr_price > curr_vwap:
            score += 10.0
            observations.append(f"Price above institutional VWAP (${curr_vwap:,.2f}) — buyers in control")
        else:
            score -= 10.0
            observations.append(f"Price below institutional VWAP (${curr_vwap:,.2f}) — sellers in control")

        # (c) Dow Theory Structure
        if dow.structure == "HH_HL":
            score += 15.0
            observations.append(dow.summary)
        elif dow.structure == "LH_LL":
            if dow.is_break_of_structure:
                score += 10.0
                observations.append(dow.summary)
            else:
                score -= 15.0
                observations.append(dow.summary)

        # (d) Wyckoff Method
        if wyckoff.phase == "MARKUP":
            score += 15.0
            observations.append(wyckoff.summary)
        elif wyckoff.has_spring:
            score += 20.0
            observations.append(wyckoff.summary)
        elif wyckoff.phase == "MARKDOWN":
            score -= 20.0
            observations.append(wyckoff.summary)
        elif wyckoff.phase == "DISTRIBUTION":
            score -= 15.0
            observations.append(wyckoff.summary)

        # (e) Divergences (RSI & MACD)
        if rsi_bull_div or macd_bull_div:
            score += 20.0
            observations.append("BULLISH DIVERGENCE: Price making lower lows while momentum forms higher lows (reversal signal).")
        elif rsi_bear_div or macd_bear_div:
            score -= 20.0
            observations.append("BEARISH DIVERGENCE: Price making higher highs while momentum forms lower highs (exhaustion signal).")

        # (f) Stochastic Oscillator
        stoch_oversold_cross = (prev_stoch_k <= prev_stoch_d) and (curr_stoch_k > curr_stoch_d) and (curr_stoch_k < 25.0)
        stoch_overbought_cross = (prev_stoch_k >= prev_stoch_d) and (curr_stoch_k < curr_stoch_d) and (curr_stoch_k > 75.0)
        if stoch_oversold_cross:
            score += 15.0
            observations.append(f"Stochastic Oscillator bullish crossover in oversold zone (%K: {curr_stoch_k:.1f} > %D: {curr_stoch_d:.1f})")
        elif stoch_overbought_cross:
            score -= 15.0
            observations.append(f"Stochastic Oscillator bearish crossover in overbought zone (%K: {curr_stoch_k:.1f} < %D: {curr_stoch_d:.1f})")

        # (g) Fibonacci Retracement Golden Pocket Support
        if fib.is_near_golden_pocket:
            if curr_price >= fib.fib_618:
                score += 10.0
                observations.append(f"Fibonacci Support: Price testing Golden Pocket zone (0.618 at ${fib.fib_618:,.2f})")
            else:
                score -= 5.0
                observations.append(f"Fibonacci: Price breaking below Golden Pocket (${fib.fib_618:,.2f}) towards 0.786")

        # (h) Classical Chart Patterns
        if patterns.is_bullish:
            score += 20.0
            observations.append(patterns.summary)
        elif patterns.is_bearish:
            score -= 15.0
            observations.append(patterns.summary)

        # (i) Elliott Wave Context
        if elliott.phase == "IMPULSE_BEAR":
            score -= 20.0
            observations.append(elliott.summary)
        elif elliott.phase == "IMPULSE_BULL":
            score += 15.0
            observations.append(elliott.summary)

        # (j) MACD Histogram Momentum
        if curr_hist > 0:
            if curr_hist > prev_hist:
                score += 10.0
                observations.append(f"MACD Histogram positive and expanding (+{curr_hist:.2f})")
            else:
                score += 5.0
        else:
            if curr_hist < prev_hist:
                score -= 10.0
                observations.append(f"MACD Histogram negative and expanding ({curr_hist:.2f})")
            else:
                score -= 5.0

        # (k) Supertrend & Volume
        curr_st_dir = int(st_dir.iloc[-1])
        if curr_st_dir == 1:
            score += 10.0
            observations.append(f"Supertrend is BULLISH (Support trail at ${st_val.iloc[-1]:,.2f})")
        else:
            score -= 10.0
            observations.append(f"Supertrend is BEARISH (Resistance trail at ${st_val.iloc[-1]:,.2f})")

        if curr_obv > curr_obv_ema:
            observations.append("On-Balance Volume (OBV) above its 20-period trendline (accumulation)")
        else:
            observations.append("On-Balance Volume (OBV) below its 20-period trendline (distribution)")

        # Clamp composite score between -100 and +100
        composite_score = max(-100.0, min(100.0, score))

        # Determine Verdict strictly
        if composite_score >= 45.0:
            verdict = "STRONG_BUY"
        elif composite_score >= 15.0:
            verdict = "BUY"
        elif composite_score <= -45.0:
            verdict = "STRONG_SELL"
        elif composite_score <= -15.0:
            verdict = "SELL"
        else:
            verdict = "NEUTRAL"

        indicators_summary = {
            "price": float(curr_price),
            "ema9": float(c_ema9),
            "ema21": float(c_ema21),
            "ema50": float(c_ema50),
            "ema200": float(c_ema200),
            "vwap": float(curr_vwap),
            "vwap_bias": "BULLISH" if curr_price > curr_vwap else "BEARISH",
            "rsi": float(curr_rsi),
            "stoch_k": float(curr_stoch_k),
            "stoch_d": float(curr_stoch_d),
            "rsi_bullish_div": bool(rsi_bull_div),
            "rsi_bearish_div": bool(rsi_bear_div),
            "macd": float(macd_line.iloc[-1]),
            "macd_signal": float(sig_line.iloc[-1]),
            "macd_hist": float(curr_hist),
            "macd_bullish_div": bool(macd_bull_div),
            "bollinger_upper": float(bb_up.iloc[-1]),
            "bollinger_middle": float(bb_mid.iloc[-1]),
            "bollinger_lower": float(bb_low.iloc[-1]),
            "bollinger_bandwidth": float(bb_width.iloc[-1]),
            "bollinger_percent_b": float(bb_pct_b.iloc[-1]),
            "atr": float(curr_atr),
            "atr_percent": float((curr_atr / curr_price) * 100.0),
            "adx": float(regime_res.adx_value),
            "supertrend_direction": "BULLISH" if curr_st_dir == 1 else "BEARISH",
            "supertrend_value": float(st_val.iloc[-1]),
            "volume_ratio": float(vol_ratio),
            "dow_trend": str(dow.trend),
            "dow_structure": str(dow.structure),
            "dow_bos": bool(dow.is_break_of_structure),
            "wyckoff_phase": str(wyckoff.phase),
            "wyckoff_spring": bool(wyckoff.has_spring),
            "wyckoff_safe_to_buy": bool(wyckoff.is_safe_to_buy),
            "chart_pattern": str(patterns.pattern_name),
            "chart_pattern_bullish": bool(patterns.is_bullish),
            "elliott_phase": str(elliott.phase),
            "elliott_safe_to_buy": bool(elliott.is_safe_for_long),
            "fib_nearest": str(fib.nearest_level_name),
            "fib_golden_pocket": bool(fib.is_near_golden_pocket),
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
            dow_structure=dow,
            wyckoff=wyckoff,
            patterns=patterns,
            elliott_wave=elliott,
            fibonacci=fib,
        )
