"""Mean-Reversion Strategy for Ranging & Consolidating Markets.

Trades high-confluence statistical overextensions using Bollinger Bands,
Stochastic Oscillator, Fibonacci Retracements, Wyckoff accumulation/springs,
and RSI bullish divergences.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from bot.strategies.base import BaseStrategy, TradeSignal, SignalAction
from bot.indicators.technical import (
    bollinger_bands,
    rsi,
    atr,
    adx,
    sma,
    vwap,
    stochastic_oscillator,
    fibonacci_retracement,
    detect_divergence,
    detect_dow_market_structure,
    detect_wyckoff_phase,
    detect_chart_patterns,
    detect_elliott_wave_context,
)


class MeanReversionStrategy(BaseStrategy):
    """Mean-reversion strategy with strict institutional confluence filters."""

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 38.0,
        rsi_overbought: float = 65.0,
        stop_atr_mult: float = 1.8,
    ):
        super().__init__(name="MeanReversionStrategy")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stop_atr_mult = stop_atr_mult

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute indicators for mean reversion."""
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        upper, mid, lower, width, pct_b = bollinger_bands(close, self.bb_period, self.bb_std)
        df["bb_upper"] = upper
        df["bb_mid"] = mid
        df["bb_lower"] = lower
        df["bb_width"] = width
        df["bb_pct_b"] = pct_b

        df["rsi"] = rsi(close, self.rsi_period)
        df["atr"] = atr(high, low, close, 14)
        adx_val, _, _ = adx(high, low, close, 14)
        df["adx"] = adx_val

        stoch_k, stoch_d = stochastic_oscillator(high, low, close, 14, 3, 3)
        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_d
        df["vwap"] = vwap(high, low, close, volume)

        return df

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate signal based on strict mean reversion and reversal criteria."""
        if len(df) < max(self.bb_period, 50):
            return TradeSignal(
                timestamp=df.index[-1] if len(df) > 0 else None,
                symbol=symbol,
                action=SignalAction.HOLD,
                price=float(df["close"].iloc[-1]) if len(df) > 0 else 0.0,
                strategy_name=self.name,
            )

        if "bb_lower" not in df.columns or "stoch_k" not in df.columns:
            df = self.prepare_indicators(df)

        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        curr_price = float(curr["close"])
        curr_atr = float(curr["atr"]) if not np.isnan(curr["atr"]) and curr["atr"] > 0 else (curr_price * 0.02)
        curr_ts = df.index[-1]

        # 1. Exit / Close Condition
        # Exit when price hits upper Bollinger Band target, or becomes severely overbought
        hit_upper = curr_price >= curr["bb_upper"]
        overbought = (curr["rsi"] >= self.rsi_overbought) and (curr["stoch_k"] >= 75.0)

        if hit_upper or overbought:
            return TradeSignal(
                timestamp=curr_ts,
                symbol=symbol,
                action=SignalAction.CLOSE,
                price=curr_price,
                strategy_name=self.name,
                metadata={"reason": "Reversion target reached (BB upper or overbought oscillators)"},
            )

        # 2. Institutional Framework Analysis
        dow = detect_dow_market_structure(high, low, close, lookback=40)
        wyckoff = detect_wyckoff_phase(close, high, low, volume, lookback=50)
        patterns = detect_chart_patterns(high, low, close, lookback=35)
        fib = fibonacci_retracement(high, low, curr_price, lookback=50)
        rsi_bull_div, _ = detect_divergence(close, df["rsi"], lookback=30)
        stoch_bull_div, _ = detect_divergence(close, df["stoch_k"], lookback=30)
        elliott = detect_elliott_wave_context(close, high, low, lookback=50)

        # Rule A: NEVER buy during an active unconfirmed Markdown or Wave 3 / Wave C crash
        if elliott.phase == "IMPULSE_BEAR" and not patterns.is_bullish and not wyckoff.has_spring and not rsi_bull_div:
            return TradeSignal(
                timestamp=curr_ts,
                symbol=symbol,
                action=SignalAction.HOLD,
                price=curr_price,
                strategy_name=self.name,
                metadata={"reason": "Stand aside: Elliott Wave Impulsive Bear Crash"},
            )

        # Rule B: If Dow structure is Lower Lows + Lower Highs or Markdown phase, require structural confirmation
        if (dow.structure == "LH_LL") or (wyckoff.phase == "MARKDOWN"):
            has_structural_reversal = (
                rsi_bull_div
                or stoch_bull_div
                or patterns.pattern_name in ("DOUBLE_BOTTOM", "INVERSE_HEAD_AND_SHOULDERS")
                or wyckoff.has_spring
                or dow.is_break_of_structure
            )
            if not has_structural_reversal:
                return TradeSignal(
                    timestamp=curr_ts,
                    symbol=symbol,
                    action=SignalAction.HOLD,
                    price=curr_price,
                    strategy_name=self.name,
                    metadata={"reason": "Stand aside: Downtrend without bullish divergence or bottom pattern"},
                )

        # Rule C: Confluence Entry Criteria
        near_lower_band = curr["bb_pct_b"] <= 0.15 or curr_price <= curr["bb_lower"]
        near_fib_support = fib.is_near_golden_pocket or (abs(curr_price - fib.fib_786) / curr_price <= 0.015)
        near_pattern_support = patterns.is_bullish

        support_confluence = near_lower_band and (near_fib_support or near_pattern_support or rsi_bull_div or stoch_bull_div or wyckoff.has_spring)

        # Stochastic crossover in oversold
        stoch_oversold_cross = (prev["stoch_k"] <= prev["stoch_d"]) and (curr["stoch_k"] > curr["stoch_d"]) and (curr["stoch_k"] < 35.0)
        rsi_recovering = (prev["rsi"] <= self.rsi_oversold) and (curr["rsi"] > prev["rsi"])
        oscillator_confluence = stoch_oversold_cross or (rsi_recovering and (rsi_bull_div or stoch_bull_div))

        candle_body = curr["close"] - curr["open"]
        lower_wick = min(curr["open"], curr["close"]) - curr["low"]
        candle_range = max(curr["high"] - curr["low"], 1e-6)
        bullish_candle = (curr["close"] > curr["open"]) and (lower_wick / candle_range >= 0.25 or curr["close"] > prev["high"])

        adx_acceptable = (curr["adx"] < 28.0) or wyckoff.has_spring or rsi_bull_div or patterns.is_bullish

        if support_confluence and oscillator_confluence and bullish_candle and adx_acceptable:
            stop_loss = curr_price - (self.stop_atr_mult * curr_atr)
            take_profit = max(float(curr["bb_upper"]), curr_price + (2.5 * curr_atr))
            confidence = 0.80
            if rsi_bull_div or wyckoff.has_spring or patterns.is_bullish:
                confidence = 0.90

            return TradeSignal(
                timestamp=curr_ts,
                symbol=symbol,
                action=SignalAction.BUY,
                price=curr_price,
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                confidence=confidence,
                strategy_name=self.name,
                metadata={
                    "confluence": "MeanReversion+InstitutionalSupport",
                    "dow_trend": dow.trend,
                    "wyckoff_phase": wyckoff.phase,
                    "chart_pattern": patterns.pattern_name,
                    "rsi_bull_div": rsi_bull_div,
                    "fib_nearest": fib.nearest_level_name,
                },
            )

        return TradeSignal(
            timestamp=curr_ts,
            symbol=symbol,
            action=SignalAction.HOLD,
            price=curr_price,
            strategy_name=self.name,
        )
