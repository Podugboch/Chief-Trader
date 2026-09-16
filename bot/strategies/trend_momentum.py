"""Trend-Following & Momentum Strategy.

Combines EMA Ribbon alignment (9/21/50), MACD histogram momentum,
volume filter, and dynamic ATR stops to capture sustained crypto trends.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from bot.strategies.base import BaseStrategy, TradeSignal, SignalAction
from bot.indicators.technical import (
    ema,
    sma,
    macd,
    rsi,
    atr,
    supertrend,
)


class TrendMomentumStrategy(BaseStrategy):
    """Trend-following strategy leveraging EMA Ribbon, MACD, and ATR stops."""

    def __init__(
        self,
        ema_fast: int = 9,
        ema_mid: int = 21,
        ema_slow: int = 50,
        atr_period: int = 14,
        stop_atr_mult: float = 2.0,
        tp_atr_mult: float = 3.5,
        min_volume_ratio: float = 1.0,
    ):
        super().__init__(name="TrendMomentumStrategy")
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.min_volume_ratio = min_volume_ratio

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-computes all indicator columns required for execution."""
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        df["ema_fast"] = ema(close, self.ema_fast)
        df["ema_mid"] = ema(close, self.ema_mid)
        df["ema_slow"] = ema(close, self.ema_slow)

        macd_line, sig_line, hist = macd(close, 12, 26, 9)
        df["macd"] = macd_line
        df["macd_signal"] = sig_line
        df["macd_hist"] = hist

        df["rsi"] = rsi(close, 14)
        df["atr"] = atr(high, low, close, self.atr_period)
        df["vol_sma"] = sma(volume, 20)
        df["vol_ratio"] = volume / df["vol_sma"].replace(0, np.nan)

        st_dir, st_line = supertrend(high, low, close, 10, 3.0)
        df["supertrend_dir"] = st_dir
        df["supertrend_val"] = st_line

        from bot.indicators.technical import vwap
        df["vwap"] = vwap(high, low, close, volume)
        df["ema_200"] = ema(close, min(200, len(df) - 1))

        return df

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """Evaluate the latest bar and return a TradeSignal."""
        if len(df) < max(self.ema_slow, 50):
            return TradeSignal(
                timestamp=df.index[-1] if len(df) > 0 else None,
                symbol=symbol,
                action=SignalAction.HOLD,
                price=float(df["close"].iloc[-1]) if len(df) > 0 else 0.0,
                strategy_name=self.name,
            )

        # Prepare indicators if not already present
        if "ema_fast" not in df.columns or "vwap" not in df.columns:
            df = self.prepare_indicators(df)

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        curr_price = float(curr["close"])
        curr_atr = float(curr["atr"]) if not np.isnan(curr["atr"]) and curr["atr"] > 0 else (curr_price * 0.02)
        curr_ts = df.index[-1]

        # 1. Check Exit / Close conditions
        # Bearish breakdown: Fast crosses below Mid OR Supertrend turns Bearish OR MACD negative divergence
        is_bearish_cross = (prev["ema_fast"] >= prev["ema_mid"]) and (curr["ema_fast"] < curr["ema_mid"])
        supertrend_bearish = curr["supertrend_dir"] == -1 and prev["supertrend_dir"] == 1

        if is_bearish_cross or supertrend_bearish:
            return TradeSignal(
                timestamp=curr_ts,
                symbol=symbol,
                action=SignalAction.CLOSE,
                price=curr_price,
                strategy_name=self.name,
                metadata={"reason": "Trend breakdown (EMA cross or Supertrend flip)"},
            )

        # 2. Institutional Framework Checks
        from bot.indicators.technical import detect_dow_market_structure, detect_wyckoff_phase
        dow = detect_dow_market_structure(df["high"], df["low"], df["close"], lookback=40)
        wyckoff = detect_wyckoff_phase(df["close"], df["high"], df["low"], df["volume"], lookback=50)

        # Dow filter: Do not buy if in confirmed downtrend unless active Break of Structure
        if dow.trend == "DOWNTREND" and not dow.is_break_of_structure:
            return TradeSignal(
                timestamp=curr_ts,
                symbol=symbol,
                action=SignalAction.HOLD,
                price=curr_price,
                strategy_name=self.name,
                metadata={"reason": "Stand aside: Dow Theory Downtrend (LH+LL)"},
            )

        # Wyckoff filter: Avoid Markdown phase
        if wyckoff.phase == "MARKDOWN":
            return TradeSignal(
                timestamp=curr_ts,
                symbol=symbol,
                action=SignalAction.HOLD,
                price=curr_price,
                strategy_name=self.name,
                metadata={"reason": "Stand aside: Wyckoff Markdown distribution in progress"},
            )

        # 3. Check Long Entry conditions
        # Ribbon Alignment: Fast > Mid > Slow
        ribbon_bullish = curr["ema_fast"] > curr["ema_mid"] > curr["ema_slow"]
        # VWAP confirmation: Price trading above institutional VWAP
        vwap_bullish = curr_price >= curr["vwap"]
        # Momentum check: MACD histogram positive and expanding or positive crossover
        macd_bullish = (curr["macd_hist"] > 0)
        # Momentum trigger: Recent crossover or bounce off mid EMA
        recent_cross = (
            (prev["ema_fast"] <= prev["ema_mid"] and curr["ema_fast"] > curr["ema_mid"])
            or (prev["close"] <= curr["ema_mid"] and curr_price > curr["ema_mid"])
            or (curr_price > curr["ema_fast"] and prev["close"] <= prev["ema_fast"])
        )
        # Healthy RSI: between 45 and 70 (not overbought)
        rsi_healthy = 45.0 <= curr["rsi"] <= 72.0
        # Volume check: above minimum ratio threshold
        vol_healthy = curr["vol_ratio"] >= self.min_volume_ratio
        # Supertrend confirmation
        st_bullish = curr["supertrend_dir"] == 1

        if ribbon_bullish and vwap_bullish and macd_bullish and recent_cross and rsi_healthy and vol_healthy and st_bullish:
            stop_loss = curr_price - (self.stop_atr_mult * curr_atr)
            take_profit = curr_price + (self.tp_atr_mult * curr_atr)
            confidence = min(1.0, 0.7 + (0.15 if curr["vol_ratio"] > 1.5 else 0.0) + (0.15 if curr["macd_hist"] > prev["macd_hist"] else 0.0))

            return TradeSignal(
                timestamp=curr_ts,
                symbol=symbol,
                action=SignalAction.BUY,
                price=curr_price,
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                confidence=round(confidence, 2),
                strategy_name=self.name,
                metadata={
                    "atr": curr_atr,
                    "rsi": float(curr["rsi"]),
                    "macd_hist": float(curr["macd_hist"]),
                    "vol_ratio": float(curr["vol_ratio"]),
                    "vwap": float(curr["vwap"]),
                    "dow_trend": dow.trend,
                },
            )

        return TradeSignal(
            timestamp=curr_ts,
            symbol=symbol,
            action=SignalAction.HOLD,
            price=curr_price,
            strategy_name=self.name,
        )
