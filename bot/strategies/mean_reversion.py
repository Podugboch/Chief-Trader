"""Mean-Reversion Strategy for Ranging & Consolidating Markets.

Trades statistical overextensions using Bollinger Bands, RSI turnarounds,
and dynamic volatility targets.
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
)


class MeanReversionStrategy(BaseStrategy):
    """Mean-reversion strategy designed for range-bound regimes."""

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        stop_atr_mult: float = 1.5,
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

        return df

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate signal based on mean reversion criteria."""
        if len(df) < max(self.bb_period, 30):
            return TradeSignal(
                timestamp=df.index[-1] if len(df) > 0 else None,
                symbol=symbol,
                action=SignalAction.HOLD,
                price=float(df["close"].iloc[-1]) if len(df) > 0 else 0.0,
                strategy_name=self.name,
            )

        if "bb_lower" not in df.columns:
            df = self.prepare_indicators(df)

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        curr_price = float(curr["close"])
        curr_atr = float(curr["atr"]) if not np.isnan(curr["atr"]) and curr["atr"] > 0 else (curr_price * 0.02)
        curr_ts = df.index[-1]

        # 1. Exit / Close Condition
        # Take profit when price reaches or crosses the middle Bollinger Band or RSI becomes overbought
        reverted_to_mean = curr_price >= curr["bb_mid"] and prev["close"] < prev["bb_mid"]
        hit_upper = curr_price >= curr["bb_upper"]
        overbought = curr["rsi"] >= self.rsi_overbought

        if reverted_to_mean or hit_upper or overbought:
            return TradeSignal(
                timestamp=curr_ts,
                symbol=symbol,
                action=SignalAction.CLOSE,
                price=curr_price,
                strategy_name=self.name,
                metadata={"reason": "Reversion target reached (BB middle/upper or RSI overbought)"},
            )

        # 2. Entry Long Condition
        # Condition 1: Market is not in a fierce runaway trend (ADX < 28)
        not_trending = curr["adx"] < 28.0
        # Condition 2: Price touched or is near lower band (%B <= 0.15)
        near_lower_band = curr["bb_pct_b"] <= 0.15 or curr_price <= curr["bb_lower"]
        # Condition 3: RSI is recovering from oversold (RSI increasing)
        rsi_recovering = (prev["rsi"] <= self.rsi_oversold) and (curr["rsi"] > prev["rsi"])
        # Candle shape: Bullish rejection wick or green candle
        bullish_candle = curr["close"] > curr["open"]

        if not_trending and near_lower_band and rsi_recovering and bullish_candle:
            stop_loss = curr_price - (self.stop_atr_mult * curr_atr)
            take_profit = float(curr["bb_mid"])  # First target is the mean
            confidence = 0.75

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
                    "bb_lower": float(curr["bb_lower"]),
                    "bb_mid": float(curr["bb_mid"]),
                    "rsi": float(curr["rsi"]),
                    "adx": float(curr["adx"]),
                },
            )

        return TradeSignal(
            timestamp=curr_ts,
            symbol=symbol,
            action=SignalAction.HOLD,
            price=curr_price,
            strategy_name=self.name,
        )
