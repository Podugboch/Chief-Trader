"""Regime-Adaptive Meta Strategy.

Dynamically routes market analysis between Trend-Following and Mean-Reversion
models based on objective market regime classification. Protects capital
by standing aside during extreme volatility clusters or choppiness.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd

from bot.strategies.base import BaseStrategy, TradeSignal, SignalAction
from bot.strategies.trend_momentum import TrendMomentumStrategy
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.analysis.regime import MarketRegimeClassifier, StrategyMode, MarketRegime


class RegimeAdaptiveStrategy(BaseStrategy):
    """Adaptive strategy switching between Trend Following and Mean Reversion."""

    def __init__(
        self,
        trend_strategy: Optional[TrendMomentumStrategy] = None,
        range_strategy: Optional[MeanReversionStrategy] = None,
        regime_classifier: Optional[MarketRegimeClassifier] = None,
    ):
        super().__init__(name="RegimeAdaptiveStrategy")
        self.trend_strat = trend_strategy or TrendMomentumStrategy()
        self.range_strat = range_strategy or MeanReversionStrategy()
        self.classifier = regime_classifier or MarketRegimeClassifier()

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.trend_strat.prepare_indicators(df)
        df = self.range_strat.prepare_indicators(df)
        return df

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        if len(df) < 50:
            return TradeSignal(
                timestamp=df.index[-1] if len(df) > 0 else None,
                symbol=symbol,
                action=SignalAction.HOLD,
                price=float(df["close"].iloc[-1]) if len(df) > 0 else 0.0,
                strategy_name=self.name,
            )

        regime_result = self.classifier.classify(df)
        curr_price = float(df["close"].iloc[-1])
        curr_ts = df.index[-1]

        # 1. Stand aside during violent volatility spikes
        if regime_result.strategy_mode == StrategyMode.STAND_ASIDE:
            return TradeSignal(
                timestamp=curr_ts,
                symbol=symbol,
                action=SignalAction.HOLD,
                price=curr_price,
                strategy_name=self.name,
                metadata={
                    "regime": regime_result.regime.value,
                    "mode": "STAND_ASIDE",
                    "reason": regime_result.summary,
                },
            )

        # 2. Trending Mode -> Trend Momentum Strategy
        if regime_result.strategy_mode == StrategyMode.TREND_FOLLOWING:
            sig = self.trend_strat.generate_signal(df, symbol)
            sig.strategy_name = f"Adaptive[{self.trend_strat.name}]"
            sig.metadata["regime"] = regime_result.regime.value
            sig.metadata["mode"] = "TREND_FOLLOWING"
            return sig

        # 3. Mean Reversion Mode -> Range Strategy with Bear Regime Protection
        if regime_result.strategy_mode == StrategyMode.MEAN_REVERSION:
            # If in macro bear trend, strictly require high-conviction bottom confirmation
            if regime_result.regime in (MarketRegime.STRONG_BEAR, MarketRegime.WEAK_BEAR):
                from bot.indicators.technical import (
                    detect_chart_patterns,
                    detect_wyckoff_phase,
                    detect_divergence,
                    detect_dow_market_structure,
                    rsi,
                )
                close_s = df["close"]
                high_s = df["high"]
                low_s = df["low"]
                vol_s = df["volume"]
                patterns = detect_chart_patterns(high_s, low_s, close_s, lookback=35)
                wyckoff = detect_wyckoff_phase(close_s, high_s, low_s, vol_s, lookback=50)
                dow = detect_dow_market_structure(high_s, low_s, close_s, lookback=40)
                r = df["rsi"] if "rsi" in df.columns else rsi(close_s, 14)
                rsi_bull_div, _ = detect_divergence(close_s, r, lookback=30)

                has_institutional_reversal = (
                    patterns.is_bullish
                    or wyckoff.has_spring
                    or rsi_bull_div
                    or dow.is_break_of_structure
                )
                if not has_institutional_reversal:
                    return TradeSignal(
                        timestamp=curr_ts,
                        symbol=symbol,
                        action=SignalAction.HOLD,
                        price=curr_price,
                        strategy_name=self.name,
                        metadata={
                            "regime": regime_result.regime.value,
                            "mode": "STAND_ASIDE_BEAR",
                            "reason": "Bear regime active: Standing aside to protect capital without confirmed reversal pattern.",
                        },
                    )

            sig = self.range_strat.generate_signal(df, symbol)
            sig.strategy_name = f"Adaptive[{self.range_strat.name}]"
            sig.metadata["regime"] = regime_result.regime.value
            sig.metadata["mode"] = "MEAN_REVERSION"
            return sig

        return TradeSignal(
            timestamp=curr_ts,
            symbol=symbol,
            action=SignalAction.HOLD,
            price=curr_price,
            strategy_name=self.name,
            metadata={"regime": regime_result.regime.value},
        )
