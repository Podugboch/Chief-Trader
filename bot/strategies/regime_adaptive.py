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

        # 3. Mean Reversion Mode -> Range Strategy
        if regime_result.strategy_mode == StrategyMode.MEAN_REVERSION:
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
