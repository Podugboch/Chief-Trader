"""Base Strategy Framework.

Defines standard signal representations and abstract strategy interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
import pandas as pd


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"  # Short (or spot sell)
    HOLD = "HOLD"
    CLOSE = "CLOSE"  # Exit position


@dataclass
class TradeSignal:
    timestamp: Any
    symbol: str
    action: SignalAction
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 1.0  # 0.0 to 1.0
    strategy_name: str = "BaseStrategy"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_entry(self) -> bool:
        return self.action in (SignalAction.BUY, SignalAction.SELL)

    def is_exit(self) -> bool:
        return self.action == SignalAction.CLOSE


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradeSignal:
        """Generate trade signal for the latest bar in the DataFrame.

        Must prevent lookahead bias by only evaluating data up to the current bar.
        """
        pass

    @abstractmethod
    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute technical indicators needed by the strategy on the full DataFrame."""
        pass
