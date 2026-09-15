"""Strict Bar-by-Bar Backtesting Engine.

Simulates strategy execution without lookahead bias, incorporating
intra-bar stop/target checking, dynamic position sizing, slippage,
and exchange transaction fees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from bot.config import RiskConfig
from bot.strategies.base import BaseStrategy, SignalAction, TradeSignal
from bot.risk.position_sizer import PositionSizer
from bot.risk.stop_manager import StopManager, StopLevels
from bot.risk.circuit_breaker import CircuitBreaker
from bot.backtesting.metrics import calculate_metrics, BacktestMetrics


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    strategy_name: str
    metrics: BacktestMetrics
    equity_curve: pd.Series
    trades: List[Dict[str, Any]]
    total_bars_evaluated: int


class BacktestEngine:
    """Event-driven bar-by-bar backtest simulator."""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_cash: float = 10_000.0,
        risk_config: Optional[RiskConfig] = None,
    ):
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.risk_config = risk_config or RiskConfig()
        self.sizer = PositionSizer(
            max_risk_pct=self.risk_config.max_risk_per_trade_pct,
            max_position_equity_pct=0.35,
        )
        self.stop_manager = StopManager(
            atr_multiplier=self.risk_config.atr_stop_multiplier,
            break_even_atr_trigger=1.2,
        )
        self.fee_rate = self.risk_config.fee_rate
        self.slippage_rate = self.risk_config.slippage_bps / 10_000.0

    def run(self, df: pd.DataFrame, symbol: str = "BTCUSDT", timeframe: str = "1h") -> BacktestResult:
        """Run backtest over historical OHLCV data."""
        if len(df) < 50:
            raise ValueError(f"Need at least 50 bars for backtest, got {len(df)}")

        # Precompute indicators on whole dataframe for execution speed
        df_ind = self.strategy.prepare_indicators(df)

        circuit_breaker = CircuitBreaker(
            max_daily_drawdown_pct=self.risk_config.max_daily_drawdown_pct,
            max_total_drawdown_pct=self.risk_config.max_total_drawdown_pct,
            volatility_atr_multiplier=self.risk_config.volatility_circuit_breaker_atr_mult,
            cooldown_bars=self.risk_config.circuit_breaker_cooldown_bars,
        )

        cash = self.initial_cash
        equity_series: List[float] = []
        equity_timestamps: List[Any] = []
        trades: List[Dict[str, Any]] = []

        total_fees = 0.0
        total_slippage = 0.0

        # Position tracking
        in_position = False
        units = 0.0
        entry_price = 0.0
        entry_time = None
        stop_state: Optional[StopLevels] = None

        min_warmup = 50
        n_bars = len(df_ind)

        for i in range(min_warmup, n_bars):
            # Window up to current bar (avoids lookahead)
            current_bar = df_ind.iloc[i]
            prev_bar = df_ind.iloc[i - 1]
            curr_time = df_ind.index[i]

            o_px = float(current_bar["open"])
            h_px = float(current_bar["high"])
            l_px = float(current_bar["low"])
            c_px = float(current_bar["close"])
            bar_atr = float(current_bar.get("atr", c_px * 0.02))

            # Mark to market equity at start of bar
            curr_equity = cash + (units * c_px if in_position else 0.0)
            cb_status = circuit_breaker.update_equity(curr_equity, curr_time)
            circuit_breaker.step_bar()

            # 1. Intra-bar stop & target management for open positions
            if in_position and stop_state is not None:
                # Update trailing stop levels
                stop_state, exit_trigger = self.stop_manager.update_stop_level(
                    stop_state, c_px, bar_atr
                )

                # Check if stop-loss was triggered by intra-bar low
                if l_px <= stop_state.current_stop:
                    exit_px = min(o_px, stop_state.current_stop) * (1.0 - self.slippage_rate)
                    gross_proceeds = units * exit_px
                    fee = gross_proceeds * self.fee_rate
                    net_proceeds = gross_proceeds - fee
                    slippage = (stop_state.current_stop - exit_px) * units

                    cash += net_proceeds
                    total_fees += fee
                    total_slippage += slippage

                    pnl_usd = net_proceeds - (units * entry_price) - (units * entry_price * self.fee_rate)
                    pnl_pct = ((exit_px - entry_price) / entry_price) * 100.0

                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": curr_time,
                        "symbol": symbol,
                        "units": units,
                        "entry_price": entry_price,
                        "exit_price": exit_px,
                        "fee": fee,
                        "pnl_usd": pnl_usd,
                        "pnl_pct": pnl_pct,
                        "reason": "STOP_LOSS",
                    })

                    in_position = False
                    units = 0.0
                    stop_state = None

                # Check if take-profit was triggered by intra-bar high
                elif stop_state.take_profit > 0 and h_px >= stop_state.take_profit:
                    exit_px = max(o_px, stop_state.take_profit) * (1.0 - self.slippage_rate)
                    gross_proceeds = units * exit_px
                    fee = gross_proceeds * self.fee_rate
                    net_proceeds = gross_proceeds - fee
                    slippage = (exit_px - stop_state.take_profit) * units

                    cash += net_proceeds
                    total_fees += fee
                    total_slippage += slippage

                    pnl_usd = net_proceeds - (units * entry_price) - (units * entry_price * self.fee_rate)
                    pnl_pct = ((exit_px - entry_price) / entry_price) * 100.0

                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": curr_time,
                        "symbol": symbol,
                        "units": units,
                        "entry_price": entry_price,
                        "exit_price": exit_px,
                        "fee": fee,
                        "pnl_usd": pnl_usd,
                        "pnl_pct": pnl_pct,
                        "reason": "TAKE_PROFIT",
                    })

                    in_position = False
                    units = 0.0
                    stop_state = None

            # 2. Strategy Signal Generation (evaluated at close of bar)
            # Pass slice up to current bar
            window_df = df_ind.iloc[max(0, i - 100) : i + 1]
            signal: TradeSignal = self.strategy.generate_signal(window_df, symbol)

            # 3. Strategy Exit Signal (CLOSE)
            if in_position and signal.action == SignalAction.CLOSE:
                exit_px = c_px * (1.0 - self.slippage_rate)
                gross_proceeds = units * exit_px
                fee = gross_proceeds * self.fee_rate
                net_proceeds = gross_proceeds - fee
                slippage = (c_px - exit_px) * units

                cash += net_proceeds
                total_fees += fee
                total_slippage += slippage

                pnl_usd = net_proceeds - (units * entry_price) - (units * entry_price * self.fee_rate)
                pnl_pct = ((exit_px - entry_price) / entry_price) * 100.0

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": curr_time,
                    "symbol": symbol,
                    "units": units,
                    "entry_price": entry_price,
                    "exit_price": exit_px,
                    "fee": fee,
                    "pnl_usd": pnl_usd,
                    "pnl_pct": pnl_pct,
                    "reason": signal.metadata.get("reason", "SIGNAL_CLOSE"),
                })

                in_position = False
                units = 0.0
                stop_state = None

            # 4. Strategy Entry Signal (BUY)
            elif not in_position and signal.action == SignalAction.BUY:
                # Only enter if circuit breaker is not tripped
                if not cb_status.is_tripped:
                    fill_px = c_px * (1.0 + self.slippage_rate)
                    target_stop = signal.stop_loss if signal.stop_loss and signal.stop_loss > 0 else (fill_px - 2.0 * bar_atr)
                    target_tp = signal.take_profit if signal.take_profit and signal.take_profit > 0 else (fill_px + 3.5 * bar_atr)

                    current_equity = cash
                    sizing = self.sizer.calculate_size(
                        account_equity=current_equity,
                        available_cash=cash,
                        entry_price=fill_px,
                        stop_loss_price=target_stop,
                        current_open_positions_count=0,
                        max_open_positions=1,
                    )

                    if sizing.allowed and sizing.units > 0:
                        order_cost = sizing.units * fill_px
                        fee = order_cost * self.fee_rate
                        total_deduction = order_cost + fee

                        if total_deduction <= cash:
                            cash -= total_deduction
                            units = sizing.units
                            entry_price = fill_px
                            entry_time = curr_time
                            in_position = True

                            total_fees += fee
                            total_slippage += (fill_px - c_px) * units

                            stop_state = StopLevels(
                                entry_price=entry_price,
                                current_stop=target_stop,
                                take_profit=target_tp,
                                highest_price_seen=entry_price,
                                atr_multiplier=self.risk_config.atr_stop_multiplier,
                            )

            # Record final bar equity
            bar_equity = cash + (units * c_px if in_position else 0.0)
            equity_series.append(bar_equity)
            equity_timestamps.append(curr_time)

        # Build equity curve series
        eq_curve = pd.Series(equity_series, index=pd.DatetimeIndex(equity_timestamps))

        benchmark_initial = float(df["close"].iloc[min_warmup])
        benchmark_final = float(df["close"].iloc[-1])

        metrics = calculate_metrics(
            equity_curve=eq_curve,
            trades=trades,
            initial_equity=self.initial_cash,
            benchmark_initial_price=benchmark_initial,
            benchmark_final_price=benchmark_final,
            total_fees=total_fees,
            total_slippage=total_slippage,
        )

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            strategy_name=self.strategy.name,
            metrics=metrics,
            equity_curve=eq_curve,
            trades=trades,
            total_bars_evaluated=len(equity_series),
        )
