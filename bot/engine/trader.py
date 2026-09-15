"""Trading Bot Orchestration Engine.

Manages data ingestion, indicator pipelines, strategy signals,
risk guardrails, order execution, state persistence, and terminal UI.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import pandas as pd

from bot.config import BotConfig, TradingMode
from bot.data.fetcher import DataFetcher
from bot.strategies.base import BaseStrategy, SignalAction, TradeSignal
from bot.strategies.regime_adaptive import RegimeAdaptiveStrategy
from bot.strategies.trend_momentum import TrendMomentumStrategy
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.analysis.market_analyzer import MarketAnalyzer
from bot.analysis.regime import MarketRegimeClassifier
from bot.risk.position_sizer import PositionSizer
from bot.risk.stop_manager import StopManager, StopLevels
from bot.risk.circuit_breaker import CircuitBreaker
from bot.execution.order import OrderSide, OrderType, Position
from bot.execution.paper_exchange import PaperExchange
from bot.execution.binance_client import BinanceClient
from bot.engine.state import BotState
from bot.engine.kill_switch import KillSwitch
from bot.ui.dashboard import BotDashboard

logger = logging.getLogger("bot.engine.trader")


class TradingBot:
    """Production crypto trading bot engine."""

    def __init__(
        self,
        config: Optional[BotConfig] = None,
        strategy: Optional[BaseStrategy] = None,
        enable_ui: bool = True,
    ):
        self.config = config or BotConfig.load_from_env()
        self.strategy = strategy or RegimeAdaptiveStrategy()
        self.enable_ui = enable_ui

        # Data and Analysis
        self.data_fetcher = DataFetcher(base_url=self.config.exchange.public_data_base_url)
        self.market_analyzer = MarketAnalyzer()
        self.regime_classifier = MarketRegimeClassifier()

        # Risk Controls
        self.sizer = PositionSizer(
            max_risk_pct=self.config.risk.max_risk_per_trade_pct,
            max_position_equity_pct=0.35,
        )
        self.stop_manager = StopManager(
            atr_multiplier=self.config.risk.atr_stop_multiplier,
            break_even_atr_trigger=1.2,
        )
        self.circuit_breaker = CircuitBreaker(
            max_daily_drawdown_pct=self.config.risk.max_daily_drawdown_pct,
            max_total_drawdown_pct=self.config.risk.max_total_drawdown_pct,
            volatility_atr_multiplier=self.config.risk.volatility_circuit_breaker_atr_mult,
            cooldown_bars=self.config.risk.circuit_breaker_cooldown_bars,
        )

        # Execution
        self.paper_exchange = PaperExchange(
            initial_cash_usd=self.config.initial_cash,
            fee_rate=self.config.risk.fee_rate,
            slippage_bps=self.config.risk.slippage_bps,
        )
        self.binance_client: Optional[BinanceClient] = None
        self.exchange_client = None
        if self.config.trading_mode in (TradingMode.TESTNET, TradingMode.LIVE):
            from bot.execution.multi_exchange_client import MultiExchangeClient
            self.exchange_client = MultiExchangeClient(
                exchange_id=self.config.exchange.exchange_name,
                api_key=self.config.exchange.api_key,
                api_secret=self.config.exchange.api_secret,
                passphrase=self.config.exchange.api_passphrase,
                use_testnet=self.config.exchange.use_testnet,
            )
            if self.config.exchange.exchange_name == "binance":
                self.binance_client = BinanceClient(
                    api_key=self.config.exchange.api_key,
                    api_secret=self.config.exchange.api_secret,
                    use_testnet=self.config.exchange.use_testnet,
                )

        # Persistence & Safety
        self.state = BotState(state_dir=self.config.state_dir)
        self.kill_switch = KillSwitch(flag_path=self.config.kill_switch_flag)
        self.kill_switch.register_callback(self._on_emergency_shutdown)
        self.dashboard = BotDashboard()

        # Restore saved state if exists
        self._restore_state()

    def _restore_state(self) -> None:
        """Restore persisted cash, positions, and trades if available."""
        saved = self.state.load()
        if saved:
            self.paper_exchange.cash = float(saved.get("cash", self.config.initial_cash))
            self.paper_exchange.balances = saved.get("balances", {})
            self.paper_exchange.positions = saved.get("reconstructed_positions", {})
            self.paper_exchange.trade_history = saved.get("trade_history", [])
            self.circuit_breaker.high_water_mark = float(saved.get("high_water_mark", self.paper_exchange.cash))
            logger.info("Restored %d open positions from state file.", len(self.paper_exchange.positions))

    def _on_emergency_shutdown(self, reason: str) -> None:
        """Callback invoked when kill switch is tripped."""
        logger.critical("Shutdown initiated via kill switch: %s", reason)
        self.save_state()

    def save_state(self) -> None:
        """Persist current bot runtime state to disk."""
        self.state.save(
            cash=self.paper_exchange.cash,
            balances=self.paper_exchange.balances,
            positions=self.paper_exchange.positions,
            high_water_mark=self.circuit_breaker.high_water_mark,
            trade_history=self.paper_exchange.trade_history,
        )

    def emergency_liquidate(self, symbol: Optional[str] = None) -> None:
        """Immediately market-sell open position(s)."""
        logger.warning("Emergency liquidation triggered!")
        symbols_to_close = [symbol] if symbol else list(self.paper_exchange.positions.keys())

        for sym in symbols_to_close:
            if sym in self.paper_exchange.positions:
                pos = self.paper_exchange.positions[sym]
                current_price = self.data_fetcher.fetch_ticker_price(sym)
                self.paper_exchange.execute_market_order(
                    symbol=sym,
                    side=OrderSide.SELL,
                    quantity=pos.units,
                    current_market_price=current_price,
                    strategy_name="EMERGENCY_LIQUIDATE",
                )
                logger.warning("Liquidated position in %s", sym)

        self.save_state()

    def step(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
        """Execute evaluation cycle across specified symbol or all configured trading pairs."""
        # 1. Safety Check: Kill Switch
        if self.kill_switch.is_triggered():
            logger.warning("Kill switch active: %s. Skipping evaluation.", self.kill_switch.reason)
            return

        symbols = [symbol] if symbol else list(self.config.trading_pairs)
        tf = timeframe or self.config.default_timeframe

        for sym in symbols:
            try:
                self._evaluate_symbol(sym, tf)
            except Exception as e:
                logger.error("Error evaluating pair %s: %s", sym, e)

        # Save State after evaluation cycle
        self.save_state()

        # Render Terminal Dashboard if enabled
        if self.enable_ui:
            primary_sym = symbols[0] if symbols else self.config.default_symbol
            try:
                mark = self.data_fetcher.fetch_ticker_price(primary_sym)
            except Exception:
                mark = 0.0
            equity = self.paper_exchange.get_portfolio_equity({primary_sym: mark})
            realized_pnl = sum(t.get("pnl_usd", 0.0) for t in self.paper_exchange.trade_history)
            unrealized_pnl = sum(p.unrealized_pnl for p in self.paper_exchange.positions.values())
            self.dashboard.render_summary(
                mode=self.config.trading_mode.value,
                symbol=primary_sym,
                timeframe=tf,
                current_price=mark,
                equity=equity,
                cash=self.paper_exchange.cash,
                realized_pnl=realized_pnl,
            )

    def _evaluate_symbol(self, sym: str, tf: str) -> None:
        """Evaluates market data, positions, and strategy signals for a single symbol."""
        # 1. Ingest Latest Market Data
        try:
            current_price = self.data_fetcher.fetch_ticker_price(sym)
            df = self.data_fetcher.fetch_klines(sym, timeframe=tf, limit=100)
            self.circuit_breaker.record_api_success()
        except Exception as e:
            logger.error("Failed to fetch market data for %s: %s", sym, e)
            self.circuit_breaker.record_api_error(str(e))
            return

        if len(df) < 50:
            logger.warning("Insufficient candles (%d < 50) for %s analysis.", len(df), sym)
            return

        # 2. Market Analysis and Regime Classification
        analysis_report = self.market_analyzer.analyze(df, symbol=sym, timeframe=tf)
        curr_atr = analysis_report.indicators.get("atr", current_price * 0.02)

        # 3. Update Portfolio Valuation & Circuit Breakers
        prices_map = {sym: current_price}
        for pos_sym, pos in self.paper_exchange.positions.items():
            if pos_sym not in prices_map:
                prices_map[pos_sym] = pos.current_price
        equity = self.paper_exchange.get_portfolio_equity(prices_map)
        cb_status = self.circuit_breaker.update_equity(equity)

        # 4. Manage Open Positions (Trailing Stops / Take Profits)
        if sym in self.paper_exchange.positions:
            pos = self.paper_exchange.positions[sym]
            pos.current_price = current_price

            stop_state = StopLevels(
                entry_price=pos.entry_price,
                current_stop=pos.stop_loss,
                take_profit=pos.take_profit,
                highest_price_seen=pos.highest_price_seen,
                atr_multiplier=self.config.risk.atr_stop_multiplier,
                break_even_activated=pos.break_even_activated,
            )

            updated_stops, exit_trigger = self.stop_manager.update_stop_level(
                stop_state, current_price, curr_atr
            )

            # Update position state
            pos.stop_loss = updated_stops.current_stop
            pos.highest_price_seen = updated_stops.highest_price_seen
            pos.break_even_activated = updated_stops.break_even_activated

            # Check if stop or profit target was hit
            if exit_trigger is not None:
                logger.info("Exiting %s: %s triggered at $%.2f", sym, exit_trigger, current_price)
                self.paper_exchange.execute_market_order(
                    symbol=sym,
                    side=OrderSide.SELL,
                    quantity=pos.units,
                    current_market_price=current_price,
                    strategy_name=f"{pos.strategy_name}:{exit_trigger}",
                )
                return

        # 5. Generate Strategy Signal
        signal: TradeSignal = self.strategy.generate_signal(df, sym)

        # 6. Execute Strategy Exit (CLOSE)
        if sym in self.paper_exchange.positions and signal.action == SignalAction.CLOSE:
            pos = self.paper_exchange.positions[sym]
            reason = signal.metadata.get("reason", "STRATEGY_CLOSE")
            logger.info("Strategy requested position close for %s: %s", sym, reason)
            self.paper_exchange.execute_market_order(
                symbol=sym,
                side=OrderSide.SELL,
                quantity=pos.units,
                current_market_price=current_price,
                strategy_name=signal.strategy_name,
            )
            return

        # 7. Execute Strategy Entry (BUY)
        if sym not in self.paper_exchange.positions and signal.action == SignalAction.BUY:
            if cb_status.is_tripped:
                logger.warning("Entry signal for %s ignored: Circuit breaker is tripped (%s)", sym, cb_status.reason)
            else:
                target_stop = signal.stop_loss or (current_price - (self.config.risk.atr_stop_multiplier * curr_atr))
                target_tp = signal.take_profit or (current_price + (self.config.risk.atr_take_profit_multiplier * curr_atr))

                sizing = self.sizer.calculate_size(
                    account_equity=equity,
                    available_cash=self.paper_exchange.cash,
                    entry_price=current_price,
                    stop_loss_price=target_stop,
                    current_open_positions_count=len(self.paper_exchange.positions),
                    max_open_positions=self.config.risk.max_open_positions,
                )

                if sizing.allowed and sizing.units > 0:
                    logger.info(
                        "Signal triggered BUY for %s: %.6f units ($%.2f notional, risked $%.2f)",
                        sym, sizing.units, sizing.notional_usd, sizing.risk_usd
                    )
                    self.paper_exchange.execute_market_order(
                        symbol=sym,
                        side=OrderSide.BUY,
                        quantity=sizing.units,
                        current_market_price=current_price,
                        stop_loss=target_stop,
                        take_profit=target_tp,
                        strategy_name=signal.strategy_name,
                    )
                else:
                    logger.info("Position sizing for %s: %s", sym, sizing.reason)

    def run_loop(self, poll_interval_seconds: int = 10, max_iterations: Optional[int] = None) -> None:
        """Run continuous live / paper monitoring loop."""
        logger.info("Starting bot event loop (Interval: %ds)...", poll_interval_seconds)
        iteration = 0

        try:
            while not self.kill_switch.is_triggered():
                self.step()
                iteration += 1
                if max_iterations is not None and iteration >= max_iterations:
                    logger.info("Reached maximum iterations (%d). Stopping.", max_iterations)
                    break
                time.sleep(poll_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down gracefully.")
            self.kill_switch.trigger("KeyboardInterrupt")
        finally:
            self.save_state()
            logger.info("Bot execution finished safely. Final state saved.")
