"""Web Server and API for Crypto Trading Bot Dashboard.

Provides a modern localhost web interface and REST API for real-time monitoring,
interactive backtesting, market analysis, multi-exchange API key configuration,
and emergency kill switches.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import threading
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, BackgroundTasks, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from bot.config import BotConfig, TradingMode
from bot.data.fetcher import DataFetcher
from bot.analysis.market_analyzer import MarketAnalyzer
from bot.strategies.regime_adaptive import RegimeAdaptiveStrategy
from bot.strategies.trend_momentum import TrendMomentumStrategy
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.backtesting.engine import BacktestEngine
from bot.execution.binance_client import BinanceClient
from bot.execution.multi_exchange_client import MultiExchangeClient, POPULAR_EXCHANGES
from bot.execution.order import OrderSide, OrderStatus
from bot.engine.trader import TradingBot
from bot.engine.kill_switch import KillSwitch
from bot.engine.state import BotState

logger = logging.getLogger("bot.web.server")

app = FastAPI(title="Crypto Trading Bot Dashboard", version="1.2.0")


def _write_env_updates(updates: Dict[str, str]) -> None:
    """Updates key-value pairs in .env while preserving comments and structure."""
    if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        return
    try:
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        lines = []
        updated_keys = set()

        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#") and "=" in stripped:
                        key, _ = stripped.split("=", 1)
                        key = key.strip()
                        if key in updates:
                            lines.append(f"{key}={updates[key]}\n")
                            updated_keys.add(key)
                            continue
                    lines.append(line)

        for k, v in updates.items():
            if k not in updated_keys:
                lines.append(f"{k}={v}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except OSError:
        pass


# Shared Bot State Holder
class BotManager:
    def __init__(self):
        self.config = BotConfig.load_from_env()
        self.strategy = RegimeAdaptiveStrategy()
        self.bot = TradingBot(config=self.config, strategy=self.strategy, enable_ui=False)
        self.is_running = False
        self.interval_seconds = 10
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.last_step_time: Optional[str] = None
        self.last_step_error: Optional[str] = None

    def start_loop(self):
        with self.lock:
            if self.is_running:
                return
            self.is_running = True
            self.thread = threading.Thread(target=self._run_loop_worker, daemon=True)
            self.thread.start()
            logger.info("Background bot worker thread started.")

    def stop_loop(self):
        with self.lock:
            self.is_running = False
            logger.info("Background bot worker stopped.")

    def _run_loop_worker(self):
        while self.is_running:
            if not self.bot.kill_switch.is_triggered():
                try:
                    self.bot.step()
                    self.last_step_time = datetime.now(timezone.utc).isoformat()
                    self.last_step_error = None
                except Exception as e:
                    self.last_step_error = str(e)
                    logger.error("Error in bot worker step: %s", e)
            time.sleep(self.interval_seconds)

    def manual_step(self):
        try:
            self.bot.step()
            self.last_step_time = datetime.now(timezone.utc).isoformat()
            self.last_step_error = None
            return True, "Step executed successfully"
        except Exception as e:
            self.last_step_error = str(e)
            return False, str(e)


manager = BotManager()


class BacktestRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    strategy: str = "adaptive"
    days: int = 60
    cash: float = 10_000.0


class SettingsPayload(BaseModel):
    exchange_name: str = "binance"
    trading_mode: str = "paper"
    api_key: Optional[str] = ""
    api_secret: Optional[str] = ""
    api_passphrase: Optional[str] = ""
    use_testnet: bool = True
    default_symbol: str = "BTCUSDT"
    trading_pairs: Optional[str] = None
    default_timeframe: str = "1h"
    max_risk_per_trade_pct: float = 1.5
    max_daily_drawdown_pct: float = 5.0
    max_total_drawdown_pct: float = 15.0


class TestConnectionPayload(BaseModel):
    exchange_name: str = "binance"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_passphrase: Optional[str] = None
    use_testnet: bool = True


@app.get("/api/status")
def get_status():
    """Returns real-time status of the trading bot."""
    bot = manager.bot
    exchange = bot.paper_exchange
    current_symbol = bot.config.default_symbol

    # Fetch live mark price
    mark_price = 0.0
    try:
        mark_price = bot.data_fetcher.fetch_ticker_price(current_symbol)
    except Exception:
        pass

    equity = exchange.get_portfolio_equity({current_symbol: mark_price}) if mark_price > 0 else exchange.cash
    cb_status = bot.circuit_breaker.update_equity(equity)

    positions_data = []
    for sym, p in exchange.positions.items():
        positions_data.append({
            "symbol": p.symbol,
            "units": p.units,
            "entry_price": p.entry_price,
            "current_price": p.current_price,
            "stop_loss": p.stop_loss,
            "take_profit": p.take_profit,
            "unrealized_pnl": round(p.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 2),
            "strategy": p.strategy_name,
            "break_even_activated": p.break_even_activated,
        })

    realized_pnl = sum(t.get("pnl_usd", 0.0) for t in exchange.trade_history)
    unrealized_pnl = sum(p.unrealized_pnl for p in exchange.positions.values())

    return {
        "is_running": manager.is_running,
        "mode": bot.config.trading_mode.value,
        "exchange_name": bot.config.exchange.exchange_name,
        "default_symbol": bot.config.default_symbol,
        "trading_pairs": bot.config.trading_pairs,
        "default_timeframe": bot.config.default_timeframe,
        "mark_price": mark_price,
        "equity": round(equity, 2),
        "cash": round(exchange.cash, 2),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "total_fees": round(exchange.total_fees_paid, 2),
        "total_slippage": round(exchange.total_slippage_lost, 2),
        "kill_switch_triggered": bot.kill_switch.is_triggered(),
        "kill_switch_reason": bot.kill_switch.reason,
        "circuit_breaker": {
            "is_tripped": cb_status.is_tripped,
            "reason": cb_status.reason,
            "daily_drawdown_pct": round(cb_status.daily_drawdown_pct * 100, 2),
            "total_drawdown_pct": round(cb_status.total_drawdown_pct * 100, 2),
            "high_water_mark": round(cb_status.high_water_mark, 2),
        },
        "open_positions": positions_data,
        "recent_trades": (exchange.order_history[-20:] if getattr(exchange, "order_history", None) else exchange.trade_history[-20:]),
        "last_step_time": manager.last_step_time,
        "last_step_error": manager.last_step_error,
        "strategy_name": manager.strategy.name,
    }


@app.get("/api/analysis")
def get_analysis(symbol: str = "BTCUSDT", timeframe: str = "1h"):
    """Fetch live multi-factor market analysis."""
    fetcher = manager.bot.data_fetcher
    analyzer = manager.bot.market_analyzer

    try:
        df = fetcher.fetch_klines(symbol.upper(), timeframe=timeframe, limit=200)
        report = analyzer.analyze(df, symbol=symbol.upper(), timeframe=timeframe)

        return {
            "symbol": report.symbol,
            "timeframe": report.timeframe,
            "latest_price": report.latest_price,
            "composite_score": report.composite_score,
            "verdict": report.verdict,
            "regime": {
                "name": report.regime.regime.value,
                "strategy_mode": report.regime.strategy_mode.value,
                "adx": round(report.regime.adx_value, 2),
                "atr": round(report.regime.atr_value, 2),
                "atr_percent": round(report.regime.atr_percent, 2),
                "volatility_zscore": round(report.regime.volatility_zscore, 2),
                "trend_bias": report.regime.trend_bias,
                "summary": report.regime.summary,
            },
            "indicators": report.indicators,
            "levels": {
                "resistance_2": round(report.levels.resistance_2, 2),
                "resistance_1": round(report.levels.resistance_1, 2),
                "pivot": round(report.levels.pivot, 2),
                "support_1": round(report.levels.support_1, 2),
                "support_2": round(report.levels.support_2, 2),
                "recent_swing_high": round(report.levels.recent_swing_high, 2),
                "recent_swing_low": round(report.levels.recent_swing_low, 2),
            },
            "key_observations": report.key_observations,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/scanner")
def get_scanner(timeframe: str = "1h"):
    """Scans all configured trading pairs and returns market analysis summaries."""
    fetcher = manager.bot.data_fetcher
    analyzer = manager.bot.market_analyzer
    pairs = manager.bot.config.trading_pairs
    positions = manager.bot.paper_exchange.positions

    results = []
    for sym in pairs:
        try:
            current_price = fetcher.fetch_ticker_price(sym)
            df = fetcher.fetch_klines(sym, timeframe=timeframe, limit=80)
            report = analyzer.analyze(df, symbol=sym, timeframe=timeframe)
            results.append({
                "symbol": sym,
                "price": report.latest_price,
                "verdict": report.verdict,
                "composite_score": report.composite_score,
                "regime": report.regime.regime.value,
                "strategy_mode": report.regime.strategy_mode.value,
                "trend_bias": report.regime.trend_bias,
                "adx": round(report.regime.adx_value, 1),
                "rsi": round(report.indicators.get("rsi", 50.0), 1),
                "supertrend": report.indicators.get("supertrend_direction", "NEUTRAL"),
                "has_position": sym in positions,
            })
        except Exception as e:
            logger.warning("Scanner error for %s: %s", sym, e)
            results.append({
                "symbol": sym,
                "price": 0.0,
                "verdict": "NEUTRAL",
                "composite_score": 0.0,
                "regime": "STAND_ASIDE",
                "strategy_mode": "STAND_ASIDE",
                "trend_bias": "NEUTRAL",
                "adx": 0.0,
                "rsi": 50.0,
                "supertrend": "NEUTRAL",
                "has_position": sym in positions,
            })
    return results


@app.get("/api/settings")
def get_settings():
    """Retrieve current bot configuration and API status."""
    cfg = manager.bot.config
    api_key = cfg.exchange.api_key
    has_secret = bool(cfg.exchange.api_secret)
    masked_key = (api_key[:6] + "..." + api_key[-4:]) if len(api_key) > 10 else api_key

    return {
        "exchange_name": cfg.exchange.exchange_name,
        "trading_mode": cfg.trading_mode.value,
        "api_key": api_key,
        "masked_key": masked_key,
        "has_secret": has_secret,
        "api_passphrase": cfg.exchange.api_passphrase,
        "use_testnet": cfg.exchange.use_testnet,
        "default_symbol": cfg.default_symbol,
        "trading_pairs": ",".join(cfg.trading_pairs),
        "trading_pairs_list": cfg.trading_pairs,
        "default_timeframe": cfg.default_timeframe,
        "max_risk_per_trade_pct": round(cfg.risk.max_risk_per_trade_pct * 100.0, 2),
        "max_daily_drawdown_pct": round(cfg.risk.max_daily_drawdown_pct * 100.0, 2),
        "max_total_drawdown_pct": round(cfg.risk.max_total_drawdown_pct * 100.0, 2),
        "supported_exchanges": POPULAR_EXCHANGES,
    }


@app.post("/api/settings")
def update_settings(payload: SettingsPayload):
    """Save API keys and bot parameters to .env and reload running configuration."""
    ex_name = payload.exchange_name.lower().strip()
    updates = {
        "EXCHANGE_NAME": ex_name,
        "TRADING_MODE": payload.trading_mode.lower(),
        "USE_TESTNET": "true" if payload.use_testnet else "false",
        "BINANCE_USE_TESTNET": "true" if payload.use_testnet else "false",
        "DEFAULT_SYMBOL": payload.default_symbol.upper(),
        "DEFAULT_TIMEFRAME": payload.default_timeframe,
        "MAX_RISK_PER_TRADE_PCT": str(payload.max_risk_per_trade_pct / 100.0),
        "MAX_DAILY_DRAWDOWN_PCT": str(payload.max_daily_drawdown_pct / 100.0),
        "MAX_TOTAL_DRAWDOWN_PCT": str(payload.max_total_drawdown_pct / 100.0),
    }

    if payload.trading_pairs:
        cleaned_pairs = ",".join([p.strip().upper() for p in payload.trading_pairs.split(",") if p.strip()])
        if cleaned_pairs:
            updates["TRADING_PAIRS"] = cleaned_pairs

    if payload.api_key is not None:
        updates["API_KEY"] = payload.api_key.strip()
        updates["BINANCE_API_KEY"] = payload.api_key.strip()
    if payload.api_secret is not None:
        updates["API_SECRET"] = payload.api_secret.strip()
        updates["BINANCE_API_SECRET"] = payload.api_secret.strip()
    if payload.api_passphrase is not None:
        updates["API_PASSPHRASE"] = payload.api_passphrase.strip()

    _write_env_updates(updates)

    # Reload configuration
    new_cfg = BotConfig.load_from_env()
    manager.config = new_cfg
    manager.bot.config = new_cfg

    if new_cfg.trading_mode in (TradingMode.TESTNET, TradingMode.LIVE):
        manager.bot.exchange_client = MultiExchangeClient(
            exchange_id=new_cfg.exchange.exchange_name,
            api_key=new_cfg.exchange.api_key,
            api_secret=new_cfg.exchange.api_secret,
            passphrase=new_cfg.exchange.api_passphrase,
            use_testnet=new_cfg.exchange.use_testnet,
        )
        if new_cfg.exchange.exchange_name == "binance":
            manager.bot.binance_client = BinanceClient(
                api_key=new_cfg.exchange.api_key,
                api_secret=new_cfg.exchange.api_secret,
                use_testnet=new_cfg.exchange.use_testnet,
            )
    else:
        manager.bot.exchange_client = None
        manager.bot.binance_client = None

    ex_label = POPULAR_EXCHANGES.get(ex_name, ex_name.upper())
    logger.info("Bot configuration updated: Exchange=%s, Mode=%s", ex_label, new_cfg.trading_mode.value)
    return {
        "status": "success",
        "message": f"Settings and credentials saved! Exchange: {ex_label} | Mode: {new_cfg.trading_mode.value.upper()}",
        "mode": new_cfg.trading_mode.value,
        "exchange": ex_name,
    }


@app.post("/api/settings/test-connection")
def test_connection(payload: TestConnectionPayload):
    """Test exchange connectivity and authentication using CCXT across any selected exchange."""
    cfg = manager.bot.config
    ex_id = (payload.exchange_name or cfg.exchange.exchange_name or "binance").lower()
    key = payload.api_key if (payload.api_key is not None and payload.api_key != "") else cfg.exchange.api_key
    sec = payload.api_secret if (payload.api_secret is not None and payload.api_secret != "") else cfg.exchange.api_secret
    passphrase = payload.api_passphrase if (payload.api_passphrase is not None and payload.api_passphrase != "") else cfg.exchange.api_passphrase
    use_testnet = payload.use_testnet

    ex_label = POPULAR_EXCHANGES.get(ex_id, ex_id.upper())

    if not key or not sec:
        return {
            "success": False,
            "message": f"Both API Key and API Secret are required to test {ex_label} authentication.",
        }

    try:
        client = MultiExchangeClient(
            exchange_id=ex_id,
            api_key=key,
            api_secret=sec,
            passphrase=passphrase,
            use_testnet=use_testnet,
            timeout=10,
        )

        # Test ping / market load
        if not client.ping():
            return {"success": False, "message": f"Could not reach {ex_label} servers. Check network or DNS."}

        # Query authenticated balances
        balances = client.get_account_balances()
        dest = f"{ex_label} (Sandbox/Testnet)" if use_testnet else f"{ex_label} (Live API)"
        return {
            "success": True,
            "message": f"Successfully authenticated with {dest}!",
            "exchange": ex_label,
            "balances": balances,
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"{ex_label} authentication test failed: {str(e)}",
        }


@app.post("/api/backtest")
def run_backtest_endpoint(req: BacktestRequest):
    """Run backtest on demand and return analytics and trades."""
    symbol = req.symbol.upper()
    fetcher = manager.bot.data_fetcher

    if req.strategy.lower() in ("trend", "momentum"):
        strat = TrendMomentumStrategy()
    elif req.strategy.lower() in ("mean_reversion", "range"):
        strat = MeanReversionStrategy()
    else:
        strat = RegimeAdaptiveStrategy()

    try:
        df = fetcher.fetch_historical_range(symbol, timeframe=req.timeframe, days=req.days)
        engine = BacktestEngine(strategy=strat, initial_cash=req.cash)
        res = engine.run(df, symbol=symbol, timeframe=req.timeframe)

        # Sample equity curve for lightweight JSON transfer (up to 150 points)
        eq_series = res.equity_curve
        step = max(1, len(eq_series) // 150)
        sampled_curve = [
            {"time": str(t)[:16], "equity": round(val, 2)}
            for t, val in zip(eq_series.index[::step], eq_series.values[::step])
        ]

        return {
            "symbol": res.symbol,
            "timeframe": res.timeframe,
            "strategy": res.strategy_name,
            "metrics": res.metrics.to_dict(),
            "trades": res.trades[-30:],
            "total_trades": len(res.trades),
            "equity_curve": sampled_curve,
        }
    except Exception as e:
        logger.error("Backtest failed: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/bot/start")
def start_bot():
    manager.start_loop()
    return {"status": "started", "is_running": manager.is_running}


@app.post("/api/bot/stop")
def stop_bot():
    manager.stop_loop()
    return {"status": "stopped", "is_running": manager.is_running}


@app.post("/api/bot/step")
def trigger_step():
    ok, msg = manager.manual_step()
    return {"success": ok, "message": msg}


@app.post("/api/bot/kill")
def kill_bot():
    manager.bot.kill_switch.trigger("Web Dashboard Kill Switch")
    manager.stop_loop()
    return {"status": "killed", "reason": "Web Dashboard Kill Switch"}


@app.post("/api/bot/arm")
def arm_bot():
    manager.bot.kill_switch.reset()
    return {"status": "armed"}


@app.post("/api/bot/liquidate")
def emergency_liquidate():
    manager.bot.emergency_liquidate()
    return {"status": "liquidated"}


class TestTradePayload(BaseModel):
    symbol: Optional[str] = None
    side: str = "BUY"
    amount_usd: float = 500.0


@app.post("/api/bot/test-trade")
def execute_test_trade(payload: TestTradePayload = Body(default=TestTradePayload())):
    """Execute an instant demo/paper test trade for user verification."""
    bot = manager.bot
    sym = (payload.symbol or bot.config.default_symbol).upper()
    try:
        current_price = bot.data_fetcher.fetch_ticker_price(sym)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to get price for {sym}: {e}"})

    if current_price <= 0:
        return JSONResponse(status_code=400, content={"error": f"Invalid market price for {sym}: {current_price}"})

    if payload.side.upper() == "BUY":
        qty = round(payload.amount_usd / current_price, 5)
        if qty <= 0:
            qty = 0.0001
        # Risk protection: 2% stop, 3% take profit
        stop_loss = round(current_price * 0.98, 2)
        take_profit = round(current_price * 1.03, 2)

        order = bot.paper_exchange.execute_market_order(
            symbol=sym,
            side=OrderSide.BUY,
            quantity=qty,
            current_market_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name="DemoTestFill",
        )
        bot.save_state()
        return {
            "status": "filled" if order.status == OrderStatus.FILLED else "rejected",
            "symbol": sym,
            "side": "BUY",
            "quantity": qty,
            "price": order.executed_price,
            "fee": order.fee_paid,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }
    elif payload.side.upper() == "SELL":
        if sym not in bot.paper_exchange.positions:
            return JSONResponse(status_code=400, content={"error": f"No open position for {sym} to sell"})
        bot.emergency_liquidate(sym)
        return {"status": "closed", "symbol": sym}


class ClosePositionPayload(BaseModel):
    symbol: str


@app.post("/api/bot/close-position")
def close_single_position(payload: ClosePositionPayload):
    """Closes a specific open position at live market price."""
    bot = manager.bot
    sym = payload.symbol.upper()
    if sym not in bot.paper_exchange.positions:
        return JSONResponse(status_code=400, content={"error": f"No open position for {sym}"})
    bot.emergency_liquidate(sym)
    return {"status": "closed", "symbol": sym}


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the complete, interactive Web Dashboard."""
    return HTML_CONTENT


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Crypto Trading Bot Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: '#3b82f6',
            card: '#1e293b',
            darkbg: '#0f172a',
            bull: '#10b981',
            bear: '#ef4444',
          }
        }
      }
    }
  </script>
</head>
<body class="bg-darkbg text-slate-100 min-h-screen font-sans">
  <!-- Top Navigation -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold shadow-lg shadow-blue-500/30">
          <i class="fa-solid fa-chart-line"></i>
        </div>
        <div>
          <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
            Crypto Trading Bot
            <span id="modeBadge" class="text-xs px-2.5 py-0.5 rounded-full font-semibold uppercase bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">Paper</span>
            <span id="exchangeBadge" class="text-xs px-2 py-0.5 rounded font-semibold bg-slate-800 text-slate-300 border border-slate-700">BINANCE</span>
          </h1>
          <p class="text-xs text-slate-400">Adaptive Multi-Regime Strategy Engine</p>
        </div>
      </div>

      <!-- Quick Control Actions -->
      <div class="flex items-center space-x-2.5">
        <button onclick="switchTab('settings')" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center gap-1.5 transition border border-slate-700">
          <i class="fa-solid fa-key text-amber-400"></i> API Keys & Exchange
        </button>
        <button onclick="executeTestDemoBuy()" class="px-3.5 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-500 text-white text-xs font-semibold flex items-center gap-1.5 transition shadow">
          <i class="fa-solid fa-cart-shopping"></i> Test Demo Buy
        </button>
        <button id="btnStart" onclick="startBot()" class="px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center gap-1.5 transition shadow">
          <i class="fa-solid fa-play"></i> Start Bot
        </button>
        <button id="btnStop" onclick="stopBot()" class="px-3.5 py-1.5 rounded-lg bg-slate-700 hover:bg-slate-600 text-white text-xs font-semibold flex items-center gap-1.5 transition hidden">
          <i class="fa-solid fa-pause"></i> Pause
        </button>
        <button onclick="stepBot()" class="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center gap-1.5 transition">
          <i class="fa-solid fa-rotate-right"></i> Evaluate Now
        </button>
        <button id="btnKill" onclick="killBot()" class="px-3.5 py-1.5 rounded-lg bg-red-600 hover:bg-red-500 text-white text-xs font-semibold flex items-center gap-1.5 transition shadow-lg shadow-red-500/20">
          <i class="fa-solid fa-skull-crossbones"></i> Kill Switch
        </button>
        <button id="btnArm" onclick="armBot()" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition hidden">
          <i class="fa-solid fa-shield"></i> Re-Arm
        </button>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

    <!-- Kill Switch / Circuit Breaker Banner (hidden unless tripped) -->
    <div id="alertBanner" class="hidden p-4 rounded-xl border border-red-500/50 bg-red-950/40 text-red-200 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <i class="fa-solid fa-triangle-exclamation text-2xl text-red-400"></i>
        <div>
          <h3 class="font-bold text-sm text-red-300" id="alertTitle">Emergency Kill Switch Tripped</h3>
          <p class="text-xs text-red-200/80" id="alertMessage">Trading has been halted.</p>
        </div>
      </div>
      <button onclick="armBot()" class="px-3 py-1 rounded-lg bg-red-600 hover:bg-red-500 text-white text-xs font-bold transition">Re-Arm Bot</button>
    </div>

    <!-- Portfolio Metrics Grid -->
    <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3.5">
      <div class="bg-card p-4 rounded-xl border border-slate-800">
        <div class="text-xs font-medium text-slate-400">Total Portfolio</div>
        <div class="text-xl font-bold text-white mt-1" id="valEquity">$10,000.00</div>
        <div class="text-[11px] text-slate-500 mt-0.5">Mark-to-Market</div>
      </div>
      <div class="bg-card p-4 rounded-xl border border-slate-800">
        <div class="text-xs font-medium text-slate-400">Available Cash</div>
        <div class="text-xl font-bold text-white mt-1" id="valCash">$10,000.00</div>
        <div class="text-[11px] text-slate-500 mt-0.5">Liquid USDT Buffer</div>
      </div>
      <div class="bg-card p-4 rounded-xl border border-slate-800">
        <div class="text-xs font-medium text-slate-400">Unrealized PnL</div>
        <div class="text-xl font-bold text-emerald-400 mt-1" id="valUnrealized">$0.00</div>
        <div class="text-[11px] text-slate-500 mt-0.5">Open Positions</div>
      </div>
      <div class="bg-card p-4 rounded-xl border border-slate-800">
        <div class="text-xs font-medium text-slate-400">Realized PnL</div>
        <div class="text-xl font-bold text-emerald-400 mt-1" id="valRealized">$0.00</div>
        <div class="text-[11px] text-slate-500 mt-0.5">Closed Trades</div>
      </div>
      <div class="bg-card p-4 rounded-xl border border-slate-800">
        <div class="text-xs font-medium text-slate-400">Fees Paid</div>
        <div class="text-xl font-bold text-amber-400 mt-1" id="valFees">$0.00</div>
        <div class="text-[11px] text-slate-500 mt-0.5">0.10% Explicit Cost</div>
      </div>
      <div class="bg-card p-4 rounded-xl border border-slate-800">
        <div class="text-xs font-medium text-slate-400">Circuit Breaker</div>
        <div class="text-xl font-bold text-emerald-400 mt-1" id="valCircuit">ARMED</div>
        <div class="text-[11px] text-slate-500 mt-0.5" id="valMaxDD">DD: 0.0% / Max 15%</div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="flex border-b border-slate-800 space-x-6 text-sm font-semibold">
      <button onclick="switchTab('dashboard')" id="tabBtnDashboard" class="py-2.5 border-b-2 border-blue-500 text-blue-400">Live Dashboard & Market Intelligence</button>
      <button onclick="switchTab('backtest')" id="tabBtnBacktest" class="py-2.5 border-b-2 border-transparent text-slate-400 hover:text-slate-200">Interactive Backtesting</button>
      <button onclick="switchTab('trades')" id="tabBtnTrades" class="py-2.5 border-b-2 border-transparent text-slate-400 hover:text-slate-200">Execution Trade History</button>
      <button onclick="switchTab('settings')" id="tabBtnSettings" class="py-2.5 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center gap-1.5">
        <i class="fa-solid fa-gear text-amber-400"></i> API Keys & Settings
      </button>
    </div>

    <!-- TAB 1: Live Dashboard -->
    <div id="tabDashboard" class="space-y-6">
      <!-- Market Intelligence Card -->
      <div class="bg-card rounded-xl border border-slate-800 p-5">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
          <div class="flex items-center gap-3">
            <div class="flex items-center gap-1.5">
              <span class="text-xs text-slate-400 font-medium">Pair:</span>
              <select id="marketPairSelect" onchange="switchActivePair(this.value)" class="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs font-bold text-white focus:outline-none focus:border-blue-500">
                <option value="BTCUSDT">BTC/USDT</option>
                <option value="ETHUSDT">ETH/USDT</option>
                <option value="SOLUSDT">SOL/USDT</option>
                <option value="BNBUSDT">BNB/USDT</option>
                <option value="XRPUSDT">XRP/USDT</option>
                <option value="ADAUSDT">ADA/USDT</option>
                <option value="DOGEUSDT">DOGE/USDT</option>
                <option value="AVAXUSDT">AVAX/USDT</option>
              </select>
            </div>
            <span id="marketPrice" class="text-xl font-extrabold text-amber-400">$0.00</span>
            <span id="regimeBadge" class="px-2.5 py-0.5 rounded-full text-xs font-bold bg-slate-700 text-slate-300">ANALYZING...</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-xs text-slate-400">Verdict:</span>
            <span id="verdictBadge" class="px-3 py-1 rounded-lg text-xs font-bold bg-blue-600/30 text-blue-400 border border-blue-500/30">NEUTRAL</span>
            <span id="compositeScore" class="text-xs font-mono font-bold text-slate-300 ml-1">Score: 0.0/100</span>
          </div>
        </div>

        <!-- Indicator Matrix Grid -->
        <div class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 pt-4">
          <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
            <div class="text-[11px] text-slate-400">RSI (14)</div>
            <div id="indRSI" class="text-base font-bold text-white mt-0.5">--</div>
            <div id="indRSIState" class="text-[10px] text-slate-500">Momentum</div>
          </div>
          <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
            <div class="text-[11px] text-slate-400">MACD Histogram</div>
            <div id="indMACD" class="text-base font-bold text-white mt-0.5">--</div>
            <div class="text-[10px] text-slate-500">Acceleration</div>
          </div>
          <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
            <div class="text-[11px] text-slate-400">ADX (14)</div>
            <div id="indADX" class="text-base font-bold text-white mt-0.5">--</div>
            <div class="text-[10px] text-slate-500">Trend Strength</div>
          </div>
          <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
            <div class="text-[11px] text-slate-400">ATR Volatility</div>
            <div id="indATR" class="text-base font-bold text-white mt-0.5">--</div>
            <div id="indATRPct" class="text-[10px] text-slate-500">Per candle</div>
          </div>
          <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
            <div class="text-[11px] text-slate-400">Supertrend</div>
            <div id="indSupertrend" class="text-base font-bold text-white mt-0.5">--</div>
            <div class="text-[10px] text-slate-500">Trailing Support</div>
          </div>
          <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
            <div class="text-[11px] text-slate-400">Volume Ratio</div>
            <div id="indVolume" class="text-base font-bold text-white mt-0.5">--</div>
            <div class="text-[10px] text-slate-500">vs 20-period avg</div>
          </div>
          <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
            <div class="text-[11px] text-slate-400">Bollinger %B</div>
            <div id="indBB" class="text-base font-bold text-white mt-0.5">--</div>
            <div class="text-[10px] text-slate-500">Envelope Pos</div>
          </div>
        </div>

        <!-- Support / Resistance Pivots & Key Insights -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 pt-4 border-t border-slate-800">
          <div>
            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">🎯 Pivot & Support/Resistance Levels</h3>
            <div class="grid grid-cols-5 gap-2 text-center text-xs">
              <div class="bg-slate-900 p-2 rounded border border-slate-800">
                <div class="text-slate-500 text-[10px]">S2</div>
                <div id="lvlS2" class="font-bold text-red-400">--</div>
              </div>
              <div class="bg-slate-900 p-2 rounded border border-slate-800">
                <div class="text-slate-500 text-[10px]">S1</div>
                <div id="lvlS1" class="font-bold text-red-400">--</div>
              </div>
              <div class="bg-slate-900 p-2 rounded border border-slate-800">
                <div class="text-slate-500 text-[10px]">Pivot</div>
                <div id="lvlPivot" class="font-bold text-slate-200">--</div>
              </div>
              <div class="bg-slate-900 p-2 rounded border border-slate-800">
                <div class="text-slate-500 text-[10px]">R1</div>
                <div id="lvlR1" class="font-bold text-emerald-400">--</div>
              </div>
              <div class="bg-slate-900 p-2 rounded border border-slate-800">
                <div class="text-slate-500 text-[10px]">R2</div>
                <div id="lvlR2" class="font-bold text-emerald-400">--</div>
              </div>
            </div>
          </div>
          <div>
            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">💡 Key Market Observations</h3>
            <ul id="marketInsights" class="text-xs text-slate-300 space-y-1">
              <li>Loading market intelligence...</li>
            </ul>
          </div>
        </div>
      </div>

      <!-- Multi-Pair Watchlist Scanner -->
      <div class="bg-card rounded-xl border border-slate-800 overflow-hidden">
        <div class="p-4 border-b border-slate-800 flex items-center justify-between">
          <div class="flex items-center gap-2">
            <h3 class="text-sm font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-satellite-dish text-indigo-400"></i> Multi-Pair Market Scanner
            </h3>
            <span class="text-xs text-slate-400 font-mono" id="scannerPairCount">(Scanning...)</span>
          </div>
          <button onclick="fetchScanner()" class="text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition flex items-center gap-1.5">
            <i class="fa-solid fa-arrows-rotate"></i> Scan All Pairs
          </button>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs">
            <thead class="bg-slate-900/60 text-slate-400 uppercase tracking-wider text-[10px]">
              <tr>
                <th class="py-2.5 px-4">Pair</th>
                <th class="py-2.5 px-4 text-right">Price</th>
                <th class="py-2.5 px-4 text-center">Verdict</th>
                <th class="py-2.5 px-4 text-center">Score</th>
                <th class="py-2.5 px-4 text-center">Regime</th>
                <th class="py-2.5 px-4 text-right">RSI</th>
                <th class="py-2.5 px-4 text-right">ADX</th>
                <th class="py-2.5 px-4 text-center">Trend</th>
                <th class="py-2.5 px-4 text-center">Actions</th>
              </tr>
            </thead>
            <tbody id="scannerTableBody" class="divide-y divide-slate-800/60">
              <tr>
                <td colspan="9" class="py-4 text-center text-slate-500">Scanning pairs...</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Active Open Positions Table -->
      <div class="bg-card rounded-xl border border-slate-800 overflow-hidden">
        <div class="p-4 border-b border-slate-800 flex items-center justify-between">
          <h3 class="text-sm font-bold text-white flex items-center gap-2">
            <i class="fa-solid fa-briefcase text-blue-400"></i> Open Positions
          </h3>
          <div class="flex items-center gap-2">
            <button onclick="executeTestDemoBuy()" class="text-xs px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold transition flex items-center gap-1.5 shadow">
              <i class="fa-solid fa-cart-shopping"></i> Test Demo Buy ($500 BTC)
            </button>
            <button onclick="liquidatePositions()" class="text-xs px-2.5 py-1.5 rounded-lg bg-red-950/60 hover:bg-red-900/60 text-red-300 border border-red-800/60 transition">
              Close All Positions
            </button>
          </div>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs">
            <thead class="bg-slate-900/60 text-slate-400 uppercase tracking-wider text-[10px]">
              <tr>
                <th class="py-2.5 px-4">Symbol</th>
                <th class="py-2.5 px-4 text-right">Units</th>
                <th class="py-2.5 px-4 text-right">Entry Price</th>
                <th class="py-2.5 px-4 text-right">Mark Price</th>
                <th class="py-2.5 px-4 text-right">Stop Loss</th>
                <th class="py-2.5 px-4 text-right">Take Profit</th>
                <th class="py-2.5 px-4 text-right">Unrealized PnL</th>
                <th class="py-2.5 px-4 text-center">Strategy</th>
                <th class="py-2.5 px-4 text-center">Action</th>
              </tr>
            </thead>
            <tbody id="positionsTableBody" class="divide-y divide-slate-800/60">
              <tr>
                <td colspan="9" class="py-4 text-center text-slate-500">No open positions (Flat). Waiting for valid signal or click "Test Demo Buy".</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 2: Interactive Backtesting -->
    <div id="tabBacktest" class="hidden space-y-6">
      <div class="bg-card rounded-xl border border-slate-800 p-5">
        <h2 class="text-base font-bold text-white mb-4 flex items-center gap-2">
          <i class="fa-solid fa-flask text-blue-400"></i> Historical Backtest Simulator (No Lookahead Bias)
        </h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Symbol</label>
            <select id="btSymbol" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white">
              <option value="BTCUSDT">BTCUSDT</option>
              <option value="ETHUSDT">ETHUSDT</option>
              <option value="SOLUSDT">SOLUSDT</option>
              <option value="BNBUSDT">BNBUSDT</option>
            </select>
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Timeframe</label>
            <select id="btTimeframe" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white">
              <option value="15m">15m</option>
              <option value="1h" selected>1h</option>
              <option value="4h">4h</option>
              <option value="1d">1d</option>
            </select>
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Strategy</label>
            <select id="btStrategy" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white">
              <option value="adaptive" selected>Regime-Adaptive Meta Strategy</option>
              <option value="trend">Trend Momentum (EMA Ribbon + MACD)</option>
              <option value="mean_reversion">Mean Reversion (Bollinger + RSI)</option>
            </select>
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Lookback (Days)</label>
            <input type="number" id="btDays" value="60" min="7" max="180" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white">
          </div>
          <div class="flex items-end">
            <button onclick="runBacktest()" id="btnRunBT" class="w-full py-2 px-4 rounded-lg bg-blue-600 hover:bg-blue-500 font-bold text-xs text-white transition flex items-center justify-center gap-2">
              <i class="fa-solid fa-play"></i> Run Backtest
            </button>
          </div>
        </div>
      </div>

      <!-- Backtest Results Display (Hidden initially) -->
      <div id="btResultsContainer" class="hidden space-y-6">
        <!-- Backtest Metrics Cards -->
        <div class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3.5">
          <div class="bg-card p-4 rounded-xl border border-slate-800">
            <div class="text-xs text-slate-400">Strategy Return</div>
            <div id="btMetricReturn" class="text-xl font-bold mt-1 text-white">--</div>
          </div>
          <div class="bg-card p-4 rounded-xl border border-slate-800">
            <div class="text-xs text-slate-400">Benchmark (B&H)</div>
            <div id="btMetricBenchmark" class="text-xl font-bold mt-1 text-white">--</div>
          </div>
          <div class="bg-card p-4 rounded-xl border border-slate-800">
            <div class="text-xs text-slate-400">Sharpe Ratio</div>
            <div id="btMetricSharpe" class="text-xl font-bold mt-1 text-white">--</div>
          </div>
          <div class="bg-card p-4 rounded-xl border border-slate-800">
            <div class="text-xs text-slate-400">Max Drawdown</div>
            <div id="btMetricMDD" class="text-xl font-bold mt-1 text-red-400">--</div>
          </div>
          <div class="bg-card p-4 rounded-xl border border-slate-800">
            <div class="text-xs text-slate-400">Win Rate</div>
            <div id="btMetricWinRate" class="text-xl font-bold mt-1 text-white">--</div>
          </div>
          <div class="bg-card p-4 rounded-xl border border-slate-800">
            <div class="text-xs text-slate-400">Profit Factor</div>
            <div id="btMetricPF" class="text-xl font-bold mt-1 text-white">--</div>
          </div>
        </div>

        <!-- Equity Curve Chart -->
        <div class="bg-card p-5 rounded-xl border border-slate-800">
          <h3 class="text-sm font-bold text-white mb-3 flex items-center gap-2">
            <i class="fa-solid fa-chart-area text-blue-400"></i> Equity Curve ($)
          </h3>
          <div class="h-64">
            <canvas id="equityChartCanvas"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: Execution Trade History -->
    <div id="tabTrades" class="hidden space-y-6">
      <div class="bg-card rounded-xl border border-slate-800 overflow-hidden">
        <div class="p-4 border-b border-slate-800">
          <h3 class="text-sm font-bold text-white flex items-center gap-2">
            <i class="fa-solid fa-clock-rotate-left text-blue-400"></i> Executed Order & Trade Log
          </h3>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs">
            <thead class="bg-slate-900/60 text-slate-400 uppercase tracking-wider text-[10px]">
              <tr>
                <th class="py-2.5 px-4">Timestamp</th>
                <th class="py-2.5 px-4">Symbol</th>
                <th class="py-2.5 px-4 text-center">Side</th>
                <th class="py-2.5 px-4 text-right">Units</th>
                <th class="py-2.5 px-4 text-right">Fill Price</th>
                <th class="py-2.5 px-4 text-right">Fee ($)</th>
                <th class="py-2.5 px-4 text-right">Net PnL ($)</th>
                <th class="py-2.5 px-4 text-left">Reason / Strategy</th>
              </tr>
            </thead>
            <tbody id="tradesTableBody" class="divide-y divide-slate-800/60">
              <tr>
                <td colspan="8" class="py-4 text-center text-slate-500">No executed trades recorded yet.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 4: API Keys & Configuration Settings -->
    <div id="tabSettings" class="hidden space-y-6">
      <!-- Status Feedback Alert Box -->
      <div id="settingsFeedback" class="hidden p-4 rounded-xl border text-xs flex items-center justify-between">
        <div class="flex items-center gap-2.5">
          <i id="settingsFeedbackIcon" class="fa-solid text-lg"></i>
          <span id="settingsFeedbackMsg" class="font-medium"></span>
        </div>
        <button onclick="document.getElementById('settingsFeedback').classList.add('hidden')" class="text-slate-400 hover:text-white">
          <i class="fa-solid fa-xmark"></i>
        </button>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Main Form (Left 2 cols) -->
        <div class="lg:col-span-2 space-y-6">
          <!-- Trading Mode & Exchange Credentials Card -->
          <div class="bg-card rounded-xl border border-slate-800 p-6 space-y-5">
            <div class="flex items-center justify-between pb-3 border-b border-slate-800">
              <div>
                <h3 class="text-sm font-bold text-white flex items-center gap-2">
                  <i class="fa-solid fa-key text-amber-400"></i> Multi-Exchange & API Credentials
                </h3>
                <p class="text-xs text-slate-400 mt-0.5">Connect Binance, Bybit, OKX, Kraken, Coinbase, KuCoin, or Bitget.</p>
              </div>
              <span id="currentModeDisplay" class="text-xs font-bold px-2.5 py-1 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
                PAPER MODE
              </span>
            </div>

            <!-- Exchange Selector Dropdown -->
            <div>
              <label class="block text-xs font-bold text-slate-300 mb-1.5">Select Your Exchange</label>
              <select id="selectExchange" onchange="onExchangeChange(this.value)" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3.5 py-2.5 text-xs text-white font-semibold focus:outline-none focus:border-blue-500">
                <option value="binance" selected>Binance (Spot / Testnet)</option>
                <option value="bybit">Bybit (Unified / Spot / Testnet)</option>
                <option value="okx">OKX (Spot / Demo)</option>
                <option value="kraken">Kraken</option>
                <option value="coinbase">Coinbase Advanced</option>
                <option value="kucoin">KuCoin</option>
                <option value="bitget">Bitget</option>
                <option value="gateio">Gate.io</option>
                <option value="mexc">MEXC</option>
                <option value="htx">HTX (Huobi)</option>
              </select>
            </div>

            <!-- Trading Mode Selector -->
            <div>
              <label class="block text-xs font-bold text-slate-300 mb-2">Trading Execution Mode</label>
              <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <label class="cursor-pointer border border-slate-800 rounded-xl p-3 bg-slate-900 hover:border-slate-700 flex flex-col justify-between mode-option" id="labelModePaper">
                  <div class="flex items-center justify-between">
                    <span class="text-xs font-bold text-emerald-400 flex items-center gap-1.5">
                      <i class="fa-solid fa-shield-halved"></i> Paper Trading
                    </span>
                    <input type="radio" name="tradingModeRadio" value="paper" checked onchange="onModeChange(this.value)" class="text-blue-600 focus:ring-blue-500">
                  </div>
                  <p class="text-[11px] text-slate-400 mt-2">Zero risk simulated orders against live exchange market data.</p>
                </label>

                <label class="cursor-pointer border border-slate-800 rounded-xl p-3 bg-slate-900 hover:border-slate-700 flex flex-col justify-between mode-option" id="labelModeTestnet">
                  <div class="flex items-center justify-between">
                    <span class="text-xs font-bold text-cyan-400 flex items-center gap-1.5">
                      <i class="fa-solid fa-network-wired"></i> Sandbox / Testnet
                    </span>
                    <input type="radio" name="tradingModeRadio" value="testnet" onchange="onModeChange(this.value)" class="text-blue-600 focus:ring-blue-500">
                  </div>
                  <p class="text-[11px] text-slate-400 mt-2">Simulated funds on exchange demo matching engine.</p>
                </label>

                <label class="cursor-pointer border border-slate-800 rounded-xl p-3 bg-slate-900 hover:border-slate-700 flex flex-col justify-between mode-option" id="labelModeLive">
                  <div class="flex items-center justify-between">
                    <span class="text-xs font-bold text-amber-400 flex items-center gap-1.5">
                      <i class="fa-solid fa-coins"></i> Live Trading
                    </span>
                    <input type="radio" name="tradingModeRadio" value="live" onchange="onModeChange(this.value)" class="text-blue-600 focus:ring-blue-500">
                  </div>
                  <p class="text-[11px] text-slate-400 mt-2">Real orders with live capital on your selected exchange.</p>
                </label>
              </div>
            </div>

            <!-- API Key Input -->
            <div>
              <div class="flex items-center justify-between mb-1.5">
                <label id="lblApiKey" class="text-xs font-bold text-slate-300">Exchange API Key</label>
                <span id="apiKeyStatus" class="text-[10px] text-slate-500">Not configured</span>
              </div>
              <div class="relative">
                <input type="text" id="inputApiKey" placeholder="Enter your API Key" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3.5 py-2 text-xs font-mono text-white focus:outline-none focus:border-blue-500 placeholder-slate-600">
              </div>
              <p class="text-[11px] text-slate-500 mt-1">Leave blank if running purely in Paper Mode.</p>
            </div>

            <!-- API Secret Input -->
            <div>
              <div class="flex items-center justify-between mb-1.5">
                <label id="lblApiSecret" class="text-xs font-bold text-slate-300">Exchange API Secret</label>
                <span id="apiSecretStatus" class="text-[10px] text-slate-500">Not configured</span>
              </div>
              <div class="relative">
                <input type="password" id="inputApiSecret" placeholder="Enter your API Secret" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3.5 py-2 text-xs font-mono text-white focus:outline-none focus:border-blue-500 placeholder-slate-600 pr-10">
                <button type="button" onclick="toggleSecretVisibility('inputApiSecret', 'eyeIconSecret')" class="absolute right-3 top-2.5 text-slate-500 hover:text-slate-300">
                  <i id="eyeIconSecret" class="fa-solid fa-eye text-xs"></i>
                </button>
              </div>
            </div>

            <!-- API Passphrase Input (shown for OKX, KuCoin, Bitget) -->
            <div id="passphraseContainer" class="hidden">
              <div class="flex items-center justify-between mb-1.5">
                <label class="text-xs font-bold text-slate-300">API Passphrase / Password</label>
                <span class="text-[10px] text-amber-400">Required for OKX, KuCoin & Bitget</span>
              </div>
              <div class="relative">
                <input type="password" id="inputApiPassphrase" placeholder="Enter your API Passphrase" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3.5 py-2 text-xs font-mono text-white focus:outline-none focus:border-blue-500 placeholder-slate-600 pr-10">
                <button type="button" onclick="toggleSecretVisibility('inputApiPassphrase', 'eyeIconPass')" class="absolute right-3 top-2.5 text-slate-500 hover:text-slate-300">
                  <i id="eyeIconPass" class="fa-solid fa-eye text-xs"></i>
                </button>
              </div>
            </div>

            <!-- Test Connection Result Box -->
            <div id="connTestResult" class="hidden p-3.5 rounded-lg border text-xs"></div>

            <!-- Actions Row -->
            <div class="flex items-center justify-between pt-2">
              <button type="button" onclick="testApiConnection()" id="btnTestConn" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center gap-2 transition border border-slate-700">
                <i class="fa-solid fa-vial"></i> Test API Connection
              </button>
              <button type="button" onclick="saveSettings()" id="btnSaveSettings" class="px-5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold flex items-center gap-2 transition shadow-lg shadow-blue-500/20">
                <i class="fa-solid fa-floppy-disk"></i> Save Settings & API Keys
              </button>
            </div>
          </div>

          <!-- Risk Management & Pair Configuration Card -->
          <div class="bg-card rounded-xl border border-slate-800 p-6 space-y-4">
            <h3 class="text-sm font-bold text-white flex items-center gap-2 pb-3 border-b border-slate-800">
              <i class="fa-solid fa-sliders text-blue-400"></i> Pair & Risk Limits Configuration
            </h3>

            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label class="block text-xs font-bold text-slate-300 mb-1.5">Primary Active Pair</label>
                <input type="text" id="inputSymbol" value="BTCUSDT" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3.5 py-2 text-xs font-mono text-white uppercase focus:outline-none focus:border-blue-500">
                <p class="text-[10px] text-slate-500 mt-1">Default selected pair for dashboard view</p>
              </div>

              <div class="sm:col-span-2">
                <label class="block text-xs font-bold text-slate-300 mb-1.5">Multi-Pair Trading Watchlist (Comma-separated)</label>
                <input type="text" id="inputTradingPairs" value="BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, AVAXUSDT" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3.5 py-2 text-xs font-mono text-white uppercase focus:outline-none focus:border-blue-500">
                <p class="text-[10px] text-slate-500 mt-1">The bot continuously monitors and trades all pairs in this list (e.g. BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT)</p>
              </div>

              <div>
                <label class="block text-xs font-bold text-slate-300 mb-1.5">Default Timeframe</label>
                <select id="inputTimeframe" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3.5 py-2 text-xs text-white focus:outline-none focus:border-blue-500">
                  <option value="5m">5m (Scalping)</option>
                  <option value="15m">15m (Short-Term)</option>
                  <option value="1h" selected>1h (Standard Swing)</option>
                  <option value="4h">4h (Trend Follow)</option>
                  <option value="1d">1d (Macro Trend)</option>
                </select>
                <p class="text-[10px] text-slate-500 mt-1">Candle evaluation frequency</p>
              </div>

              <div>
                <label class="block text-xs font-bold text-slate-300 mb-1.5">Max Risk Per Trade (%)</label>
                <input type="number" id="inputRiskPct" step="0.1" min="0.1" max="5.0" value="1.5" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3.5 py-2 text-xs text-white focus:outline-none focus:border-blue-500">
                <p class="text-[10px] text-slate-500 mt-1">Fraction of equity risked via ATR stop-loss (Default: 1.5%)</p>
              </div>

              <div>
                <label class="block text-xs font-bold text-slate-300 mb-1.5">Max Daily Drawdown Breaker (%)</label>
                <input type="number" id="inputDailyDDPct" step="0.5" min="1.0" max="20.0" value="5.0" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3.5 py-2 text-xs text-white focus:outline-none focus:border-blue-500">
                <p class="text-[10px] text-slate-500 mt-1">Daily loss limit that pauses trading (Default: 5.0%)</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Sidebar: Security Guide & Info (Right 1 col) -->
        <div class="space-y-6">
          <!-- Security Notice Card -->
          <div class="bg-card rounded-xl border border-slate-800 p-5 space-y-3">
            <h4 class="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <i class="fa-solid fa-lock text-emerald-400"></i> Security First (Discipline #8)
            </h4>
            <div class="text-xs text-slate-300 space-y-2.5 leading-relaxed">
              <div class="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-500/30 text-emerald-300 text-[11px]">
                <i class="fa-solid fa-check-circle mr-1"></i> <strong>Safe Permissions:</strong> Enable only <em>"Read"</em> and <em>"Spot Trading"</em> permissions.
              </div>
              <div class="p-2.5 rounded-lg bg-red-950/40 border border-red-500/30 text-red-300 text-[11px]">
                <i class="fa-solid fa-triangle-exclamation mr-1"></i> <strong>NEVER</strong> enable <em>"Withdrawals"</em> on your API key.
              </div>
              <p class="text-[11px] text-slate-400">
                Your keys are stored exclusively in your local machine's <code class="text-blue-400">.env</code> file, protected by <code class="text-blue-400">.gitignore</code>.
              </p>
            </div>
          </div>

          <!-- Multi-Exchange Info Card -->
          <div class="bg-card rounded-xl border border-slate-800 p-5 space-y-3">
            <h4 class="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <i class="fa-solid fa-globe text-blue-400"></i> Supported Exchanges
            </h4>
            <p class="text-[11px] text-slate-400">
              The bot supports 100+ exchanges powered by CCXT:
            </p>
            <div class="flex flex-wrap gap-1.5 pt-1">
              <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-900 border border-slate-800 text-yellow-400">Binance</span>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-900 border border-slate-800 text-amber-400">Bybit</span>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-900 border border-slate-800 text-blue-400">OKX</span>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-900 border border-slate-800 text-purple-400">Kraken</span>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-900 border border-slate-800 text-cyan-400">Coinbase</span>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-900 border border-slate-800 text-emerald-400">KuCoin</span>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-900 border border-slate-800 text-indigo-400">Bitget</span>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-900 border border-slate-800 text-rose-400">Gate.io</span>
            </div>
          </div>

          <!-- Quick Actions -->
          <div class="bg-card rounded-xl border border-slate-800 p-5 space-y-2.5">
            <h4 class="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <i class="fa-solid fa-terminal text-purple-400"></i> Local Secrets File
            </h4>
            <p class="text-[11px] text-slate-400">You can also inspect or edit keys directly in the terminal via:</p>
            <div class="p-2 rounded bg-slate-900 font-mono text-[11px] text-slate-300 select-all border border-slate-800">
              cat /home/recoding/crypto_trading_bot/.env
            </div>
          </div>
        </div>
      </div>
    </div>

  </main>

  <script>
    let equityChart = null;

    function switchTab(tab) {
      document.getElementById('tabDashboard').classList.toggle('hidden', tab !== 'dashboard');
      document.getElementById('tabBacktest').classList.toggle('hidden', tab !== 'backtest');
      document.getElementById('tabTrades').classList.toggle('hidden', tab !== 'trades');
      document.getElementById('tabSettings').classList.toggle('hidden', tab !== 'settings');

      const tabs = ['dashboard', 'backtest', 'trades', 'settings'];
      tabs.forEach(t => {
        const btn = document.getElementById('tabBtn' + t.charAt(0).toUpperCase() + t.slice(1));
        if (t === tab) {
          btn.className = 'py-2.5 border-b-2 border-blue-500 text-blue-400 font-semibold flex items-center gap-1.5';
        } else {
          btn.className = 'py-2.5 border-b-2 border-transparent text-slate-400 hover:text-slate-200 font-semibold flex items-center gap-1.5';
        }
      });

      if (tab === 'settings') {
        loadSettings();
      }
    }

    function onExchangeChange(ex) {
      const needsPassphrase = ['okx', 'kucoin', 'bitget'].includes(ex);
      document.getElementById('passphraseContainer').classList.toggle('hidden', !needsPassphrase);

      const exNames = {
        'binance': 'Binance',
        'bybit': 'Bybit',
        'okx': 'OKX',
        'kraken': 'Kraken',
        'coinbase': 'Coinbase Advanced',
        'kucoin': 'KuCoin',
        'bitget': 'Bitget',
        'gateio': 'Gate.io',
        'mexc': 'MEXC',
        'htx': 'HTX'
      };
      const label = exNames[ex] || ex.toUpperCase();
      document.getElementById('lblApiKey').innerText = label + ' API Key';
      document.getElementById('lblApiSecret').innerText = label + ' API Secret';
      document.getElementById('inputApiKey').placeholder = 'Enter your ' + label + ' API Key';
      document.getElementById('inputApiSecret').placeholder = 'Enter your ' + label + ' API Secret';
    }

    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();

        // Update Header & Mode
        document.getElementById('modeBadge').innerText = data.mode.toUpperCase() + ' MODE';
        if (data.exchange_name) {
          document.getElementById('exchangeBadge').innerText = data.exchange_name.toUpperCase();
        }
        const isRunning = data.is_running;
        document.getElementById('btnStart').classList.toggle('hidden', isRunning);
        document.getElementById('btnStop').classList.toggle('hidden', !isRunning);

        // Update Portfolio Cards
        document.getElementById('valEquity').innerText = '$' + Number(data.equity).toLocaleString('en-US', {minimumFractionDigits: 2});
        document.getElementById('valCash').innerText = '$' + Number(data.cash).toLocaleString('en-US', {minimumFractionDigits: 2});

        const unRealEl = document.getElementById('valUnrealized');
        unRealEl.innerText = (data.unrealized_pnl >= 0 ? '+$' : '-$') + Math.abs(data.unrealized_pnl).toFixed(2);
        unRealEl.className = 'text-xl font-bold mt-1 ' + (data.unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-red-400');

        const realEl = document.getElementById('valRealized');
        realEl.innerText = (data.realized_pnl >= 0 ? '+$' : '-$') + Math.abs(data.realized_pnl).toFixed(2);
        realEl.className = 'text-xl font-bold mt-1 ' + (data.realized_pnl >= 0 ? 'text-emerald-400' : 'text-red-400');

        document.getElementById('valFees').innerText = '$' + Number(data.total_fees).toFixed(2);

        // Circuit Breaker & Kill Switch Alerts
        const alertBanner = document.getElementById('alertBanner');
        const cbEl = document.getElementById('valCircuit');
        const btnKill = document.getElementById('btnKill');
        const btnArm = document.getElementById('btnArm');

        if (data.kill_switch_triggered) {
          alertBanner.classList.remove('hidden');
          document.getElementById('alertTitle').innerText = 'EMERGENCY KILL SWITCH ACTIVE';
          document.getElementById('alertMessage').innerText = data.kill_switch_reason || 'Trading halted by operator flag.';
          cbEl.innerText = 'TRIPPED';
          cbEl.className = 'text-xl font-bold mt-1 text-red-500';
          btnKill.classList.add('hidden');
          btnArm.classList.remove('hidden');
        } else if (data.circuit_breaker.is_tripped) {
          alertBanner.classList.remove('hidden');
          document.getElementById('alertTitle').innerText = 'CIRCUIT BREAKER TRIGGERED';
          document.getElementById('alertMessage').innerText = data.circuit_breaker.reason || 'Drawdown limit reached.';
          cbEl.innerText = 'TRIPPED';
          cbEl.className = 'text-xl font-bold mt-1 text-red-500';
          btnKill.classList.remove('hidden');
          btnArm.classList.add('hidden');
        } else {
          alertBanner.classList.add('hidden');
          cbEl.innerText = 'ARMED';
          cbEl.className = 'text-xl font-bold mt-1 text-emerald-400';
          btnKill.classList.remove('hidden');
          btnArm.classList.add('hidden');
        }

        document.getElementById('valMaxDD').innerText = 'DD: ' + data.circuit_breaker.total_drawdown_pct + '% / Max 15%';

        // Update Positions Table
        const pTbody = document.getElementById('positionsTableBody');
        if (data.open_positions.length === 0) {
          pTbody.innerHTML = '<tr><td colspan="9" class="py-4 text-center text-slate-500">No open positions (Flat). Waiting for valid signal or click "Test Demo Buy".</td></tr>';
        } else {
          pTbody.innerHTML = data.open_positions.map(p => {
            const pnlCol = p.unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-red-400';
            const sign = p.unrealized_pnl >= 0 ? '+' : '';
            return `
              <tr class="hover:bg-slate-800/40">
                <td class="py-2.5 px-4 font-bold text-white">${p.symbol}</td>
                <td class="py-2.5 px-4 text-right font-mono">${p.units.toFixed(4)}</td>
                <td class="py-2.5 px-4 text-right">$${p.entry_price.toLocaleString()}</td>
                <td class="py-2.5 px-4 text-right font-bold text-amber-300">$${p.current_price.toLocaleString()}</td>
                <td class="py-2.5 px-4 text-right text-red-400">$${p.stop_loss.toLocaleString()}</td>
                <td class="py-2.5 px-4 text-right text-emerald-400">$${p.take_profit.toLocaleString()}</td>
                <td class="py-2.5 px-4 text-right font-bold ${pnlCol}">${sign}$${p.unrealized_pnl.toFixed(2)} (${p.unrealized_pnl_pct.toFixed(2)}%)</td>
                <td class="py-2.5 px-4 text-center text-slate-400">${p.strategy}</td>
                <td class="py-2.5 px-4 text-center">
                  <button onclick="closePosition('${p.symbol}')" class="px-2 py-0.5 rounded bg-red-600/30 hover:bg-red-600/60 text-red-300 text-[11px] font-semibold border border-red-500/30 transition">
                    Close
                  </button>
                </td>
              </tr>
            `;
          }).join('');
        }

        // Update Trades Table
        const tTbody = document.getElementById('tradesTableBody');
        if (data.recent_trades.length === 0) {
          tTbody.innerHTML = '<tr><td colspan="8" class="py-4 text-center text-slate-500">No executed trades recorded yet.</td></tr>';
        } else {
          tTbody.innerHTML = [...data.recent_trades].reverse().map(t => {
            const pnl = t.pnl_usd || 0;
            const pnlCol = pnl >= 0 ? 'text-emerald-400' : 'text-red-400';
            const pnlDisplay = t.side === 'BUY'
              ? '<span class="text-slate-500 font-normal">-</span>'
              : `<span class="${pnlCol}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}</span>`;
            return `
              <tr class="hover:bg-slate-800/40">
                <td class="py-2.5 px-4 font-mono text-[11px] text-slate-400">${(t.timestamp || '').slice(0, 19)}</td>
                <td class="py-2.5 px-4 font-bold text-white">${t.symbol}</td>
                <td class="py-2.5 px-4 text-center"><span class="px-2 py-0.5 rounded text-[10px] font-bold ${t.side === 'BUY' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}">${t.side}</span></td>
                <td class="py-2.5 px-4 text-right font-mono">${(t.quantity || 0).toFixed(4)}</td>
                <td class="py-2.5 px-4 text-right font-mono">$${(t.price || 0).toLocaleString()}</td>
                <td class="py-2.5 px-4 text-right text-slate-400">$${(t.fee || 0).toFixed(2)}</td>
                <td class="py-2.5 px-4 text-right font-bold">${pnlDisplay}</td>
                <td class="py-2.5 px-4 text-left text-slate-400">${t.reason || t.strategy || '-'}</td>
              </tr>
            `;
          }).join('');
        }

      } catch (err) {
        console.error('Failed to fetch status:', err);
      }
    let currentMarketSymbol = 'BTCUSDT';

    function switchActivePair(sym) {
      if (!sym) return;
      currentMarketSymbol = sym;
      const select = document.getElementById('marketPairSelect');
      if (select) select.value = sym;
      fetchAnalysis();
    }

    async function fetchAnalysis() {
      try {
        const res = await fetch('/api/analysis?symbol=' + encodeURIComponent(currentMarketSymbol) + '&timeframe=1h');
        const a = await res.json();
        if (a.error) return;

        document.getElementById('marketSymbol').innerText = a.symbol;
        document.getElementById('marketPrice').innerText = '$' + Number(a.latest_price).toLocaleString('en-US', {minimumFractionDigits: 2});

        // Regime Badge
        const rBadge = document.getElementById('regimeBadge');
        rBadge.innerText = a.regime.name.replace(/_/g, ' ');
        if (a.regime.name.includes('BULL')) rBadge.className = 'px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
        else if (a.regime.name.includes('BEAR')) rBadge.className = 'px-2.5 py-0.5 rounded-full text-xs font-bold bg-red-500/20 text-red-400 border border-red-500/30';
        else rBadge.className = 'px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30';

        // Verdict Badge
        const vBadge = document.getElementById('verdictBadge');
        vBadge.innerText = a.verdict.replace(/_/g, ' ');
        if (a.verdict.includes('BUY')) vBadge.className = 'px-3 py-1 rounded-lg text-xs font-bold bg-emerald-600 text-white';
        else if (a.verdict.includes('SELL')) vBadge.className = 'px-3 py-1 rounded-lg text-xs font-bold bg-red-600 text-white';
        else vBadge.className = 'px-3 py-1 rounded-lg text-xs font-bold bg-slate-700 text-slate-300';

        document.getElementById('compositeScore').innerText = 'Score: ' + (a.composite_score >= 0 ? '+' : '') + a.composite_score.toFixed(1) + '/100';

        // Indicators
        const ind = a.indicators;
        document.getElementById('indRSI').innerText = ind.rsi.toFixed(1);
        document.getElementById('indMACD').innerText = (ind.macd_hist >= 0 ? '+' : '') + ind.macd_hist.toFixed(2);
        document.getElementById('indADX').innerText = ind.adx.toFixed(1);
        document.getElementById('indATR').innerText = '$' + ind.atr.toFixed(2);
        document.getElementById('indATRPct').innerText = ind.atr_percent.toFixed(2) + '% of price';
        document.getElementById('indSupertrend').innerText = ind.supertrend_direction;
        document.getElementById('indVolume').innerText = ind.volume_ratio.toFixed(2) + 'x';
        document.getElementById('indBB').innerText = ind.bollinger_percent_b.toFixed(2);

        // Pivots
        document.getElementById('lvlS2').innerText = '$' + a.levels.support_2.toLocaleString();
        document.getElementById('lvlS1').innerText = '$' + a.levels.support_1.toLocaleString();
        document.getElementById('lvlPivot').innerText = '$' + a.levels.pivot.toLocaleString();
        document.getElementById('lvlR1').innerText = '$' + a.levels.resistance_1.toLocaleString();
        document.getElementById('lvlR2').innerText = '$' + a.levels.resistance_2.toLocaleString();

        // Insights
        document.getElementById('marketInsights').innerHTML = a.key_observations.map(obs => `<li>• ${obs}</li>`).join('');

      } catch (err) {
        console.error('Failed to fetch analysis:', err);
      }
    }

    async function loadSettings() {
      try {
        const res = await fetch('/api/settings');
        const s = await res.json();

        // Exchange Selector
        if (s.exchange_name) {
          document.getElementById('selectExchange').value = s.exchange_name;
          onExchangeChange(s.exchange_name);
        }

        // Mode Radio
        const modeRadios = document.getElementsByName('tradingModeRadio');
        for (let r of modeRadios) {
          if (r.value === s.trading_mode) {
            r.checked = true;
            onModeChange(r.value);
          }
        }

        document.getElementById('inputApiKey').value = s.api_key || '';
        document.getElementById('inputApiPassphrase').value = s.api_passphrase || '';
        document.getElementById('apiKeyStatus').innerText = s.api_key ? 'Configured (' + s.masked_key + ')' : 'Not configured';
        document.getElementById('apiSecretStatus').innerText = s.has_secret ? 'Secret Saved (Hidden)' : 'Not configured';

        document.getElementById('inputSymbol').value = s.default_symbol || 'BTCUSDT';
        if (document.getElementById('inputTradingPairs')) {
          document.getElementById('inputTradingPairs').value = s.trading_pairs || 'BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, AVAXUSDT';
        }
        document.getElementById('inputTimeframe').value = s.default_timeframe || '1h';
        document.getElementById('inputRiskPct').value = s.max_risk_per_trade_pct || 1.5;
        document.getElementById('inputDailyDDPct').value = s.max_daily_drawdown_pct || 5.0;

      } catch (err) {
        console.error('Failed to load settings:', err);
      }
    }

    function onModeChange(val) {
      document.getElementById('currentModeDisplay').innerText = val.toUpperCase() + ' MODE';
      const borderMap = {
        'paper': 'border-emerald-500/80',
        'testnet': 'border-cyan-500/80',
        'live': 'border-amber-500/80'
      };

      ['labelModePaper', 'labelModeTestnet', 'labelModeLive'].forEach(id => {
        const el = document.getElementById(id);
        el.className = 'cursor-pointer border border-slate-800 rounded-xl p-3 bg-slate-900 hover:border-slate-700 flex flex-col justify-between mode-option';
      });

      const activeId = val === 'paper' ? 'labelModePaper' : (val === 'testnet' ? 'labelModeTestnet' : 'labelModeLive');
      document.getElementById(activeId).className = 'cursor-pointer border ' + borderMap[val] + ' rounded-xl p-3 bg-slate-900 flex flex-col justify-between mode-option ring-1 ring-blue-500/30';
    }

    function toggleSecretVisibility(inputId, iconId) {
      const input = document.getElementById(inputId);
      const icon = document.getElementById(iconId);
      if (input.type === 'password') {
        input.type = 'text';
        icon.className = 'fa-solid fa-eye-slash text-xs text-blue-400';
      } else {
        input.type = 'password';
        icon.className = 'fa-solid fa-eye text-xs text-slate-500';
      }
    }

    async function testApiConnection() {
      const btn = document.getElementById('btnTestConn');
      const resultBox = document.getElementById('connTestResult');
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Testing Connection...';
      resultBox.classList.add('hidden');

      let selectedMode = 'paper';
      const modeRadios = document.getElementsByName('tradingModeRadio');
      for (let r of modeRadios) {
        if (r.checked) selectedMode = r.value;
      }

      const payload = {
        exchange_name: document.getElementById('selectExchange').value,
        api_key: document.getElementById('inputApiKey').value.trim(),
        api_secret: document.getElementById('inputApiSecret').value.trim(),
        api_passphrase: document.getElementById('inputApiPassphrase').value.trim(),
        use_testnet: (selectedMode === 'testnet')
      };

      try {
        const res = await fetch('/api/settings/test-connection', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        resultBox.classList.remove('hidden');
        if (data.success) {
          resultBox.className = 'p-3.5 rounded-lg border border-emerald-500/40 bg-emerald-950/40 text-emerald-200 text-xs';
          let balHtml = '';
          if (data.balances && Object.keys(data.balances).length > 0) {
            balHtml = '<div class="mt-2 font-mono text-[11px]">' + Object.entries(data.balances).map(([k, v]) => `• ${k}: ${v}`).join('  |  ') + '</div>';
          }
          resultBox.innerHTML = `<strong><i class="fa-solid fa-circle-check mr-1 text-emerald-400"></i> ${data.message}</strong>` + balHtml;
        } else {
          resultBox.className = 'p-3.5 rounded-lg border border-red-500/40 bg-red-950/40 text-red-200 text-xs';
          resultBox.innerHTML = `<strong><i class="fa-solid fa-circle-xmark mr-1 text-red-400"></i> ${data.message}</strong>`;
        }
      } catch (err) {
        resultBox.classList.remove('hidden');
        resultBox.className = 'p-3.5 rounded-lg border border-red-500/40 bg-red-950/40 text-red-200 text-xs';
        resultBox.innerHTML = `<strong>Connection test failed:</strong> ${err}`;
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-vial"></i> Test API Connection';
      }
    }

    async function saveSettings() {
      const btn = document.getElementById('btnSaveSettings');
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Saving...';

      let selectedMode = 'paper';
      const modeRadios = document.getElementsByName('tradingModeRadio');
      for (let r of modeRadios) {
        if (r.checked) selectedMode = r.value;
      }

      const payload = {
        exchange_name: document.getElementById('selectExchange').value,
        trading_mode: selectedMode,
        api_key: document.getElementById('inputApiKey').value.trim(),
        api_secret: document.getElementById('inputApiSecret').value.trim(),
        api_passphrase: document.getElementById('inputApiPassphrase').value.trim(),
        use_testnet: (selectedMode === 'testnet'),
        default_symbol: document.getElementById('inputSymbol').value.trim().toUpperCase(),
        trading_pairs: document.getElementById('inputTradingPairs') ? document.getElementById('inputTradingPairs').value.trim() : null,
        default_timeframe: document.getElementById('inputTimeframe').value,
        max_risk_per_trade_pct: parseFloat(document.getElementById('inputRiskPct').value) || 1.5,
        max_daily_drawdown_pct: parseFloat(document.getElementById('inputDailyDDPct').value) || 5.0,
      };

      try {
        const res = await fetch('/api/settings', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        // Show Feedback Box
        const feedback = document.getElementById('settingsFeedback');
        const icon = document.getElementById('settingsFeedbackIcon');
        const msg = document.getElementById('settingsFeedbackMsg');
        feedback.classList.remove('hidden');

        if (data.status === 'success') {
          feedback.className = 'p-4 rounded-xl border border-emerald-500/50 bg-emerald-950/40 text-emerald-200 text-xs flex items-center justify-between';
          icon.className = 'fa-solid fa-circle-check text-lg text-emerald-400';
          msg.innerText = data.message;
        } else {
          feedback.className = 'p-4 rounded-xl border border-red-500/50 bg-red-950/40 text-red-200 text-xs flex items-center justify-between';
          icon.className = 'fa-solid fa-triangle-exclamation text-lg text-red-400';
          msg.innerText = data.message || 'Failed to save settings.';
        }

        fetchStatus();
        loadSettings();

      } catch (err) {
        alert('Failed to save settings: ' + err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> Save Settings & API Keys';
      }
    }

    async function runBacktest() {
      const btn = document.getElementById('btnRunBT');
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Simulating...';

      const payload = {
        symbol: document.getElementById('btSymbol').value,
        timeframe: document.getElementById('btTimeframe').value,
        strategy: document.getElementById('btStrategy').value,
        days: parseInt(document.getElementById('btDays').value),
        cash: 10000.0
      };

      try {
        const res = await fetch('/api/backtest', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.error) {
          alert('Backtest error: ' + data.error);
          return;
        }

        document.getElementById('btResultsContainer').classList.remove('hidden');
        document.getElementById('btMetricReturn').innerText = data.metrics['Strategy Return'];
        document.getElementById('btMetricBenchmark').innerText = data.metrics['Benchmark (Buy & Hold)'];
        document.getElementById('btMetricSharpe').innerText = data.metrics['Sharpe Ratio'];
        document.getElementById('btMetricMDD').innerText = data.metrics['Max Drawdown'];
        document.getElementById('btMetricWinRate').innerText = data.metrics['Win Rate'];
        document.getElementById('btMetricPF').innerText = data.metrics['Profit Factor'];

        // Render Equity Chart
        renderEquityChart(data.equity_curve);

      } catch (err) {
        alert('Backtest failed: ' + err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-play"></i> Run Backtest';
      }
    }

    function renderEquityChart(curve) {
      const ctx = document.getElementById('equityChartCanvas').getContext('2d');
      if (equityChart) equityChart.destroy();

      const labels = curve.map(c => c.time);
      const values = curve.map(c => c.equity);

      equityChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [{
            label: 'Portfolio Equity ($)',
            data: values,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            fill: true,
            tension: 0.1,
            pointRadius: 0,
            borderWidth: 2
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { display: false },
            y: {
              grid: { color: '#334155' },
              ticks: { color: '#94a3b8' }
            }
          },
          plugins: {
            legend: { display: false }
          }
        }
      });
    }

    async function startBot() {
      await fetch('/api/bot/start', {method: 'POST'});
      fetchStatus();
    }

    async function stopBot() {
      await fetch('/api/bot/stop', {method: 'POST'});
      fetchStatus();
    }

    async function stepBot() {
      await fetch('/api/bot/step', {method: 'POST'});
      fetchStatus();
      fetchAnalysis();
    }

    async function killBot() {
      if (confirm('Are you sure you want to trigger the EMERGENCY KILL SWITCH?')) {
        await fetch('/api/bot/kill', {method: 'POST'});
        fetchStatus();
      }
    }

    async function armBot() {
      await fetch('/api/bot/arm', {method: 'POST'});
      fetchStatus();
    }

    async function liquidatePositions() {
      if (confirm('Close all open positions at market price?')) {
        await fetch('/api/bot/liquidate', {method: 'POST'});
        fetchStatus();
      }
    }

    async function executeTestDemoBuy(symbol) {
      const sym = symbol || currentMarketSymbol || 'BTCUSDT';
      try {
        const res = await fetch('/api/bot/test-trade', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({symbol: sym, amount_usd: 500.0})
        });
        const data = await res.json();
        if (res.ok) {
          fetchStatus();
          fetchScanner();
          alert(`Demo Order Executed!\nBought ${data.quantity} ${data.symbol} @ $${Number(data.price).toLocaleString()}\nStop Loss: $${data.stop_loss} | Take Profit: $${data.take_profit}`);
        } else {
          alert(`Error: ${data.error || 'Failed to execute test trade'}`);
        }
      } catch (err) {
        alert('Test trade error: ' + err.message);
      }
    }

    async function closePosition(symbol) {
      if (confirm(`Market sell & close position in ${symbol}?`)) {
        try {
          const res = await fetch('/api/bot/close-position', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({symbol: symbol})
          });
          const data = await res.json();
          if (res.ok) {
            fetchStatus();
            fetchScanner();
          } else {
            alert(`Error: ${data.error || 'Failed to close position'}`);
          }
        } catch (err) {
          alert('Error closing position: ' + err.message);
        }
      }
    }

    async function fetchScanner() {
      const tbody = document.getElementById('scannerTableBody');
      if (!tbody) return;
      try {
        const res = await fetch('/api/scanner?timeframe=1h');
        const pairs = await res.json();
        if (!pairs || pairs.length === 0) {
          tbody.innerHTML = '<tr><td colspan="9" class="py-4 text-center text-slate-500">No pairs scanned.</td></tr>';
          return;
        }
        document.getElementById('scannerPairCount').innerText = `(${pairs.length} pairs monitored)`;
        tbody.innerHTML = pairs.map(p => {
          let verdictBadge = 'bg-blue-600/30 text-blue-400 border-blue-500/30';
          if (p.verdict.includes('BUY')) verdictBadge = 'bg-emerald-600/30 text-emerald-400 border-emerald-500/30';
          else if (p.verdict.includes('SELL')) verdictBadge = 'bg-red-600/30 text-red-400 border-red-500/30';

          const scoreCol = p.composite_score > 15 ? 'text-emerald-400' : (p.composite_score < -15 ? 'text-red-400' : 'text-slate-300');
          const posBadge = p.has_position ? '<span class="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-bold text-[10px]">OPEN</span>' : '<span class="text-slate-500 text-[10px]">FLAT</span>';

          return `
            <tr class="hover:bg-slate-800/40">
              <td class="py-2.5 px-4 font-bold text-white flex items-center gap-1.5">
                ${p.symbol} ${posBadge}
              </td>
              <td class="py-2.5 px-4 text-right font-mono font-bold text-amber-300">$${Number(p.price).toLocaleString()}</td>
              <td class="py-2.5 px-4 text-center"><span class="px-2 py-0.5 rounded text-[10px] font-bold border ${verdictBadge}">${p.verdict}</span></td>
              <td class="py-2.5 px-4 text-center font-mono font-bold ${scoreCol}">${p.composite_score > 0 ? '+' : ''}${p.composite_score}</td>
              <td class="py-2.5 px-4 text-center text-slate-400">${p.regime.replace(/_/g, ' ')}</td>
              <td class="py-2.5 px-4 text-right font-mono">${p.rsi}</td>
              <td class="py-2.5 px-4 text-right font-mono">${p.adx}</td>
              <td class="py-2.5 px-4 text-center text-slate-300">${p.trend_bias}</td>
              <td class="py-2.5 px-4 text-center space-x-1.5">
                <button onclick="switchActivePair('${p.symbol}')" class="px-2 py-0.5 rounded bg-blue-600/30 hover:bg-blue-600/60 text-blue-300 text-[11px] font-semibold border border-blue-500/30 transition">Analyze</button>
                <button onclick="executeTestDemoBuy('${p.symbol}')" class="px-2 py-0.5 rounded bg-emerald-600/30 hover:bg-emerald-600/60 text-emerald-300 text-[11px] font-semibold border border-emerald-500/30 transition">Test Buy</button>
              </td>
            </tr>
          `;
        }).join('');
      } catch (err) {
        console.error('Failed to fetch scanner:', err);
      }
    }

    // Initial Load & Interval Polling
    fetchStatus();
    fetchAnalysis();
    fetchScanner();
    setInterval(fetchStatus, 4000);
    setInterval(fetchAnalysis, 15000);
    setInterval(fetchScanner, 20000);
  </script>
</body>
</html>
"""
