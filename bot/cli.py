"""Command Line Interface (CLI) for Crypto Trading Bot.

Provides intuitive commands to run the live bot, execute backtests,
inspect market analysis, check status, and trigger kill switches.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from bot.config import BotConfig, TradingMode
from bot.data.fetcher import DataFetcher
from bot.analysis.market_analyzer import MarketAnalyzer
from bot.strategies.regime_adaptive import RegimeAdaptiveStrategy
from bot.strategies.trend_momentum import TrendMomentumStrategy
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.backtesting.engine import BacktestEngine
from bot.engine.trader import TradingBot
from bot.engine.kill_switch import KillSwitch
from bot.engine.state import BotState

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get_strategy(strat_name: str):
    name = strat_name.lower()
    if name in ("adaptive", "regime", "regime_adaptive"):
        return RegimeAdaptiveStrategy()
    elif name in ("trend", "momentum", "trend_momentum"):
        return TrendMomentumStrategy()
    elif name in ("mean_reversion", "mean", "range"):
        return MeanReversionStrategy()
    else:
        console.print(f"[yellow]Unknown strategy '{strat_name}', defaulting to 'adaptive'.[/yellow]")
        return RegimeAdaptiveStrategy()


def cmd_run(args: argparse.Namespace) -> None:
    """Run bot in paper, testnet, or live mode."""
    cfg = BotConfig.load_from_env()
    if args.mode:
        cfg.trading_mode = TradingMode(args.mode)
    if args.symbol:
        cfg.default_symbol = args.symbol.upper()
    if args.timeframe:
        cfg.default_timeframe = args.timeframe

    cfg.validate_safety()
    strategy = get_strategy(args.strategy)

    console.print(Panel(
        f"[bold green]Starting Crypto Trading Bot[/]\n"
        f"Mode: [bold cyan]{cfg.trading_mode.value.upper()}[/] | Symbol: [bold yellow]{cfg.default_symbol}[/] | Timeframe: [bold]{cfg.default_timeframe}[/]\n"
        f"Strategy: [bold magenta]{strategy.name}[/]",
        title="🤖 Initialization",
        style="blue",
    ))

    bot = TradingBot(config=cfg, strategy=strategy, enable_ui=not args.no_ui)

    if args.once:
        bot.step()
    else:
        bot.run_loop(poll_interval_seconds=args.interval, max_iterations=args.max_iterations)


def cmd_backtest(args: argparse.Namespace) -> None:
    """Execute historical backtest with institutional analytics."""
    symbol = args.symbol.upper()
    timeframe = args.timeframe
    days = args.days
    cash = args.cash

    console.print(Panel(
        f"Running Backtest for [bold yellow]{symbol}[/] ({timeframe})\n"
        f"Window: [cyan]{days} days[/] | Initial Equity: [green]${cash:,.2f}[/]\n"
        f"Strategy: [magenta]{args.strategy}[/]",
        title="📈 Backtesting Engine",
        style="cyan",
    ))

    fetcher = DataFetcher()
    with console.status(f"[bold cyan]Fetching {days} days of {timeframe} data for {symbol}...[/]"):
        df = fetcher.fetch_historical_range(symbol, timeframe=timeframe, days=days)

    console.print(f"Loaded [bold]{len(df)}[/] historical candles ({df.index[0]} to {df.index[-1]}).")

    strat = get_strategy(args.strategy)
    engine = BacktestEngine(strategy=strat, initial_cash=cash)

    with console.status("[bold green]Simulating bar-by-bar execution without lookahead bias...[/]"):
        res = engine.run(df, symbol=symbol, timeframe=timeframe)

    # Render Performance Table
    table = Table(title=f"📊 Backtest Performance: {symbol} ({strat.name})", expand=True)
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("Value", style="bold yellow", justify="right")

    for k, v in res.metrics.to_dict().items():
        table.add_row(k, str(v))

    console.print(table)

    # Render Trades Table
    if res.trades:
        trades_table = Table(title=f"📜 Trade Executions ({len(res.trades)} total)", expand=True)
        trades_table.add_column("Entry Time", justify="left")
        trades_table.add_column("Exit Time", justify="left")
        trades_table.add_column("Units", justify="right")
        trades_table.add_column("Entry Px", justify="right")
        trades_table.add_column("Exit Px", justify="right")
        trades_table.add_column("Fee ($)", justify="right")
        trades_table.add_column("PnL ($)", justify="right")
        trades_table.add_column("PnL %", justify="right")
        trades_table.add_column("Exit Reason", justify="left")

        for t in res.trades[-15:]:  # show up to last 15
            pnl = t.get("pnl_usd", 0.0)
            col = "green" if pnl >= 0 else "red"
            trades_table.add_row(
                str(t.get("entry_time"))[:16],
                str(t.get("exit_time"))[:16],
                f"{t.get('units', 0.0):.4f}",
                f"${t.get('entry_price', 0.0):,.2f}",
                f"${t.get('exit_price', 0.0):,.2f}",
                f"${t.get('fee', 0.0):.2f}",
                f"[{col}]${pnl:+,.2f}[/]",
                f"[{col}]{t.get('pnl_pct', 0.0):+.2f}%[/]",
                str(t.get("reason", "")),
            )
        console.print(trades_table)
    else:
        console.print("[dim]No trades executed with given strategy parameters in this period.[/dim]")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run top-notch multi-factor market analysis report."""
    symbol = args.symbol.upper()
    timeframe = args.timeframe

    with console.status(f"[bold cyan]Fetching real-time data for {symbol} ({timeframe})...[/]"):
        fetcher = DataFetcher()
        df = fetcher.fetch_klines(symbol, timeframe=timeframe, limit=200)

    analyzer = MarketAnalyzer()
    report = analyzer.analyze(df, symbol=symbol, timeframe=timeframe)

    # 1. Verdict Banner
    verdict_colors = {
        "STRONG_BUY": "bold white on green",
        "BUY": "bold black on green",
        "NEUTRAL": "bold black on yellow",
        "SELL": "bold white on red",
        "STRONG_SELL": "bold white on dark_red",
    }
    v_style = verdict_colors.get(report.verdict, "bold white on blue")

    console.print(Panel(
        f"[bold]Symbol:[/] {report.symbol} | [bold]Timeframe:[/] {report.timeframe} | [bold]Price:[/] ${report.latest_price:,.2f}\n"
        f"[bold]Market Regime:[/] [cyan]{report.regime.regime.value}[/] ({report.regime.summary})\n"
        f"[bold]Composite Technical Score:[/] [{v_style}] {report.composite_score:+.1f} / 100 [{report.verdict}] [/]",
        title="🔍 Top-Notch Market Intelligence Report",
        style="blue",
    ))

    # 2. Indicators Table
    ind_table = Table(title="📐 Key Technical Indicators", expand=True)
    ind_table.add_column("Indicator", justify="left")
    ind_table.add_column("Value", justify="right")
    ind_table.add_column("Interpretation", justify="left")

    ind = report.indicators
    rsi_val = ind["rsi"]
    rsi_interp = "Overbought (>70)" if rsi_val > 70 else ("Oversold (<30)" if rsi_val < 30 else "Neutral / Healthy")
    ind_table.add_row("RSI (14)", f"{rsi_val:.1f}", rsi_interp)

    macd_hist = ind["macd_hist"]
    macd_interp = "Bullish momentum acceleration" if macd_hist > 0 else "Bearish momentum expansion"
    ind_table.add_row("MACD Histogram", f"{macd_hist:+.2f}", macd_interp)

    adx_val = ind["adx"]
    adx_interp = "Strong directional trend" if adx_val >= 25 else "Ranging / Consolidating"
    ind_table.add_row("ADX (14)", f"{adx_val:.1f}", adx_interp)

    ind_table.add_row("ATR (14)", f"${ind['atr']:.2f} ({ind['atr_percent']:.2f}%)", "Expected average candle volatility")
    ind_table.add_row("EMA 9 / 21 / 50", f"${ind['ema9']:.1f} / ${ind['ema21']:.1f} / ${ind['ema50']:.1f}", "Moving average ribbon")
    ind_table.add_row("Bollinger Bandwidth", f"{ind['bollinger_bandwidth']*100:.2f}% (%B: {ind['bollinger_percent_b']:.2f})", "Volatility envelope")
    ind_table.add_row("Supertrend", f"{ind['supertrend_direction']} (${ind['supertrend_value']:.2f})", "Dynamic trend trail")
    ind_table.add_row("Volume Ratio", f"{ind['volume_ratio']:.2f}x", "Current volume relative to 20-period SMA")

    console.print(ind_table)

    # 3. Support & Resistance Table
    sr_table = Table(title="🎯 Key Pivot & Support/Resistance Levels", expand=True)
    sr_table.add_column("Level Type", justify="left")
    sr_table.add_column("Price Level", justify="right")
    sr_table.add_column("Distance", justify="right")

    levels = [
        ("Resistance 2 (R2)", report.levels.resistance_2),
        ("Resistance 1 (R1)", report.levels.resistance_1),
        ("Pivot Point (P)", report.levels.pivot),
        ("Support 1 (S1)", report.levels.support_1),
        ("Support 2 (S2)", report.levels.support_2),
        ("20-Bar Swing High", report.levels.recent_swing_high),
        ("20-Bar Swing Low", report.levels.recent_swing_low),
    ]

    for name, lvl in levels:
        dist_pct = ((lvl - report.latest_price) / report.latest_price) * 100.0
        sr_table.add_row(name, f"${lvl:,.2f}", f"{dist_pct:+.2f}%")

    console.print(sr_table)

    # 4. Key Observations
    obs_text = "\n".join(f"• {obs}" for obs in report.key_observations)
    console.print(Panel(obs_text, title="💡 Actionable Market Insights", style="yellow"))


def cmd_status(args: argparse.Namespace) -> None:
    """Inspect saved bot state, open positions, and cash balance."""
    state = BotState()
    data = state.load()
    if not data:
        console.print("[yellow]No active state found. Bot is clean and ready.[/yellow]")
        return

    table = Table(title="📦 Bot State Snapshot", expand=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Available Cash", f"${data.get('cash', 0.0):,.2f}")
    table.add_row("High Water Mark", f"${data.get('high_water_mark', 0.0):,.2f}")
    table.add_row("Open Positions", str(len(data.get("positions", {}))))
    table.add_row("Total Trades Recorded", str(len(data.get("trade_history", []))))
    table.add_row("Last Saved Timestamp", str(data.get("last_saved", "N/A")))

    console.print(table)


def cmd_kill(args: argparse.Namespace) -> None:
    """Trigger emergency kill switch."""
    ks = KillSwitch()
    reason = args.reason or "Manual CLI invocation"
    ks.trigger(reason)
    console.print(f"[bold red]EMERGENCY KILL SWITCH ACTIVATED: {reason}[/bold red]")


def cmd_arm(args: argparse.Namespace) -> None:
    """Reset emergency kill switch."""
    ks = KillSwitch()
    ks.reset()
    console.print("[bold green]Kill switch cleared. Bot is re-armed and ready to trade.[/bold green]")


def cmd_web(args: argparse.Namespace) -> None:
    """Launch the interactive web dashboard on localhost."""
    import uvicorn
    from bot.web.server import app
    port = args.port
    host = args.host
    console.print(Panel(
        f"[bold green]Starting Crypto Trading Bot Web Dashboard[/]\n"
        f"URL: [bold cyan]http://localhost:{port}[/] (or http://127.0.0.1:{port})\n"
        f"Interactive controls: Live metrics, Market intelligence, Backtesting & Kill switch.",
        title="🌐 Localhost Web Dashboard",
        style="cyan",
    ))
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Crypto Trading Bot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Web command
    p_web = subparsers.add_parser("web", help="Start web dashboard on localhost")
    p_web.add_argument("--port", type=int, default=8000, help="Port to bind server (default: 8000)")
    p_web.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    p_web.set_defaults(func=cmd_web)

    # Run command
    p_run = subparsers.add_parser("run", help="Run the trading bot")
    p_run.add_argument("--mode", choices=["paper", "testnet", "live"], default=None, help="Trading execution mode")
    p_run.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol")
    p_run.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe (e.g. 15m, 1h, 4h)")
    p_run.add_argument("--strategy", type=str, default="adaptive", help="Strategy (adaptive, trend, mean_reversion)")
    p_run.add_argument("--interval", type=int, default=15, help="Poll interval in seconds")
    p_run.add_argument("--max-iterations", type=int, default=None, help="Max loop iterations (optional)")
    p_run.add_argument("--no-ui", action="store_true", help="Disable Rich dashboard UI")
    p_run.add_argument("--once", action="store_true", help="Run a single evaluation step and exit")
    p_run.set_defaults(func=cmd_run)

    # Backtest command
    p_bt = subparsers.add_parser("backtest", help="Run backtest over historical data")
    p_bt.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol (e.g. BTCUSDT, ETHUSDT)")
    p_bt.add_argument("--timeframe", type=str, default="1h", help="Timeframe (e.g. 15m, 1h, 4h)")
    p_bt.add_argument("--days", type=int, default=60, help="Number of historical days to backtest")
    p_bt.add_argument("--strategy", type=str, default="adaptive", help="Strategy to evaluate")
    p_bt.add_argument("--cash", type=float, default=10_000.0, help="Starting cash capital")
    p_bt.set_defaults(func=cmd_backtest)

    # Analyze command
    p_an = subparsers.add_parser("analyze", help="Run top-notch multi-factor market analysis")
    p_an.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to analyze")
    p_an.add_argument("--timeframe", type=str, default="1h", help="Timeframe (e.g. 15m, 1h, 4h)")
    p_an.set_defaults(func=cmd_analyze)

    # Status command
    p_st = subparsers.add_parser("status", help="Inspect saved bot state")
    p_st.set_defaults(func=cmd_status)

    # Kill command
    p_kl = subparsers.add_parser("kill", help="Trigger emergency kill switch")
    p_kl.add_argument("--reason", type=str, default="Manual kill command", help="Reason for emergency stop")
    p_kl.set_defaults(func=cmd_kill)

    # Arm command
    p_arm = subparsers.add_parser("arm", help="Reset and arm kill switch")
    p_arm.set_defaults(func=cmd_arm)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
