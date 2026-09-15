"""Rich Terminal Dashboard for Real-Time Monitoring.

Renders high-visibility terminal panels displaying market analysis,
active positions, portfolio health, risk limits, and trade history.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout

from bot.execution.order import Position
from bot.risk.circuit_breaker import CircuitBreakerStatus
from bot.analysis.regime import RegimeAnalysisResult


class BotDashboard:
    """Renders formatted console dashboard using rich."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def render_summary(
        self,
        mode: str,
        symbol: str,
        timeframe: str,
        current_price: float,
        equity: float,
        cash: float,
        realized_pnl: float,
        unrealized_pnl: float,
        total_fees: float,
        regime: Optional[RegimeAnalysisResult],
        indicators: Dict[str, Any],
        positions: Dict[str, Position],
        cb_status: Optional[CircuitBreakerStatus],
        recent_trades: List[Dict[str, Any]],
        strategy_name: str,
    ) -> None:
        """Render snapshot table to console."""
        self.console.clear()

        # 1. Header Table
        header_table = Table.grid(expand=True)
        header_table.add_column(justify="left")
        header_table.add_column(justify="right")

        mode_badge = f"[bold white on red] {mode.upper()} [/]" if mode == "live" else f"[bold black on cyan] {mode.upper()} MODE [/]"
        header_table.add_row(
            Text.from_markup(f"🚀 [bold cyan]CRYPTO TRADING BOT[/] | {mode_badge} | [yellow]{symbol}[/] ({timeframe}) | Strategy: [bold green]{strategy_name}[/]"),
            Text.from_markup(f"[dim]{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}[/]"),
        )
        self.console.print(Panel(header_table, style="bold blue"))

        # 2. Portfolio Overview Table
        p_table = Table(title="💰 Portfolio & Risk Accounting", expand=True)
        p_table.add_column("Total Equity", justify="right")
        p_table.add_column("Available Cash", justify="right")
        p_table.add_column("Unrealized PnL", justify="right")
        p_table.add_column("Realized PnL", justify="right")
        p_table.add_column("Total Fees", justify="right")
        p_table.add_column("Max Drawdown", justify="right")
        p_table.add_column("Circuit Breaker", justify="center")

        u_color = "green" if unrealized_pnl >= 0 else "red"
        r_color = "green" if realized_pnl >= 0 else "red"
        cb_color = "bold green" if (not cb_status or not cb_status.is_tripped) else "bold red"
        cb_text = "ARMED (OK)" if (not cb_status or not cb_status.is_tripped) else f"TRIPPED: {cb_status.reason}"

        p_table.add_row(
            f"[bold]${equity:,.2f}[/]",
            f"${cash:,.2f}",
            f"[{u_color}]${unrealized_pnl:+,.2f}[/]",
            f"[{r_color}]${realized_pnl:+,.2f}[/]",
            f"${total_fees:,.2f}",
            f"{cb_status.total_drawdown_pct*100:.2f}%" if cb_status else "0.00%",
            f"[{cb_color}]{cb_text}[/]",
        )
        self.console.print(p_table)

        # 3. Market Analysis Panel
        m_table = Table(title=f"📊 Market Analysis ({symbol})", expand=True)
        m_table.add_column("Mark Price", justify="right")
        m_table.add_column("Regime", justify="center")
        m_table.add_column("Strategy Mode", justify="center")
        m_table.add_column("RSI(14)", justify="center")
        m_table.add_column("ADX(14)", justify="center")
        m_table.add_column("ATR", justify="right")
        m_table.add_column("MACD Hist", justify="right")
        m_table.add_column("Supertrend", justify="center")

        reg_name = regime.regime.value if regime else "N/A"
        strat_mode = regime.strategy_mode.value if regime else "N/A"
        rsi_val = indicators.get("rsi", 50.0)
        rsi_style = "red" if rsi_val > 70 or rsi_val < 30 else "green"

        m_table.add_row(
            f"[bold yellow]${current_price:,.2f}[/]",
            f"[cyan]{reg_name}[/]",
            f"[magenta]{strat_mode}[/]",
            f"[{rsi_style}]{rsi_val:.1f}[/]",
            f"{indicators.get('adx', 0.0):.1f}",
            f"${indicators.get('atr', 0.0):.2f}",
            f"{indicators.get('macd_hist', 0.0):+.2f}",
            f"{indicators.get('supertrend_direction', 'N/A')}",
        )
        self.console.print(m_table)

        # 4. Open Positions Table
        pos_table = Table(title="📌 Open Positions", expand=True)
        pos_table.add_column("Symbol", justify="left")
        pos_table.add_column("Units", justify="right")
        pos_table.add_column("Entry Price", justify="right")
        pos_table.add_column("Current Price", justify="right")
        pos_table.add_column("Stop Loss", justify="right")
        pos_table.add_column("Take Profit", justify="right")
        pos_table.add_column("Unrealized PnL", justify="right")
        pos_table.add_column("PnL %", justify="right")

        if not positions:
            pos_table.add_row("-", "0.0", "$0.00", "$0.00", "$0.00", "$0.00", "$0.00", "0.00%")
        else:
            for s, pos in positions.items():
                p_pnl = pos.unrealized_pnl
                p_col = "green" if p_pnl >= 0 else "red"
                pos_table.add_row(
                    f"[bold]{s}[/]",
                    f"{pos.units:.6f}",
                    f"${pos.entry_price:,.2f}",
                    f"${pos.current_price:,.2f}",
                    f"[red]${pos.stop_loss:,.2f}[/]",
                    f"[green]${pos.take_profit:,.2f}[/]",
                    f"[{p_col}]${p_pnl:+,.2f}[/]",
                    f"[{p_col}]{pos.unrealized_pnl_pct:+.2f}%[/]",
                )
        self.console.print(pos_table)

        # 5. Recent Trades Table
        t_table = Table(title="📜 Recent Executed Trades", expand=True)
        t_table.add_column("Timestamp", justify="left")
        t_table.add_column("Symbol", justify="left")
        t_table.add_column("Side", justify="center")
        t_table.add_column("Units", justify="right")
        t_table.add_column("Fill Price", justify="right")
        t_table.add_column("PnL ($)", justify="right")
        t_table.add_column("Reason", justify="left")

        recent = recent_trades[-5:]
        if not recent:
            t_table.add_row("-", "-", "-", "0.0", "$0.00", "$0.00", "No trades yet")
        else:
            for t in reversed(recent):
                pnl = t.get("pnl_usd", 0.0)
                t_col = "green" if pnl >= 0 else "red"
                t_table.add_row(
                    str(t.get("timestamp", ""))[:19],
                    str(t.get("symbol", "")),
                    str(t.get("side", "")),
                    f"{t.get('quantity', 0.0):.6f}",
                    f"${t.get('price', 0.0):,.2f}",
                    f"[{t_col}]${pnl:+,.2f}[/]",
                    str(t.get("reason", t.get("strategy", ""))),
                )
        self.console.print(t_table)
        self.console.print("[dim]Press Ctrl+C or run 'python -m bot.cli kill' to safely halt bot.[/dim]")
