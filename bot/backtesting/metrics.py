"""Backtest Performance Analytics and Metrics.

Calculates institutional-grade risk and performance statistics:
Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor, and Fee impact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    initial_equity: float
    final_equity: float
    total_return_pct: float
    benchmark_return_pct: float  # Buy & Hold return
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    avg_win_usd: float
    avg_loss_usd: float
    payoff_ratio: float  # avg_win / abs(avg_loss)
    total_fees_paid: float
    total_slippage_lost: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Initial Equity": f"${self.initial_equity:,.2f}",
            "Final Equity": f"${self.final_equity:,.2f}",
            "Strategy Return": f"{self.total_return_pct:+.2f}%",
            "Benchmark (Buy & Hold)": f"{self.benchmark_return_pct:+.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{self.sortino_ratio:.2f}",
            "Max Drawdown": f"{self.max_drawdown_pct:.2f}%",
            "Win Rate": f"{self.win_rate_pct:.1f}% ({self.winning_trades}/{self.total_trades})",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Payoff Ratio": f"{self.payoff_ratio:.2f}",
            "Avg Win": f"${self.avg_win_usd:.2f}",
            "Avg Loss": f"${self.avg_loss_usd:.2f}",
            "Total Fees Paid": f"${self.total_fees_paid:.2f}",
            "Total Slippage": f"${self.total_slippage_lost:.2f}",
        }


def calculate_metrics(
    equity_curve: pd.Series,
    trades: List[Dict[str, Any]],
    initial_equity: float,
    benchmark_initial_price: float,
    benchmark_final_price: float,
    total_fees: float,
    total_slippage: float,
    risk_free_rate: float = 0.02,  # 2% annual risk-free rate
    annualization_factor: float = 365.25 * 24,  # Hourly bars default
) -> BacktestMetrics:
    """Calculate institutional-grade trading metrics."""
    final_equity = float(equity_curve.iloc[-1]) if len(equity_curve) > 0 else initial_equity
    total_return_pct = ((final_equity - initial_equity) / initial_equity) * 100.0

    # Benchmark return
    benchmark_return_pct = (
        ((benchmark_final_price - benchmark_initial_price) / benchmark_initial_price) * 100.0
        if benchmark_initial_price > 0
        else 0.0
    )

    # Returns series
    returns = equity_curve.pct_change().dropna()

    # Drawdown series
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd_pct = abs(float(drawdown.min())) * 100.0 if len(drawdown) > 0 else 0.0

    # Sharpe Ratio
    if len(returns) > 1 and returns.std() > 0:
        rf_per_period = risk_free_rate / annualization_factor
        excess_returns = returns - rf_per_period
        sharpe = float(np.sqrt(annualization_factor) * excess_returns.mean() / returns.std())
    else:
        sharpe = 0.0

    # Sortino Ratio (downside deviation only)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1 and downside_returns.std() > 0:
        downside_std = downside_returns.std()
        rf_per_period = risk_free_rate / annualization_factor
        sortino = float(np.sqrt(annualization_factor) * (returns.mean() - rf_per_period) / downside_std)
    else:
        sortino = 0.0

    # Trade statistics
    trade_pnls = [t.get("pnl_usd", 0.0) for t in trades]
    total_trades = len(trade_pnls)
    winning_trades = [p for p in trade_pnls if p > 0]
    losing_trades = [p for p in trade_pnls if p < 0]

    n_wins = len(winning_trades)
    n_losses = len(losing_trades)
    win_rate = (n_wins / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = sum(winning_trades)
    gross_loss = abs(sum(losing_trades))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)

    avg_win = (gross_profit / n_wins) if n_wins > 0 else 0.0
    avg_loss = (gross_loss / n_losses) if n_losses > 0 else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    # CAGR
    n_periods = len(equity_curve)
    n_years = n_periods / annualization_factor if annualization_factor > 0 else 1.0
    if n_years > 0 and final_equity > 0 and initial_equity > 0:
        cagr = ((final_equity / initial_equity) ** (1.0 / n_years) - 1.0) * 100.0
    else:
        cagr = 0.0

    return BacktestMetrics(
        initial_equity=initial_equity,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
        benchmark_return_pct=benchmark_return_pct,
        cagr_pct=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=max_dd_pct,
        win_rate_pct=win_rate,
        total_trades=total_trades,
        winning_trades=n_wins,
        losing_trades=n_losses,
        profit_factor=profit_factor,
        avg_win_usd=avg_win,
        avg_loss_usd=avg_loss,
        payoff_ratio=payoff_ratio,
        total_fees_paid=total_fees,
        total_slippage_lost=total_slippage,
    )
