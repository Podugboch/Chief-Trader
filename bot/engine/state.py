"""State Persistence and Recovery.

Atomically saves and loads bot runtime state (cash, crypto balances,
open positions, stop levels, high-water mark, and trade log) to disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from bot.execution.order import Position

logger = logging.getLogger("bot.engine.state")


class BotState:
    """Manages bot persistence across restarts and crashes."""

    def __init__(self, state_dir: Optional[Path] = None):
        if state_dir is None:
            self.state_dir = Path(__file__).resolve().parent.parent.parent / "state"
        else:
            self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "bot_state.json"

    def save(
        self,
        cash: float,
        balances: Dict[str, float],
        positions: Dict[str, Position],
        high_water_mark: float,
        trade_history: list,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Atomically persist state to JSON."""
        pos_dict = {}
        for sym, pos in positions.items():
            pos_dict[sym] = {
                "symbol": pos.symbol,
                "units": pos.units,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "highest_price_seen": pos.highest_price_seen,
                "entry_time": pos.entry_time.isoformat() if isinstance(pos.entry_time, datetime) else str(pos.entry_time),
                "strategy_name": pos.strategy_name,
                "break_even_activated": pos.break_even_activated,
            }

        payload = {
            "cash": cash,
            "balances": balances,
            "positions": pos_dict,
            "high_water_mark": high_water_mark,
            "trade_history": trade_history[-50:],  # retain last 50 trades in state
            "last_saved": datetime.now(timezone.utc).isoformat(),
            "meta": extra_meta or {},
        }

        # Write to temp file then rename for atomic safety
        tmp_file = self.state_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            tmp_file.replace(self.state_file)
            logger.debug("Bot state saved successfully.")
        except Exception as e:
            logger.error("Failed to save bot state: %s", e)

    def load(self) -> Optional[Dict[str, Any]]:
        """Load state from disk if exists."""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Reconstruct Position objects
            positions: Dict[str, Position] = {}
            for sym, p in data.get("positions", {}).items():
                entry_dt = datetime.now(timezone.utc)
                if "entry_time" in p:
                    try:
                        entry_dt = datetime.fromisoformat(p["entry_time"])
                    except Exception:
                        pass

                positions[sym] = Position(
                    symbol=p["symbol"],
                    units=p["units"],
                    entry_price=p["entry_price"],
                    current_price=p["current_price"],
                    stop_loss=p["stop_loss"],
                    take_profit=p["take_profit"],
                    highest_price_seen=p.get("highest_price_seen", p["entry_price"]),
                    entry_time=entry_dt,
                    strategy_name=p.get("strategy_name", "Recovered"),
                    break_even_activated=p.get("break_even_activated", False),
                )

            data["reconstructed_positions"] = positions
            logger.info("Bot state restored from %s", self.state_file)
            return data
        except Exception as e:
            logger.error("Error loading state file: %s. Starting fresh.", e)
            return None

    def clear(self) -> None:
        """Remove state file."""
        if self.state_file.exists():
            try:
                self.state_file.unlink()
                logger.info("State file cleared.")
            except Exception as e:
                logger.error("Failed to delete state file: %s", e)
