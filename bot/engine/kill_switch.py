"""Kill Switch and Emergency Shutdown Handler.

Provides instantaneous emergency stops via file flag, OS signal handlers,
or programmatic triggers to protect capital during anomalous events.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
import sys
from typing import Callable, Optional, Any

logger = logging.getLogger("bot.engine.kill_switch")


class KillSwitch:
    """Monitors emergency abort conditions."""

    def __init__(self, flag_path: Optional[Path] = None):
        if flag_path is None:
            self.flag_path = Path(__file__).resolve().parent.parent.parent / "kill_switch.flag"
        else:
            self.flag_path = Path(flag_path)

        self._shutdown_requested = False
        self._emergency_reason = ""
        self._on_shutdown_callback: Optional[Callable[[str], None]] = None

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def register_callback(self, callback: Callable[[str], None]) -> None:
        """Register callback invoked when kill switch fires."""
        self._on_shutdown_callback = callback

    def _signal_handler(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.warning("Received termination signal: %s", sig_name)
        self.trigger(f"OS Signal {sig_name} received")

    def trigger(self, reason: str = "Manual emergency shutdown triggered") -> None:
        """Trigger immediate bot shutdown."""
        self._shutdown_requested = True
        self._emergency_reason = reason
        logger.critical("!!! EMERGENCY KILL SWITCH TRIGGERED: %s !!!", reason)

        # Create flag file on disk
        try:
            self.flag_path.write_text(f"TRIGGERED: {reason}\n")
        except Exception as e:
            logger.error("Failed to write kill switch flag file: %s", e)

        if self._on_shutdown_callback:
            try:
                self._on_shutdown_callback(reason)
            except Exception as e:
                logger.error("Error in shutdown callback: %s", e)

    def is_triggered(self) -> bool:
        """Check if shutdown was requested via signal, method, or disk flag file."""
        if self._shutdown_requested:
            return True

        if self.flag_path.exists():
            reason = "Flag file detected on disk"
            try:
                content = self.flag_path.read_text().strip()
                if content:
                    reason = content
            except Exception:
                pass
            self.trigger(reason)
            return True

        return False

    def reset(self) -> None:
        """Reset kill switch and remove disk flag file."""
        self._shutdown_requested = False
        self._emergency_reason = ""
        if self.flag_path.exists():
            try:
                self.flag_path.unlink()
                logger.info("Kill switch flag file removed. Bot armed.")
            except Exception as e:
                logger.error("Could not remove flag file: %s", e)

    @property
    def reason(self) -> str:
        return self._emergency_reason
