"""Local Data Storage and Caching.

Caches historical OHLCV data to disk to allow ultra-fast offline
backtesting and avoid unnecessary repeated exchange API calls.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import pandas as pd


class DataStorage:
    """Handles saving and loading OHLCV data from disk."""

    def __init__(self, cache_dir: Optional[Path] = None):
        is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
        if cache_dir is None:
            if is_serverless:
                self.cache_dir = Path("/tmp/data_cache")
            else:
                self.cache_dir = Path(__file__).resolve().parent.parent.parent / "data_cache"
        else:
            self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Fallback to /tmp in serverless/read-only filesystems
            self.cache_dir = Path("/tmp/data_cache")
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        clean_symbol = symbol.upper().replace("/", "")
        return self.cache_dir / f"{clean_symbol}_{timeframe}.parquet"

    def _get_csv_path(self, symbol: str, timeframe: str) -> Path:
        clean_symbol = symbol.upper().replace("/", "")
        return self.cache_dir / f"{clean_symbol}_{timeframe}.csv"

    def save(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Save DataFrame to CSV and/or Parquet."""
        try:
            csv_path = self._get_csv_path(symbol, timeframe)
            # Ensure timestamp is preserved
            df.to_csv(csv_path, index=True)
        except Exception:
            pass

    def load(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache if available."""
        csv_path = self._get_csv_path(symbol, timeframe)
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col=0)
                # Ensure numeric types
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
            except Exception:
                return None
        return None

    def exists(self, symbol: str, timeframe: str) -> bool:
        return self._get_csv_path(symbol, timeframe).exists()
