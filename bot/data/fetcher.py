"""Market Data Fetcher.

Fetches real-time and historical cryptocurrency market data from Binance
public data endpoints with automatic pagination, rate-limiting respect,
and exponential backoff retry.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
import requests

from bot.data.storage import DataStorage

logger = logging.getLogger("bot.data.fetcher")


class DataFetcher:
    """Fetches market data from public exchange endpoints."""

    def __init__(
        self,
        base_url: str = "https://data-api.binance.vision",
        storage: Optional[DataStorage] = None,
        max_retries: int = 3,
        timeout: int = 10,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "CryptoTradingBot/1.0", "Accept": "application/json"}
        )
        self.storage = storage or DataStorage()
        self.max_retries = max_retries
        self.timeout = timeout

    def _request_with_retry(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Executes GET request with exponential backoff on network errors."""
        url = f"{self.base_url}{endpoint}"
        last_err = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                # Check for rate limit warnings
                used_weight = resp.headers.get("x-mbx-used-weight-1m")
                if used_weight and int(used_weight) > 1000:
                    logger.warning("Exchange rate limit weight high: %s/1200", used_weight)
                    time.sleep(1.0)

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", "5"))
                    logger.warning("Rate limited (429). Sleeping for %d seconds", retry_after)
                    time.sleep(retry_after)
                    continue

                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as err:
                last_err = err
                wait_time = 1.5**attempt
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s. Retrying in %.1fs",
                    url,
                    attempt,
                    self.max_retries,
                    err,
                    wait_time,
                )
                time.sleep(wait_time)

        raise RuntimeError(f"Failed to fetch data from {url} after {self.max_retries} retries: {last_err}")

    def fetch_ticker_price(self, symbol: str) -> float:
        """Fetch real-time ticker price for a symbol."""
        clean_symbol = symbol.upper().replace("/", "")
        data = self._request_with_retry("/api/v3/ticker/price", {"symbol": clean_symbol})
        return float(data["price"])

    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Fetch top order book depth."""
        clean_symbol = symbol.upper().replace("/", "")
        data = self._request_with_retry("/api/v3/depth", {"symbol": clean_symbol, "limit": limit})
        return {
            "bids": [(float(p), float(q)) for p, q in data.get("bids", [])],
            "asks": [(float(p), float(q)) for p, q in data.get("asks", [])],
        }

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch single batch of OHLCV candles (up to 1000 bars)."""
        clean_symbol = symbol.upper().replace("/", "")
        params: Dict[str, Any] = {
            "symbol": clean_symbol,
            "interval": timeframe,
            "limit": min(limit, 1000),
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)

        raw_data = self._request_with_retry("/api/v3/klines", params)
        if not raw_data:
            return pd.DataFrame()

        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        df = pd.DataFrame(raw_data, columns=columns)

        # Type conversion
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume", "trades"]]

    def fetch_historical_range(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 60,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch deep historical dataset across multiple pages for backtesting."""
        if use_cache and self.storage.exists(symbol, timeframe):
            cached_df = self.storage.load(symbol, timeframe)
            if cached_df is not None and len(cached_df) >= (days * 12):  # reasonable check
                logger.info("Loaded %d bars for %s (%s) from disk cache", len(cached_df), symbol, timeframe)
                return cached_df

        clean_symbol = symbol.upper().replace("/", "")
        now_dt = datetime.now(timezone.utc)
        start_dt = now_dt - timedelta(days=days)
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(now_dt.timestamp() * 1000)

        all_dfs: List[pd.DataFrame] = []
        current_start = start_ts

        logger.info("Fetching %d days of historical %s data for %s...", days, timeframe, clean_symbol)

        while current_start < end_ts:
            batch_df = self.fetch_klines(
                clean_symbol,
                timeframe=timeframe,
                limit=1000,
                start_time=current_start,
                end_time=end_ts,
            )
            if batch_df.empty:
                break

            all_dfs.append(batch_df)
            last_ts = int(batch_df.index[-1].timestamp() * 1000)
            if last_ts <= current_start:
                break
            current_start = last_ts + 1
            time.sleep(0.1)  # small throttle

        if not all_dfs:
            raise RuntimeError(f"Could not fetch historical data for {clean_symbol}")

        combined_df = pd.concat(all_dfs)
        # Drop duplicates by index
        combined_df = combined_df[~combined_df.index.duplicated(keep="first")].sort_index()

        if use_cache:
            self.storage.save(combined_df, symbol, timeframe)
            logger.info("Cached %d bars to disk for %s (%s)", len(combined_df), symbol, timeframe)

        return combined_df
