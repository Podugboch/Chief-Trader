"""Configuration module for Crypto Trading Bot.

Loads configuration from environment variables and provides strict typing
and safety validations.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load .env file if present
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class TradingMode(str, Enum):
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"


class ExchangeConfig(BaseModel):
    """Configuration for exchange connectivity."""
    exchange_name: str = Field(default_factory=lambda: os.getenv("EXCHANGE_NAME", "binance").lower())
    api_key: str = Field(default_factory=lambda: os.getenv("API_KEY", os.getenv("BINANCE_API_KEY", "")))
    api_secret: str = Field(default_factory=lambda: os.getenv("API_SECRET", os.getenv("BINANCE_API_SECRET", "")))
    api_passphrase: str = Field(default_factory=lambda: os.getenv("API_PASSPHRASE", ""))
    use_testnet: bool = Field(
        default_factory=lambda: os.getenv("USE_TESTNET", os.getenv("BINANCE_USE_TESTNET", "true")).lower() == "true"
    )
    # Official Binance public data and testnet URLs
    public_data_base_url: str = "https://data-api.binance.vision"
    testnet_base_url: str = "https://testnet.binance.vision"
    live_base_url: str = "https://api.binance.com"
    request_timeout_seconds: int = 10
    max_retries: int = 3
    retry_backoff_base: float = 1.5


class RiskConfig(BaseModel):
    """Rigorous risk management parameters (Discipline #3)."""
    # Max risk as fraction of total equity per individual trade (e.g. 0.015 = 1.5%)
    max_risk_per_trade_pct: float = Field(default=0.015, ge=0.001, le=0.05)
    # Maximum leverage (1.0 for spot)
    max_leverage: float = Field(default=1.0, ge=1.0, le=1.0)
    # Maximum open positions
    max_open_positions: int = Field(default=3, ge=1, le=10)
    # Maximum portfolio drawdown allowed before trigger kill switch
    max_daily_drawdown_pct: float = Field(default=0.05, ge=0.01, le=0.20)
    max_total_drawdown_pct: float = Field(default=0.15, ge=0.02, le=0.40)
    # Dynamic ATR Multipliers
    atr_stop_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    atr_take_profit_multiplier: float = Field(default=3.5, ge=1.0, le=10.0)
    # Slippage & Fees (Discipline #5)
    fee_rate: float = Field(default=0.001, ge=0.0, le=0.01)  # 0.1% maker/taker
    slippage_bps: float = Field(default=5.0, ge=0.0, le=50.0)  # 5 bps = 0.05%
    # Circuit Breakers
    volatility_circuit_breaker_atr_mult: float = Field(default=3.0, ge=1.5, le=6.0)
    circuit_breaker_cooldown_bars: int = Field(default=4, ge=1, le=24)


class BotConfig(BaseModel):
    """Global bot configuration."""
    trading_mode: TradingMode = Field(default=TradingMode.PAPER)
    default_symbol: str = Field(default="BTCUSDT")
    trading_pairs: List[str] = Field(
        default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"]
    )
    default_timeframe: str = Field(default="1h")
    initial_cash: float = Field(default=10_000.0, gt=0.0)

    # Directories
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data_cache")
    state_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "state")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "logs")
    kill_switch_flag: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "kill_switch.flag")

    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)

    @classmethod
    def load_from_env(cls) -> "BotConfig":
        """Instantiate configuration from environment variables."""
        load_dotenv(dotenv_path=ENV_PATH, override=True)
        mode_str = os.getenv("TRADING_MODE", "paper").lower()
        mode = TradingMode(mode_str) if mode_str in TradingMode._value2member_map_ else TradingMode.PAPER

        risk_cfg = RiskConfig(
            max_risk_per_trade_pct=float(os.getenv("MAX_RISK_PER_TRADE_PCT", "0.015")),
            max_leverage=float(os.getenv("MAX_ACCOUNT_LEVERAGE", "1.0")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "3")),
            max_daily_drawdown_pct=float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "0.05")),
            max_total_drawdown_pct=float(os.getenv("MAX_TOTAL_DRAWDOWN_PCT", "0.15")),
            atr_stop_multiplier=float(os.getenv("ATR_STOP_MULTIPLIER", "2.0")),
            atr_take_profit_multiplier=float(os.getenv("ATR_TAKE_PROFIT_MULTIPLIER", "3.5")),
            fee_rate=float(os.getenv("FEE_RATE", "0.001")),
            slippage_bps=float(os.getenv("SLIPPAGE_BPS", "5")),
            volatility_circuit_breaker_atr_mult=float(os.getenv("VOLATILITY_CIRCUIT_BREAKER_ATR_MULT", "3.0")),
            circuit_breaker_cooldown_bars=int(os.getenv("CIRCUIT_BREAKER_COOLDOWN_BARS", "4")),
        )

        pairs_raw = os.getenv("TRADING_PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT")
        pairs_list = [p.strip().upper() for p in pairs_raw.split(",") if p.strip()]
        if not pairs_list:
            pairs_list = ["BTCUSDT"]

        is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
        base_d = Path("/tmp") if is_serverless else Path(__file__).resolve().parent.parent
        data_d = Path("/tmp/data_cache") if is_serverless else (base_d / "data_cache")
        state_d = Path("/tmp/state") if is_serverless else (base_d / "state")
        logs_d = Path("/tmp/logs") if is_serverless else (base_d / "logs")
        flag_f = Path("/tmp/kill_switch.flag") if is_serverless else (base_d / "kill_switch.flag")

        cfg = cls(
            trading_mode=mode,
            default_symbol=os.getenv("DEFAULT_SYMBOL", pairs_list[0]),
            trading_pairs=pairs_list,
            default_timeframe=os.getenv("DEFAULT_TIMEFRAME", "1h"),
            exchange=ExchangeConfig(),
            risk=risk_cfg,
            base_dir=base_d,
            data_dir=data_d,
            state_dir=state_d,
            logs_dir=logs_d,
            kill_switch_flag=flag_f,
        )

        # Create directories if they do not exist
        try:
            cfg.data_dir.mkdir(parents=True, exist_ok=True)
            cfg.state_dir.mkdir(parents=True, exist_ok=True)
            cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        return cfg

    def validate_safety(self) -> None:
        """Enforces security discipline #8: no live trading without valid keys."""
        if self.trading_mode == TradingMode.LIVE:
            if not self.exchange.api_key or not self.exchange.api_secret:
                raise ValueError(
                    "SECURITY GUARD: Live trading requires BINANCE_API_KEY and BINANCE_API_SECRET. "
                    "Cannot start bot in LIVE mode without credentials. "
                    "Use TRADING_MODE=paper or TRADING_MODE=testnet for testing."
                )
            if self.exchange.use_testnet:
                raise ValueError("Incompatible: TRADING_MODE is live but BINANCE_USE_TESTNET is True.")
