# 🚀 Chief Trader - Autonomous Multi-Pair Crypto Trading System

A robust, modular, and disciplined cryptocurrency trading bot built in Python. Designed around the **9 core disciplines of successful trading bots**: clear testable strategy logic, rigorous backtesting & forward-testing, risk-management first, low-latency execution, fee/slippage awareness, fault-tolerant monitoring & kill switches, adaptability without overfitting, strict security, and continuous market analysis.

---

## 📑 Core Disciplines Architecture

| Discipline | Implementation Component | File Link |
| :--- | :--- | :--- |
| **1. Clear, Testable Logic** | Modular Strategy Engine (`BaseStrategy`, `TrendMomentumStrategy`, `MeanReversionStrategy`, `RegimeAdaptiveStrategy`) | [`bot/strategies/`](file:///home/recoding/crypto_trading_bot/bot/strategies/) |
| **2. Rigorous Testing** | Event-Driven Backtesting Engine (no lookahead bias) + Paper Trading Simulator | [`bot/backtesting/engine.py`](file:///home/recoding/crypto_trading_bot/bot/backtesting/engine.py) |
| **3. Risk Management First** | ATR Volatility-based Position Sizing, Dynamic Trailing Stops, Portfolio Exposure Caps, Multi-Tier Circuit Breakers | [`bot/risk/`](file:///home/recoding/crypto_trading_bot/bot/risk/) |
| **4. Low-Latency Execution** | Resilient API Client with HMAC-SHA256 authentication, Binance Spot Testnet & Live routing | [`bot/execution/binance_client.py`](file:///home/recoding/crypto_trading_bot/bot/execution/binance_client.py) |
| **5. Fee & Slippage Awareness** | Explicit Maker/Taker Fee Modeling (0.10%) and dynamic/basis-point slippage deduction on every fill | [`bot/execution/paper_exchange.py`](file:///home/recoding/crypto_trading_bot/bot/execution/paper_exchange.py) |
| **6. Error Handling & Monitoring** | Exponential backoff retry, rate limit weight tracking, atomic state persistence, file/signal kill switches | [`bot/engine/kill_switch.py`](file:///home/recoding/crypto_trading_bot/bot/engine/kill_switch.py), [`bot/engine/state.py`](file:///home/recoding/crypto_trading_bot/bot/engine/state.py) |
| **7. Adaptability Without Overfitting** | Objective Market Regime Classifier (`STRONG_BULL`, `STRONG_BEAR`, `RANGING`, `HIGH_VOLATILITY_CHOP`) | [`bot/analysis/regime.py`](file:///home/recoding/crypto_trading_bot/bot/analysis/regime.py) |
| **8. Security** | Strict `.env` credential management, Trade-only API policy (no withdrawals), default safe `paper` mode | [`bot/config.py`](file:///home/recoding/crypto_trading_bot/bot/config.py) |
| **9. Top-Notch Market Analysis** | Multi-factor technical indicator suite: EMA Ribbon, RSI, MACD, Bollinger Bands, ATR, ADX, Supertrend, OBV, S/R Pivots | [`bot/analysis/market_analyzer.py`](file:///home/recoding/crypto_trading_bot/bot/analysis/market_analyzer.py) |

---

## 🛠️ Project Structure

```
crypto_trading_bot/
├── .env.example               # Template environment configuration with safety limits
├── .gitignore                 # Excludes secrets, caches, logs, and state
├── README.md                  # Comprehensive user and developer guide
├── pyproject.toml             # Pytest & package configuration
├── requirements.txt           # Python dependencies (requests, pandas, numpy, rich, etc.)
├── bot/
│   ├── config.py              # Strongly-typed configuration & security validations
│   ├── analysis/
│   │   ├── regime.py          # Market Regime Classifier (Trend strength & Volatility z-score)
│   │   └── market_analyzer.py # Multi-factor technical synthesis & composite scoring
│   ├── indicators/
│   │   └── technical.py       # Vectorized pure-Python EMA, SMA, RSI, MACD, BB, ATR, ADX, Supertrend, OBV
│   ├── strategies/
│   │   ├── base.py            # BaseStrategy interface & TradeSignal model
│   │   ├── trend_momentum.py  # EMA Ribbon + MACD + Volume expansion strategy
│   │   ├── mean_reversion.py  # Bollinger Bands + RSI turnaround range strategy
│   │   └── regime_adaptive.py # Meta-strategy dynamically routing between trend & range models
│   ├── risk/
│   │   ├── position_sizer.py  # ATR dollar-budgeted position sizer with exposure caps
│   │   ├── stop_manager.py    # Trailing stops, dynamic ATR stops, break-even ratcheting
│   │   └── circuit_breaker.py # Daily loss limits, peak drawdown stop, volatility pause
│   ├── execution/
│   │   ├── order.py           # Order, Position, OrderStatus, and OrderSide dataclasses
│   │   ├── binance_client.py  # Binance Spot REST client (HMAC auth, testnet & live)
│   │   └── paper_exchange.py  # High-fidelity paper exchange with fees & slippage
│   ├── backtesting/
│   │   ├── engine.py          # Event-driven bar-by-bar backtest simulator
│   │   └── metrics.py         # Sharpe, Sortino, Drawdown, Win Rate, Profit Factor, Fees
│   ├── engine/
│   │   ├── trader.py          # Trading orchestration engine & event loop
│   │   ├── state.py           # Atomic JSON state persistence across restarts
│   │   └── kill_switch.py     # File-based and OS signal emergency kill switch
│   ├── ui/
│   │   └── dashboard.py       # Live Rich terminal UI dashboard
│   └── cli.py                 # Command line interface
└── tests/
    ├── test_indicators.py     # Unit tests for technical indicators
    ├── test_regime.py         # Unit tests for market regime classification
    ├── test_risk_manager.py   # Unit tests for position sizer, stops, and circuit breakers
    ├── test_paper_exchange.py # Unit tests for paper execution, fees, and slippage
    ├── test_strategies.py     # Unit tests for strategy signal generation
    ├── test_backtester.py     # Unit tests for backtesting simulation
    └── test_kill_switch_and_state.py # Unit tests for persistence and emergency stops
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup
The project uses Python 3.10+ (tested with Python 3.12).

```bash
cd /home/recoding/crypto_trading_bot
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
By default, the bot runs in `paper` mode (100% simulated, zero capital risk). To configure API keys for Binance Testnet:

```ini
TRADING_MODE=paper
BINANCE_API_KEY=your_testnet_key_here
BINANCE_API_SECRET=your_testnet_secret_here
BINANCE_USE_TESTNET=true
DEFAULT_SYMBOL=BTCUSDT
DEFAULT_TIMEFRAME=1h
```

---

## 🌐 Interactive Web Dashboard (Localhost)

Launch the modern real-time Web Dashboard in your browser at `http://localhost:8000`:

```bash
cd /home/recoding/crypto_trading_bot
.venv/bin/python -m bot.cli web --port 8000
```

Features included in the Web UI:
- **Live Portfolio & PnL Metrics**: Real-time equity, liquid cash, unrealized/realized PnL, circuit breaker status.
- **Market Intelligence Dial**: Real-time BTC mark price, Market Regime badge, Composite Technical Score (-100 to +100), and indicator matrix (RSI, MACD, ADX, ATR, Supertrend, Bollinger %B).
- **Interactive Backtester**: Select symbol, timeframe, strategy, and lookback period to generate instant performance metrics and interactive Chart.js equity curves.
- **Position & Trade Management**: View open positions with trailing stop markers, view past executed orders, and close positions.
- **One-Click Controls**: Start Bot, Pause, Evaluate Now, and Emergency Kill Switch.

---

## 💻 CLI Commands & Usage

### 1. Market Analysis (`analyze`)
Runs institutional-grade multi-factor market analysis, printing the market regime, composite technical score, indicator matrix, and support/resistance levels:

```bash
python -m bot.cli analyze --symbol BTCUSDT --timeframe 1h
```

Output includes:
- **Market Regime**: `STRONG_BULL`, `STRONG_BEAR`, `RANGING`, or `HIGH_VOLATILITY_CHOP`.
- **Composite Score**: Normalized from `-100.0` (extreme bear) to `+100.0` (extreme bull).
- **Verdict**: `STRONG_BUY`, `BUY`, `NEUTRAL`, `SELL`, or `STRONG_SELL`.
- **Key Indicators**: RSI(14), MACD Histogram, ADX(14), ATR(14), EMA Ribbon, Bollinger %B, Supertrend.
- **Pivot Levels**: Fibonacci & Classical R2, R1, Pivot, S1, S2, and 20-bar swing extremes.

### 2. Backtesting (`backtest`)
Validates strategies against deep historical data without lookahead bias:

```bash
# Backtest Regime-Adaptive strategy on BTCUSDT over 60 days
python -m bot.cli backtest --symbol BTCUSDT --timeframe 1h --days 60 --strategy adaptive

# Backtest Trend-Momentum strategy on ETHUSDT over 30 days
python -m bot.cli backtest --symbol ETHUSDT --timeframe 1h --days 30 --strategy trend
```

Metrics reported:
- **Strategy Return vs. Benchmark (Buy & Hold)**
- **Sharpe Ratio & Sortino Ratio** (annualized)
- **Maximum Drawdown %**
- **Win Rate % & Total Trades**
- **Profit Factor & Payoff Ratio (Avg Win / Avg Loss)**
- **Total Fees Paid & Total Slippage Lost**

### 3. Live / Paper Trading (`run`)
Launches the real-time bot with an interactive Rich terminal dashboard:

```bash
# Run paper trading loop (refreshes every 15 seconds)
python -m bot.cli run --symbol BTCUSDT --timeframe 1h --strategy adaptive --interval 15

# Run a single evaluation step and exit
python -m bot.cli run --symbol BTCUSDT --timeframe 1h --once
```

### 4. Status Inspection (`status`)
Inspects the saved state file (cash balance, open positions, unrealized PnL, and trade history):

```bash
python -m bot.cli status
```

### 5. Emergency Kill Switch (`kill` & `arm`)
Immediately halts the bot and halts new order entries:

```bash
# Trigger kill switch with reason
python -m bot.cli kill --reason "Emergency market halt"

# Clear kill switch and re-arm the bot
python -m bot.cli arm
```

Alternatively, you can trigger the kill switch at any time by creating the flag file:
```bash
touch kill_switch.flag
```

---

## 🛡️ Risk Management Parameters

All risk parameters are configurable via `.env` or in [`bot/config.py`](file:///home/recoding/crypto_trading_bot/bot/config.py):

- **`MAX_RISK_PER_TRADE_PCT`** (default: `0.015` = 1.5%): Maximum dollars risked per trade based on distance to ATR stop loss.
- **`MAX_ACCOUNT_LEVERAGE`** (default: `1.0`): Spot trading constraint (no leverage).
- **`MAX_OPEN_POSITIONS`** (default: `3`): Caps total simultaneous asset exposure.
- **`MAX_DAILY_DRAWDOWN_PCT`** (default: `0.05` = 5.0%): Pauses trading for the remainder of the UTC day if breached.
- **`MAX_TOTAL_DRAWDOWN_PCT`** (default: `0.15` = 15.0%): Circuit breaker halts the bot if drawdown from peak equity exceeds 15%.
- **`ATR_STOP_MULTIPLIER`** (default: `2.0`): Dynamic stop-loss distance.
- **`ATR_TAKE_PROFIT_MULTIPLIER`** (default: `3.5`): Profit target, maintaining a favorable risk-to-reward ratio (> 1.75:1).
- **`VOLATILITY_CIRCUIT_BREAKER_ATR_MULT`** (default: `3.0`): Temporary trading pause during sudden volatility clusters.

---

## 🧪 Test Suite

Run the full automated test suite:

```bash
cd /home/recoding/crypto_trading_bot
.venv/bin/pytest -v
```

All 24 unit and integration tests validate:
- Indicator calculations across synthetic and real market data.
- Regime classification logic under trending, consolidating, and abnormal regimes.
- Position sizing constraints, single-asset exposure caps, and available cash checks.
- Dynamic trailing stops and break-even ratcheting.
- Circuit breaker trip mechanisms for daily and total portfolio drawdowns.
- Realistic paper execution including fees and slippage.
- Kill switch triggers and atomic state restoration.
