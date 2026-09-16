"""Technical Analysis Indicators.

Vectorized, pure-Python / pandas technical indicator library designed for
high-performance backtesting and real-time market analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's Exponential Moving Average
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    # Replace division by zero / edge cases
    return rsi_val.fillna(50.0)


def macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD Line, Signal Line, and Histogram.

    Returns:
        (macd_line, signal_line, histogram)
    """
    fast_ema = ema(series, fast_period)
    slow_ema = ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Returns:
        (upper_band, middle_band, lower_band, bandwidth, percent_b)
    """
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    percent_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, middle, lower, bandwidth, percent_b


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate True Range (TR)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR) using Wilder's smoothing."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Average Directional Index (ADX), +DI, and -DI.

    Returns:
        (adx_line, plus_di, minus_di)
    """
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=high.index)
    minus_dm_s = pd.Series(minus_dm, index=low.index)

    tr = true_range(high, low, close)
    atr_val = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * (
        plus_dm_s.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        / atr_val.replace(0, np.nan)
    )
    minus_di = 100.0 * (
        minus_dm_s.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        / atr_val.replace(0, np.nan)
    )

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_line = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return adx_line.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume (OBV)."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum()


def donchian_channel(
    high: pd.Series, low: pd.Series, period: int = 20
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Donchian Channels (Upper, Lower, Middle)."""
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle = (upper + lower) / 2.0
    return upper, lower, middle


def supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Supertrend Indicator.

    Returns:
        (trend_direction, supertrend_line)
        trend_direction is 1 for Bullish, -1 for Bearish.
    """
    atr_val = atr(high, low, close, period=period)
    hl2 = (high + low) / 2.0

    basic_upper = hl2 + (multiplier * atr_val)
    basic_lower = hl2 - (multiplier * atr_val)

    final_upper = pd.Series(index=close.index, dtype=float)
    final_lower = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=int)

    n = len(close)
    if n == 0:
        return trend, final_upper

    basic_upper = basic_upper.bfill()
    basic_lower = basic_lower.bfill()

    final_upper.iloc[0] = basic_upper.iloc[0]
    final_lower.iloc[0] = basic_lower.iloc[0]
    trend.iloc[0] = 1

    for i in range(1, n):
        # Upper band adjustment
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # Lower band adjustment
        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        # Trend direction determination
        if trend.iloc[i - 1] == 1:
            if close.iloc[i] < final_lower.iloc[i]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = 1
        else:
            if close.iloc[i] > final_upper.iloc[i]:
                trend.iloc[i] = 1
            else:
                trend.iloc[i] = -1

    st_line = pd.Series(
        np.where(trend == 1, final_lower, final_upper),
        index=close.index,
    )
    return trend, st_line


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Volume-Weighted Average Price (VWAP)."""
    typical_price = (high + low + close) / 3.0
    cum_pv = (typical_price * volume).cumsum()
    cum_v = volume.cumsum().replace(0, np.nan)
    return (cum_pv / cum_v).ffill()


def rolling_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 50
) -> pd.Series:
    """Calculate Rolling Volume-Weighted Average Price."""
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    roll_pv = pv.rolling(window=window, min_periods=1).sum()
    roll_v = volume.rolling(window=window, min_periods=1).sum().replace(0, np.nan)
    return (roll_pv / roll_v).ffill()


def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    slowing: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator (%K and %D lines).

    Returns:
        (fast_k, slow_d)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    raw_k = 100.0 * ((close - lowest_low) / denom)
    fast_k = raw_k.rolling(window=slowing).mean().fillna(50.0)
    slow_d = fast_k.rolling(window=d_period).mean().fillna(50.0)
    return fast_k, slow_d


@dataclass
class FibonacciLevels:
    swing_high: float
    swing_low: float
    fib_0: float        # 0.0% (Swing High)
    fib_236: float      # 23.6% retracement
    fib_382: float      # 38.2% retracement
    fib_500: float      # 50.0% retracement
    fib_618: float      # 61.8% retracement (Golden Pocket)
    fib_786: float      # 78.6% retracement
    fib_100: float      # 100.0% (Swing Low)
    nearest_level_name: str
    nearest_level_price: float
    distance_pct: float
    is_near_golden_pocket: bool  # within 1.5% of 0.618 or 0.500


def fibonacci_retracement(
    high: pd.Series, low: pd.Series, current_price: float, lookback: int = 50
) -> FibonacciLevels:
    """Calculate key Fibonacci Retracement levels from recent swing high & swing low."""
    s_high = float(high.tail(lookback).max())
    s_low = float(low.tail(lookback).min())
    diff = max(s_high - s_low, 1e-6)

    f0 = s_high
    f236 = s_high - (0.236 * diff)
    f382 = s_high - (0.382 * diff)
    f500 = s_high - (0.500 * diff)
    f618 = s_high - (0.618 * diff)
    f786 = s_high - (0.786 * diff)
    f100 = s_low

    levels = [
        ("Fib 0.0 (High)", f0),
        ("Fib 0.236", f236),
        ("Fib 0.382", f382),
        ("Fib 0.500 (Mid)", f500),
        ("Fib 0.618 (Golden Pocket)", f618),
        ("Fib 0.786", f786),
        ("Fib 1.0 (Low)", f100),
    ]

    nearest_name, nearest_price = min(
        levels, key=lambda x: abs(current_price - x[1])
    )
    dist_pct = abs(current_price - nearest_price) / max(current_price, 1e-6) * 100.0

    near_golden_pocket = (
        abs(current_price - f618) / max(current_price, 1e-6) <= 0.015
        or abs(current_price - f500) / max(current_price, 1e-6) <= 0.012
    )

    return FibonacciLevels(
        swing_high=s_high,
        swing_low=s_low,
        fib_0=f0,
        fib_236=f236,
        fib_382=f382,
        fib_500=f500,
        fib_618=f618,
        fib_786=f786,
        fib_100=f100,
        nearest_level_name=nearest_name,
        nearest_level_price=nearest_price,
        distance_pct=dist_pct,
        is_near_golden_pocket=bool(near_golden_pocket),
    )


def detect_divergence(
    price: pd.Series, indicator: pd.Series, lookback: int = 30
) -> Tuple[bool, bool]:
    """Detect regular Bullish and Bearish Divergence between price and an oscillator.

    - Bullish Divergence: Price forms lower low, oscillator forms higher low (exhaustion of sellers).
    - Bearish Divergence: Price forms higher high, oscillator forms lower high (exhaustion of buyers).

    Returns:
        (is_bullish_divergence, is_bearish_divergence)
    """
    if len(price) < lookback:
        return False, False

    p_sub = price.iloc[-lookback:].values
    ind_sub = indicator.iloc[-lookback:].values

    # Find local minima
    lows = []
    for i in range(2, len(p_sub) - 2):
        if p_sub[i] <= p_sub[i - 1] and p_sub[i] <= p_sub[i - 2] and p_sub[i] <= p_sub[i + 1] and p_sub[i] <= p_sub[i + 2]:
            lows.append(i)

    bullish_div = False
    if len(lows) >= 2:
        prev_low_idx, curr_low_idx = lows[-2], lows[-1]
        if len(p_sub) - curr_low_idx <= 12:
            p1, p2 = p_sub[prev_low_idx], p_sub[curr_low_idx]
            i1, i2 = ind_sub[prev_low_idx], ind_sub[curr_low_idx]
            if p2 < p1 * 0.998 and i2 > i1:
                bullish_div = True

    # Find local maxima
    highs = []
    for i in range(2, len(p_sub) - 2):
        if p_sub[i] >= p_sub[i - 1] and p_sub[i] >= p_sub[i - 2] and p_sub[i] >= p_sub[i + 1] and p_sub[i] >= p_sub[i + 2]:
            highs.append(i)

    bearish_div = False
    if len(highs) >= 2:
        prev_high_idx, curr_high_idx = highs[-2], highs[-1]
        if len(p_sub) - curr_high_idx <= 12:
            p1, p2 = p_sub[prev_high_idx], p_sub[curr_high_idx]
            i1, i2 = ind_sub[prev_high_idx], ind_sub[curr_high_idx]
            if p2 > p1 * 1.002 and i2 < i1:
                bearish_div = True

    return bullish_div, bearish_div


@dataclass
class DowMarketStructure:
    trend: str  # "UPTREND", "DOWNTREND", "NEUTRAL"
    structure: str  # "HH_HL", "LH_LL", "TRANSITIONING"
    last_swing_high: float
    last_swing_low: float
    is_break_of_structure: bool  # Bullish BOS / CHoCH: price broke above previous lower high
    summary: str


def detect_dow_market_structure(
    high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 40
) -> DowMarketStructure:
    """Classifies market structure using Dow Theory principles."""
    if len(close) < 20:
        return DowMarketStructure("NEUTRAL", "TRANSITIONING", 0.0, 0.0, False, "Insufficient data")

    h_vals = high.iloc[-lookback:].values
    l_vals = low.iloc[-lookback:].values
    c_curr = float(close.iloc[-1])

    swing_highs = []
    swing_lows = []
    for i in range(2, len(h_vals) - 2):
        if h_vals[i] > h_vals[i - 1] and h_vals[i] > h_vals[i - 2] and h_vals[i] > h_vals[i + 1] and h_vals[i] > h_vals[i + 2]:
            swing_highs.append((i, h_vals[i]))
        if l_vals[i] < l_vals[i - 1] and l_vals[i] < l_vals[i - 2] and l_vals[i] < l_vals[i + 1] and l_vals[i] < l_vals[i + 2]:
            swing_lows.append((i, l_vals[i]))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        mid_idx = len(h_vals) // 2
        first_h, second_h = np.max(h_vals[:mid_idx]), np.max(h_vals[mid_idx:])
        first_l, second_l = np.min(l_vals[:mid_idx]), np.min(l_vals[mid_idx:])
        if second_h > first_h and second_l > first_l:
            return DowMarketStructure("UPTREND", "HH_HL", float(second_h), float(second_l), False, "Dow Theory: Higher Highs & Higher Lows (Bullish).")
        elif second_h < first_h and second_l < first_l:
            return DowMarketStructure("DOWNTREND", "LH_LL", float(second_h), float(second_l), False, "Dow Theory: Lower Highs & Lower Lows (Bearish).")
        return DowMarketStructure("NEUTRAL", "TRANSITIONING", float(second_h), float(second_l), False, "Dow Theory: Transitional consolidation.")

    sh1, sh2 = swing_highs[-2][1], swing_highs[-1][1]
    sl1, sl2 = swing_lows[-2][1], swing_lows[-1][1]

    is_bos = (sh2 < sh1) and (c_curr > sh2)

    if sh2 > sh1 and sl2 > sl1:
        trend = "UPTREND"
        struct = "HH_HL"
        summary = "Dow Theory: Bullish structure with Higher Highs & Higher Lows."
    elif sh2 < sh1 and sl2 < sl1:
        trend = "DOWNTREND"
        struct = "LH_LL"
        if is_bos:
            summary = "Dow Theory: Bearish structure with active Bullish Break of Structure (BOS) above last lower high."
        else:
            summary = "Dow Theory: Bearish structure with Lower Highs & Lower Lows. Dips are dangerous to buy."
    else:
        trend = "NEUTRAL"
        struct = "TRANSITIONING"
        summary = "Dow Theory: Transitional consolidation / mixed structure."

    return DowMarketStructure(
        trend=trend,
        structure=struct,
        last_swing_high=float(sh2),
        last_swing_low=float(sl2),
        is_break_of_structure=bool(is_bos),
        summary=summary,
    )


@dataclass
class WyckoffAnalysis:
    phase: str  # "ACCUMULATION", "MARKUP", "DISTRIBUTION", "MARKDOWN"
    has_spring: bool  # False breakdown below support quickly reclaimed with volume
    is_accumulation: bool
    is_safe_to_buy: bool
    summary: str


def detect_wyckoff_phase(
    close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, lookback: int = 50
) -> WyckoffAnalysis:
    """Classifies Wyckoff market phase (Accumulation, Markup, Distribution, Markdown)."""
    if len(close) < lookback:
        return WyckoffAnalysis("ACCUMULATION", False, True, True, "Insufficient data for Wyckoff analysis")

    c = close.iloc[-lookback:]
    h = high.iloc[-lookback:]
    l = low.iloc[-lookback:]
    v = volume.iloc[-lookback:]

    curr_price = float(c.iloc[-1])
    recent_range_low = float(l.iloc[:-3].min())

    min_last_3 = float(l.iloc[-3:].min())
    has_spring = (min_last_3 < recent_range_low) and (curr_price > recent_range_low)

    price_change_pct = (curr_price - float(c.iloc[0])) / float(c.iloc[0]) * 100.0
    vol_mean = float(v.mean()) if float(v.mean()) > 0 else 1.0
    curr_vol = float(v.iloc[-1])

    sma20 = float(c.rolling(20).mean().iloc[-1])
    sma50 = float(c.rolling(min(50, len(c))).mean().iloc[-1])

    if price_change_pct > 6.0 and curr_price > sma20 > sma50:
        phase = "MARKUP"
        safe_to_buy = True
        summary = "Wyckoff: MARKUP Phase — Aggressive institutional demand pushing price upward."
    elif price_change_pct < -6.0 and curr_price < sma20 < sma50:
        if has_spring:
            phase = "ACCUMULATION"
            safe_to_buy = True
            summary = "Wyckoff: ACCUMULATION (Spring detected! False breakdown reclaimed with absorption)."
        else:
            phase = "MARKDOWN"
            safe_to_buy = False
            summary = "Wyckoff: MARKDOWN Phase — Institutional distribution in progress. Avoid buying."
    else:
        if has_spring:
            phase = "ACCUMULATION"
            safe_to_buy = True
            summary = "Wyckoff: ACCUMULATION with active Spring pattern (Prime low-risk entry)."
        elif curr_price < sma20 and curr_vol > 1.5 * vol_mean:
            phase = "DISTRIBUTION"
            safe_to_buy = False
            summary = "Wyckoff: DISTRIBUTION — Heavy selling volume overwhelming buyers."
        else:
            phase = "ACCUMULATION"
            safe_to_buy = True
            summary = "Wyckoff: Neutral Consolidation / Accumulation range."

    return WyckoffAnalysis(
        phase=phase,
        has_spring=bool(has_spring),
        is_accumulation=bool(phase == "ACCUMULATION"),
        is_safe_to_buy=bool(safe_to_buy),
        summary=summary,
    )


@dataclass
class ChartPatternResult:
    pattern_name: str  # "DOUBLE_BOTTOM", "INVERSE_HEAD_AND_SHOULDERS", "ASCENDING_TRIANGLE", "DESCENDING_TRIANGLE", "NONE"
    is_bullish: bool
    is_bearish: bool
    confidence: float
    summary: str


def detect_chart_patterns(
    high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 35
) -> ChartPatternResult:
    """Detects classic reversal and continuation chart patterns (Double Bottom, Inverse H&S, Triangles)."""
    if len(close) < lookback:
        return ChartPatternResult("NONE", False, False, 0.0, "Insufficient bars for pattern recognition")

    h = high.iloc[-lookback:].values
    l = low.iloc[-lookback:].values
    c = close.iloc[-lookback:].values

    swing_lows = []
    for i in range(2, len(l) - 2):
        if l[i] <= l[i-1] and l[i] <= l[i-2] and l[i] <= l[i+1] and l[i] <= l[i+2]:
            swing_lows.append((i, l[i]))

    swing_highs = []
    for i in range(2, len(h) - 2):
        if h[i] >= h[i-1] and h[i] >= h[i-2] and h[i] >= h[i+1] and h[i] >= h[i+2]:
            swing_highs.append((i, h[i]))

    # 1. Double Bottom (W Pattern)
    if len(swing_lows) >= 2:
        idx1, low1 = swing_lows[-2]
        idx2, low2 = swing_lows[-1]
        if idx2 - idx1 >= 4 and abs(low1 - low2) / low1 <= 0.018:
            intermediate_highs = [h[k] for k in range(idx1, idx2)]
            if intermediate_highs and max(intermediate_highs) > max(low1, low2) * 1.01:
                neckline = max(intermediate_highs)
                curr_c = c[-1]
                if curr_c >= low2 and (curr_c <= neckline * 1.025):
                    return ChartPatternResult(
                        pattern_name="DOUBLE_BOTTOM",
                        is_bullish=True,
                        is_bearish=False,
                        confidence=0.85,
                        summary=f"Double Bottom (W-Pattern) confirmed: Troughs at ${low1:,.2f} & ${low2:,.2f}, Neckline at ${neckline:,.2f}.",
                    )

    # 2. Inverse Head and Shoulders
    if len(swing_lows) >= 3:
        idx_ls, ls = swing_lows[-3]
        idx_h, head = swing_lows[-2]
        idx_rs, rs = swing_lows[-1]
        if head < ls * 0.99 and head < rs * 0.99 and abs(ls - rs) / ls <= 0.03:
            return ChartPatternResult(
                pattern_name="INVERSE_HEAD_AND_SHOULDERS",
                is_bullish=True,
                is_bearish=False,
                confidence=0.90,
                summary=f"Inverse Head & Shoulders confirmed: Left Shoulder (${ls:,.2f}), Head (${head:,.2f}), Right Shoulder (${rs:,.2f}).",
            )

    # 3. Triangles
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        sh1, sh2 = swing_highs[-2][1], swing_highs[-1][1]
        sl1, sl2 = swing_lows[-2][1], swing_lows[-1][1]
        if abs(sh1 - sh2) / sh1 <= 0.012 and sl2 > sl1 * 1.008:
            return ChartPatternResult(
                pattern_name="ASCENDING_TRIANGLE",
                is_bullish=True,
                is_bearish=False,
                confidence=0.80,
                summary=f"Ascending Triangle: Flat resistance at ${sh2:,.2f} with rising swing lows (${sl1:,.2f} -> ${sl2:,.2f}).",
            )
        elif abs(sl1 - sl2) / sl1 <= 0.012 and sh2 < sh1 * 0.992:
            return ChartPatternResult(
                pattern_name="DESCENDING_TRIANGLE",
                is_bullish=False,
                is_bearish=True,
                confidence=0.80,
                summary=f"Descending Triangle: Flat support at ${sl2:,.2f} with declining swing highs (${sh1:,.2f} -> ${sh2:,.2f}).",
            )

    return ChartPatternResult("NONE", False, False, 0.0, "No dominant classical chart pattern formed.")


@dataclass
class ElliottWaveContext:
    phase: str  # "IMPULSE_BULL", "IMPULSE_BEAR", "CORRECTIVE_ABC", "NEUTRAL"
    wave_estimate: str
    is_safe_for_long: bool
    summary: str


def detect_elliott_wave_context(
    close: pd.Series, high: pd.Series, low: pd.Series, lookback: int = 50
) -> ElliottWaveContext:
    """Estimates Elliott Wave state based on price velocity, multi-period momentum, and trend length."""
    if len(close) < lookback:
        return ElliottWaveContext("NEUTRAL", "Unknown", True, "Insufficient data for wave analysis")

    c = close.iloc[-lookback:]
    ret_50 = (float(c.iloc[-1]) - float(c.iloc[0])) / float(c.iloc[0]) * 100.0
    ret_15 = (float(c.iloc[-1]) - float(c.iloc[-15])) / float(c.iloc[-15]) * 100.0

    if ret_50 > 8.0 and ret_15 > 3.0:
        return ElliottWaveContext(
            phase="IMPULSE_BULL",
            wave_estimate="Wave 3 / 5 Impulsive Bull",
            is_safe_for_long=True,
            summary="Elliott Wave: Strong Bullish Impulse wave in progress. Favor trend following.",
        )
    elif ret_50 < -8.0 and ret_15 < -3.0:
        return ElliottWaveContext(
            phase="IMPULSE_BEAR",
            wave_estimate="Wave 3 / Wave C Bearish Impulse",
            is_safe_for_long=False,
            summary="Elliott Wave: Impulsive downward Wave 3 or Wave C crash. High risk: Do NOT buy dips.",
        )
    elif ret_50 < -5.0 and ret_15 > 0.0:
        return ElliottWaveContext(
            phase="CORRECTIVE_ABC",
            wave_estimate="Wave B / C Consolidation",
            is_safe_for_long=False,
            summary="Elliott Wave: Counter-trend corrective consolidation inside bear cycle.",
        )
    return ElliottWaveContext(
        phase="NEUTRAL",
        wave_estimate="Wave 1 / 4 Consolidation",
        is_safe_for_long=True,
        summary="Elliott Wave: Structural consolidation / equilibrium.",
    )
