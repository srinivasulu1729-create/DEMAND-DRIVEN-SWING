"""
indicators.py — Demand-Driven Swing & Position System
All technical indicators computed from canonical OHLCV DataFrames.
Every function is pure: input → output, no side-effects.
"""

import numpy as np
import pandas as pd
from config import (
    EMA_10, EMA_20, EMA_30W, EMA_40W,
    ATR_PERIOD, BB_PERIOD, BB_STD, RSI_PERIOD,
    VOL_AVG_PERIOD, POSITION_LOOKBACK_DAYS,
    WEEKLY_GREEN_WINDOW, SWING_LOOKBACK_DAYS
)


# ═══════════════════════════════════════════════════════════════════════════
# WEEKLY RESAMPLING
# ═══════════════════════════════════════════════════════════════════════════

def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily OHLCV to weekly (Monday-anchored).
    Expects 'date' column and canonical OHLCV columns.
    Returns weekly DataFrame with 'date' = week-start Monday.
    """
    tmp = df.set_index("date")
    weekly = tmp.resample("W-FRI").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna(subset=["Close"])
    weekly.index.name = "date"
    weekly = weekly.reset_index()
    weekly = weekly[weekly["Close"] > 0]
    return weekly


# ═══════════════════════════════════════════════════════════════════════════
# EXPONENTIAL MOVING AVERAGES
# ═══════════════════════════════════════════════════════════════════════════

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def add_daily_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA-10 and EMA-20 columns to daily df."""
    df = df.copy()
    df[f"EMA{EMA_10}"]  = ema(df["Close"], EMA_10)
    df[f"EMA{EMA_20}"]  = ema(df["Close"], EMA_20)
    return df


def add_weekly_emas(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Add 30W and 40W EMA columns to weekly df."""
    wdf = weekly_df.copy()
    wdf[f"EMA{EMA_30W}W"] = ema(wdf["Close"], EMA_30W)
    wdf[f"EMA{EMA_40W}W"] = ema(wdf["Close"], EMA_40W)
    return wdf


# ═══════════════════════════════════════════════════════════════════════════
# ATR — Average True Range
# ═══════════════════════════════════════════════════════════════════════════

def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Standard ATR over daily OHLC."""
    high = df["High"]
    low  = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def weekly_atr(weekly_df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """ATR on weekly bars."""
    return atr(weekly_df, period)


# ═══════════════════════════════════════════════════════════════════════════
# BOLLINGER BANDS
# ═══════════════════════════════════════════════════════════════════════════

def bollinger_bands(df: pd.DataFrame,
                    period: int = BB_PERIOD,
                    std_mult: float = BB_STD) -> pd.DataFrame:
    """Returns df with BB_mid, BB_upper, BB_lower, BB_width columns."""
    df = df.copy()
    mid   = df["Close"].rolling(period).mean()
    sigma = df["Close"].rolling(period).std()
    df["BB_mid"]   = mid
    df["BB_upper"] = mid + std_mult * sigma
    df["BB_lower"] = mid - std_mult * sigma
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / mid
    return df


# ═══════════════════════════════════════════════════════════════════════════
# RSI
# ═══════════════════════════════════════════════════════════════════════════

def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs     = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ═══════════════════════════════════════════════════════════════════════════
# RELATIVE STRENGTH vs INDEX
# ═══════════════════════════════════════════════════════════════════════════

def rs_ratio(stock_df: pd.DataFrame,
             index_df: pd.DataFrame,
             lookback: int = POSITION_LOOKBACK_DAYS) -> float:
    """
    RS ratio = (stock % change) / (index % change) over `lookback` days.
    Returns float; NaN if either side is zero or data insufficient.
    """
    if len(stock_df) < lookback + 1 or len(index_df) < lookback + 1:
        return float("nan")
    stock_chg = (stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[-lookback] - 1)
    index_chg = (index_df["Close"].iloc[-1] / index_df["Close"].iloc[-lookback] - 1)
    if abs(index_chg) < 1e-9:
        return float("nan")
    return stock_chg / index_chg


def rs_line(stock_df: pd.DataFrame, index_df: pd.DataFrame) -> pd.Series:
    """
    RS line = stock close / index close, normalised to start at 100.
    Both DataFrames must already be date-aligned.
    """
    raw = stock_df["Close"].values / index_df["Close"].values
    return pd.Series(raw / raw[0] * 100, index=stock_df.index, name="RS_line")


# ═══════════════════════════════════════════════════════════════════════════
# VOLUME INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

def vol_avg(df: pd.DataFrame, period: int = VOL_AVG_PERIOD) -> pd.Series:
    """Rolling mean volume over `period` days (including current bar)."""
    return df["Volume"].rolling(period, min_periods=1).mean()


def vol_ratio(df: pd.DataFrame, period: int = VOL_AVG_PERIOD) -> pd.Series:
    """Today's volume / N-day average volume."""
    avg = vol_avg(df, period)
    return df["Volume"] / avg


# ═══════════════════════════════════════════════════════════════════════════
# WEEKLY GREEN / RED CANDLE COUNT
# ═══════════════════════════════════════════════════════════════════════════

def weekly_green_count(weekly_df: pd.DataFrame, n: int = WEEKLY_GREEN_WINDOW) -> int:
    """Count green candles (close > open) in last `n` weekly bars."""
    last_n = weekly_df.tail(n)
    return int((last_n["Close"] > last_n["Open"]).sum())


def daily_green_count(df: pd.DataFrame, n: int = SWING_LOOKBACK_DAYS) -> int:
    """Count green daily candles in last `n` bars."""
    last_n = df.tail(n)
    return int((last_n["Close"] > last_n["Open"]).sum())


# ═══════════════════════════════════════════════════════════════════════════
# PRICE RANGE / BASE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def base_range(df: pd.DataFrame, lookback: int = 40) -> tuple[float, float]:
    """Return (base_low, base_high) over the last `lookback` bars."""
    w = df.tail(lookback)
    return w["Low"].min(), w["High"].max()


def price_pct_change(df: pd.DataFrame, lookback: int = POSITION_LOOKBACK_DAYS) -> float:
    """% change in close over `lookback` bars."""
    if len(df) <= lookback:
        return float("nan")
    return df["Close"].iloc[-1] / df["Close"].iloc[-lookback] - 1


def atr_contraction(weekly_df: pd.DataFrame,
                    recent_weeks: int = 8,
                    prior_weeks: int = 8) -> float:
    """
    ATR contraction ratio = 1 − (recent_avg_ATR / prior_avg_ATR).
    Positive value means contraction. Returns 0 if insufficient data.
    """
    if len(weekly_df) < recent_weeks + prior_weeks:
        return 0.0
    atr_vals = weekly_atr(weekly_df)
    recent_atr = atr_vals.iloc[-(recent_weeks):].mean()
    prior_atr  = atr_vals.iloc[-(recent_weeks + prior_weeks):-recent_weeks].mean()
    if prior_atr == 0:
        return 0.0
    return float(1 - recent_atr / prior_atr)


# ═══════════════════════════════════════════════════════════════════════════
# INDEX REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def index_regime(index_weekly: pd.DataFrame) -> str:
    """
    Returns 'bull', 'sideways', or 'bear' based on 30W and 40W EMA.
    Requires at least 40 weekly bars.
    """
    if len(index_weekly) < EMA_40W:
        return "bull"  # default if insufficient data
    wdf = add_weekly_emas(index_weekly)
    last = wdf.iloc[-1]
    close    = last["Close"]
    ema30    = last[f"EMA{EMA_30W}W"]
    ema40    = last[f"EMA{EMA_40W}W"]
    if close > ema30 and close > ema40:
        return "bull"
    elif close < ema30 and close < ema40:
        return "bear"
    else:
        return "sideways"


def risk_pct_for_regime(regime: str) -> float:
    """Return risk-per-trade % for the given regime string."""
    from config import RISK_BULL, RISK_SIDEWAYS, RISK_BEAR
    return {"bull": RISK_BULL, "sideways": RISK_SIDEWAYS, "bear": RISK_BEAR}.get(regime, RISK_BEAR)


# ═══════════════════════════════════════════════════════════════════════════
# CANDLE PATTERN HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def is_hammer(row: pd.Series, body_ratio: float = 0.35) -> bool:
    """Bullish hammer: small body, long lower shadow, little upper shadow."""
    body  = abs(row["Close"] - row["Open"])
    rng   = row["High"] - row["Low"]
    if rng == 0:
        return False
    lower_shadow = min(row["Open"], row["Close"]) - row["Low"]
    upper_shadow = row["High"] - max(row["Open"], row["Close"])
    return (body / rng < body_ratio) and (lower_shadow >= 2 * body) and (upper_shadow < body)


def is_bullish_engulfing(curr: pd.Series, prev: pd.Series) -> bool:
    """Current green candle fully engulfs previous red candle."""
    prev_red = prev["Close"] < prev["Open"]
    curr_green = curr["Close"] > curr["Open"]
    engulfs = (curr["Open"] <= prev["Close"]) and (curr["Close"] >= prev["Open"])
    return bool(prev_red and curr_green and engulfs)


def is_inside_bar(curr: pd.Series, prev: pd.Series) -> bool:
    """Current bar's high/low within prev bar's range."""
    return bool(curr["High"] <= prev["High"] and curr["Low"] >= prev["Low"])


def is_reversal_candle(df: pd.DataFrame, idx: int) -> bool:
    """
    True if bar at `idx` is a hammer, bullish engulfing vs idx-1, or inside bar vs idx-1.
    """
    if idx < 1 or idx >= len(df):
        return False
    curr = df.iloc[idx]
    prev = df.iloc[idx - 1]
    return (is_hammer(curr)
            or is_bullish_engulfing(curr, prev)
            or is_inside_bar(curr, prev))
