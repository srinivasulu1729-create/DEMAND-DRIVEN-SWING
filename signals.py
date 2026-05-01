"""
signals.py — Demand-Driven Swing & Position System
All entry, exit, and filter rules (Sections 1–6).
Each rule function returns True/False (or a numeric score).
Rule numbers match the specification exactly.
"""

import numpy as np
import pandas as pd

from config import (
    POSITION_LOOKBACK_DAYS, WEEKLY_GREEN_MIN, WEEKLY_GREEN_WINDOW,
    RS_MIN_RATIO, LOW_VOL_PULLBACK_RATIO, HIGH_VOL_EXPANSION_RATIO,
    ATR_CONTRACTION_MIN, BASE_BOTTOM_PCT, BREAKOUT_WEEKS,
    SWING_LOOKBACK_DAYS, SWING_GREEN_MIN, SWING_VOL_DRY,
    SELL_PROFIT_TARGET_LO, SELL_PROFIT_TARGET_HI,
    EMA10W_EXTENSION, RSI_OVERBOUGHT, ACCEL_DAYS,
    HEAVY_RED_VOL_RATIO, HEAVY_RED_CONSEC,
    POSITION_SL_PCT, SWING_SL_PCT,
    POSITION_BREAKEVEN_TRIGGER, SWING_BREAKEVEN_TRIGGER,
    EMA_10, EMA_20, VOL_AVG_PERIOD,
    MIN_GAIN_RULE26_STANDALONE,
    MIN_GAIN_RULE3233,
    R01_WATCHLIST_DAYS, R01_MIN_MOVE,
    EXTENDED_10EMA_DAILY_PCT, N_CONFIRM_REQUIRED,
    RULE31_POSITION_USE_WEEKLY, RULE31_USE_EMA20, RULE31_VOL_CONFIRM,
    BASE_MIN_WEEKS,
    BASE_DEPTH_MIN, BASE_DEPTH_MAX,
)
from indicators import (
    resample_weekly, add_daily_emas, add_weekly_emas,
    rs_ratio, rs_line, vol_ratio, vol_avg,
    weekly_green_count, daily_green_count,
    base_range, price_pct_change, atr_contraction,
    index_regime, atr, rsi, weekly_atr,
    is_reversal_candle,
)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1-A  LONG-TERM POSITION FILTER
# ═══════════════════════════════════════════════════════════════════════════

def rule01_price_moved_50pct(stock_df: pd.DataFrame,
                              lookback: int = POSITION_LOOKBACK_DAYS,
                              watchlist_window: int = R01_WATCHLIST_DAYS) -> bool:
    """
    [1] Price moved ≥ 50% in ANY 40-day window within the last R01_WATCHLIST_DAYS bars.

    Original: only checks the most recent 40-day window.
    Updated (watchlist mechanism): scans rolling 40-day sub-windows over the past
    R01_WATCHLIST_DAYS (default 90) trading days.

    Rationale: In practice, a stock's 50%-in-40-days event identifies it as a
    high-momentum candidate ("alert fires"). The system should then remain ready
    to enter for up to 90 days while the stock consolidates and sets up the next
    entry pattern (VCP bottom, EMA support, inside bar). The old 40-day-only check
    meant the window had already expired by the time the consolidation entry
    appeared, causing the system to miss second-leg continuation trades on the
    year's biggest winners (e.g. SHAKTIPUMP +532%, TARIL +380%, CUPID +333%).
    r03 (40-day RS >= 3x) still guards quality: stocks in genuine decline will
    have low RS and be filtered out regardless of r01 passing.
    """
    if len(stock_df) < lookback + 1:
        return False
    close = stock_df["Close"].values
    n = len(close)
    # Scan each bar in the last `watchlist_window` trading days
    scan_start = max(lookback, n - watchlist_window)
    for i in range(scan_start, n):
        base = close[i - lookback]
        if base > 0 and (close[i] / base - 1) >= R01_MIN_MOVE:
            return True
    return False


def rule02_green_weekly_candles(weekly_df: pd.DataFrame,
                                 n: int = WEEKLY_GREEN_WINDOW,
                                 min_green: int = WEEKLY_GREEN_MIN) -> bool:
    """[2] At least 6 out of last 8 weekly candles are green."""
    return weekly_green_count(weekly_df, n) >= min_green


def rule03_rs_ratio(stock_df: pd.DataFrame,
                    index_df: pd.DataFrame,
                    lookback: int = POSITION_LOOKBACK_DAYS) -> tuple[bool, float]:
    """[3] RS ≥ 3× NIFTYMIDSML400 over 40 days. Returns (pass, rs_value)."""
    rs = rs_ratio(stock_df, index_df, lookback)
    return (rs >= RS_MIN_RATIO if not np.isnan(rs) else False), rs


def rule04_index_regime(index_weekly: pd.DataFrame) -> bool:
    """[4] Index must be ABOVE both 30W and 40W EMA to allow position trades."""
    regime = index_regime(index_weekly)
    return regime == "bull"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1-B  DEMAND–SUPPLY FILTER
# ═══════════════════════════════════════════════════════════════════════════

def rule05_low_vol_pullbacks(stock_df: pd.DataFrame) -> bool:
    """[5] Volume on red days < 50% of 20-day avg."""
    df = stock_df.copy()
    avg = vol_avg(df)
    red_days = df[df["Close"] < df["Open"]]
    if len(red_days) == 0:
        return True
    last_red = red_days.tail(5)
    # All recent red days must have low volume
    passed = (last_red["Volume"] < LOW_VOL_PULLBACK_RATIO * avg.loc[last_red.index]).all()
    return bool(passed)


def rule06_high_vol_expansions(stock_df: pd.DataFrame) -> bool:
    """[6] Volume on green days ≥ 1.5× 20-day avg."""
    df = stock_df.copy()
    avg = vol_avg(df)
    green_days = df[df["Close"] > df["Open"]]
    if len(green_days) == 0:
        return False
    last_green = green_days.tail(5)
    passed = (last_green["Volume"] >= HIGH_VOL_EXPANSION_RATIO * avg.loc[last_green.index]).any()
    return bool(passed)


def rule07_atr_contraction(weekly_df: pd.DataFrame,
                            min_contraction: float = ATR_CONTRACTION_MIN) -> bool:
    """[7] Weekly ATR contraction ≥ 30% vs prior 8 weeks."""
    c = atr_contraction(weekly_df)
    return c >= min_contraction


def rule08_rs_line_near_highs(stock_df: pd.DataFrame, index_df: pd.DataFrame,
                               lookback: int = 20) -> bool:
    """[8] RS line at or near new highs (within 5% of its rolling max)."""
    rs = rs_line(stock_df, index_df)
    if len(rs) < lookback:
        return False
    current  = rs.iloc[-1]
    rs_max   = rs.rolling(lookback).max().iloc[-1]
    return float(current) >= 0.95 * float(rs_max)


def rule09_stage2(stock_df: pd.DataFrame, weekly_df: pd.DataFrame) -> bool:
    """[9] Price above 30W EMA AND 30W EMA is sloping up."""
    if len(weekly_df) < 32:
        return False
    wdf = add_weekly_emas(weekly_df)
    last  = wdf.iloc[-1]
    prev  = wdf.iloc[-2]
    above_ema30 = last["Close"] > last[f"EMA{30}W"]
    slope_up    = last[f"EMA{30}W"] > prev[f"EMA{30}W"]
    return bool(above_ema30 and slope_up)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2-A  BASE BOTTOM ENTRY
# ═══════════════════════════════════════════════════════════════════════════

def rule10_base_bottom_entry(stock_df: pd.DataFrame,
                              lookback: int = 40) -> bool:
    """[10] Price in lower 30% of current base range."""
    lo, hi = base_range(stock_df, lookback)
    if hi <= lo:
        return False
    current = stock_df["Close"].iloc[-1]
    pct_in_range = (current - lo) / (hi - lo)
    return float(pct_in_range) <= BASE_BOTTOM_PCT


def rule11_volatility_contraction(stock_df: pd.DataFrame) -> bool:
    """[11] BB width or daily ATR narrowing (current < 20-bar avg of ATR)."""
    df = stock_df.copy()
    atr_vals = atr(df)
    if len(atr_vals) < 21:
        return False
    current_atr = atr_vals.iloc[-1]
    avg_atr     = atr_vals.iloc[-21:-1].mean()
    return float(current_atr) < float(avg_atr)


def rule12_low_vol_red_days(stock_df: pd.DataFrame) -> bool:
    """[12] Volume on red days < 50% of 20-day avg (same as rule 5 — re-affirmed)."""
    return rule05_low_vol_pullbacks(stock_df)


def rule13_break_above_resistance(stock_df: pd.DataFrame,
                                   lookback: int = 20) -> bool:
    """[13] Price breaks and closes above minor resistance within base."""
    if len(stock_df) < lookback + 2:
        return False
    recent = stock_df.tail(lookback + 1)
    resistance = recent["High"].iloc[:-1].max()   # prior highs = resistance
    current_close = recent["Close"].iloc[-1]
    return float(current_close) > float(resistance)


def rule_base_formed(weekly_df: pd.DataFrame,
                     min_weeks: int = BASE_MIN_WEEKS) -> bool:
    """
    Base formation check: the stock must have PAUSED for at least `min_weeks`
    weeks WITHOUT making a new swing high above the previous week's high.

    Rationale (per user stock lifecycle spec):
      After a Stage 1 breakout (50%+ in 6-8 weeks), the stock MUST consolidate
      before the Stage 2 entry. A base that forms in < 2 weeks is just noise —
      we need genuine sideways/digestion action before buying.

    Implementation:
      Look at the last (min_weeks + 1) weekly bars.
      If none of the last `min_weeks` weeks' highs exceeded the high of the
      week before the base started, the base is confirmed.
      Also used as a confirmatory signal — NOT a hard gate (so some setups
      where the base is still fresh can still pass via other confirmations).
    """
    if len(weekly_df) < min_weeks + 2:
        return False
    # The "prior swing high" is the highest High in the week just before
    # the base window (bar at index -(min_weeks+1))
    prior_swing_high = float(weekly_df["High"].iloc[-(min_weeks + 1)])
    # Check that NO week in the last `min_weeks` weeks exceeded that high
    base_window = weekly_df.tail(min_weeks)
    no_new_high = (base_window["High"] <= prior_swing_high).all()
    return bool(no_new_high)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2-B  BREAKOUT ENTRY
# ═══════════════════════════════════════════════════════════════════════════

def rule_weekly_10wma_support_bounce(weekly_df: pd.DataFrame) -> bool:
    """
    10-WMA support + bounce entry — 2nd most common real setup group.

    Trades from TradeLog:
      "Weekly 10 WMA support and weekly inside candle breakout" (DIXON, IFBAGRO)
      "10 WMA support then breakout" (DIXON)
      "Near 10 WMA inside candle breakout" (IFBAGRO, HINDCOPPER)
      "EMA Support + Inside candle breakout Daily candle" (CUPID x3)
      "Weekly inside candle breakout near 10WMA support" (HINDCOPPER)

    Pattern:
      1. Price pulled back to within 3% of the weekly 10-EMA (the support level)
      2. Current or most recent week formed a bullish candle (green close, or
         the prior week was a pullback and this week is bouncing back)
      3. No new 8-week low (stock is in uptrend, just resting at the EMA)
    """
    if len(weekly_df) < 15:
        return False
    wdf    = add_weekly_emas(weekly_df)
    last   = wdf.iloc[-1]
    prev   = wdf.iloc[-2]
    ema10w = float(wdf["EMA10W"].iloc[-1])
    if ema10w <= 0:
        return False

    # Condition 1: current close is near weekly 10-EMA (within 8%)
    # WOCKPHARMA analysis: all 5 entries were 5–8% above 10-WMA.
    # The prior 3% threshold was too tight and missed every real pullback.
    close = float(last["Close"])
    near_ema10w = abs(close - ema10w) / ema10w <= 0.08

    # Condition 2: price is bouncing (current week is green, or this is
    # the first green week after 1-2 red pullback weeks)
    curr_green = close > float(last["Open"])
    prev_red   = float(prev["Close"]) < float(prev["Open"])

    # Condition 3: not making a new 8-week low (still in uptrend)
    low_8w  = float(wdf.tail(8)["Low"].min())
    not_low = float(last["Low"]) > low_8w * 0.99  # within 1% of 8-week low is not a bounce

    return bool(near_ema10w and (curr_green or prev_red) and not_low)


def rule_weekly_hammer_or_engulfing(weekly_df: pd.DataFrame) -> bool:
    """
    Weekly Hammer or Bullish Engulfing near weekly 10-EMA.

    Trades from TradeLog:
      "Weekly Hammer breakout" (GRMOVER)
      "Weekly Engulfing near 10 WMA" (GRMOVER)
      "Weekly green engulfing candle" (APOLLO, COCHINSHIP)
      "Green candle engulfing at 10 EMA support" (COCHINSHIP)
      "Inverted hammer and next candle open above" (APOLLO)

    Pattern:
      - Last week was a hammer OR the current week's green candle engulfs last week's red
      - Price is within 5% above or at the weekly 10-EMA (at a support level)
    """
    if len(weekly_df) < 13:
        return False
    wdf    = add_weekly_emas(weekly_df)
    curr   = wdf.iloc[-1]
    prev   = wdf.iloc[-2]
    ema10w = float(wdf["EMA10W"].iloc[-1])

    # Price near weekly 10-EMA (within 5% above)
    close = float(curr["Close"])
    near_support = 0.95 * ema10w <= close <= 1.05 * ema10w

    # Weekly hammer: close in upper half of range, long lower shadow
    rng = float(curr["High"]) - float(curr["Low"])
    if rng > 0:
        body_lo = min(float(curr["Open"]), float(curr["Close"]))
        body_hi = max(float(curr["Open"]), float(curr["Close"]))
        lower_shadow = body_lo - float(curr["Low"])
        body_size    = body_hi - body_lo
        is_hammer = (lower_shadow >= 2 * body_size) and (close > float(curr["Open"]))
    else:
        is_hammer = False

    # Bullish engulfing: this green candle body fully covers last red candle
    is_engulfing = (float(curr["Close"]) > float(curr["Open"])           # green
                    and float(prev["Close"]) < float(prev["Open"])       # prev red
                    and float(curr["Open"]) <= float(prev["Close"])      # opens at or below prev close
                    and float(curr["Close"]) >= float(prev["Open"]))     # closes above prev open

    return bool((is_hammer or is_engulfing) and near_support)


def rule_vcp_breakout(weekly_df: pd.DataFrame, n_weeks: int = 8) -> bool:
    """
    VCP (Volatility Contraction Pattern) breakout.

    Trades from TradeLog:
      "VCP breakout in Daily and weekly 10WMA support" (DIXON)
      "2nd VCP bottom" (ASHAPURMIN)
      "VCP breakout" (HINDCOPPER)
      "Contraction in volatility" (CUPID)

    Pattern:
      The stock makes a series of smaller and smaller price swings (coiling).
      ATR is contracting over the base period.
      Current price is breaking out near the upper end of the coil.

    Implementation:
      - Recent 3 weeks' ATR < prior 3 weeks' ATR (contraction confirmed)
      - Current close within 5% of the 8-week high (at breakout point)
      - ATR contraction ≥ 20% (meaningful tightening, not noise)
    """
    if len(weekly_df) < n_weeks + 3:
        return False
    recent   = weekly_df.tail(n_weeks)
    atr_vals = weekly_atr(recent)
    if len(atr_vals) < 6:
        return False

    recent_atr = float(atr_vals.iloc[-3:].mean())
    prior_atr  = float(atr_vals.iloc[-6:-3].mean())
    if prior_atr <= 0:
        return False

    contraction_pct = 1 - recent_atr / prior_atr
    atr_contracting = contraction_pct >= 0.20   # at least 20% ATR contraction

    # Price within 5% of the 8-week high — near the pivot/breakout point
    high_8w  = float(recent["High"].max())
    close    = float(weekly_df["Close"].iloc[-1])
    near_breakout = close >= high_8w * 0.95

    return bool(atr_contracting and near_breakout)


def rule_w_pattern_breakout(weekly_df: pd.DataFrame,
                             lookback: int = 12) -> bool:
    """
    W-pattern (Double Bottom) breakout.

    Trades from TradeLog:
      "W pattern in Daily and Weekly 10 WMA" (AXISCADES)
      "W pattern in Daily and Weekly near 10 WMA" (ASHAPURMIN)
      "Weekly W pattern breakout" (GRMOVER)

    Pattern (weekly bars):
      1. Two lows at similar price levels (second low ≥ 90% of first low)
      2. A peak between them (the middle of the W)
      3. Current price is breaking above that middle peak

    Implementation:
      - Find the lowest close in the lookback window (first bottom)
      - Find the highest close AFTER the first bottom (middle peak)
      - Find the second lowest close AFTER the middle peak (second bottom)
        that is within 15% of the first bottom (but not a new low)
      - Current close > middle peak (breakout above W neckline)
    """
    if len(weekly_df) < lookback:
        return False
    wnd    = weekly_df.tail(lookback).reset_index(drop=True)
    closes = wnd["Close"].values
    n      = len(closes)

    # Scan for the double-bottom structure
    for first_idx in range(0, n - 6):
        first_bottom = closes[first_idx]
        if first_bottom <= 0:
            continue
        # Middle peak: highest close in the window after first_idx
        mid_slice  = closes[first_idx + 1: first_idx + 7]
        if len(mid_slice) < 2:
            continue
        peak_offset = int(np.argmax(mid_slice))
        peak_idx    = first_idx + 1 + peak_offset
        middle_peak = closes[peak_idx]

        # Middle peak must be at least 5% above first bottom
        if middle_peak < first_bottom * 1.05:
            continue

        # Second bottom: after the peak, a close within 15% of first bottom
        for second_idx in range(peak_idx + 1, min(peak_idx + 6, n - 1)):
            second_bottom = closes[second_idx]
            # Second bottom within 15% of first (not lower = not new low)
            if (second_bottom >= first_bottom * 0.85
                    and second_bottom <= first_bottom * 1.10):
                # Breakout: current close > middle peak
                if closes[-1] >= middle_peak * 0.98:
                    return True
    return False


def rule_ihs_breakout(weekly_df: pd.DataFrame, lookback: int = 16) -> bool:
    """
    Inverted Head and Shoulders (IHS) breakout on weekly bars.

    Trades from TradeLog:
      "Daily Inverted head and shoulder breakout" (HUBTOWN: +39% in 48d)
      "HIS breakout weekly" (AXISCADES: +29.7% in 175d)
      "HIS breakout" (ASHAPURMIN: +19.3% in 27d)
      Avg return: +29.3% — worth detecting.

    Pattern:
      Three troughs in order: Left Shoulder (LS), Head (H), Right Shoulder (RS)
      - H must be LOWER than both LS and RS (deepest trough = the head)
      - LS and RS at similar levels (RS within 20% of LS price)
      - Two peaks between the troughs form a neckline
      - Current close BREAKS ABOVE the neckline (breakout confirmation)

    Implementation (weekly bars):
      Scan the lookback window for a 3-trough structure using weekly lows.
    """
    if len(weekly_df) < lookback:
        return False
    wnd    = weekly_df.tail(lookback).reset_index(drop=True)
    lows   = wnd["Low"].values
    closes = wnd["Close"].values
    highs  = wnd["High"].values
    n      = len(lows)

    # Find local troughs (simple: lower than both neighbors)
    troughs = [i for i in range(1, n - 1)
               if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]]

    if len(troughs) < 3:
        return False

    # Try all combinations of 3 consecutive troughs as LS, Head, RS
    for t_idx in range(len(troughs) - 2):
        ls_i = troughs[t_idx]
        hd_i = troughs[t_idx + 1]
        rs_i = troughs[t_idx + 2]

        ls_low = lows[ls_i]
        hd_low = lows[hd_i]
        rs_low = lows[rs_i]

        # Head must be the deepest trough
        if not (hd_low < ls_low and hd_low < rs_low):
            continue

        # LS and RS at similar levels (RS within 25% of LS)
        if rs_low < ls_low * 0.75 or rs_low > ls_low * 1.25:
            continue

        # Neckline = average of the peaks between LS-Head and Head-RS
        peak1 = float(highs[ls_i:hd_i + 1].max()) if hd_i > ls_i else 0
        peak2 = float(highs[hd_i:rs_i + 1].max()) if rs_i > hd_i else 0
        if peak1 <= 0 or peak2 <= 0:
            continue
        neckline = (peak1 + peak2) / 2

        # Breakout: current close above neckline
        if closes[-1] >= neckline * 0.98:
            return True

    return False


def rule14_new_high_breakout(stock_df: pd.DataFrame,
                              weeks: int = BREAKOUT_WEEKS) -> bool:
    """[14] New 8-week high OR all-time high."""
    trading_days = weeks * 5
    if len(stock_df) < trading_days:
        return False
    prev_high = stock_df["High"].iloc[-(trading_days + 1):-1].max()
    current   = stock_df["Close"].iloc[-1]
    ath        = stock_df["High"].max()
    return (float(current) > float(prev_high)) or (float(current) >= float(ath) * 0.99)


def rule15_breakout_volume(stock_df: pd.DataFrame) -> bool:
    """[15] Breakout volume ≥ 1.5× 20-day avg."""
    vr = vol_ratio(stock_df)
    return float(vr.iloc[-1]) >= HIGH_VOL_EXPANSION_RATIO


def rule16_rs_line_breakout(stock_df: pd.DataFrame, index_df: pd.DataFrame,
                             lookback: int = 40) -> bool:
    """[16] RS line breaks out before or simultaneously with price."""
    rs = rs_line(stock_df, index_df)
    if len(rs) < lookback:
        return False
    current_rs = rs.iloc[-1]
    prior_max  = rs.iloc[-(lookback + 1):-1].max()
    return float(current_rs) >= float(prior_max) * 0.99


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2-C  INSIDE CANDLE BREAKOUT ENTRIES  (primary real-trade pattern)
# ═══════════════════════════════════════════════════════════════════════════

def rule_weekly_inside_candle_breakout(weekly_df: pd.DataFrame) -> bool:
    """
    Weekly inside candle breakout — the #1 real entry pattern (12/42 trades).
    Pattern: week[-2] is an inside bar vs week[-3],
             current week[-1] Close breaks above week[-2].High.

    Note: 10-WMA proximity is NOT gated here — that would merge this pattern with
    rule_weekly_10wma_support_bounce (the pure pullback pattern). Kept separate
    so each pattern's win-rate is measured independently.
    """
    if len(weekly_df) < 3:
        return False
    mother  = weekly_df.iloc[-3]   # the wide 'mother' candle
    inside  = weekly_df.iloc[-2]   # the inside (tight) candle
    current = weekly_df.iloc[-1]   # current week's close must exceed inside's high
    is_inside = (float(inside["High"]) < float(mother["High"]) and
                 float(inside["Low"])  > float(mother["Low"]))
    # Require 1.5% buffer above inside bar's high — filters false signals from
    # partial week bars (e.g. Tuesday with only 2 days of data).
    breakout  = float(current["Close"]) > float(inside["High"]) * 1.015
    green     = float(current["Close"]) > float(current["Open"])
    return bool(is_inside and breakout and green)


def rule_weekly_prev_high_near_10wma(weekly_df: pd.DataFrame) -> bool:
    """
    Breakout of previous week's high while price is near 10-WMA.

    This is the PRIMARY WOCKPHARMA entry pattern — all 5 annotated entries were:
      "Breakout of last week candle and price near 10 WMA"
      "Engulfing the previous weekly candle"
      "10WMA support and engulfing weekly candle"

    WOCKPHARMA data confirms: entries were 5–8% above 10-WMA (NOT at new 8-week highs).
    This is a PULLBACK entry during an uptrend — stock rests near 10-WMA then resumes.

    Pattern:
      1. Current week close > previous week high  (breakout of last week's candle)
      2. Price within 10% of weekly 10-WMA        (pullback zone, not extended)
      3. Green weekly candle                       (bullish confirmation)
    """
    if len(weekly_df) < 12:
        return False
    wdf    = add_weekly_emas(weekly_df)
    last   = wdf.iloc[-1]
    prev   = wdf.iloc[-2]
    ema10w = float(wdf["EMA10W"].iloc[-1])
    if ema10w <= 0:
        return False

    close     = float(last["Close"])
    prev_high = float(prev["High"])

    breakout   = close > prev_high                          # breaks last week's candle
    near_10wma = abs(close - ema10w) / ema10w <= 0.10      # within 10% of 10-WMA
    green      = close > float(last["Open"])                # bullish weekly close

    # Not making a new 8-week low (must be in uptrend, not breaking down)
    not_low = float(last["Low"]) > float(wdf.tail(8)["Low"].min()) * 0.99

    return bool(breakout and near_10wma and green and not_low)


def rule_daily_inside_candle_near_10ema(stock_df: pd.DataFrame) -> bool:
    """
    Daily inside candle breakout near 10-DMA.
    Pattern: bar[-2] is an inside bar vs bar[-3],
             today (bar[-1]) breaks above bar[-2].High,
             and bar[-2] was within 3% of the 10-EMA.
    """
    if len(stock_df) < 4:
        return False
    df     = add_daily_emas(stock_df)
    mother = df.iloc[-3]
    inside = df.iloc[-2]
    curr   = df.iloc[-1]
    ema10  = float(df[f"EMA{EMA_10}"].iloc[-2])   # 10-EMA at inside bar
    if ema10 <= 0:
        return False
    is_inside = (float(inside["High"]) < float(mother["High"]) and
                 float(inside["Low"])  > float(mother["Low"]))
    breakout  = float(curr["Close"]) > float(inside["High"])
    near_ema  = abs(float(inside["Close"]) - ema10) / ema10 <= 0.03
    return bool(is_inside and breakout and near_ema)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2-D  SWING ENTRY
# ═══════════════════════════════════════════════════════════════════════════

def rule17_swing_rs(stock_df: pd.DataFrame, index_df: pd.DataFrame,
                    lookback: int = SWING_LOOKBACK_DAYS) -> tuple[bool, float]:
    """[17] Stock strength ≥ 3× index over last 9 trading days."""
    rs = rs_ratio(stock_df, index_df, lookback)
    return (rs >= RS_MIN_RATIO if not np.isnan(rs) else False), rs


def rule18_swing_green_candles(stock_df: pd.DataFrame,
                                n: int = SWING_LOOKBACK_DAYS,
                                min_green: int = SWING_GREEN_MIN) -> bool:
    """[18] At least 6 out of 9 recent daily candles are green."""
    return daily_green_count(stock_df, n) >= min_green


def rule19_ema_pullback(stock_df: pd.DataFrame) -> tuple[bool, str]:
    """
    [19] Price pulled back to or near 10-EMA or 20-EMA (within 2%).
    Returns (pass, which_ema).
    """
    df = add_daily_emas(stock_df)
    current = df["Close"].iloc[-1]
    ema10   = df[f"EMA{EMA_10}"].iloc[-1]
    ema20   = df[f"EMA{EMA_20}"].iloc[-1]
    near10  = abs(current - ema10) / ema10 <= 0.02
    near20  = abs(current - ema20) / ema20 <= 0.02
    if near10:
        return True, "EMA10"
    if near20:
        return True, "EMA20"
    return False, ""


def rule20_reversal_candle(stock_df: pd.DataFrame) -> bool:
    """[20] Reversal candle at EMA: hammer, bullish engulfing, or inside bar."""
    idx = len(stock_df) - 1
    return is_reversal_candle(stock_df.reset_index(drop=True), idx)


def rule21_vol_dryup(stock_df: pd.DataFrame) -> bool:
    """[21] Volume dry-up on pullback: volume < 40% of 20-day avg."""
    vr = vol_ratio(stock_df)
    return float(vr.iloc[-1]) < SWING_VOL_DRY


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3  POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════

def compute_shares(capital: float, risk_pct: float,
                   entry: float, stop: float) -> int:
    """
    Shares = (Capital × Risk%) / (Entry − Stop).
    Returns 0 if entry <= stop.
    """
    if entry <= stop or entry <= 0:
        return 0
    risk_amount = capital * risk_pct
    shares = risk_amount / (entry - stop)
    return max(0, int(shares))


def stop_loss_price(entry: float, trade_type: str = "position") -> float:
    """Initial stop-loss price."""
    pct = POSITION_SL_PCT if trade_type == "position" else SWING_SL_PCT
    return entry * (1 - pct)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4  SELL RULES
# ═══════════════════════════════════════════════════════════════════════════

def sell_on_strength(stock_df: pd.DataFrame, entry: float,
                     weekly_df: pd.DataFrame | None = None) -> dict:
    """
    Check all 'sell on strength' rules [25–28].
    Returns dict with individual rule results and overall sell signal.
    """
    result = {
        "rule25_target_hit": False,
        "rule26_wide_range_vol": False,
        "rule27_extended_10W": False,
        "rule28_ignite_bar": False,
        "sell_strength": False,
    }
    current = stock_df["Close"].iloc[-1]
    gain    = (current - entry) / entry

    # [25] +20–25% from breakout
    result["rule25_target_hit"] = gain >= SELL_PROFIT_TARGET_LO

    # [26] Wide-range weekly candle + above-avg volume
    if weekly_df is not None and len(weekly_df) >= 2:
        last_w  = weekly_df.iloc[-1]
        w_range = (last_w["High"] - last_w["Low"]) / last_w["Close"]
        avg_w_vol = weekly_df["Volume"].rolling(10).mean().iloc[-1]
        result["rule26_wide_range_vol"] = (w_range > 0.05
                                            and last_w["Volume"] > avg_w_vol)

    # [27] ≥40% above 10-week EMA
    if weekly_df is not None and len(weekly_df) >= 12:
        wdf = add_weekly_emas(weekly_df)
        ema10w = wdf.iloc[-1].get(f"EMA10W", wdf["Close"].ewm(span=10).mean().iloc[-1])
        result["rule27_extended_10W"] = float(current) >= float(ema10w) * (1 + EMA10W_EXTENSION)

    # [28] Ignite bar: high volume + large single-day move
    #      Only trigger after position has gained ≥5% (avoids selling on entry breakout itself)
    last_day = stock_df.iloc[-1]
    day_move = abs(last_day["Close"] - last_day["Open"]) / last_day["Open"]
    vr       = vol_ratio(stock_df).iloc[-1]
    result["rule28_ignite_bar"] = (day_move > 0.05 and vr > 2.0 and gain >= 0.05)

    # Rule 26 standalone is a 52% win-rate coin-flip when fired early.
    # Only allow it to trigger a sell on its own when the position is
    # already up ≥ MIN_GAIN_RULE26_STANDALONE (default 5%).  When rule27
    # or rule28 ALSO fires, rule 26 is just a confirming signal and the
    # gain-gate is not needed — those combos have 74–100% win rates.
    rule26_qualified = (
        result["rule26_wide_range_vol"] and
        (gain >= MIN_GAIN_RULE26_STANDALONE
         or result["rule27_extended_10W"]
         or result["rule28_ignite_bar"])
    )
    result["sell_strength"] = any([
        result["rule25_target_hit"],
        rule26_qualified,
        result["rule27_extended_10W"],
        result["rule28_ignite_bar"],
    ])
    return result


def sell_when_extended(stock_df: pd.DataFrame) -> dict:
    """
    Check 'sell when extended' rules [29–30].
    """
    result = {
        "rule29_accel_days": False,
        "rule30_rsi_overbought": False,
        "sell_extended": False,
    }
    # [29] 3–5 consecutive accelerating gains
    if len(stock_df) >= ACCEL_DAYS + 1:
        recent = stock_df.tail(ACCEL_DAYS + 1)
        gains  = recent["Close"].pct_change().dropna()
        accel  = all(gains.iloc[i] > gains.iloc[i - 1] for i in range(1, len(gains)))
        result["rule29_accel_days"] = accel and (gains > 0).all()

    # [30] RSI > 85
    rsi_vals = rsi(stock_df["Close"])
    if not rsi_vals.empty:
        result["rule30_rsi_overbought"] = float(rsi_vals.iloc[-1]) > RSI_OVERBOUGHT

    # [extra] Price >100% above daily 10-EMA — "Extended from 10 EMA" exit seen in real trades
    # Threshold raised from 0.30 to EXTENDED_10EMA_DAILY_PCT (default 1.00) for 10X strategy:
    # a 10X stock routinely trades 50-80% above its 10-EMA during the big move —
    # only exit at truly parabolic extension (>100% above EMA).
    df_ema  = add_daily_emas(stock_df)
    ema10_d = float(df_ema[f"EMA{EMA_10}"].iloc[-1])
    current_p = float(df_ema["Close"].iloc[-1])
    pct_above = (current_p / ema10_d - 1) if ema10_d > 0 else 0.0
    result["rule_extended_10ema_daily"] = pct_above >= EXTENDED_10EMA_DAILY_PCT

    result["sell_extended"] = (result["rule29_accel_days"]
                                or result["rule30_rsi_overbought"]
                                or result["rule_extended_10ema_daily"])
    return result


def sell_on_weakness(stock_df: pd.DataFrame, index_df: pd.DataFrame,
                     entry_ema20: float | None = None,
                     trade_type: str = "swing",
                     weekly_df: pd.DataFrame | None = None,
                     current_gain: float = 0.0) -> dict:
    """
    Check 'sell on weakness' rules [31–34].
    `entry_ema20` is the 20-EMA level at current bar (pre-computed if available).
    `trade_type`: "position" uses WEEKLY 10-EMA for rule31 (survives daily pullbacks,
                  lets winners ride to the weekly trend break).
                  "swing" uses DAILY 10-EMA (fast exit appropriate for short holds).
    `current_gain`: reserved for future hybrid logic (unused in current version).
    """
    result = {
        "rule31_close_below_10ema": False,
        "rule32_lower_low": False,
        "rule33_rs_breakdown": False,
        "rule34_heavy_red_vol": False,
        "sell_weakness": False,
    }
    df = add_daily_emas(stock_df)
    current     = df["Close"].iloc[-1]
    ema10_level = df[f"EMA{EMA_10}"].iloc[-1]
    ema20_level = df[f"EMA{EMA_20}"].iloc[-1]

    # [31] Break below 20-EMA on volume — PRIMARY weakness exit (per user spec).
    #      Strategy: "Sell on Weakness: Break below 20 EMA on volume"
    #      Using 20-EMA (not 10-EMA) reduces false exits during healthy pullbacks.
    #      Volume confirmation ensures the breach has real selling pressure behind it.
    #
    #      RULE31_USE_EMA20=True  → use daily 20-EMA (user spec)
    #      RULE31_VOL_CONFIRM=True → also require volume >= avg (confirms selling pressure)
    #
    #      Position trades with RULE31_POSITION_USE_WEEKLY=True still use weekly EMA
    #      (retained as an override option for very long holds if needed).
    if (RULE31_POSITION_USE_WEEKLY
            and trade_type == "position"
            and weekly_df is not None
            and len(weekly_df) >= 12):
        # Position trades trail on weekly 10W-EMA — lets winners ride for weeks/months.
        wdf = add_weekly_emas(weekly_df)
        weekly_close = float(wdf["Close"].iloc[-1])
        weekly_ema10 = float(wdf["EMA10W"].iloc[-1])
        result["rule31_close_below_10ema"] = weekly_close < weekly_ema10
    else:
        # Daily 10-EMA for swing trades (fast trail appropriate for short holds)
        ema_level = float(ema20_level) if RULE31_USE_EMA20 else float(ema10_level)
        below_ema = bool(current < ema_level)
        if RULE31_VOL_CONFIRM and below_ema:
            vr = vol_ratio(df).iloc[-1]
            result["rule31_close_below_10ema"] = bool(vr >= 1.0)
        else:
            result["rule31_close_below_10ema"] = below_ema

    # [32] Sustained lower lows: 3 consecutive lower lows below the 20-EMA
    # Gated by MIN_GAIN_RULE3233: below this gain the position hasn't developed
    # yet — normal consolidation looks like lower lows. Let SL handle real losers.
    if len(df) >= 5 and current_gain >= MIN_GAIN_RULE3233:
        lows = df["Low"].iloc[-4:].values
        three_lower = (lows[-1] < lows[-2] < lows[-3])
        below_ema   = float(df["Close"].iloc[-1]) < float(ema20_level)
        result["rule32_lower_low"] = bool(three_lower and below_ema)

    # [33] RS line breaks down (current RS < 10-bar ago RS — avoids daily noise)
    # Same gate: RS naturally dips during early consolidation before trend resumes.
    rs = rs_line(stock_df, index_df)
    if len(rs) >= 11 and current_gain >= MIN_GAIN_RULE3233:
        result["rule33_rs_breakdown"] = float(rs.iloc[-1]) < float(rs.iloc[-11])

    # [34] 2+ consecutive heavy red-volume candles
    if len(df) >= HEAVY_RED_CONSEC:
        tail = df.tail(HEAVY_RED_CONSEC)
        heavy_red = (
            (tail["Close"] < tail["Open"]) &
            (tail["Volume"] > HEAVY_RED_VOL_RATIO * vol_avg(df).tail(HEAVY_RED_CONSEC))
        )
        result["rule34_heavy_red_vol"] = heavy_red.all()

    # Rule 31 (10-EMA breach) is the primary trailing stop — fires alone.
    # Other signals are secondary and need 2+ to avoid noise exits.
    secondary_count = sum([
        result["rule32_lower_low"],
        result["rule33_rs_breakdown"],
        result["rule34_heavy_red_vol"],
    ])
    result["sell_weakness"] = result["rule31_close_below_10ema"] or secondary_count >= 2
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5  TRAILING STOP
# ═══════════════════════════════════════════════════════════════════════════

def update_trailing_stop(current_price: float, entry: float,
                          current_stop: float, trade_type: str,
                          stock_df: pd.DataFrame,
                          weekly_df: pd.DataFrame | None = None) -> float:
    """
    Returns the new (higher) stop-loss level.
    Position trades trail below 10-week EMA; swing trades trail below 10-day EMA.
    """
    gain = (current_price - entry) / entry

    # Move to breakeven
    be_trigger = POSITION_BREAKEVEN_TRIGGER if trade_type == "position" else SWING_BREAKEVEN_TRIGGER
    if gain >= be_trigger and current_stop < entry:
        current_stop = entry

    # Position trades: trail with WEEKLY 10-EMA (slow, survives pullbacks).
    # Swing trades:    trail with DAILY  10-EMA (fast, fits short holds).
    # Stop only ever moves UP (ratchet).
    if trade_type == "swing":
        df    = add_daily_emas(stock_df)
        ema10 = float(df[f"EMA{EMA_10}"].iloc[-1])
        trail = ema10 * 0.99           # 1% cushion below daily 10-EMA
    else:
        if weekly_df is not None and len(weekly_df) >= 12:
            wema10 = weekly_df["Close"].ewm(span=10, adjust=False).mean().iloc[-1]
            trail  = float(wema10) * 0.99   # 1% cushion below weekly 10-EMA
        else:
            trail = current_stop

    return max(current_stop, trail)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6  FULL ENTRY CHECKLIST
# ═══════════════════════════════════════════════════════════════════════════

def entry_checklist(stock_df: pd.DataFrame, index_df: pd.DataFrame,
                    weekly_df: pd.DataFrame, index_weekly: pd.DataFrame,
                    trade_type: str = "position") -> dict:
    """
    Evaluate all 10 checklist items from Section 6.
    Returns dict with each item result and 'all_pass' boolean.
    Non-negotiable items: rule01, rule03, rule04 — failing any means discard.
    """
    r01 = rule01_price_moved_50pct(stock_df)
    r02 = rule02_green_weekly_candles(weekly_df)
    r03_pass, rs_val = rule03_rs_ratio(stock_df, index_df)
    r04 = rule04_index_regime(index_weekly)
    r09 = rule09_stage2(stock_df, weekly_df)

    # Base depth 8–25% — measures RETRACEMENT from 40-day high to current close.
    # Spec: "base must retrace ≥ 8%" means the stock has pulled back 8–25% from
    # its recent swing high.  The previous (hi-lo)/hi formula measured OSCILLATION
    # (total range), not retracement — it failed stocks in tight pullbacks (<8% range)
    # even when they were 12–20% below their high, and also failed stocks with wide
    # ranges (>25%) that were actually forming valid deep bases.
    lo, hi = base_range(stock_df, 40)
    current_close = float(stock_df["Close"].iloc[-1])
    base_depth = (hi - current_close) / hi if hi > 0 else 0
    r_base = BASE_DEPTH_MIN <= base_depth <= BASE_DEPTH_MAX

    r11 = rule11_volatility_contraction(stock_df)

    # ── Base formation check (Stage 2 lifecycle) ──────────────────────────
    # Per user spec: "Base formation: pause of 2 weeks, no new swing high"
    # Must have formed a base (at least BASE_MIN_WEEKS=2 weeks no new high)
    # before entering — we do NOT buy the initial Stage 1 surge.
    r_base_formed = rule_base_formed(weekly_df)

    # ── Entry pattern (ANY ONE is sufficient for r_entry) ─────────────────
    # GROUP 1: Inside candle breakout (most common — 12+7 = 19 real trades)
    r_weekly_inside = rule_weekly_inside_candle_breakout(weekly_df)
    r_daily_inside  = False  # rule_daily_inside_candle_near_10ema(stock_df)
    # DISABLED: 22 trades in 2023 at only 36.4% WR — fires on daily noise, not clean setups.
    # Re-enable only after adding a stricter proximity / RS confirmation gate.

    # GROUP 2: 10-WMA support + bounce (8 real trades: DIXON, IFBAGRO, HINDCOPPER, CUPID)
    r_10wma_bounce  = rule_weekly_10wma_support_bounce(weekly_df)

    # GROUP 2b: Primary WOCKPHARMA pattern — prev week high breakout near 10-WMA
    # All 5 WOCKPHARMA entries were this pattern (5–8% above 10-WMA, close > prev week high).
    # Wider proximity (10%) vs r_10wma_bounce (8%) catches slightly more extended pullbacks.
    r_prev_high_near_10wma = rule_weekly_prev_high_near_10wma(weekly_df)

    # GROUP 3: Weekly Hammer or Engulfing near 10-WMA (5 real trades)
    r_wkly_hammer_engulf = rule_weekly_hammer_or_engulfing(weekly_df)

    # GROUP 4: VCP breakout (3 real trades: DIXON, ASHAPURMIN, HINDCOPPER, CUPID)
    r_vcp = rule_vcp_breakout(weekly_df)

    # GROUP 5: W-pattern — DISABLED.
    # Algorithm found 24 trades (vs 3 real) with 38% win rate — over-triggers on any
    # two-trough consolidation.  Keep function for future calibration but disable in entry.
    r_w_pattern = False  # rule_w_pattern_breakout(weekly_df)

    # GROUP 6: IHS breakout — DISABLED.
    # Algorithm found 31 trades (vs 3 real) with 32% win rate — detects random 3-trough
    # structures.  Keep function for future calibration but disable in entry.
    r_ihs = False  # rule_ihs_breakout(weekly_df)

    # GROUP 7: Traditional breakout patterns
    r10 = rule10_base_bottom_entry(stock_df)
    r14 = rule14_new_high_breakout(stock_df)
    # r14 standalone (new 8-week high) enters at the TOP of every expansion leg —
    # WOCKPHARMA analysis shows this produces stop-outs far from 10-WMA support.
    # Keep r14 only as a CONFIRMATORY signal; pullback rules above handle actual entries.
    r14_vol_confirmed = r14 and rule15_breakout_volume(stock_df)

    # Any valid entry pattern is sufficient
    # NOTE: r14_vol_confirmed removed from r_entry — it fires at leg tops (wrong timing).
    # The r_prev_high_near_10wma + r_10wma_bounce + r_weekly_inside rules cover the same
    # breakout moment but only when price is correctly positioned near 10-WMA support.
    r_entry = (r10
               or r_weekly_inside or r_daily_inside
               or r_10wma_bounce or r_prev_high_near_10wma
               or r_wkly_hammer_engulf
               or r_vcp or r_w_pattern or r_ihs)

    # Volume confirms (for n_confirm score)
    r_vol = rule15_breakout_volume(stock_df) or rule12_low_vol_red_days(stock_df)

    # Position size (regime check → non-zero size means rule passes)
    regime  = "bull" if r04 else index_regime(index_weekly)
    r_size  = regime in ("bull", "sideways")   # bear → no new position trades

    r08 = rule08_rs_line_near_highs(stock_df, index_df)

    # ── Confirmatory score ──────────────────────────────────────────────────
    # 7 confirmatory signals (one added: r_base_formed)
    # N_CONFIRM_REQUIRED of 7 must pass.
    # r_base_formed: validates that a Stage 2 base has genuinely formed (2+ weeks
    # no new swing high) — prevents chasing the initial Stage 1 surge directly.
    n_confirm = sum([r02, r_base, r11, r_vol, r08, r09, r_base_formed])

    checklist = {
        "r01_price_50pct":           r01,
        "r02_green_weekly":          r02,
        "r03_rs_ratio":              r03_pass,
        "r03_rs_value":              round(rs_val, 2) if not np.isnan(rs_val) else None,
        "r04_index_regime":          r04,
        "r09_stage2":                r09,
        "r_base_depth_8_25pct":      r_base,
        "r11_vol_contracting":       r11,
        "r_base_formed_2wk":         r_base_formed,
        "r_weekly_inside_breakout":  r_weekly_inside,
        "r_daily_inside_near_10ema": r_daily_inside,
        "r_10wma_support_bounce":    r_10wma_bounce,
        "r_prev_high_near_10wma":    r_prev_high_near_10wma,
        "r_weekly_hammer_engulfing": r_wkly_hammer_engulf,
        "r_vcp_breakout":            r_vcp,
        "r_w_pattern":               r_w_pattern,
        "r_ihs_breakout":            r_ihs,
        "r14_vol_confirmed":         r14_vol_confirmed,
        "r_entry_valid":             r_entry,
        "r_volume_confirms":         r_vol,
        "r_position_sized":          r_size,
        "r08_rs_near_highs":         r08,
        "n_confirm_score":           n_confirm,
        # Non-negotiable gates: Stage 1 momentum (RS >=3x) + bull index
        "non_negotiable_pass":       r01 and r03_pass and r04,
        # all_pass: non-negotiables + valid entry + N_CONFIRM_REQUIRED of 7 confirmatory
        # (7 signals: r02, r_base, r11, r_vol, r08, r09, r_base_formed)
        "all_pass":                  (r01 and r03_pass and r04 and r_size
                                      and r_entry and n_confirm >= N_CONFIRM_REQUIRED),
    }
    return checklist
