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
                              lookback: int = POSITION_LOOKBACK_DAYS) -> bool:
    """[1] Price moved ≥ 50% in the last 6–8 weeks (40 trading days)."""
    chg = price_pct_change(stock_df, lookback)
    return float(chg) >= 0.50 if not np.isnan(chg) else False


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


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2-B  BREAKOUT ENTRY
# ═══════════════════════════════════════════════════════════════════════════

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
    """
    if len(weekly_df) < 3:
        return False
    mother  = weekly_df.iloc[-3]   # the wide 'mother' candle
    inside  = weekly_df.iloc[-2]   # the inside (tight) candle
    current = weekly_df.iloc[-1]   # current week's close must exceed inside's high
    is_inside = (float(inside["High"]) < float(mother["High"]) and
                 float(inside["Low"])  > float(mother["Low"]))
    breakout  = float(current["Close"]) > float(inside["High"])
    return bool(is_inside and breakout)


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

    # [extra] Price >30% above daily 10-EMA — "Extended from 10 EMA" exit seen in real trades
    df_ema  = add_daily_emas(stock_df)
    ema10_d = float(df_ema[f"EMA{EMA_10}"].iloc[-1])
    current_p = float(df_ema["Close"].iloc[-1])
    pct_above = (current_p / ema10_d - 1) if ema10_d > 0 else 0.0
    result["rule_extended_10ema_daily"] = pct_above >= 0.30

    result["sell_extended"] = (result["rule29_accel_days"]
                                or result["rule30_rsi_overbought"]
                                or result["rule_extended_10ema_daily"])
    return result


def sell_on_weakness(stock_df: pd.DataFrame, index_df: pd.DataFrame,
                     entry_ema20: float | None = None) -> dict:
    """
    Check 'sell on weakness' rules [31–34].
    `entry_ema20` is the 20-EMA level at current bar (pre-computed if available).
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

    # [31] Close below 10-EMA daily — PRIMARY trailing stop (most common real exit).
    #      Standalone trigger: no secondary condition required.
    result["rule31_close_below_10ema"] = bool(current < float(ema10_level))

    # [32] Sustained lower lows: 3 consecutive lower lows below the 20-EMA
    if len(df) >= 5:
        lows = df["Low"].iloc[-4:].values
        three_lower = (lows[-1] < lows[-2] < lows[-3])
        below_ema   = float(df["Close"].iloc[-1]) < float(ema20_level)
        result["rule32_lower_low"] = bool(three_lower and below_ema)


    # [33] RS line breaks down (current RS < 10-bar ago RS — avoids daily noise)
    rs = rs_line(stock_df, index_df)
    if len(rs) >= 11:
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

    # Trail stop with daily 10-EMA for both swing and position trades.
    # The stop only ever moves UP (ratchet).  Using daily EMA means the
    # trail responds within a week of a trend change, protecting profits
    # without being too noisy.
    df   = add_daily_emas(stock_df)
    ema10 = float(df[f"EMA{EMA_10}"].iloc[-1])
    trail = ema10 * 0.99              # 1% cushion below daily 10-EMA

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

    # Base depth 8–25%
    lo, hi = base_range(stock_df, 40)
    base_depth = (hi - lo) / hi if hi > 0 else 0
    r_base = 0.08 <= base_depth <= 0.25

    r11 = rule11_volatility_contraction(stock_df)

    # ── Entry pattern (any one valid pattern is sufficient) ───────────────
    # Inside candle breakout patterns (primary real-trade setups)
    r_weekly_inside = rule_weekly_inside_candle_breakout(weekly_df)
    r_daily_inside  = rule_daily_inside_candle_near_10ema(stock_df)
    # Traditional breakout patterns
    r10 = rule10_base_bottom_entry(stock_df)
    r14 = rule14_new_high_breakout(stock_df)
    r_entry = r10 or r14 or r_weekly_inside or r_daily_inside

    # Volume confirms
    r_vol = rule15_breakout_volume(stock_df) or rule12_low_vol_red_days(stock_df)

    # Position size (regime check → non-zero size means rule passes)
    regime  = "bull" if r04 else index_regime(index_weekly)
    r_size  = regime in ("bull", "sideways")   # bear → no new position trades

    r08 = rule08_rs_line_near_highs(stock_df, index_df)

    # Confirmatory score (3 of 6 required — not all simultaneously)
    n_confirm = sum([r02, r_base, r11, r_vol, r08, r09])

    checklist = {
        "r01_price_50pct":           r01,
        "r02_green_weekly":          r02,
        "r03_rs_ratio":              r03_pass,
        "r03_rs_value":              round(rs_val, 2) if not np.isnan(rs_val) else None,
        "r04_index_regime":          r04,
        "r09_stage2":                r09,
        "r_base_depth_8_25pct":      r_base,
        "r11_vol_contracting":       r11,
        "r_weekly_inside_breakout":  r_weekly_inside,
                "r_daily_inside_near_10ema": r_daily_inside,
        "r_entry_valid":             r_entry,
        "r_volume_confirms":         r_vol,
        "r_position_sized":          r_size,
        "r08_rs_near_highs":         r08,
        # Non-negotiable gates: momentum + RS + not bear
        "non_negotiable_pass":       r01 and r03_pass and r04,
        # all_pass: non-negotiables + a valid entry pattern + 3 of 6 confirmatory
        "all_pass":                  (r01 and r03_pass and r04 and r_size
                                      and r_entry and n_confirm >= 3),
    }
    return checklist
