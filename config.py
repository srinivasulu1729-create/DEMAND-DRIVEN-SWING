"""
config.py — Demand-Driven Swing & Position System
Central configuration. Change values here only.

STRATEGY: Trade Stage 2 base formations after a stock's initial 50%+ move.
           The stock lifecycle: 12w expansion → base → 12w expansion → base
           → 8w expansion → base → 3–6w expansion.
           Enter at base BOTTOM or base BREAKOUT, ride expansion legs.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = r"C:\Z_Srinu\WorkingDat\StocksDatabase\NSE Data\Day\market_data"
INDEX_SYMBOL = "NIFTYMIDSMALLCAP400"
SYMBOLS_CSV = os.path.join(os.path.dirname(__file__), "symbols.csv")
TRADELOG_XLSX = os.path.join(os.path.dirname(__file__), "TradeLog.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ── Capital & costs ────────────────────────────────────────────────────────
STARTING_CAPITAL = 1_000_000          # Rs 10,00,000
SLIPPAGE_PCT = 0.001                  # 0.1% per side
FLAT_BROKERAGE = 20                   # Rs 20 flat per trade
PCT_BROKERAGE = 0.0003                # 0.03% per trade (lower of two)

# ── Risk regime ────────────────────────────────────────────────────────────
# Per user spec: Bull 1%, Sideways 0.5%, Bear 0.25%
RISK_BULL = 0.010                     # 1.0% of capital
RISK_SIDEWAYS = 0.005                 # 0.5%
RISK_BEAR = 0.0025                    # 0.25%

# ── Universe filters ───────────────────────────────────────────────────────
MIN_PRICE = 50.0                      # Exclude penny stocks below Rs 50
MIN_AVG_DAILY_VOL = 50_000            # Exclude thinly traded stocks
MIN_TRADING_ROWS = 200                # skip if fewer rows

# ── Indicator periods ──────────────────────────────────────────────────────
EMA_10 = 10
EMA_20 = 20
EMA_30W = 30                          # 30-week EMA
EMA_40W = 40                          # 40-week EMA
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2
RSI_PERIOD = 14
VOL_AVG_PERIOD = 20                   # volume 20-day avg

# ── Stage 1 / Watchlist gates ─────────────────────────────────────────────
# Stage 1 = the initial strong breakout that puts stock on watchlist.
# We do NOT enter during Stage 1. We wait for the base (Stage 2) to form.
R01_MIN_MOVE = 0.43                   # Stage 1 alert threshold: 43%+ in 40 days
                                      # (was 0.50 — lowered to capture ARIHANT, BIRLA CABLE,
                                      #  ASIAN ENERGY and other stocks with 43-49% alerts)
POSITION_LOOKBACK_DAYS = 40           # ~6-8 weeks in trading days
R01_WATCHLIST_DAYS = 750              # Watchlist window: stock qualifies if 43%+ in any
                                      # 40-day window within last 750 trading days (~3 yrs).
                                      # Real trade log shows entries up to 766 CALENDAR days
                                      # (≈550 trading days) after the alert event:
                                      #   COCHINSHIP: alerted Oct 2022, traded Nov 2024 (766d)
                                      #   63MOONS:    alerted Jul 2023, traded Nov 2024 (507d)
                                      #   CUPID:      alerted Sep 2023, traded Jul 2025 (665d)
                                      # Once Stage 1 confirms a stock's DNA as a strong mover,
                                      # it stays eligible through its full multi-year lifecycle.
                                      # RS ≥ 3× + Stage-2 check naturally filter declining stocks.
WEEKLY_GREEN_WINDOW = 8               # last N weekly candles for green-candle check
WEEKLY_GREEN_MIN = 6                  # at least 6 must be green (Stage 1 confirmation)
RS_MIN_RATIO = 3.0                    # RS >= 3x NIFTYMIDSML400. Tested 4x: caused complete
                                      # Feb-Apr 2023 drought (0 trades, 86 days flat) which
                                      # dropped CAGR from 71.88% -> 51.19%. Reverted to 3x.

# ── Base formation parameters ─────────────────────────────────────────────
BASE_MIN_WEEKS = 2                    # base must pause at least 2 weeks (no new swing high)
                                      # prevents entering immediately after Stage 1 surge
BASE_DEPTH_MIN = 0.08                 # base must retrace ≥ 8%
BASE_DEPTH_MAX = 0.25                 # base must retrace ≤ 25% (deeper = failed breakout)
BASE_RS_RELATIVE_MULT = 2.5           # base depth ≤ 2.5× index depth (relative tightness)

# ── Entry rules ───────────────────────────────────────────────────────────
BREAKOUT_WEEKS = 8                    # new 8-week high for breakout entry
LOW_VOL_PULLBACK_RATIO = 0.50         # vol < 50% of 20d avg (low-vol decline in base)
HIGH_VOL_EXPANSION_RATIO = 1.50       # vol >= 1.5x 20d avg (breakout volume)
ATR_CONTRACTION_MIN = 0.30            # >=30% ATR contraction (volatility compression)
BASE_BOTTOM_PCT = 0.30                # lower 30% of base range = base bottom entry zone
SWING_LOOKBACK_DAYS = 9               # swing RS lookback (9 days)
SWING_GREEN_MIN = 6                   # 6 green candles out of 9 for swing
SWING_VOL_DRY = 0.40                  # vol < 40% 20d avg = volume dry-up

# ── Swing trades toggle ────────────────────────────────────────────────────
ENABLE_SWING_TRADES = False           # DISABLED: 31/33 swing trades hit StopLoss (24% WR)
                                      # Swing entries fire on short-term bounces that reverse
                                      # immediately. Position-only system is 53% WR vs 24%.
                                      # Re-enable only after fixing swing entry quality.

# ── Add-on rules ──────────────────────────────────────────────────────────
ADDON_MIN_GAIN_PCT = 0.05             # add-on only after 4-6% gain (use 5% midpoint)
ADDON_MAX_SIZE_RATIO = 0.50           # <=50% of original shares

# ── Sell rules ────────────────────────────────────────────────────────────
# Per user spec:
# Strength:  +20-25% from breakout | wide-range weekly + big volume | 40-50% above 10W EMA
# Extended:  parabolic 3-5 day rally | RSI > 75-80
# Weakness:  break below 20 EMA on volume | lower low | RS breakdown | heavy red vol candles
SELL_PROFIT_TARGET_LO = 0.80          # +80% from entry → sell (backstop only)
                                      # [was 0.20 — WRONG: the 20% target was cutting 40/60
                                      #  real trades (67%) short. Real exit = 10-EMA break.
                                      #  At 80%: only 7/60 real trades cut early (12%).
                                      #  10-EMA trailing stop handles all normal exits.]
SELL_PROFIT_TARGET_HI = 1.20          # +120% → sell (extreme backstop)
EMA10W_EXTENSION = 0.45               # >=45% above 10W EMA → extended (midpoint 40-50%)
RSI_OVERBOUGHT = 88                   # RSI > 88 → extreme overbought backstop
                                      # (was 78 — fired after 1-2 days in momentum stocks;
                                      #  real trades never use RSI as exit. 88 = safety net only.)
ACCEL_DAYS = 4                        # 3-5 consecutive acceleration days → parabolic
HEAVY_RED_VOL_RATIO = 1.50            # >=1.5x avg volume for red candle (rule 34)
HEAVY_RED_CONSEC = 2                  # 2+ consecutive heavy red candles

# ── Rule-31 weakness exit ──────────────────────────────────────────────────
# Real TradeLog analysis (60 trades): 28/60 exits = "price close below 10 EMA in daily"
# This is the PRIMARY exit — standalone, NO volume condition needed.
# The "break below 20 EMA on volume" in the spec description is secondary context;
# actual trade exits are clean 10-EMA daily closes.
RULE31_USE_EMA20 = False              # False = 10-EMA (matches real 28/60 exit pattern)
RULE31_VOL_CONFIRM = False            # False = standalone close below EMA (no volume gate)
RULE31_POSITION_USE_WEEKLY = True     # True = weekly 10W-EMA for position trades
                                      # (daily 10-EMA is too tight — TITAGARH exited at +0.65%
                                      #  instead of riding the full +300% 2023 move. Position
                                      #  trades are designed to hold weeks/months, not days.)

# ── Extended exit threshold (daily 10-EMA extension) ──────────────────────
EXTENDED_10EMA_DAILY_PCT = 0.30       # price >30% above daily 10-EMA → parabolic exit
                                      # (per original spec — fires on truly extended moves)

# ── Stop-loss ─────────────────────────────────────────────────────────────
# Per user spec: Position 7-10% (use 8%); Swing 3-4% (use 3.5%)
POSITION_SL_PCT = 0.08                # 8% below entry (midpoint of 7-10%)
SWING_SL_PCT = 0.035                  # 3.5% below entry (midpoint of 3-4%)
POSITION_BREAKEVEN_TRIGGER = 0.12     # move to BE after 1.5R = 12% gain (8% SL × 1.5)
                                      # (was 0.04 = 0.5R — too tight, stopped out at BE
                                      #  immediately on minor pullbacks before trade developed)
SWING_BREAKEVEN_TRIGGER = 0.05        # move to BE after 1.5R ≈ 5% gain (3.5% SL × 1.5)

# ── Position sizing ────────────────────────────────────────────────────────
# With 1% risk and 8% SL: position = 1%/8% = 12.5% of capital per trade
# With 8 positions: ~100% deployed in a strong bull market
MAX_POSITION_PCT = 0.20               # max 20% of capital per position
                                      # (2x safety buffer over the 12.5% calculated size)
MAX_POSITION_ABS = 2_000_000          # Rs 20 lakh hard cap per position
MAX_POSITIONS_HARD_CAP = 8            # max 8 simultaneous positions

# ── Hold-period guards ─────────────────────────────────────────────────────
MIN_GAIN_STRENGTH_EXIT = 0.15         # strength/extended exits require ≥15% gain first
                                      # Prevents Rule26/27/30 firing on entry candle itself.
                                      # Real trades: RSI never primary exit; 10-EMA is primary.
MAX_HOLD_TRADING_DAYS = 150           # force-close after 150 trading days (~7 months)
                                      # Real trade log: CUPID held 192 calendar days ≈ 137
                                      # trading days (+396%). 150 covers this with buffer.
                                      # Most trades close much earlier via 10-EMA exit.
MIN_HOLD_DAYS_WEAKNESS = 15           # minimum 15 days before weakness exits can fire
                                      # raised from 10: a Stage 2 base typically needs
                                      # 2-3 weeks to develop momentum after entry

# Rule 26 (wide-range weekly candle + volume) fires standalone only after this gain
MIN_GAIN_RULE26_STANDALONE = 0.15     # 15% gain required for rule26 standalone trigger
                                      # (was 0.05 — too easy to fire on entry week breakout candle)

# Rule 32/33 minimum gain gate
MIN_GAIN_RULE3233 = 0.05              # rule32 (lower lows) and rule33 (RS breakdown) only fire
                                      # after a 5% gain. Below this threshold they fire on normal
                                      # post-entry consolidation — 24/30 of all rule32/33 exits
                                      # had gains < 5% (avg +0.16%). The SL handles true losers;
                                      # rule32/33 should only prune deteriorating winners.

# ── Entry quality gate ─────────────────────────────────────────────────────
N_CONFIRM_REQUIRED = 3                # 3 of 7 confirmatory signals required for entry
                                      # Confirmatory: r02, r_base, r11, r_vol, r08, r09, r_base_formed
                                      # Tested N=4: reduced trades from 144→103, dropped CAGR from
                                      # 71.88%→59.49%. Audit shows 17 of 44 valid ENTRY signals for
                                      # known winners (ANANTRAJ, ZENTEC etc.) blocked at n_confirm=3.
                                      # The low Jan WR was cold-start (slot flood), not confirm quality.
                                      # Fix cold-start via warm-up (--start 2022-11-01), not N_CONFIRM.
                                      # At 4/7, only well-formed Stage 2 bases enter.

# ── Pass criteria ─────────────────────────────────────────────────────────
TARGET_CAGR = 5.00                    # 500% CAGR target
TARGET_WIN_RATE = 0.70                # 70% win rate target
MIN_TRADES = 30                       # minimum trades per year
