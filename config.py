"""
config.py — Demand-Driven Swing & Position System
Central configuration. Change values here only.
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
RISK_BULL = 0.020                     # 2.0% of capital
RISK_SIDEWAYS = 0.005                 # 0.5%
RISK_BEAR = 0.0025                    # 0.25%

# ── Universe filters ───────────────────────────────────────────────────────
MIN_PRICE = 50.0                      # Exclude penny stocks below Rs 50
MIN_AVG_DAILY_VOL = 50_000            # Exclude thinly traded stocks
MIN_TRADING_ROWS = 200                # [V5] skip if fewer rows

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

# ── Position-trade filter windows ─────────────────────────────────────────
POSITION_LOOKBACK_DAYS = 40           # ~6-8 weeks in trading days
WEEKLY_GREEN_WINDOW = 8               # last N weekly candles
WEEKLY_GREEN_MIN = 6                  # at least 6 must be green
RS_MIN_RATIO = 4.0                    # RS >= 4x index

# ── Entry rules ───────────────────────────────────────────────────────────
BREAKOUT_WEEKS = 8                    # new 8-week high for breakout
LOW_VOL_PULLBACK_RATIO = 0.50         # vol < 50% of 20d avg (rules 5, 12)
HIGH_VOL_EXPANSION_RATIO = 1.50       # vol >= 1.5x 20d avg (rules 6, 15)
ATR_CONTRACTION_MIN = 0.30            # >=30% ATR contraction (rule 7)
BASE_BOTTOM_PCT = 0.30                # lower 30% of base (rule 10)
SWING_LOOKBACK_DAYS = 9               # rule 17/18
SWING_GREEN_MIN = 6                   # rule 18
SWING_VOL_DRY = 0.40                  # rule 21: vol < 40% 20d avg

# ── Add-on rules ──────────────────────────────────────────────────────────
ADDON_MIN_GAIN_PCT = 0.04             # +4% min before add-on
ADDON_MAX_SIZE_RATIO = 0.50           # <=50% of original shares

# ── Sell rules ────────────────────────────────────────────────────────────
SELL_PROFIT_TARGET_LO = 0.18          # 3R target (3x stop) -> sell
SELL_PROFIT_TARGET_HI = 0.24          # 4R target -> sell
EMA10W_EXTENSION = 0.55               # >=55% above 10W EMA (rule 27)
RSI_OVERBOUGHT = 85                   # rule 30
ACCEL_DAYS = 3                        # consecutive acceleration days (rule 29)
HEAVY_RED_VOL_RATIO = 1.50            # >=1.5x avg (rule 34)
HEAVY_RED_CONSEC = 2                  # 2+ consecutive (rule 34)

# ── Stop-loss ─────────────────────────────────────────────────────────────
POSITION_SL_PCT = 0.06                # 6% below entry (position, weekly close basis)
SWING_SL_PCT = 0.035                  # 3.5% below entry (swing, daily close basis)
POSITION_BREAKEVEN_TRIGGER = 0.03     # move to BE after +3% (position)
SWING_BREAKEVEN_TRIGGER = 0.04        # move to BE after +4% (swing)

# ── Position sizing cap ────────────────────────────────────────────────────
MAX_POSITION_PCT = 0.25               # max 25% of capital per position
MAX_POSITION_ABS = 2_000_000          # Rs 20 lakh hard cap per position
MAX_POSITIONS_HARD_CAP = 20           # never more than 20 simultaneous

# ── Hold-period guards
MIN_GAIN_STRENGTH_EXIT = 0.18         # block rule29/30 until +18% (3R)
MAX_HOLD_TRADING_DAYS = 20            # force-close after 20 trading days
# Weakness exits (rule31/32/33/34) are blocked for this many calendar days
# after entry. 5 cal-days ~= 3-4 trading days -- lets the setup develop
# before a routine EMA re-test kills a perfectly good position.
MIN_HOLD_DAYS_WEAKNESS = 7

# Rule 26 (wide-range weekly candle + volume) fires as a standalone sell-
# strength signal only AFTER the position is already up >= this threshold.
# Below this gain the candle is just normal volatility, not climactic action.
MIN_GAIN_RULE26_STANDALONE = 0.05     # 5% gain required for rule26 alone

# ── Pass criteria ─────────────────────────────────────────────────────────
TARGET_CAGR = 1.00                    # 100% CAGR
TARGET_WIN_RATE = 0.60                # 60%
MIN_TRADES = 100
