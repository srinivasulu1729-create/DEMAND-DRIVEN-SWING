"""
debug_single.py — trace exactly why the backtest misses a known winner.

Usage:
    python debug_single.py ZENTEC
    python debug_single.py MANINDS --start 2023-01-01 --end 2023-12-31

For every trading day the backtest processes, prints:
  - open positions + slots available
  - RS rank of the target symbol in the candidate pool
  - entry_checklist result (key fields)
  - whether it was entered / why not
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from data_loader import load_stock, load_index, align_stock_index
from indicators import resample_weekly, index_regime, add_weekly_emas
from signals import entry_checklist
from config import (
    RS_MIN_RATIO, MAX_POSITIONS_HARD_CAP, MAX_POSITION_PCT, MAX_POSITION_ABS,
    POSITION_LOOKBACK_DAYS, R01_WATCHLIST_DAYS, R01_MIN_MOVE,
    STARTING_CAPITAL, POSITION_SL_PCT,
)

# ── CLI ────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("symbol", help="Symbol to debug, e.g. ZENTEC")
ap.add_argument("--start", default="2023-01-01")
ap.add_argument("--end",   default="2023-12-31")
args = ap.parse_args()

TARGET  = args.symbol.upper()
T_START = pd.Timestamp(args.start)
T_END   = pd.Timestamp(args.end)

print(f"\nLoading index ...")
idx_full = load_index()
idx_full["date"] = pd.to_datetime(idx_full["date"])

print(f"Loading target stock: {TARGET} ...")
tgt_raw = load_stock(TARGET)
if tgt_raw is None:
    print(f"ERROR: no data for {TARGET}"); sys.exit(1)
tgt_raw["date"] = pd.to_datetime(tgt_raw["date"])

# Align to index
from data_loader import align_stock_index
tgt_s, tgt_i = align_stock_index(tgt_raw, idx_full)

# Trading dates in range
all_dates = sorted(idx_full["date"].unique())
sim_dates = [d for d in all_dates if T_START <= d <= T_END]

print(f"\n{'='*70}")
print(f"  SINGLE-STOCK BACKTEST DEBUG: {TARGET}  ({args.start} → {args.end})")
print(f"{'='*70}\n")
print(f"{'DATE':<12}  {'SLOTS':>5}  {'RS_RANK':>8}  {'RS_VAL':>8}  "
      f"{'r01':>4}  {'r03':>4}  {'r04':>4}  {'r_ent':>5}  "
      f"{'nconf':>5}  {'ALL':>4}  REASON")
print("-"*110)

# Simulate open positions (just for THIS debug — we run target-only sim)
# Actually track a pseudo-portfolio so n_slots is realistic.
# Simple model: pretend we always have 0 open positions (worst-case for target).
# The real issue is whether entry_checklist passes + RS rank is within cap.

entry_count = 0

for date in sim_dates:
    # Build history slices up to this date
    s_h = tgt_s[tgt_s["date"] <= date]
    i_h = tgt_i[tgt_i["date"] <= date]

    if len(s_h) < 60:
        continue

    # Compute RS for ranking
    lb = POSITION_LOOKBACK_DAYS
    if len(s_h) < lb + 1 or len(i_h) < lb + 1:
        continue
    s_ret = float(s_h["Close"].pct_change(lb).iloc[-1])
    i_ret = float(i_h["Close"].pct_change(lb).iloc[-1])
    rs_val = (s_ret / i_ret) if i_ret != 0 else np.nan

    # Load weekly data
    wkly     = resample_weekly(s_h)
    idx_wkly = resample_weekly(i_h)

    if len(wkly) < 10:
        continue

    # Run entry_checklist
    try:
        cl = entry_checklist(s_h, i_h, wkly, idx_wkly, "position")
    except Exception as exc:
        print(f"{str(date.date()):<12}  EXCEPTION: {exc}")
        continue

    r01   = cl.get("r01_price_50pct", False)
    r03   = cl.get("r03_rs_ratio", False)
    r04   = cl.get("r04_index_regime", False)
    r_ent = cl.get("r_entry_valid", False)
    nconf = cl.get("n_confirm_score", 0)
    all_p = cl.get("all_pass", False)
    nn    = cl.get("non_negotiable_pass", False)

    # Determine blocking reason
    if all_p:
        reason = ">>> ENTRY SIGNAL <<<"
        entry_count += 1
    elif not r01:
        reason = "r01 FAIL (no 43%+ move in 750d)"
    elif not r03:
        reason = f"r03 FAIL (RS={rs_val:.2f}x < 3x)"
    elif not r04:
        reason = "r04 FAIL (index not bull)"
    elif not r_ent:
        # Which entry patterns fired?
        wi  = cl.get("r_weekly_inside_breakout", False)
        wm  = cl.get("r_10wma_support_bounce", False)
        ph  = cl.get("r_prev_high_near_10wma", False)
        vcp = cl.get("r_vcp_breakout", False)
        reason = f"NO entry pattern (wi={wi} 10wma={wm} prevhi={ph} vcp={vcp})"
    elif nconf < 3:
        r02  = cl.get("r02_green_weekly", False)
        rb   = cl.get("r_base_depth_8_25pct", False)
        r11  = cl.get("r11_vol_contracting", False)
        r08  = cl.get("r08_rs_near_highs", False)
        r09  = cl.get("r09_stage2", False)
        rbf  = cl.get("r_base_formed_2wk", False)
        rvol = cl.get("r_volume_confirms", False)
        reason = (f"n_confirm={nconf}<3  "
                  f"r02={int(r02)} rb={int(rb)} r11={int(r11)} "
                  f"rvol={int(rvol)} r08={int(r08)} r09={int(r09)} rbf={int(rbf)}")
    else:
        reason = f"BLOCKED other  nn={nn}"

    # Only print significant rows: entry signals + days where r01+r03+r04 all pass
    if all_p or (r01 and r03 and r04):
        rs_str = f"{rs_val:.1f}x" if not np.isnan(rs_val) else "  nan"
        print(f"{str(date.date()):<12}  {'?':>5}  {'N/A':>8}  {rs_str:>8}  "
              f"{'✓' if r01 else '✗':>4}  {'✓' if r03 else '✗':>4}  "
              f"{'✓' if r04 else '✗':>4}  {'✓' if r_ent else '✗':>5}  "
              f"{nconf:>5}  {'✓' if all_p else '✗':>4}  {reason}")

print(f"\n  Total ENTRY signals found: {entry_count}")

# ── Now check RS rank on each entry date ──────────────────────────────────
print(f"\n{'='*70}")
print(f"  RS RANK CHECK on entry signal dates")
print(f"  (Shows where {TARGET} ranks among all RS >= 3x stocks that day)")
print(f"{'='*70}")

# Load a batch of symbols to compute ranks
import csv, random
symbols_csv = os.path.join(os.path.dirname(__file__), "symbols.csv")
try:
    with open(symbols_csv) as f:
        all_syms = [row[0].strip() for row in csv.reader(f) if row]
    # Sample 300 random symbols + target for rank estimation
    sample_syms = random.sample([s for s in all_syms if s != TARGET], min(299, len(all_syms)-1))
    sample_syms.append(TARGET)
    print(f"  Sampling {len(sample_syms)} symbols for rank estimation (includes {TARGET})\n")

    # Find entry signal dates
    entry_dates = []
    for date in sim_dates:
        s_h = tgt_s[tgt_s["date"] <= date]
        i_h = tgt_i[tgt_i["date"] <= date]
        if len(s_h) < 60: continue
        wkly = resample_weekly(s_h)
        idx_wkly = resample_weekly(i_h)
        if len(wkly) < 10: continue
        try:
            cl = entry_checklist(s_h, i_h, wkly, idx_wkly, "position")
            if cl.get("all_pass"):
                entry_dates.append(date)
        except:
            pass

    if not entry_dates:
        print("  No entry dates to rank-check.")
    else:
        for edate in entry_dates[:5]:  # check first 5 entry dates
            # Compute RS for all sample symbols on this date
            rs_list = []
            for sym in sample_syms:
                if sym == TARGET:
                    # Use pre-loaded target data
                    sh = tgt_s[tgt_s["date"] <= edate]
                    ih = tgt_i[tgt_i["date"] <= edate]
                else:
                    raw = load_stock(sym)
                    if raw is None: continue
                    raw["date"] = pd.to_datetime(raw["date"])
                    sh, ih = align_stock_index(raw, idx_full)

                if len(sh) < POSITION_LOOKBACK_DAYS + 1: continue
                sh2 = sh[sh["date"] <= edate]
                ih2 = ih[ih["date"] <= edate]
                if len(sh2) < POSITION_LOOKBACK_DAYS + 1: continue
                s_r = float(sh2["Close"].pct_change(POSITION_LOOKBACK_DAYS).iloc[-1])
                i_r = float(ih2["Close"].pct_change(POSITION_LOOKBACK_DAYS).iloc[-1])
                if i_r == 0 or np.isnan(s_r) or np.isnan(i_r): continue
                rv = s_r / i_r
                if rv >= RS_MIN_RATIO:
                    rs_list.append((rv, sym))

            rs_list.sort(reverse=True)
            tgt_pos = next((i+1 for i, (_, s) in enumerate(rs_list) if s == TARGET), None)
            print(f"  {edate.date()}  RS >= 3x pool size: {len(rs_list):3d}  "
                  f"{TARGET} rank: {tgt_pos or 'NOT IN POOL'}")
            print(f"    Top 10: {', '.join(f'{s}({rv:.1f}x)' for rv, s in rs_list[:10])}")
            if tgt_pos:
                print(f"    {TARGET}: rank {tgt_pos} of {len(rs_list)}  RS={next(rv for rv,s in rs_list if s==TARGET):.1f}x")
            print()

except Exception as e:
    print(f"  Could not load symbols for rank check: {e}")
