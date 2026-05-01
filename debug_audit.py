"""
debug_audit.py — diagnose why entry_checklist misses known winners

Run: python debug_audit.py
"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from data_loader import load_stock, load_index, align_stock_index
from signals import entry_checklist, rule01_price_moved_50pct
from config import POSITION_LOOKBACK_DAYS, R01_WATCHLIST_DAYS, R01_MIN_MOVE

SYMBOLS = ["TITAGARH", "ZENTEC", "HBLENGINE"]

# Known alert dates from _scan_r01_detail output
KNOWN_ALERT_END = {
    "TITAGARH":  "2023-04-21",
    "ZENTEC":    "2023-04-18",
    "HBLENGINE": "2023-06-14",
}

print("Loading index ...")
index_df = load_index()

for sym in SYMBOLS:
    print(f"\n{'='*60}")
    print(f"  DEBUG: {sym}")
    print(f"{'='*60}")

    stock = load_stock(sym)
    if stock is None:
        print("  ERROR: load_stock returned None")
        continue

    s_al, i_al = align_stock_index(stock, index_df)
    print(f"  s_al shape: {s_al.shape}  columns: {list(s_al.columns)}")
    print(f"  date col type: {type(s_al['date'].iloc[0])}")

    # ── Test R01 directly on full and sliced data ──────────────────────────
    test_date = pd.Timestamp(KNOWN_ALERT_END[sym])

    # Full data
    r01_full = rule01_price_moved_50pct(s_al)
    print(f"\n  rule01 on FULL s_al:          {r01_full}")

    # Sliced data (as audit does it)
    s_hist = s_al[s_al["date"] <= test_date]
    i_hist = i_al[i_al["date"] <= test_date]
    r01_sliced = rule01_price_moved_50pct(s_hist)
    print(f"  rule01 on SLICED s_hist:      {r01_sliced}")
    print(f"  s_hist rows: {len(s_hist)}  last date: {s_hist['date'].iloc[-1]}")

    # ── Try entry_checklist and print full exception if any ───────────────
    print(f"\n  Calling entry_checklist on {test_date.date()} ...")
    try:
        cl = entry_checklist(s_hist, i_hist)
        print(f"  SUCCESS — checklist returned {len(cl)} keys")
        print(f"\n  KEY VALUES:")
        for k in ["r01_price_50pct","r03_rs_ratio","r03_rs_value","r04_index_regime",
                  "r02_green_weekly","r09_stage2","r_base_depth_8_25pct",
                  "r11_vol_contracting","r08_rs_near_highs","r_base_formed_2wk",
                  "r_weekly_inside_breakout","r_10wma_support_bounce",
                  "r_prev_high_near_10wma","r_vcp_breakout","r_entry_valid",
                  "n_confirm_score","non_negotiable_pass","all_pass"]:
            v = cl.get(k, "NOT IN DICT")
            flag = "  ← BLOCKING" if k in ("r01_price_50pct","r03_rs_ratio","r04_index_regime",
                                             "non_negotiable_pass","all_pass") and not v else ""
            print(f"    {k:<32}: {v}{flag}")
    except Exception as exc:
        print(f"  EXCEPTION in entry_checklist: {exc}")
        traceback.print_exc()

    # ── If r01 comes back False, scan manually ────────────────────────────
    if not r01_sliced:
        print(f"\n  r01 False on sliced data — manual scan:")
        close = s_hist["Close"].values
        n = len(close)
        lb = POSITION_LOOKBACK_DAYS
        scan_start = max(lb, n - R01_WATCHLIST_DAYS)
        print(f"  n={n}  scan_start={scan_start}  scanning [{scan_start}..{n-1}]")
        hits = 0
        for i in range(scan_start, n):
            base = close[i - lb]
            if base > 0 and (close[i] / base - 1) >= R01_MIN_MOVE:
                hits += 1
        print(f"  Manual scan found {hits} qualifying windows in sliced data")
        # Try wider range
        for extra in [10, 20]:
            hits2 = 0
            for i in range(max(lb+extra, n - R01_WATCHLIST_DAYS), n):
                if i < lb+extra: continue
                base = close[i - (lb+extra)]
                if base > 0 and (close[i] / base - 1) >= R01_MIN_MOVE:
                    hits2 += 1
            if hits2 > 0:
                print(f"  With {lb+extra}-bar window: {hits2} hits found")

    # ── Scan for first entry date (with exception logging) ────────────────
    print(f"\n  Scanning for first entry date (May-Sep 2023) ...")
    t0 = pd.Timestamp("2023-05-01")
    t1 = pd.Timestamp("2023-09-30")
    scan_dates = s_al[(s_al["date"] >= t0) & (s_al["date"] <= t1)]["date"].iloc[::5]  # every 5th day

    exceptions_seen = {}
    entries_found = []
    r01_true_count = 0

    for date in scan_dates:
        sh = s_al[s_al["date"] <= date]
        ih = i_al[i_al["date"] <= date]
        try:
            cl = entry_checklist(sh, ih)
            if cl.get("r01_price_50pct"):
                r01_true_count += 1
            if cl.get("all_pass"):
                entries_found.append((date.date(), cl))
        except Exception as exc:
            key = type(exc).__name__ + str(exc)[:60]
            exceptions_seen[key] = exceptions_seen.get(key, 0) + 1

    print(f"  Dates scanned: {len(scan_dates)}")
    print(f"  r01=True count: {r01_true_count}")
    print(f"  all_pass=True count: {len(entries_found)}")
    if exceptions_seen:
        print(f"  Exceptions encountered:")
        for k, v in exceptions_seen.items():
            print(f"    [{v}x] {k}")
    if r01_true_count == 0:
        print(f"\n  r01 is ALWAYS False — checking why with first date in range:")
        d = scan_dates.iloc[0]
        sh = s_al[s_al["date"] <= d]
        ih = i_al[i_al["date"] <= d]
        close = sh["Close"].values
        n = len(close)
        lb = POSITION_LOOKBACK_DAYS
        scan_start = max(lb, n - R01_WATCHLIST_DAYS)
        print(f"    date={d.date()}  n={n}  R01_WATCHLIST_DAYS={R01_WATCHLIST_DAYS}  scan_start={scan_start}")
        # show last 5 windows
        for i in range(max(scan_start, n-5), n):
            base = close[i-lb]
            pct = (close[i]/base - 1)*100 if base > 0 else 0
            print(f"    bar {i}: pct={pct:.1f}%  needed={R01_MIN_MOVE*100:.0f}%  pass={pct>=R01_MIN_MOVE*100}")
