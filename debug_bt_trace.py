"""
debug_bt_trace.py — trace exactly what the backtest does with target symbols.

Run:  python debug_bt_trace.py TITAGARH ZENTEC MANINDS ANANTRAJ
      python debug_bt_trace.py TITAGARH --start 2023-01-01 --end 2023-06-30

Shows for EACH trading day:
  - Whether the target symbol is in the RS >= 3x candidate pool
  - Its RS rank among all candidates
  - How many positions are open
  - Whether entry_checklist passes
  - Why it was/wasn't entered
"""
import sys, os, argparse, logging
sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.WARNING)   # suppress backtest noise

import numpy as np
import pandas as pd

from data_loader import load_index, load_all_stocks
from backtest_engine import parallel_load_all_stocks
from indicators  import resample_weekly, index_regime
from signals     import entry_checklist
from config import (
    RS_MIN_RATIO, MAX_POSITIONS_HARD_CAP,
    MAX_POSITION_PCT, MAX_POSITION_ABS,
    POSITION_LOOKBACK_DAYS, STARTING_CAPITAL,
    POSITION_SL_PCT, SYMBOLS_CSV,
    MIN_TRADING_ROWS,
)

# ── CLI ────────────────────────────────────────────────────────────────────
# NOTE: must be guarded with if __name__ == '__main__' so Windows multiprocessing
# workers (spawned by parallel_load_all_stocks) don't re-run this whole script.
if __name__ != '__main__':
    import sys; sys.exit(0)

ap = argparse.ArgumentParser()
ap.add_argument("targets", nargs="+", help="Symbols to trace, e.g. TITAGARH ZENTEC")
ap.add_argument("--start", default="2023-01-01")
ap.add_argument("--end",   default="2023-12-31")
ap.add_argument("--workers", type=int, default=6)
args = ap.parse_args()

TARGETS  = [s.upper() for s in args.targets]
T_START  = pd.Timestamp(args.start)
T_END    = pd.Timestamp(args.end)

# ── Load universe ──────────────────────────────────────────────────────────
print("Loading index ...")
index_df = load_index()
index_df["date"] = pd.to_datetime(index_df["date"])

import csv as _csv
with open(SYMBOLS_CSV) as f:
    all_syms = [r[0].strip() for r in _csv.reader(f) if r]

# Confirm targets are in symbols.csv
for t in TARGETS:
    in_csv = t in all_syms
    print(f"  {t} in symbols.csv: {in_csv}")

print(f"\nPre-loading {len(all_syms)} symbols ...")
stock_data = parallel_load_all_stocks(all_syms, index_df, n_workers=args.workers)
print(f"Universe loaded: {len(stock_data)} symbols")

# Confirm targets loaded
for t in TARGETS:
    if t in stock_data:
        rows = len(stock_data[t][0])
        print(f"  {t}: {rows} rows loaded  ✓")
    else:
        print(f"  {t}: NOT IN stock_data — failed to load or < {MIN_TRADING_ROWS} rows  ✗")

# ── Build position table (same as backtest) ────────────────────────────────
from data_loader import align_stock_index
import numpy as np

all_dates_dti = pd.DatetimeIndex(sorted(index_df["date"].unique()))
n_td = len(all_dates_dti)

idx_dti = pd.DatetimeIndex(index_df["date"])
idx_pos = idx_dti.searchsorted(all_dates_dti, side="right").astype(np.int32)

pos_table = {}
rs_arr    = {}
for sym, (sdf, idf) in stock_data.items():
    dti = pd.DatetimeIndex(sdf["date"])
    pos_table[sym] = dti.searchsorted(all_dates_dti, side="right").astype(np.int32)
    s_ret = sdf["Close"].pct_change(POSITION_LOOKBACK_DAYS).values
    i_ret = idf["Close"].pct_change(POSITION_LOOKBACK_DAYS).values
    with np.errstate(divide="ignore", invalid="ignore"):
        rs_arr[sym] = np.where(i_ret != 0, s_ret / i_ret, np.nan)

# ── Simulate entry scan (no exits — just entry decisions) ─────────────────
sim_dates  = [d for d in all_dates_dti if T_START <= d <= T_END]
date_index = {d: i for i, d in enumerate(all_dates_dti)}

open_trades = []   # list of (symbol, entry_date)
results     = []   # per-date trace for targets

_cached_week     = None
_cached_idx_wkly = None
_cached_regime   = "bull"

print(f"\n{'='*80}")
print(f"  ENTRY SCAN TRACE  {args.start} → {args.end}")
print(f"  Targets: {', '.join(TARGETS)}")
print(f"{'='*80}")
print(f"\n{'DATE':<12}  {'OPEN':>4}  {'SLOTS':>5}  ", end="")
for t in TARGETS:
    print(f"  {t[:10]:<10} [RS/rank/pass]", end="")
print()
print("-"*80)

entries_made  = {t: [] for t in TARGETS}
missed_reason = {t: {} for t in TARGETS}  # reason → count

for date in sim_dates:
    date_idx = date_index[date]
    ip = int(idx_pos[date_idx])
    if ip < 40:
        continue

    # Cache weekly index
    iso_week = date.isocalendar()[1]
    if iso_week != _cached_week:
        _cached_week     = iso_week
        _cached_idx_wkly = resample_weekly(index_df.iloc[:ip])
        _cached_regime   = index_regime(_cached_idx_wkly)
    idx_wkly = _cached_idx_wkly
    regime   = _cached_regime

    open_syms = {sym for sym, _ in open_trades}
    n_open    = len(open_trades)
    n_slots   = MAX_POSITIONS_HARD_CAP - n_open

    # Build RS >= 3x candidate pool (same as fixed backtest)
    cands = []
    for sym, (sdf, idf) in stock_data.items():
        if sym in open_syms:
            continue
        sp = int(pos_table[sym][date_idx])
        if sp < 60:
            continue
        rv = float(rs_arr[sym][sp - 1])
        if np.isnan(rv) or rv < RS_MIN_RATIO:
            continue
        cands.append((rv, sym, sp))
    cands.sort(key=lambda x: x[0], reverse=True)

    # Find rank of each target
    tgt_info = {}
    for t in TARGETS:
        rank = next((i+1 for i, (_, s, _) in enumerate(cands) if s == t), None)
        rs_v = next((rv for rv, s, _ in cands if s == t), None)
        tgt_info[t] = (rank, rs_v)

    # Simulate entries (try top candidates, stop when slots full)
    added    = 0
    tried    = set()
    entries_today = []
    for rv, sym, sp in cands:
        if added >= n_slots:
            break
        tried.add(sym)
        sdf, idf = stock_data[sym]
        s_h  = sdf.iloc[:sp]
        i_h  = idf.iloc[:sp]
        wkly = resample_weekly(s_h)
        if len(wkly) < 10:
            continue
        try:
            cl = entry_checklist(s_h, i_h, wkly, idx_wkly, "position")
        except Exception as exc:
            continue
        if cl.get("all_pass"):
            open_trades.append((sym, date))
            entries_today.append(sym)
            added += 1

    # Simple exit simulation: close after 30 days to free slots
    open_trades = [(s, d) for s, d in open_trades
                   if (date - d).days < 30]

    # Check each target
    any_target_news = False
    row_parts = []
    for t in TARGETS:
        rank, rs_v = tgt_info[t]
        entered = t in entries_today
        if entered:
            entries_made[t].append(date)
            status = "ENTERED"
            any_target_news = True
        elif rank is None:
            # Not in RS >= 3x pool at all
            sp_t = int(pos_table.get(t, [0]*len(all_dates_dti))[date_idx]) if t in pos_table else 0
            if t not in stock_data:
                status = "NOT_LOADED"
            elif sp_t < 60:
                status = "TOO_FEW_ROWS"
            else:
                rv_t = float(rs_arr[t][sp_t - 1]) if sp_t > 0 else float('nan')
                status = f"RS_FILTER  rv={rv_t:.2f}x"
                if t in (s for _, s, _ in cands[:3]):
                    any_target_news = True
            missed_reason[t][status.split()[0]] = missed_reason[t].get(status.split()[0], 0) + 1
        elif t in open_syms:
            status = "ALREADY_OPEN"
        elif rank > n_slots and n_slots <= 0:
            status = f"NO_SLOTS   (rank={rank}/{len(cands)})"
            missed_reason[t]["NO_SLOTS"] = missed_reason[t].get("NO_SLOTS", 0) + 1
            any_target_news = True
        elif t not in tried:
            status = f"NOT_TRIED  slots_used_before_rank  (rank={rank}/{len(cands)})"
            missed_reason[t]["NOT_TRIED"] = missed_reason[t].get("NOT_TRIED", 0) + 1
            if rank <= 15:
                any_target_news = True
        else:
            # Was tried but entry_checklist failed
            sp_t = int(pos_table[t][date_idx])
            sdf_t, idf_t = stock_data[t]
            s_h  = sdf_t.iloc[:sp_t]
            i_h  = idf_t.iloc[:sp_t]
            wkly_t = resample_weekly(s_h)
            try:
                cl = entry_checklist(s_h, i_h, wkly_t, idx_wkly, "position")
                r01 = cl.get("r01_price_50pct", False)
                r03 = cl.get("r03_rs_ratio", False)
                r04 = cl.get("r04_index_regime", False)
                re  = cl.get("r_entry_valid", False)
                nc  = cl.get("n_confirm_score", 0)
                if not r01:   blk = "r01_FAIL"
                elif not r03: blk = f"r03_FAIL(rs={rs_v:.1f}x)"
                elif not r04: blk = "r04_FAIL"
                elif not re:  blk = "no_entry_pattern"
                elif nc < 3:  blk = f"n_confirm={nc}<3"
                else:         blk = "UNKNOWN"
                status = f"CHECKED_FAIL  {blk}  (rank={rank}/{len(cands)})"
                any_target_news = True
                missed_reason[t][blk] = missed_reason[t].get(blk, 0) + 1
            except Exception as exc:
                status = f"EXCEPTION: {exc}"
                any_target_news = True

        rs_str = f"{rs_v:.1f}x" if rs_v else "  ---"
        row_parts.append(f"  {rs_str:>7} #{rank or '---':>4}  {status}")

    if any_target_news:
        print(f"{str(date.date()):<12}  {n_open:>4}  {n_slots:>5}  {''.join(row_parts)}")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  SUMMARY")
print(f"{'='*80}")
for t in TARGETS:
    print(f"\n  {t}:")
    if entries_made[t]:
        print(f"    Entered {len(entries_made[t])} times: "
              f"{', '.join(str(d.date()) for d in entries_made[t][:10])}")
    else:
        print(f"    NEVER ENTERED")
    if missed_reason[t]:
        print(f"    Missed reasons:")
        for reason, cnt in sorted(missed_reason[t].items(), key=lambda x: -x[1]):
            print(f"      {reason:<30}: {cnt} days")
