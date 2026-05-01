"""
debug_date_format.py — diagnose date parsing failures in old parquet files.

Run:  python debug_date_format.py JAIBALAJI
      python debug_date_format.py JAIBALAJI AURIONPRO INOXWIND
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))

if __name__ != '__main__':
    sys.exit(0)

import duckdb, pandas as pd
from config import BASE_DIR

SYMBOLS = [s.upper() for s in sys.argv[1:]] if len(sys.argv) > 1 else ["JAIBALAJI"]
con = duckdb.connect(":memory:")

for sym in SYMBOLS:
    pattern = os.path.join(BASE_DIR, f"symbol={sym}", "*.parquet").replace("\\", "/")
    files   = sorted(glob.glob(pattern.replace("/", os.sep)))
    print(f"\n{'='*60}")
    print(f"  {sym}  —  {len(files)} parquet file(s) found")
    print(f"{'='*60}")
    if not files:
        print("  NOT FOUND on disk")
        continue

    total_strict_ok = 0
    total_auto_ok   = 0
    total_fail      = 0

    for f in files:
        year = os.path.basename(f).replace(".parquet", "")
        fp   = f.replace("\\", "/")
        try:
            raw = con.execute(
                f"SELECT \"Date\" FROM read_parquet('{fp}') LIMIT 5"
            ).df()
            n   = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{fp}')"
            ).fetchone()[0]

            sample         = raw["Date"].tolist()
            parsed_strict  = pd.to_datetime(raw["Date"], format="%d-%b-%Y", errors="coerce")
            parsed_auto    = pd.to_datetime(raw["Date"], errors="coerce")
            strict_ok      = parsed_strict.notna().all()
            auto_ok        = parsed_auto.notna().all()

            if strict_ok:
                status = "✓ STRICT OK"
                total_strict_ok += n
            elif auto_ok:
                status = "⚠ STRICT FAIL, AUTO OK  ← these rows get DROPPED currently"
                total_auto_ok += n
            else:
                status = "✗ BOTH FAIL"
                total_fail += n

            print(f"  {year:>6}  {n:>6} rows  {status}")
            print(f"           sample dates: {sample[:3]}")
        except Exception as e:
            print(f"  {year:>6}  ERROR: {e}")

    print(f"\n  Rows loaded correctly (strict format): {total_strict_ok:>6}")
    print(f"  Rows SILENTLY DROPPED (auto ok, strict fails): {total_auto_ok:>6}  ← FIX NEEDED")
    print(f"  Rows truly unparseable: {total_fail:>6}")
