"""
generate_symbols.py — One-time utility
Scans BASE_DIR for symbol=<NAME> sub-folders and writes symbols.csv
next to this script.  Run once before --backtest or --screen.

Usage:
    python generate_symbols.py
    python generate_symbols.py --min-files 2       # skip symbols with < 2 parquet files
    python generate_symbols.py --exclude SYMBOL1 SYMBOL2
"""

import argparse
import os
import sys

from config import BASE_DIR, INDEX_SYMBOL, SYMBOLS_CSV


# Index-format column signature (lower-case): has 'eodtime' or lacks 'close price'
_INDEX_COLS = {"eodtime", "datetime"}


def _is_stock_schema(folder_path: str) -> bool:
    """Return True only if the parquet files use the stock OHLCV schema
    (columns: Date, Open Price, High Price, Low Price, Close Price, Total Traded Quantity).
    Folders using the index schema (eodtime / open / close / volume) are rejected.
    """
    import pyarrow.parquet as pq
    pq_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".parquet")
    ]
    if not pq_files:
        return False
    try:
        schema = pq.read_schema(pq_files[0])
        col_names_lower = {c.lower() for c in schema.names}
        # Must have the stock-specific column name
        return "close price" in col_names_lower
    except Exception:
        return False


def scan_symbols(base_dir: str, min_files: int = 1, exclude: set = None) -> list[str]:
    exclude = exclude or set()
    symbols = []
    skipped_index_schema = []

    if not os.path.isdir(base_dir):
        print(f"ERROR: BASE_DIR not found: {base_dir}")
        sys.exit(1)

    for entry in sorted(os.scandir(base_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("symbol="):
            continue
        sym = name[len("symbol="):]
        if not sym:
            continue
        if sym == INDEX_SYMBOL:
            continue                          # skip benchmark index
        if sym in exclude:
            print(f"  EXCLUDED: {sym}")
            continue

        # Count parquet files
        pq_files = [
            f for f in os.listdir(entry.path)
            if f.endswith(".parquet")
        ]
        if len(pq_files) < min_files:
            print(f"  SKIP (only {len(pq_files)} parquet files): {sym}")
            continue

        # Schema check: skip index-format folders (NIFTY* indices, misclassified files, etc.)
        if not _is_stock_schema(entry.path):
            skipped_index_schema.append(sym)
            continue

        symbols.append(sym)

    if skipped_index_schema:
        print(f"  Skipped {len(skipped_index_schema)} index-schema folders "
              f"(not stock data): {skipped_index_schema[:8]}"
              + (" ..." if len(skipped_index_schema) > 8 else ""))

    return symbols


def main():
    parser = argparse.ArgumentParser(description="Generate symbols.csv from parquet store")
    parser.add_argument("--min-files", type=int, default=1,
                        help="Minimum parquet files a symbol must have (default 1)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Symbols to exclude")
    args = parser.parse_args()

    print(f"Scanning: {BASE_DIR}")
    symbols = scan_symbols(BASE_DIR, min_files=args.min_files, exclude=set(args.exclude))

    if not symbols:
        print("No symbols found. Check BASE_DIR in config.py.")
        sys.exit(1)

    with open(SYMBOLS_CSV, "w") as f:
        f.write("Symbol\n")
        for sym in symbols:
            f.write(sym + "\n")

    print(f"\nWrote {len(symbols)} symbols to: {SYMBOLS_CSV}")
    print(f"First 10: {symbols[:10]}")


if __name__ == "__main__":
    main()
