"""
main.py — Demand-Driven Swing & Position System

Usage:
  python main.py --backtest            Run full backtest (parallel symbol load)
  python main.py --screen              Run live screener (parallel, needs backtest pass)
  python main.py --validate            Cross-check TradeLog.xlsx (full audit report)
  python main.py --all                 Backtest -> validate -> screen
  python main.py --force-screen        Screen without confirmed backtest pass
  python main.py --workers N           Set parallel worker count (default: CPU-1)
  python main.py --top N               Show top N screener results (default 20)
  python main.py --log DEBUG           Set log level
"""

import argparse
import json
import logging
import os
import sys

from config import OUTPUT_DIR, TARGET_CAGR, TARGET_WIN_RATE, MIN_TRADES


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _backtest_has_passed() -> bool:
    metrics_path = os.path.join(OUTPUT_DIR, "backtest_metrics.json")
    if not os.path.exists(metrics_path):
        return False
    try:
        with open(metrics_path) as f:
            m = json.load(f)
        return bool(m.get("OVERALL_PASS", False))
    except Exception:
        return False


def run_backtest(n_workers=None, start_date=None, end_date=None):
    print("\n[1/3] Loading index ...")
    from data_loader import load_index, load_symbols
    from backtest_engine import Backtester, parallel_load_all_stocks

    index_df   = load_index()
    symbols    = load_symbols()
    print(f"[2/3] Pre-loading {len(symbols)} symbols in parallel ...")
    stock_data = parallel_load_all_stocks(symbols, index_df, n_workers=n_workers)
    print(f"[3/3] Running backtest on {len(stock_data)} symbols ...")
    bt     = Backtester(stock_data, index_df)
    result = bt.run(start_date=start_date, end_date=end_date)
    result.print_report()
    result.save(OUTPUT_DIR)
    return result.metrics.get("OVERALL_PASS", False)


def run_screener(force=False, top_n=20, n_workers=None):
    passed = _backtest_has_passed()
    if not passed and not force:
        print("\n  Backtest has NOT been confirmed as PASS.")
        print("  Run: python main.py --backtest  first.")
        print("  To override: use --force-screen\n")
        sys.exit(1)
    from live_screener import screen, print_screen_results, save_screen_results
    df = screen(backtest_passed=passed or force, n_workers=n_workers)
    print_screen_results(df, top_n=top_n)
    save_screen_results(df, output_dir=OUTPUT_DIR)
    return df


def run_validation():
    from audit_report import run_audit
    return run_audit(output_dir=OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Demand-Driven Swing & Position System"
    )
    parser.add_argument("--backtest",     action="store_true")
    parser.add_argument("--screen",       action="store_true")
    parser.add_argument("--validate",     action="store_true")
    parser.add_argument("--all",          action="store_true")
    parser.add_argument("--force-screen", action="store_true")
    parser.add_argument("--top",     type=int,  default=20)
    parser.add_argument("--workers", type=int,  default=None,
                        help="Parallel workers (default: CPU-1)")
    parser.add_argument("--start",   default=None,
                        help="Backtest start date  YYYY-MM-DD  (default: all data)")
    parser.add_argument("--end",     default=None,
                        help="Backtest end date    YYYY-MM-DD  (default: today)")
    parser.add_argument("--log",     default="INFO")
    args = parser.parse_args()

    setup_logging(args.log)
    logger = logging.getLogger(__name__)

    if not any([args.backtest, args.screen, args.validate, args.all]):
        parser.print_help()
        sys.exit(0)

    bt_passed = False

    if args.all or args.backtest:
        logger.info("=== BACKTEST ===")
        bt_passed = run_backtest(n_workers=args.workers,
                                 start_date=args.start, end_date=args.end)

    if args.all or args.validate:
        logger.info("=== TRADE LOG VALIDATION / AUDIT ===")
        run_validation()

    if args.all or args.screen:
        logger.info("=== LIVE SCREENER ===")
        run_screener(force=args.force_screen or bt_passed,
                     top_n=args.top, n_workers=args.workers)

    logger.info("Done.")


if __name__ == "__main__":
    main()
