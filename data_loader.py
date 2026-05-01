"""
data_loader.py — Demand-Driven Swing & Position System
DuckDB-powered data loading. Queries parquet files directly.
Validation rules V1-V6 enforced after each load.
"""

import logging
import os
import glob

import duckdb
import pandas as pd

from config import (
    BASE_DIR, INDEX_SYMBOL, SYMBOLS_CSV,
    MIN_PRICE, MIN_AVG_DAILY_VOL, MIN_TRADING_ROWS,
)

logger = logging.getLogger(__name__)
_con = duckdb.connect(database=":memory:")


def _parquet_glob(symbol: str) -> str:
    return os.path.join(BASE_DIR, f"symbol={symbol}", "*.parquet").replace("\\", "/")


def _query_index() -> pd.DataFrame:
    pattern = _parquet_glob(INDEX_SYMBOL)
    sql = (
        "SELECT strptime(eodtime, '%d-%b-%Y')::DATE AS date, "
        "CAST(\"open\" AS DOUBLE) AS Open, CAST(\"high\" AS DOUBLE) AS High, "
        "CAST(\"low\" AS DOUBLE) AS Low, CAST(\"close\" AS DOUBLE) AS Close, "
        "CAST(\"volume\" AS BIGINT) AS Volume "
        f"FROM read_parquet('{pattern}', union_by_name=true) "
        "WHERE TRY_CAST(\"close\" AS DOUBLE) > 0 AND TRY_CAST(\"volume\" AS BIGINT) > 0 ORDER BY date"
    )
    try:
        df = _con.execute(sql).df()
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as exc:
        raise FileNotFoundError(f"Index data not found at: {pattern}") from exc


def _query_stock(symbol: str) -> pd.DataFrame | None:
    pattern = _parquet_glob(symbol)
    if not glob.glob(pattern.replace("/", os.sep)):
        return None
    # Read raw columns without any SQL-level casting so DuckDB type mismatches
    # (VARCHAR vs DOUBLE across year-files) never cause BinderException or
    # ConversionError.  All cleaning is done in pandas below.
    sql = (
        "SELECT \"Date\", \"Open Price\", \"High Price\", \"Low Price\", "
        "\"Close Price\", \"Total Traded Quantity\" "
        f"FROM read_parquet('{pattern}', union_by_name=true)"
    )
    try:
        raw = _con.execute(sql).df()
        if raw.empty:
            return None
        df = pd.DataFrame()
        # FIX: do NOT pass format= — old parquet years (2010-2021) may store
        # dates in a different format (e.g. "2010-01-03" vs "03-Jan-2023").
        # Strict format="%d-%b-%Y" silently converts mismatches to NaT which
        # then get dropped, causing stocks to appear as if they have no history
        # before ~2022 even when full data exists on disk.
        df["date"] = pd.to_datetime(raw["Date"], errors="coerce")
        # Price columns: handle both DOUBLE (some years) and VARCHAR-with-commas (other years)
        for raw_col, alias in [
            ("Open Price",  "Open"),
            ("High Price",  "High"),
            ("Low Price",   "Low"),
            ("Close Price", "Close"),
        ]:
            s = raw[raw_col]
            if s.dtype == object:           # VARCHAR — strip thousand-separators
                s = s.astype(str).str.replace(",", "", regex=False)
            df[alias] = pd.to_numeric(s, errors="coerce")
        # Volume: may be stored as string or numeric
        vol = raw["Total Traded Quantity"]
        if vol.dtype == object:
            vol = vol.astype(str).str.replace(",", "", regex=False)
        df["Volume"] = pd.to_numeric(vol, errors="coerce").fillna(0).astype(int)
        # Drop rows with unparseable dates or zero prices/volume
        df = df.dropna(subset=["date"])
        df = df[(df["Close"] > 0) & (df["Volume"] > 0)].reset_index(drop=True)
        return df
    except Exception as exc:
        logger.warning("[V6] %s query failed: %s", symbol, exc)
        return None


def _validate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    n0 = len(df)
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    df = df.reset_index(drop=True)
    if n0 - len(df):
        logger.debug("%s: dropped %d duplicate rows", label, n0 - len(df))
    return df


def _universe_ok(df: pd.DataFrame, symbol: str) -> bool:
    if len(df) < MIN_TRADING_ROWS:
        return False
    if df["Close"].iloc[-1] < MIN_PRICE:
        return False
    if df["Volume"].tail(20).mean() < MIN_AVG_DAILY_VOL:
        return False
    return True


def load_index() -> pd.DataFrame:
    df = _query_index()
    df = _validate(df, INDEX_SYMBOL)
    logger.info("Index loaded: %d rows [%s -> %s]",
                len(df), df["date"].min().date(), df["date"].max().date())
    return df



def _detect_and_truncate_splits(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Detect stock splits / bonus issues from single-day price discontinuities.

    Raw NSE EOD data is unadjusted — a 2:1 split shows as a ~50% overnight
    price drop on ex-date.  Any close-to-close ratio below 0.55 (catches
    splits 2:1, 3:1, 4:1, bonus 1:1) or above 1.80 (reverse splits) is
    treated as a corporate action.  Data BEFORE the last such event is
    discarded so that EMA, RS, and base calculations all use a single
    price scale consistent with current market prices.

    Threshold rationale:
      - Legitimate single-day crashes rarely exceed -45% in NSE mid/small caps.
      - 2:1 split  → ratio ~0.50  (caught by < 0.55)
      - 3:1 split  → ratio ~0.33  (caught)
      - 1:1 bonus  → ratio ~0.50  (caught)
      - Reverse split 1:2 → ratio ~2.0  (caught by > 1.80)
    """
    if len(df) < 2:
        return df

    close    = df["Close"].values
    prev_cls = df["Close"].shift(1).values
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        ratio = close / prev_cls          # NaN for first row — ignored below

    # Find indices where ratio signals a corporate action
    import numpy as _np
    split_mask = (_np.array(ratio) < 0.55) | (_np.array(ratio) > 1.80)
    split_mask[0] = False                 # first row always NaN ratio — skip
    split_indices = _np.where(split_mask)[0]

    if len(split_indices) == 0:
        return df

    # Keep only data from the last split date onward
    last_idx = int(split_indices[-1])
    df_clean = df.iloc[last_idx:].reset_index(drop=True)
    logger.debug(
        "[SPLIT] %s: %d corporate action(s) detected; keeping data from %s "
        "(%d rows, discarded %d pre-event rows)",
        symbol, len(split_indices),
        df_clean["date"].iloc[0].date(), len(df_clean), last_idx
    )
    return df_clean


def load_stock(symbol: str) -> pd.DataFrame | None:
    df = _query_stock(symbol)
    if df is None:
        return None
    df = _validate(df, symbol)
    df = _detect_and_truncate_splits(df, symbol)  # remove pre-split history
    if not _universe_ok(df, symbol):
        return None
    return df


def load_symbols() -> list[str]:
    if not os.path.exists(SYMBOLS_CSV):
        raise FileNotFoundError(f"symbols.csv not found: {SYMBOLS_CSV}")
    syms = pd.read_csv(SYMBOLS_CSV)["Symbol"].dropna().str.strip().tolist()
    logger.info("Loaded %d symbols", len(syms))
    return syms


def align_stock_index(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """[V4] Inner-join stock and index on common trading dates via DuckDB."""
    _con.register("_stock_tmp", stock_df)
    _con.register("_index_tmp", index_df)

    stock_out = _con.execute(
        "SELECT s.date, s.Open, s.High, s.Low, s.Close, s.Volume "
        "FROM _stock_tmp s INNER JOIN _index_tmp i USING (date) ORDER BY s.date"
    ).df()
    stock_out["date"] = pd.to_datetime(stock_out["date"])

    index_out = _con.execute(
        "SELECT i.date, i.Open, i.High, i.Low, i.Close, i.Volume "
        "FROM _index_tmp i INNER JOIN _stock_tmp s USING (date) ORDER BY i.date"
    ).df()
    index_out["date"] = pd.to_datetime(index_out["date"])

    _con.unregister("_stock_tmp")
    _con.unregister("_index_tmp")
    return stock_out.reset_index(drop=True), index_out.reset_index(drop=True)


def load_all_stocks(
    symbols: list[str],
    index_df: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    result, failed = {}, 0
    for i, sym in enumerate(symbols, 1):
        if i % 50 == 0:
            logger.info("  Loaded %d / %d ...", i, len(symbols))
        stock = load_stock(sym)
        if stock is None:
            failed += 1
            continue
        s_al, i_al = align_stock_index(stock, index_df)
        if len(s_al) < MIN_TRADING_ROWS:
            failed += 1
            continue
        result[sym] = (s_al, i_al)
    logger.info("Universe ready: %d symbols (%d skipped)", len(result), failed)
    return result


def fast_stock_summary(symbol: str) -> dict | None:
    """Lightweight pre-filter using pandas on raw parquet data."""
    pattern = _parquet_glob(symbol)
    pq_files = glob.glob(pattern.replace("/", os.sep))
    if not pq_files:
        return None
    try:
        sql = (
            "SELECT \"Close Price\", \"Total Traded Quantity\" "
            f"FROM read_parquet('{pattern}', union_by_name=true)"
        )
        raw = _con.execute(sql).df()
        if raw.empty:
            return None
        cp = raw["Close Price"]
        if cp.dtype == object:
            cp = pd.to_numeric(cp.astype(str).str.replace(",", "", regex=False), errors="coerce")
        else:
            cp = pd.to_numeric(cp, errors="coerce")
        vol = raw["Total Traded Quantity"]
        if vol.dtype == object:
            vol = pd.to_numeric(vol.astype(str).str.replace(",", "", regex=False), errors="coerce")
        else:
            vol = pd.to_numeric(vol, errors="coerce")
        mask = (cp > 0) & (vol > 0)
        cp  = cp[mask]
        vol = vol[mask]
        n = len(cp)
        last_c = float(cp.iloc[-1]) if n else 0.0
        avg_v  = float(vol.mean())  if n else 0.0
        if n < MIN_TRADING_ROWS or last_c < MIN_PRICE or avg_v < MIN_AVG_DAILY_VOL:
            return None
        return {"symbol": symbol, "n_rows": n, "last_close": last_c, "avg_vol": avg_v}
    except Exception as exc:
        logger.debug("fast_summary %s: %s", symbol, exc)
        return None
