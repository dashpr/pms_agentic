"""
Backfill NIFTY universe OHLCV CSVs from yfinance.

Designed to fill missing symbols in data/csvs based on symbol_master.
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Backfill NIFTY universe OHLCV CSVs")
    p.add_argument("--db-path", default="data/ownership.duckdb")
    p.add_argument("--out-dir", default="data/csvs")
    p.add_argument("--start-date", default="2010-01-01")
    p.add_argument("--end-date", default=None)
    p.add_argument("--min-trading-days", type=int, default=2520, help="~10y minimum rows")
    p.add_argument("--max-symbols", type=int, default=0, help="0 = no cap")
    p.add_argument("--sleep-ms", type=int, default=250)
    p.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Refresh existing CSVs that are stale vs end-date/today.",
    )
    p.add_argument(
        "--stale-days",
        type=int,
        default=2,
        help="If last local date is older than this many days from end-date/today, refresh it.",
    )
    p.add_argument(
        "--include-all-symbol-master",
        action="store_true",
        help="Use all symbol_master entries (in_universe true/false).",
    )
    p.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Force full-history overwrite for all selected symbols.",
    )
    p.add_argument(
        "--allow-fallback-exchanges",
        action="store_true",
        help="Try .BO and bare ticker if .NS fails. Disabled by default for cleaner NSE history.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _load_universe(db_path: str, include_all_symbol_master: bool = False) -> pd.DataFrame:
    where = "" if include_all_symbol_master else "WHERE in_universe=true"
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            """
            select
                upper(trim(canonical_symbol)) as canonical_symbol,
                upper(trim(coalesce(nullif(nse_symbol, ''), canonical_symbol))) as nse_symbol
            from symbol_master
            """
            + where
            + """
            order by canonical_symbol
            """
        ).fetchdf()
    finally:
        con.close()
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    need = ["date", "open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in need):
        return pd.DataFrame()

    out = df[need].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    return out


def _read_existing_csv(path: Path, cutoff_date: date | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    try:
        df = pd.read_csv(path)
        out = _normalize(df)
        if cutoff_date is not None and not out.empty:
            out = out[pd.to_datetime(out["date"]).dt.date <= cutoff_date].copy()
        return out
    except Exception:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])


def _end_date_or_today(end_date: str | None) -> date:
    if end_date:
        return pd.to_datetime(end_date).date()
    return pd.Timestamp.utcnow().date()


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    universe_df = _load_universe(args.db_path, include_all_symbol_master=bool(args.include_all_symbol_master))
    have = {p.stem.upper() for p in out_dir.glob("*.csv")}
    universe = universe_df["canonical_symbol"].tolist()
    end_dt = _end_date_or_today(args.end_date)

    if args.overwrite_existing:
        missing_df = universe_df.copy()
    elif args.refresh_existing:
        refresh_rows = []
        for row in universe_df.itertuples(index=False):
            sym = str(row.canonical_symbol).strip().upper()
            fp = out_dir / f"{sym}.csv"
            ex = _read_existing_csv(fp, cutoff_date=end_dt)
            if ex.empty:
                refresh_rows.append({"canonical_symbol": sym, "nse_symbol": str(row.nse_symbol).strip().upper()})
                continue
            last_local = pd.to_datetime(ex["date"]).max().date()
            if (end_dt - last_local).days > int(args.stale_days):
                refresh_rows.append({"canonical_symbol": sym, "nse_symbol": str(row.nse_symbol).strip().upper()})
        missing_df = pd.DataFrame(refresh_rows)
        if missing_df.empty:
            missing_df = pd.DataFrame(columns=["canonical_symbol", "nse_symbol"])
    else:
        missing_df = universe_df[~universe_df["canonical_symbol"].isin(have)].copy()
    missing = missing_df["canonical_symbol"].tolist() if not missing_df.empty else []

    if args.max_symbols and args.max_symbols > 0:
        missing_df = missing_df.iloc[: args.max_symbols].copy()
        missing = missing_df["canonical_symbol"].tolist()

    report = pd.DataFrame({"symbol": missing})
    report_path = Path("data/reports/missing_universe_symbols_v1.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print("universe_symbols", len(universe))
    print("existing_csv", len(have))
    print("missing_symbols", len(missing))
    print("missing_report", report_path)

    if args.dry_run:
        return

    import yfinance as yf

    ok = 0
    fail = []
    short = []
    used_tickers = []

    for i, row in enumerate(missing_df.itertuples(index=False), start=1):
        sym = str(row.canonical_symbol).strip().upper()
        nse_sym = str(row.nse_symbol).strip().upper()
        out_path = out_dir / f"{sym}.csv"
        existing = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        if not args.overwrite_existing:
            existing = _read_existing_csv(out_path, cutoff_date=end_dt)
        fetch_start = args.start_date
        if args.refresh_existing and (not args.overwrite_existing) and not existing.empty:
            last_local = pd.to_datetime(existing["date"]).max()
            fetch_start = (last_local - pd.Timedelta(days=7)).date().isoformat()

        candidate_bases = []
        for c in [nse_sym, sym]:
            if c and c != "NAN" and c not in candidate_bases:
                candidate_bases.append(c)

        selected_df = pd.DataFrame()
        selected_ticker = None
        last_error = "empty"

        suffixes = [".NS"] if not args.allow_fallback_exchanges else [".NS", ".BO", ""]
        for base in candidate_bases:
            for suffix in suffixes:
                ticker = f"{base}{suffix}"
                try:
                    df = yf.download(
                        ticker,
                        start=fetch_start,
                        end=args.end_date,
                        progress=False,
                        auto_adjust=False,
                    )
                    df = _normalize(df.reset_index())
                    if not df.empty:
                        selected_df = df
                        selected_ticker = ticker
                        break
                except Exception as exc:
                    last_error = str(exc)
            if selected_ticker is not None:
                break

        if selected_df.empty:
            fail.append((sym, nse_sym, last_error))
            continue

        if len(selected_df) < int(args.min_trading_days):
            short.append((sym, len(selected_df), selected_ticker))

        if not existing.empty:
            merged = pd.concat([existing, selected_df], ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.date
            for c in ["open", "high", "low", "close", "volume"]:
                merged[c] = pd.to_numeric(merged[c], errors="coerce")
            selected_df = (
                merged.dropna(subset=["date", "open", "high", "low", "close"])
                .sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
            )

        selected_df.to_csv(out_path, index=False)
        used_tickers.append((sym, selected_ticker))
        ok += 1

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

        if i % 10 == 0:
            print("done", i, "ok", ok, "fail", len(fail), "short", len(short))

    fail_df = pd.DataFrame(fail, columns=["symbol", "nse_symbol", "reason"])
    short_df = pd.DataFrame(short, columns=["symbol", "rows", "ticker"])
    used_df = pd.DataFrame(used_tickers, columns=["symbol", "ticker"])
    fail_path = Path("data/reports/backfill_failures_v1.csv")
    short_path = Path("data/reports/backfill_short_history_v1.csv")
    used_path = Path("data/reports/backfill_downloaded_tickers_v1.csv")
    fail_df.to_csv(fail_path, index=False)
    short_df.to_csv(short_path, index=False)
    used_df.to_csv(used_path, index=False)

    print("\n===== BACKFILL SUMMARY =====")
    print("downloaded", ok)
    print("failed", len(fail))
    print("short_history", len(short))
    print("fail_report", fail_path)
    print("short_report", short_path)
    print("downloaded_report", used_path)


if __name__ == "__main__":
    main()
