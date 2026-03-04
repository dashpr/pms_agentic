"""
Backfill NIFTY universe OHLCV CSVs from yfinance.

Designed to fill missing symbols in data/csvs based on symbol_master.
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import requests


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
    p.add_argument(
        "--prefer-nse-bhavcopy",
        action="store_true",
        help="Use NSE bhavcopy fallback for refresh-existing stale symbols before yfinance.",
    )
    p.add_argument(
        "--nse-bhavcopy-lookback-days",
        type=int,
        default=14,
        help="Lookback window for NSE bhavcopy fallback updates.",
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


def _download_nse_bhavcopy_csv(
    session: requests.Session,
    d: date,
) -> pd.DataFrame:
    ymd = d.strftime("%Y%m%d")
    year = d.strftime("%Y")
    mon = d.strftime("%b").upper()
    ddmonyyyy = d.strftime("%d%b%Y").upper()
    urls = [
        f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip",
        f"https://archives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip",
        f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{year}/{mon}/cm{ddmonyyyy}bhav.csv.zip",
        f"https://archives.nseindia.com/content/historical/EQUITIES/{year}/{mon}/cm{ddmonyyyy}bhav.csv.zip",
    ]
    for u in urls:
        try:
            r = session.get(u, timeout=20)
            if r.status_code != 200 or not r.content:
                continue
            raw = r.content
            if zipfile.is_zipfile(io.BytesIO(raw)):
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    csv_names = [n for n in zf.namelist() if str(n).lower().endswith(".csv")]
                    if not csv_names:
                        continue
                    txt = zf.read(csv_names[0]).decode("utf-8", errors="ignore")
            else:
                txt = raw.decode("utf-8", errors="ignore")
            if not txt:
                continue
            x = pd.read_csv(io.StringIO(txt))
            if x.empty:
                continue
            x.columns = [str(c).strip().upper() for c in x.columns]
            if "SYMBOL" not in x.columns or "CLOSE" not in x.columns:
                continue
            if "SERIES" in x.columns:
                x = x[x["SERIES"].astype(str).str.upper().eq("EQ")].copy()
            x["SYMBOL"] = x["SYMBOL"].astype(str).str.upper().str.strip()
            x["OPEN"] = pd.to_numeric(x.get("OPEN"), errors="coerce")
            x["HIGH"] = pd.to_numeric(x.get("HIGH"), errors="coerce")
            x["LOW"] = pd.to_numeric(x.get("LOW"), errors="coerce")
            x["CLOSE"] = pd.to_numeric(x.get("CLOSE"), errors="coerce")
            vol_col = "TOTTRDQTY" if "TOTTRDQTY" in x.columns else ("VOLUME" if "VOLUME" in x.columns else None)
            x["VOLUME"] = pd.to_numeric(x.get(vol_col), errors="coerce") if vol_col else 0.0
            x["DATE"] = pd.to_datetime(d).date()
            x = x.dropna(subset=["SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE"]).copy()
            if x.empty:
                continue
            return x[["SYMBOL", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
        except Exception:
            continue
    return pd.DataFrame(columns=["SYMBOL", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])


def _update_from_nse_bhavcopy(
    universe_df: pd.DataFrame,
    out_dir: Path,
    target_symbols: list[str],
    end_dt: date,
    lookback_days: int,
) -> tuple[int, int]:
    if not target_symbols:
        return (0, 0)
    target = sorted({str(s).strip().upper() for s in target_symbols if str(s).strip()})
    if not target:
        return (0, 0)
    map_row = universe_df.copy()
    map_row["canonical_symbol"] = map_row["canonical_symbol"].astype(str).str.upper().str.strip()
    map_row["nse_symbol"] = map_row["nse_symbol"].astype(str).str.upper().str.strip()
    sym_to_nse = map_row.drop_duplicates(subset=["canonical_symbol"], keep="first").set_index("canonical_symbol")["nse_symbol"].to_dict()
    nse_to_sym = {}
    for cs in target:
        ns = str(sym_to_nse.get(cs, cs)).upper().strip()
        if ns:
            nse_to_sym[ns] = cs
    if not nse_to_sym:
        return (0, 0)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,application/octet-stream,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    sess = requests.Session()
    sess.headers.update(headers)

    start_dt = end_dt - pd.Timedelta(days=max(1, int(lookback_days)))
    days = pd.date_range(start=start_dt, end=end_dt, freq="D")
    bh_rows = []
    for d in days:
        z = _download_nse_bhavcopy_csv(sess, d.date())
        if z.empty:
            continue
        bh_rows.append(z)
    if not bh_rows:
        return (0, 0)

    bh = pd.concat(bh_rows, ignore_index=True)
    bh = bh[bh["SYMBOL"].isin(set(nse_to_sym.keys()))].copy()
    if bh.empty:
        return (0, 0)
    bh["canonical_symbol"] = bh["SYMBOL"].map(nse_to_sym)
    bh = bh.dropna(subset=["canonical_symbol"]).copy()
    if bh.empty:
        return (0, 0)

    updated_symbols = 0
    inserted_rows = 0
    for sym, g in bh.groupby("canonical_symbol"):
        fp = out_dir / f"{sym}.csv"
        ex = _read_existing_csv(fp, cutoff_date=end_dt)
        old_rows = len(ex)
        add = g.rename(
            columns={
                "DATE": "date",
                "OPEN": "open",
                "HIGH": "high",
                "LOW": "low",
                "CLOSE": "close",
                "VOLUME": "volume",
            }
        )[["date", "open", "high", "low", "close", "volume"]].copy()
        add["date"] = pd.to_datetime(add["date"], errors="coerce").dt.date
        for c in ["open", "high", "low", "close", "volume"]:
            add[c] = pd.to_numeric(add[c], errors="coerce")
        add = add.dropna(subset=["date", "open", "high", "low", "close"]).copy()
        if add.empty:
            continue
        merged = pd.concat([ex, add], ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.date
        for c in ["open", "high", "low", "close", "volume"]:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
        merged = (
            merged.dropna(subset=["date", "open", "high", "low", "close"])
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
        )
        new_rows = len(merged)
        if new_rows > old_rows or (not ex.empty and not merged.equals(ex)):
            merged.to_csv(fp, index=False)
            updated_symbols += 1
            inserted_rows += max(new_rows - old_rows, 0)
    return (updated_symbols, inserted_rows)


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

    if args.refresh_existing and bool(args.prefer_nse_bhavcopy) and not missing_df.empty:
        nse_updated, nse_inserted = _update_from_nse_bhavcopy(
            universe_df=universe_df,
            out_dir=out_dir,
            target_symbols=missing_df["canonical_symbol"].astype(str).tolist(),
            end_dt=end_dt,
            lookback_days=int(args.nse_bhavcopy_lookback_days),
        )
        if nse_updated > 0:
            refreshed_rows = []
            for row in missing_df.itertuples(index=False):
                sym = str(row.canonical_symbol).strip().upper()
                fp = out_dir / f"{sym}.csv"
                ex = _read_existing_csv(fp, cutoff_date=end_dt)
                if ex.empty:
                    refreshed_rows.append({"canonical_symbol": sym, "nse_symbol": str(row.nse_symbol).strip().upper()})
                    continue
                last_local = pd.to_datetime(ex["date"]).max().date()
                if (end_dt - last_local).days > int(args.stale_days):
                    refreshed_rows.append({"canonical_symbol": sym, "nse_symbol": str(row.nse_symbol).strip().upper()})
            missing_df = pd.DataFrame(refreshed_rows) if refreshed_rows else pd.DataFrame(columns=["canonical_symbol", "nse_symbol"])
            missing = missing_df["canonical_symbol"].tolist() if not missing_df.empty else []
            print("nse_bhavcopy_pre_refresh_updated_symbols", int(nse_updated))
            print("nse_bhavcopy_pre_refresh_inserted_rows", int(nse_inserted))
            print("remaining_after_nse_bhavcopy", len(missing))

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

    # Final NSE bhavcopy fallback for unresolved symbols.
    unresolved = sorted(set([str(x[0]).strip().upper() for x in fail]))
    if unresolved:
        nse_updated_f, nse_inserted_f = _update_from_nse_bhavcopy(
            universe_df=universe_df,
            out_dir=out_dir,
            target_symbols=unresolved,
            end_dt=end_dt,
            lookback_days=int(args.nse_bhavcopy_lookback_days),
        )
        print("nse_bhavcopy_post_fallback_updated_symbols", int(nse_updated_f))
        print("nse_bhavcopy_post_fallback_inserted_rows", int(nse_inserted_f))

    print("\n===== BACKFILL SUMMARY =====")
    print("downloaded", ok)
    print("failed", len(fail))
    print("short_history", len(short))
    print("fail_report", fail_path)
    print("short_report", short_path)
    print("downloaded_report", used_path)


if __name__ == "__main__":
    main()
