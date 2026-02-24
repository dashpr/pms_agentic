"""
Rebuild Canonical Delivery Table from Local NSE Raw Files (v1)

Source: data_raw/nse_delivery/*.DAT
Target: delivery_daily_v1 (canonical_symbol, date, delivery_pct)

Institutional safeguards:
- Strict record parsing for row type 20
- Atomic table replacement (temp -> swap)
- Delivery bounds validation (0-100)
- Universe coverage checks before replace
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


DB_PATH = Path("data/ownership.duckdb")
RAW_DIR = Path("data_raw/nse_delivery")
TARGET_TABLE = "delivery_daily_v1"


@dataclass
class RebuildConfig:
    workers: int = 8
    min_symbols: int = 80
    min_rows: int = 100_000


def _extract_date_from_path(path: Path) -> datetime.date:
    return datetime.strptime(path.stem, "%Y-%m-%d").date()


def parse_delivery_file(path: Path) -> pd.DataFrame:
    rows = []
    trade_date = _extract_date_from_path(path)

    with path.open("r", errors="ignore") as handle:
        for raw in handle:
            parts = [p.strip() for p in raw.split(",")]
            # Record format:
            # 20,SrNo,Symbol,Series,QtyTraded,DeliverableQty,DeliveryPct
            if len(parts) < 7 or parts[0] != "20":
                continue

            symbol = parts[2].upper()
            if not symbol:
                continue

            try:
                delivery_pct = float(parts[6])
            except ValueError:
                continue

            if delivery_pct < 0 or delivery_pct > 100:
                continue

            rows.append((symbol, trade_date, delivery_pct))

    if not rows:
        return pd.DataFrame(columns=["canonical_symbol", "date", "delivery_pct"])

    return pd.DataFrame(rows, columns=["canonical_symbol", "date", "delivery_pct"])


def parse_all_files(files: Iterable[Path], workers: int) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(parse_delivery_file, f): f for f in files}
        for future in as_completed(futures):
            chunk = future.result()
            if not chunk.empty:
                chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=["canonical_symbol", "date", "delivery_pct"])

    return pd.concat(chunks, ignore_index=True)


def rebuild_delivery_from_raw(config: RebuildConfig) -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw delivery folder missing: {RAW_DIR}")

    files = sorted(RAW_DIR.glob("*.DAT"))
    if not files:
        raise FileNotFoundError(f"No .DAT files found in: {RAW_DIR}")

    print("\n===== REBUILD DELIVERY DAILY V1 FROM RAW =====\n")
    print(f"Files discovered: {len(files)}")

    parsed = parse_all_files(files, workers=config.workers)
    if parsed.empty:
        raise ValueError("Parsed delivery dataset is empty")

    parsed["canonical_symbol"] = parsed["canonical_symbol"].astype(str).str.strip().str.upper()
    parsed["date"] = pd.to_datetime(parsed["date"]).dt.date

    conn = duckdb.connect(DB_PATH)
    try:
        universe = conn.execute(
            """
            SELECT canonical_symbol
            FROM symbol_master
            WHERE in_universe = TRUE
            """
        ).fetchdf()
        if universe.empty:
            raise ValueError("symbol_master has no active universe symbols")

        parsed = parsed.merge(universe, on="canonical_symbol", how="inner")
        if parsed.empty:
            raise ValueError("No parsed delivery rows matched symbol_master universe")

        temp_table = f"{TARGET_TABLE}__new"
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        conn.register("delivery_raw_df", parsed)
        conn.execute(
            f"""
            CREATE TABLE {temp_table} AS
            SELECT
                canonical_symbol,
                date,
                AVG(delivery_pct) AS delivery_pct
            FROM delivery_raw_df
            GROUP BY canonical_symbol, date
            """
        )
        conn.unregister("delivery_raw_df")

        rows = conn.execute(f"SELECT COUNT(*) FROM {temp_table}").fetchone()[0]
        symbols = conn.execute(
            f"SELECT COUNT(DISTINCT canonical_symbol) FROM {temp_table}"
        ).fetchone()[0]
        min_date, max_date = conn.execute(
            f"SELECT MIN(date), MAX(date) FROM {temp_table}"
        ).fetchone()

        if rows < config.min_rows:
            raise ValueError(
                f"{temp_table} row count too low ({rows} < {config.min_rows})"
            )
        if symbols < config.min_symbols:
            raise ValueError(
                f"{temp_table} symbol count too low ({symbols} < {config.min_symbols})"
            )

        conn.execute(f"DROP TABLE IF EXISTS {TARGET_TABLE}")
        conn.execute(f"ALTER TABLE {temp_table} RENAME TO {TARGET_TABLE}")

        print(f"Rows in {TARGET_TABLE} : {rows}")
        print(f"Symbols covered        : {symbols}")
        print(f"Date range             : {min_date} -> {max_date}")
        print("\n===== REBUILD COMPLETE =====\n")
    finally:
        conn.close()


def main() -> None:
    rebuild_delivery_from_raw(RebuildConfig())


if __name__ == "__main__":
    main()
