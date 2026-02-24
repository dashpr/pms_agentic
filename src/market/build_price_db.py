"""
Canonical Market Price Loader -> DuckDB.

Builds `prices_daily` table from local CSV universe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build prices_daily from CSV universe")
    p.add_argument("--data-folder", default="data/csvs")
    p.add_argument("--db-path", default="data/ownership.duckdb")
    return p.parse_args(argv)


def load_csv(file_path: Path):
    df = pd.read_csv(file_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # yfinance CSVs may be written with a 3-line header:
    # Price,Close,High,Low,Open,Volume
    # Ticker,XXX,...
    # Date,,,,,
    # In this shape, actual data starts after the "Date" marker row.
    if "date" not in df.columns:
        first_col = df.columns[0] if len(df.columns) else None
        if first_col is None:
            raise ValueError(f"{file_path.name} is empty")

        marker = df[first_col].astype(str).str.strip().str.lower().eq("date")
        if not marker.any():
            raise ValueError(f"{file_path.name} missing date header/marker")

        marker_label = marker[marker].index[0]
        marker_pos = df.index.get_loc(marker_label)
        df = df.iloc[marker_pos + 1 :].copy()
        df.columns = ["date"] + [str(c).strip().lower() for c in list(df.columns[1:])]

    if "close" not in df.columns and "adj close" in df.columns:
        df["close"] = df["adj close"]

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name} missing columns: {', '.join(missing)}")

    out = df[required].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out[out["date"] <= pd.Timestamp.utcnow().date()]
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out["volume"] = out["volume"].fillna(0.0)
    out["symbol"] = file_path.stem.upper()

    return out[["symbol", "date", "open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------
# BUILD DATABASE
# ---------------------------------------------------------------------
def build_price_db(
    data_folder: str | Path = "data/csvs",
    db_path: str | Path = "data/ownership.duckdb",
):

    print("\n===== BUILDING CANONICAL PRICE DATABASE =====\n")

    data_folder = Path(data_folder)
    db_path = Path(db_path)
    data_folder.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(db_path)
    temp_table = "prices_daily__new"

    conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
    conn.execute(f"""
        CREATE TABLE {temp_table} (
            symbol TEXT,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        )
    """)

    total_rows = 0
    success_files = 0
    skipped_files = 0
    files = list(data_folder.glob("*.csv"))

    print("CSV folder:", data_folder)
    print("CSV files found:", len(files), "\n")

    for f in files:
        try:
            df = load_csv(f)

            conn.register("temp_df", df)
            conn.execute(f"INSERT INTO {temp_table} SELECT * FROM temp_df")

            rows = len(df)
            total_rows += rows
            success_files += 1

            print(f"{f.name} → {rows} rows inserted")

        except Exception as e:
            skipped_files += 1
            print(f"{f.name} → skipped ({e})")

    if total_rows == 0:
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        conn.close()
        raise ValueError("No rows parsed from CSV folder; aborted prices_daily replacement.")

    conn.execute("DROP TABLE IF EXISTS prices_daily")
    conn.execute(f"ALTER TABLE {temp_table} RENAME TO prices_daily")
    conn.close()

    print("\n===== PRICE DATABASE COMPLETE =====")
    print("Files processed:", len(files))
    print("Files loaded:", success_files)
    print("Files skipped:", skipped_files)
    print("Total rows inserted:", total_rows)
    print("===================================\n")


def main(argv=None):
    args = parse_args(argv)
    build_price_db(data_folder=str(args.data_folder), db_path=str(args.db_path))


if __name__ == "__main__":
    main()
