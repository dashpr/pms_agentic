"""
R2 Institutional Delivery Rebuild (Parallel)

• Full RAW wipe
• Parallel NSE download
• Canonical parsing using symbol_master
• Writes delivery_daily_v1
• Produces audit summary

FINAL INFRA SCRIPT BEFORE DATA FREEZE
"""

import duckdb
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import os


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DB_PATH = Path("data/ownership.duckdb")

RAW_DIR = Path("data_raw/nse_delivery")
BASE_URL = "https://archives.nseindia.com/archives/equities/mto/MTO_{date}.DAT"

START_DATE = datetime(2015, 1, 1)
END_DATE = datetime.today()

MAX_WORKERS = 8
MIN_FILE_BYTES = 200  # basic integrity check


# ---------------------------------------------------------------------
# UTIL
# ---------------------------------------------------------------------
def trading_days(start, end):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


# ---------------------------------------------------------------------
# RAW WIPE (R2 RULE)
# ---------------------------------------------------------------------
def reset_raw_folder():
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# DOWNLOAD ONE FILE
# ---------------------------------------------------------------------
def download_day(date):

    url = BASE_URL.format(date=date.strftime("%d%m%Y"))
    out_file = RAW_DIR / f"{date.strftime('%Y-%m-%d')}.DAT"

    try:
        r = requests.get(url, timeout=10)

        if r.status_code != 200 or len(r.content) < MIN_FILE_BYTES:
            return None

        with open(out_file, "wb") as f:
            f.write(r.content)

        return out_file

    except Exception:
        return None


# ---------------------------------------------------------------------
# PARSE NSE DAT → DataFrame
# TRUE STRUCTURE:
# col2 = SYMBOL
# col6 = DELIVERY %
# ---------------------------------------------------------------------
def parse_file(path):

    rows = []
    date = datetime.strptime(path.stem, "%Y-%m-%d").date()

    with open(path, "r", errors="ignore") as f:
        for line in f:

            parts = [p.strip() for p in line.split(",")]

            if len(parts) >= 7:
                try:
                    symbol = parts[2]
                    delivery_pct = float(parts[6])
                except:
                    continue

                rows.append((symbol, date, delivery_pct))

    if not rows:
        return None

    return pd.DataFrame(rows, columns=["canonical_symbol", "date", "delivery_pct"])


# ---------------------------------------------------------------------
# MAIN REBUILD
# ---------------------------------------------------------------------
def main():

    print("\n===== R2 PARALLEL DELIVERY REBUILD START =====\n")

    # Step-1: RAW reset
    reset_raw_folder()
    print("RAW folder reset complete.")

    # Step-2: Parallel download
    days = list(trading_days(START_DATE, END_DATE))

    print(f"Downloading {len(days)} trading days in parallel...\n")

    downloaded_files = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(download_day, d): d for d in days}

        for f in as_completed(futures):
            result = f.result()
            if result:
                downloaded_files.append(result)

    print(f"Files downloaded: {len(downloaded_files)}")

    # Step-3: Parse all files
    dfs = []

    for f in downloaded_files:
        df = parse_file(f)
        if df is not None:
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No delivery data parsed.")

    full_df = pd.concat(dfs, ignore_index=True)

    print(f"Parsed rows: {len(full_df)}")

    # Step-4: Canonical filter via symbol_master
    conn = duckdb.connect(DB_PATH)

    conn.execute("DROP TABLE IF EXISTS delivery_daily_v1")

    conn.register("temp_df", full_df)

    conn.execute("""
        CREATE TABLE delivery_daily_v1 AS
        SELECT t.*
        FROM temp_df t
        JOIN symbol_master s
          ON t.canonical_symbol = s.canonical_symbol
        WHERE s.in_universe = TRUE
    """)

    # Step-5: Audit summary
    rows = conn.execute(
        "SELECT COUNT(*) FROM delivery_daily_v1"
    ).fetchone()[0]

    symbols = conn.execute(
        "SELECT COUNT(DISTINCT canonical_symbol) FROM delivery_daily_v1"
    ).fetchone()[0]

    min_date, max_date = conn.execute(
        "SELECT MIN(date), MAX(date) FROM delivery_daily_v1"
    ).fetchone()

    conn.close()

    print("\n===== DELIVERY REBUILD COMPLETE =====")
    print(f"Rows in delivery_daily_v1 : {rows}")
    print(f"Symbols covered           : {symbols}")
    print(f"Date range                : {min_date} → {max_date}")
    print("=====================================\n")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
