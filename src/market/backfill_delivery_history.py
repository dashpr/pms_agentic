"""
Historical Delivery % Backfill → FINAL CORRECT PARSER

Fixes:
✔ Correct SYMBOL extraction from NSE DAT
✔ Clean numeric delivery %
✔ Proper DB rebuild
"""

import duckdb
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DB_PATH = Path("data/ownership.duckdb")
BASE_URL = "https://archives.nseindia.com/archives/equities/mto/MTO_{date}.DAT"

START_DATE = datetime(2015, 1, 1)
END_DATE = datetime.today()

SLEEP = 0.2


# ---------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------
def get_conn():
    return duckdb.connect(DB_PATH)


def recreate_table():
    conn = get_conn()
    conn.execute("DROP TABLE IF EXISTS delivery_daily")
    conn.execute("""
        CREATE TABLE delivery_daily (
            symbol TEXT,
            date DATE,
            delivery_pct DOUBLE
        )
    """)
    conn.close()


# ---------------------------------------------------------------------
# DATE LOOP
# ---------------------------------------------------------------------
def trading_days(start, end):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


# ---------------------------------------------------------------------
# CORRECT NSE PARSER
# ---------------------------------------------------------------------
def fetch_day(date):

    url = BASE_URL.format(date=date.strftime("%d%m%Y"))

    try:
        r = requests.get(url, timeout=10)

        if r.status_code != 200 or len(r.text) < 100:
            return None

        rows = []

        for line in r.text.splitlines():

            parts = [p.strip() for p in line.split(",")]

            # Expected NSE structure:
            # SYMBOL, SERIES, ... , DELIVERY_QTY, DELIVERY_PCT
            if len(parts) >= 6:

                symbol = parts[0]  # ← correct column

                try:
                    delivery_pct = float(parts[-1])
                except:
                    continue

                rows.append((symbol, date.date(), delivery_pct))

        if not rows:
            return None

        return pd.DataFrame(rows, columns=["symbol", "date", "delivery_pct"])

    except Exception:
        return None


# ---------------------------------------------------------------------
# MAIN BACKFILL
# ---------------------------------------------------------------------
def run_backfill():

    print("\n===== DELIVERY HISTORY BACKFILL (FINAL) =====\n")

    recreate_table()
    conn = get_conn()

    total_rows = 0
    checked_days = 0

    for d in trading_days(START_DATE, END_DATE):

        checked_days += 1

        df = fetch_day(d)
        if df is None:
            continue

        conn.register("temp_df", df)
        conn.execute("INSERT INTO delivery_daily SELECT * FROM temp_df")

        rows = len(df)
        total_rows += rows

        print(f"{d.date()} → {rows} rows")

        time.sleep(SLEEP)

    conn.close()

    print("\n===== BACKFILL COMPLETE =====")
    print("Days checked :", checked_days)
    print("Rows inserted:", total_rows)
    print("=============================\n")


# ---------------------------------------------------------------------
# ENTRY
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_backfill()
