"""
Phase 1D — Structured Bulk Deal Collector (Robust Version)

✔ Handles variable Screener column names
✔ Future-proof HTML parsing
✔ Safe DuckDB insertion
"""

import duckdb
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from io import StringIO


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DB_PATH = Path("data/ownership.duckdb")
BULK_URL = "https://www.screener.in/screens/343087/bulk-deals/"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------
def get_conn():
    return duckdb.connect(DB_PATH)


def ensure_table():
    conn = get_conn()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS bulk_block_deals (
            symbol TEXT,
            date DATE,
            side TEXT,
            qty DOUBLE,
            price DOUBLE,
            participant TEXT
        )
    """)

    conn.close()


# ---------------------------------------------------------------------
# FETCH TABLE
# ---------------------------------------------------------------------
def fetch_bulk_table():

    r = requests.get(BULK_URL, headers=HEADERS, timeout=15)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))

    if not tables:
        raise Exception("No tables found on bulk-deal page")

    return tables[0]


# ---------------------------------------------------------------------
# COLUMN NORMALIZATION
# ---------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame):

    df.columns = [c.strip().lower() for c in df.columns]

    # Print columns once for debugging clarity
    print("Detected columns:", list(df.columns))

    column_map = {}

    for col in df.columns:

        if "stock" in col or "company" in col:
            column_map[col] = "symbol"

        elif "date" in col:
            column_map[col] = "date"

        elif "qty" in col or "quantity" in col:
            column_map[col] = "qty"

        elif "price" in col:
            column_map[col] = "price"

        elif "buy" in col or "sell" in col or "action" in col:
            column_map[col] = "side"

        elif "buyer" in col or "seller" in col or "client" in col:
            column_map[col] = "participant"

    df = df.rename(columns=column_map)

    required = ["symbol", "date", "qty", "price"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Still missing required column: {col}")

    if "side" not in df.columns:
        df["side"] = ""

    if "participant" not in df.columns:
        df["participant"] = ""

    return df


# ---------------------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------------------
def transform_df(df: pd.DataFrame):

    df = normalize_columns(df)

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.date

    df = df.dropna(subset=["qty", "price"])

    return df[["symbol", "date", "side", "qty", "price", "participant"]]


# ---------------------------------------------------------------------
# INSERT
# ---------------------------------------------------------------------
def insert_into_db(df: pd.DataFrame):

    conn = get_conn()

    conn.register("temp_bulk", df)

    conn.execute("""
        INSERT INTO bulk_block_deals
        SELECT * FROM temp_bulk
    """)

    conn.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def run_structured_bulk_ingestion():

    print("\n====== STRUCTURED BULK DEAL INGESTION ======\n")

    ensure_table()

    try:
        df_raw = fetch_bulk_table()
        print("Fetched rows:", len(df_raw))

        df_clean = transform_df(df_raw)
        print("Clean rows:", len(df_clean))

        insert_into_db(df_clean)

        print("\n✔ Bulk-deal data stored in DuckDB.")

    except Exception as e:
        print("ERROR:", e)

    print("\n============================================\n")


# ---------------------------------------------------------------------
# ENTRY
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_structured_bulk_ingestion()
