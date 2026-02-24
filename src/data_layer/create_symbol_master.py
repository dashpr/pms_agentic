"""
Create Symbol Master — Canonical Universe Table (U2 Historical Tracking)
Robust Excel loader with automatic symbol column detection.
"""

import duckdb
from pathlib import Path
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DB_PATH = Path("data/ownership.duckdb")
UNIVERSE_FILE = Path("data/data_NSE200.xlsx")  # change if needed
DATA_VERSION = "v1"


# ---------------------------------------------------------------------
# CREATE TABLE
# ---------------------------------------------------------------------
def create_table(conn):

    conn.execute("DROP TABLE IF EXISTS symbol_master")

    conn.execute("""
        CREATE TABLE symbol_master (
            canonical_symbol TEXT PRIMARY KEY,
            nse_symbol TEXT,
            series TEXT,
            sector TEXT,
            is_active BOOLEAN,
            in_universe BOOLEAN,
            start_date DATE,
            end_date DATE,
            data_version TEXT,
            created_at TIMESTAMP
        )
    """)


# ---------------------------------------------------------------------
# AUTO-DETECT SYMBOL COLUMN
# ---------------------------------------------------------------------
def detect_symbol_column(columns):

    candidates = [
        "symbol",
        "nse_symbol",
        "security id",
        "security_id",
        "ticker",
        "code",
        "company symbol",
    ]

    cols_lower = {c.lower(): c for c in columns}

    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]

    raise ValueError(
        f"Could not find symbol column. Available columns: {list(columns)}"
    )


# ---------------------------------------------------------------------
# LOAD EXCEL
# ---------------------------------------------------------------------
def load_universe():

    df = pd.read_excel(UNIVERSE_FILE)

    # Detect symbol column automatically
    symbol_col = detect_symbol_column(df.columns)

    df = df.rename(columns={symbol_col: "canonical_symbol"})

    # Clean symbols
    df["canonical_symbol"] = (
        df["canonical_symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Default institutional assumptions
    df["nse_symbol"] = df["canonical_symbol"]
    df["series"] = "EQ"
    df["sector"] = None
    df["is_active"] = True
    df["in_universe"] = True

    # Historical tracking (U2)
    df["start_date"] = pd.to_datetime("2000-01-01").date()
    df["end_date"] = None

    df["data_version"] = DATA_VERSION
    df["created_at"] = datetime.utcnow()

    return df[
        [
            "canonical_symbol",
            "nse_symbol",
            "series",
            "sector",
            "is_active",
            "in_universe",
            "start_date",
            "end_date",
            "data_version",
            "created_at",
        ]
    ]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():

    print("\n===== CREATING SYMBOL MASTER =====\n")

    conn = duckdb.connect(DB_PATH)

    create_table(conn)

    df = load_universe()

    conn.register("temp_df", df)
    conn.execute("INSERT INTO symbol_master SELECT * FROM temp_df")

    count = conn.execute("SELECT COUNT(*) FROM symbol_master").fetchone()[0]

    conn.close()

    print(f"Symbol master created with {count} symbols.")
    print("==================================\n")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
