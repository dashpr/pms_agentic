"""
Rebuild Canonical Prices Table → prices_daily_v1
Uses symbol_master as universe authority.
Versioned institutional data layer (P2).
"""

import duckdb
from pathlib import Path


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DB_PATH = Path("data/ownership.duckdb")
SOURCE_TABLE = "prices_daily"       # legacy table
TARGET_TABLE = "prices_daily_v1"    # canonical version


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():

    print("\n===== REBUILDING CANONICAL PRICES (v1) =====\n")

    conn = duckdb.connect(DB_PATH)
    temp_table = f"{TARGET_TABLE}__new"

    existing = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
        """,
        [TARGET_TABLE],
    ).fetchone()[0]
    existing_rows = 0
    if existing:
        existing_rows = conn.execute(
            f"SELECT COUNT(*) FROM {TARGET_TABLE}"
        ).fetchone()[0]

    conn.execute(f"DROP TABLE IF EXISTS {temp_table}")

    # Build into temp and replace only if non-empty.
    conn.execute(f"""
        CREATE TABLE {temp_table} AS
        SELECT
            p.symbol AS canonical_symbol,
            p.date,
            p.open,
            p.high,
            p.low,
            p.close,
            p.volume
        FROM {SOURCE_TABLE} p
        JOIN symbol_master s
            ON p.symbol = s.canonical_symbol
        WHERE s.in_universe = TRUE
    """)

    # Row count
    count = conn.execute(
        f"SELECT COUNT(*) FROM {temp_table}"
    ).fetchone()[0]

    if count == 0:
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        conn.close()
        raise ValueError(
            "Canonical prices rebuild produced 0 rows. "
            f"Existing {TARGET_TABLE} rows before rebuild: {existing_rows}. "
            "Aborted replacement to prevent data loss."
        )

    conn.execute(f"DROP TABLE IF EXISTS {TARGET_TABLE}")
    conn.execute(f"ALTER TABLE {temp_table} RENAME TO {TARGET_TABLE}")

    # Distinct symbols
    symbols = conn.execute(
        f"SELECT COUNT(DISTINCT canonical_symbol) FROM {TARGET_TABLE}"
    ).fetchone()[0]

    conn.close()

    print(f"Rows in prices_daily_v1  : {count}")
    print(f"Symbols covered          : {symbols}")
    print("===========================================\n")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
