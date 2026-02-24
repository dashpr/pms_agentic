"""
Rebuild Canonical Delivery Table → delivery_daily_v1
Keeps FULL delivery history (D2) but enforces canonical symbols via symbol_master.
Institutional versioned data layer.
"""

import duckdb
from pathlib import Path


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DB_PATH = Path("data/ownership.duckdb")
SOURCE_TABLE = "delivery_daily"        # legacy raw delivery
TARGET_TABLE = "delivery_daily_v1"     # canonical version


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():

    print("\n===== REBUILDING CANONICAL DELIVERY (v1) =====\n")

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

    # Create canonical delivery using symbol_master filter in a temp table.
    conn.execute(f"""
        CREATE TABLE {temp_table} AS
        SELECT
            d.symbol AS canonical_symbol,
            d.date,
            d.delivery_pct
        FROM {SOURCE_TABLE} d
        JOIN symbol_master s
            ON d.symbol = s.canonical_symbol
        WHERE s.in_universe = TRUE
    """)

    # Row count
    rows = conn.execute(
        f"SELECT COUNT(*) FROM {temp_table}"
    ).fetchone()[0]

    if rows == 0:
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        conn.close()
        raise ValueError(
            "Canonical delivery rebuild produced 0 rows. "
            f"Existing {TARGET_TABLE} rows before rebuild: {existing_rows}. "
            "Aborted replacement to prevent data loss."
        )

    conn.execute(f"DROP TABLE IF EXISTS {TARGET_TABLE}")
    conn.execute(f"ALTER TABLE {temp_table} RENAME TO {TARGET_TABLE}")

    # Distinct symbols
    symbols = conn.execute(
        f"SELECT COUNT(DISTINCT canonical_symbol) FROM {TARGET_TABLE}"
    ).fetchone()[0]

    # Date range
    min_date, max_date = conn.execute(
        f"SELECT MIN(date), MAX(date) FROM {TARGET_TABLE}"
    ).fetchone()

    conn.close()

    print(f"Rows in delivery_daily_v1 : {rows}")
    print(f"Symbols covered           : {symbols}")
    print(f"Date range                : {min_date} → {max_date}")
    print("============================================\n")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
