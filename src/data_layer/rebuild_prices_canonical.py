"""
Rebuild Canonical Prices Table → prices_daily_v1
Uses symbol_master as universe authority.
Versioned institutional data layer (P2).
"""

from __future__ import annotations

import argparse
import duckdb
from pathlib import Path


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DB_PATH = Path("data/ownership.duckdb")
SOURCE_TABLE = "prices_daily"       # legacy table
TARGET_TABLE = "prices_daily_v1"    # canonical version


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Rebuild canonical prices table prices_daily_v1")
    p.add_argument("--db-path", default="data/ownership.duckdb")
    return p.parse_args(argv)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main(argv=None, db_path: str | Path | None = None):

    print("\n===== REBUILDING CANONICAL PRICES (v1) =====\n")

    if db_path is None:
        if argv is not None:
            args = parse_args(argv)
            db_path = args.db_path
        else:
            db_path = DB_PATH

    conn = duckdb.connect(str(db_path))
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
    import sys

    main(argv=sys.argv[1:])
