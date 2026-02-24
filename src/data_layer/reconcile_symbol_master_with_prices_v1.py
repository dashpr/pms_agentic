"""
Reconcile symbol_master with available local price history.

Institutional intent:
- Promote symbols with actual OHLCV history into in_universe=true
- Demote symbols with no OHLCV history to in_universe=false
- Add missing symbols from prices_daily into symbol_master
"""

from __future__ import annotations

from pathlib import Path

import duckdb


DB_PATH = Path("data/ownership.duckdb")
REPORT_DIR = Path("data/reports")
DATA_VERSION = "reconciled_v1"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(DB_PATH)
    try:
        before_universe = con.execute(
            "SELECT COUNT(*) FROM symbol_master WHERE in_universe=TRUE"
        ).fetchone()[0]
        before_master = con.execute("SELECT COUNT(*) FROM symbol_master").fetchone()[0]

        con.execute("DROP TABLE IF EXISTS __price_symbols")
        con.execute(
            """
            CREATE TEMP TABLE __price_symbols AS
            SELECT DISTINCT UPPER(TRIM(symbol)) AS symbol
            FROM prices_daily
            WHERE symbol IS NOT NULL AND TRIM(symbol) <> ''
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE __deactivated AS
            SELECT canonical_symbol
            FROM symbol_master
            WHERE in_universe=TRUE
              AND canonical_symbol NOT IN (SELECT symbol FROM __price_symbols)
            ORDER BY canonical_symbol
            """
        )

        con.execute(
            """
            UPDATE symbol_master
            SET
                in_universe=TRUE,
                is_active=TRUE,
                nse_symbol=COALESCE(NULLIF(TRIM(nse_symbol), ''), canonical_symbol),
                data_version=?
            WHERE canonical_symbol IN (SELECT symbol FROM __price_symbols)
            """,
            [DATA_VERSION],
        )

        con.execute(
            """
            UPDATE symbol_master
            SET
                in_universe=FALSE,
                data_version=?
            WHERE canonical_symbol NOT IN (SELECT symbol FROM __price_symbols)
              AND in_universe=TRUE
            """,
            [DATA_VERSION],
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE __added AS
            SELECT p.symbol AS canonical_symbol
            FROM __price_symbols p
            LEFT JOIN symbol_master s
              ON s.canonical_symbol = p.symbol
            WHERE s.canonical_symbol IS NULL
            ORDER BY p.symbol
            """
        )

        con.execute(
            """
            INSERT INTO symbol_master (
                canonical_symbol,
                nse_symbol,
                series,
                sector,
                is_active,
                in_universe,
                start_date,
                end_date,
                data_version,
                created_at
            )
            SELECT
                canonical_symbol,
                canonical_symbol,
                'EQ',
                NULL,
                TRUE,
                TRUE,
                DATE '2000-01-01',
                NULL,
                ?,
                NOW()
            FROM __added
            """,
            [DATA_VERSION],
        )

        after_universe = con.execute(
            "SELECT COUNT(*) FROM symbol_master WHERE in_universe=TRUE"
        ).fetchone()[0]
        after_master = con.execute("SELECT COUNT(*) FROM symbol_master").fetchone()[0]

        deactivated_df = con.execute("SELECT * FROM __deactivated").fetchdf()
        added_df = con.execute("SELECT * FROM __added").fetchdf()

        deactivated_path = REPORT_DIR / "symbol_master_deactivated_v1.csv"
        added_path = REPORT_DIR / "symbol_master_added_from_prices_v1.csv"
        deactivated_df.to_csv(deactivated_path, index=False)
        added_df.to_csv(added_path, index=False)

    finally:
        con.close()

    print("\n===== SYMBOL MASTER RECONCILIATION v1 =====")
    print("master_before", before_master)
    print("universe_before", before_universe)
    print("master_after", after_master)
    print("universe_after", after_universe)
    print("deactivated_report", deactivated_path)
    print("added_report", added_path)
    print("===========================================\n")


if __name__ == "__main__":
    main()
