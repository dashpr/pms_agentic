from __future__ import annotations

from pathlib import Path

import duckdb


DB_PATH = Path("data/ownership.duckdb")


def main(argv=None) -> None:
    conn = duckdb.connect(str(DB_PATH))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS delivery_daily_v1 (
                canonical_symbol VARCHAR,
                date DATE,
                delivery_pct DOUBLE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS delivery_daily (
                symbol VARCHAR,
                date DATE,
                delivery_pct DOUBLE,
                volume DOUBLE,
                price DOUBLE
            )
            """
        )

        conn.execute("DROP TABLE IF EXISTS delivery_sync_stage_v1")
        conn.execute(
            """
            CREATE TABLE delivery_sync_stage_v1 AS
            SELECT
                UPPER(TRIM(sm.canonical_symbol)) AS canonical_symbol,
                CAST(d.date AS DATE) AS date,
                CAST(d.delivery_pct AS DOUBLE) AS delivery_pct
            FROM delivery_daily d
            JOIN symbol_master sm
              ON UPPER(TRIM(d.symbol)) = UPPER(TRIM(sm.canonical_symbol))
            WHERE d.delivery_pct IS NOT NULL
              AND CAST(d.delivery_pct AS DOUBLE) BETWEEN 0 AND 100
              AND COALESCE(sm.in_universe, TRUE)=TRUE
            """
        )
        stage_rows = conn.execute("SELECT COUNT(*) FROM delivery_sync_stage_v1").fetchone()[0]
        if int(stage_rows) <= 0:
            # Fallback for legacy/corrupted delivery_daily schemas:
            # keep canonical table usable by sourcing latest known valid rows.
            conn.execute("DROP TABLE IF EXISTS delivery_sync_stage_v1")
            conn.execute(
                """
                CREATE TABLE delivery_sync_stage_v1 AS
                SELECT
                    UPPER(TRIM(canonical_symbol)) AS canonical_symbol,
                    CAST(date AS DATE) AS date,
                    CAST(delivery_pct AS DOUBLE) AS delivery_pct
                FROM delivery_daily_v1
                WHERE delivery_pct IS NOT NULL
                  AND CAST(delivery_pct AS DOUBLE) BETWEEN 0 AND 100
                """
            )
            stage_rows = conn.execute("SELECT COUNT(*) FROM delivery_sync_stage_v1").fetchone()[0]
            if int(stage_rows) <= 0:
                print("delivery_sync: no live delivery rows to merge")
                conn.execute("DROP TABLE IF EXISTS delivery_sync_stage_v1")
                return
            print("delivery_sync: using canonical fallback stage from delivery_daily_v1")

        before_target = conn.execute("SELECT COUNT(*) FROM delivery_daily_v1").fetchone()[0]
        conn.execute(
            """
            INSERT INTO delivery_daily_v1
            SELECT s.canonical_symbol, s.date, s.delivery_pct
            FROM delivery_sync_stage_v1 s
            LEFT JOIN delivery_daily_v1 t
              ON UPPER(TRIM(t.canonical_symbol)) = UPPER(TRIM(s.canonical_symbol))
             AND CAST(t.date AS DATE) = CAST(s.date AS DATE)
            WHERE t.canonical_symbol IS NULL
            """
        )
        after_target = conn.execute("SELECT COUNT(*) FROM delivery_daily_v1").fetchone()[0]
        inserted = int(after_target) - int(before_target)

        conn.execute("DROP TABLE IF EXISTS delivery_sync_stage_v1")
        print(f"delivery_sync: inserted_rows={int(inserted or 0)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
