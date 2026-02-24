"""
Data Quality Gates v1 (model-free)

Validates core data tables only so the project can rebuild models from a clean
data foundation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

try:
    from src.data_layer.schema_validation import require_columns, require_non_empty
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.data_layer.schema_validation import require_columns, require_non_empty


DB_PATH = Path("data/ownership.duckdb")


@dataclass
class GateConfig:
    min_symbol_master_rows: int = 100
    min_prices_rows: int = 100_000
    min_prices_symbols: int = 80
    min_delivery_rows: int = 100_000
    min_delivery_symbols: int = 80
    max_price_null_ratio: float = 0.0
    max_delivery_null_ratio: float = 0.0


def check_duplicates(conn: duckdb.DuckDBPyConnection, table: str, keys: list[str]) -> None:
    key_sql = ", ".join(keys)
    dup_count = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT {key_sql}, COUNT(*) c
            FROM {table}
            GROUP BY {key_sql}
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    if dup_count > 0:
        raise ValueError(f"{table} has {dup_count} duplicate key groups on ({key_sql})")


def check_symbol_master(conn: duckdb.DuckDBPyConnection, cfg: GateConfig) -> None:
    require_columns(
        conn,
        "symbol_master",
        ["canonical_symbol", "in_universe"],
    )
    require_non_empty(conn, "symbol_master")

    total_rows = conn.execute("SELECT COUNT(*) FROM symbol_master").fetchone()[0]
    if total_rows < cfg.min_symbol_master_rows:
        raise ValueError(
            f"symbol_master rows too low: {total_rows} < {cfg.min_symbol_master_rows}"
        )

    active_rows = conn.execute(
        "SELECT COUNT(*) FROM symbol_master WHERE COALESCE(in_universe, FALSE)=TRUE"
    ).fetchone()[0]
    if active_rows <= 0:
        raise ValueError("symbol_master has zero active in_universe symbols")

    check_duplicates(conn, "symbol_master", ["canonical_symbol"])


def check_prices(conn: duckdb.DuckDBPyConnection, cfg: GateConfig) -> None:
    require_columns(
        conn,
        "prices_daily_v1",
        ["canonical_symbol", "date", "open", "high", "low", "close", "volume"],
    )
    require_non_empty(conn, "prices_daily_v1")
    check_duplicates(conn, "prices_daily_v1", ["canonical_symbol", "date"])

    rows, syms = conn.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT canonical_symbol)
        FROM prices_daily_v1
        """
    ).fetchone()
    if rows < cfg.min_prices_rows:
        raise ValueError(f"prices_daily_v1 rows too low: {rows} < {cfg.min_prices_rows}")
    if syms < cfg.min_prices_symbols:
        raise ValueError(
            f"prices_daily_v1 symbol count too low: {syms} < {cfg.min_prices_symbols}"
        )

    null_ratio = conn.execute(
        """
        SELECT AVG(
            CASE
                WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL
                THEN 1.0 ELSE 0.0
            END
        )
        FROM prices_daily_v1
        """
    ).fetchone()[0]
    if null_ratio is None:
        raise ValueError("prices_daily_v1 null-ratio query returned NULL")
    if float(null_ratio) > cfg.max_price_null_ratio:
        raise ValueError(
            f"prices_daily_v1 null ratio too high: {float(null_ratio):.4f} > "
            f"{cfg.max_price_null_ratio:.4f}"
        )


def check_delivery(conn: duckdb.DuckDBPyConnection, cfg: GateConfig) -> None:
    require_columns(
        conn,
        "delivery_daily_v1",
        ["canonical_symbol", "date", "delivery_pct"],
    )
    require_non_empty(conn, "delivery_daily_v1")
    check_duplicates(conn, "delivery_daily_v1", ["canonical_symbol", "date"])

    rows, syms = conn.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT canonical_symbol)
        FROM delivery_daily_v1
        """
    ).fetchone()
    if rows < cfg.min_delivery_rows:
        raise ValueError(f"delivery_daily_v1 rows too low: {rows} < {cfg.min_delivery_rows}")
    if syms < cfg.min_delivery_symbols:
        raise ValueError(
            f"delivery_daily_v1 symbol count too low: {syms} < {cfg.min_delivery_symbols}"
        )

    null_ratio = conn.execute(
        """
        SELECT AVG(CASE WHEN delivery_pct IS NULL THEN 1.0 ELSE 0.0 END)
        FROM delivery_daily_v1
        """
    ).fetchone()[0]
    if null_ratio is None:
        raise ValueError("delivery_daily_v1 null-ratio query returned NULL")
    if float(null_ratio) > cfg.max_delivery_null_ratio:
        raise ValueError(
            f"delivery_daily_v1 null ratio too high: {float(null_ratio):.4f} > "
            f"{cfg.max_delivery_null_ratio:.4f}"
        )

    out_of_range = conn.execute(
        """
        SELECT COUNT(*)
        FROM delivery_daily_v1
        WHERE delivery_pct < 0 OR delivery_pct > 100
        """
    ).fetchone()[0]
    if out_of_range > 0:
        raise ValueError(
            f"delivery_daily_v1 has {out_of_range} rows with delivery_pct outside [0,100]"
        )


def check_news_raw_optional(conn: duckdb.DuckDBPyConnection) -> None:
    exists = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema='main' AND table_name='news_social_raw_v1'
        """
    ).fetchone()[0]
    if not exists:
        return

    require_columns(
        conn,
        "news_social_raw_v1",
        [
            "event_id",
            "date",
            "canonical_symbol",
            "provider",
            "sentiment_score",
            "relevance_score",
            "source_confidence",
        ],
    )
    check_duplicates(conn, "news_social_raw_v1", ["event_id"])


def run_quality_gates(cfg: GateConfig) -> None:
    conn = duckdb.connect(DB_PATH)
    try:
        check_symbol_master(conn, cfg)
        check_prices(conn, cfg)
        check_delivery(conn, cfg)
        check_news_raw_optional(conn)
        print("All data quality gates passed.")
    finally:
        conn.close()


def main() -> None:
    run_quality_gates(GateConfig())


if __name__ == "__main__":
    main()
