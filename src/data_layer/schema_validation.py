from __future__ import annotations

from typing import Iterable

import duckdb


def get_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = ?
        """,
        [table_name],
    ).fetchall()
    return {r[0] for r in rows}


def require_columns(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    required_columns: Iterable[str],
) -> None:
    actual = get_columns(conn, table_name)

    if not actual:
        raise ValueError(f"Required table missing: {table_name}")

    missing = [c for c in required_columns if c not in actual]
    if missing:
        raise ValueError(
            f"Table {table_name} is missing columns: {missing}. "
            f"Available columns: {sorted(actual)}"
        )


def require_non_empty(conn: duckdb.DuckDBPyConnection, table_name: str) -> None:
    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    if count == 0:
        raise ValueError(f"Table {table_name} is empty")
