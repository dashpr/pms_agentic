from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


@dataclass(frozen=True)
class PipelineRule:
    pipeline: str
    table_name: str
    date_col: str
    symbol_col: str | None
    required: bool
    lag_warn_days: int
    lag_fail_days: int
    min_rows: int = 1
    min_symbols: int = 0
    notes: str = ""


def _table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    x = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema='main' AND table_name=?
        """,
        [table_name],
    ).fetchone()
    return bool(x and x[0] > 0)


def _safe_count_distinct(conn: duckdb.DuckDBPyConnection, table: str, col: str | None) -> int:
    if not col:
        return 0
    try:
        return int(
            conn.execute(f"SELECT COUNT(DISTINCT {col}) FROM {table}").fetchone()[0] or 0
        )
    except Exception:
        return 0


def _eval_rule(
    conn: duckdb.DuckDBPyConnection,
    as_of_date: date,
    rule: PipelineRule,
) -> dict[str, Any]:
    out = {
        "pipeline": rule.pipeline,
        "table_name": rule.table_name,
        "required": bool(rule.required),
        "status": "UNKNOWN",
        "last_date": None,
        "lag_days": None,
        "row_count": 0,
        "symbol_count": 0,
        "notes": rule.notes,
        "message": "",
    }

    if not _table_exists(conn, rule.table_name):
        out["status"] = "BROKEN" if rule.required else "OPTIONAL_OFF"
        out["message"] = "table_missing"
        return out

    q = f"SELECT MAX(CAST({rule.date_col} AS DATE)), COUNT(*) FROM {rule.table_name}"
    last_date, row_count = conn.execute(q).fetchone()
    out["last_date"] = last_date
    out["row_count"] = int(row_count or 0)
    out["symbol_count"] = _safe_count_distinct(conn, rule.table_name, rule.symbol_col)

    if out["row_count"] < int(rule.min_rows):
        out["status"] = "BROKEN" if rule.required else "OPTIONAL_OFF"
        out["message"] = f"rows_below_min({out['row_count']}<{rule.min_rows})"
        return out
    if out["symbol_count"] < int(rule.min_symbols):
        out["status"] = "BROKEN" if rule.required else "OPTIONAL_OFF"
        out["message"] = f"symbols_below_min({out['symbol_count']}<{rule.min_symbols})"
        return out
    if last_date is None:
        out["status"] = "BROKEN" if rule.required else "OPTIONAL_OFF"
        out["message"] = "no_last_date"
        return out

    lag_days = int((pd.Timestamp(as_of_date) - pd.Timestamp(last_date)).days)
    # If data is newer than the requested as_of_date (e.g., forward-filled cache),
    # treat lag as zero for freshness-gate purposes.
    lag_days = max(lag_days, 0)
    out["lag_days"] = lag_days
    if lag_days > int(rule.lag_fail_days):
        out["status"] = "BROKEN"
        out["message"] = f"stale_{lag_days}d"
    elif lag_days > int(rule.lag_warn_days):
        out["status"] = "STALE"
        out["message"] = f"lag_warn_{lag_days}d"
    else:
        out["status"] = "WORKING"
        out["message"] = "ok"
    return out


def compute_pipeline_health(
    conn: duckdb.DuckDBPyConnection,
    as_of_date: date | None = None,
    require_news: bool = False,
    require_fundamentals: bool = False,
) -> pd.DataFrame:
    as_of = as_of_date
    if as_of is None:
        try:
            px_max = conn.execute("SELECT MAX(CAST(date AS DATE)) FROM prices_daily_v1").fetchone()[0]
            as_of = pd.to_datetime(px_max).date() if px_max is not None else None
        except Exception:
            as_of = None
    if as_of is None:
        as_of = pd.Timestamp.utcnow().date()
    rules = [
        PipelineRule(
            pipeline="prices_daily_v1",
            table_name="prices_daily_v1",
            date_col="date",
            symbol_col="canonical_symbol",
            required=True,
            lag_warn_days=2,
            lag_fail_days=5,
            min_rows=100000,
            min_symbols=200,
            notes="Core market data feed",
        ),
        PipelineRule(
            pipeline="delivery_daily_v1",
            table_name="delivery_daily_v1",
            date_col="date",
            symbol_col="canonical_symbol",
            required=True,
            lag_warn_days=3,
            lag_fail_days=8,
            min_rows=100000,
            min_symbols=180,
            notes="Ownership delivery feed",
        ),
        PipelineRule(
            pipeline="bulk_block_deals",
            table_name="bulk_block_deals",
            date_col="date",
            symbol_col="symbol",
            required=True,
            # Event-driven feed can naturally have sparse dates.
            lag_warn_days=14,
            lag_fail_days=45,
            min_rows=1,
            min_symbols=1,
            notes="Ownership block/bulk flow feed",
        ),
        PipelineRule(
            pipeline="news_social_raw_v1",
            table_name="news_social_raw_v1",
            date_col="date",
            symbol_col="canonical_symbol",
            required=bool(require_news),
            lag_warn_days=2,
            lag_fail_days=7,
            min_rows=1 if require_news else 0,
            min_symbols=1 if require_news else 0,
            notes="News/social sentiment feed",
        ),
        PipelineRule(
            pipeline="fundamentals_daily_v1",
            table_name="fundamentals_daily_v1",
            date_col="date",
            symbol_col="symbol",
            required=bool(require_fundamentals),
            lag_warn_days=45,
            lag_fail_days=120,
            min_rows=1 if require_fundamentals else 0,
            min_symbols=1 if require_fundamentals else 0,
            notes="Fundamental data feed",
        ),
        PipelineRule(
            pipeline="agentic_runs_v1",
            table_name="agentic_runs_v1",
            date_col="as_of_date",
            symbol_col=None,
            required=True,
            lag_warn_days=1,
            lag_fail_days=3,
            min_rows=1,
            notes="Decision cycle freshness",
        ),
        PipelineRule(
            pipeline="agentic_consensus_v1",
            table_name="agentic_consensus_v1",
            date_col="as_of_date",
            symbol_col="symbol",
            required=True,
            lag_warn_days=1,
            lag_fail_days=3,
            min_rows=1,
            min_symbols=100,
            notes="Fusion score freshness",
        ),
    ]

    rows = [_eval_rule(conn, as_of, r) for r in rules]
    return pd.DataFrame(rows)


def persist_pipeline_health(
    conn: duckdb.DuckDBPyConnection,
    health_df: pd.DataFrame,
) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_health_checks_v1 (
            run_ts TIMESTAMP,
            pipeline VARCHAR,
            table_name VARCHAR,
            required BOOLEAN,
            status VARCHAR,
            last_date DATE,
            lag_days INTEGER,
            row_count BIGINT,
            symbol_count BIGINT,
            notes VARCHAR,
            message VARCHAR,
            payload_json VARCHAR
        )
        """
    )
    z = health_df.copy()
    z["run_ts"] = pd.Timestamp.utcnow()
    z["payload_json"] = [
        json.dumps(
            {
                "pipeline": str(r.pipeline),
                "status": str(r.status),
                "last_date": None if pd.isna(r.last_date) else str(r.last_date),
                "lag_days": None if pd.isna(r.lag_days) else int(r.lag_days),
                "rows": int(r.row_count),
                "symbols": int(r.symbol_count),
                "message": str(r.message),
            },
            ensure_ascii=True,
        )
        for r in z.itertuples(index=False)
    ]
    cols = [
        "run_ts",
        "pipeline",
        "table_name",
        "required",
        "status",
        "last_date",
        "lag_days",
        "row_count",
        "symbol_count",
        "notes",
        "message",
        "payload_json",
    ]
    conn.register("health_df", z[cols])
    conn.execute("INSERT INTO pipeline_health_checks_v1 SELECT * FROM health_df")
    conn.unregister("health_df")


def main(argv=None):
    db_path = Path("data/ownership.duckdb")
    with duckdb.connect(str(db_path)) as conn:
        h = compute_pipeline_health(
            conn=conn,
            as_of_date=None,
            require_news=False,
            require_fundamentals=False,
        )
        persist_pipeline_health(conn, h)
    print("===== PIPELINE HEALTH =====")
    for r in h.itertuples(index=False):
        print(
            f"{r.pipeline}: {r.status} | last={r.last_date} | lag={r.lag_days} | "
            f"rows={r.row_count} | symbols={r.symbol_count} | msg={r.message}"
        )
    print("===========================")


if __name__ == "__main__":
    main()
