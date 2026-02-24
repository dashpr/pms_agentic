from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DEFAULT_TABLES = [
    "agentic_runs_v1",
    "agentic_watchlist_v1",
    "agentic_portfolio_targets_v1",
    "agentic_consensus_v1",
    "agentic_agent_signals_v1",
    "agentic_backtest_runs_v1",
    "agentic_backtest_equity_v1",
    "agentic_backtest_trades_v1",
    "symbol_master",
    "prices_daily_v1",
    "delivery_daily_v1",
    "bulk_block_deals",
    "pipeline_health_checks_v1",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build slim runtime DuckDB for cloud dashboard")
    p.add_argument("--src-db", default="data/ownership.duckdb")
    p.add_argument("--out-db", default="data/ownership_runtime.duckdb")
    p.add_argument(
        "--tables",
        default=",".join(DEFAULT_TABLES),
        help="Comma-separated table list to copy into runtime DB",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    src = Path(str(args.src_db))
    out = Path(str(args.out_db))
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    tables = [t.strip() for t in str(args.tables).split(",") if t.strip()]
    if not tables:
        raise SystemExit("No tables specified for runtime snapshot.")

    with duckdb.connect(str(out)) as conn:
        conn.execute(f"ATTACH '{src}' AS src (READ_ONLY)")
        existing = {
            str(r[0])
            for r in conn.execute(
                """
                SELECT table_name
                FROM duckdb_tables()
                WHERE database_name='src'
                  AND schema_name='main'
                """
            ).fetchall()
        }
        copied = 0
        skipped: list[str] = []
        for t in tables:
            if t not in existing:
                skipped.append(t)
                continue
            conn.execute(f"CREATE TABLE {t} AS SELECT * FROM src.{t}")
            copied += 1
        conn.execute("CHECKPOINT")

    print("===== BUILD RUNTIME DASHBOARD DB =====")
    print("src_db:", src)
    print("out_db:", out)
    print("copied_tables:", copied)
    print("skipped_tables:", len(skipped))
    if skipped:
        print("skipped:", ",".join(skipped))
    print("size_bytes:", out.stat().st_size if out.exists() else 0)
    print("======================================")


if __name__ == "__main__":
    main()
