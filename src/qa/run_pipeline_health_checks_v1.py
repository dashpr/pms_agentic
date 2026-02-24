from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

try:
    from src.qa.pipeline_health_v1 import compute_pipeline_health, persist_pipeline_health
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.qa.pipeline_health_v1 import compute_pipeline_health, persist_pipeline_health


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pipeline health checks v1")
    p.add_argument("--db-path", default="data/ownership.duckdb")
    p.add_argument("--as-of-date", default=None, help="YYYY-MM-DD, default latest prices_daily_v1 date")
    p.add_argument("--require-news", action="store_true")
    p.add_argument("--require-fundamentals", action="store_true")
    p.add_argument("--persist", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    as_of = pd.to_datetime(args.as_of_date).date() if args.as_of_date else None
    with duckdb.connect(str(args.db_path)) as conn:
        health = compute_pipeline_health(
            conn=conn,
            as_of_date=as_of,
            require_news=bool(args.require_news),
            require_fundamentals=bool(args.require_fundamentals),
        )
        if args.persist:
            persist_pipeline_health(conn, health)
    print("===== PIPELINE HEALTH CHECKS v1 =====")
    for r in health.itertuples(index=False):
        print(
            f"{r.pipeline}: {r.status} | last={r.last_date} | lag={r.lag_days} | "
            f"rows={r.row_count} | symbols={r.symbol_count} | msg={r.message}"
        )
    print("=====================================")


if __name__ == "__main__":
    main()
