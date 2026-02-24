"""
Data Pipeline Runner (v1)

Data-only orchestration after model reset.

Scope:
1) Optional price CSV refresh (yfinance backfill)
2) Build/refresh `prices_daily` from local CSVs
3) Rebuild canonical `prices_daily_v1` and `delivery_daily_v1`
4) Optional live-news raw ingestion (`news_social_raw_v1`)
5) Data-quality gates on core data tables only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from src.agentic.contracts_v1 import OrchestrationPolicy
    from src.agentic.orchestrator_v1 import run_agentic_cycle
    from src.data_layer.rebuild_delivery_from_raw_v1 import main as rebuild_delivery
    from src.data_layer.rebuild_prices_canonical import main as rebuild_prices
    from src.market.backfill_nifty_universe_prices_v1 import main as backfill_prices
    from src.market.build_price_db import build_price_db
    from src.news_social.ingest_news_social_raw_v1 import IngestConfig, ingest_news_social_raw
    from src.qa.run_quality_gates_v1 import main as run_data_quality_gates
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.agentic.contracts_v1 import OrchestrationPolicy
    from src.agentic.orchestrator_v1 import run_agentic_cycle
    from src.data_layer.rebuild_delivery_from_raw_v1 import main as rebuild_delivery
    from src.data_layer.rebuild_prices_canonical import main as rebuild_prices
    from src.market.backfill_nifty_universe_prices_v1 import main as backfill_prices
    from src.market.build_price_db import build_price_db
    from src.news_social.ingest_news_social_raw_v1 import IngestConfig, ingest_news_social_raw
    from src.qa.run_quality_gates_v1 import main as run_data_quality_gates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run data-only ownership pipeline v1")
    p.add_argument(
        "--refresh-prices-csv",
        action="store_true",
        help="Backfill/refresh local OHLCV CSV files from data source.",
    )
    p.add_argument(
        "--refresh-existing-prices-csv",
        action="store_true",
        help="When refreshing price CSVs, update existing stale files too.",
    )
    p.add_argument(
        "--build-prices-table",
        action="store_true",
        help="Rebuild legacy `prices_daily` from local CSV files.",
    )
    p.add_argument(
        "--rebuild-canonical",
        action="store_true",
        help="Rebuild canonical `prices_daily_v1` and `delivery_daily_v1`.",
    )
    p.add_argument(
        "--skip-news-ingestion",
        action="store_true",
        help="Skip external news raw ingestion (`news_social_raw_v1`).",
    )
    p.add_argument(
        "--news-lookback-days",
        type=int,
        default=3,
        help="Lookback window for external news ingestion.",
    )
    p.add_argument(
        "--news-max-symbols",
        type=int,
        default=120,
        help="Maximum symbols queried for external news ingestion.",
    )
    p.add_argument(
        "--skip-quality-gates",
        action="store_true",
        help="Skip core data quality gates.",
    )
    p.add_argument(
        "--with-agentic-cycle",
        action="store_true",
        help="Run agentic watchlist/portfolio decision cycle after data quality gates.",
    )
    p.add_argument(
        "--agentic-as-of-date",
        default=None,
        help="As-of date for agentic cycle (YYYY-MM-DD). Defaults to today UTC date.",
    )
    p.add_argument("--agentic-watchlist-size", type=int, default=25)
    p.add_argument("--agentic-portfolio-min", type=int, default=10)
    p.add_argument("--agentic-portfolio-max", type=int, default=15)
    p.add_argument("--agentic-portfolio-target", type=int, default=12)
    p.add_argument("--agentic-universe-limit", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("\n===== DATA PIPELINE v1 (MODEL-FREE) =====\n")

    if args.refresh_prices_csv:
        argv = ["--refresh-existing"] if args.refresh_existing_prices_csv else None
        backfill_prices(argv)

    if args.build_prices_table:
        build_price_db()

    if args.rebuild_canonical:
        rebuild_prices()
        rebuild_delivery()

    if not args.skip_news_ingestion:
        ingest_news_social_raw(
            IngestConfig(
                as_of_date=pd.Timestamp.utcnow().date(),
                lookback_days=int(args.news_lookback_days),
                max_symbols=int(args.news_max_symbols),
            )
        )

    if not args.skip_quality_gates:
        run_data_quality_gates()

    if args.with_agentic_cycle:
        as_of = (
            pd.to_datetime(args.agentic_as_of_date).date()
            if args.agentic_as_of_date
            else pd.Timestamp.utcnow().date()
        )
        policy = OrchestrationPolicy(
            watchlist_size=int(args.agentic_watchlist_size),
            portfolio_min_positions=int(args.agentic_portfolio_min),
            portfolio_max_positions=int(args.agentic_portfolio_max),
            portfolio_target_positions=int(args.agentic_portfolio_target),
        )
        out = run_agentic_cycle(
            db_path="data/ownership.duckdb",
            as_of_date=as_of,
            policy=policy,
            universe_limit=int(args.agentic_universe_limit),
        )
        print("agentic_cycle:", out)

    print("\n===== DATA PIPELINE COMPLETE =====\n")


if __name__ == "__main__":
    main()
