from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

try:
    from src.agentic.run_ai_agent_stable_stocks_v1 import main as run_locked_model
    from src.agentic.run_stage2_optimizer_walkforward_v1 import main as run_stage2
    from src.data_layer.rebuild_delivery_from_raw_v1 import main as rebuild_delivery
    from src.data_layer.rebuild_prices_canonical import main as rebuild_prices
    from src.data_layer.sync_delivery_daily_to_v1 import main as sync_delivery_v1
    from src.market.backfill_nifty_universe_prices_v1 import main as backfill_prices
    from src.market.build_price_db import build_price_db
    from src.news_social.ingest_news_social_raw_v1 import IngestConfig, ingest_news_social_raw
    from src.ownership.collect_bulk_deals_nse import run_nse_bulk_ingestion as update_bulk_live
    from src.ownership.collect_delivery_data import run_delivery_update as update_delivery_live
    from src.qa.pipeline_health_v1 import compute_pipeline_health, persist_pipeline_health
    from src.qa.run_release_checks_ai_agent_stable_v1 import parse_args as parse_release_args
    from src.qa.run_release_checks_ai_agent_stable_v1 import run_release_checks
    from src.versioning.record_version_snapshot_v1 import record_version_snapshot
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.agentic.run_ai_agent_stable_stocks_v1 import main as run_locked_model
    from src.agentic.run_stage2_optimizer_walkforward_v1 import main as run_stage2
    from src.data_layer.rebuild_delivery_from_raw_v1 import main as rebuild_delivery
    from src.data_layer.rebuild_prices_canonical import main as rebuild_prices
    from src.data_layer.sync_delivery_daily_to_v1 import main as sync_delivery_v1
    from src.market.backfill_nifty_universe_prices_v1 import main as backfill_prices
    from src.market.build_price_db import build_price_db
    from src.news_social.ingest_news_social_raw_v1 import IngestConfig, ingest_news_social_raw
    from src.ownership.collect_bulk_deals_nse import run_nse_bulk_ingestion as update_bulk_live
    from src.ownership.collect_delivery_data import run_delivery_update as update_delivery_live
    from src.qa.pipeline_health_v1 import compute_pipeline_health, persist_pipeline_health
    from src.qa.run_release_checks_ai_agent_stable_v1 import parse_args as parse_release_args
    from src.qa.run_release_checks_ai_agent_stable_v1 import run_release_checks
    from src.versioning.record_version_snapshot_v1 import record_version_snapshot


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Incremental production updater + scheduled retraining cycle",
    )
    p.add_argument("--db-path", default="data/ownership.duckdb")
    p.add_argument("--as-of-date", default=None, help="YYYY-MM-DD; defaults to latest prices_daily_v1 date")
    p.add_argument("--daily-auto", action="store_true", help="Run full daily production refresh + decision.")

    p.add_argument("--refresh-prices", action="store_true", help="Refresh stale OHLCV CSVs from source.")
    p.add_argument("--stale-days", type=int, default=1)
    p.add_argument("--refresh-max-symbols", type=int, default=0)
    p.add_argument("--build-prices-table", action="store_true", help="Rebuild prices_daily from local CSVs.")
    p.add_argument("--rebuild-prices-canonical", action="store_true")
    p.add_argument("--rebuild-delivery-canonical", action="store_true")
    p.add_argument("--update-delivery-live", action="store_true", help="Fetch latest NSE delivery DAT into delivery_daily.")
    p.add_argument("--sync-delivery-v1", action="store_true", help="Merge delivery_daily into delivery_daily_v1.")
    p.add_argument("--update-bulk-live", action="store_true", help="Fetch latest/rolling bulk-block deals from NSE.")

    p.add_argument("--ingest-news", action="store_true")
    p.add_argument("--news-lookback-days", type=int, default=3)
    p.add_argument("--news-max-symbols", type=int, default=120)

    p.add_argument("--run-release-checks", action="store_true")
    p.add_argument("--run-pipeline-health", action="store_true")
    p.add_argument("--run-locked-decision", action="store_true")
    p.add_argument("--persist-backtest", action="store_true")
    p.add_argument("--bt-start-date", default="2016-01-01")

    p.add_argument(
        "--retrain-schedule",
        default="none",
        choices=["none", "weekly", "monthly"],
        help="weekly=Sunday, monthly=first Sunday.",
    )
    p.add_argument("--run-retrain-now", action="store_true")
    p.add_argument("--retrain-start-date", default="2016-01-01")
    p.add_argument("--retrain-optimizer-trials", type=int, default=64)
    p.add_argument("--retrain-wf-trials-per-fold", type=int, default=24)
    p.add_argument("--retrain-min-trade-win-rate", type=float, default=0.55)
    p.add_argument("--retrain-max-drawdown-floor", type=float, default=-0.20)
    p.add_argument("--skip-version-snapshot", action="store_true")
    return p.parse_args(argv)


def _resolve_as_of(raw: str | None, db_path: str) -> date:
    if raw:
        return date.fromisoformat(str(raw))
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            d = conn.execute("SELECT MAX(CAST(date AS DATE)) FROM prices_daily_v1").fetchone()[0]
            if d is not None:
                return pd.to_datetime(d).date()
    except Exception:
        pass
    return pd.Timestamp.utcnow().date()


def _is_retrain_due(as_of: date, schedule: str) -> bool:
    wd = pd.Timestamp(as_of).weekday()  # Mon=0..Sun=6
    if schedule == "weekly":
        return wd == 6
    if schedule == "monthly":
        return wd == 6 and pd.Timestamp(as_of).day <= 7
    return False


def main(argv=None):
    args = parse_args(argv)
    as_of = _resolve_as_of(args.as_of_date, str(args.db_path))
    print("\n===== INCREMENTAL UPDATE + RETRAIN CYCLE =====")
    print("as_of_date:", as_of)

    if args.daily_auto:
        args.refresh_prices = True
        args.build_prices_table = True
        args.rebuild_prices_canonical = True
        args.update_delivery_live = True
        args.sync_delivery_v1 = True
        args.update_bulk_live = True
        args.ingest_news = True
        args.run_pipeline_health = True
        args.run_release_checks = True
        args.run_locked_decision = True

    cycle_status: dict[str, Any] = {
        "as_of_date": as_of.isoformat(),
        "steps": {},
    }

    def _run_step(name: str, fn):
        try:
            fn()
            cycle_status["steps"][name] = {"status": "OK"}
        except Exception as e:
            cycle_status["steps"][name] = {"status": "ERROR", "error": str(e)[:500]}

    if args.refresh_prices:
        def _step_prices():
            px_argv = [
                "--db-path",
                str(args.db_path),
                "--refresh-existing",
                "--stale-days",
                str(int(args.stale_days)),
            ]
            if int(args.refresh_max_symbols) > 0:
                px_argv += ["--max-symbols", str(int(args.refresh_max_symbols))]
            backfill_prices(px_argv)

        _run_step("refresh_prices", _step_prices)

    if args.build_prices_table:
        _run_step("build_prices_table", build_price_db)

    if args.rebuild_prices_canonical:
        _run_step("rebuild_prices_canonical", rebuild_prices)

    if args.update_delivery_live:
        def _step_delivery_live():
            with duckdb.connect(str(args.db_path), read_only=True) as c0:
                before = c0.execute("SELECT MAX(CAST(date AS DATE)) FROM delivery_daily").fetchone()[0]
            upd = update_delivery_live(max_lookback=21, prefer_cache=True, raise_on_error=True)
            with duckdb.connect(str(args.db_path), read_only=True) as c1:
                after = c1.execute("SELECT MAX(CAST(date AS DATE)) FROM delivery_daily").fetchone()[0]
            cycle_status["update_delivery_live"] = upd
            if after is None:
                raise RuntimeError("delivery_daily still empty after live update")
            if before is not None and pd.to_datetime(after).date() < pd.to_datetime(before).date():
                raise RuntimeError("delivery_daily max date moved backwards")
            lag_days = int((pd.Timestamp(as_of) - pd.Timestamp(after)).days)
            if before is not None and pd.to_datetime(after).date() == pd.to_datetime(before).date() and lag_days > 2:
                raise RuntimeError(f"delivery_daily not refreshed; lag still {lag_days} days")

        _run_step("update_delivery_live", _step_delivery_live)

    if args.sync_delivery_v1:
        _run_step("sync_delivery_v1", sync_delivery_v1)

    if args.rebuild_delivery_canonical:
        _run_step("rebuild_delivery_canonical", rebuild_delivery)

    if args.update_bulk_live:
        def _step_bulk():
            with duckdb.connect(str(args.db_path), read_only=True) as c0:
                before_rows = int(c0.execute("SELECT COUNT(*) FROM bulk_block_deals").fetchone()[0] or 0)
            upd = update_bulk_live(lookback_days=120, batch_days=30, raise_on_error=True)
            with duckdb.connect(str(args.db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS bulk_block_deals (
                        symbol VARCHAR,
                        date DATE,
                        side VARCHAR,
                        qty DOUBLE,
                        price DOUBLE,
                        participant VARCHAR
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE OR REPLACE TABLE bulk_block_deals AS
                    SELECT DISTINCT
                        UPPER(TRIM(symbol)) AS symbol,
                        CAST(date AS DATE) AS date,
                        UPPER(TRIM(COALESCE(side, ''))) AS side,
                        CAST(qty AS DOUBLE) AS qty,
                        CAST(price AS DOUBLE) AS price,
                        COALESCE(participant, '') AS participant
                    FROM bulk_block_deals
                    WHERE qty IS NOT NULL
                      AND price IS NOT NULL
                    """
                )
                after_rows = int(conn.execute("SELECT COUNT(*) FROM bulk_block_deals").fetchone()[0] or 0)
            cycle_status["update_bulk_live"] = upd
            if after_rows <= 0:
                raise RuntimeError("bulk_block_deals has no rows after live update")
            if after_rows < before_rows:
                raise RuntimeError("bulk_block_deals row count decreased unexpectedly")

        _run_step("update_bulk_live", _step_bulk)

    if args.ingest_news:
        def _step_news():
            ingest_news_social_raw(
                IngestConfig(
                    as_of_date=as_of,
                    lookback_days=int(args.news_lookback_days),
                    max_symbols=int(args.news_max_symbols),
                )
            )

        _run_step("ingest_news", _step_news)

    if args.run_pipeline_health:
        def _step_health():
            with duckdb.connect(str(args.db_path)) as conn:
                h = compute_pipeline_health(
                    conn=conn,
                    as_of_date=as_of,
                    require_news=bool(args.ingest_news),
                    require_fundamentals=False,
                )
                persist_pipeline_health(conn, h)
                cycle_status["pipeline_health"] = h.to_dict(orient="records")

        _run_step("pipeline_health", _step_health)

    if args.run_release_checks:
        def _step_release():
            r_args = parse_release_args(["--db-path", str(args.db_path), "--as-of-date", as_of.isoformat()])
            run_release_checks(r_args)
            print("release_checks: PASS")

        _run_step("release_checks", _step_release)

    if args.run_locked_decision:
        def _step_locked():
            locked_args = [
                "--db-path",
                str(args.db_path),
                "--mode",
                "both" if args.persist_backtest else "decision",
                "--as-of-date",
                as_of.isoformat(),
                "--out-json",
                "data/reports/ai_agent_stable_stocks_v1_latest.json",
            ]
            if args.persist_backtest:
                locked_args += ["--persist-backtest", "--bt-start-date", str(args.bt_start_date)]
            run_locked_model(locked_args)

        _run_step("locked_decision", _step_locked)

    retrain_due = bool(args.run_retrain_now) or _is_retrain_due(as_of, str(args.retrain_schedule))
    if retrain_due:
        def _step_retrain():
            s2_args = [
                "--mode",
                "both",
                "--db-path",
                str(args.db_path),
                "--start-date",
                str(args.retrain_start_date),
                "--end-date",
                as_of.isoformat(),
                "--optimizer-trials",
                str(int(args.retrain_optimizer_trials)),
                "--wf-trials-per-fold",
                str(int(args.retrain_wf_trials_per_fold)),
                "--min-trade-win-rate",
                str(float(args.retrain_min_trade_win_rate)),
                "--max-drawdown-floor",
                str(float(args.retrain_max_drawdown_floor)),
                "--watchlist-size",
                "25",
                "--portfolio-min",
                "10",
                "--portfolio-max",
                "15",
                "--portfolio-target",
                "12",
            ]
            run_stage2(s2_args)
            print("retrain_cycle: COMPLETED (challenger generated)")

        _run_step("retrain_cycle", _step_retrain)
    else:
        print("retrain_cycle: SKIPPED (not due)")
        cycle_status["steps"]["retrain_cycle"] = {"status": "SKIPPED"}

    if not bool(args.skip_version_snapshot):
        def _step_version():
            v = record_version_snapshot(
                db_path=str(args.db_path),
                context="incremental_cycle",
                notes=f"as_of={as_of.isoformat()}",
            )
            cycle_status["version_snapshot"] = {
                "snapshot_id": v.get("snapshot_id"),
                "model_version": v.get("model_version"),
                "dashboard_version": v.get("dashboard_version"),
                "data_pipeline_version": v.get("data_pipeline_version"),
            }

        _run_step("version_snapshot", _step_version)

    def _json_safe(x: Any):
        if isinstance(x, dict):
            return {str(k): _json_safe(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_json_safe(v) for v in x]
        if isinstance(x, tuple):
            return [_json_safe(v) for v in x]
        if isinstance(x, (pd.Timestamp, date)):
            return str(x)
        return x

    out = Path("data/reports/incremental_cycle_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(cycle_status), ensure_ascii=True, indent=2), encoding="utf-8")
    print("cycle_report:", out)

    print("=============================================\n")


if __name__ == "__main__":
    main()
