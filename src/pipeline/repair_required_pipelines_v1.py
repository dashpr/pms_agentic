from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

try:
    from src.ownership.backfill_bulk_archives_to_local_v1 import main as backfill_bulk_local
    from src.data_layer.sync_delivery_daily_to_v1 import main as sync_delivery_v1
    from src.ownership.import_bulk_block_deals_from_files_v1 import main as import_bulk_local
    from src.ownership.collect_bulk_deals_nse import run_nse_bulk_ingestion
    from src.ownership.collect_delivery_data import run_delivery_update
    from src.qa.pipeline_health_v1 import compute_pipeline_health, persist_pipeline_health
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.ownership.backfill_bulk_archives_to_local_v1 import main as backfill_bulk_local
    from src.data_layer.sync_delivery_daily_to_v1 import main as sync_delivery_v1
    from src.ownership.import_bulk_block_deals_from_files_v1 import main as import_bulk_local
    from src.ownership.collect_bulk_deals_nse import run_nse_bulk_ingestion
    from src.ownership.collect_delivery_data import run_delivery_update
    from src.qa.pipeline_health_v1 import compute_pipeline_health, persist_pipeline_health


REQUIRED_PIPELINES = {"prices_daily_v1", "delivery_daily_v1", "bulk_block_deals"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Repair required pipelines until healthy")
    p.add_argument("--db-path", default="data/ownership.duckdb")
    p.add_argument("--as-of-date", default=None, help="YYYY-MM-DD; default latest prices_daily_v1 date")
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--bulk-lookback-days", type=int, default=120)
    p.add_argument("--bulk-local-input-dir", default="data_raw/nse_bulk")
    p.add_argument("--out-json", default="data/reports/repair_required_pipelines_v1_latest.json")
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


def _health(db_path: str, as_of: date) -> pd.DataFrame:
    with duckdb.connect(str(db_path)) as conn:
        h = compute_pipeline_health(
            conn=conn,
            as_of_date=as_of,
            require_news=False,
            require_fundamentals=False,
        )
        persist_pipeline_health(conn, h)
    return h


def _failing_required(h: pd.DataFrame) -> pd.DataFrame:
    z = h[h["pipeline"].isin(sorted(REQUIRED_PIPELINES))].copy()
    z["status"] = z["status"].astype(str).str.upper()
    return z[z["status"] != "WORKING"].copy()


def main(argv=None):
    args = parse_args(argv)
    as_of = _resolve_as_of(args.as_of_date, str(args.db_path))
    report: dict[str, Any] = {
        "as_of_date": as_of.isoformat(),
        "required_pipelines": sorted(REQUIRED_PIPELINES),
        "rounds": [],
        "passed": False,
    }

    h = _health(str(args.db_path), as_of)
    fail = _failing_required(h)
    for i in range(1, max(1, int(args.max_rounds)) + 1):
        if fail.empty:
            report["passed"] = True
            break
        failing_set = {str(x) for x in fail["pipeline"].tolist()}
        actions: dict[str, Any] = {}

        if "delivery_daily_v1" in failing_set:
            actions["update_delivery_live"] = run_delivery_update(
                max_lookback=21,
                prefer_cache=True,
                raise_on_error=False,
            )
            try:
                sync_delivery_v1()
                actions["sync_delivery_v1"] = {"ok": True}
            except Exception as e:
                actions["sync_delivery_v1"] = {"ok": False, "error": str(e)}

        if "bulk_block_deals" in failing_set:
            actions["update_bulk_live"] = run_nse_bulk_ingestion(
                lookback_days=int(args.bulk_lookback_days),
                batch_days=30,
                raise_on_error=False,
            )
            if not bool((actions["update_bulk_live"] or {}).get("ok", False)):
                try:
                    backfill_bulk_local(["--days", str(int(args.bulk_lookback_days)), "--out-dir", str(args.bulk_local_input_dir)])
                    actions["download_bulk_archives_local"] = {
                        "ok": True,
                        "output_dir": str(args.bulk_local_input_dir),
                    }
                except BaseException as e:
                    actions["download_bulk_archives_local"] = {
                        "ok": False,
                        "output_dir": str(args.bulk_local_input_dir),
                        "error": str(e),
                    }
                try:
                    import_bulk_local(["--input-dir", str(args.bulk_local_input_dir), "--recursive"])
                    actions["update_bulk_from_local"] = {
                        "ok": True,
                        "input_dir": str(args.bulk_local_input_dir),
                    }
                except BaseException as e:
                    actions["update_bulk_from_local"] = {
                        "ok": False,
                        "input_dir": str(args.bulk_local_input_dir),
                        "error": str(e),
                    }

        h = _health(str(args.db_path), as_of)
        fail = _failing_required(h)
        report["rounds"].append(
            {
                "round": i,
                "failing_before": sorted(failing_set),
                "actions": actions,
                "failing_after": fail[
                    ["pipeline", "status", "last_date", "lag_days", "message"]
                ].to_dict(orient="records"),
            }
        )
        if fail.empty:
            report["passed"] = True
            break

    out = Path(str(args.out_json))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=True, default=str, indent=2), encoding="utf-8")

    print("===== REPAIR REQUIRED PIPELINES =====")
    print("as_of_date:", as_of)
    print("passed:", bool(report["passed"]))
    print("rounds:", len(report["rounds"]))
    if not fail.empty:
        print("remaining_failures:")
        for r in fail.itertuples(index=False):
            print(f"- {r.pipeline}: {r.status} | msg={r.message} | last={r.last_date}")
    print("out_json:", out)
    print("====================================")

    if not bool(report["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
