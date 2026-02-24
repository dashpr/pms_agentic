from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.agentic.run_ai_agent_stable_stocks_v1 import main as run_locked_model
    from src.qa.run_release_checks_ai_agent_stable_v1 import parse_args as parse_release_args
    from src.qa.run_release_checks_ai_agent_stable_v1 import run_release_checks
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.agentic.run_ai_agent_stable_stocks_v1 import main as run_locked_model
    from src.qa.run_release_checks_ai_agent_stable_v1 import parse_args as parse_release_args
    from src.qa.run_release_checks_ai_agent_stable_v1 import run_release_checks


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Production cycle runner for AI Agent Stable Stocks v1",
    )
    p.add_argument("--db-path", default="data/ownership.duckdb")
    p.add_argument("--as-of-date", default=date.today().isoformat(), help="YYYY-MM-DD")
    p.add_argument("--with-backtest", action="store_true")
    p.add_argument("--bt-start-date", default="2016-01-01")
    p.add_argument("--bt-end-date", default=None)
    p.add_argument("--skip-release-checks", action="store_true")
    p.add_argument("--release-skip-model-smoke", action="store_true")
    p.add_argument("--out-json", default="data/reports/production_cycle_ai_agent_stable_v1_latest.json")
    return p.parse_args(argv)


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    if isinstance(x, tuple):
        return [_json_safe(v) for v in x]
    if isinstance(x, (pd.Timestamp, date)):
        return x.isoformat()
    return x


def main(argv=None):
    args = parse_args(argv)
    payload: dict[str, Any] = {
        "as_of_date": str(args.as_of_date),
        "release_checks": None,
        "locked_model_report_json": "data/reports/ai_agent_stable_stocks_v1_latest.json",
    }

    if not bool(args.skip_release_checks):
        r_args = parse_release_args(
            [
                "--db-path",
                str(args.db_path),
                "--as-of-date",
                str(args.as_of_date),
                *(["--skip-model-smoke"] if bool(args.release_skip_model_smoke) else []),
            ]
        )
        payload["release_checks"] = run_release_checks(r_args)

    locked_args = [
        "--db-path",
        str(args.db_path),
        "--mode",
        "both" if bool(args.with_backtest) else "decision",
        "--as-of-date",
        str(args.as_of_date),
        "--out-json",
        "data/reports/ai_agent_stable_stocks_v1_latest.json",
    ]
    if bool(args.with_backtest):
        locked_args.extend(
            [
                "--bt-start-date",
                str(args.bt_start_date),
                *(
                    ["--bt-end-date", str(args.bt_end_date)]
                    if args.bt_end_date
                    else []
                ),
                "--persist-backtest",
            ]
        )
    run_locked_model(locked_args)

    out = Path(str(args.out_json))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(payload), ensure_ascii=True, indent=2), encoding="utf-8")
    print("===== PRODUCTION CYCLE COMPLETE =====")
    print("as_of_date:", args.as_of_date)
    print("release_checks:", "SKIPPED" if args.skip_release_checks else "DONE")
    print("mode:", "decision+backtest" if args.with_backtest else "decision")
    print("out_json:", out)
    print("=====================================")


if __name__ == "__main__":
    main()
