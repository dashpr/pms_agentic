from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Scheduled daily automation cycle (cloud/local)")
    p.add_argument("--db-path", default="data/ownership.duckdb")
    p.add_argument("--as-of-date", default=None, help="YYYY-MM-DD; default latest prices_daily_v1 date")
    p.add_argument("--max-repair-rounds", type=int, default=5)
    p.add_argument("--bulk-lookback-days", type=int, default=180)
    p.add_argument("--strict-rebalance-pretrade", action="store_true")
    p.add_argument("--allow-degraded-fallback", action="store_true")
    p.add_argument("--out-json", default="data/reports/scheduled_daily_cycle_v1_latest.json")
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


def _run_step(name: str, cmd: list[str], cwd: Path) -> dict[str, Any]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    out = (p.stdout or "")[-4000:]
    err = (p.stderr or "")[-4000:]
    return {
        "step": name,
        "cmd": " ".join(cmd),
        "returncode": int(p.returncode),
        "status": "OK" if int(p.returncode) == 0 else "ERROR",
        "stdout_tail": out,
        "stderr_tail": err,
    }


def main(argv=None):
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    py = str(Path(sys.executable))
    db_path = str(args.db_path)
    as_of = _resolve_as_of(args.as_of_date, db_path).isoformat()
    out_path = Path(str(args.out_json))
    if not out_path.is_absolute():
        out_path = repo / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not Path(db_path).is_absolute():
        db_file = repo / db_path
    else:
        db_file = Path(db_path)
    if not db_file.exists():
        payload = {
            "as_of_date": as_of,
            "status": "ERROR",
            "error": f"DB missing: {db_file}",
            "steps": [],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        raise SystemExit(1)

    steps: list[dict[str, Any]] = []

    steps.append(
        _run_step(
            "repair_required_pipelines",
            [
                py,
                "src/pipeline/repair_required_pipelines_v1.py",
                "--db-path",
                db_path,
                "--as-of-date",
                as_of,
                "--max-rounds",
                str(int(args.max_repair_rounds)),
                "--bulk-lookback-days",
                str(int(args.bulk_lookback_days)),
            ],
            cwd=repo,
        )
    )
    steps.append(
        _run_step(
            "pipeline_health_checks",
            [
                py,
                "src/qa/run_pipeline_health_checks_v1.py",
                "--db-path",
                db_path,
                "--as-of-date",
                as_of,
                "--persist",
            ],
            cwd=repo,
        )
    )

    strict_cmd = [
        py,
        "src/agentic/run_ai_agent_stable_stocks_v1.py",
        "--db-path",
        db_path,
        "--mode",
        "decision",
        "--as-of-date",
        as_of,
        "--out-json",
        "data/reports/ai_agent_stable_stocks_v1_latest.json",
    ]
    if bool(args.strict_rebalance_pretrade):
        strict_cmd.append("--strict-rebalance-pretrade")
    strict_res = _run_step("locked_decision_strict", strict_cmd, cwd=repo)
    steps.append(strict_res)

    decision_mode = "STRICT" if strict_res["status"] == "OK" else "FAILED"
    if strict_res["status"] != "OK" and bool(args.allow_degraded_fallback):
        fallback_res = _run_step(
            "locked_decision_degraded_fallback",
            [
                py,
                "src/agentic/run_ai_agent_stable_stocks_v1.py",
                "--db-path",
                db_path,
                "--mode",
                "decision",
                "--as-of-date",
                as_of,
                "--out-json",
                "data/reports/ai_agent_stable_stocks_v1_latest.json",
            ],
            cwd=repo,
        )
        steps.append(fallback_res)
        if fallback_res["status"] == "OK":
            decision_mode = "DEGRADED_FALLBACK"

    overall_ok = decision_mode in {"STRICT", "DEGRADED_FALLBACK"}
    payload = {
        "as_of_date": as_of,
        "status": "OK" if overall_ok else "ERROR",
        "decision_mode": decision_mode,
        "strict_rebalance_pretrade": bool(args.strict_rebalance_pretrade),
        "allow_degraded_fallback": bool(args.allow_degraded_fallback),
        "steps": steps,
        "reports": {
            "model_report": "data/reports/ai_agent_stable_stocks_v1_latest.json",
            "pipeline_health": "data/reports/repair_required_pipelines_v1_latest.json",
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print("===== SCHEDULED DAILY CYCLE v1 =====")
    print("as_of_date:", as_of)
    print("status:", payload["status"])
    print("decision_mode:", decision_mode)
    print("out_json:", out_path)
    print("=====================================")

    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
