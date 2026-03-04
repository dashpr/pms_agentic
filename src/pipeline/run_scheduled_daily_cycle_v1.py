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
try:
    from src.qa.pipeline_health_v1 import compute_pipeline_health
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.qa.pipeline_health_v1 import compute_pipeline_health


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Scheduled daily automation cycle (cloud/local)")
    p.add_argument("--db-path", default="data/ownership.duckdb")
    p.add_argument(
        "--as-of-date",
        default=None,
        help="YYYY-MM-DD; if omitted, decision uses latest market date after repair and repair checks use UTC today.",
    )
    p.add_argument("--max-repair-rounds", type=int, default=5)
    p.add_argument("--price-csv-dir", default="data/csvs")
    p.add_argument("--prices-stale-days", type=int, default=1)
    p.add_argument("--prices-max-symbols", type=int, default=0, help="0 = no cap")
    p.add_argument("--prices-sleep-ms", type=int, default=150)
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
            # Anchor decisions to the latest available market price date.
            try:
                px = conn.execute("SELECT MAX(CAST(date AS DATE)) FROM prices_daily_v1").fetchone()[0]
                if px is not None:
                    return pd.to_datetime(px).date()
            except Exception:
                pass

            candidates: list[date] = []
            probes = [
                "SELECT MAX(CAST(date AS DATE)) FROM delivery_daily_v1",
                "SELECT MAX(CAST(as_of_date AS DATE)) FROM agentic_runs_v1",
            ]
            for q in probes:
                try:
                    d = conn.execute(q).fetchone()[0]
                    if d is not None:
                        candidates.append(pd.to_datetime(d).date())
                except Exception:
                    continue
            if candidates:
                return max(candidates)
    except Exception:
        pass
    return pd.Timestamp.now(tz="UTC").date()


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


def _freshness_gate(
    db_path: str,
    as_of: date,
    strict_rebalance_pretrade: bool,
) -> dict[str, Any]:
    core_required = {"prices_daily_v1", "delivery_daily_v1"}
    bulk_required = bool(strict_rebalance_pretrade)
    required = set(core_required)
    if bulk_required:
        required.add("bulk_block_deals")
    with duckdb.connect(str(db_path), read_only=True) as conn:
        h = compute_pipeline_health(
            conn=conn,
            as_of_date=as_of,
            require_news=False,
            require_fundamentals=False,
        )
    z = h[h["pipeline"].isin(sorted(required))].copy()
    z["status"] = z["status"].astype(str).str.upper()
    core = z[z["pipeline"].isin(sorted(core_required))].copy()
    core_fail = core[core["status"] != "WORKING"].copy()

    bulk_warn: list[dict[str, Any]] = []
    bulk_fail = pd.DataFrame(columns=z.columns)
    if bulk_required:
        bulk = z[z["pipeline"] == "bulk_block_deals"].copy()
        # Bulk feed can naturally be sparse on some periods. Treat STALE as warning,
        # but keep BROKEN as blocking for institutional visibility.
        if not bulk.empty:
            bulk_warn = bulk[bulk["status"] == "STALE"][
                ["pipeline", "status", "last_date", "lag_days", "message"]
            ].to_dict(orient="records")
            bulk_fail = bulk[bulk["status"].isin(["BROKEN", "ERROR"])].copy()
        else:
            bulk_fail = pd.DataFrame(
                [{"pipeline": "bulk_block_deals", "status": "BROKEN", "last_date": None, "lag_days": None, "message": "table_missing"}]
            )

    fail = pd.concat([core_fail, bulk_fail], ignore_index=True) if (not core_fail.empty or not bulk_fail.empty) else pd.DataFrame(columns=z.columns)
    return {
        "as_of_date": as_of.isoformat(),
        "required_pipelines": sorted(required),
        "core_required_pipelines": sorted(core_required),
        "strict_bulk_enabled": bulk_required,
        "passed": bool(fail.empty),
        "failing": fail[
            ["pipeline", "status", "last_date", "lag_days", "message"]
        ].to_dict(orient="records"),
        "warnings": bulk_warn,
    }


def main(argv=None):
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    py = str(Path(sys.executable))
    db_path = str(args.db_path)
    decision_as_of_initial = _resolve_as_of(args.as_of_date, db_path).isoformat()
    repair_as_of = (
        str(args.as_of_date)
        if args.as_of_date
        else pd.Timestamp.now(tz="UTC").date().isoformat()
    )
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
            "as_of_date": decision_as_of_initial,
            "as_of_date_initial": decision_as_of_initial,
            "repair_as_of_date": repair_as_of,
            "status": "ERROR",
            "error": f"DB missing: {db_file}",
            "steps": [],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
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
                repair_as_of,
                "--max-rounds",
                str(int(args.max_repair_rounds)),
                "--price-csv-dir",
                str(args.price_csv_dir),
                "--prices-stale-days",
                str(int(args.prices_stale_days)),
                "--prices-max-symbols",
                str(int(args.prices_max_symbols)),
                "--prices-sleep-ms",
                str(int(args.prices_sleep_ms)),
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
                repair_as_of,
                "--persist",
            ],
            cwd=repo,
        )
    )

    repair_gate = _freshness_gate(
        db_path=db_path,
        as_of=date.fromisoformat(str(repair_as_of)),
        strict_rebalance_pretrade=bool(args.strict_rebalance_pretrade),
    )
    steps.append(
        {
            "step": "required_freshness_gate",
            "cmd": "internal_freshness_gate",
            "returncode": 0 if bool(repair_gate.get("passed")) else 1,
            "status": "OK" if bool(repair_gate.get("passed")) else "ERROR",
            "stdout_tail": "",
            "stderr_tail": "" if bool(repair_gate.get("passed")) else json.dumps(repair_gate.get("failing", []), ensure_ascii=True, default=str),
        }
    )

    # Resolve decision as-of after repair has had a chance to refresh prices.
    decision_as_of = (
        str(args.as_of_date)
        if args.as_of_date
        else _resolve_as_of(None, db_path).isoformat()
    )

    decision_mode = "SKIPPED_FRESHNESS_BLOCK"
    strict_res = {
        "step": "locked_decision_strict",
        "cmd": "",
        "returncode": 1,
        "status": "ERROR",
        "stdout_tail": "",
        "stderr_tail": "Skipped due to failed required_freshness_gate",
    }
    if bool(repair_gate.get("passed")):
        strict_cmd = [
            py,
            "src/agentic/run_ai_agent_stable_stocks_v1.py",
            "--db-path",
            db_path,
            "--mode",
            "decision",
            "--as-of-date",
            decision_as_of,
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
                    decision_as_of,
                    "--out-json",
                    "data/reports/ai_agent_stable_stocks_v1_latest.json",
                ],
                cwd=repo,
            )
            steps.append(fallback_res)
            if fallback_res["status"] == "OK":
                decision_mode = "DEGRADED_FALLBACK"
    else:
        steps.append(strict_res)

    overall_ok = bool(repair_gate.get("passed")) and (decision_mode in {"STRICT", "DEGRADED_FALLBACK"})
    payload = {
        "as_of_date": decision_as_of,
        "as_of_date_initial": decision_as_of_initial,
        "repair_as_of_date": repair_as_of,
        "status": "OK" if overall_ok else "ERROR",
        "decision_mode": decision_mode,
        "required_freshness_gate": repair_gate,
        "strict_rebalance_pretrade": bool(args.strict_rebalance_pretrade),
        "allow_degraded_fallback": bool(args.allow_degraded_fallback),
        "steps": steps,
        "reports": {
            "model_report": "data/reports/ai_agent_stable_stocks_v1_latest.json",
            "pipeline_health": "data/reports/repair_required_pipelines_v1_latest.json",
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")

    print("===== SCHEDULED DAILY CYCLE v1 =====")
    print("decision_as_of_date_initial:", decision_as_of_initial)
    print("decision_as_of_date:", decision_as_of)
    print("repair_as_of_date:", repair_as_of)
    print("required_freshness_gate_passed:", bool(repair_gate.get("passed")))
    if repair_gate.get("warnings"):
        print("required_freshness_gate_warnings:", json.dumps(repair_gate.get("warnings"), ensure_ascii=True, default=str))
    print("status:", payload["status"])
    print("decision_mode:", decision_mode)
    if not overall_ok:
        print("failed_steps:")
        for s in steps:
            if str(s.get("status", "")).upper() != "OK":
                print(f"- {s.get('step')} rc={s.get('returncode')}")
                st = str(s.get("stdout_tail", "")).strip()
                er = str(s.get("stderr_tail", "")).strip()
                if st:
                    print("  stdout_tail:")
                    print(st[-800:])
                if er:
                    print("  stderr_tail:")
                    print(er[-800:])
    print("out_json:", out_path)
    print("=====================================")

    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
