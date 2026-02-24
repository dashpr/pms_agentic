from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

try:
    from src.agentic.backtest_engine_v1 import BacktestConfig, run_agentic_backtest
    from src.agentic.model_lock_ai_agent_stable_stocks_v1 import (
        MODEL_ID,
        MODEL_VERSION,
        locked_agent_weights,
        locked_policy,
        locked_reference_metrics,
        model_metadata,
    )
    from src.agentic.orchestrator_v1 import run_agentic_cycle
    from src.data_layer.sync_delivery_daily_to_v1 import main as sync_delivery_v1
    from src.ownership.backfill_bulk_archives_to_local_v1 import main as backfill_bulk_local
    from src.ownership.collect_bulk_deals_nse import run_nse_bulk_ingestion
    from src.ownership.collect_delivery_data import run_delivery_update
    from src.ownership.import_bulk_block_deals_from_files_v1 import main as import_bulk_local
    from src.qa.pipeline_health_v1 import compute_pipeline_health
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.agentic.backtest_engine_v1 import BacktestConfig, run_agentic_backtest
    from src.agentic.model_lock_ai_agent_stable_stocks_v1 import (
        MODEL_ID,
        MODEL_VERSION,
        locked_agent_weights,
        locked_policy,
        locked_reference_metrics,
        model_metadata,
    )
    from src.agentic.orchestrator_v1 import run_agentic_cycle
    from src.data_layer.sync_delivery_daily_to_v1 import main as sync_delivery_v1
    from src.ownership.backfill_bulk_archives_to_local_v1 import main as backfill_bulk_local
    from src.ownership.collect_bulk_deals_nse import run_nse_bulk_ingestion
    from src.ownership.collect_delivery_data import run_delivery_update
    from src.ownership.import_bulk_block_deals_from_files_v1 import main as import_bulk_local
    from src.qa.pipeline_health_v1 import compute_pipeline_health


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run locked AI Agent Stable Stocks v1 model")
    p.add_argument("--db-path", default="data/ownership.duckdb")
    p.add_argument("--mode", default="both", choices=["decision", "backtest", "both"])
    p.add_argument("--as-of-date", default=date.today().isoformat(), help="YYYY-MM-DD")

    p.add_argument("--bt-start-date", default="2016-01-01")
    p.add_argument("--bt-end-date", default=None)
    p.add_argument("--bt-initial-capital", type=float, default=1_000_000.0)
    p.add_argument("--bt-rebalance-mode", default="weekly", choices=["weekly", "daily"])
    p.add_argument("--bt-rebalance-weekday", type=int, default=2, help="0=Mon ... 6=Sun")
    p.add_argument("--bt-fees-bps", type=float, default=5.0)
    p.add_argument("--bt-slippage-bps", type=float, default=10.0)
    p.add_argument("--bt-universe-limit", type=int, default=0)
    p.add_argument("--persist-backtest", action="store_true")
    p.add_argument("--pretrade-heal-attempts", type=int, default=2)
    p.add_argument("--pretrade-heal-bulk-lookback-days", type=int, default=120)
    p.add_argument("--pretrade-heal-bulk-local-input-dir", default="data_raw/nse_bulk")
    p.add_argument(
        "--enforce-full-pretrade-gate",
        action="store_true",
        help="If set, block decision run unless full rebalance gate is WORKING.",
    )
    p.add_argument(
        "--strict-rebalance-pretrade",
        action="store_true",
        help="If set, rebalance_allowed requires strict full gate (including bulk feed).",
    )

    p.add_argument("--out-json", default="data/reports/ai_agent_stable_stocks_v1_latest.json")
    return p.parse_args(argv)


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, tuple):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, (pd.Timestamp, date, datetime)):
        return obj.isoformat()
    return obj


def _evaluate_pretrade_gate(
    db_path: str,
    as_of: date,
    required_pipelines: set[str],
) -> dict[str, Any]:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        health = compute_pipeline_health(
            conn=conn,
            as_of_date=as_of,
            require_news=False,
            require_fundamentals=False,
        )
    hv = health[health["pipeline"].isin(sorted(required_pipelines))].copy()
    hv["status"] = hv["status"].astype(str).str.upper()
    fail = hv[hv["status"] != "WORKING"].copy()
    failing = fail[["pipeline", "status", "last_date", "lag_days", "message"]].to_dict(orient="records")
    return {
        "as_of_date": str(as_of),
        "required_pipelines": sorted(required_pipelines),
        "passed": bool(fail.empty),
        "failing": failing,
    }


def _raise_gate_block(stage: str, gate_result: dict[str, Any]) -> None:
    failing = gate_result.get("failing", []) or []
    detail = "; ".join(
        [
            f"{r.get('pipeline')}={r.get('status')}({r.get('message')})"
            for r in failing
        ]
    )
    raise RuntimeError(
        f"PRETRADE_GATE_BLOCKED[{stage}] as_of={gate_result.get('as_of_date')} :: {detail}"
    )


def _auto_heal_required_pipelines(
    db_path: str,
    as_of: date,
    required_pipelines: set[str],
    max_attempts: int,
    bulk_lookback_days: int,
    bulk_local_input_dir: str,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    gate = _evaluate_pretrade_gate(
        db_path=db_path,
        as_of=as_of,
        required_pipelines=required_pipelines,
    )
    for i in range(1, max(0, int(max_attempts)) + 1):
        if bool(gate.get("passed")):
            break
        failing_before = {str(r.get("pipeline")) for r in (gate.get("failing") or [])}
        actions: dict[str, Any] = {}

        if "delivery_daily_v1" in failing_before:
            try:
                actions["update_delivery_live"] = run_delivery_update(
                    max_lookback=21,
                    prefer_cache=True,
                    raise_on_error=False,
                )
            except Exception as e:
                actions["update_delivery_live"] = {"ok": False, "error": str(e)}
            try:
                sync_delivery_v1()
                actions["sync_delivery_v1"] = {"ok": True}
            except Exception as e:
                actions["sync_delivery_v1"] = {"ok": False, "error": str(e)}

        if "bulk_block_deals" in failing_before:
            try:
                actions["update_bulk_live"] = run_nse_bulk_ingestion(
                    lookback_days=int(bulk_lookback_days),
                    batch_days=30,
                    raise_on_error=False,
                )
            except Exception as e:
                actions["update_bulk_live"] = {"ok": False, "error": str(e)}
            if not bool((actions.get("update_bulk_live") or {}).get("ok", False)):
                try:
                    backfill_bulk_local(["--days", str(int(bulk_lookback_days)), "--out-dir", str(bulk_local_input_dir)])
                    actions["download_bulk_archives_local"] = {
                        "ok": True,
                        "output_dir": str(bulk_local_input_dir),
                    }
                except BaseException as e:
                    actions["download_bulk_archives_local"] = {
                        "ok": False,
                        "output_dir": str(bulk_local_input_dir),
                        "error": str(e),
                    }
                try:
                    import_bulk_local(["--input-dir", str(bulk_local_input_dir), "--recursive"])
                    actions["update_bulk_from_local"] = {
                        "ok": True,
                        "input_dir": str(bulk_local_input_dir),
                    }
                except BaseException as e:
                    actions["update_bulk_from_local"] = {
                        "ok": False,
                        "input_dir": str(bulk_local_input_dir),
                        "error": str(e),
                    }

        gate_after = _evaluate_pretrade_gate(
            db_path=db_path,
            as_of=as_of,
            required_pipelines=required_pipelines,
        )
        attempts.append(
            {
                "attempt": i,
                "failing_before": sorted(failing_before),
                "actions": actions,
                "gate_after": gate_after,
            }
        )
        gate = gate_after
    return {
        "attempt_count": len(attempts),
        "attempts": attempts,
        "final_gate": gate,
    }


def main(argv=None):
    args = parse_args(argv)
    out: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "frozen_model": model_metadata(),
        "decision_cycle": None,
        "backtest": None,
        "pretrade_gate": {},
    }

    policy = locked_policy()
    weights = locked_agent_weights()
    as_of = date.fromisoformat(str(args.as_of_date))

    if args.mode in {"decision", "both"}:
        decision_required = {"prices_daily_v1", "delivery_daily_v1"}
        rebalance_required = {"prices_daily_v1", "delivery_daily_v1", "bulk_block_deals"}
        rebalance_core_required = {"prices_daily_v1", "delivery_daily_v1", "agentic_runs_v1", "agentic_consensus_v1"}
        pre_gate = _evaluate_pretrade_gate(
            db_path=str(args.db_path),
            as_of=as_of,
            required_pipelines=decision_required,
        )
        out["pretrade_gate"]["pre_decision"] = pre_gate
        heal = _auto_heal_required_pipelines(
            db_path=str(args.db_path),
            as_of=as_of,
            required_pipelines=rebalance_required,
            max_attempts=int(args.pretrade_heal_attempts),
            bulk_lookback_days=int(args.pretrade_heal_bulk_lookback_days),
            bulk_local_input_dir=str(args.pretrade_heal_bulk_local_input_dir),
        )
        out["pretrade_gate"]["auto_heal"] = heal
        pre_gate = _evaluate_pretrade_gate(
            db_path=str(args.db_path),
            as_of=as_of,
            required_pipelines=decision_required,
        )
        out["pretrade_gate"]["pre_decision"] = pre_gate
        if not bool(pre_gate.get("passed")):
            out_path = Path(str(args.out_json))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(_sanitize(out), ensure_ascii=True, indent=2), encoding="utf-8")
            _raise_gate_block("pre_decision", pre_gate)

        decision = run_agentic_cycle(
            db_path=str(args.db_path),
            as_of_date=as_of,
            policy=policy,
            universe_limit=0,
            agent_weight_overrides=weights,
        )
        out["decision_cycle"] = decision

        post_gate = _evaluate_pretrade_gate(
            db_path=str(args.db_path),
            as_of=as_of,
            required_pipelines=rebalance_required,
        )
        post_gate_full = _evaluate_pretrade_gate(
            db_path=str(args.db_path),
            as_of=as_of,
            required_pipelines=rebalance_required.union({"agentic_runs_v1", "agentic_consensus_v1"}),
        )
        post_gate_core = _evaluate_pretrade_gate(
            db_path=str(args.db_path),
            as_of=as_of,
            required_pipelines=rebalance_core_required,
        )
        out["pretrade_gate"]["post_decision"] = post_gate
        out["pretrade_gate"]["post_decision_core"] = post_gate_core
        out["pretrade_gate"]["post_decision_full"] = post_gate_full
        strict_rebalance = bool(args.strict_rebalance_pretrade) or bool(args.enforce_full_pretrade_gate)
        if strict_rebalance:
            rebalance_allowed = bool(post_gate_full.get("passed"))
            rebalance_mode = "STRICT"
        else:
            rebalance_allowed = bool(post_gate_core.get("passed"))
            rebalance_mode = "DEGRADED_ALLOWED" if rebalance_allowed and (not bool(post_gate_full.get("passed"))) else "NORMAL"
        out["pretrade_gate"]["rebalance_allowed"] = bool(rebalance_allowed)
        out["pretrade_gate"]["rebalance_mode"] = rebalance_mode
        out["pretrade_gate"]["rebalance_risk_flags"] = (
            [f"{r.get('pipeline')}:{r.get('message')}" for r in (post_gate_full.get("failing") or [])]
            if rebalance_mode == "DEGRADED_ALLOWED"
            else []
        )
        if bool(args.enforce_full_pretrade_gate) and not bool(post_gate_full.get("passed")):
            out_path = Path(str(args.out_json))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(_sanitize(out), ensure_ascii=True, indent=2), encoding="utf-8")
            _raise_gate_block("post_decision_full", post_gate_full)

    if args.mode in {"backtest", "both"}:
        cfg = BacktestConfig(
            db_path=str(args.db_path),
            start_date=str(args.bt_start_date),
            end_date=str(args.bt_end_date) if args.bt_end_date else None,
            initial_capital=float(args.bt_initial_capital),
            rebalance_mode=str(args.bt_rebalance_mode),
            rebalance_weekday=int(args.bt_rebalance_weekday),
            fees_bps=float(args.bt_fees_bps),
            slippage_bps=float(args.bt_slippage_bps),
            universe_limit=int(args.bt_universe_limit),
        )
        bt = run_agentic_backtest(
            cfg=cfg,
            policy=policy,
            agent_weight_overrides=weights,
            persist=bool(args.persist_backtest),
        )
        out["backtest"] = {
            "config": asdict(cfg),
            "run_id": bt.get("run_id"),
            "stats": bt.get("stats", {}),
            "reference_metrics": locked_reference_metrics(),
        }

    out_path = Path(str(args.out_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_sanitize(out), ensure_ascii=True, indent=2), encoding="utf-8")

    print("===== AI AGENT STABLE STOCKS V1 =====")
    print("model_id:", MODEL_ID)
    print("model_version:", MODEL_VERSION)
    pg = out.get("pretrade_gate", {}) or {}
    if pg:
        pre = pg.get("pre_decision", {}) or {}
        heal = pg.get("auto_heal", {}) or {}
        print("pretrade_gate_pre_decision_passed:", pre.get("passed"))
        if heal:
            print("pretrade_gate_heal_attempts:", int(heal.get("attempt_count", 0)))
        print("pretrade_gate_rebalance_allowed:", pg.get("rebalance_allowed"))
        print("pretrade_gate_rebalance_mode:", pg.get("rebalance_mode"))
    if out.get("decision_cycle"):
        dc = out["decision_cycle"]
        print("decision_run_id:", dc.get("run_id"))
        print("as_of_date:", dc.get("as_of_date"))
        print("portfolio_size:", dc.get("portfolio_size"))
    if out.get("backtest"):
        st = out["backtest"]["stats"]
        print(f"CAGR: {float(st.get('CAGR', 0.0)):.2%}")
        print(f"Trade Win: {float(st.get('Trade Win Rate', 0.0)):.2%}")
        print(f"MaxDD: {float(st.get('Max Drawdown', 0.0)):.2%}")
        print(f"Sharpe: {float(st.get('Sharpe', 0.0)):.2f}")
    print("out_json:", out_path)
    print("======================================")


if __name__ == "__main__":
    main()
