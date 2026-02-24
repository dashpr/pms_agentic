from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

try:
    from src.agentic.backtest_engine_v1 import BacktestConfig, run_agentic_backtest
    from src.agentic.model_lock_ai_agent_stable_stocks_v1 import (
        BEST_TRIAL_ID,
        MODEL_ID,
        MODEL_SOURCE,
        locked_agent_weights,
        locked_policy,
        model_metadata,
    )
    from src.agentic.orchestrator_v1 import run_agentic_cycle
    from src.qa.run_quality_gates_v1 import (
        GateConfig,
        check_delivery,
        check_news_raw_optional,
        check_prices,
        check_symbol_master,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.agentic.backtest_engine_v1 import BacktestConfig, run_agentic_backtest
    from src.agentic.model_lock_ai_agent_stable_stocks_v1 import (
        BEST_TRIAL_ID,
        MODEL_ID,
        MODEL_SOURCE,
        locked_agent_weights,
        locked_policy,
        model_metadata,
    )
    from src.agentic.orchestrator_v1 import run_agentic_cycle
    from src.qa.run_quality_gates_v1 import (
        GateConfig,
        check_delivery,
        check_news_raw_optional,
        check_prices,
        check_symbol_master,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Release checks for AI Agent Stable Stocks v1 (data + model + smoke backtest)"
    )
    p.add_argument("--db-path", default="data/ownership.duckdb")
    p.add_argument("--as-of-date", default=None, help="YYYY-MM-DD. Defaults to max price date.")
    p.add_argument("--require-min-universe", type=int, default=240)
    p.add_argument("--require-min-price-symbols", type=int, default=240)
    p.add_argument("--require-min-delivery-symbols", type=int, default=220)
    p.add_argument("--require-min-history-years", type=float, default=10.0)
    p.add_argument("--smoke-backtest-days", type=int, default=252)
    p.add_argument("--skip-model-smoke", action="store_true")
    p.add_argument(
        "--out-json",
        default="data/reports/release_checks_ai_agent_stable_v1.json",
    )
    return p.parse_args(argv)


def _to_serializable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_serializable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_serializable(v) for v in x]
    if isinstance(x, tuple):
        return [_to_serializable(v) for v in x]
    if isinstance(x, (pd.Timestamp, date)):
        return x.isoformat()
    return x


def _load_data_coverage(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    x = conn.execute(
        """
        SELECT
            MIN(CAST(date AS DATE)) AS min_date,
            MAX(CAST(date AS DATE)) AS max_date,
            COUNT(*) AS rows,
            COUNT(DISTINCT UPPER(TRIM(canonical_symbol))) AS symbols
        FROM prices_daily_v1
        """
    ).fetchone()
    d_min, d_max, rows, symbols = x

    y = conn.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT UPPER(TRIM(canonical_symbol))) AS symbols
        FROM delivery_daily_v1
        """
    ).fetchone()
    d_rows, d_symbols = y

    z = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN COALESCE(in_universe, FALSE)=TRUE THEN 1 ELSE 0 END) AS active
        FROM symbol_master
        """
    ).fetchone()
    total_sym, active_sym = z

    years = 0.0
    if d_min is not None and d_max is not None:
        years = max((pd.Timestamp(d_max) - pd.Timestamp(d_min)).days / 365.25, 0.0)
    return {
        "prices_min_date": d_min,
        "prices_max_date": d_max,
        "prices_rows": int(rows or 0),
        "prices_symbols": int(symbols or 0),
        "delivery_rows": int(d_rows or 0),
        "delivery_symbols": int(d_symbols or 0),
        "symbol_master_total": int(total_sym or 0),
        "symbol_master_active": int(active_sym or 0),
        "history_years": float(years),
    }


def _assert_release_thresholds(coverage: dict[str, Any], args) -> None:
    if int(coverage["symbol_master_active"]) < int(args.require_min_universe):
        raise ValueError(
            f"active universe too low: {coverage['symbol_master_active']} < {args.require_min_universe}"
        )
    if int(coverage["prices_symbols"]) < int(args.require_min_price_symbols):
        raise ValueError(
            f"price symbol coverage too low: {coverage['prices_symbols']} < {args.require_min_price_symbols}"
        )
    if int(coverage["delivery_symbols"]) < int(args.require_min_delivery_symbols):
        raise ValueError(
            f"delivery symbol coverage too low: {coverage['delivery_symbols']} < {args.require_min_delivery_symbols}"
        )
    if float(coverage["history_years"]) < float(args.require_min_history_years):
        raise ValueError(
            f"history years too low: {coverage['history_years']:.2f} < {args.require_min_history_years:.2f}"
        )


def _resolve_as_of_date(conn: duckdb.DuckDBPyConnection, raw: str | None) -> date:
    if raw:
        return date.fromisoformat(str(raw))
    d = conn.execute("SELECT MAX(CAST(date AS DATE)) FROM prices_daily_v1").fetchone()[0]
    if d is None:
        raise ValueError("cannot resolve as_of_date: prices_daily_v1 is empty")
    return pd.to_datetime(d).date()


def run_release_checks(args) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_source": MODEL_SOURCE,
        "best_trial_id": BEST_TRIAL_ID,
        "model_metadata": model_metadata(),
        "checks": {},
    }

    # 1) Baseline data-quality gates.
    gate_cfg = GateConfig(
        min_symbol_master_rows=max(int(args.require_min_universe), 100),
        min_prices_symbols=max(int(args.require_min_price_symbols), 80),
        min_delivery_symbols=max(int(args.require_min_delivery_symbols), 80),
    )
    with duckdb.connect(str(args.db_path)) as conn:
        check_symbol_master(conn, gate_cfg)
        check_prices(conn, gate_cfg)
        check_delivery(conn, gate_cfg)
        check_news_raw_optional(conn)
    out["checks"]["data_quality_gates"] = "PASS"

    # 2) Coverage checks.
    with duckdb.connect(str(args.db_path), read_only=True) as ro_conn:
        coverage = _load_data_coverage(ro_conn)
        as_of = _resolve_as_of_date(ro_conn, args.as_of_date)
    _assert_release_thresholds(coverage, args)
    out["checks"]["coverage"] = coverage
    out["checks"]["as_of_date"] = as_of

    # 3) Locked-model smoke checks (writes audit rows).
    if not bool(args.skip_model_smoke):
        policy = locked_policy()
        weights = locked_agent_weights()
        decision = run_agentic_cycle(
            db_path=str(args.db_path),
            as_of_date=as_of,
            policy=policy,
            universe_limit=0,
            agent_weight_overrides=weights,
        )
        if int(decision.get("watchlist_size", 0)) != int(policy.watchlist_size):
            raise ValueError("decision smoke failed: watchlist size mismatch")
        psize = int(decision.get("portfolio_size", 0))
        if psize < int(policy.portfolio_min_positions) or psize > int(policy.portfolio_max_positions):
            raise ValueError("decision smoke failed: portfolio size outside policy bounds")
        out["checks"]["decision_smoke"] = {
            "run_id": decision.get("run_id"),
            "watchlist_size": int(decision.get("watchlist_size", 0)),
            "portfolio_size": psize,
            "status": decision.get("status"),
        }

        start_bt = (pd.Timestamp(as_of) - pd.Timedelta(days=max(int(args.smoke_backtest_days), 120))).date()
        cfg = BacktestConfig(
            db_path=str(args.db_path),
            start_date=start_bt.isoformat(),
            end_date=as_of.isoformat(),
            initial_capital=1_000_000.0,
            rebalance_mode="weekly",
            rebalance_weekday=2,
            fees_bps=5.0,
            slippage_bps=10.0,
            universe_limit=0,
        )
        bt = run_agentic_backtest(
            cfg=cfg,
            policy=policy,
            agent_weight_overrides=weights,
            persist=False,
        )
        stats = bt.get("stats", {})
        if "CAGR" not in stats or "Max Drawdown" not in stats or "Trade Win Rate" not in stats:
            raise ValueError("backtest smoke failed: missing key metrics")
        out["checks"]["backtest_smoke"] = {
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "stats": {
                "CAGR": float(stats.get("CAGR", 0.0)),
                "Trade Win Rate": float(stats.get("Trade Win Rate", 0.0)),
                "Max Drawdown": float(stats.get("Max Drawdown", 0.0)),
                "Sharpe": float(stats.get("Sharpe", 0.0)),
            },
        }
    else:
        out["checks"]["decision_smoke"] = "SKIPPED"
        out["checks"]["backtest_smoke"] = "SKIPPED"

    return out


def main(argv=None):
    args = parse_args(argv)
    out = run_release_checks(args)
    out_path = Path(str(args.out_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_to_serializable(out), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    coverage = out["checks"]["coverage"]
    print("===== RELEASE CHECKS: AI AGENT STABLE STOCKS V1 =====")
    print("model_id:", out["model_id"])
    print("source:", out["model_source"])
    print("best_trial_id:", out["best_trial_id"])
    print("as_of_date:", out["checks"]["as_of_date"])
    print(
        "coverage:",
        f"prices_symbols={coverage['prices_symbols']}, "
        f"delivery_symbols={coverage['delivery_symbols']}, "
        f"history_years={coverage['history_years']:.2f}",
    )
    if isinstance(out["checks"]["decision_smoke"], dict):
        print("decision_run_id:", out["checks"]["decision_smoke"]["run_id"])
        print("portfolio_size:", out["checks"]["decision_smoke"]["portfolio_size"])
    if isinstance(out["checks"]["backtest_smoke"], dict):
        s = out["checks"]["backtest_smoke"]["stats"]
        print(f"smoke_CAGR: {float(s['CAGR']):.2%}")
        print(f"smoke_trade_win: {float(s['Trade Win Rate']):.2%}")
        print(f"smoke_maxdd: {float(s['Max Drawdown']):.2%}")
    print("out_json:", out_path)
    print("======================================================")


if __name__ == "__main__":
    main()
