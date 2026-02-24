from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

try:
    from src.agentic.backtest_engine_v1 import BacktestConfig, run_agentic_backtest
    from src.agentic.contracts_v1 import OrchestrationPolicy
    from src.agentic.optimizer_v1 import OptimizerConfig, optimize_policy_for_period
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.agentic.backtest_engine_v1 import BacktestConfig, run_agentic_backtest
    from src.agentic.contracts_v1 import OrchestrationPolicy
    from src.agentic.optimizer_v1 import OptimizerConfig, optimize_policy_for_period


@dataclass
class WalkForwardConfig:
    db_path: str = "data/ownership.duckdb"
    start_date: str = "2016-01-01"
    end_date: str = "2026-02-19"
    train_days: int = 756
    test_days: int = 252
    step_days: int = 126
    optimizer_trials_per_fold: int = 12
    min_trade_win_rate: float = 0.52
    max_drawdown_floor: float = -0.18
    persist_results: bool = True


def _ensure_walkforward_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agentic_walkforward_folds_v1 (
            run_ts TIMESTAMP,
            fold_id INTEGER,
            train_start DATE,
            train_end DATE,
            test_start DATE,
            test_end DATE,
            cagr DOUBLE,
            sharpe DOUBLE,
            max_drawdown DOUBLE,
            trade_win_rate DOUBLE,
            daily_win_rate DOUBLE,
            policy_json VARCHAR,
            agent_weights_json VARCHAR
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agentic_walkforward_summary_v1 (
            run_ts TIMESTAMP,
            start_date DATE,
            end_date DATE,
            folds INTEGER,
            median_cagr DOUBLE,
            mean_cagr DOUBLE,
            cagr_std DOUBLE,
            median_trade_win DOUBLE,
            worst_max_drawdown DOUBLE,
            median_sharpe DOUBLE,
            config_json VARCHAR
        )
        """
    )


def _generate_folds(cfg: WalkForwardConfig) -> list[dict[str, Any]]:
    start = pd.to_datetime(cfg.start_date).date()
    end = pd.to_datetime(cfg.end_date).date()
    folds: list[dict[str, Any]] = []
    cursor = start
    k = 0
    while True:
        train_start = cursor
        train_end = train_start + timedelta(days=int(cfg.train_days) - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=int(cfg.test_days) - 1)
        if test_end > end:
            break
        k += 1
        folds.append(
            {
                "fold_id": k,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        cursor = cursor + timedelta(days=int(cfg.step_days))
    return folds


def run_walkforward_validation(
    wf_cfg: WalkForwardConfig,
    bt_template: BacktestConfig,
    base_policy: OrchestrationPolicy,
) -> dict[str, Any]:
    folds = _generate_folds(wf_cfg)
    if not folds:
        raise ValueError("No walk-forward folds generated for the given date range/settings")

    fold_rows: list[dict[str, Any]] = []
    for f in folds:
        opt_cfg = OptimizerConfig(
            db_path=wf_cfg.db_path,
            train_start_date=str(f["train_start"]),
            train_end_date=str(f["train_end"]),
            max_trials=int(wf_cfg.optimizer_trials_per_fold),
            min_trade_win_rate=float(wf_cfg.min_trade_win_rate),
            max_drawdown_floor=float(wf_cfg.max_drawdown_floor),
            persist_results=False,
        )
        opt_out = optimize_policy_for_period(
            opt_cfg=opt_cfg,
            bt_template=bt_template,
            base_policy=base_policy,
        )
        best_policy = OrchestrationPolicy(**opt_out["best_policy"])
        best_weights = dict(opt_out["best_agent_weights"])

        test_cfg = replace(
            bt_template,
            db_path=wf_cfg.db_path,
            start_date=str(f["test_start"]),
            end_date=str(f["test_end"]),
        )
        test_bt = run_agentic_backtest(
            cfg=test_cfg,
            policy=best_policy,
            agent_weight_overrides=best_weights,
            persist=False,
        )
        st = test_bt["stats"]
        fold_rows.append(
            {
                "fold_id": int(f["fold_id"]),
                "train_start": f["train_start"],
                "train_end": f["train_end"],
                "test_start": f["test_start"],
                "test_end": f["test_end"],
                "cagr": float(st.get("CAGR", 0.0)),
                "sharpe": float(st.get("Sharpe", 0.0)),
                "max_drawdown": float(st.get("Max Drawdown", 0.0)),
                "trade_win_rate": float(st.get("Trade Win Rate", 0.0)),
                "daily_win_rate": float(st.get("Daily Win Rate", 0.0)),
                "policy_json": json.dumps(asdict(best_policy), ensure_ascii=True),
                "agent_weights_json": json.dumps(best_weights, ensure_ascii=True),
            }
        )

    folds_df = pd.DataFrame(fold_rows).sort_values("fold_id").reset_index(drop=True)
    summary = {
        "start_date": wf_cfg.start_date,
        "end_date": wf_cfg.end_date,
        "folds": int(len(folds_df)),
        "median_cagr": float(folds_df["cagr"].median()),
        "mean_cagr": float(folds_df["cagr"].mean()),
        "cagr_std": float(folds_df["cagr"].std(ddof=0)),
        "median_trade_win": float(folds_df["trade_win_rate"].median()),
        "worst_max_drawdown": float(folds_df["max_drawdown"].min()),
        "median_sharpe": float(folds_df["sharpe"].median()),
        "config": asdict(wf_cfg),
    }

    if wf_cfg.persist_results:
        conn = duckdb.connect(wf_cfg.db_path)
        try:
            _ensure_walkforward_tables(conn)
            z = folds_df.copy()
            z.insert(0, "run_ts", pd.Timestamp.utcnow())
            conn.register("wf_folds", z)
            conn.execute("INSERT INTO agentic_walkforward_folds_v1 SELECT * FROM wf_folds")
            conn.unregister("wf_folds")

            srow = pd.DataFrame(
                [
                    {
                        "run_ts": pd.Timestamp.utcnow(),
                        "start_date": wf_cfg.start_date,
                        "end_date": wf_cfg.end_date,
                        "folds": int(summary["folds"]),
                        "median_cagr": float(summary["median_cagr"]),
                        "mean_cagr": float(summary["mean_cagr"]),
                        "cagr_std": float(summary["cagr_std"]),
                        "median_trade_win": float(summary["median_trade_win"]),
                        "worst_max_drawdown": float(summary["worst_max_drawdown"]),
                        "median_sharpe": float(summary["median_sharpe"]),
                        "config_json": json.dumps(summary["config"], ensure_ascii=True),
                    }
                ]
            )
            conn.register("wf_sum", srow)
            conn.execute("INSERT INTO agentic_walkforward_summary_v1 SELECT * FROM wf_sum")
            conn.unregister("wf_sum")
        finally:
            conn.close()

    return {
        "summary": summary,
        "folds": folds_df,
    }
