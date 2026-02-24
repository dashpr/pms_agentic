from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
import time
from typing import Any

import duckdb
import pandas as pd

try:
    from src.agentic.backtest_engine_v1 import BacktestConfig, run_agentic_backtest
    from src.agentic.contracts_v1 import OrchestrationPolicy
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.agentic.backtest_engine_v1 import BacktestConfig, run_agentic_backtest
    from src.agentic.contracts_v1 import OrchestrationPolicy


@dataclass
class OptimizerConfig:
    db_path: str = "data/ownership.duckdb"
    train_start_date: str = "2018-01-01"
    train_end_date: str = "2026-01-31"
    max_trials: int = 24
    min_trade_win_rate: float = 0.52
    max_drawdown_floor: float = -0.18
    objective_mode: str = "risk_adjusted"  # risk_adjusted | cagr_max
    full_grid: bool = False
    progress_every: int = 0
    checkpoint_csv_path: str = ""
    checkpoint_every: int = 0
    resume_from_checkpoint: bool = False
    persist_results: bool = True


def _ensure_optimizer_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agentic_optimizer_results_v1 (
            run_ts TIMESTAMP,
            trial_id INTEGER,
            train_start_date DATE,
            train_end_date DATE,
            cagr DOUBLE,
            sharpe DOUBLE,
            max_drawdown DOUBLE,
            trade_win_rate DOUBLE,
            daily_win_rate DOUBLE,
            score DOUBLE,
            pass_constraints BOOLEAN,
            objective_mode VARCHAR,
            policy_json VARCHAR,
            agent_weights_json VARCHAR
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agentic_optimizer_best_v1 (
            run_ts TIMESTAMP,
            train_start_date DATE,
            train_end_date DATE,
            best_trial_id INTEGER,
            cagr DOUBLE,
            sharpe DOUBLE,
            max_drawdown DOUBLE,
            trade_win_rate DOUBLE,
            score DOUBLE,
            objective_mode VARCHAR,
            policy_json VARCHAR,
            agent_weights_json VARCHAR
        )
        """
    )


def _candidate_agent_weights() -> list[dict[str, float]]:
    return [
        {},  # default
        {"technical": 0.50, "ownership": 0.20, "fundamental": 0.20, "news": 0.10},
        {"technical": 0.45, "ownership": 0.30, "fundamental": 0.20, "news": 0.05},
        {"technical": 0.35, "ownership": 0.35, "fundamental": 0.25, "news": 0.05},
        {"technical": 0.30, "ownership": 0.30, "fundamental": 0.35, "news": 0.05},
        {"technical": 0.55, "ownership": 0.25, "fundamental": 0.15, "news": 0.05},
    ]


def _policy_grid(base: OrchestrationPolicy) -> list[OrchestrationPolicy]:
    buy_vals = [0.62, 0.64, 0.66, 0.68, 0.70]
    sell_vals = [0.28, 0.30, 0.32, 0.34, 0.36]
    tgt_vals = [10, 11, 12, 13, 14, 15]
    stale_vals = [3, 5, 7]
    liq_vals = [2.0e7, 5.0e7, 8.0e7]
    out: list[OrchestrationPolicy] = []
    for b in buy_vals:
        for s in sell_vals:
            if s >= b:
                continue
            for tgt in tgt_vals:
                for st in stale_vals:
                    for liq in liq_vals:
                        p = replace(
                            base,
                            consensus_buy_threshold=float(b),
                            consensus_sell_threshold=float(s),
                            portfolio_target_positions=int(tgt),
                            max_price_staleness_days=int(st),
                            min_median_turnover_inr=float(liq),
                        )
                        p.validate()
                        out.append(p)
    return out


def _sample_candidates(
    base_policy: OrchestrationPolicy,
    max_trials: int,
) -> list[tuple[OrchestrationPolicy, dict[str, float]]]:
    grid = _policy_grid(base_policy)
    aw = _candidate_agent_weights()
    candidates: list[tuple[OrchestrationPolicy, dict[str, float]]] = []
    for p in grid:
        for w in aw:
            candidates.append((p, w))
    m = int(max(1, max_trials))
    if len(candidates) > m:
        step = max(1, len(candidates) // m)
        candidates = candidates[::step][:m]
    return candidates


def _score_candidate(
    stats: dict[str, float],
    min_trade_win_rate: float,
    max_drawdown_floor: float,
    objective_mode: str,
) -> tuple[float, bool]:
    cagr = float(stats.get("CAGR", 0.0))
    sharpe = float(stats.get("Sharpe", 0.0))
    maxdd = float(stats.get("Max Drawdown", 0.0))
    twr = float(stats.get("Trade Win Rate", 0.0))

    pass_constraints = (twr >= min_trade_win_rate) and (maxdd >= max_drawdown_floor)

    mode = str(objective_mode).strip().lower()
    if mode == "cagr_max":
        return cagr, bool(pass_constraints)

    dd_pen = max(0.0, abs(maxdd) - 0.15)
    tw_pen = max(0.0, 0.55 - twr)
    sh_pen = max(0.0, 1.0 - sharpe)
    score = cagr - (1.25 * dd_pen) - (0.60 * tw_pen) - (0.20 * sh_pen)
    return float(score), bool(pass_constraints)


def annotate_trials_with_constraints(
    trials_df: pd.DataFrame,
    min_trade_win_rate: float,
    max_drawdown_floor: float,
    objective_mode: str = "risk_adjusted",
) -> pd.DataFrame:
    x = trials_df.copy()
    scores = []
    passes = []
    for cagr, sharpe, maxdd, twr in zip(
        x["cagr"], x["sharpe"], x["max_drawdown"], x["trade_win_rate"]
    ):
        st = {
            "CAGR": float(cagr),
            "Sharpe": float(sharpe),
            "Max Drawdown": float(maxdd),
            "Trade Win Rate": float(twr),
        }
        sc, ps = _score_candidate(
            stats=st,
            min_trade_win_rate=float(min_trade_win_rate),
            max_drawdown_floor=float(max_drawdown_floor),
            objective_mode=str(objective_mode),
        )
        scores.append(float(sc))
        passes.append(bool(ps))
    x["score"] = scores
    x["pass_constraints"] = passes
    x["objective_mode"] = str(objective_mode)
    return x


def select_best_trial(
    annotated_trials: pd.DataFrame,
) -> dict[str, Any]:
    if annotated_trials.empty:
        raise ValueError("annotated_trials is empty")
    x = annotated_trials.copy().sort_values("trial_id")
    best_row = None
    for _, row in x.iterrows():
        if best_row is None:
            best_row = row
            continue
        b_pass = bool(best_row["pass_constraints"])
        c_pass = bool(row["pass_constraints"])
        replace_best = False
        if c_pass and not b_pass:
            replace_best = True
        elif c_pass == b_pass:
            if float(row["score"]) > float(best_row["score"]):
                replace_best = True
            elif (
                float(row["score"]) == float(best_row["score"])
                and float(row["cagr"]) > float(best_row["cagr"])
            ):
                replace_best = True
        if replace_best:
            best_row = row
    if best_row is None:
        raise RuntimeError("Unable to select best trial")
    return dict(best_row)


def evaluate_trial_grid(
    opt_cfg: OptimizerConfig,
    bt_template: BacktestConfig,
    base_policy: OrchestrationPolicy,
) -> pd.DataFrame:
    if opt_cfg.full_grid:
        candidates = _sample_candidates(base_policy, max_trials=10_000_000)
    else:
        candidates = _sample_candidates(base_policy, opt_cfg.max_trials)
    base_snapshot_cache: dict[date, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    total = len(candidates)
    t0 = time.perf_counter()
    done_trial_ids: set[int] = set()
    checkpoint_path = str(opt_cfg.checkpoint_csv_path).strip()
    use_checkpoint = checkpoint_path != ""

    if use_checkpoint and bool(opt_cfg.resume_from_checkpoint):
        cp = Path(checkpoint_path)
        if cp.exists():
            try:
                old = pd.read_csv(cp)
                if "trial_id" in old.columns:
                    done_trial_ids = {
                        int(x) for x in old["trial_id"].dropna().astype(int).tolist()
                    }
                    if len(done_trial_ids) > 0:
                        print(
                            f"[optimizer] resume: loaded {len(done_trial_ids)} completed trials from {cp}",
                            flush=True,
                        )
            except Exception:
                pass

    pending_buffer: list[dict[str, Any]] = []

    def _flush_checkpoint(buf: list[dict[str, Any]]) -> None:
        if not use_checkpoint or not buf:
            return
        cp = Path(checkpoint_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        z = pd.DataFrame(buf)
        if not cp.exists() or cp.stat().st_size == 0:
            z.to_csv(cp, index=False)
        else:
            z.to_csv(cp, mode="a", header=False, index=False)

    for i, (policy, agent_weights) in enumerate(candidates, start=1):
        if i in done_trial_ids:
            continue
        cfg = replace(
            bt_template,
            db_path=opt_cfg.db_path,
            start_date=opt_cfg.train_start_date,
            end_date=opt_cfg.train_end_date,
        )
        bt = run_agentic_backtest(
            cfg=cfg,
            policy=policy,
            agent_weight_overrides=agent_weights,
            base_snapshot_cache=base_snapshot_cache,
            persist=False,
        )
        st = bt["stats"]
        rows.append(
            row := {
                "trial_id": i,
                "train_start_date": opt_cfg.train_start_date,
                "train_end_date": opt_cfg.train_end_date,
                "cagr": float(st.get("CAGR", 0.0)),
                "sharpe": float(st.get("Sharpe", 0.0)),
                "max_drawdown": float(st.get("Max Drawdown", 0.0)),
                "trade_win_rate": float(st.get("Trade Win Rate", 0.0)),
                "daily_win_rate": float(st.get("Daily Win Rate", 0.0)),
                "policy_json": json.dumps(asdict(policy), ensure_ascii=True),
                "agent_weights_json": json.dumps(agent_weights, ensure_ascii=True),
            }
        )
        pending_buffer.append(row)
        if int(opt_cfg.checkpoint_every) > 0 and len(pending_buffer) >= int(opt_cfg.checkpoint_every):
            _flush_checkpoint(pending_buffer)
            pending_buffer = []
        if int(opt_cfg.progress_every) > 0 and (i % int(opt_cfg.progress_every) == 0 or i == total):
            elapsed = time.perf_counter() - t0
            print(
                f"[optimizer] trials {i}/{total} complete | elapsed {elapsed:.1f}s",
                flush=True,
            )
    if pending_buffer:
        _flush_checkpoint(pending_buffer)
    if not rows:
        if use_checkpoint and Path(checkpoint_path).exists():
            out = pd.read_csv(checkpoint_path)
            if out.empty:
                raise RuntimeError("No trial results generated")
            return out
        raise RuntimeError("No trial results generated")

    out = pd.DataFrame(rows)
    if use_checkpoint and Path(checkpoint_path).exists():
        try:
            cp_df = pd.read_csv(checkpoint_path)
            if not cp_df.empty:
                out = pd.concat([cp_df, out], ignore_index=True)
                out = out.drop_duplicates(subset=["trial_id"], keep="last")
        except Exception:
            pass
    return out.sort_values("trial_id").reset_index(drop=True)


def optimize_policy_for_period(
    opt_cfg: OptimizerConfig,
    bt_template: BacktestConfig,
    base_policy: OrchestrationPolicy,
) -> dict[str, Any]:
    trials_df = evaluate_trial_grid(
        opt_cfg=opt_cfg,
        bt_template=bt_template,
        base_policy=base_policy,
    )
    annotated = annotate_trials_with_constraints(
        trials_df=trials_df,
        min_trade_win_rate=float(opt_cfg.min_trade_win_rate),
        max_drawdown_floor=float(opt_cfg.max_drawdown_floor),
        objective_mode=str(opt_cfg.objective_mode),
    )
    best = select_best_trial(annotated)

    if opt_cfg.persist_results:
        conn = duckdb.connect(opt_cfg.db_path)
        try:
            _ensure_optimizer_tables(conn)
            z = annotated.copy()
            z.insert(0, "run_ts", pd.Timestamp.utcnow())
            conn.register("opt_res", z)
            conn.execute("INSERT INTO agentic_optimizer_results_v1 SELECT * FROM opt_res")
            conn.unregister("opt_res")

            best_row = pd.DataFrame(
                [
                    {
                        "run_ts": pd.Timestamp.utcnow(),
                        "train_start_date": opt_cfg.train_start_date,
                        "train_end_date": opt_cfg.train_end_date,
                        "best_trial_id": int(best["trial_id"]),
                        "cagr": float(best["cagr"]),
                        "sharpe": float(best["sharpe"]),
                        "max_drawdown": float(best["max_drawdown"]),
                        "trade_win_rate": float(best["trade_win_rate"]),
                        "score": float(best["score"]),
                        "objective_mode": str(best["objective_mode"]),
                        "policy_json": str(best["policy_json"]),
                        "agent_weights_json": str(best["agent_weights_json"]),
                    }
                ]
            )
            conn.register("opt_best", best_row)
            conn.execute("INSERT INTO agentic_optimizer_best_v1 SELECT * FROM opt_best")
            conn.unregister("opt_best")
        finally:
            conn.close()

    best_policy = json.loads(str(best["policy_json"]))
    best_weights = json.loads(str(best["agent_weights_json"]))
    return {
        "optimizer_config": asdict(opt_cfg),
        "best_policy": best_policy,
        "best_agent_weights": best_weights,
        "best_row": best,
        "trials": annotated,
    }
