from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd

try:
    from src.agentic.agents_v1 import (
        default_agents,
        detect_regime,
        evaluate_fundamental_agent,
        evaluate_news_agent,
        evaluate_ownership_agent,
        evaluate_technical_agent,
        load_market_context,
    )
    from src.agentic.contracts_v1 import ActionVote, OrchestrationPolicy, clamp
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.agentic.agents_v1 import (
        default_agents,
        detect_regime,
        evaluate_fundamental_agent,
        evaluate_news_agent,
        evaluate_ownership_agent,
        evaluate_technical_agent,
        load_market_context,
    )
    from src.agentic.contracts_v1 import ActionVote, OrchestrationPolicy, clamp


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=True, default=str)


def _create_audit_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agentic_runs_v1 (
            run_id VARCHAR,
            run_ts TIMESTAMP,
            as_of_date DATE,
            status VARCHAR,
            objective VARCHAR,
            policy_json VARCHAR,
            regime_json VARCHAR,
            summary_json VARCHAR
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agentic_agent_signals_v1 (
            run_id VARCHAR,
            as_of_date DATE,
            symbol VARCHAR,
            agent_name VARCHAR,
            agent_weight DOUBLE,
            score DOUBLE,
            confidence DOUBLE,
            action_vote VARCHAR,
            rationale_json VARCHAR,
            created_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agentic_consensus_v1 (
            run_id VARCHAR,
            as_of_date DATE,
            symbol VARCHAR,
            consensus_score DOUBLE,
            consensus_confidence DOUBLE,
            consensus_action VARCHAR,
            risk_pass BOOLEAN,
            risk_reason VARCHAR,
            median_turnover_63 DOUBLE,
            stale_days INTEGER,
            created_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agentic_watchlist_v1 (
            run_id VARCHAR,
            as_of_date DATE,
            watch_rank INTEGER,
            symbol VARCHAR,
            consensus_score DOUBLE,
            consensus_confidence DOUBLE,
            consensus_action VARCHAR,
            risk_pass BOOLEAN,
            risk_reason VARCHAR,
            created_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agentic_portfolio_targets_v1 (
            run_id VARCHAR,
            as_of_date DATE,
            portfolio_rank INTEGER,
            symbol VARCHAR,
            target_weight DOUBLE,
            consensus_score DOUBLE,
            consensus_confidence DOUBLE,
            final_action VARCHAR,
            rationale_json VARCHAR,
            created_at TIMESTAMP
        )
        """
    )


def _build_liquidity_snapshot(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(columns=["symbol", "median_turnover_63", "stale_days"])
    x = panel.copy()
    x["turnover"] = pd.to_numeric(x["close"], errors="coerce").fillna(0.0) * pd.to_numeric(
        x["volume"], errors="coerce"
    ).fillna(0.0)
    x = x.sort_values(["symbol", "date"])
    med = (
        x.groupby("symbol", as_index=False)
        .tail(63)
        .groupby("symbol", as_index=False)
        .agg(median_turnover_63=("turnover", "median"), last_date=("date", "max"))
    )
    return med


def _discuss_and_aggregate(
    agent_outputs: dict[str, pd.DataFrame],
    policy: OrchestrationPolicy,
    agent_weight_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = []
    for agent_name, df in agent_outputs.items():
        z = df.copy()
        z["agent_name"] = agent_name
        rows.append(z)
    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "consensus_score",
                "consensus_confidence",
                "consensus_action",
            ]
        )
    long = pd.concat(rows, ignore_index=True)

    defs = {d.name: d for d in default_agents(agent_weight_overrides) if d.enabled}
    long["agent_weight"] = long["agent_name"].map(lambda n: float(defs[n].weight) if n in defs else 0.0)
    long["score"] = pd.to_numeric(long["score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    long["confidence"] = pd.to_numeric(long["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    long["effective_weight"] = long["agent_weight"] * long["confidence"]
    long["weighted_score_component"] = long["score"] * long["effective_weight"]
    agg = long.groupby("symbol", as_index=False).agg(
        weighted_score=("weighted_score_component", "sum"),
        weight_sum=("effective_weight", "sum"),
        conf_sum=("effective_weight", "sum"),
    )
    agg["consensus_score"] = np.where(
        agg["weight_sum"] > 0,
        agg["weighted_score"] / agg["weight_sum"],
        0.5,
    )

    total_agent_weight = float(sum(d.weight for d in defs.values())) if defs else 1.0
    agg["consensus_confidence"] = np.where(
        total_agent_weight > 0,
        (agg["conf_sum"] / total_agent_weight).clip(0.0, 1.0),
        0.0,
    )

    def _action(score: float, conf: float) -> str:
        if conf < 0.15:
            return ActionVote.WAIT.value
        if score >= policy.consensus_buy_threshold:
            return ActionVote.BUY.value
        if score <= policy.consensus_sell_threshold:
            return ActionVote.SELL.value
        return ActionVote.HOLD.value

    agg["consensus_action"] = [
        _action(float(s), float(c))
        for s, c in zip(agg["consensus_score"], agg["consensus_confidence"])
    ]
    return agg[["symbol", "consensus_score", "consensus_confidence", "consensus_action"]]


def _apply_risk_gates(
    consensus: pd.DataFrame,
    liq: pd.DataFrame,
    as_of_date: date,
    policy: OrchestrationPolicy,
    price_stale_days_global: int | None,
) -> pd.DataFrame:
    if consensus.empty:
        return consensus.assign(
            risk_pass=pd.Series(dtype=bool),
            risk_reason=pd.Series(dtype=str),
            median_turnover_63=pd.Series(dtype=float),
            stale_days=pd.Series(dtype=int),
        )

    x = consensus.merge(liq, on="symbol", how="left")
    x["median_turnover_63"] = pd.to_numeric(x["median_turnover_63"], errors="coerce").fillna(0.0)
    x["last_date"] = pd.to_datetime(x["last_date"], errors="coerce")
    x["stale_days"] = np.where(
        x["last_date"].notna(),
        (pd.Timestamp(as_of_date) - x["last_date"]).dt.days,
        9999,
    ).astype(int)

    stale_global_block = (
        (price_stale_days_global is not None)
        and (int(price_stale_days_global) > int(policy.max_price_staleness_days))
    )
    stale_symbol_block = x["stale_days"] > int(policy.max_price_staleness_days)
    liquidity_block = x["median_turnover_63"] < float(policy.min_median_turnover_inr)

    x["risk_pass"] = ~(stale_symbol_block | liquidity_block | stale_global_block)
    reasons = []
    for stale_sym, liq_block in zip(stale_symbol_block.tolist(), liquidity_block.tolist()):
        r = []
        if stale_global_block:
            r.append("global_price_staleness_breach")
        if stale_sym:
            r.append("symbol_price_stale")
        if liq_block:
            r.append("liquidity_below_floor")
        reasons.append(",".join(r) if r else "pass")
    x["risk_reason"] = reasons
    return x[
        [
            "symbol",
            "consensus_score",
            "consensus_confidence",
            "consensus_action",
            "risk_pass",
            "risk_reason",
            "median_turnover_63",
            "stale_days",
        ]
    ].copy()


def _build_watchlist_and_targets(
    consensus: pd.DataFrame,
    regime_exposure_scalar: float,
    policy: OrchestrationPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _normalize_with_bounds(raw: pd.Series, min_w: float, max_w: float, gross_target: float) -> pd.Series:
        if raw.empty:
            return raw
        n = len(raw)
        gross_target = float(clamp(gross_target, 0.0, 1.0))
        if gross_target <= 0.0:
            return pd.Series(0.0, index=raw.index, dtype=float)
        if (min_w * n) > gross_target or (max_w * n) < gross_target:
            base = min(max_w, gross_target / n)
            return pd.Series(base, index=raw.index, dtype=float)
        z = pd.to_numeric(raw, errors="coerce").fillna(0.0).clip(lower=0.0)
        if float(z.sum()) <= 0.0:
            z = pd.Series(1.0, index=raw.index, dtype=float)
        w = (z / float(z.sum())) * gross_target
        w = w.clip(lower=min_w, upper=max_w)
        for _ in range(40):
            s = float(w.sum())
            diff = gross_target - s
            if abs(diff) <= 1e-10:
                break
            if diff > 0:
                room = (max_w - w).clip(lower=0.0)
                if float(room.sum()) <= 0.0:
                    break
                w = w + (room / float(room.sum())) * diff
            else:
                room = (w - min_w).clip(lower=0.0)
                if float(room.sum()) <= 0.0:
                    break
                w = w - (room / float(room.sum())) * (-diff)
            w = w.clip(lower=min_w, upper=max_w)
        return w

    x = consensus.sort_values(
        ["risk_pass", "consensus_score", "consensus_confidence", "symbol"],
        ascending=[False, False, False, True],
    ).copy()
    watch = x.head(int(policy.watchlist_size)).copy()
    watch["watch_rank"] = np.arange(1, len(watch) + 1)

    target_n = policy.target_positions(regime_exposure_scalar)
    tradable = watch[watch["risk_pass"] == True].copy()
    tradable = tradable[tradable["consensus_action"] != ActionVote.SELL.value]
    if tradable.empty:
        return watch, pd.DataFrame(
            columns=[
                "portfolio_rank",
                "symbol",
                "target_weight",
                "consensus_score",
                "consensus_confidence",
                "final_action",
                "rationale_json",
            ]
        )

    n = int(clamp(target_n, policy.portfolio_min_positions, policy.portfolio_max_positions))
    n = min(n, len(tradable))
    if n < policy.portfolio_min_positions:
        n = min(len(tradable), policy.portfolio_min_positions)

    pf = tradable.head(n).copy()
    pf["portfolio_rank"] = np.arange(1, len(pf) + 1)
    # Risk-adjusted sizing:
    # expected alpha proxy from consensus score, adjusted by confidence and liquidity.
    score = pd.to_numeric(pf["consensus_score"], errors="coerce").fillna(0.5)
    conf = pd.to_numeric(pf["consensus_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    liq = pd.to_numeric(pf["median_turnover_63"], errors="coerce").fillna(0.0)
    liq_rank = liq.rank(pct=True).fillna(0.5).clip(0.05, 1.0)

    alpha = ((score - float(policy.consensus_sell_threshold)) / max(1.0 - float(policy.consensus_sell_threshold), 1e-6)).clip(0.0, 1.0)
    risk_proxy = (1.0 - conf).clip(0.05, 1.0)
    liq_penalty = (1.15 - liq_rank).clip(0.15, 1.15)
    raw_weight = (alpha * conf) / (risk_proxy * liq_penalty)
    avg_conf = float(conf.mean()) if len(conf) > 0 else 0.5
    gross_target = float(
        clamp(
            (0.70 + 0.30 * avg_conf) * float(clamp(regime_exposure_scalar, 0.75, 1.15)),
            0.55,
            1.0,
        )
    )
    pf["target_weight"] = _normalize_with_bounds(
        raw=raw_weight,
        min_w=float(policy.min_single_weight),
        max_w=float(policy.max_single_weight),
        gross_target=gross_target,
    )

    pf["final_action"] = np.where(
        pf["consensus_action"].isin([ActionVote.BUY.value, ActionVote.HOLD.value]),
        ActionVote.BUY.value,
        ActionVote.WAIT.value,
    )
    pf["rationale_json"] = [
        _to_json(
            {
                "consensus_score": float(s),
                "consensus_confidence": float(c),
                "risk_reason": rr,
                "regime_scalar": float(regime_exposure_scalar),
                "sizing_method": "risk_adjusted_alpha_confidence_liquidity",
                "gross_target": gross_target,
            }
        )
        for s, c, rr in zip(
            pf["consensus_score"], pf["consensus_confidence"], pf["risk_reason"]
        )
    ]
    return watch, pf[
        [
            "portfolio_rank",
            "symbol",
            "target_weight",
            "consensus_score",
            "consensus_confidence",
            "final_action",
            "rationale_json",
        ]
    ].copy()


def _persist_agent_signals(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    as_of_date: date,
    agent_outputs: dict[str, pd.DataFrame],
    agent_weight_overrides: dict[str, float] | None = None,
) -> None:
    defs = {d.name: d for d in default_agents(agent_weight_overrides)}
    rows = []
    for agent_name, df in agent_outputs.items():
        if df.empty:
            continue
        w = float(defs[agent_name].weight) if agent_name in defs else 0.0
        z = df.copy()
        z["run_id"] = run_id
        z["as_of_date"] = as_of_date
        z["agent_name"] = agent_name
        z["agent_weight"] = w
        z["created_at"] = _utc_now()
        z = z[
            [
                "run_id",
                "as_of_date",
                "symbol",
                "agent_name",
                "agent_weight",
                "score",
                "confidence",
                "action",
                "rationale_json",
                "created_at",
            ]
        ].copy()
        z = z.rename(columns={"action": "action_vote"})
        rows.append(z)
    if not rows:
        return
    all_rows = pd.concat(rows, ignore_index=True)
    conn.register("agent_sig", all_rows)
    conn.execute("INSERT INTO agentic_agent_signals_v1 SELECT * FROM agent_sig")
    conn.unregister("agent_sig")


def _persist_consensus(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    as_of_date: date,
    consensus: pd.DataFrame,
) -> None:
    if consensus.empty:
        return
    z = consensus.copy()
    z["run_id"] = run_id
    z["as_of_date"] = as_of_date
    z["created_at"] = _utc_now()
    z = z[
        [
            "run_id",
            "as_of_date",
            "symbol",
            "consensus_score",
            "consensus_confidence",
            "consensus_action",
            "risk_pass",
            "risk_reason",
            "median_turnover_63",
            "stale_days",
            "created_at",
        ]
    ]
    conn.register("consensus_df", z)
    conn.execute("INSERT INTO agentic_consensus_v1 SELECT * FROM consensus_df")
    conn.unregister("consensus_df")


def _persist_watchlist(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    as_of_date: date,
    watchlist: pd.DataFrame,
) -> None:
    if watchlist.empty:
        return
    z = watchlist.copy()
    z["run_id"] = run_id
    z["as_of_date"] = as_of_date
    z["created_at"] = _utc_now()
    z = z[
        [
            "run_id",
            "as_of_date",
            "watch_rank",
            "symbol",
            "consensus_score",
            "consensus_confidence",
            "consensus_action",
            "risk_pass",
            "risk_reason",
            "created_at",
        ]
    ]
    conn.register("watch_df", z)
    conn.execute("INSERT INTO agentic_watchlist_v1 SELECT * FROM watch_df")
    conn.unregister("watch_df")


def _persist_portfolio_targets(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    as_of_date: date,
    targets: pd.DataFrame,
) -> None:
    if targets.empty:
        return
    z = targets.copy()
    z["run_id"] = run_id
    z["as_of_date"] = as_of_date
    z["created_at"] = _utc_now()
    z = z[
        [
            "run_id",
            "as_of_date",
            "portfolio_rank",
            "symbol",
            "target_weight",
            "consensus_score",
            "consensus_confidence",
            "final_action",
            "rationale_json",
            "created_at",
        ]
    ]
    conn.register("targets_df", z)
    conn.execute("INSERT INTO agentic_portfolio_targets_v1 SELECT * FROM targets_df")
    conn.unregister("targets_df")


def _empty_snapshot(as_of_date: date) -> dict[str, Any]:
    return {
        "status": "NO_UNIVERSE",
        "as_of_date": str(as_of_date),
        "agent_outputs": {},
        "consensus": pd.DataFrame(),
        "watchlist": pd.DataFrame(),
        "targets": pd.DataFrame(),
        "regime": {},
        "price_stale_days_global": None,
        "universe_size": 0,
        "risk_pass_count": 0,
    }


def build_agentic_base_snapshot(
    conn: duckdb.DuckDBPyConnection,
    as_of_date: date,
    universe_limit: int = 0,
) -> dict[str, Any]:
    market = load_market_context(conn, as_of_date=as_of_date, universe_limit=universe_limit)
    if not market.symbols:
        return _empty_snapshot(as_of_date)

    regime = detect_regime(market)
    agent_outputs = {
        "technical": evaluate_technical_agent(market),
        "ownership": evaluate_ownership_agent(conn, market.symbols, as_of_date),
        "fundamental": evaluate_fundamental_agent(conn, market.symbols, as_of_date),
        "news": evaluate_news_agent(conn, market.symbols, as_of_date),
    }
    liq = _build_liquidity_snapshot(market.panel)
    price_stale_days = (
        (as_of_date - market.max_price_date).days if market.max_price_date is not None else None
    )
    return {
        "status": "OK",
        "as_of_date": str(as_of_date),
        "agent_outputs": agent_outputs,
        "liq": liq,
        "regime": {
            "label": regime.label,
            "score": regime.score,
            "exposure_scalar": regime.exposure_scalar,
            "breadth": regime.breadth,
            "market_ret_63": regime.market_ret_63,
        },
        "price_stale_days_global": None if price_stale_days is None else int(price_stale_days),
        "universe_size": int(len(market.symbols)),
    }


def materialize_policy_snapshot(
    base_snapshot: dict[str, Any],
    as_of_date: date,
    policy: OrchestrationPolicy,
    agent_weight_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    if base_snapshot.get("status") != "OK":
        return _empty_snapshot(as_of_date)

    consensus = _discuss_and_aggregate(
        agent_outputs=base_snapshot["agent_outputs"],
        policy=policy,
        agent_weight_overrides=agent_weight_overrides,
    )
    consensus = _apply_risk_gates(
        consensus=consensus,
        liq=base_snapshot["liq"],
        as_of_date=as_of_date,
        policy=policy,
        price_stale_days_global=base_snapshot.get("price_stale_days_global"),
    )
    watchlist, targets = _build_watchlist_and_targets(
        consensus=consensus,
        regime_exposure_scalar=float(base_snapshot["regime"]["exposure_scalar"]),
        policy=policy,
    )
    return {
        "status": "OK",
        "as_of_date": str(as_of_date),
        "agent_outputs": base_snapshot["agent_outputs"],
        "consensus": consensus,
        "watchlist": watchlist,
        "targets": targets,
        "regime": base_snapshot["regime"],
        "price_stale_days_global": base_snapshot.get("price_stale_days_global"),
        "universe_size": int(base_snapshot.get("universe_size", 0)),
        "risk_pass_count": int(consensus["risk_pass"].sum()) if not consensus.empty else 0,
    }


def build_agentic_snapshot(
    conn: duckdb.DuckDBPyConnection,
    as_of_date: date,
    policy: OrchestrationPolicy,
    universe_limit: int = 0,
    agent_weight_overrides: dict[str, float] | None = None,
    base_snapshot_cache: dict[date, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    policy.validate()
    if base_snapshot_cache is not None and as_of_date in base_snapshot_cache:
        base = base_snapshot_cache[as_of_date]
    else:
        base = build_agentic_base_snapshot(
            conn=conn,
            as_of_date=as_of_date,
            universe_limit=universe_limit,
        )
        if base_snapshot_cache is not None:
            base_snapshot_cache[as_of_date] = base

    return materialize_policy_snapshot(
        base_snapshot=base,
        as_of_date=as_of_date,
        policy=policy,
        agent_weight_overrides=agent_weight_overrides,
    )


def run_agentic_cycle(
    db_path: str | Path,
    as_of_date: date,
    policy: OrchestrationPolicy,
    universe_limit: int = 0,
    agent_weight_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    policy.validate()
    run_id = f"agentic_{uuid4().hex[:12]}"
    objective = "Max CAGR toward >=25% with minimum-risk constraints and institutional controls."

    conn = duckdb.connect(str(db_path))
    try:
        _create_audit_tables(conn)
        snap = build_agentic_snapshot(
            conn=conn,
            as_of_date=as_of_date,
            policy=policy,
            universe_limit=universe_limit,
            agent_weight_overrides=agent_weight_overrides,
        )
        if snap["status"] != "OK":
            summary = {
                "run_id": run_id,
                "status": "NO_UNIVERSE",
                "as_of_date": str(as_of_date),
                "message": "No active symbols loaded from prices_daily_v1 + symbol_master.",
            }
            run_row = pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "run_ts": _utc_now(),
                        "as_of_date": as_of_date,
                        "status": "NO_UNIVERSE",
                        "objective": objective,
                        "policy_json": _to_json(asdict(policy)),
                        "regime_json": _to_json({}),
                        "summary_json": _to_json(summary),
                    }
                ]
            )
            conn.register("run_row", run_row)
            conn.execute("INSERT INTO agentic_runs_v1 SELECT * FROM run_row")
            conn.unregister("run_row")
            return summary

        _persist_agent_signals(
            conn=conn,
            run_id=run_id,
            as_of_date=as_of_date,
            agent_outputs=snap["agent_outputs"],
            agent_weight_overrides=agent_weight_overrides,
        )
        _persist_consensus(conn, run_id, as_of_date, snap["consensus"])
        _persist_watchlist(conn, run_id, as_of_date, snap["watchlist"])
        _persist_portfolio_targets(conn, run_id, as_of_date, snap["targets"])

        summary = {
            "run_id": run_id,
            "status": "OK",
            "as_of_date": str(as_of_date),
            "objective": objective,
            "policy": asdict(policy),
            "agent_weight_overrides": agent_weight_overrides or {},
            "regime": snap["regime"],
            "universe_size": int(snap["universe_size"]),
            "watchlist_size": int(len(snap["watchlist"])),
            "portfolio_size": int(len(snap["targets"])),
            "price_stale_days_global": snap["price_stale_days_global"],
            "risk_pass_count": int(snap["risk_pass_count"]),
        }
        run_row = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "run_ts": _utc_now(),
                    "as_of_date": as_of_date,
                    "status": "OK",
                    "objective": objective,
                    "policy_json": _to_json(asdict(policy)),
                    "regime_json": _to_json(summary["regime"]),
                    "summary_json": _to_json(summary),
                }
            ]
        )
        conn.register("run_row", run_row)
        conn.execute("INSERT INTO agentic_runs_v1 SELECT * FROM run_row")
        conn.unregister("run_row")
        return summary
    finally:
        conn.close()
