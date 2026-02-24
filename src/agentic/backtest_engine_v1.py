from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd

try:
    from src.agentic.contracts_v1 import OrchestrationPolicy
    from src.agentic.orchestrator_v1 import build_agentic_snapshot
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.agentic.contracts_v1 import OrchestrationPolicy
    from src.agentic.orchestrator_v1 import build_agentic_snapshot


@dataclass
class BacktestConfig:
    db_path: str = "data/ownership.duckdb"
    start_date: str = "2016-01-01"
    end_date: str | None = None
    initial_capital: float = 1_000_000.0
    rebalance_mode: str = "weekly"
    rebalance_weekday: int = 2
    fees_bps: float = 5.0
    slippage_bps: float = 10.0
    universe_limit: int = 0

    def tc_rate(self) -> float:
        return max((float(self.fees_bps) + float(self.slippage_bps)) / 10_000.0, 0.0)


def _load_close_matrix(
    conn: duckdb.DuckDBPyConnection,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    df = conn.execute(
        """
        SELECT
            CAST(p.date AS DATE) AS date,
            UPPER(TRIM(p.canonical_symbol)) AS symbol,
            CAST(p.close AS DOUBLE) AS close
        FROM prices_daily_v1 p
        JOIN symbol_master sm
          ON UPPER(TRIM(sm.canonical_symbol)) = UPPER(TRIM(p.canonical_symbol))
        WHERE COALESCE(sm.in_universe, TRUE)=TRUE
          AND CAST(p.date AS DATE) BETWEEN ? AND ?
          AND p.close IS NOT NULL
        ORDER BY date, symbol
        """,
        [start_date, end_date],
    ).df()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "close"]).copy()
    px = (
        df.pivot(index="date", columns="symbol", values="close")
        .sort_index()
        .ffill()
    )
    # Data-quality guardrail:
    # remove symbols that repeatedly show impossible one-day moves, which
    # indicates corrupted source rows (common paise/rupee or mapping glitches).
    if not px.empty and len(px.index) >= 3:
        ret = px.pct_change()
        extreme = ret.abs() > 0.70
        bad_counts = extreme.sum(axis=0).astype(int)
        bad_syms = bad_counts[bad_counts >= 3].index.tolist()
        if bad_syms:
            px = px.drop(columns=bad_syms, errors="ignore")
    return px


def _rebalance_dates(
    dates: pd.DatetimeIndex,
    mode: str,
    weekday: int,
) -> set[pd.Timestamp]:
    if len(dates) == 0:
        return set()
    if mode.lower() == "daily":
        out = set(pd.DatetimeIndex(dates))
    else:
        out = set(pd.DatetimeIndex(dates)[pd.DatetimeIndex(dates).weekday == int(weekday)])
    out.add(pd.Timestamp(dates[0]))
    return out


def _compute_stats(equity: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float]:
    if equity.empty or len(equity) < 2:
        return {
            "CAGR": 0.0,
            "Total Return": 0.0,
            "Sharpe": 0.0,
            "Max Drawdown": 0.0,
            "Volatility": 0.0,
            "Daily Win Rate": 0.0,
            "Trade Win Rate": 0.0,
            "Trades": 0.0,
            "Avg Turnover": 0.0,
            "Years": 0.0,
        }
    eq = equity.sort_values("date").copy()
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)
    years = max((eq["date"].iloc[-1] - eq["date"].iloc[0]).days / 365.25, 1 / 365.25)
    total_ret = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0)
    cagr = float((eq["equity"].iloc[-1] / eq["equity"].iloc[0]) ** (1.0 / years) - 1.0)
    vol = float(eq["ret"].std(ddof=0) * np.sqrt(252.0))
    sharpe = float((eq["ret"].mean() / eq["ret"].std(ddof=0)) * np.sqrt(252.0)) if eq["ret"].std(ddof=0) > 0 else 0.0
    dd = (eq["equity"] / eq["equity"].cummax() - 1.0).astype(float)
    maxdd = float(dd.min()) if not dd.empty else 0.0
    dwin = float((eq["ret"] > 0).mean()) if len(eq) > 0 else 0.0

    tr = trades.copy()
    trade_count = int(len(tr))
    twr = 0.0
    if not tr.empty and "win_flag" in tr.columns:
        z = tr[tr["win_flag"].notna()].copy()
        twr = float(z["win_flag"].mean()) if not z.empty else 0.0
    avg_turnover = float(trades["turnover_fraction"].mean()) if (not trades.empty and "turnover_fraction" in trades.columns) else 0.0
    return {
        "CAGR": cagr,
        "Total Return": total_ret,
        "Sharpe": sharpe,
        "Max Drawdown": maxdd,
        "Volatility": vol,
        "Daily Win Rate": dwin,
        "Trade Win Rate": twr,
        "Trades": float(trade_count),
        "Avg Turnover": avg_turnover,
        "Years": float(years),
    }


def _ensure_backtest_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agentic_backtest_runs_v1 (
            run_id VARCHAR,
            run_ts TIMESTAMP,
            start_date DATE,
            end_date DATE,
            config_json VARCHAR,
            policy_json VARCHAR,
            agent_weights_json VARCHAR,
            stats_json VARCHAR
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agentic_backtest_equity_v1 (
            run_id VARCHAR,
            date DATE,
            equity DOUBLE,
            cash DOUBLE,
            gross_exposure DOUBLE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agentic_backtest_trades_v1 (
            run_id VARCHAR,
            date DATE,
            symbol VARCHAR,
            side VARCHAR,
            quantity DOUBLE,
            price DOUBLE,
            notional DOUBLE,
            turnover_fraction DOUBLE,
            tc_cost DOUBLE,
            realized_pnl DOUBLE,
            win_flag DOUBLE
        )
        """
    )


def run_agentic_backtest(
    cfg: BacktestConfig,
    policy: OrchestrationPolicy,
    agent_weight_overrides: dict[str, float] | None = None,
    base_snapshot_cache: dict[date, dict[str, object]] | None = None,
    persist: bool = True,
) -> dict[str, object]:
    policy.validate()
    start = pd.to_datetime(cfg.start_date).date()

    conn = duckdb.connect(cfg.db_path)
    try:
        max_date_row = conn.execute("SELECT MAX(CAST(date AS DATE)) FROM prices_daily_v1").fetchone()
        if max_date_row is None or max_date_row[0] is None:
            raise ValueError("prices_daily_v1 has no rows for backtest")
        end = pd.to_datetime(cfg.end_date).date() if cfg.end_date else pd.to_datetime(max_date_row[0]).date()
        if end <= start:
            raise ValueError("end_date must be after start_date")

        close_px = _load_close_matrix(conn, start, end)
        if close_px.empty or len(close_px.index) < 2:
            raise ValueError("Insufficient close price data for backtest range")
        tradable_symbols = set(close_px.columns.tolist())

        rebalance_set = _rebalance_dates(close_px.index, cfg.rebalance_mode, cfg.rebalance_weekday)
        tc_rate = cfg.tc_rate()
        run_id = f"abt_{uuid4().hex[:12]}"

        cash = float(cfg.initial_capital)
        shares: dict[str, float] = {}
        avg_cost: dict[str, float] = {}
        equity_rows: list[dict[str, object]] = []
        trade_rows: list[dict[str, object]] = []

        for dt in close_px.index:
            px_row = close_px.loc[dt]
            nav_pre = cash + float(sum(shares.get(s, 0.0) * float(px_row.get(s, np.nan)) for s in shares if pd.notna(px_row.get(s, np.nan))))
            nav_pre = float(max(nav_pre, 0.0))

            if dt in rebalance_set:
                snap = build_agentic_snapshot(
                    conn=conn,
                    as_of_date=dt.date(),
                    policy=policy,
                    universe_limit=int(cfg.universe_limit),
                    agent_weight_overrides=agent_weight_overrides,
                    base_snapshot_cache=base_snapshot_cache,
                )
                targets = snap["targets"] if isinstance(snap, dict) and "targets" in snap else pd.DataFrame()
                target_w = pd.Series(dtype=float)
                if isinstance(targets, pd.DataFrame) and not targets.empty:
                    target_w = targets.set_index("symbol")["target_weight"].astype(float)
                    target_w = target_w[target_w.index.isin(tradable_symbols)].copy()
                    target_w = target_w[target_w > 0].copy()
                    tw_sum = float(target_w.sum()) if not target_w.empty else 0.0
                    if tw_sum > 1.0:
                        target_w = target_w / tw_sum

                current_symbols = set(shares.keys())
                target_symbols = set(target_w.index.tolist())
                all_symbols = sorted(current_symbols | target_symbols)

                turnover_notional = 0.0
                tc_cost = 0.0
                for sym in all_symbols:
                    px = float(px_row.get(sym, np.nan))
                    if not np.isfinite(px) or px <= 0:
                        continue
                    old_sh = float(shares.get(sym, 0.0))
                    tw = float(target_w.get(sym, 0.0)) if not target_w.empty else 0.0
                    target_val = nav_pre * tw
                    new_sh = float(target_val / px) if target_val > 0 else 0.0
                    d_sh = new_sh - old_sh
                    if abs(d_sh) <= 1e-12:
                        continue

                    notional = abs(d_sh) * px
                    turnover_notional += notional
                    realized = np.nan
                    win_flag = np.nan
                    side = "BUY" if d_sh > 0 else "SELL"
                    if d_sh < 0:
                        sold = abs(d_sh)
                        ac = float(avg_cost.get(sym, px))
                        realized = (px - ac) * sold
                        win_flag = 1.0 if realized > 0 else 0.0
                    trade_rows.append(
                        {
                            "run_id": run_id,
                            "date": dt.date(),
                            "symbol": sym,
                            "side": side,
                            "quantity": float(abs(d_sh)),
                            "price": px,
                            "notional": float(notional),
                            "turnover_fraction": 0.0,  # filled after tc calc
                            "tc_cost": 0.0,  # filled after tc calc
                            "realized_pnl": realized,
                            "win_flag": win_flag,
                        }
                    )

                    cash -= d_sh * px
                    shares[sym] = new_sh
                    if new_sh <= 1e-12:
                        shares.pop(sym, None)
                        avg_cost.pop(sym, None)
                    elif d_sh > 0:
                        prev_sh = old_sh
                        prev_cost = float(avg_cost.get(sym, px))
                        total_sh = prev_sh + d_sh
                        if total_sh > 0:
                            avg_cost[sym] = ((prev_sh * prev_cost) + (d_sh * px)) / total_sh

                if nav_pre > 0 and turnover_notional > 0:
                    tc_cost = turnover_notional * tc_rate
                    cash -= tc_cost
                    tf = turnover_notional / nav_pre
                    for r in range(len(trade_rows) - 1, -1, -1):
                        if trade_rows[r]["run_id"] != run_id or trade_rows[r]["date"] != dt.date():
                            break
                        trade_rows[r]["turnover_fraction"] = float(tf)
                        trade_rows[r]["tc_cost"] = float(tc_cost)

            nav = cash + float(sum(shares.get(s, 0.0) * float(px_row.get(s, np.nan)) for s in shares if pd.notna(px_row.get(s, np.nan))))
            gross_exposure = 0.0
            if nav > 0:
                gross_exposure = float(sum(abs(shares.get(s, 0.0) * float(px_row.get(s, np.nan))) for s in shares if pd.notna(px_row.get(s, np.nan))) / nav)
            equity_rows.append(
                {
                    "run_id": run_id,
                    "date": dt.date(),
                    "equity": float(nav),
                    "cash": float(cash),
                    "gross_exposure": float(gross_exposure),
                }
            )

        equity = pd.DataFrame(equity_rows)
        trades = pd.DataFrame(trade_rows)
        stats = _compute_stats(equity[["date", "equity"]], trades)

        out = {
            "run_id": run_id,
            "start_date": str(start),
            "end_date": str(end),
            "stats": stats,
            "equity": equity,
            "trades": trades,
            "config": asdict(cfg),
            "policy": asdict(policy),
            "agent_weights": agent_weight_overrides or {},
        }

        if persist:
            _ensure_backtest_tables(conn)
            run_row = pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "run_ts": pd.Timestamp.utcnow(),
                        "start_date": start,
                        "end_date": end,
                        "config_json": json.dumps(asdict(cfg), ensure_ascii=True),
                        "policy_json": json.dumps(asdict(policy), ensure_ascii=True),
                        "agent_weights_json": json.dumps(agent_weight_overrides or {}, ensure_ascii=True),
                        "stats_json": json.dumps(stats, ensure_ascii=True),
                    }
                ]
            )
            conn.register("bt_run", run_row)
            conn.execute("INSERT INTO agentic_backtest_runs_v1 SELECT * FROM bt_run")
            conn.unregister("bt_run")

            conn.register("bt_eq", equity[["run_id", "date", "equity", "cash", "gross_exposure"]])
            conn.execute("INSERT INTO agentic_backtest_equity_v1 SELECT * FROM bt_eq")
            conn.unregister("bt_eq")

            if not trades.empty:
                conn.register(
                    "bt_tr",
                    trades[
                        [
                            "run_id",
                            "date",
                            "symbol",
                            "side",
                            "quantity",
                            "price",
                            "notional",
                            "turnover_fraction",
                            "tc_cost",
                            "realized_pnl",
                            "win_flag",
                        ]
                    ],
                )
                conn.execute("INSERT INTO agentic_backtest_trades_v1 SELECT * FROM bt_tr")
                conn.unregister("bt_tr")

        return out
    finally:
        conn.close()
