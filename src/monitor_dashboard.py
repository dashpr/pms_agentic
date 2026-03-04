from __future__ import annotations

import json
import os
from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

try:
    from src.agentic.model_lock_ai_agent_stable_stocks_v1 import (
        MODEL_ID,
        MODEL_VERSION,
        locked_agent_weights,
        locked_policy,
        locked_reference_metrics,
    )
    from src.qa.pipeline_health_v1 import compute_pipeline_health
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.agentic.model_lock_ai_agent_stable_stocks_v1 import (
        MODEL_ID,
        MODEL_VERSION,
        locked_agent_weights,
        locked_policy,
        locked_reference_metrics,
    )
    from src.qa.pipeline_health_v1 import compute_pipeline_health


st.set_page_config(page_title="AI Agent Stable Stocks", layout="wide")

DEFAULT_REMOTE_DB_URL = (
    "https://raw.githubusercontent.com/dashpr/pms_agentic/runtime-data/data/ownership_runtime.duckdb"
)
REMOTE_DB_LOCAL_PATH = Path("/tmp/ownership_runtime.duckdb")
REMOTE_DB_META_PATH = Path("/tmp/ownership_runtime_meta.json")


def _safe_json_load(x: Any) -> dict[str, Any]:
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    try:
        return json.loads(str(x))
    except Exception:
        return {}


def _ensure_remote_db_snapshot(url: str, refresh_minutes: int = 60) -> str:
    url = str(url or "").strip()
    if not url:
        raise ValueError("Remote DB URL is empty.")
    refresh_minutes = max(int(refresh_minutes), 1)
    now = pd.Timestamp.utcnow()
    if REMOTE_DB_LOCAL_PATH.exists() and REMOTE_DB_META_PATH.exists():
        meta = _safe_json_load(REMOTE_DB_META_PATH.read_text(encoding="utf-8"))
        last_dl = pd.to_datetime(meta.get("downloaded_at_utc"), errors="coerce")
        if pd.notna(last_dl):
            age_min = (now - last_dl).total_seconds() / 60.0
            if age_min <= float(refresh_minutes):
                # Validate cached file before returning.
                with duckdb.connect(str(REMOTE_DB_LOCAL_PATH), read_only=True) as conn:
                    conn.execute("SELECT 1").fetchone()
                return str(REMOTE_DB_LOCAL_PATH)

    try:
        import requests
    except Exception as e:
        raise RuntimeError(f"requests not available for remote DB download: {e}") from e

    tmp_path = REMOTE_DB_LOCAL_PATH.with_suffix(".download")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                if chunk:
                    f.write(chunk)
    if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
        raise RuntimeError("Downloaded remote DB snapshot is empty.")
    head = tmp_path.read_bytes()[:256].lower()
    if (b"<html" in head) or (b"<!doctype" in head) or (b"internal server error" in head):
        raise RuntimeError("Remote DB URL returned HTML/text error content, not DuckDB bytes.")

    # Integrity check before activating downloaded file.
    with duckdb.connect(str(tmp_path), read_only=True) as conn:
        conn.execute("SELECT 1").fetchone()
    tmp_path.replace(REMOTE_DB_LOCAL_PATH)
    REMOTE_DB_META_PATH.write_text(
        json.dumps(
            {
                "downloaded_at_utc": now.isoformat(),
                "source_url": url,
                "size_bytes": int(REMOTE_DB_LOCAL_PATH.stat().st_size),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(REMOTE_DB_LOCAL_PATH)


def _runtime_meta_url(remote_db_url: str) -> str:
    u = str(remote_db_url or "").strip()
    if not u:
        return ""
    if u.endswith("/ownership_runtime.duckdb"):
        return u.rsplit("/", 1)[0] + "/runtime_meta.json"
    if u.endswith(".duckdb"):
        return u.rsplit("/", 1)[0] + "/runtime_meta.json"
    return ""


@st.cache_data(ttl=300)
def _fetch_runtime_meta(remote_db_url: str) -> dict[str, Any]:
    meta_url = _runtime_meta_url(remote_db_url)
    if not meta_url:
        return {}
    try:
        import requests
    except Exception:
        return {}
    try:
        r = requests.get(meta_url, timeout=20)
        if r.status_code != 200:
            return {}
        return _safe_json_load(r.text)
    except Exception:
        return {}


@st.cache_data(ttl=180)
def _load_data_provenance(db_path: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with duckdb.connect(db_path, read_only=True) as conn:
        row = conn.execute(
            """
            SELECT
                (SELECT MAX(CAST(date AS DATE)) FROM prices_daily_v1) AS prices_max_date,
                (SELECT MAX(CAST(date AS DATE)) FROM delivery_daily_v1) AS delivery_max_date,
                (SELECT MAX(CAST(date AS DATE)) FROM bulk_block_deals) AS bulk_max_date,
                (SELECT MAX(CAST(as_of_date AS DATE)) FROM agentic_runs_v1) AS agentic_max_asof,
                (SELECT MAX(run_ts) FROM agentic_runs_v1) AS agentic_max_run_ts,
                (SELECT COUNT(*) FROM agentic_runs_v1) AS agentic_run_count,
                (SELECT COUNT(*) FROM agentic_consensus_v1) AS consensus_total_rows
            """
        ).fetchone()
        out = {
            "prices_max_date": row[0],
            "delivery_max_date": row[1],
            "bulk_max_date": row[2],
            "agentic_max_asof": row[3],
            "agentic_max_run_ts": row[4],
            "agentic_run_count": int(row[5] or 0),
            "consensus_total_rows": int(row[6] or 0),
        }
        lr = conn.execute(
            """
            SELECT run_id, run_ts, as_of_date, status
            FROM agentic_runs_v1
            ORDER BY run_ts DESC
            LIMIT 1
            """
        ).fetchone()
        if lr:
            out["latest_run_id"] = str(lr[0])
            out["latest_run_ts"] = lr[1]
            out["latest_run_asof"] = lr[2]
            out["latest_run_status"] = str(lr[3])
    return out


def _to_dt_ns(s: pd.Series) -> pd.Series:
    z = pd.to_datetime(s, errors="coerce")
    if hasattr(z, "dtype") and str(z.dtype).startswith("datetime64"):
        return z.astype("datetime64[ns]")
    return z


def _period_stats(equity: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float]:
    if equity.empty or len(equity) < 2:
        return {
            "CAGR": 0.0,
            "Total Return": 0.0,
            "Max Drawdown": 0.0,
            "Volatility": 0.0,
            "Sharpe": 0.0,
            "Daily Win Rate": 0.0,
            "Trade Win Rate": 0.0,
            "Trades": 0.0,
        }
    eq = equity.sort_values("date").copy()
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)
    years = max((_to_dt_ns(eq["date"]).max() - _to_dt_ns(eq["date"]).min()).days / 365.25, 1 / 365.25)
    total_ret = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0)
    cagr = float((eq["equity"].iloc[-1] / eq["equity"].iloc[0]) ** (1.0 / years) - 1.0)
    vol = float(eq["ret"].std(ddof=0) * np.sqrt(252.0))
    sharpe = float((eq["ret"].mean() / eq["ret"].std(ddof=0)) * np.sqrt(252.0)) if eq["ret"].std(ddof=0) > 0 else 0.0
    dd = (eq["equity"] / eq["equity"].cummax() - 1.0).astype(float)
    maxdd = float(dd.min()) if not dd.empty else 0.0
    dwr = float((eq["ret"] > 0).mean()) if len(eq) > 0 else 0.0
    twr = 0.0
    if not trades.empty and "win_flag" in trades.columns:
        z = trades[trades["win_flag"].notna()].copy()
        twr = float(z["win_flag"].mean()) if not z.empty else 0.0
    return {
        "CAGR": cagr,
        "Total Return": total_ret,
        "Max Drawdown": maxdd,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Daily Win Rate": dwr,
        "Trade Win Rate": twr,
        "Trades": float(len(trades)),
    }


def _signal_from_score(score: float, buy_cutoff: float, sell_cutoff: float, in_portfolio: bool) -> str:
    mid = (buy_cutoff + sell_cutoff) / 2.0
    if score <= sell_cutoff:
        return "SELL"
    if in_portfolio and score >= buy_cutoff:
        return "ACCUMULATE"
    if score >= buy_cutoff:
        return "BUY"
    if score >= mid:
        return "HOLD"
    return "WAIT"


def _signal_color(signal: str) -> str:
    s = str(signal).strip().upper()
    if s == "BUY":
        return "background-color: #8FD694; color: #1d3b1f;"
    if s == "ACCUMULATE":
        return "background-color: #C6F6C6; color: #1d3b1f;"
    if s == "HOLD":
        return "background-color: #FFD166; color: #4a3a00;"
    if s == "SELL":
        return "background-color: #EF476F; color: #ffffff;"
    return "background-color: #E6E6E6; color: #333333;"


def _pct_change_color(v: object) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return "color: #666666;"
    if float(x) > 0:
        return "color: #1b8a3a; font-weight: 600;"
    if float(x) < 0:
        return "color: #c12f2f; font-weight: 600;"
    return "color: #666666;"


def _status_color(v: object) -> str:
    s = str(v).upper()
    if s in {"OK", "WORKING"}:
        return "background-color: #C6F6C6; color: #1d3b1f; font-weight: 600;"
    if s in {"SKIPPED", "STALE"}:
        return "background-color: #FFD166; color: #4a3a00; font-weight: 600;"
    if s in {"OPTIONAL_OFF", "N/A", "NA"}:
        return "background-color: #E6E6E6; color: #333333; font-weight: 600;"
    if s in {"BROKEN", "ERROR"}:
        return "background-color: #EF476F; color: #ffffff; font-weight: 600;"
    return "background-color: #E6E6E6; color: #333333;"


def _style_signal_and_change(
    df: pd.DataFrame,
    signal_col: str = "signal",
    change_col: str = "change_1d_pct",
) -> pd.io.formats.style.Styler:
    sty = df.style
    if signal_col in df.columns:
        sty = sty.apply(
            lambda s: [_signal_color(v) if s.name == signal_col else "" for v in s],
            subset=[signal_col],
            axis=0,
        )
    if change_col in df.columns:
        sty = sty.apply(
            lambda s: [_pct_change_color(v) if s.name == change_col else "" for v in s],
            subset=[change_col],
            axis=0,
        )
        sty = sty.format({change_col: lambda v: f"{float(v):.2f}%" if pd.notna(v) else "-"})
    if "fusion_score_display" in df.columns and "fusion_delta_pct" in df.columns:
        sty = sty.apply(
            lambda row: [
                _pct_change_color(row["fusion_delta_pct"]) if c == "fusion_score_display" else ""
                for c in row.index
            ],
            axis=1,
        )
        sty = sty.format({"fusion_score_display": lambda v: str(v)})
    return sty


def _df_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        for name, df in sheets.items():
            safe = str(name)[:31]
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_excel(
                xw,
                sheet_name=safe,
                index=False,
            )
    return bio.getvalue()


def _signal_strength_label(
    signal: str,
    score: float,
    confidence: float,
    buy_cutoff: float,
    sell_cutoff: float,
) -> str:
    sig = str(signal).strip().upper()
    mid = (buy_cutoff + sell_cutoff) / 2.0
    if sig in {"BUY", "ACCUMULATE"}:
        edge = score - buy_cutoff
        if edge >= 0.12 and confidence >= 0.60:
            return f"STRONG {sig}"
        if edge >= 0.05 and confidence >= 0.50:
            return sig
        return f"WEAK {sig}"
    if sig == "SELL":
        edge = sell_cutoff - score
        if edge >= 0.10 and confidence >= 0.60:
            return "STRONG SELL"
        if edge >= 0.04 and confidence >= 0.50:
            return "SELL"
        return "WEAK SELL"
    if sig == "HOLD":
        if abs(score - mid) <= 0.06 and confidence >= 0.55:
            return "STRONG HOLD"
        return "HOLD"
    return sig


def _filter_dates(df: pd.DataFrame, date_col: str, years: list[int], months: list[int]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    z = df.copy()
    z[date_col] = _to_dt_ns(z[date_col])
    z = z[z[date_col].notna()].copy()
    if years:
        z = z[z[date_col].dt.year.isin(years)].copy()
    if months:
        z = z[z[date_col].dt.month.isin(months)].copy()
    return z


def _build_closed_trips(trades: pd.DataFrame) -> pd.DataFrame:
    cols = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "pnl_pct", "holding_days", "win_flag"]
    if trades.empty:
        return pd.DataFrame(columns=cols)
    z = trades.copy()
    z["date"] = _to_dt_ns(z["date"])
    z["symbol"] = z["symbol"].astype(str).str.strip().str.upper()
    z["side"] = z["side"].astype(str).str.strip().str.upper()
    z["quantity"] = pd.to_numeric(z["quantity"], errors="coerce").fillna(0.0)
    z["price"] = pd.to_numeric(z["price"], errors="coerce").fillna(0.0)
    z = z.sort_values(["symbol", "date"]).copy()
    out: list[dict[str, Any]] = []
    for sym, g in z.groupby("symbol"):
        qty = 0.0
        avg_cost = 0.0
        entry_dt: pd.Timestamp | None = None
        for _, r in g.iterrows():
            q = float(r["quantity"])
            p = float(r["price"])
            d = pd.Timestamp(r["date"])
            if q <= 0 or p <= 0 or pd.isna(d):
                continue
            side = str(r["side"])
            if side == "BUY":
                if qty <= 1e-12:
                    entry_dt = d
                total_cost = avg_cost * qty + p * q
                qty += q
                avg_cost = total_cost / qty if qty > 0 else 0.0
            elif side == "SELL" and qty > 1e-12:
                sell_q = min(q, qty)
                qty -= sell_q
                if qty <= 1e-9 and entry_dt is not None and avg_cost > 0:
                    pnl_pct = (p - avg_cost) / avg_cost
                    out.append(
                        {
                            "symbol": sym,
                            "entry_date": entry_dt,
                            "exit_date": d,
                            "entry_price": avg_cost,
                            "exit_price": p,
                            "pnl_pct": float(pnl_pct),
                            "holding_days": int((d - entry_dt).days),
                            "win_flag": 1.0 if pnl_pct > 0 else 0.0,
                        }
                    )
                    qty = 0.0
                    avg_cost = 0.0
                    entry_dt = None
    if not out:
        return pd.DataFrame(columns=cols)
    o = pd.DataFrame(out)
    o["entry_date"] = _to_dt_ns(o["entry_date"])
    o["exit_date"] = _to_dt_ns(o["exit_date"])
    return o[cols].sort_values(["exit_date", "symbol"], ascending=[False, True]).reset_index(drop=True)


@st.cache_data(ttl=60)
def _load_db_bundle(db_path: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    with duckdb.connect(db_path, read_only=True) as conn:
        out["runs"] = conn.execute(
            """
            SELECT run_id, run_ts, as_of_date, status, objective, policy_json, regime_json, summary_json
            FROM agentic_runs_v1
            ORDER BY run_ts DESC
            """
        ).df()
        out["watchlist"] = conn.execute(
            """
            SELECT run_id, as_of_date, watch_rank, symbol, consensus_score, consensus_confidence, consensus_action, risk_pass, risk_reason, created_at
            FROM agentic_watchlist_v1
            ORDER BY as_of_date DESC, watch_rank ASC
            """
        ).df()
        out["portfolio"] = conn.execute(
            """
            SELECT run_id, as_of_date, portfolio_rank, symbol, target_weight, consensus_score, consensus_confidence, final_action, rationale_json, created_at
            FROM agentic_portfolio_targets_v1
            ORDER BY as_of_date DESC, portfolio_rank ASC
            """
        ).df()
        out["consensus"] = conn.execute(
            """
            SELECT run_id, as_of_date, symbol, consensus_score, consensus_confidence, consensus_action, risk_pass, risk_reason, median_turnover_63, stale_days, created_at
            FROM agentic_consensus_v1
            ORDER BY as_of_date DESC, consensus_score DESC
            """
        ).df()
        out["agent_signals"] = conn.execute(
            """
            SELECT run_id, as_of_date, symbol, agent_name, agent_weight, score, confidence, action_vote, rationale_json, created_at
            FROM agentic_agent_signals_v1
            ORDER BY as_of_date DESC, symbol, agent_name
            """
        ).df()
        out["bt_runs"] = conn.execute(
            """
            SELECT run_id, run_ts, start_date, end_date, stats_json, config_json
            FROM agentic_backtest_runs_v1
            ORDER BY run_ts DESC
            """
        ).df()
        out["bt_equity"] = conn.execute(
            """
            SELECT run_id, date, equity, cash, gross_exposure
            FROM agentic_backtest_equity_v1
            ORDER BY date
            """
        ).df()
        out["bt_trades"] = conn.execute(
            """
            SELECT run_id, date, symbol, side, quantity, price, notional, turnover_fraction, tc_cost, realized_pnl, win_flag
            FROM agentic_backtest_trades_v1
            ORDER BY date, symbol
            """
        ).df()
        sm_cols = {
            r[0].lower()
            for r in conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name='symbol_master'
                """
            ).fetchall()
        }
        if "sector" in sm_cols:
            out["symbol_meta"] = conn.execute(
                """
                SELECT
                    UPPER(TRIM(canonical_symbol)) AS symbol,
                    COALESCE(NULLIF(TRIM(sector), ''), 'UNKNOWN') AS sector
                FROM symbol_master
                WHERE COALESCE(in_universe, TRUE)=TRUE
                ORDER BY symbol
                """
            ).df()
        else:
            out["symbol_meta"] = conn.execute(
                """
                SELECT
                    UPPER(TRIM(canonical_symbol)) AS symbol,
                    'UNKNOWN' AS sector
                FROM symbol_master
                WHERE COALESCE(in_universe, TRUE)=TRUE
                ORDER BY symbol
                """
            ).df()
        out["symbols"] = out["symbol_meta"][["symbol"]].copy()
        out["quotes_db"] = conn.execute(
            """
            WITH px AS (
                SELECT
                    UPPER(TRIM(canonical_symbol)) AS symbol,
                    CAST(date AS DATE) AS date,
                    CAST(close AS DOUBLE) AS close,
                    ROW_NUMBER() OVER (
                        PARTITION BY UPPER(TRIM(canonical_symbol))
                        ORDER BY CAST(date AS DATE) DESC
                    ) AS rn
                FROM prices_daily_v1
            )
            SELECT
                a.symbol,
                a.date AS px_date,
                a.close AS last_price,
                b.close AS prev_close
            FROM px a
            LEFT JOIN px b
              ON a.symbol = b.symbol
             AND b.rn = 2
            WHERE a.rn = 1
            """
        ).df()
    return out


@st.cache_data(ttl=180)
def _fetch_live_quotes(symbols: tuple[str, ...]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "live_price", "live_prev_close", "live_change_1d_pct", "live_ts"])
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame(columns=["symbol", "live_price", "live_prev_close", "live_change_1d_pct", "live_ts"])

    rows: list[dict[str, Any]] = []
    for sym in symbols:
        ticker = f"{sym}.NS"
        try:
            t = yf.Ticker(ticker)
            fi = getattr(t, "fast_info", {}) or {}
            lp = fi.get("lastPrice", None)
            pc = fi.get("previousClose", None)
            if lp is None:
                hist = t.history(period="2d", interval="1d")
                if not hist.empty:
                    lp = float(hist["Close"].iloc[-1])
                    pc = float(hist["Close"].iloc[-2]) if len(hist) > 1 else lp
            if lp is None:
                continue
            lp = float(lp)
            pc = float(pc) if pc is not None else np.nan
            chg = ((lp / pc) - 1.0) * 100.0 if np.isfinite(pc) and pc != 0 else np.nan
            rows.append(
                {
                    "symbol": sym,
                    "live_price": lp,
                    "live_prev_close": pc,
                    "live_change_1d_pct": chg,
                    "live_ts": pd.Timestamp.utcnow(),
                }
            )
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["symbol", "live_price", "live_prev_close", "live_change_1d_pct", "live_ts"])
    return pd.DataFrame(rows)


def _build_signal_reason(
    symbol: str,
    signal: str,
    score: float,
    confidence: float,
    risk_reason: str,
    buy_cutoff: float,
    sell_cutoff: float,
    signal_rows: pd.DataFrame,
    day_change_pct: float | None = None,
) -> str:
    s = signal_rows.copy()
    s["score"] = pd.to_numeric(s["score"], errors="coerce").fillna(0.5)
    s["confidence"] = pd.to_numeric(s["confidence"], errors="coerce").fillna(0.0)
    s["agent_weight"] = pd.to_numeric(s["agent_weight"], errors="coerce").fillna(0.0)
    s = s[s["symbol"] == symbol].copy()
    contributors = "no agent details"
    score_dispersion = np.nan
    if not s.empty:
        s["contrib"] = s["score"] * s["confidence"] * s["agent_weight"]
        score_dispersion = float(s["score"].std(ddof=0))
        top = s.sort_values("contrib", ascending=False).head(2)
        contributors = ", ".join(
            [f"{r.agent_name}:{int(round(float(r.score) * 100.0, 0))}" for r in top.itertuples()]
        )
    strength = _signal_strength_label(signal, score, confidence, buy_cutoff, sell_cutoff)
    sig = str(signal).strip().upper()
    if sig in {"BUY", "ACCUMULATE"}:
        why = (
            f"{strength}: score {score:.2f} is above buy cutoff {buy_cutoff:.2f} "
            f"with confidence {confidence:.2f}. Dominant contributors: {contributors}."
        )
    elif sig == "SELL":
        why = (
            f"{strength}: score {score:.2f} is below sell cutoff {sell_cutoff:.2f} "
            f"with confidence {confidence:.2f}. Dominant contributors: {contributors}."
        )
    elif sig == "HOLD":
        why = (
            f"{strength}: score {score:.2f} sits between sell {sell_cutoff:.2f} and buy {buy_cutoff:.2f}; "
            f"confirmation is incomplete. Dominant contributors: {contributors}."
        )
    else:
        why = (
            f"{strength}: score {score:.2f} does not clear action thresholds. "
            f"Dominant contributors: {contributors}."
        )

    watchouts: list[str] = []
    if str(risk_reason).lower() != "pass":
        watchouts.append(f"risk gate flagged ({risk_reason})")
    if confidence < 0.45:
        watchouts.append("low confidence; conviction is weak")
    if np.isfinite(score_dispersion) and float(score_dispersion) > 0.16:
        watchouts.append("agent disagreement is elevated")
    if day_change_pct is not None and np.isfinite(day_change_pct) and abs(float(day_change_pct)) >= 4.0:
        watchouts.append(f"1-day move is stretched ({float(day_change_pct):.2f}%)")
    if not watchouts:
        watchouts.append("no immediate red flags; monitor regime change and weekly rebalance")
    return f"{why} Watch-outs: {'; '.join(watchouts)}."


def _build_symbol_advisor(
    symbol: str,
    consensus: pd.DataFrame,
    signals: pd.DataFrame,
    in_portfolio: set[str],
    buy_cutoff: float,
    sell_cutoff: float,
) -> dict[str, Any]:
    sym = str(symbol).strip().upper()
    if consensus.empty:
        return {
            "symbol": sym,
            "signal": "WAIT",
            "fusion_score": 0,
            "score_percentile": 0,
            "expected_upside_pct": 0,
            "expected_horizon_days": 0,
            "why": "No consensus snapshot available.",
        }
    c = consensus.copy()
    c["symbol"] = c["symbol"].astype(str).str.strip().str.upper()
    c["consensus_score"] = pd.to_numeric(c["consensus_score"], errors="coerce").fillna(0.5)
    c["consensus_confidence"] = pd.to_numeric(c["consensus_confidence"], errors="coerce").fillna(0.0)
    c["pct_rank"] = c["consensus_score"].rank(pct=True).fillna(0.0)
    row = c[c["symbol"] == sym]
    if row.empty:
        return {
            "symbol": sym,
            "signal": "WAIT",
            "fusion_score": 0,
            "score_percentile": 0,
            "expected_upside_pct": 0,
            "expected_horizon_days": 0,
            "why": "Symbol not in latest consensus universe.",
        }
    r = row.iloc[0]
    score = float(r["consensus_score"])
    conf = float(r["consensus_confidence"])
    in_pf = sym in in_portfolio
    signal = _signal_from_score(score, buy_cutoff, sell_cutoff, in_pf)
    raw_up = max((score - buy_cutoff) / max(1.0 - buy_cutoff, 1e-6), 0.0)
    upside = int(round(raw_up * 30.0, 0))
    horizon = int(round(30 + (1.0 - conf) * 60.0, 0))
    why = _build_signal_reason(
        symbol=sym,
        signal=signal,
        score=score,
        confidence=conf,
        risk_reason=str(r.get("risk_reason", "pass")),
        buy_cutoff=buy_cutoff,
        sell_cutoff=sell_cutoff,
        signal_rows=signals,
        day_change_pct=None,
    )
    return {
        "symbol": sym,
        "signal": signal,
        "fusion_score": int(round(score * 100.0, 0)),
        "score_percentile": int(round(float(r["pct_rank"]) * 100.0, 0)),
        "expected_upside_pct": upside,
        "expected_horizon_days": horizon,
        "why": why,
    }


@st.cache_data(ttl=300)
def _load_pipeline_health(db_path: str) -> pd.DataFrame:
    with duckdb.connect(db_path, read_only=True) as conn:
        # Freshness must be evaluated against "today", not max table date,
        # otherwise stale pipelines can appear falsely healthy.
        as_of = pd.Timestamp.now(tz="UTC").date()
        h = compute_pipeline_health(
            conn=conn,
            as_of_date=as_of,
            require_news=False,
            require_fundamentals=False,
        )
    h = h.copy()
    h["as_of_ref_date"] = pd.to_datetime(as_of).date()
    h["last_date"] = pd.to_datetime(h["last_date"], errors="coerce")
    today = pd.Timestamp.utcnow().date()
    h["calendar_lag_days"] = h["last_date"].apply(
        lambda x: int((today - pd.Timestamp(x).date()).days) if pd.notna(x) else None
    )
    return h


def _next_weekday(d: pd.Timestamp, weekday: int) -> pd.Timestamp:
    days_ahead = (int(weekday) - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (d + pd.Timedelta(days=days_ahead)).normalize()


def main() -> None:
    st.title("AI Agent Stable Stocks")
    st.caption(f"Model: {MODEL_ID} | Version: {MODEL_VERSION}")

    policy = locked_policy()
    buy_cutoff = float(policy.consensus_buy_threshold)
    sell_cutoff = float(policy.consensus_sell_threshold)

    with st.sidebar:
        st.header("Controls")
        default_db_path = os.getenv("PMS_DB_PATH", "data/ownership.duckdb")
        db_path = st.text_input("DuckDB Path", value=default_db_path)
        remote_db_url = os.getenv("OWNERSHIP_DB_URL", DEFAULT_REMOTE_DB_URL).strip()
        remote_refresh_minutes = int(os.getenv("OWNERSHIP_DB_REFRESH_MINUTES", "60"))
        top_n_watchlist = st.slider("Top watchlist rows", min_value=5, max_value=25, value=25, step=1)
        bt_scope = st.radio("Backtest scope", options=["All runs", "Latest run"], index=0)
        use_live_quotes = st.checkbox("Use real-time Yahoo quotes (best effort)", value=True)
        tracking_capital_inr = st.number_input(
            "Tracking capital (INR) for indicative quantity",
            min_value=100000.0,
            value=1_000_000.0,
            step=50000.0,
        )
        st.markdown("Cutoffs")
        st.write(f"Buy >= {int(round(buy_cutoff * 100.0, 0))}")
        st.write(f"Exit <= {int(round(sell_cutoff * 100.0, 0))}")
        if st.button("Reload Data"):
            _load_db_bundle.clear()
            _fetch_live_quotes.clear()
            _load_pipeline_health.clear()

    if not Path(db_path).exists() and remote_db_url:
        try:
            with st.spinner("Bootstrapping cloud DB snapshot..."):
                db_path = _ensure_remote_db_snapshot(remote_db_url, refresh_minutes=remote_refresh_minutes)
            st.caption(f"Using runtime DB snapshot: {db_path}")
        except Exception as e:
            st.error(f"Remote DB bootstrap failed: {e}")
            return

    if not Path(db_path).exists():
        st.error(f"DB not found: {db_path}")
        return

    bundle = _load_db_bundle(db_path)
    provenance = _load_data_provenance(db_path)
    remote_runtime_meta = _fetch_runtime_meta(remote_db_url) if remote_db_url else {}
    local_runtime_meta = _safe_json_load(REMOTE_DB_META_PATH.read_text(encoding="utf-8")) if REMOTE_DB_META_PATH.exists() else {}
    symbol_meta = bundle.get("symbol_meta", pd.DataFrame(columns=["symbol", "sector"])).copy()
    if not symbol_meta.empty:
        symbol_meta["symbol"] = symbol_meta["symbol"].astype(str).str.upper()
        symbol_meta["sector"] = symbol_meta["sector"].astype(str).replace("", "UNKNOWN").fillna("UNKNOWN")
    runs = bundle["runs"].copy()
    if runs.empty:
        st.warning("No agentic runs found. Run the model first.")
        return

    runs["run_ts"] = _to_dt_ns(runs["run_ts"])
    runs["as_of_date"] = _to_dt_ns(runs["as_of_date"])
    latest_run = runs.sort_values("run_ts", ascending=False).iloc[0]
    latest_run_id = str(latest_run["run_id"])
    latest_as_of = pd.to_datetime(latest_run["as_of_date"]).date()
    runs_sorted = runs.sort_values("run_ts", ascending=False).reset_index(drop=True)
    latest_run_ts = pd.to_datetime(latest_run["run_ts"])
    prev_distinct_day_runs = runs_sorted[
        _to_dt_ns(runs_sorted["run_ts"]).dt.date < pd.Timestamp(latest_run_ts).date()
    ]
    if not prev_distinct_day_runs.empty:
        prev_run_id = str(prev_distinct_day_runs.iloc[0]["run_id"])
    elif len(runs_sorted) > 1:
        prev_run_id = str(runs_sorted.iloc[1]["run_id"])
    else:
        prev_run_id = ""
    regime = _safe_json_load(latest_run.get("regime_json"))
    summary = _safe_json_load(latest_run.get("summary_json"))

    watch = bundle["watchlist"].copy()
    watch["as_of_date"] = _to_dt_ns(watch["as_of_date"])
    watch_latest = watch[watch["run_id"] == latest_run_id].copy().sort_values("watch_rank")

    portfolio = bundle["portfolio"].copy()
    portfolio["as_of_date"] = _to_dt_ns(portfolio["as_of_date"])
    portfolio_latest = portfolio[portfolio["run_id"] == latest_run_id].copy().sort_values("portfolio_rank")
    in_pf = set(portfolio_latest["symbol"].astype(str).str.upper().tolist())

    consensus = bundle["consensus"].copy()
    consensus["as_of_date"] = _to_dt_ns(consensus["as_of_date"])
    consensus_latest = consensus[consensus["run_id"] == latest_run_id].copy()
    consensus_latest["symbol"] = consensus_latest["symbol"].astype(str).str.upper()
    prev_consensus = consensus[consensus["run_id"] == prev_run_id].copy() if prev_run_id else pd.DataFrame()
    prev_consensus["symbol"] = prev_consensus["symbol"].astype(str).str.upper() if not prev_consensus.empty else pd.Series(dtype=str)
    prev_score_map = (
        prev_consensus.dropna(subset=["symbol"])
        .drop_duplicates(subset=["symbol"], keep="first")
        .set_index("symbol")["consensus_score"]
        .to_dict()
        if not prev_consensus.empty
        else {}
    )
    risk_reason_map = (
        consensus_latest.dropna(subset=["symbol"])
        .drop_duplicates(subset=["symbol"], keep="first")
        .set_index("symbol")["risk_reason"]
        .to_dict()
    )

    agent_signals = bundle["agent_signals"].copy()
    agent_signals["as_of_date"] = _to_dt_ns(agent_signals["as_of_date"])
    signals_latest = agent_signals[agent_signals["run_id"] == latest_run_id].copy()
    signals_latest["symbol"] = signals_latest["symbol"].astype(str).str.upper()

    bt_runs = bundle["bt_runs"].copy()
    bt_runs["run_ts"] = _to_dt_ns(bt_runs["run_ts"])
    latest_bt_cagr = 0.0
    if not bt_runs.empty:
        stz = _safe_json_load(bt_runs.sort_values("run_ts", ascending=False).iloc[0]["stats_json"])
        latest_bt_cagr = float(stz.get("CAGR", 0.0))

    quotes_db = bundle["quotes_db"].copy()
    quotes_db["symbol"] = quotes_db["symbol"].astype(str).str.upper()
    quotes_db["px_date"] = _to_dt_ns(quotes_db["px_date"])
    quotes_db["last_price"] = pd.to_numeric(quotes_db["last_price"], errors="coerce")
    quotes_db["prev_close"] = pd.to_numeric(quotes_db["prev_close"], errors="coerce")
    quotes_db["db_change_1d_pct"] = ((quotes_db["last_price"] / quotes_db["prev_close"]) - 1.0) * 100.0

    use_symbols = sorted(set(watch_latest["symbol"].astype(str).str.upper().tolist()) | in_pf)
    live_quotes = _fetch_live_quotes(tuple(use_symbols)) if use_live_quotes else pd.DataFrame()
    if not live_quotes.empty:
        live_quotes["symbol"] = live_quotes["symbol"].astype(str).str.upper()

    max_px_date = quotes_db["px_date"].max() if not quotes_db.empty else pd.NaT
    latency_days = None
    if pd.notna(max_px_date):
        now_ts = pd.Timestamp.utcnow()
        now_naive = now_ts.tz_localize(None) if getattr(now_ts, "tzinfo", None) is not None else now_ts
        px_ts = pd.Timestamp(max_px_date)
        px_naive = px_ts.tz_localize(None) if getattr(px_ts, "tzinfo", None) is not None else px_ts
        latency_days = int((now_naive.normalize() - px_naive.normalize()).days)

    st.subheader("Live Snapshot")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Snapshot Time", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
    c2.metric("Model As Of", str(latest_as_of))
    c3.metric("Regime", str(regime.get("label", "unknown")).upper())
    c4.metric("Portfolio", f"{int(summary.get('portfolio_size', len(portfolio_latest)))}")
    c5.metric("Portfolio CAGR", f"{latest_bt_cagr:.2%}")
    c6.metric("Data Latency (days)", "-" if latency_days is None else str(int(latency_days)))

    st.subheader("Data Provenance")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("DB Source", Path(db_path).name)
    p2.metric(
        "Runtime Published (UTC)",
        str(remote_runtime_meta.get("published_at_utc", "-")),
    )
    p3.metric(
        "Snapshot Downloaded (UTC)",
        str(local_runtime_meta.get("downloaded_at_utc", "-")),
    )
    p4.metric(
        "Latest Decision Run",
        str(provenance.get("latest_run_id", "-"))[:18],
    )
    prov_rows = [
        {"key": "latest_run_status", "value": provenance.get("latest_run_status")},
        {"key": "latest_run_asof", "value": str(provenance.get("latest_run_asof"))},
        {"key": "latest_run_ts", "value": str(provenance.get("latest_run_ts"))},
        {"key": "prices_daily_v1.max_date", "value": str(provenance.get("prices_max_date"))},
        {"key": "delivery_daily_v1.max_date", "value": str(provenance.get("delivery_max_date"))},
        {"key": "bulk_block_deals.max_date", "value": str(provenance.get("bulk_max_date"))},
        {"key": "agentic_runs_v1.count", "value": int(provenance.get("agentic_run_count", 0))},
        {"key": "agentic_consensus_v1.rows", "value": int(provenance.get("consensus_total_rows", 0))},
        {"key": "runtime_meta.source_sha", "value": str(remote_runtime_meta.get("source_sha", "-"))},
        {"key": "runtime_meta.source_run_id", "value": str(remote_runtime_meta.get("source_run_id", "-"))},
    ]
    latest_run_asof_ts = pd.to_datetime(provenance.get("latest_run_asof"), errors="coerce")
    if pd.notna(latest_run_asof_ts):
        run_asof_lag_days = int((pd.Timestamp.now(tz="UTC").date() - latest_run_asof_ts.date()).days)
        prov_rows.append({"key": "latest_run_asof_lag_days", "value": run_asof_lag_days})
        if run_asof_lag_days > 1:
            st.warning(
                f"Latest decision as-of is stale by {run_asof_lag_days} day(s). "
                "Automation is running, but market data refresh is lagging."
            )
    st.dataframe(pd.DataFrame(prov_rows), hide_index=True, use_container_width=True)

    st.subheader("Weekly Rebalance Tracker")
    rebalance_weekday = 2  # Wednesday
    now_utc = pd.Timestamp.utcnow()
    next_rebalance_dt = _next_weekday(now_utc, rebalance_weekday)
    countdown = next_rebalance_dt - now_utc
    cd_days = max(int(countdown.total_seconds() // 86400), 0)
    cd_hours = max(int((countdown.total_seconds() % 86400) // 3600), 0)
    run_ts_series = _to_dt_ns(runs_sorted["run_ts"])
    wed_runs = runs_sorted[run_ts_series.dt.weekday == rebalance_weekday]
    if not wed_runs.empty:
        last_rebalance_run_id = str(wed_runs.iloc[0]["run_id"])
        last_rebalance_ts = pd.to_datetime(wed_runs.iloc[0]["run_ts"])
    else:
        # Fallback: no Wednesday execution found yet, use latest execution snapshot.
        last_rebalance_run_id = latest_run_id
        last_rebalance_ts = pd.to_datetime(latest_run["run_ts"])
    last_rebalance_date = pd.Timestamp(last_rebalance_ts).date()

    prev_reb_candidates = runs_sorted[
        _to_dt_ns(runs_sorted["run_ts"]) < pd.Timestamp(last_rebalance_ts)
    ].copy()
    prev_reb_distinct_day = prev_reb_candidates[
        _to_dt_ns(prev_reb_candidates["run_ts"]).dt.date < pd.Timestamp(last_rebalance_ts).date()
    ]
    if not prev_reb_distinct_day.empty:
        prev_rebalance_run_id = str(prev_reb_distinct_day.iloc[0]["run_id"])
    elif not prev_reb_candidates.empty:
        prev_rebalance_run_id = str(prev_reb_candidates.iloc[0]["run_id"])
    else:
        prev_rebalance_run_id = (prev_run_id if prev_run_id else "")
    today_utc = now_utc.date()
    this_week_wed = (pd.Timestamp(today_utc) - pd.Timedelta(days=pd.Timestamp(today_utc).weekday()) + pd.Timedelta(days=2)).date()
    rebalance_done_today = bool(last_rebalance_date == today_utc and pd.Timestamp(last_rebalance_ts).weekday() == rebalance_weekday)
    rebalance_done_this_week = bool(last_rebalance_date >= this_week_wed)
    rebalance_status = (
        "DONE_TODAY"
        if rebalance_done_today
        else ("DONE_THIS_WEEK" if rebalance_done_this_week else ("PENDING_TODAY" if today_utc.weekday() == rebalance_weekday else "PENDING"))
    )
    rb1, rb2, rb3 = st.columns(3)
    rb1.metric("Last Rebalance Date", str(last_rebalance_date))
    rb2.metric("Next Rebalance Date", str(next_rebalance_dt.date()))
    rb3.metric("Countdown", f"{cd_days}d {cd_hours}h")
    st.caption(f"Rebalance status: {rebalance_status} | run_id: {last_rebalance_run_id[:16]}")

    rebalance_portfolio = portfolio[portfolio["run_id"] == last_rebalance_run_id].copy() if last_rebalance_run_id else pd.DataFrame()
    rebalance_portfolio["symbol"] = (
        rebalance_portfolio["symbol"].astype(str).str.upper() if not rebalance_portfolio.empty else pd.Series(dtype=str)
    )
    prev_portfolio = portfolio[portfolio["run_id"] == prev_rebalance_run_id].copy() if prev_rebalance_run_id else pd.DataFrame()
    prev_portfolio["symbol"] = (
        prev_portfolio["symbol"].astype(str).str.upper() if not prev_portfolio.empty else pd.Series(dtype=str)
    )
    prev_syms = set(prev_portfolio["symbol"].tolist()) if not prev_portfolio.empty else set()
    cur_syms = set(rebalance_portfolio["symbol"].tolist()) if not rebalance_portfolio.empty else set()
    bought = sorted(cur_syms - prev_syms)
    exited = sorted(prev_syms - cur_syms)

    price_map = (
        quotes_db.dropna(subset=["symbol"])
        .drop_duplicates(subset=["symbol"], keep="first")
        .set_index("symbol")["last_price"]
        .to_dict()
    )
    latest_w_map = (
        rebalance_portfolio.set_index(rebalance_portfolio["symbol"].astype(str).str.upper())["target_weight"].to_dict()
        if not rebalance_portfolio.empty
        else {}
    )
    prev_w_map = (
        prev_portfolio.set_index(prev_portfolio["symbol"].astype(str).str.upper())["target_weight"].to_dict()
        if not prev_portfolio.empty
        else {}
    )
    buy_rows = []
    for s in bought:
        px = float(price_map.get(s, np.nan)) if s in price_map else np.nan
        w = float(latest_w_map.get(s, 0.0))
        qty = int((float(tracking_capital_inr) * w) / px) if np.isfinite(px) and px > 0 else 0
        buy_rows.append({"symbol": s, "target_weight_pct": round(w * 100.0, 2), "indicative_qty": qty})
    exit_rows = []
    for s in exited:
        px = float(price_map.get(s, np.nan)) if s in price_map else np.nan
        w = float(prev_w_map.get(s, 0.0))
        qty = int((float(tracking_capital_inr) * w) / px) if np.isfinite(px) and px > 0 else 0
        exit_rows.append({"symbol": s, "prev_weight_pct": round(w * 100.0, 2), "indicative_qty": qty})

    rr1, rr2 = st.columns(2)
    rr1.markdown("**New Buys (vs previous rebalance run)**")
    rr1.dataframe(pd.DataFrame(buy_rows), hide_index=True, use_container_width=True)
    rr2.markdown("**Exits (vs previous rebalance run)**")
    rr2.dataframe(pd.DataFrame(exit_rows), hide_index=True, use_container_width=True)
    if not buy_rows and not exit_rows:
        st.info("No symbol-level entries/exits vs previous rebalance snapshot.")

    st.markdown("**Rebalance Order Sheet (Zerodha Upload Helper)**")
    all_syms = sorted(set(latest_w_map.keys()) | set(prev_w_map.keys()))
    order_rows = []
    for s in all_syms:
        tgt_w = float(latest_w_map.get(s, 0.0))
        prv_w = float(prev_w_map.get(s, 0.0))
        delta_w = float(tgt_w - prv_w)
        if abs(delta_w) < 1e-6:
            continue
        px = float(price_map.get(s, np.nan)) if s in price_map else np.nan
        qty = int((float(tracking_capital_inr) * abs(delta_w)) / px) if np.isfinite(px) and px > 0 else 0
        txn = "BUY" if delta_w > 0 else "SELL"
        order_rows.append(
            {
                "tradingsymbol": f"{s}",
                "exchange": "NSE",
                "transaction_type": txn,
                "quantity": int(max(qty, 0)),
                "order_type": "MARKET",
                "product": "CNC",
                "validity": "DAY",
                "last_price": round(px, 2) if np.isfinite(px) else np.nan,
                "prev_weight_pct": round(prv_w * 100.0, 2),
                "target_weight_pct": round(tgt_w * 100.0, 2),
                "delta_weight_pct": round(delta_w * 100.0, 2),
                "as_of_date": str(last_rebalance_date),
                "run_id": last_rebalance_run_id,
            }
        )
    orders_df = pd.DataFrame(order_rows)
    if orders_df.empty:
        st.info("No rebalance orders generated (weights unchanged vs previous rebalance snapshot).")
    else:
        st.dataframe(orders_df, hide_index=True, use_container_width=True)
        cdl1, cdl2 = st.columns(2)
        csv_bytes = orders_df.to_csv(index=False).encode("utf-8")
        cdl1.download_button(
            "Download Rebalance CSV",
            data=csv_bytes,
            file_name=f"zerodha_rebalance_orders_{last_rebalance_date}.csv",
            mime="text/csv",
        )
        try:
            xlsx_bytes = _df_to_excel_bytes(
                {
                    "orders": orders_df,
                    "new_buys": pd.DataFrame(buy_rows),
                    "exits": pd.DataFrame(exit_rows),
                }
            )
            cdl2.download_button(
                "Download Rebalance Excel",
                data=xlsx_bytes,
                file_name=f"zerodha_rebalance_orders_{last_rebalance_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            cdl2.warning(f"Excel export unavailable: {e}")

    st.subheader("Agent Weights")
    w_df = pd.DataFrame(
        [{"agent": k, "weight_pct": int(round(v * 100.0, 0))} for k, v in locked_agent_weights().items()]
    ).sort_values("weight_pct", ascending=False)
    st.dataframe(w_df, hide_index=True, use_container_width=True)

    st.subheader("Portfolio Targets (Risk-Adjusted Weights)")
    if portfolio_latest.empty:
        st.info("No portfolio targets found for latest run.")
    else:
        pf = portfolio_latest.copy()
        pf["symbol"] = pf["symbol"].astype(str).str.upper()
        pf["score"] = pd.to_numeric(pf["consensus_score"], errors="coerce").fillna(0.0)
        pf["target_weight_pct"] = (pd.to_numeric(pf["target_weight"], errors="coerce").fillna(0.0) * 100.0).round(2)
        pf["signal"] = pf["score"].map(lambda x: _signal_from_score(float(x), buy_cutoff, sell_cutoff, True))
        pf["prev_score"] = pf["symbol"].map(prev_score_map)
        pf["fusion_delta_pct"] = np.where(
            pd.to_numeric(pf["prev_score"], errors="coerce").abs() > 1e-9,
            ((pf["score"] - pd.to_numeric(pf["prev_score"], errors="coerce")) / pd.to_numeric(pf["prev_score"], errors="coerce").abs()) * 100.0,
            np.nan,
        )
        pf["fusion_score_display"] = [
            f"{int(round(sc * 100.0, 0))} ({float(dl):+.2f}%)" if pd.notna(dl) else f"{int(round(sc * 100.0, 0))} (-)"
            for sc, dl in zip(pf["score"], pf["fusion_delta_pct"])
        ]
        pf = pf.merge(
            quotes_db[["symbol", "px_date", "last_price", "db_change_1d_pct"]],
            on="symbol",
            how="left",
        )
        if not live_quotes.empty:
            pf = pf.merge(
                live_quotes[["symbol", "live_price", "live_change_1d_pct"]],
                on="symbol",
                how="left",
            )
            pf["last_price"] = np.where(pf["live_price"].notna(), pf["live_price"], pf["last_price"])
            pf["db_change_1d_pct"] = np.where(pf["live_change_1d_pct"].notna(), pf["live_change_1d_pct"], pf["db_change_1d_pct"])
        pf["reason"] = [
            _build_signal_reason(
                symbol=s,
                signal=str(sig),
                score=float(sc),
                confidence=float(cf),
                risk_reason=str(risk_reason_map.get(str(s), "pass")),
                buy_cutoff=buy_cutoff,
                sell_cutoff=sell_cutoff,
                signal_rows=signals_latest,
                day_change_pct=float(chg) if pd.notna(chg) else None,
            )
            for s, sig, sc, cf, chg in zip(
                pf["symbol"],
                pf["signal"],
                pf["score"],
                pf["consensus_confidence"],
                pf["db_change_1d_pct"],
            )
        ]
        pf["db_change_1d_pct"] = pd.to_numeric(pf["db_change_1d_pct"], errors="coerce").round(2)
        pv = pf[
            [
                "portfolio_rank",
                "symbol",
                "signal",
                "target_weight_pct",
                "fusion_score_display",
                "fusion_delta_pct",
                "last_price",
                "db_change_1d_pct",
                "reason",
            ]
        ]
        st.dataframe(_style_signal_and_change(pv, "signal", "db_change_1d_pct"), hide_index=True, use_container_width=True)

        st.markdown("**Sector Allocation (Current Portfolio)**")
        pf_sec = pf[["symbol", "target_weight"]].copy()
        pf_sec = pf_sec.merge(symbol_meta, on="symbol", how="left") if not symbol_meta.empty else pf_sec.assign(sector="UNKNOWN")
        pf_sec["sector"] = pf_sec["sector"].fillna("UNKNOWN")
        sec = (
            pf_sec.groupby("sector", as_index=False)
            .agg(target_weight=("target_weight", "sum"))
            .sort_values("target_weight", ascending=False)
        )
        sec["allocation_pct"] = (pd.to_numeric(sec["target_weight"], errors="coerce").fillna(0.0) * 100.0).round(2)
        invested = float(pd.to_numeric(pf_sec["target_weight"], errors="coerce").fillna(0.0).sum())
        cash_pct = max(0.0, (1.0 - invested) * 100.0)
        if cash_pct > 0.01:
            sec = pd.concat(
                [
                    sec[["sector", "allocation_pct"]],
                    pd.DataFrame([{"sector": "CASH", "allocation_pct": round(cash_pct, 2)}]),
                ],
                ignore_index=True,
            )
        else:
            sec = sec[["sector", "allocation_pct"]]
        st.dataframe(sec, hide_index=True, use_container_width=True)

    st.subheader("Watchlist (Top 25)")
    if watch_latest.empty:
        st.info("No watchlist rows found.")
    else:
        wx = watch_latest.copy()
        wx["symbol"] = wx["symbol"].astype(str).str.upper()
        wx["score"] = pd.to_numeric(wx["consensus_score"], errors="coerce").fillna(0.0)
        wx["prev_score"] = wx["symbol"].map(prev_score_map)
        wx["fusion_delta_pct"] = np.where(
            pd.to_numeric(wx["prev_score"], errors="coerce").abs() > 1e-9,
            ((wx["score"] - pd.to_numeric(wx["prev_score"], errors="coerce")) / pd.to_numeric(wx["prev_score"], errors="coerce").abs()) * 100.0,
            np.nan,
        )
        wx["fusion_score_display"] = [
            f"{int(round(sc * 100.0, 0))} ({float(dl):+.2f}%)" if pd.notna(dl) else f"{int(round(sc * 100.0, 0))} (-)"
            for sc, dl in zip(wx["score"], wx["fusion_delta_pct"])
        ]
        wx["signal"] = [
            _signal_from_score(float(sc), buy_cutoff, sell_cutoff, str(sym) in in_pf)
            for sym, sc in zip(wx["symbol"], wx["score"])
        ]
        wx["expected_upside_pct"] = (
            ((wx["score"] - buy_cutoff) / max(1.0 - buy_cutoff, 1e-6))
            .clip(lower=0.0)
            .mul(30.0)
            .round(0)
            .astype(int)
        )
        wx["upside_horizon_days"] = (
            30 + (1.0 - pd.to_numeric(wx["consensus_confidence"], errors="coerce").fillna(0.0)).clip(0.0, 1.0) * 60
        ).round(0).astype(int)
        wx = wx.merge(quotes_db[["symbol", "last_price", "db_change_1d_pct"]], on="symbol", how="left")
        if not live_quotes.empty:
            wx = wx.merge(live_quotes[["symbol", "live_price", "live_change_1d_pct"]], on="symbol", how="left")
            wx["last_price"] = np.where(wx["live_price"].notna(), wx["live_price"], wx["last_price"])
            wx["db_change_1d_pct"] = np.where(wx["live_change_1d_pct"].notna(), wx["live_change_1d_pct"], wx["db_change_1d_pct"])
        wx["db_change_1d_pct"] = pd.to_numeric(wx["db_change_1d_pct"], errors="coerce").round(2)
        wx["explainability"] = [
            _build_signal_reason(
                symbol=s,
                signal=str(sig),
                score=float(sc),
                confidence=float(cf),
                risk_reason=str(rr),
                buy_cutoff=buy_cutoff,
                sell_cutoff=sell_cutoff,
                signal_rows=signals_latest,
                day_change_pct=float(chg) if pd.notna(chg) else None,
            )
            for s, sig, sc, cf, rr, chg in zip(
                wx["symbol"],
                wx["signal"],
                wx["score"],
                wx["consensus_confidence"],
                wx["risk_reason"],
                wx["db_change_1d_pct"],
            )
        ]
        wv = wx[
            [
                "watch_rank",
                "symbol",
                "signal",
                "fusion_score_display",
                "fusion_delta_pct",
                "last_price",
                "db_change_1d_pct",
                "expected_upside_pct",
                "upside_horizon_days",
                "explainability",
            ]
        ].head(int(top_n_watchlist))
        st.dataframe(_style_signal_and_change(wv, "signal", "db_change_1d_pct"), hide_index=True, use_container_width=True)

    st.subheader("Backtest Analytics")
    bt_equity = bundle["bt_equity"].copy()
    bt_trades = bundle["bt_trades"].copy()
    if bt_runs.empty or bt_equity.empty:
        st.warning("No persisted backtest data found.")
    else:
        bt_runs["stats"] = bt_runs["stats_json"].map(_safe_json_load)
        bt_runs["CAGR"] = bt_runs["stats"].map(lambda x: float(x.get("CAGR", 0.0)))
        bt_runs["Trade Win Rate"] = bt_runs["stats"].map(lambda x: float(x.get("Trade Win Rate", 0.0)))
        bt_runs["Max Drawdown"] = bt_runs["stats"].map(lambda x: float(x.get("Max Drawdown", 0.0)))
        st.dataframe(
            bt_runs[["run_id", "run_ts", "start_date", "end_date", "CAGR", "Trade Win Rate", "Max Drawdown"]],
            hide_index=True,
            use_container_width=True,
        )

        selected_run_ids = bt_runs["run_id"].astype(str).tolist() if bt_scope == "All runs" else [str(bt_runs.iloc[0]["run_id"])]
        eq = bt_equity[bt_equity["run_id"].astype(str).isin(selected_run_ids)].copy()
        tr = bt_trades[bt_trades["run_id"].astype(str).isin(selected_run_ids)].copy()
        eq["date"] = _to_dt_ns(eq["date"])
        tr["date"] = _to_dt_ns(tr["date"])
        all_years = sorted(eq["date"].dt.year.dropna().astype(int).unique().tolist())
        all_months = sorted(eq["date"].dt.month.dropna().astype(int).unique().tolist())
        fc1, fc2 = st.columns(2)
        years_sel = fc1.multiselect("Year filter", all_years, default=all_years)
        months_sel = fc2.multiselect("Month filter", all_months, default=all_months)
        eq_f = _filter_dates(eq, "date", years_sel, months_sel)
        tr_f = _filter_dates(tr, "date", years_sel, months_sel)
        stats_f = _period_stats(eq_f, tr_f)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Filtered CAGR", f"{stats_f['CAGR']:.2%}")
        k2.metric("Filtered Trade Win", f"{stats_f['Trade Win Rate']:.2%}")
        k3.metric("Filtered MaxDD", f"{stats_f['Max Drawdown']:.2%}")
        k4.metric("Filtered Sharpe", f"{stats_f['Sharpe']:.2f}")
        k5.metric("Trades", f"{int(stats_f['Trades'])}")
        if not eq_f.empty:
            st.line_chart(eq_f.sort_values("date").set_index("date")[["equity"]], height=280)
        closed = _build_closed_trips(tr_f)
        st.markdown("Past Portfolio Holdings (Entry/Exit/PnL)")
        if closed.empty:
            st.info("No closed holdings for selected period.")
        else:
            cv = closed.copy()
            cv["pnl_pct"] = (pd.to_numeric(cv["pnl_pct"], errors="coerce").fillna(0.0) * 100.0).round(2)
            st.dataframe(cv, hide_index=True, use_container_width=True)

    st.subheader("Stock Advisor")
    symbols = bundle["symbols"]["symbol"].astype(str).tolist() if not bundle["symbols"].empty else []
    q = st.text_input("Search symbol (Nifty-300 universe)").strip().upper()
    suggestions = [s for s in symbols if q in s][:100] if q else symbols[:100]
    if not suggestions:
        st.info("No symbols available.")
    else:
        selected_symbol = st.selectbox("Symbol", options=suggestions, index=0)
        advisor = _build_symbol_advisor(selected_symbol, consensus_latest, signals_latest, in_pf, buy_cutoff, sell_cutoff)
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Signal", advisor["signal"])
        a2.metric("Fusion Score", advisor["fusion_score"])
        a3.metric("Score Percentile", advisor["score_percentile"])
        a4.metric("Expected Upside", f"{advisor['expected_upside_pct']}%")
        st.write(f"Expected waiting period: ~{advisor['expected_horizon_days']} days")
        st.write(f"Why: {advisor['why']}")
        st.write(f"Cutoffs: Buy >= {int(round(buy_cutoff*100,0))}, Exit <= {int(round(sell_cutoff*100,0))}")

    st.subheader("Operations Console")
    ref = locked_reference_metrics()
    rc_path = Path("data/reports/release_checks_ai_agent_stable_v1.json")
    rc = _safe_json_load(rc_path.read_text(encoding="utf-8")) if rc_path.exists() else {}
    model_report_path = Path("data/reports/ai_agent_stable_stocks_v1_latest.json")
    model_report = _safe_json_load(model_report_path.read_text(encoding="utf-8")) if model_report_path.exists() else {}
    rebalance_allowed = bool(
        (
            ((model_report.get("pretrade_gate") or {}).get("rebalance_allowed"))
            if isinstance(model_report, dict)
            else False
        )
    )
    c1, c2 = st.columns(2)
    c1.write(
        {
            "reference_cagr": f"{float(ref.get('cagr', 0.0)):.2%}",
            "reference_trade_win": f"{float(ref.get('trade_win_rate', 0.0)):.2%}",
            "reference_maxdd": f"{float(ref.get('max_drawdown', 0.0)):.2%}",
            "reference_sharpe": f"{float(ref.get('sharpe', 0.0)):.2f}",
        }
    )
    c2.write(
        {
            "broker_auto_execution": "DISABLED (risk gate)",
            "rebalance_allowed": rebalance_allowed,
            "rebalance_mode": (
                ((model_report.get("pretrade_gate") or {}).get("rebalance_mode"))
                if isinstance(model_report, dict)
                else None
            ),
            "zerodha_keys_present": bool(os.getenv("KITE_API_KEY") and os.getenv("KITE_API_SECRET")),
            "newsapi_key_present": bool(os.getenv("NEWSAPI_KEY")),
            "finnhub_key_present": bool(os.getenv("FINNHUB_API_KEY")),
            "release_report_found": bool(rc),
        }
    )
    health_ops = pd.DataFrame()
    decision_core_status = "BROKEN"
    rebalance_strict_status = "BROKEN"
    try:
        health_ops = _load_pipeline_health(db_path)
    except Exception:
        health_ops = pd.DataFrame()

    def _rollup_pipeline_status(df: pd.DataFrame, pipelines: list[str]) -> str:
        if df.empty:
            return "BROKEN"
        z = df[df["pipeline"].isin(pipelines)].copy()
        if z.empty:
            return "BROKEN"
        s = z["status"].astype(str).str.upper()
        if s.isin(["BROKEN", "ERROR"]).any():
            return "BROKEN"
        if (s == "STALE").any():
            return "STALE"
        if (s == "WORKING").all():
            return "OK"
        return "BROKEN"

    if not health_ops.empty:
        decision_core_status = _rollup_pipeline_status(health_ops, ["prices_daily_v1", "delivery_daily_v1"])
        rebalance_strict_status = _rollup_pipeline_status(
            health_ops,
            ["prices_daily_v1", "delivery_daily_v1", "bulk_block_deals"],
        )

    runtime_snapshot_mode = Path(db_path).name.lower() == "ownership_runtime.duckdb"
    has_incremental_report = Path("data/reports/incremental_cycle_latest.json").exists()
    has_scheduled_report = Path("data/reports/scheduled_daily_cycle_v1_latest.json").exists()

    ops_rows = [
        {
            "item": "Decision core pipelines",
            "status": decision_core_status,
            "action": "Core feeds required for decision generation.",
            "how_to_fix": ".venv/bin/python src/pipeline/repair_required_pipelines_v1.py --db-path data/ownership.duckdb --max-rounds 3",
        },
        {
            "item": "Rebalance strict feeds",
            "status": rebalance_strict_status,
            "action": "Strict mode requires bulk feed also WORKING.",
            "how_to_fix": ".venv/bin/python src/pipeline/repair_required_pipelines_v1.py --db-path data/ownership.duckdb --max-rounds 3",
        },
        {
            "item": "Release checks report",
            "status": ("OK" if bool(rc) else ("OPTIONAL_OFF" if runtime_snapshot_mode else "BROKEN")),
            "action": "Run release checks to regenerate report.",
            "how_to_fix": (
                "Published in runtime snapshot branch; local file may be absent in cloud app container."
                if runtime_snapshot_mode
                else ".venv/bin/python src/qa/run_release_checks_ai_agent_stable_v1.py --db-path data/ownership.duckdb"
            ),
        },
        {
            "item": "Zerodha keys",
            "status": (
                "OK"
                if bool(os.getenv("KITE_API_KEY") and os.getenv("KITE_API_SECRET"))
                else ("OPTIONAL_OFF" if runtime_snapshot_mode else "BROKEN")
            ),
            "action": "Set broker keys only when enabling broker execution gate.",
            "how_to_fix": "export KITE_API_KEY=... && export KITE_API_SECRET=...",
        },
        {
            "item": "News provider keys",
            "status": (
                "OK"
                if bool(os.getenv("NEWSAPI_KEY") and os.getenv("FINNHUB_API_KEY"))
                else ("OPTIONAL_OFF" if runtime_snapshot_mode else "BROKEN")
            ),
            "action": "Set NewsAPI + Finnhub keys for live sentiment feed.",
            "how_to_fix": "export NEWSAPI_KEY=... && export FINNHUB_API_KEY=...",
        },
        {
            "item": "Daily automation report",
            "status": (
                "OK"
                if (has_incremental_report or has_scheduled_report)
                else ("OPTIONAL_OFF" if runtime_snapshot_mode else "BROKEN")
            ),
            "action": "Run daily automation cycle.",
            "how_to_fix": (
                "Cloud app reads runtime snapshot; automation report is tracked in GitHub Actions artifacts."
                if runtime_snapshot_mode
                else ".venv/bin/python src/pipeline/run_incremental_data_and_retrain_v1.py --daily-auto --db-path data/ownership.duckdb"
            ),
        },
        {
            "item": "Version snapshot registry",
            "status": (
                "OK"
                if Path("data/reports/version_snapshot_latest.json").exists()
                else ("OPTIONAL_OFF" if runtime_snapshot_mode else "BROKEN")
            ),
            "action": "Record version snapshot for current increment.",
            "how_to_fix": (
                "Run version snapshot in pipeline runner; cloud app container does not persist local report files."
                if runtime_snapshot_mode
                else ".venv/bin/python src/versioning/record_version_snapshot_v1.py --db-path data/ownership.duckdb --context manual"
            ),
        },
    ]
    ops_df = pd.DataFrame(ops_rows)
    ops_sty = ops_df.style.apply(
        lambda s: [_status_color(v) if s.name == "status" else "" for v in s],
        subset=["status"],
        axis=0,
    )
    st.markdown("**Operational Action Board**")
    st.dataframe(ops_sty, hide_index=True, use_container_width=True)

    st.markdown("**Pipeline Health (Data + Model Refresh)**")
    try:
        health = _load_pipeline_health(db_path)
    except Exception as e:
        st.error(f"Failed to compute pipeline health: {e}")
        health = pd.DataFrame()
    if health.empty:
        st.info("No pipeline health rows available.")
    else:
        hv = health[
            [
                "pipeline",
                "status",
                "as_of_ref_date",
                "last_date",
                "lag_days",
                "calendar_lag_days",
                "row_count",
                "symbol_count",
                "message",
                "notes",
            ]
        ].copy()
        hv["last_date"] = _to_dt_ns(hv["last_date"]).dt.date
        hv = hv.rename(
            columns={
                "lag_days": "market_lag_days",
            }
        )
        hsty = hv.style.apply(
            lambda s: [_status_color(v) if s.name == "status" else "" for v in s],
            subset=["status"],
            axis=0,
        )
        st.dataframe(hsty, hide_index=True, use_container_width=True)

    cycle_report_path = Path("data/reports/incremental_cycle_latest.json")
    scheduled_report_path = Path("data/reports/scheduled_daily_cycle_v1_latest.json")
    selected_cycle_path = cycle_report_path if cycle_report_path.exists() else (
        scheduled_report_path if scheduled_report_path.exists() else None
    )
    if selected_cycle_path is not None:
        cyc = _safe_json_load(selected_cycle_path.read_text(encoding="utf-8"))
        st.markdown("**Daily Automation Snapshot**")
        st.write(
            {
                "source_report": selected_cycle_path.name,
                "last_cycle_as_of_date": cyc.get("as_of_date"),
                "last_cycle_repair_as_of_date": cyc.get("repair_as_of_date"),
                "version_snapshot": cyc.get("version_snapshot", {}),
                "status": cyc.get("status"),
                "decision_mode": cyc.get("decision_mode"),
            }
        )
        step_actions = {
            "refresh_prices": "Refresh stale market CSV and canonical prices.",
            "build_prices_table": "Rebuild prices_daily from refreshed CSV.",
            "rebuild_prices_canonical": "Rebuild prices_daily_v1.",
            "update_delivery_live": "Fetch latest NSE delivery file.",
            "sync_delivery_v1": "Sync delivery_daily into delivery_daily_v1.",
            "update_bulk_live": "Fetch latest NSE bulk/block deals.",
            "ingest_news": "Ingest news/social feed from API providers.",
            "pipeline_health": "Evaluate pipeline health and persist status.",
            "release_checks": "Run institutional release gates.",
            "locked_decision": "Run daily fusion cycle and persist consensus.",
            "retrain_cycle": "Run scheduled optimizer/walk-forward challenger.",
        }
        step_fix = {
            "update_delivery_live": ".venv/bin/python src/pipeline/repair_required_pipelines_v1.py --db-path data/ownership.duckdb --max-rounds 3",
            "update_bulk_live": ".venv/bin/python src/pipeline/repair_required_pipelines_v1.py --db-path data/ownership.duckdb --max-rounds 3",
            "ingest_news": "Set NEWSAPI_KEY/FINNHUB_API_KEY and rerun.",
            "locked_decision": "Run locked model runner manually and inspect logs.",
        }
        rows = []
        raw_steps = cyc.get("steps") or {}
        if isinstance(raw_steps, dict):
            for step, payload in raw_steps.items():
                status = str((payload or {}).get("status", "UNKNOWN")).upper()
                rows.append(
                    {
                        "step": step,
                        "status": status,
                        "action": step_actions.get(step, "Review step output."),
                        "details": str((payload or {}).get("error", ""))[:180],
                        "how_to_fix": step_fix.get(step, "Run --daily-auto and inspect this step status."),
                    }
                )
        elif isinstance(raw_steps, list):
            for payload in raw_steps:
                step = str((payload or {}).get("step", "unknown"))
                status = str((payload or {}).get("status", "UNKNOWN")).upper()
                details = str((payload or {}).get("stderr_tail") or (payload or {}).get("stdout_tail") or "")[:180]
                rows.append(
                    {
                        "step": step,
                        "status": status,
                        "action": step_actions.get(step, "Review step output."),
                        "details": details,
                        "how_to_fix": step_fix.get(step, "Inspect workflow logs for this failed step."),
                    }
                )
        if rows:
            ds = pd.DataFrame(rows)
            ds_sty = ds.style.apply(
                lambda s: [_status_color(v) if s.name == "status" else "" for v in s],
                subset=["status"],
                axis=0,
            )
            st.dataframe(ds_sty, hide_index=True, use_container_width=True)
    else:
        st.markdown("**Daily Automation Snapshot**")
        st.info("No automation report available in this runtime snapshot yet.")


if __name__ == "__main__":
    main()
