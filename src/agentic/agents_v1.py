from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

try:
    from src.agentic.contracts_v1 import ActionVote, AgentDefinition, RegimeState, clamp
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.agentic.contracts_v1 import ActionVote, AgentDefinition, RegimeState, clamp


def table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    out = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema='main' AND table_name=?
        """,
        [table_name],
    ).fetchone()
    return bool(out and out[0] > 0)


def _vote_from_score(score: float, buy_threshold: float, sell_threshold: float) -> ActionVote:
    if score >= buy_threshold:
        return ActionVote.BUY
    if score <= sell_threshold:
        return ActionVote.SELL
    return ActionVote.HOLD


def _safe_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(0.5, index=series.index, dtype=float)
    return x.rank(pct=True, ascending=ascending).fillna(0.5).clip(0.0, 1.0)


def _neutral_frame(symbols: Iterable[str], reason: str, confidence: float = 0.05) -> pd.DataFrame:
    syms = pd.Index(sorted(set(str(s).strip().upper() for s in symbols if str(s).strip() != "")))
    out = pd.DataFrame({"symbol": syms})
    out["score"] = 0.5
    out["confidence"] = float(clamp(confidence, 0.0, 1.0))
    out["action"] = ActionVote.HOLD.value
    out["rationale_json"] = json.dumps({"status": "neutral", "reason": reason}, ensure_ascii=True)
    return out


@dataclass(frozen=True)
class MarketContext:
    panel: pd.DataFrame
    latest: pd.DataFrame
    symbols: list[str]
    max_price_date: date | None


def load_market_context(
    conn: duckdb.DuckDBPyConnection,
    as_of_date: date,
    lookback_days: int = 420,
    universe_limit: int = 0,
) -> MarketContext:
    if not table_exists(conn, "prices_daily_v1"):
        return MarketContext(
            panel=pd.DataFrame(columns=["date", "symbol", "close", "volume"]),
            latest=pd.DataFrame(columns=["date", "symbol", "close", "volume"]),
            symbols=[],
            max_price_date=None,
        )
    if not table_exists(conn, "symbol_master"):
        return MarketContext(
            panel=pd.DataFrame(columns=["date", "symbol", "close", "volume"]),
            latest=pd.DataFrame(columns=["date", "symbol", "close", "volume"]),
            symbols=[],
            max_price_date=None,
        )

    start_date = as_of_date - timedelta(days=int(max(lookback_days, 120)))
    df = conn.execute(
        """
        SELECT
            CAST(p.date AS DATE) AS date,
            UPPER(TRIM(p.canonical_symbol)) AS symbol,
            CAST(p.close AS DOUBLE) AS close,
            CAST(p.volume AS DOUBLE) AS volume
        FROM prices_daily_v1 p
        JOIN symbol_master sm
          ON UPPER(TRIM(sm.canonical_symbol)) = UPPER(TRIM(p.canonical_symbol))
        WHERE COALESCE(sm.in_universe, TRUE) = TRUE
          AND CAST(p.date AS DATE) BETWEEN ? AND ?
          AND p.close IS NOT NULL
        ORDER BY symbol, date
        """,
        [start_date, as_of_date],
    ).df()

    if df.empty:
        return MarketContext(
            panel=pd.DataFrame(columns=["date", "symbol", "close", "volume"]),
            latest=pd.DataFrame(columns=["date", "symbol", "close", "volume"]),
            symbols=[],
            max_price_date=None,
        )

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date", "symbol", "close"]).copy()
    if universe_limit and universe_limit > 0:
        keep = sorted(df["symbol"].dropna().unique().tolist())[: int(universe_limit)]
        df = df[df["symbol"].isin(keep)].copy()

    latest = (
        df.sort_values(["symbol", "date"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .sort_values(["date", "symbol"], ascending=[False, True])
        .reset_index(drop=True)
    )
    symbols = latest["symbol"].dropna().astype(str).unique().tolist()
    max_price_date = (
        latest["date"].max().date() if not latest.empty and latest["date"].notna().any() else None
    )
    return MarketContext(panel=df, latest=latest, symbols=symbols, max_price_date=max_price_date)


def evaluate_technical_agent(ctx: MarketContext) -> pd.DataFrame:
    if ctx.panel.empty or not ctx.symbols:
        return _neutral_frame([], "prices_daily_v1 unavailable for technical agent")

    df = ctx.panel.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", group_keys=False)
    df["ret_1"] = g["close"].pct_change()
    df["ret_21"] = g["close"].pct_change(21)
    df["ret_63"] = g["close"].pct_change(63)
    df["vol_21"] = g["ret_1"].transform(lambda s: s.rolling(21, min_periods=10).std())
    df["high_252"] = g["close"].transform(lambda s: s.rolling(252, min_periods=40).max())
    df["dist_high"] = (df["close"] / df["high_252"]) - 1.0
    latest = df.groupby("symbol", as_index=False).tail(1).copy()

    r_ret63 = _safe_rank(latest["ret_63"], ascending=True)
    r_ret21 = _safe_rank(latest["ret_21"], ascending=True)
    r_dist = _safe_rank(latest["dist_high"], ascending=True)
    r_vol = _safe_rank(latest["vol_21"], ascending=False)

    latest["score"] = (
        0.38 * r_ret63
        + 0.27 * r_ret21
        + 0.20 * r_dist
        + 0.15 * r_vol
    ).clip(0.0, 1.0)

    feature_cov = latest[["ret_21", "ret_63", "vol_21", "dist_high"]].notna().sum(axis=1) / 4.0
    latest["confidence"] = (0.20 + 0.80 * feature_cov).clip(0.0, 1.0)
    latest["action"] = latest["score"].map(
        lambda x: _vote_from_score(float(x), buy_threshold=0.67, sell_threshold=0.33).value
    )

    latest["rationale_json"] = [
        json.dumps(
            {
                "agent": "technical",
                "ret_21": None if pd.isna(r21) else float(r21),
                "ret_63": None if pd.isna(r63) else float(r63),
                "vol_21": None if pd.isna(v21) else float(v21),
                "dist_high_252": None if pd.isna(dh) else float(dh),
            },
            ensure_ascii=True,
        )
        for r21, r63, v21, dh in zip(
            latest["ret_21"], latest["ret_63"], latest["vol_21"], latest["dist_high"]
        )
    ]
    return latest[["symbol", "score", "confidence", "action", "rationale_json"]].reset_index(drop=True)


def evaluate_ownership_agent(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
    as_of_date: date,
) -> pd.DataFrame:
    if not symbols:
        return _neutral_frame([], "empty universe for ownership agent")
    if not table_exists(conn, "delivery_daily_v1"):
        return _neutral_frame(symbols, "delivery_daily_v1 missing for ownership agent")

    start_date = as_of_date - timedelta(days=220)
    delivery = conn.execute(
        """
        SELECT
            UPPER(TRIM(canonical_symbol)) AS symbol,
            CAST(date AS DATE) AS date,
            CAST(delivery_pct AS DOUBLE) AS delivery_pct
        FROM delivery_daily_v1
        WHERE CAST(date AS DATE) BETWEEN ? AND ?
        """,
        [start_date, as_of_date],
    ).df()
    if delivery.empty:
        return _neutral_frame(symbols, "delivery_daily_v1 has no rows in lookback window")

    delivery["date"] = pd.to_datetime(delivery["date"])
    delivery = delivery[delivery["symbol"].isin(symbols)].copy()
    delivery = delivery.sort_values(["symbol", "date"])
    g = delivery.groupby("symbol", group_keys=False)
    delivery["mean_60"] = g["delivery_pct"].transform(lambda s: s.rolling(60, min_periods=15).mean())
    delivery["std_60"] = g["delivery_pct"].transform(lambda s: s.rolling(60, min_periods=15).std(ddof=0))
    delivery["delivery_z"] = (
        (delivery["delivery_pct"] - delivery["mean_60"])
        / delivery["std_60"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    d_latest = delivery.groupby("symbol", as_index=False).tail(1).copy()

    bulk = pd.DataFrame(columns=["symbol", "net_notional", "event_count"])
    if table_exists(conn, "bulk_block_deals"):
        b_start = as_of_date - timedelta(days=90)
        bulk_raw = conn.execute(
            """
            SELECT
                UPPER(TRIM(symbol)) AS symbol,
                UPPER(TRIM(COALESCE(side, ''))) AS side,
                CAST(COALESCE(qty, 0.0) AS DOUBLE) AS qty,
                CAST(COALESCE(price, 0.0) AS DOUBLE) AS price
            FROM bulk_block_deals
            WHERE CAST(date AS DATE) BETWEEN ? AND ?
            """,
            [b_start, as_of_date],
        ).df()
        if not bulk_raw.empty:
            bulk_raw = bulk_raw[bulk_raw["symbol"].isin(symbols)].copy()
            side = bulk_raw["side"].astype(str)
            sign = np.where(
                side.str.contains("BUY", na=False),
                1.0,
                np.where(side.str.contains("SELL", na=False), -1.0, 0.0),
            )
            bulk_raw["signed_notional"] = sign * bulk_raw["qty"] * bulk_raw["price"]
            bulk = (
                bulk_raw.groupby("symbol", as_index=False)
                .agg(
                    net_notional=("signed_notional", "sum"),
                    event_count=("signed_notional", "count"),
                )
            )

    all_syms = pd.DataFrame({"symbol": sorted(set(symbols))})
    out = all_syms.merge(d_latest[["symbol", "delivery_pct", "delivery_z"]], on="symbol", how="left")
    out = out.merge(bulk, on="symbol", how="left")
    out["net_notional"] = pd.to_numeric(out["net_notional"], errors="coerce").fillna(0.0)
    out["event_count"] = pd.to_numeric(out["event_count"], errors="coerce").fillna(0.0)

    r_dz = _safe_rank(out["delivery_z"], ascending=True)
    r_bulk = _safe_rank(out["net_notional"], ascending=True)
    out["score"] = (0.70 * r_dz + 0.30 * r_bulk).clip(0.0, 1.0)

    has_d = out["delivery_z"].notna().astype(float)
    has_b = (out["event_count"] > 0).astype(float)
    bulk_strength = np.minimum(out["event_count"] / 20.0, 1.0)
    out["confidence"] = (0.20 + 0.45 * has_d + 0.20 * has_b + 0.15 * bulk_strength).clip(0.0, 1.0)
    out["action"] = out["score"].map(
        lambda x: _vote_from_score(float(x), buy_threshold=0.65, sell_threshold=0.35).value
    )
    out["rationale_json"] = [
        json.dumps(
            {
                "agent": "ownership",
                "delivery_pct": None if pd.isna(d) else float(d),
                "delivery_z": None if pd.isna(z) else float(z),
                "bulk_net_notional": float(n),
                "bulk_event_count": int(c),
            },
            ensure_ascii=True,
        )
        for d, z, n, c in zip(
            out["delivery_pct"], out["delivery_z"], out["net_notional"], out["event_count"]
        )
    ]
    return out[["symbol", "score", "confidence", "action", "rationale_json"]].reset_index(drop=True)


def evaluate_news_agent(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
    as_of_date: date,
) -> pd.DataFrame:
    if not symbols:
        return _neutral_frame([], "empty universe for news agent")
    if not table_exists(conn, "news_social_raw_v1"):
        return _neutral_frame(symbols, "news_social_raw_v1 missing for news agent")

    start_date = as_of_date - timedelta(days=10)
    raw = conn.execute(
        """
        SELECT
            UPPER(TRIM(canonical_symbol)) AS symbol,
            CAST(COALESCE(sentiment_score, 0.5) AS DOUBLE) AS sentiment_score,
            CAST(COALESCE(relevance_score, 0.5) AS DOUBLE) AS relevance_score,
            CAST(COALESCE(source_confidence, 0.5) AS DOUBLE) AS source_confidence
        FROM news_social_raw_v1
        WHERE CAST(date AS DATE) BETWEEN ? AND ?
        """,
        [start_date, as_of_date],
    ).df()
    if raw.empty:
        return _neutral_frame(symbols, "news_social_raw_v1 has no rows in lookback window")

    raw = raw[raw["symbol"].isin(symbols)].copy()
    if raw.empty:
        return _neutral_frame(symbols, "news rows exist but none map to active universe symbols")

    raw["sentiment_score"] = pd.to_numeric(raw["sentiment_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    raw["relevance_score"] = pd.to_numeric(raw["relevance_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    raw["source_confidence"] = pd.to_numeric(raw["source_confidence"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    raw["w"] = (raw["relevance_score"] * raw["source_confidence"]).clip(lower=1e-6)
    raw["wx"] = raw["w"] * raw["sentiment_score"]

    agg = raw.groupby("symbol", as_index=False).agg(
        weighted_sent=("wx", "sum"),
        total_w=("w", "sum"),
        avg_source_conf=("source_confidence", "mean"),
        event_count=("w", "count"),
    )
    agg["weighted_sentiment"] = (agg["weighted_sent"] / agg["total_w"].replace(0.0, np.nan)).fillna(0.5)

    all_syms = pd.DataFrame({"symbol": sorted(set(symbols))})
    out = all_syms.merge(
        agg[["symbol", "weighted_sentiment", "avg_source_conf", "event_count"]],
        on="symbol",
        how="left",
    )
    out["weighted_sentiment"] = pd.to_numeric(out["weighted_sentiment"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    out["avg_source_conf"] = pd.to_numeric(out["avg_source_conf"], errors="coerce").fillna(0.3).clip(0.0, 1.0)
    out["event_count"] = pd.to_numeric(out["event_count"], errors="coerce").fillna(0.0)
    buzz_rank = _safe_rank(np.log1p(out["event_count"]), ascending=True)
    out["score"] = (0.80 * out["weighted_sentiment"] + 0.20 * buzz_rank).clip(0.0, 1.0)
    out["confidence"] = (
        0.15
        + 0.65 * out["avg_source_conf"]
        + 0.20 * np.minimum(out["event_count"] / 25.0, 1.0)
    ).clip(0.0, 1.0)
    out["action"] = out["score"].map(
        lambda x: _vote_from_score(float(x), buy_threshold=0.70, sell_threshold=0.30).value
    )
    out["rationale_json"] = [
        json.dumps(
            {
                "agent": "news",
                "weighted_sentiment": float(s),
                "event_count": int(c),
                "avg_source_conf": float(cf),
            },
            ensure_ascii=True,
        )
        for s, c, cf in zip(
            out["weighted_sentiment"], out["event_count"], out["avg_source_conf"]
        )
    ]
    return out[["symbol", "score", "confidence", "action", "rationale_json"]].reset_index(drop=True)


def evaluate_fundamental_agent(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
    as_of_date: date,
) -> pd.DataFrame:
    if not symbols:
        return _neutral_frame([], "empty universe for fundamental agent")
    if not table_exists(conn, "fundamentals_daily_v1"):
        return _neutral_frame(
            symbols,
            "fundamentals_daily_v1 missing; fundamental agent parked to neutral",
            confidence=0.02,
        )

    cols = {
        r[0]
        for r in conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name='fundamentals_daily_v1'
            """
        ).fetchall()
    }
    required = {"symbol", "date", "roe", "debt_to_equity", "eps_growth"}
    if not required.issubset(cols):
        return _neutral_frame(
            symbols,
            "fundamentals_daily_v1 missing required columns (symbol,date,roe,debt_to_equity,eps_growth)",
            confidence=0.02,
        )

    query = """
        SELECT
            UPPER(TRIM(symbol)) AS symbol,
            CAST(roe AS DOUBLE) AS roe,
            CAST(debt_to_equity AS DOUBLE) AS debt_to_equity,
            CAST(eps_growth AS DOUBLE) AS eps_growth
        FROM (
            SELECT
                symbol,
                date,
                roe,
                debt_to_equity,
                eps_growth,
                ROW_NUMBER() OVER (PARTITION BY UPPER(TRIM(symbol)) ORDER BY CAST(date AS DATE) DESC) AS rn
            FROM fundamentals_daily_v1
            WHERE CAST(date AS DATE) <= ?
        ) t
        WHERE rn = 1
    """
    base = conn.execute(query, [as_of_date]).df()
    all_syms = pd.DataFrame({"symbol": sorted(set(symbols))})
    out = all_syms.merge(base, on="symbol", how="left")

    r_roe = _safe_rank(out["roe"], ascending=True)
    r_eps = _safe_rank(out["eps_growth"], ascending=True)
    r_debt = _safe_rank(out["debt_to_equity"], ascending=False)
    out["score"] = (0.40 * r_roe + 0.35 * r_eps + 0.25 * r_debt).clip(0.0, 1.0)
    cov = out[["roe", "eps_growth", "debt_to_equity"]].notna().sum(axis=1) / 3.0
    out["confidence"] = (0.10 + 0.90 * cov).clip(0.0, 1.0)
    out["action"] = out["score"].map(
        lambda x: _vote_from_score(float(x), buy_threshold=0.68, sell_threshold=0.32).value
    )
    out["rationale_json"] = [
        json.dumps(
            {
                "agent": "fundamental",
                "roe": None if pd.isna(roe) else float(roe),
                "eps_growth": None if pd.isna(eps) else float(eps),
                "debt_to_equity": None if pd.isna(de) else float(de),
            },
            ensure_ascii=True,
        )
        for roe, eps, de in zip(out["roe"], out["eps_growth"], out["debt_to_equity"])
    ]
    return out[["symbol", "score", "confidence", "action", "rationale_json"]].reset_index(drop=True)


def detect_regime(ctx: MarketContext) -> RegimeState:
    if ctx.panel.empty:
        return RegimeState(
            label="unknown",
            score=0.50,
            exposure_scalar=1.0,
            breadth=0.50,
            market_ret_63=0.0,
        )

    x = ctx.panel.sort_values(["symbol", "date"]).copy()
    g = x.groupby("symbol", group_keys=False)
    x["sma_200"] = g["close"].transform(lambda s: s.rolling(200, min_periods=40).mean())
    x["above_sma200"] = (x["close"] > x["sma_200"]).astype(float)
    breadth = x.groupby("date")["above_sma200"].mean().sort_index()

    mkt = x.groupby("date")["close"].mean().sort_index()
    mkt_ma200 = mkt.rolling(200, min_periods=40).mean()
    mkt_ret_63 = mkt.pct_change(63)
    d = mkt.index.max()
    if pd.isna(d):
        return RegimeState(
            label="unknown",
            score=0.50,
            exposure_scalar=1.0,
            breadth=0.50,
            market_ret_63=0.0,
        )

    m = float(mkt.loc[d])
    ma = float(mkt_ma200.loc[d]) if pd.notna(mkt_ma200.loc[d]) else m
    r63 = float(mkt_ret_63.loc[d]) if pd.notna(mkt_ret_63.loc[d]) else 0.0
    br = float(breadth.loc[d]) if pd.notna(breadth.loc[d]) else 0.50

    if (m > ma) and (r63 > 0.0) and (br >= 0.55):
        return RegimeState(
            label="bull",
            score=0.75,
            exposure_scalar=1.15,
            breadth=br,
            market_ret_63=r63,
        )
    if (m < ma) and (r63 < 0.0) and (br < 0.45):
        return RegimeState(
            label="bear",
            score=0.30,
            exposure_scalar=0.85,
            breadth=br,
            market_ret_63=r63,
        )
    return RegimeState(
        label="sideways",
        score=0.50,
        exposure_scalar=1.0,
        breadth=br,
        market_ret_63=r63,
    )


def default_agents(weight_overrides: dict[str, float] | None = None) -> list[AgentDefinition]:
    base = [
        AgentDefinition(name="technical", weight=0.40, enabled=True, min_confidence=0.05),
        AgentDefinition(name="ownership", weight=0.25, enabled=True, min_confidence=0.05),
        AgentDefinition(name="fundamental", weight=0.25, enabled=True, min_confidence=0.02),
        AgentDefinition(name="news", weight=0.10, enabled=True, min_confidence=0.02),
    ]
    if not weight_overrides:
        return base

    out = []
    for d in base:
        if d.name in weight_overrides:
            w = float(weight_overrides[d.name])
            out.append(
                AgentDefinition(
                    name=d.name,
                    weight=max(w, 0.0),
                    enabled=d.enabled,
                    min_confidence=d.min_confidence,
                )
            )
        else:
            out.append(d)
    s = sum(x.weight for x in out if x.enabled)
    if s <= 0:
        return base
    return [
        AgentDefinition(
            name=x.name,
            weight=(x.weight / s) if x.enabled else 0.0,
            enabled=x.enabled,
            min_confidence=x.min_confidence,
        )
        for x in out
    ]
