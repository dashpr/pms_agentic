"""
Ingest real external news/social events into canonical news_social_raw_v1.

Providers supported:
- NewsAPI (env: NEWSAPI_KEY)
- Finnhub company news (env: FINNHUB_API_KEY)

If keys are missing, ingestion exits successfully with zero rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import duckdb
import pandas as pd
import requests

DB_PATH = Path("data/ownership.duckdb")

POS_WORDS = {
    "beat", "beats", "strong", "upgrade", "upgrades", "growth", "surge", "record", "profit", "bullish",
    "outperform", "buy", "acquire", "expands", "expansion", "wins", "approval", "rebound", "rally",
}
NEG_WORDS = {
    "miss", "misses", "weak", "downgrade", "downgrades", "decline", "falls", "loss", "bearish", "sell",
    "probe", "fraud", "penalty", "lawsuit", "default", "cut", "cuts", "slump", "crash", "concern",
}


@dataclass
class IngestConfig:
    as_of_date: date
    lookback_days: int = 3
    max_symbols: int = 80
    request_timeout_sec: int = 20


@dataclass
class ProviderStats:
    provider: str
    queried_symbols: int = 0
    http_non_200: int = 0
    exceptions: int = 0
    rows_emitted: int = 0
    sample_error: str = ""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_text(x: object) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _score_sentiment(text: str) -> float:
    t = text.lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    raw = (pos - neg) / max(pos + neg, 1)
    # map [-1,1] -> [0,1]
    return max(0.0, min(1.0, 0.5 + 0.5 * raw))


def _event_id(provider: str, canonical_symbol: str, url: str, title: str, event_ts: str) -> str:
    key = "|".join([provider, canonical_symbol, url, title, event_ts])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _ensure_raw_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_social_raw_v1 (
            event_id VARCHAR,
            event_ts TIMESTAMP,
            date DATE,
            canonical_symbol VARCHAR,
            ticker VARCHAR,
            provider VARCHAR,
            source VARCHAR,
            title VARCHAR,
            summary VARCHAR,
            url VARCHAR,
            sentiment_score DOUBLE,
            relevance_score DOUBLE,
            novelty_score DOUBLE,
            buzz_score DOUBLE,
            source_confidence DOUBLE,
            payload_json VARCHAR,
            ingested_at TIMESTAMP
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_news_social_raw_v1_event_id ON news_social_raw_v1(event_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_news_social_raw_v1_symbol_date ON news_social_raw_v1(canonical_symbol, date)"
    )


def _load_symbol_map(conn: duckdb.DuckDBPyConnection, max_symbols: int) -> pd.DataFrame:
    q = """
    SELECT
        canonical_symbol,
        COALESCE(NULLIF(TRIM(nse_symbol), ''), canonical_symbol) AS nse_symbol,
        COALESCE(NULLIF(TRIM(sector), ''), 'UNKNOWN') AS sector,
        COALESCE(in_universe, TRUE) AS in_universe
    FROM symbol_master
    ORDER BY canonical_symbol
    """
    sm = conn.execute(q).df()
    if sm.empty:
        return sm

    sm["canonical_symbol"] = sm["canonical_symbol"].astype(str).str.strip().str.upper()
    sm["nse_symbol"] = sm["nse_symbol"].astype(str).str.strip().str.upper()
    sm = sm[sm["in_universe"] == True].copy()
    if max_symbols > 0:
        sm = sm.head(max_symbols)
    sm["ticker"] = sm["nse_symbol"].str.replace(r"\.NS$", "", regex=True) + ".NS"
    return sm[["canonical_symbol", "ticker", "sector"]]


def _fetch_newsapi(symbol_df: pd.DataFrame, cfg: IngestConfig) -> tuple[List[Dict[str, object]], ProviderStats]:
    api_key = os.getenv("NEWSAPI_KEY", "").strip()
    stats = ProviderStats(provider="newsapi")
    if not api_key:
        stats.sample_error = "missing NEWSAPI_KEY"
        return [], stats

    out: List[Dict[str, object]] = []
    fr = (cfg.as_of_date - timedelta(days=cfg.lookback_days)).isoformat()
    to = cfg.as_of_date.isoformat()

    base_url = "https://newsapi.org/v2/everything"
    headers = {"X-Api-Key": api_key}

    for _, row in symbol_df.iterrows():
        stats.queried_symbols += 1
        canonical = row["canonical_symbol"]
        ticker = row["ticker"]
        q = f'"{ticker}" OR "{canonical}"'
        params = {
            "q": q,
            "from": fr,
            "to": to,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50,
        }
        try:
            r = requests.get(base_url, params=params, headers=headers, timeout=cfg.request_timeout_sec)
            if r.status_code != 200:
                stats.http_non_200 += 1
                if not stats.sample_error:
                    stats.sample_error = f"http_{r.status_code}"
                continue
            payload = r.json()
            for a in payload.get("articles", []) or []:
                title = _safe_text(a.get("title"))
                desc = _safe_text(a.get("description"))
                source = _safe_text((a.get("source") or {}).get("name") or "newsapi")
                url = _safe_text(a.get("url"))
                ts = _safe_text(a.get("publishedAt"))
                text = f"{title} {desc}".strip()
                out.append(
                    {
                        "provider": "newsapi",
                        "canonical_symbol": canonical,
                        "ticker": ticker,
                        "event_ts": ts,
                        "source": source,
                        "title": title,
                        "summary": desc,
                        "url": url,
                        "sentiment_score": _score_sentiment(text),
                        "relevance_score": 0.80,
                        "novelty_score": 0.70,
                        "buzz_score": 0.60,
                        "source_confidence": 0.75,
                        "payload_json": json.dumps(a, ensure_ascii=True),
                    }
                )
        except Exception as exc:
            stats.exceptions += 1
            if not stats.sample_error:
                stats.sample_error = str(exc)[:200]
            continue
    stats.rows_emitted = len(out)
    return out, stats


def _fetch_finnhub(symbol_df: pd.DataFrame, cfg: IngestConfig) -> tuple[List[Dict[str, object]], ProviderStats]:
    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    stats = ProviderStats(provider="finnhub")
    if not api_key:
        stats.sample_error = "missing FINNHUB_API_KEY"
        return [], stats

    out: List[Dict[str, object]] = []
    fr = (cfg.as_of_date - timedelta(days=cfg.lookback_days)).isoformat()
    to = cfg.as_of_date.isoformat()

    base_url = "https://finnhub.io/api/v1/company-news"

    for _, row in symbol_df.iterrows():
        stats.queried_symbols += 1
        canonical = row["canonical_symbol"]
        ticker = row["ticker"]
        params = {
            "symbol": ticker,
            "from": fr,
            "to": to,
            "token": api_key,
        }
        try:
            r = requests.get(base_url, params=params, timeout=cfg.request_timeout_sec)
            if r.status_code != 200:
                stats.http_non_200 += 1
                if not stats.sample_error:
                    stats.sample_error = f"http_{r.status_code}"
                continue
            arr = r.json()
            if not isinstance(arr, list):
                continue
            for a in arr:
                title = _safe_text(a.get("headline"))
                desc = _safe_text(a.get("summary"))
                source = _safe_text(a.get("source") or "finnhub")
                url = _safe_text(a.get("url"))
                ts_unix = a.get("datetime")
                try:
                    ts = datetime.fromtimestamp(int(ts_unix), tz=timezone.utc).isoformat()
                except Exception:
                    ts = ""
                text = f"{title} {desc}".strip()
                out.append(
                    {
                        "provider": "finnhub",
                        "canonical_symbol": canonical,
                        "ticker": ticker,
                        "event_ts": ts,
                        "source": source,
                        "title": title,
                        "summary": desc,
                        "url": url,
                        "sentiment_score": _score_sentiment(text),
                        "relevance_score": 0.75,
                        "novelty_score": 0.65,
                        "buzz_score": 0.55,
                        "source_confidence": 0.70,
                        "payload_json": json.dumps(a, ensure_ascii=True),
                    }
                )
        except Exception as exc:
            stats.exceptions += 1
            if not stats.sample_error:
                stats.sample_error = str(exc)[:200]
            continue
    stats.rows_emitted = len(out)
    return out, stats


def _prepare_rows(rows: Iterable[Dict[str, object]], as_of_date: date) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df

    df["canonical_symbol"] = df["canonical_symbol"].astype(str).str.strip().str.upper()
    df["event_ts"] = pd.to_datetime(df["event_ts"], errors="coerce", utc=True)
    df["event_ts"] = df["event_ts"].fillna(pd.Timestamp(as_of_date, tz="UTC"))
    df["date"] = df["event_ts"].dt.date

    for c in ["sentiment_score", "relevance_score", "novelty_score", "buzz_score", "source_confidence"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.5).clip(0.0, 1.0)

    df["title"] = df["title"].astype(str).str.slice(0, 1200)
    df["summary"] = df["summary"].astype(str).str.slice(0, 3000)
    df["url"] = df["url"].astype(str).str.slice(0, 2000)
    df["source"] = df["source"].astype(str).str.slice(0, 200)
    df["provider"] = df["provider"].astype(str).str.slice(0, 120)

    df["event_id"] = [
        _event_id(p, s, u, t, str(ts))
        for p, s, u, t, ts in zip(
            df["provider"], df["canonical_symbol"], df["url"], df["title"], df["event_ts"]
        )
    ]
    df["ingested_at"] = _utcnow()

    keep = [
        "event_id",
        "event_ts",
        "date",
        "canonical_symbol",
        "ticker",
        "provider",
        "source",
        "title",
        "summary",
        "url",
        "sentiment_score",
        "relevance_score",
        "novelty_score",
        "buzz_score",
        "source_confidence",
        "payload_json",
        "ingested_at",
    ]
    return df[keep].drop_duplicates(subset=["event_id"])


def ingest_news_social_raw(cfg: IngestConfig) -> int:
    conn = duckdb.connect(DB_PATH)
    try:
        _ensure_raw_table(conn)
        symbols = _load_symbol_map(conn, cfg.max_symbols)
        if symbols.empty:
            print("No symbols available in symbol_master; skipping news ingestion.")
            return 0

        rows: List[Dict[str, object]] = []
        rows_newsapi, stats_newsapi = _fetch_newsapi(symbols, cfg)
        rows_finnhub, stats_finnhub = _fetch_finnhub(symbols, cfg)
        rows.extend(rows_newsapi)
        rows.extend(rows_finnhub)
        print(
            "provider_stats",
            {
                "newsapi": stats_newsapi.__dict__,
                "finnhub": stats_finnhub.__dict__,
            },
        )

        prepared = _prepare_rows(rows, cfg.as_of_date)
        if prepared.empty:
            print("No news events fetched (missing keys or empty provider responses).")
            return 0

        conn.register("incoming_news", prepared)
        conn.execute(
            """
            INSERT INTO news_social_raw_v1
            SELECT i.*
            FROM incoming_news i
            LEFT JOIN news_social_raw_v1 t
              ON i.event_id = t.event_id
            WHERE t.event_id IS NULL
            """
        )
        inserted = conn.execute("SELECT COUNT(*) FROM incoming_news").fetchone()[0]
        conn.unregister("incoming_news")

        print(f"Fetched events: {inserted}")
        return int(inserted)
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest external news/social into news_social_raw_v1")
    p.add_argument("--as-of-date", default=None, help="YYYY-MM-DD")
    p.add_argument("--lookback-days", type=int, default=3)
    p.add_argument("--max-symbols", type=int, default=80)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    as_of = pd.to_datetime(args.as_of_date).date() if args.as_of_date else _utcnow().date()
    cfg = IngestConfig(
        as_of_date=as_of,
        lookback_days=int(args.lookback_days),
        max_symbols=int(args.max_symbols),
    )
    inserted = ingest_news_social_raw(cfg)
    print(f"news_social_raw_v1 upsert completed. inserted={inserted}")


if __name__ == "__main__":
    main()
