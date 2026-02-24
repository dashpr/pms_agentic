from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests


DB_PATH = Path("data/ownership.duckdb")
NSE_HOME = "https://www.nseindia.com"
NSE_BULK_API = "https://www.nseindia.com/api/historical/bulk-deals"
NSE_BULK_HTML = "https://www.nseindia.com/report-detail/display-bulk-and-block-deals"
NSE_BULK_ARCHIVE = "https://archives.nseindia.com/content/equities/BULKDEALS_{date}.csv"
NSE_BULK_ARCHIVE_ALT = "https://archives.nseindia.com/content/equities/BULKDEALS{date}.csv"
NSE_BULK_ARCHIVE_NEW = "https://nsearchives.nseindia.com/content/equities/BULKDEALS_{date}.csv"
NSE_BULK_ARCHIVE_NEW_ALT = "https://nsearchives.nseindia.com/content/equities/BULKDEALS{date}.csv"
SCREENER_BULK_URL = "https://www.screener.in/screens/343087/bulk-deals/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


def _get_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH))


def _ensure_table() -> None:
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bulk_block_deals (
            symbol TEXT,
            date DATE,
            side TEXT,
            qty DOUBLE,
            price DOUBLE,
            participant TEXT
        )
        """
    )
    conn.close()


def _build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get(NSE_HOME, timeout=20)
    return s


def _normalize_bulk_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["symbol", "date", "side", "qty", "price", "participant"])

    z = df.copy()
    z.columns = [str(c).strip().lower() for c in z.columns]
    rename_map = {
        "symbol": "symbol",
        "security name": "security_name",
        "name of security": "security_name",
        "security": "security_name",
        "stock": "security_name",
        "company": "security_name",
        "buysell": "side",
        "buy/sell": "side",
        "buy sell": "side",
        "quantitytraded": "qty",
        "quantity traded": "qty",
        "qty traded": "qty",
        "no of shares": "qty",
        "no. of shares": "qty",
        "price": "price",
        "trade price / wght. avg. price": "price",
        "trade price": "price",
        "wght. avg. price": "price",
        "clientname": "participant",
        "client name": "participant",
        "date": "date",
        "trade date": "date",
        "date of transaction": "date",
    }
    z = z.rename(columns=rename_map)
    # Deduplicate any accidental duplicate column names from heterogeneous source headers.
    if z.columns.duplicated().any():
        z = z.loc[:, ~z.columns.duplicated(keep="first")].copy()

    if "date" not in z.columns and "__archive_date" in z.columns:
        z["date"] = z["__archive_date"]

    for c in ["symbol", "date", "qty", "price"]:
        if c not in z.columns:
            raise ValueError(f"Missing expected column: {c}")
    if "side" not in z.columns:
        z["side"] = ""
    if "participant" not in z.columns:
        z["participant"] = ""

    z["symbol"] = z["symbol"].astype(str).str.strip().str.upper()
    z["side"] = z["side"].astype(str).str.strip().str.upper()
    z["qty"] = pd.to_numeric(z["qty"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    z["price"] = pd.to_numeric(z["price"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    z["date"] = pd.to_datetime(z["date"], errors="coerce").dt.date
    z = z.dropna(subset=["symbol", "date", "qty", "price"]).copy()
    return z[["symbol", "date", "side", "qty", "price", "participant"]]


def _fetch_bulk_api(session: requests.Session, start: datetime, end: datetime) -> pd.DataFrame:
    params = {"from": start.strftime("%d-%m-%Y"), "to": end.strftime("%d-%m-%Y")}
    r = session.get(NSE_BULK_API, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict):
        rows = payload.get("data") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    return pd.DataFrame(rows)


def _fetch_bulk_html(session: requests.Session) -> pd.DataFrame:
    r = session.get(NSE_BULK_HTML, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    if not tables:
        return pd.DataFrame()
    return tables[0]


def _fetch_bulk_archives(lookback_days: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    today = datetime.now(timezone.utc).date()
    for i in range(max(1, int(lookback_days))):
        d = today - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        d_str = d.strftime("%d%m%Y")
        for tmpl in [NSE_BULK_ARCHIVE, NSE_BULK_ARCHIVE_ALT, NSE_BULK_ARCHIVE_NEW, NSE_BULK_ARCHIVE_NEW_ALT]:
            url = tmpl.format(date=d_str)
            try:
                r = requests.get(url, headers=HEADERS, timeout=12)
                if r.status_code != 200 or len(r.text) < 50:
                    continue
                part = pd.read_csv(StringIO(r.text), on_bad_lines="skip")
                part["__archive_date"] = d.isoformat()
                rows.append(part)
                break
            except Exception:
                continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _fetch_bulk_screener() -> pd.DataFrame:
    r = requests.get(SCREENER_BULK_URL, headers=HEADERS, timeout=15)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    if not tables:
        return pd.DataFrame()
    return tables[0]


def _insert_rows(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn = _get_conn()
    before = int(conn.execute("SELECT COUNT(*) FROM bulk_block_deals").fetchone()[0] or 0)
    conn.register("temp_bulk", df)
    conn.execute(
        """
        INSERT INTO bulk_block_deals
        SELECT
            UPPER(TRIM(symbol)) AS symbol,
            CAST(date AS DATE) AS date,
            UPPER(TRIM(COALESCE(side, ''))) AS side,
            CAST(qty AS DOUBLE) AS qty,
            CAST(price AS DOUBLE) AS price,
            COALESCE(participant, '') AS participant
        FROM temp_bulk
        """
    )
    conn.unregister("temp_bulk")
    after = int(conn.execute("SELECT COUNT(*) FROM bulk_block_deals").fetchone()[0] or 0)
    conn.close()
    return max(after - before, 0)


def run_nse_bulk_ingestion(
    lookback_days: int = 120,
    batch_days: int = 30,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    print("\n====== NSE BULK DEAL INGESTION ======\n")
    out: dict[str, Any] = {"ok": False, "rows": 0, "mode": "", "errors": []}
    _ensure_table()

    try:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=max(int(lookback_days), 1))
        total_inserted = 0
        mode = "api"
        session = None
        try:
            session = _build_session()
        except Exception as e:
            out["errors"].append(f"session_init: {e}")

        if session is not None:
            cur = start_dt
            while cur <= end_dt:
                batch_end = min(cur + timedelta(days=max(int(batch_days), 1) - 1), end_dt)
                try:
                    raw = _fetch_bulk_api(session, cur, batch_end)
                    clean = _normalize_bulk_df(raw)
                    total_inserted += _insert_rows(clean)
                except Exception as e:
                    out["errors"].append(f"api_batch_{cur.date()}_{batch_end.date()}: {e}")
                cur = batch_end + timedelta(days=1)

            if total_inserted <= 0:
                try:
                    mode = "html_fallback"
                    raw_html = _fetch_bulk_html(session)
                    clean_html = _normalize_bulk_df(raw_html)
                    total_inserted += _insert_rows(clean_html)
                except Exception as e:
                    out["errors"].append(f"html_fallback: {e}")

        if total_inserted <= 0:
            mode = "archive_fallback"
            raw_arch = _fetch_bulk_archives(lookback_days=max(20, int(lookback_days)))
            clean_arch = _normalize_bulk_df(raw_arch) if not raw_arch.empty else raw_arch
            total_inserted += _insert_rows(clean_arch) if isinstance(clean_arch, pd.DataFrame) else 0

        if total_inserted <= 0:
            try:
                mode = "screener_fallback"
                raw_scr = _fetch_bulk_screener()
                clean_scr = _normalize_bulk_df(raw_scr) if not raw_scr.empty else raw_scr
                total_inserted += _insert_rows(clean_scr) if isinstance(clean_scr, pd.DataFrame) else 0
            except Exception as e:
                out["errors"].append(f"screener_fallback: {e}")

        if total_inserted <= 0:
            raise RuntimeError("No bulk/block rows ingested from API, HTML, archive, or screener fallback.")

        out.update({"ok": True, "rows": int(total_inserted), "mode": mode})
        print("Inserted rows:", total_inserted)
        print("Mode:", mode)
        print("\nNSE bulk-deal data stored in DuckDB.")
    except Exception as e:
        out["errors"].append(str(e))
        print("ERROR:", e)
        if raise_on_error:
            raise

    print("\n=====================================\n")
    return out


if __name__ == "__main__":
    run_nse_bulk_ingestion()
