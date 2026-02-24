from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests


DB_PATH = Path("data/ownership.duckdb")
RAW_DIR = Path("data_raw/nse_delivery")
NSE_DELIVERY_URLS = [
    "https://archives.nseindia.com/archives/equities/mto/MTO_{date}.DAT",
    "https://nsearchives.nseindia.com/archives/equities/mto/MTO_{date}.DAT",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


def _get_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH))


def _build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def fetch_latest_delivery_file(
    max_lookback: int = 21,
    prefer_cache: bool = True,
) -> tuple[str, date, str]:
    session = _build_session()
    today = datetime.utcnow().date()

    for i in range(max(1, int(max_lookback))):
        check_date = today - timedelta(days=i)
        ymd = check_date.strftime("%Y-%m-%d")
        ddmmyyyy = check_date.strftime("%d%m%Y")
        cache_path = RAW_DIR / f"{ymd}.DAT"

        if prefer_cache and cache_path.exists() and cache_path.stat().st_size > 100:
            print(f"Using cached delivery file: {cache_path}")
            return cache_path.read_text(encoding="latin1", errors="ignore"), check_date, str(cache_path)

        for tmpl in NSE_DELIVERY_URLS:
            url = tmpl.format(date=ddmmyyyy)
            print("Trying:", url)
            try:
                r = session.get(url, timeout=20)
                if r.status_code != 200:
                    continue
                txt = r.content.decode("latin1", errors="ignore")
                if len(txt) < 100:
                    continue
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(txt, encoding="latin1")
                print(f"Found delivery file for: {ymd}")
                return txt, check_date, url
            except requests.RequestException:
                continue

    raise RuntimeError(f"No NSE delivery file found in last {max_lookback} days.")


def parse_nse_dat(text_data: str) -> pd.DataFrame:
    rows: list[tuple[str, float, float, float]] = []
    for line in text_data.splitlines():
        if not line or not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        # NSE MTO row format:
        # 20,SrNo,Symbol,Series,QtyTraded,DeliverableQty,DeliveryPct
        if len(parts) < 7 or parts[0] != "20":
            continue

        symbol = str(parts[2]).upper().strip()
        if not symbol or symbol == "SYMBOL":
            continue

        try:
            delivery_pct = float(parts[6])
        except Exception:
            continue

        try:
            volume = float(parts[4])
        except Exception:
            volume = 0.0
        try:
            price = 0.0
            # Some future formats may carry value/price fields; keep placeholder for compatibility.
            if len(parts) > 7:
                price = float(parts[7])
        except Exception:
            price = 0.0

        rows.append((symbol, delivery_pct, volume, price))

    if not rows:
        raise RuntimeError("No valid delivery rows parsed from NSE file.")
    return pd.DataFrame(rows, columns=["symbol", "delivery_pct", "volume", "price"])


def transform_delivery_df(text_data: str, file_date: date) -> pd.DataFrame:
    df = parse_nse_dat(text_data)
    df["date"] = pd.to_datetime(file_date).date()
    return df[["symbol", "date", "delivery_pct", "volume", "price"]]


def insert_into_duckdb(df: pd.DataFrame) -> int:
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS delivery_daily (
            symbol TEXT,
            date DATE,
            delivery_pct DOUBLE
        )
        """
    )
    cols = {
        r[0].lower()
        for r in conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='main' AND table_name='delivery_daily'
            """
        ).fetchall()
    }
    if "volume" not in cols:
        conn.execute("ALTER TABLE delivery_daily ADD COLUMN volume DOUBLE")
    if "price" not in cols:
        conn.execute("ALTER TABLE delivery_daily ADD COLUMN price DOUBLE")

    before = int(conn.execute("SELECT COUNT(*) FROM delivery_daily").fetchone()[0] or 0)
    conn.register("temp_df", df)
    conn.execute(
        """
        INSERT INTO delivery_daily
        SELECT t.symbol, CAST(t.date AS DATE), t.delivery_pct, t.volume, t.price
        FROM temp_df t
        WHERE NOT EXISTS (
            SELECT 1
            FROM delivery_daily d
            WHERE UPPER(TRIM(d.symbol)) = UPPER(TRIM(t.symbol))
              AND CAST(d.date AS DATE) = CAST(t.date AS DATE)
        )
        """
    )
    conn.unregister("temp_df")
    after = int(conn.execute("SELECT COUNT(*) FROM delivery_daily").fetchone()[0] or 0)
    conn.close()
    return max(after - before, 0)


def run_delivery_update(
    max_lookback: int = 21,
    prefer_cache: bool = True,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    print("\n===== DELIVERY UPDATE START =====\n")
    out: dict[str, Any] = {"ok": False, "date": None, "rows": 0, "inserted": 0, "source": ""}
    try:
        text_data, file_date, source = fetch_latest_delivery_file(
            max_lookback=int(max_lookback),
            prefer_cache=bool(prefer_cache),
        )
        print("Using trading date:", file_date)
        df_clean = transform_delivery_df(text_data, file_date)
        inserted = insert_into_duckdb(df_clean)
        print("Parsed delivery rows:", len(df_clean))
        print("Inserted new rows:", inserted)
        print("\nDelivery data stored successfully in DuckDB.")
        out.update(
            {
                "ok": True,
                "date": str(file_date),
                "rows": int(len(df_clean)),
                "inserted": int(inserted),
                "source": str(source),
            }
        )
    except Exception as e:
        print("ERROR during delivery update:", e)
        out["error"] = str(e)
        if raise_on_error:
            raise

    print("\n===== DELIVERY UPDATE END =====\n")
    return out


if __name__ == "__main__":
    run_delivery_update()
