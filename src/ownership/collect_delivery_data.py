from __future__ import annotations

import io
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
NSE_SEC_BHAV_URLS = [
    "https://archives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv",
    "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv",
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


def _parse_sec_bhav_csv(text_data: str, file_date: date) -> pd.DataFrame:
    try:
        z = pd.read_csv(io.StringIO(text_data))
    except Exception:
        return pd.DataFrame(columns=["symbol", "date", "delivery_pct", "volume", "price"])
    if z.empty:
        return pd.DataFrame(columns=["symbol", "date", "delivery_pct", "volume", "price"])

    z.columns = [str(c).strip().upper() for c in z.columns]
    for need in ["SYMBOL", "SERIES"]:
        if need not in z.columns:
            return pd.DataFrame(columns=["symbol", "date", "delivery_pct", "volume", "price"])
    z["SYMBOL"] = z["SYMBOL"].astype(str).str.strip().str.upper()
    z["SERIES"] = z["SERIES"].astype(str).str.strip().str.upper()
    z = z[z["SERIES"].eq("EQ")].copy()
    if z.empty:
        return pd.DataFrame(columns=["symbol", "date", "delivery_pct", "volume", "price"])

    # NSE sec_bhavdata_full usually contains DELIV_PER and TTL_TRD_QNTY columns.
    vol_col = "TTL_TRD_QNTY" if "TTL_TRD_QNTY" in z.columns else ("TOTTRDQTY" if "TOTTRDQTY" in z.columns else None)
    deliv_pct_col = "DELIV_PER" if "DELIV_PER" in z.columns else None
    deliv_qty_col = "DELIV_QTY" if "DELIV_QTY" in z.columns else None
    close_col = "CLOSE_PRICE" if "CLOSE_PRICE" in z.columns else ("CLOSE" if "CLOSE" in z.columns else None)

    if vol_col is None:
        return pd.DataFrame(columns=["symbol", "date", "delivery_pct", "volume", "price"])

    z["volume"] = pd.to_numeric(z.get(vol_col), errors="coerce")
    if deliv_pct_col is not None:
        z["delivery_pct"] = pd.to_numeric(z.get(deliv_pct_col), errors="coerce")
    elif deliv_qty_col is not None:
        dq = pd.to_numeric(z.get(deliv_qty_col), errors="coerce")
        z["delivery_pct"] = (dq / z["volume"]) * 100.0
    else:
        return pd.DataFrame(columns=["symbol", "date", "delivery_pct", "volume", "price"])
    z["price"] = pd.to_numeric(z.get(close_col), errors="coerce") if close_col is not None else 0.0
    z["date"] = pd.to_datetime(file_date).date()
    z = z.dropna(subset=["SYMBOL", "delivery_pct", "volume"]).copy()
    if z.empty:
        return pd.DataFrame(columns=["symbol", "date", "delivery_pct", "volume", "price"])
    return z.rename(columns={"SYMBOL": "symbol"})[["symbol", "date", "delivery_pct", "volume", "price"]]


def fetch_latest_delivery_file(
    max_lookback: int = 21,
    prefer_cache: bool = True,
) -> tuple[pd.DataFrame, date, str]:
    session = _build_session()
    today = datetime.utcnow().date()

    for i in range(max(1, int(max_lookback))):
        check_date = today - timedelta(days=i)
        ymd = check_date.strftime("%Y-%m-%d")
        ddmmyyyy = check_date.strftime("%d%m%Y")
        cache_path = RAW_DIR / f"{ymd}.DAT"
        sec_cache_path = RAW_DIR / f"{ymd}_sec_bhav.csv"

        if prefer_cache and cache_path.exists() and cache_path.stat().st_size > 100:
            print(f"Using cached delivery file: {cache_path}")
            txt = cache_path.read_text(encoding="latin1", errors="ignore")
            df = transform_delivery_df(txt, check_date)
            if not df.empty:
                return df, check_date, str(cache_path)
        if prefer_cache and sec_cache_path.exists() and sec_cache_path.stat().st_size > 100:
            print(f"Using cached sec_bhav file: {sec_cache_path}")
            txt = sec_cache_path.read_text(encoding="utf-8", errors="ignore")
            df = _parse_sec_bhav_csv(txt, check_date)
            if not df.empty:
                return df, check_date, str(sec_cache_path)

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
                df = transform_delivery_df(txt, check_date)
                if not df.empty:
                    return df, check_date, url
            except requests.RequestException:
                continue

        # Fallback: NSE daily sec_bhavdata_full csv
        for tmpl in NSE_SEC_BHAV_URLS:
            url = tmpl.format(date=ddmmyyyy)
            print("Trying:", url)
            try:
                r = session.get(url, timeout=20)
                if r.status_code != 200:
                    continue
                txt = r.content.decode("utf-8", errors="ignore")
                if len(txt) < 100:
                    continue
                df = _parse_sec_bhav_csv(txt, check_date)
                if df.empty:
                    continue
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                sec_cache_path.write_text(txt, encoding="utf-8")
                print(f"Found sec_bhav delivery proxy for: {ymd}")
                return df, check_date, url
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
        df_clean, file_date, source = fetch_latest_delivery_file(
            max_lookback=int(max_lookback),
            prefer_cache=bool(prefer_cache),
        )
        print("Using trading date:", file_date)
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
