import duckdb
from pathlib import Path
import pandas as pd

DB_PATH = Path("data/pms.duckdb")


def get_connection():
    """
    Creates DuckDB connection.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def create_schema():
    """
    Creates the main OHLCV table if it does not exist.
    """
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_ohlcv (
            symbol TEXT,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        )
    """)

    conn.close()


def upsert_daily(df: pd.DataFrame, symbol: str):
    """
    Inserts daily OHLCV data into DuckDB.
    """
    conn = get_connection()

    df = df.copy()
    df["symbol"] = symbol

    # ensure correct date format
    df["date"] = pd.to_datetime(df["date"]).dt.date

    conn.register("temp_df", df)

    conn.execute("""
        INSERT INTO daily_ohlcv
        SELECT symbol, date, open, high, low, close, volume
        FROM temp_df
    """)

    conn.unregister("temp_df")
    conn.close()
