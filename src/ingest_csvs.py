import pandas as pd
from pathlib import Path
from .db import create_schema, upsert_daily

CSV_FOLDER = Path("data/csvs")


def read_yahoo_multiheader(file: Path) -> pd.DataFrame:
    """
    Handle Yahoo Finance multi-row header CSV format.
    """
    df = pd.read_csv(file, header=None)

    # Yahoo structure:
    # row 0 → Price, Close, High, Low, Open, Volume
    # row 1 → Ticker values
    # row 2 → Date label
    # row 3+ → actual data

    # Build clean header
    header = ["date", "close", "high", "low", "open", "volume"]

    # Extract data starting from row 3
    data = df.iloc[3:].copy()
    data.columns = header

    return data


def read_standard_csv(file: Path) -> pd.DataFrame:
    """
    Handle normal OHLCV CSVs.
    """
    df = pd.read_csv(file)

    # normalize column names
    df.columns = [c.lower() for c in df.columns]

    required = {"date", "open", "high", "low", "close", "volume"}

    if not required.issubset(df.columns):
        raise ValueError("Not a standard OHLCV CSV")

    return df[["date", "open", "high", "low", "close", "volume"]]


def read_csv_safely(file: Path) -> pd.DataFrame:
    """
    Try Yahoo parser first, fallback to standard parser.
    """
    try:
        df = read_yahoo_multiheader(file)
    except Exception:
        df = read_standard_csv(file)

    # ensure datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    return df


def ingest_all():
    """
    Main ingestion pipeline.
    """
    create_schema()

    csv_files = list(CSV_FOLDER.glob("*.csv"))

    summary = {
        "total_files": len(csv_files),
        "ingested": 0,
        "failed": 0,
        "errors": [],
    }

    for file in csv_files:
        try:
            df = read_csv_safely(file)

            symbol = file.stem
            upsert_daily(df, symbol)

            summary["ingested"] += 1

        except Exception as e:
            summary["failed"] += 1
            summary["errors"].append({
                "file": file.name,
                "error": str(e)
            })

    return summary


if __name__ == "__main__":
    result = ingest_all()
    print(result)
