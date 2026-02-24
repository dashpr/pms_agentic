import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/csvs")

REQUIRED = ["date", "open", "high", "low", "close", "volume"]


def read_yahoo_style_csv(file: Path):
    """
    Handles Yahoo Finance CSVs with extra header rows.
    """
    df = pd.read_csv(file)

    # Detect Yahoo-style header
    if "Price" in df.columns[0] or "Ticker" in str(df.iloc[0].values):
        df = pd.read_csv(file, skiprows=3)

    return df


def normalize_columns(df: pd.DataFrame):
    """
    Normalizes column names to standard lower-case OHLCV format.
    """
    mapping = {}

    for col in df.columns:
        c = col.lower().strip()

        if "date" in c:
            mapping[col] = "date"
        elif "open" in c:
            mapping[col] = "open"
        elif "high" in c:
            mapping[col] = "high"
        elif "low" in c:
            mapping[col] = "low"
        elif "close" in c:
            mapping[col] = "close"
            if "adj" not in c:
                mapping[col] = "close"
        elif "vol" in c:
            mapping[col] = "volume"

    df = df.rename(columns=mapping)

    if "volume" not in df.columns:
        df["volume"] = 0

    if not set(REQUIRED).issubset(df.columns):
        raise ValueError("Missing OHLC columns")

    return df[REQUIRED]


def repair_all():
    """
    Iterates through all CSVs in DATA_DIR, normalizes them, and saves them back.
    """
    repaired = 0
    failed = 0

    for file in DATA_DIR.glob("*.csv"):
        try:
            df = read_yahoo_style_csv(file)

            df = normalize_columns(df)

            df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
            df = df.dropna(subset=["date"])

            df = df.sort_values("date")

            df.to_csv(file, index=False)

            repaired += 1

        except Exception as e:
            print(f"Failed to repair {file.name}: {e}")
            failed += 1

    print("Fully repaired:", repaired)
    print("Still failed:", failed)


if __name__ == "__main__":
    repair_all()
