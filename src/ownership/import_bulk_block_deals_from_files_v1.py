from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd

try:
    from src.ownership.collect_bulk_deals_nse import _insert_rows, _normalize_bulk_df
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.ownership.collect_bulk_deals_nse import _insert_rows, _normalize_bulk_df


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Import bulk/block deals from local CSV/HTML files")
    p.add_argument("--input-dir", default="data_raw/nse_bulk")
    p.add_argument("--recursive", action="store_true")
    return p.parse_args(argv)


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".csv", ".txt"}:
        return pd.read_csv(path, on_bad_lines="skip")
    if suf in {".htm", ".html"}:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        tables = pd.read_html(StringIO(txt))
        if not tables:
            return pd.DataFrame()
        return tables[0]
    return pd.DataFrame()


def main(argv=None):
    args = parse_args(argv)
    root = Path(str(args.input_dir))
    if not root.exists():
        print(f"input_dir_not_found: {root}")
        raise SystemExit(1)

    patt = "**/*" if bool(args.recursive) else "*"
    files = sorted([p for p in root.glob(patt) if p.is_file()])
    if not files:
        print(f"no_input_files: {root}")
        raise SystemExit(1)

    total_inserted = 0
    used = 0
    for f in files:
        try:
            raw = _read_any(f)
            if raw.empty:
                continue
            z = _normalize_bulk_df(raw)
            ins = _insert_rows(z)
            if ins > 0:
                used += 1
                total_inserted += int(ins)
                print(f"imported: {f} | rows={ins}")
        except Exception as e:
            print(f"skip: {f} | error={e}")

    print("===== LOCAL BULK IMPORT =====")
    print("files_seen:", len(files))
    print("files_used:", used)
    print("rows_inserted:", total_inserted)
    print("============================")
    if total_inserted <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
