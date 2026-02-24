from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests


URLS = [
    "https://archives.nseindia.com/content/equities/BULKDEALS_{date}.csv",
    "https://archives.nseindia.com/content/equities/BULKDEALS{date}.csv",
    "https://nsearchives.nseindia.com/content/equities/BULKDEALS_{date}.csv",
    "https://nsearchives.nseindia.com/content/equities/BULKDEALS{date}.csv",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Download NSE bulk archive CSVs to local folder")
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--out-dir", default="data_raw/nse_bulk")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).date()
    downloaded = 0
    skipped = 0
    for i in range(max(1, int(args.days))):
        d = today - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        d_str = d.strftime("%d%m%Y")
        out_file = out_dir / f"BULKDEALS_{d.isoformat()}.csv"
        if out_file.exists() and out_file.stat().st_size > 50:
            skipped += 1
            continue

        ok = False
        for tmpl in URLS:
            url = tmpl.format(date=d_str)
            try:
                r = requests.get(url, headers=HEADERS, timeout=12)
                if r.status_code == 200 and len(r.text) > 50 and "html" not in r.text[:200].lower():
                    out_file.write_text(r.text, encoding="utf-8")
                    downloaded += 1
                    ok = True
                    break
            except Exception:
                continue
        if not ok:
            continue

    print("===== BULK ARCHIVE LOCAL BACKFILL =====")
    print("out_dir:", out_dir)
    print("downloaded:", downloaded)
    print("skipped_existing:", skipped)
    print("=======================================")
    if downloaded <= 0 and skipped <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
