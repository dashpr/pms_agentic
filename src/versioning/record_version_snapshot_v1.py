from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd

try:
    from src.versioning.version_manifest_v1 import (
        DASHBOARD_VERSION,
        DATA_PIPELINE_VERSION,
        MODEL_VERSION,
        SCHEMA_VERSION,
        SYSTEM_RELEASE_TRACK,
        TRACKED_FILES,
        project_root,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.versioning.version_manifest_v1 import (
        DASHBOARD_VERSION,
        DATA_PIPELINE_VERSION,
        MODEL_VERSION,
        SCHEMA_VERSION,
        SYSTEM_RELEASE_TRACK,
        TRACKED_FILES,
        project_root,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS system_version_registry_v1 (
            snapshot_id VARCHAR,
            snapshot_ts TIMESTAMP,
            track_name VARCHAR,
            model_version VARCHAR,
            dashboard_version VARCHAR,
            data_pipeline_version VARCHAR,
            schema_version VARCHAR,
            context VARCHAR,
            notes VARCHAR,
            files_json VARCHAR
        )
        """
    )


def record_version_snapshot(
    db_path: str,
    context: str = "manual",
    notes: str = "",
) -> dict:
    root = project_root()
    file_rows = []
    for rel in TRACKED_FILES:
        fp = root / rel
        file_rows.append(
            {
                "path": rel,
                "exists": bool(fp.exists()),
                "sha256": _sha256(fp) if fp.exists() else "",
                "bytes": int(fp.stat().st_size) if fp.exists() else 0,
                "mtime": fp.stat().st_mtime if fp.exists() else 0.0,
            }
        )
    snap = {
        "snapshot_id": f"ver_{uuid4().hex[:12]}",
        "snapshot_ts": _utc_now(),
        "track_name": SYSTEM_RELEASE_TRACK,
        "model_version": MODEL_VERSION,
        "dashboard_version": DASHBOARD_VERSION,
        "data_pipeline_version": DATA_PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "context": context,
        "notes": notes,
        "files_json": json.dumps(file_rows, ensure_ascii=True),
    }

    with duckdb.connect(str(db_path)) as conn:
        _ensure_tables(conn)
        row = pd.DataFrame([snap])
        conn.register("ver_snap", row)
        conn.execute("INSERT INTO system_version_registry_v1 SELECT * FROM ver_snap")
        conn.unregister("ver_snap")

    out = {
        "snapshot_id": snap["snapshot_id"],
        "snapshot_ts": str(snap["snapshot_ts"]),
        "track_name": SYSTEM_RELEASE_TRACK,
        "model_version": MODEL_VERSION,
        "dashboard_version": DASHBOARD_VERSION,
        "data_pipeline_version": DATA_PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "context": context,
        "notes": notes,
        "tracked_files": file_rows,
    }
    out_path = Path("data/reports/version_snapshot_latest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Record version snapshot v1")
    p.add_argument("--db-path", default="data/ownership.duckdb")
    p.add_argument("--context", default="manual")
    p.add_argument("--notes", default="")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = record_version_snapshot(
        db_path=str(args.db_path),
        context=str(args.context),
        notes=str(args.notes),
    )
    print("===== VERSION SNAPSHOT RECORDED =====")
    print("snapshot_id:", out["snapshot_id"])
    print("context:", out["context"])
    print("model_version:", out["model_version"])
    print("dashboard_version:", out["dashboard_version"])
    print("data_pipeline_version:", out["data_pipeline_version"])
    print("schema_version:", out["schema_version"])
    print("=====================================")


if __name__ == "__main__":
    main()
