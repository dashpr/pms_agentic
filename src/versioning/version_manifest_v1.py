from __future__ import annotations

from pathlib import Path


SYSTEM_RELEASE_TRACK = "AI_AGENT_STABLE_STOCKS_TRACK_V1"
MODEL_VERSION = "2026-02-24"
DASHBOARD_VERSION = "2026-02-24.4"
DATA_PIPELINE_VERSION = "2026-02-24.3"
SCHEMA_VERSION = "duckdb_core_v1"

TRACKED_FILES = [
    "src/agentic/model_lock_ai_agent_stable_stocks_v1.py",
    "src/agentic/orchestrator_v1.py",
    "src/agentic/backtest_engine_v1.py",
    "src/pipeline/run_incremental_data_and_retrain_v1.py",
    "src/qa/pipeline_health_v1.py",
    "src/monitor_dashboard.py",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
