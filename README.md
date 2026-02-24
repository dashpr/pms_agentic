## PMS Data + Agentic Research Baseline

This repository currently contains:
- clean data ingestion/canonicalization pipeline
- institutional data quality gates
- agentic decision framework (watchlist + portfolio targets)
- Stage-2 research stack (backtest + optimizer + walk-forward validator)
- locked production model runner (`AI Agent Stable Stocks v1`)
- production monitoring dashboard (Streamlit)

Still intentionally excluded:
- direct broker auto-execution (kept gated by design)

## Core Commands

Run data quality gates:

```bash
.venv/bin/python src/qa/run_quality_gates_v1.py
```

Run agentic decision cycle (25 watchlist, 10-15 portfolio):

```bash
.venv/bin/python src/agentic/run_agentic_decision_cycle_v1.py \
  --db-path data/ownership.duckdb \
  --watchlist-size 25 \
  --portfolio-min 10 \
  --portfolio-max 15 \
  --portfolio-target 12
```

Run Stage-2 optimizer + walk-forward:

```bash
.venv/bin/python src/agentic/run_stage2_optimizer_walkforward_v1.py \
  --mode both \
  --db-path data/ownership.duckdb \
  --start-date 2016-01-01 \
  --end-date 2026-02-19 \
  --optimizer-trials 24 \
  --wf-train-days 756 \
  --wf-test-days 252 \
  --wf-step-days 126 \
  --wf-trials-per-fold 12 \
  --watchlist-size 25 \
  --portfolio-min 10 \
  --portfolio-max 15 \
  --portfolio-target 12
```

Run data pipeline (model-free):

```bash
.venv/bin/python src/pipeline/run_ownership_pipeline_v1.py --skip-news-ingestion
```

Run pipeline + agentic cycle in one command:

```bash
.venv/bin/python src/pipeline/run_ownership_pipeline_v1.py \
  --skip-news-ingestion \
  --with-agentic-cycle \
  --agentic-watchlist-size 25 \
  --agentic-portfolio-min 10 \
  --agentic-portfolio-max 15 \
  --agentic-portfolio-target 12
```

Rebuild canonical price/delivery tables:

```bash
.venv/bin/python src/pipeline/run_ownership_pipeline_v1.py --rebuild-canonical --skip-news-ingestion
```

Full data refresh flow:

```bash
.venv/bin/python src/pipeline/run_ownership_pipeline_v1.py \
  --refresh-prices-csv \
  --refresh-existing-prices-csv \
  --build-prices-table \
  --rebuild-canonical
```

## Locked Production Model (AI Agent Stable Stocks v1)

Architecture/readiness document:
- `src/agentic/PRODUCTION_READINESS_V1.md`

Run frozen model decision + backtest and write report JSON:

```bash
.venv/bin/python src/agentic/run_ai_agent_stable_stocks_v1.py \
  --db-path data/ownership.duckdb \
  --mode both \
  --as-of-date 2026-02-19 \
  --bt-start-date 2016-01-01 \
  --bt-rebalance-mode weekly \
  --bt-rebalance-weekday 2 \
  --persist-backtest \
  --out-json data/reports/ai_agent_stable_stocks_v1_latest.json
```

Run institutional release checks (data + locked model smoke):

```bash
.venv/bin/python src/qa/run_release_checks_ai_agent_stable_v1.py \
  --db-path data/ownership.duckdb \
  --require-min-universe 240 \
  --require-min-price-symbols 240 \
  --require-min-delivery-symbols 220 \
  --require-min-history-years 10
```

One-command production cycle (release checks + locked decision):

```bash
.venv/bin/python src/pipeline/run_production_cycle_ai_agent_stable_v1.py \
  --db-path data/ownership.duckdb \
  --as-of-date 2026-02-19
```

Incremental update + scheduled retraining cycle:

```bash
.venv/bin/python src/pipeline/run_incremental_data_and_retrain_v1.py \
  --db-path data/ownership.duckdb \
  --refresh-prices \
  --build-prices-table \
  --rebuild-prices-canonical \
  --run-release-checks \
  --run-locked-decision \
  --persist-backtest \
  --retrain-schedule weekly
```

Notes:
- This model is agentic/rule-driven (not a deep-learning fit per run). "Retraining" here means scheduled re-optimization + walk-forward challenger generation.
- Promote challenger to production only after review.

Pipeline health checks (persist status for dashboard):

```bash
.venv/bin/python src/qa/run_pipeline_health_checks_v1.py \
  --db-path data/ownership.duckdb \
  --persist
```

Daily automation runner (includes pipeline-status report):

```bash
.venv/bin/python src/pipeline/run_incremental_data_and_retrain_v1.py \
  --daily-auto \
  --db-path data/ownership.duckdb
```

Scheduled daily cycle runner (strict-first, degraded fallback, cloud/local compatible):

```bash
.venv/bin/python src/pipeline/run_scheduled_daily_cycle_v1.py \
  --db-path data/ownership.duckdb \
  --max-repair-rounds 5 \
  --price-csv-dir data/csvs \
  --prices-stale-days 1 \
  --bulk-lookback-days 180 \
  --strict-rebalance-pretrade \
  --allow-degraded-fallback
```

Record version snapshot (model/dashboard/pipeline/schema + file hashes):

```bash
.venv/bin/python src/versioning/record_version_snapshot_v1.py \
  --db-path data/ownership.duckdb \
  --context manual
```

## Dashboard

Start dashboard:

```bash
.venv/bin/streamlit run src/monitor_dashboard.py --server.port 8502
```

Cloud entrypoint (Streamlit Community Cloud):

```bash
streamlit_app.py
```

Dashboard features:
- live portfolio targets + watchlist + explainability
- filtered backtest analytics (year/month filters recalculate CAGR, win rate, drawdown, Sharpe)
- past watchlist outcomes with entry/exit and realized PnL
- stock advisor with symbol search, BUY/HOLD/WAIT/EXIT, score percentile, upside%, and horizon
- operations console with frozen reference metrics and run commands

## Cloud Automation (Laptop Off)

This repo now includes:

- workflow: `.github/workflows/daily_ai_agent_stable_v1.yml`
- scheduler script: `src/pipeline/run_scheduled_daily_cycle_v1.py`

What it does daily (Mon-Fri):

1. repairs required pipelines (`repair_required_pipelines_v1.py`)
2. persists pipeline health checks
3. runs strict decision cycle first
4. falls back to degraded decision mode if strict fails (still emits output)
5. uploads reports + DB snapshot artifacts
6. publishes latest runtime snapshot to branch `runtime-data` (for cloud dashboard consumption)

Important bootstrap note:

- first cloud run requires `data/ownership.duckdb` to exist in the workflow environment.
- easiest path: run workflow once after pushing a seeded DB snapshot (or use a prior workflow artifact/cache).
- optional: set GitHub Actions secret `OWNERSHIP_DB_URL` to a downloadable DB snapshot URL; workflow will auto-bootstrap from it if cache/artifact is missing.

## Streamlit Cloud Deployment (No Laptop Dependency)

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and connect GitHub.
2. Select repo: `dashpr/pms_agentic`.
3. Branch: `main`.
4. Main file path: `streamlit_app.py`.
5. In Streamlit app settings -> Secrets, set:
   - `OWNERSHIP_DB_URL = "https://raw.githubusercontent.com/dashpr/pms_agentic/runtime-data/data/ownership_runtime.duckdb"`
   - `OWNERSHIP_DB_REFRESH_MINUTES = 60`
6. Deploy.

After deployment, the URL is globally accessible (mobile/desktop). The app auto-downloads the runtime DuckDB snapshot and refreshes periodically.

## DuckDB Tables

Core data tables:
- `symbol_master`
- `prices_daily`, `prices_daily_v1`
- `delivery_daily`, `delivery_daily_v1`
- `shareholding_quarterly`
- `bulk_block_deals`
- `fii_dii_flows`
- `fo_oi_stats`
- `news_social_raw_v1`
- `tickertape_flows_raw_v1`

Agentic orchestration tables:
- `agentic_runs_v1`
- `agentic_agent_signals_v1`
- `agentic_consensus_v1`
- `agentic_watchlist_v1`
- `agentic_portfolio_targets_v1`

Stage-2 research tables:
- `agentic_backtest_runs_v1`
- `agentic_backtest_equity_v1`
- `agentic_backtest_trades_v1`
- `agentic_optimizer_results_v1`
- `agentic_optimizer_best_v1`
- `agentic_walkforward_folds_v1`
- `agentic_walkforward_summary_v1`
