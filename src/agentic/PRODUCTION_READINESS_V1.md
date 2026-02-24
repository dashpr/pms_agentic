# AI Agent Stable Stocks v1: Production Readiness

## Frozen Model Contract
- Model ID: `AI_AGENT_STABLE_STOCKS_V1`
- Source artifact: `data/reports/guardrail_risk_matrix_fullgrid_v7.csv`
- Frozen trial id: `114`
- Frozen policy:
  - watchlist: `25`
  - portfolio: `10..15` (target `12`)
  - buy cutoff: `0.62`
  - sell cutoff: `0.28`
  - max staleness days: `3`
  - min turnover INR: `20,000,000`
- Frozen agent weights:
  - technical `55%`
  - ownership `25%`
  - fundamental `15%`
  - news `5%`

## Runtime Architecture
1. Data layer
   - canonical prices: `prices_daily_v1`
   - canonical delivery: `delivery_daily_v1`
   - optional news/social: `news_social_raw_v1`
2. Decision layer
   - `src/agentic/run_ai_agent_stable_stocks_v1.py --mode decision`
   - persistence tables: `agentic_runs_v1`, `agentic_watchlist_v1`, `agentic_portfolio_targets_v1`
3. Validation layer
   - `src/qa/run_release_checks_ai_agent_stable_v1.py`
4. Monitoring layer
   - `src/monitor_dashboard.py` (Streamlit)

## Mandatory CI-Style Checks
- Core data quality checks pass on production DB path.
- Coverage thresholds:
  - active universe >= 240
  - prices symbols >= 240
  - delivery symbols >= 220
  - history >= 10 years
- Locked decision smoke:
  - watchlist exactly 25
  - portfolio within 10..15
- Short-horizon backtest smoke:
  - emits CAGR, trade win rate, max drawdown, sharpe.

## Known Constraint
- Current frozen full-grid artifact in this repo tops at `22.68%` CAGR (no trial >= `25%` in the 8100-row checkpoint).
- If `25%+` is mandatory, that must be solved by new data/features/regime logic and then re-frozen as a new model version.
