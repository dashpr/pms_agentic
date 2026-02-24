# Agentic PMS Architecture v1

## Objective
- Primary: maximize long-run CAGR toward `>=25%`.
- Constraint: keep risk controlled via hard guardrails (staleness, liquidity, concentration, sizing).

## Agent Roles
- `technical`: price momentum/trend + volatility context.
- `ownership`: delivery + bulk/block flow intelligence.
- `fundamental`: enabled when `fundamentals_daily_v1` exists; otherwise neutral.
- `news`: sentiment/event intensity from `news_social_raw_v1`.
- `regime overlay`: market breadth and medium-horizon market trend to scale exposure.

## Institutional Controls
- Agents can "discuss" through weighted consensus, but cannot bypass risk gates.
- Hard risk gates apply before watchlist/portfolio output:
  - data freshness
  - liquidity floor
  - action override to `WAIT` when blocked
- All outputs are persisted as an auditable chain:
  - `agentic_runs_v1`
  - `agentic_agent_signals_v1`
  - `agentic_consensus_v1`
  - `agentic_watchlist_v1`
  - `agentic_portfolio_targets_v1`

## Portfolio Policy (Current)
- Watchlist size: `25`
- Portfolio holdings: `10..15`
- Target holdings (regime-adjusted): `12` baseline

## Operating Mode
- Data pipeline first.
- Agentic decision cycle second.
- Execution is intentionally separate so model research and risk approval can gate deployment.
