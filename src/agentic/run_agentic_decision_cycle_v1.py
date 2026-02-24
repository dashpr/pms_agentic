from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

try:
    from src.agentic.contracts_v1 import OrchestrationPolicy
    from src.agentic.orchestrator_v1 import run_agentic_cycle
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.agentic.contracts_v1 import OrchestrationPolicy
    from src.agentic.orchestrator_v1 import run_agentic_cycle


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run Agentic PMS decision cycle v1")
    p.add_argument("--db-path", default="data/ownership.duckdb")
    p.add_argument("--as-of-date", default=date.today().isoformat(), help="YYYY-MM-DD")
    p.add_argument("--watchlist-size", type=int, default=25)
    p.add_argument("--portfolio-min", type=int, default=10)
    p.add_argument("--portfolio-max", type=int, default=15)
    p.add_argument("--portfolio-target", type=int, default=12)
    p.add_argument("--buy-threshold", type=float, default=0.66)
    p.add_argument("--sell-threshold", type=float, default=0.34)
    p.add_argument("--max-stale-days", type=int, default=5)
    p.add_argument("--min-turnover-inr", type=float, default=5.0e7)
    p.add_argument("--max-single-weight", type=float, default=0.10)
    p.add_argument("--min-single-weight", type=float, default=0.03)
    p.add_argument("--universe-limit", type=int, default=0, help="Optional cap for quick dry runs.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    as_of = date.fromisoformat(str(args.as_of_date))
    policy = OrchestrationPolicy(
        watchlist_size=int(args.watchlist_size),
        portfolio_min_positions=int(args.portfolio_min),
        portfolio_max_positions=int(args.portfolio_max),
        portfolio_target_positions=int(args.portfolio_target),
        consensus_buy_threshold=float(args.buy_threshold),
        consensus_sell_threshold=float(args.sell_threshold),
        max_price_staleness_days=int(args.max_stale_days),
        min_median_turnover_inr=float(args.min_turnover_inr),
        max_single_weight=float(args.max_single_weight),
        min_single_weight=float(args.min_single_weight),
    )
    out = run_agentic_cycle(
        db_path=args.db_path,
        as_of_date=as_of,
        policy=policy,
        universe_limit=int(args.universe_limit),
    )
    print("===== AGENTIC PMS DECISION CYCLE v1 =====")
    for k, v in out.items():
        print(f"{k}: {v}")
    print("==========================================")


if __name__ == "__main__":
    main()
