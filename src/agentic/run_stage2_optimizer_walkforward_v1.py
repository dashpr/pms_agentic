from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from src.agentic.backtest_engine_v1 import BacktestConfig, run_agentic_backtest
    from src.agentic.contracts_v1 import OrchestrationPolicy
    from src.agentic.optimizer_v1 import OptimizerConfig, optimize_policy_for_period
    from src.agentic.walkforward_v1 import WalkForwardConfig, run_walkforward_validation
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.agentic.backtest_engine_v1 import BacktestConfig, run_agentic_backtest
    from src.agentic.contracts_v1 import OrchestrationPolicy
    from src.agentic.optimizer_v1 import OptimizerConfig, optimize_policy_for_period
    from src.agentic.walkforward_v1 import WalkForwardConfig, run_walkforward_validation


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage-2 Agentic optimizer + walk-forward runner")
    p.add_argument("--mode", default="both", choices=["optimize", "walkforward", "both"])
    p.add_argument("--db-path", default="data/ownership.duckdb")
    p.add_argument("--start-date", default="2016-01-01")
    p.add_argument("--end-date", default=None)
    p.add_argument("--rebalance-mode", default="weekly", choices=["weekly", "daily"])
    p.add_argument("--rebalance-weekday", type=int, default=2)
    p.add_argument("--initial-capital", type=float, default=1_000_000.0)
    p.add_argument("--fees-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--universe-limit", type=int, default=0)

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

    p.add_argument("--optimizer-trials", type=int, default=24)
    p.add_argument("--min-trade-win-rate", type=float, default=0.52)
    p.add_argument("--max-drawdown-floor", type=float, default=-0.18)

    p.add_argument("--wf-train-days", type=int, default=756)
    p.add_argument("--wf-test-days", type=int, default=252)
    p.add_argument("--wf-step-days", type=int, default=126)
    p.add_argument("--wf-trials-per-fold", type=int, default=12)
    return p.parse_args(argv)


def _base_policy(args) -> OrchestrationPolicy:
    return OrchestrationPolicy(
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


def _bt_template(args) -> BacktestConfig:
    return BacktestConfig(
        db_path=str(args.db_path),
        start_date=str(args.start_date),
        end_date=str(args.end_date) if args.end_date else None,
        initial_capital=float(args.initial_capital),
        rebalance_mode=str(args.rebalance_mode),
        rebalance_weekday=int(args.rebalance_weekday),
        fees_bps=float(args.fees_bps),
        slippage_bps=float(args.slippage_bps),
        universe_limit=int(args.universe_limit),
    )


def main(argv=None):
    args = parse_args(argv)
    base_policy = _base_policy(args)
    bt_tpl = _bt_template(args)

    if args.mode in {"optimize", "both"}:
        end_date = args.end_date or args.start_date
        opt_cfg = OptimizerConfig(
            db_path=str(args.db_path),
            train_start_date=str(args.start_date),
            train_end_date=str(end_date),
            max_trials=int(args.optimizer_trials),
            min_trade_win_rate=float(args.min_trade_win_rate),
            max_drawdown_floor=float(args.max_drawdown_floor),
            persist_results=True,
        )
        opt = optimize_policy_for_period(
            opt_cfg=opt_cfg,
            bt_template=bt_tpl,
            base_policy=base_policy,
        )
        b = opt["best_row"]
        print("===== STAGE-2 OPTIMIZER =====")
        print("best_trial_id:", int(b["trial_id"]))
        print(f"CAGR: {float(b['cagr']):.2%}")
        print(f"Trade Win: {float(b['trade_win_rate']):.2%}")
        print(f"MaxDD: {float(b['max_drawdown']):.2%}")
        print(f"Sharpe: {float(b['sharpe']):.2f}")
        print(f"Score: {float(b['score']):.4f}")
        print("pass_constraints:", bool(b["pass_constraints"]))
        print("================================")

        # Run one persisted full backtest on selected configuration.
        selected_policy = OrchestrationPolicy(**opt["best_policy"])
        selected_weights = dict(opt["best_agent_weights"])
        bt_final = run_agentic_backtest(
            cfg=bt_tpl,
            policy=selected_policy,
            agent_weight_overrides=selected_weights,
            persist=True,
        )
        st = bt_final["stats"]
        print("===== STAGE-2 SELECTED BACKTEST =====")
        print("run_id:", bt_final["run_id"])
        print(f"CAGR: {float(st['CAGR']):.2%}")
        print(f"Trade Win: {float(st['Trade Win Rate']):.2%}")
        print(f"MaxDD: {float(st['Max Drawdown']):.2%}")
        print(f"Sharpe: {float(st['Sharpe']):.2f}")
        print("======================================")

    if args.mode in {"walkforward", "both"}:
        wf_cfg = WalkForwardConfig(
            db_path=str(args.db_path),
            start_date=str(args.start_date),
            end_date=str(args.end_date) if args.end_date else str(pd.Timestamp.utcnow().date()),
            train_days=int(args.wf_train_days),
            test_days=int(args.wf_test_days),
            step_days=int(args.wf_step_days),
            optimizer_trials_per_fold=int(args.wf_trials_per_fold),
            min_trade_win_rate=float(args.min_trade_win_rate),
            max_drawdown_floor=float(args.max_drawdown_floor),
            persist_results=True,
        )
        wf = run_walkforward_validation(
            wf_cfg=wf_cfg,
            bt_template=bt_tpl,
            base_policy=base_policy,
        )
        s = wf["summary"]
        print("===== STAGE-2 WALK-FORWARD =====")
        print("folds:", int(s["folds"]))
        print(f"median_cagr: {float(s['median_cagr']):.2%}")
        print(f"mean_cagr: {float(s['mean_cagr']):.2%}")
        print(f"cagr_std: {float(s['cagr_std']):.2%}")
        print(f"median_trade_win: {float(s['median_trade_win']):.2%}")
        print(f"worst_max_drawdown: {float(s['worst_max_drawdown']):.2%}")
        print(f"median_sharpe: {float(s['median_sharpe']):.2f}")
        print("=================================")


if __name__ == "__main__":
    main()
