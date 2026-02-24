from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

try:
    from src.agentic.backtest_engine_v1 import BacktestConfig
    from src.agentic.contracts_v1 import OrchestrationPolicy
    from src.agentic.optimizer_v1 import OptimizerConfig, evaluate_trial_grid
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.agentic.backtest_engine_v1 import BacktestConfig
    from src.agentic.contracts_v1 import OrchestrationPolicy
    from src.agentic.optimizer_v1 import OptimizerConfig, evaluate_trial_grid


RISK_ORDER = ["high", "moderate", "low"]


@dataclass(frozen=True)
class GuardrailThresholds:
    min_trade_win_rate: float
    max_drawdown_floor: float
    min_sharpe: float


@dataclass(frozen=True)
class GuardrailCombo:
    combo_name: str
    win_tier: str | None
    dd_tier: str | None
    sharpe_tier: str | None
    min_trade_win_rate: float | None
    max_drawdown_floor: float | None
    min_sharpe: float | None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Find max raw CAGR from one trial grid, then apply risk-tiered guardrails "
            "(high/moderate/low) and measure CAGR retention."
        )
    )
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

    p.add_argument("--optimizer-trials", type=int, default=240)
    p.add_argument(
        "--full-grid",
        action="store_true",
        help="Evaluate entire policy/weight grid (ignores --optimizer-trials cap).",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print optimizer progress every N trials (0 disables progress logs).",
    )
    p.add_argument(
        "--checkpoint-csv",
        default="",
        help="Optional checkpoint CSV for per-trial persistence.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Flush trial rows to checkpoint every N completed trials (0 disables).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint CSV (skip completed trial_ids).",
    )

    # Risk-tier thresholds (each guardrail has high/moderate/low categories)
    p.add_argument("--win-high", type=float, default=0.52)
    p.add_argument("--win-moderate", type=float, default=0.56)
    p.add_argument("--win-low", type=float, default=0.60)

    p.add_argument("--dd-high", type=float, default=-0.25)
    p.add_argument("--dd-moderate", type=float, default=-0.18)
    p.add_argument("--dd-low", type=float, default=-0.14)

    p.add_argument("--sharpe-high", type=float, default=0.85)
    p.add_argument("--sharpe-moderate", type=float, default=1.00)
    p.add_argument("--sharpe-low", type=float, default=1.15)

    p.add_argument(
        "--risk-tiers",
        default="high,moderate,low",
        help="Comma separated subset/order of risk tiers to evaluate, e.g. high,moderate",
    )
    p.add_argument(
        "--combo-mode",
        choices=["ladder", "all"],
        default="all",
        help=(
            "ladder: run tier sequence as WIN -> WIN+DD -> WIN+DD+SH by tier; "
            "all: run all cross-combinations of (win tier x dd tier x sharpe tier)."
        ),
    )
    p.add_argument("--persist", action="store_true")
    p.add_argument(
        "--out-csv",
        default="",
        help="Optional CSV output path. Defaults to data/reports/guardrail_risk_matrix_v1.csv",
    )
    return p.parse_args(argv)


def _parse_risk_tiers(raw: str) -> list[str]:
    tiers = [x.strip().lower() for x in str(raw).split(",") if x.strip()]
    if not tiers:
        raise ValueError("--risk-tiers cannot be empty")
    invalid = [x for x in tiers if x not in set(RISK_ORDER)]
    if invalid:
        raise ValueError(f"Invalid tier(s): {invalid}. Allowed: {RISK_ORDER}")
    dedup: list[str] = []
    for t in tiers:
        if t not in dedup:
            dedup.append(t)
    return dedup


def _risk_thresholds(args) -> dict[str, GuardrailThresholds]:
    return {
        "high": GuardrailThresholds(
            min_trade_win_rate=float(args.win_high),
            max_drawdown_floor=float(args.dd_high),
            min_sharpe=float(args.sharpe_high),
        ),
        "moderate": GuardrailThresholds(
            min_trade_win_rate=float(args.win_moderate),
            max_drawdown_floor=float(args.dd_moderate),
            min_sharpe=float(args.sharpe_moderate),
        ),
        "low": GuardrailThresholds(
            min_trade_win_rate=float(args.win_low),
            max_drawdown_floor=float(args.dd_low),
            min_sharpe=float(args.sharpe_low),
        ),
    }


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


def _build_combos(
    combo_mode: str,
    tiers: list[str],
    thresholds: dict[str, GuardrailThresholds],
) -> list[GuardrailCombo]:
    out: list[GuardrailCombo] = [
        GuardrailCombo(
            combo_name="RAW_CAGR_MAX",
            win_tier=None,
            dd_tier=None,
            sharpe_tier=None,
            min_trade_win_rate=None,
            max_drawdown_floor=None,
            min_sharpe=None,
        )
    ]

    if combo_mode == "ladder":
        for tier in tiers:
            t = thresholds[tier]
            out.extend(
                [
                    GuardrailCombo(
                        combo_name=f"{tier.upper()}_WIN",
                        win_tier=tier,
                        dd_tier=None,
                        sharpe_tier=None,
                        min_trade_win_rate=float(t.min_trade_win_rate),
                        max_drawdown_floor=None,
                        min_sharpe=None,
                    ),
                    GuardrailCombo(
                        combo_name=f"{tier.upper()}_WIN_DD",
                        win_tier=tier,
                        dd_tier=tier,
                        sharpe_tier=None,
                        min_trade_win_rate=float(t.min_trade_win_rate),
                        max_drawdown_floor=float(t.max_drawdown_floor),
                        min_sharpe=None,
                    ),
                    GuardrailCombo(
                        combo_name=f"{tier.upper()}_WIN_DD_SH",
                        win_tier=tier,
                        dd_tier=tier,
                        sharpe_tier=tier,
                        min_trade_win_rate=float(t.min_trade_win_rate),
                        max_drawdown_floor=float(t.max_drawdown_floor),
                        min_sharpe=float(t.min_sharpe),
                    ),
                ]
            )
        return out

    for wt in tiers:
        for dt in tiers:
            for st in tiers:
                tw = thresholds[wt]
                dd = thresholds[dt]
                sh = thresholds[st]
                out.append(
                    GuardrailCombo(
                        combo_name=f"WIN_{wt.upper()}__DD_{dt.upper()}__SH_{st.upper()}",
                        win_tier=wt,
                        dd_tier=dt,
                        sharpe_tier=st,
                        min_trade_win_rate=float(tw.min_trade_win_rate),
                        max_drawdown_floor=float(dd.max_drawdown_floor),
                        min_sharpe=float(sh.min_sharpe),
                    )
                )
    return out


def _pick_best_by_cagr(x: pd.DataFrame) -> dict[str, Any]:
    if x.empty:
        raise ValueError("No rows to pick best candidate")
    y = x.sort_values(
        ["cagr", "sharpe", "trade_win_rate", "max_drawdown", "trial_id"],
        ascending=[False, False, False, False, True],
    ).head(1)
    return dict(y.iloc[0])


def _apply_combo_filter(trials: pd.DataFrame, combo: GuardrailCombo) -> pd.DataFrame:
    x = trials.copy()
    if combo.min_trade_win_rate is not None:
        x = x[x["trade_win_rate"] >= float(combo.min_trade_win_rate)].copy()
    if combo.max_drawdown_floor is not None:
        x = x[x["max_drawdown"] >= float(combo.max_drawdown_floor)].copy()
    if combo.min_sharpe is not None:
        x = x[x["sharpe"] >= float(combo.min_sharpe)].copy()
    return x


def _tier_strength_score(tier: str | None) -> int:
    if tier is None:
        return 0
    rank = {"high": 1, "moderate": 2, "low": 3}
    return int(rank.get(tier, 0))


def _combo_strength_score(r: pd.Series) -> int:
    return (
        _tier_strength_score(None if pd.isna(r["win_tier"]) else str(r["win_tier"]))
        + _tier_strength_score(None if pd.isna(r["dd_tier"]) else str(r["dd_tier"]))
        + _tier_strength_score(None if pd.isna(r["sharpe_tier"]) else str(r["sharpe_tier"]))
    )


def _ensure_persist_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agentic_guardrail_risk_matrix_v1 (
            run_ts TIMESTAMP,
            train_start_date DATE,
            train_end_date DATE,
            optimizer_trials INTEGER,
            combo_mode VARCHAR,
            risk_tiers VARCHAR,
            combo_index INTEGER,
            combo_name VARCHAR,
            win_tier VARCHAR,
            dd_tier VARCHAR,
            sharpe_tier VARCHAR,
            min_trade_win_rate DOUBLE,
            max_drawdown_floor DOUBLE,
            min_sharpe DOUBLE,
            passed_trials INTEGER,
            best_trial_id INTEGER,
            cagr DOUBLE,
            trade_win_rate DOUBLE,
            max_drawdown DOUBLE,
            sharpe DOUBLE,
            cagr_retention_vs_raw DOUBLE,
            cagr_ge_25 BOOLEAN,
            combo_strength_score INTEGER,
            policy_json VARCHAR,
            agent_weights_json VARCHAR
        )
        """
    )


def main(argv=None):
    args = parse_args(argv)
    tiers = _parse_risk_tiers(args.risk_tiers)
    thresholds = _risk_thresholds(args)
    combos = _build_combos(args.combo_mode, tiers, thresholds)

    base_policy = _base_policy(args)
    bt_template = _bt_template(args)
    mode_txt = "full-grid" if bool(args.full_grid) else f"sampled(max_trials={int(args.optimizer_trials)})"
    print(
        f"[guardrail] start | mode={mode_txt} | combo_mode={args.combo_mode} | risk_tiers={tiers}",
        flush=True,
    )

    end_date = args.end_date or str(pd.Timestamp.utcnow().date())
    opt_cfg = OptimizerConfig(
        db_path=str(args.db_path),
        train_start_date=str(args.start_date),
        train_end_date=str(end_date),
        max_trials=int(args.optimizer_trials),
        min_trade_win_rate=0.0,
        max_drawdown_floor=-1.0,
        objective_mode="cagr_max",
        full_grid=bool(args.full_grid),
        progress_every=int(args.progress_every),
        checkpoint_csv_path=str(args.checkpoint_csv),
        checkpoint_every=int(args.checkpoint_every),
        resume_from_checkpoint=bool(args.resume),
        persist_results=False,
    )
    trials = evaluate_trial_grid(
        opt_cfg=opt_cfg,
        bt_template=bt_template,
        base_policy=base_policy,
    ).sort_values("trial_id")
    if trials.empty:
        raise RuntimeError("No trials returned from evaluate_trial_grid")

    raw_best = _pick_best_by_cagr(trials)
    raw_cagr = float(raw_best["cagr"])

    rows: list[dict[str, Any]] = []
    for i, combo in enumerate(combos, start=1):
        filtered = _apply_combo_filter(trials, combo)
        row: dict[str, Any] = {
            "combo_index": int(i),
            "combo_name": combo.combo_name,
            "win_tier": combo.win_tier,
            "dd_tier": combo.dd_tier,
            "sharpe_tier": combo.sharpe_tier,
            "min_trade_win_rate": combo.min_trade_win_rate,
            "max_drawdown_floor": combo.max_drawdown_floor,
            "min_sharpe": combo.min_sharpe,
            "passed_trials": int(len(filtered)),
            "best_trial_id": None,
            "cagr": None,
            "trade_win_rate": None,
            "max_drawdown": None,
            "sharpe": None,
            "cagr_retention_vs_raw": None,
            "cagr_ge_25": False,
            "combo_strength_score": 0,
            "policy_json": None,
            "agent_weights_json": None,
        }
        if not filtered.empty:
            best = _pick_best_by_cagr(filtered)
            cagr = float(best["cagr"])
            retention = cagr / raw_cagr if abs(raw_cagr) > 1e-12 else None
            row.update(
                {
                    "best_trial_id": int(best["trial_id"]),
                    "cagr": cagr,
                    "trade_win_rate": float(best["trade_win_rate"]),
                    "max_drawdown": float(best["max_drawdown"]),
                    "sharpe": float(best["sharpe"]),
                    "cagr_retention_vs_raw": retention,
                    "cagr_ge_25": bool(cagr >= 0.25),
                    "policy_json": str(best["policy_json"]),
                    "agent_weights_json": str(best["agent_weights_json"]),
                }
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    out["combo_strength_score"] = out.apply(_combo_strength_score, axis=1)
    out.insert(0, "risk_tiers", ",".join(tiers))
    out.insert(0, "combo_mode", str(args.combo_mode))
    out.insert(0, "optimizer_trials", int(args.optimizer_trials))
    out.insert(0, "train_end_date", str(end_date))
    out.insert(0, "train_start_date", str(args.start_date))

    out_path = (
        args.out_csv.strip()
        if str(args.out_csv).strip()
        else "data/reports/guardrail_risk_matrix_v1.csv"
    )
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, index=False)

    if args.persist:
        conn = duckdb.connect(str(args.db_path))
        try:
            _ensure_persist_table(conn)
            z = out.copy()
            z.insert(0, "run_ts", pd.Timestamp.utcnow())
            conn.register("guardrail_rows", z)
            conn.execute("INSERT INTO agentic_guardrail_risk_matrix_v1 SELECT * FROM guardrail_rows")
            conn.unregister("guardrail_rows")
        finally:
            conn.close()

    print("===== MAX CAGR BASELINE =====")
    print(f"trial_id: {int(raw_best['trial_id'])}")
    print(f"CAGR: {float(raw_best['cagr']):.2%}")
    print(f"Trade Win: {float(raw_best['trade_win_rate']):.2%}")
    print(f"MaxDD: {float(raw_best['max_drawdown']):.2%}")
    print(f"Sharpe: {float(raw_best['sharpe']):.2f}")
    print("=============================")

    print("===== GUARDRAIL COMBOS =====")
    for _, r in out.iterrows():
        if pd.isna(r["cagr"]):
            print(
                f"{int(r['combo_index']):02d}. {r['combo_name']} | "
                f"pass={int(r['passed_trials'])} | no candidate"
            )
        else:
            retain_txt = (
                f"{float(r['cagr_retention_vs_raw']):.2%}"
                if pd.notna(r["cagr_retention_vs_raw"])
                else "n/a"
            )
            print(
                f"{int(r['combo_index']):02d}. {r['combo_name']} | "
                f"pass={int(r['passed_trials'])} | "
                f"CAGR={float(r['cagr']):.2%} | "
                f"retain={retain_txt} | "
                f"win={float(r['trade_win_rate']):.2%} | "
                f"maxdd={float(r['max_drawdown']):.2%} | "
                f"sharpe={float(r['sharpe']):.2f} | "
                f"trial={int(r['best_trial_id'])}"
            )
    print("============================")

    keep_25 = out[(out["cagr"].notna()) & (out["cagr"] >= 0.25)].copy()
    if keep_25.empty:
        print("No risk-tier combo retained CAGR >= 25%.")
    else:
        best_cagr_row = keep_25.sort_values("cagr", ascending=False).iloc[0]
        strongest_row = (
            keep_25.sort_values(
                ["combo_strength_score", "cagr"],
                ascending=[False, False],
            )
            .iloc[0]
        )
        print("===== BEST CAGR (WITH CAGR >= 25%) =====")
        print(
            f"combo={best_cagr_row['combo_name']} | CAGR={float(best_cagr_row['cagr']):.2%} | "
            f"win={float(best_cagr_row['trade_win_rate']):.2%} | "
            f"maxdd={float(best_cagr_row['max_drawdown']):.2%} | "
            f"sharpe={float(best_cagr_row['sharpe']):.2f}"
        )
        print("========================================")
        print("===== STRONGEST RISK COMBO (WITH CAGR >= 25%) =====")
        print(
            f"combo={strongest_row['combo_name']} | strength={int(strongest_row['combo_strength_score'])} | "
            f"CAGR={float(strongest_row['cagr']):.2%} | "
            f"win={float(strongest_row['trade_win_rate']):.2%} | "
            f"maxdd={float(strongest_row['max_drawdown']):.2%} | "
            f"sharpe={float(strongest_row['sharpe']):.2f}"
        )
        print("====================================================")

    print(f"report_csv: {out_file}")


if __name__ == "__main__":
    main()
