from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.agentic.contracts_v1 import OrchestrationPolicy


MODEL_ID = "AI_AGENT_STABLE_STOCKS_V1"
MODEL_VERSION = "2026-02-24"
MODEL_SOURCE = "fullgrid_8100_trials_guardrail_risk_matrix_fullgrid_v7"
BEST_TRIAL_ID = 114
BEST_COMBO = "WIN_LOW__DD_HIGH__SH_LOW"


def locked_policy() -> OrchestrationPolicy:
    return OrchestrationPolicy(
        watchlist_size=25,
        portfolio_min_positions=10,
        portfolio_max_positions=15,
        portfolio_target_positions=12,
        consensus_buy_threshold=0.62,
        consensus_sell_threshold=0.28,
        max_price_staleness_days=3,
        min_median_turnover_inr=20_000_000.0,
        max_single_weight=0.10,
        min_single_weight=0.03,
    )


def locked_agent_weights() -> dict[str, float]:
    return {
        "technical": 0.55,
        "ownership": 0.25,
        "fundamental": 0.15,
        "news": 0.05,
    }


def locked_reference_metrics() -> dict[str, float]:
    return {
        "cagr": 0.2268159812514951,
        "trade_win_rate": 0.6452205882352942,
        "max_drawdown": -0.4117907905807138,
        "sharpe": 1.1760153156502529,
    }


def model_metadata() -> dict[str, Any]:
    p = locked_policy()
    return {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "source": MODEL_SOURCE,
        "best_trial_id": BEST_TRIAL_ID,
        "best_combo": BEST_COMBO,
        "policy": asdict(p),
        "agent_weights": locked_agent_weights(),
        "reference_metrics": locked_reference_metrics(),
    }
