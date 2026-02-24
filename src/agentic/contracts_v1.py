from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class ActionVote(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    WAIT = "WAIT"


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    weight: float
    enabled: bool = True
    min_confidence: float = 0.05


@dataclass(frozen=True)
class RunContext:
    run_id: str
    as_of_date: date
    db_path: str


@dataclass
class RegimeState:
    label: str
    score: float
    exposure_scalar: float
    breadth: float
    market_ret_63: float


@dataclass
class OrchestrationPolicy:
    watchlist_size: int = 25
    portfolio_min_positions: int = 10
    portfolio_max_positions: int = 15
    portfolio_target_positions: int = 12
    consensus_buy_threshold: float = 0.66
    consensus_sell_threshold: float = 0.34
    max_price_staleness_days: int = 5
    min_median_turnover_inr: float = 5.0e7
    max_single_weight: float = 0.10
    min_single_weight: float = 0.03

    def validate(self) -> None:
        if self.watchlist_size <= 0:
            raise ValueError("watchlist_size must be > 0")
        if self.portfolio_min_positions <= 0:
            raise ValueError("portfolio_min_positions must be > 0")
        if self.portfolio_max_positions < self.portfolio_min_positions:
            raise ValueError("portfolio_max_positions must be >= portfolio_min_positions")
        if not (
            self.portfolio_min_positions
            <= self.portfolio_target_positions
            <= self.portfolio_max_positions
        ):
            raise ValueError(
                "portfolio_target_positions must be between "
                "portfolio_min_positions and portfolio_max_positions"
            )
        if not (0.0 < self.consensus_sell_threshold < self.consensus_buy_threshold < 1.0):
            raise ValueError("consensus thresholds must satisfy 0 < sell < buy < 1")
        if self.max_price_staleness_days < 0:
            raise ValueError("max_price_staleness_days must be >= 0")
        if self.min_median_turnover_inr < 0:
            raise ValueError("min_median_turnover_inr must be >= 0")
        if not (0.0 < self.min_single_weight <= self.max_single_weight <= 1.0):
            raise ValueError("weight bounds must satisfy 0 < min <= max <= 1")

    def target_positions(self, exposure_scalar: float) -> int:
        raw = int(round(self.portfolio_target_positions * clamp(exposure_scalar, 0.5, 1.5)))
        return int(clamp(raw, self.portfolio_min_positions, self.portfolio_max_positions))
