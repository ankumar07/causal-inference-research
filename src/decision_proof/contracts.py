"""Shared data contracts. Every module speaks these types."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict


class Panel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: pd.DataFrame  # columns: unit_id, period, outcome, [covariates...]
    outcome: str
    period_freq: str = "W"


class Decision(BaseModel):
    decision_id: str
    description: str  # "Price +6% on 4 Daybreak metro markets"
    treated_units: list[str]
    start_period: int
    outcome: str
    owner: str | None = None


class EffectEstimate(BaseModel):
    method: str
    cum_effect: float
    cum_lo: float
    cum_hi: float  # 90% interval
    pct_effect: float
    pct_lo: float
    pct_hi: float
    path: list[dict[str, float]]  # {h, est, lo, hi} per week after the change
    lag_50: int | None
    lag_90: int | None
    diagnostics: dict[str, Any]


CheckStatus = Literal["PASS", "WARN", "FAIL"]
Verdict = Literal["BOARD_READY", "DIRECTIONAL", "NOT_PROVEN"]


class RefutationCheck(BaseModel):
    name: str
    statistic: float | None
    threshold: float | None
    status: CheckStatus
    explanation: str
    critical: bool


class RefutationReport(BaseModel):
    checks: list[RefutationCheck]
    verdict: Verdict
    reasons: list[str]


class GroundTruth(BaseModel):
    """Known answer from the simulator, used to check every method."""

    treated_units: list[str]
    start_period: int
    true_effect_path: list[float]  # per post period, summed over treated units
    true_cum_effect: float
    true_pct_effect: float
