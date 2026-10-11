"""Customer-level data for a regression-discontinuity situation: Daybreak Coffee's
loyalty program bumps a customer to "Gold" tier (better per-drink pricing) once
their trailing-90-day spend crosses a threshold. Does crossing that threshold
cause a jump in future spend, or do high-spend customers just keep spending at
the same underlying rate they already were?

Genuinely different shape from every other scenario in this repo: no unit_id x
period panel, no treated_units/start_period — a running variable (trailing
spend) and a cutoff. Lives in its own module rather than forcing dgp.py's panel
machinery to fit a problem it isn't shaped for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CUTOFF = 150.0
TRUE_JUMP = 12.0  # true causal effect of crossing into Gold tier, $ of next-period spend
RUNNING_VAR_LOW, RUNNING_VAR_HIGH = 50.0, 250.0
BASELINE_INTERCEPT = 20.0
BASELINE_SLOPE = 0.35  # smooth (non-causal) relationship: higher trailing spend -> higher future spend anyway
NOISE_SD = 18.0


@dataclass(frozen=True)
class RDDCustomerData:
    df: pd.DataFrame  # columns: customer_id, running_var, gold_tier, next_period_spend
    cutoff: float
    true_jump: float


def generate_rdd_data(n_customers: int = 4000, seed: int = 42) -> RDDCustomerData:
    rng = np.random.default_rng(seed)
    running_var = rng.uniform(RUNNING_VAR_LOW, RUNNING_VAR_HIGH, n_customers)
    gold_tier = (running_var >= CUTOFF).astype(int)

    # smooth baseline relationship, mildly non-linear so a purely linear local fit
    # isn't trivially exact - a small quadratic term, consistent on both sides of the cutoff
    centered = running_var - CUTOFF
    baseline = BASELINE_INTERCEPT + BASELINE_SLOPE * centered + 0.001 * centered**2
    noise = rng.normal(0, NOISE_SD, n_customers)

    next_period_spend = baseline + TRUE_JUMP * gold_tier + noise
    next_period_spend = np.maximum(next_period_spend, 0.0)

    df = pd.DataFrame(
        {
            "customer_id": [f"cust_{i:05d}" for i in range(n_customers)],
            "running_var": running_var,
            "gold_tier": gold_tier,
            "next_period_spend": next_period_spend,
        }
    )
    return RDDCustomerData(df=df, cutoff=CUTOFF, true_jump=TRUE_JUMP)
