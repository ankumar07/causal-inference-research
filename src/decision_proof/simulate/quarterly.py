"""Quarterly multi-experiment world: Daybreak Coffee runs a batch of price
experiments every quarter, each at a different price-change magnitude (so a
later sensitivity analysis has several (price %, effect %) points to fit a
curve through), plus a simple demographic tag per metro.

Within one quarter, several experiments run on disjoint metro groups at the
same time. Each experiment's "control" metros must exclude every *other*
experiment's treated group from that same quarter — otherwise a donor that's
itself mid-price-test gets used as if it were an untreated control, the same
contamination problem B5's conservation-constrained synthetic control solves,
just across experiments instead of within one. Handled here by filtering each
experiment's panel down to its own treated group plus the metros not involved
in *any* of that quarter's experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from decision_proof.contracts import Decision, GroundTruth, Panel
from decision_proof.simulate.dgp import TreatmentSpec, apply_treatment, make_units, metro_id, simulate_baseline

WEEKS_PER_QUARTER = 13
N_QUARTERS = 6
TAIL_WEEKS = 20  # post-period buffer after the final quarter's experiments
N_PERIODS = N_QUARTERS * WEEKS_PER_QUARTER + TAIL_WEEKS

EXPERIMENT_QUARTERS = [3, 4, 5, 6]  # quarters 1-2 are pure pre-period history, no experiments
PRICE_POINTS = [0.03, 0.06, 0.09, 0.12]  # four magnitudes per quarter, for the elasticity curve
GROUP_SIZE = 3
ELASTICITY = -0.5  # shared across all experiments, so "true" sensitivity is recoverable from the batch

# More metros than the single-decision scenarios (dgp.N_METROS=24) use. Running 4
# experiments at once each quarter means every experiment's donor pool has to exclude
# the other 3 groups too (see module docstring) - at 24 metros that leaves only ~12
# donors, materially weaker than the ~20 a single-experiment scenario gets, and
# placebo-in-space/placebo-in-time both degrade noticeably. More metros restores a
# comparable donor count per experiment without touching dgp.N_METROS (and the
# calibration numbers in docs/LAB_REPORT.md that are keyed to it).
N_METROS = 36

AGE_BANDS = ["under_30", "30_50", "50_plus"]
INCOME_TIERS = ["low", "mid", "high"]


def quarter_start_period(quarter: int) -> int:
    """First week (1-indexed) of the given quarter."""
    return (quarter - 1) * WEEKS_PER_QUARTER + 1


def quarter_label(quarter: int) -> str:
    return f"Q{quarter}"


def assign_demographics(seed: int, n_units: int = N_METROS) -> dict[str, dict[str, str]]:
    """One demographic profile per metro: dominant age band + income tier. Not
    customer-level data - a market-level tag, which is what the existing
    metro-panel simulator can support without a much larger rewrite.
    """
    rng = np.random.default_rng(seed + 9001)  # offset so it doesn't collide with make_units' own draws
    return {
        metro_id(i): {
            "age_band": str(rng.choice(AGE_BANDS)),
            "income_tier": str(rng.choice(INCOME_TIERS)),
        }
        for i in range(n_units)
    }


@dataclass(frozen=True)
class QuarterExperiment:
    quarter: int
    decision_id: str
    treated_units: tuple[str, ...]
    price_change_pct: float
    start_period: int


def plan_quarter_experiments(n_units: int = N_METROS) -> list[QuarterExperiment]:
    """Disjoint metro groups, one per price point, run simultaneously each
    experimental quarter.
    """
    needed = len(PRICE_POINTS) * GROUP_SIZE
    if needed > n_units:
        raise ValueError(f"need at least {needed} metros for {len(PRICE_POINTS)} groups of {GROUP_SIZE}, have {n_units}")

    plans = []
    for quarter in EXPERIMENT_QUARTERS:
        start = quarter_start_period(quarter)
        for i, price_pct in enumerate(PRICE_POINTS):
            treated = tuple(metro_id(j) for j in range(i * GROUP_SIZE, (i + 1) * GROUP_SIZE))
            plans.append(
                QuarterExperiment(
                    quarter=quarter,
                    decision_id=f"q{quarter}-p{round(price_pct * 100)}",
                    treated_units=treated,
                    price_change_pct=price_pct,
                    start_period=start,
                )
            )
    return plans


def generate_quarterly_world(
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]], list[QuarterExperiment]]:
    units = make_units(N_METROS, seed=seed)
    baseline = simulate_baseline(units, N_PERIODS, seed=seed)
    demographics = assign_demographics(seed)
    plans = plan_quarter_experiments()
    return baseline, demographics, plans


def build_experiment(
    baseline: pd.DataFrame, plan: QuarterExperiment, all_plans: list[QuarterExperiment]
) -> tuple[Panel, Decision, GroundTruth]:
    concurrent = [p for p in all_plans if p.quarter == plan.quarter and p.decision_id != plan.decision_id]
    other_treated = {u for p in concurrent for u in p.treated_units}
    clean_df = baseline[~baseline["unit_id"].isin(other_treated)].copy()

    spec = TreatmentSpec(
        treated_units=list(plan.treated_units),
        start_period=plan.start_period,
        effect_pct=plan.price_change_pct,
        ramp_weeks=4,
        elasticity=ELASTICITY,
    )
    df, truth = apply_treatment(clean_df, spec)
    decision = Decision(
        decision_id=plan.decision_id,
        description=(
            f"Price +{plan.price_change_pct:.0%} on {len(plan.treated_units)} Daybreak metro markets "
            f"({quarter_label(plan.quarter)})"
        ),
        treated_units=list(plan.treated_units),
        start_period=plan.start_period,
        outcome="revenue",
    )
    return Panel(df=df, outcome="revenue", period_freq="W"), decision, truth


def experiments_for_quarter(
    baseline: pd.DataFrame, all_plans: list[QuarterExperiment], quarter: int
) -> list[tuple[Panel, Decision, GroundTruth]]:
    return [build_experiment(baseline, p, all_plans) for p in all_plans if p.quarter == quarter]
