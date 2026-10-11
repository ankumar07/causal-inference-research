"""Scenarios S1-S5 for the Daybreak Coffee demo world, each reproducible from one seed.

S1: price +6% in 4 metros from week 61, 60 weeks of history, effect ramps over 4 weeks.
S2: the same decision, but only the last 12 weeks of pre-period history are visible.
S3: a promo with no true effect, timed to coincide with a seasonal spike.
S4: a price rise with spillover into a neighboring untreated metro (donor contamination).
S5 (stretch): a staggered rollout across metros over 8 weeks.
"""

from __future__ import annotations

import pandas as pd

from decision_proof.contracts import Decision, GroundTruth, Panel
from decision_proof.simulate.dgp import (
    N_METROS,
    TreatmentSpec,
    apply_treatment,
    make_units,
    metro_id,
    nearest_seasonal_peak,
    simulate_baseline,
)

PRE_WEEKS = 60
POST_WEEKS = 20
N_PERIODS = PRE_WEEKS + POST_WEEKS
START_PERIOD = PRE_WEEKS + 1  # week 61


def _panel(df: pd.DataFrame, outcome: str = "revenue") -> Panel:
    return Panel(df=df, outcome=outcome, period_freq="W")


def _decision(decision_id: str, description: str, treated_units: list[str], start_period: int) -> Decision:
    return Decision(
        decision_id=decision_id,
        description=description,
        treated_units=treated_units,
        start_period=start_period,
        outcome="revenue",
    )


def scenario_s1(seed: int = 42) -> tuple[Panel, Decision, GroundTruth]:
    """Price +6% in 4 metros, 60 weeks of history, ramps over 4 weeks. Should land BOARD-READY."""
    units = make_units(N_METROS, seed=seed)
    baseline = simulate_baseline(units, N_PERIODS, seed=seed)
    treated = [metro_id(i) for i in range(4)]
    spec = TreatmentSpec(
        treated_units=treated,
        start_period=START_PERIOD,
        effect_pct=0.06,
        ramp_weeks=4,
        elasticity=-0.3,
    )
    df, truth = apply_treatment(baseline, spec)
    decision = _decision("s1", "Price +6% on 4 Daybreak metro markets", treated, START_PERIOD)
    return _panel(df), decision, truth


def scenario_s2(seed: int = 42) -> tuple[Panel, Decision, GroundTruth]:
    """Same decision as S1, but only 12 weeks of pre-period history are visible to the estimator."""
    panel, decision, truth = scenario_s1(seed=seed)
    short_start = START_PERIOD - 12
    df = panel.df[panel.df["period"] >= short_start].copy()
    return _panel(df), decision, truth


def scenario_s3(seed: int = 42) -> tuple[Panel, Decision, GroundTruth]:
    """A promo with no true effect, timed to coincide with a seasonal spike. Should be NOT PROVEN."""
    units = make_units(N_METROS, seed=seed)
    baseline = simulate_baseline(units, N_PERIODS, seed=seed)
    treated = [metro_id(i) for i in range(4, 8)]
    start = nearest_seasonal_peak(units[4], near_period=START_PERIOD)
    spec = TreatmentSpec(
        treated_units=treated,
        start_period=start,
        effect_pct=0.06,
        ramp_weeks=4,
        elasticity=-0.3,
        zero_effect=True,
    )
    df, truth = apply_treatment(baseline, spec)
    decision = _decision("s3", "Promo on 4 Daybreak metro markets (no price change)", treated, start)
    return _panel(df), decision, truth


def scenario_s4(seed: int = 42) -> tuple[Panel, Decision, GroundTruth]:
    """Price rise with spillover into a neighboring untreated metro, contaminating it as a donor."""
    units = make_units(N_METROS, seed=seed)
    baseline = simulate_baseline(units, N_PERIODS, seed=seed)
    # non-adjacent metros, so every treated unit's ring-neighbor is untreated and spills cleanly
    treated = [metro_id(i) for i in (10, 12, 14, 16)]
    spec = TreatmentSpec(
        treated_units=treated,
        start_period=START_PERIOD,
        effect_pct=0.07,
        ramp_weeks=4,
        elasticity=-0.6,
        spillover_frac=0.7,
    )
    df, truth = apply_treatment(baseline, spec)
    decision = _decision("s4", "Price +7% on 4 Daybreak metro markets (with spillover)", treated, START_PERIOD)
    return _panel(df), decision, truth


def scenario_s5(seed: int = 42) -> tuple[Panel, Decision, GroundTruth]:
    """Stretch: a staggered rollout across 6 metros over 8 weeks."""
    units = make_units(N_METROS, seed=seed)
    baseline = simulate_baseline(units, N_PERIODS, seed=seed)
    treated = [metro_id(i) for i in range(16, 22)]
    starts = {u: START_PERIOD + i * (8 // (len(treated) - 1)) for i, u in enumerate(treated)}
    spec = TreatmentSpec(
        treated_units=treated,
        start_period=starts,
        effect_pct=0.06,
        ramp_weeks=4,
        elasticity=-0.3,
    )
    df, truth = apply_treatment(baseline, spec)
    decision = _decision(
        "s5", "Price +6% staggered across 6 Daybreak metro markets over 8 weeks", treated, min(starts.values())
    )
    return _panel(df), decision, truth


SCENARIOS = {
    "S1": scenario_s1,
    "S2": scenario_s2,
    "S3": scenario_s3,
    "S4": scenario_s4,
    "S5": scenario_s5,
}


def generate_scenario(name: str, seed: int = 42) -> tuple[Panel, Decision, GroundTruth]:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}; choose one of {sorted(SCENARIOS)}")
    return SCENARIOS[name](seed=seed)
