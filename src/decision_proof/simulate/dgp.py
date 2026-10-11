"""Data-generating process for Daybreak Coffee: a simulated multi-metro drive-thru
coffee chain (structurally modeled on Dutch Bros' public footprint, not its data).

Each unit is a metro market (a cluster of shops sharing a customer base). Weekly
outcome is revenue = transactions * avg_ticket, built from a baseline level, a
shared trend, period-52 seasonality with a random phase per unit, AR(1) noise,
and occasional shocks. A price-change treatment raises avg_ticket and (via an
elasticity) lowers transactions; the lost transactions can optionally spill into
a neighboring untreated unit, contaminating it as a synthetic-control donor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from decision_proof.contracts import GroundTruth

N_METROS = 24
WEEKS_PER_YEAR = 52


def metro_id(i: int) -> str:
    return f"metro_{i:02d}"


def neighbor_of(unit_id: str, n_units: int = N_METROS) -> str:
    """Ring topology: each metro's nearest market-sharing neighbor is the next one in the ring."""
    i = int(unit_id.split("_")[1])
    return metro_id((i + 1) % n_units)


@dataclass(frozen=True)
class UnitParams:
    unit_id: str
    baseline_transactions: float
    avg_ticket: float
    trend_weekly: float
    seasonal_amplitude: float
    seasonal_phase: float
    ar1_rho: float
    ar1_sigma: float
    shock_prob: float
    shock_sigma: float


def make_units(n_units: int = N_METROS, seed: int = 0) -> list[UnitParams]:
    rng = np.random.default_rng(seed)
    shared_trend = rng.normal(0.0008, 0.0002)  # one draw, shared by every unit (per the M1 spec)
    units = []
    for i in range(n_units):
        units.append(
            UnitParams(
                unit_id=metro_id(i),
                baseline_transactions=rng.uniform(4_000, 12_000),
                avg_ticket=rng.uniform(5.5, 7.5),
                trend_weekly=shared_trend,
                seasonal_amplitude=rng.uniform(0.05, 0.12),
                seasonal_phase=rng.uniform(0, 2 * np.pi),
                ar1_rho=rng.uniform(0.3, 0.6),
                ar1_sigma=rng.uniform(0.02, 0.04),
                shock_prob=0.03,
                shock_sigma=rng.uniform(0.05, 0.10),
            )
        )
    return units


def nearest_seasonal_peak(unit: UnitParams, near_period: int) -> int:
    """Period closest to `near_period` where the unit's seasonal term peaks."""
    best_t, best_val = near_period, -np.inf
    for t in range(max(1, near_period - WEEKS_PER_YEAR // 2), near_period + WEEKS_PER_YEAR // 2):
        val = np.sin(2 * np.pi * t / WEEKS_PER_YEAR + unit.seasonal_phase)
        if val > best_val:
            best_val, best_t = val, t
    return best_t


def _simulate_series(unit: UnitParams, n_periods: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Baseline (untreated) transactions and avg_ticket arrays, one value per period 1..n_periods."""
    t = np.arange(1, n_periods + 1)
    trend_mult = (1 + unit.trend_weekly) ** t
    seasonal_mult = 1 + unit.seasonal_amplitude * np.sin(2 * np.pi * t / WEEKS_PER_YEAR + unit.seasonal_phase)

    eps = rng.normal(0, unit.ar1_sigma, n_periods)
    ar1 = np.zeros(n_periods)
    ar1[0] = eps[0]
    for i in range(1, n_periods):
        ar1[i] = unit.ar1_rho * ar1[i - 1] + eps[i]
    noise_mult = np.exp(ar1 - 0.5 * unit.ar1_sigma**2)

    shocked = rng.random(n_periods) < unit.shock_prob
    shock_mult = np.ones(n_periods)
    shock_mult[shocked] = 1 + rng.normal(0, unit.shock_sigma, int(shocked.sum()))

    transactions = unit.baseline_transactions * trend_mult * seasonal_mult * noise_mult * shock_mult
    transactions = np.maximum(transactions, 1.0)

    avg_ticket = unit.avg_ticket * np.exp(rng.normal(0, 0.01, n_periods))

    return transactions, avg_ticket


def simulate_baseline(units: list[UnitParams], n_periods: int, seed: int) -> pd.DataFrame:
    """Untreated panel: unit_id, period, transactions, avg_ticket, revenue."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in units:
        transactions, avg_ticket = _simulate_series(unit, n_periods, rng)
        revenue = transactions * avg_ticket
        for t in range(n_periods):
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "period": t + 1,
                    "transactions": transactions[t],
                    "avg_ticket": avg_ticket[t],
                    "revenue": revenue[t],
                }
            )
    return pd.DataFrame(rows)


def ramp_progress(periods_since_start: np.ndarray, ramp_weeks: int) -> np.ndarray:
    """Linear ramp from just-above-0 to 1 over `ramp_weeks` periods, then flat at 1."""
    return np.clip((periods_since_start + 1) / ramp_weeks, 0.0, 1.0)


@dataclass(frozen=True)
class TreatmentSpec:
    treated_units: list[str]
    start_period: int | dict[str, int]  # single period, or per-unit for a staggered rollout (S5)
    effect_pct: float
    ramp_weeks: int = 4
    elasticity: float = -0.3
    spillover_frac: float = 0.0
    heterogeneity: dict[str, float] | None = None  # per-unit multiplier on effect_pct
    zero_effect: bool = False


def apply_treatment(
    baseline: pd.DataFrame, spec: TreatmentSpec
) -> tuple[pd.DataFrame, GroundTruth]:
    """Apply a price-change treatment to the baseline panel. Returns the treated panel
    plus the ground truth (the true dollar/pct effect on treated units only — spillover
    contaminates *other* units' outcomes but is not part of the treated units' own effect).
    """
    df = baseline.copy()
    starts = (
        spec.start_period
        if isinstance(spec.start_period, dict)
        else {u: spec.start_period for u in spec.treated_units}
    )
    overall_start = min(starts.values())

    true_effect_by_period: dict[int, float] = {}
    lost_by_unit_period: dict[tuple[str, int], float] = {}

    for unit_id in spec.treated_units:
        start = starts[unit_id]
        mult = 1.0 if spec.heterogeneity is None else spec.heterogeneity.get(unit_id, 1.0)
        mask = (df["unit_id"] == unit_id).to_numpy()
        periods = df.loc[mask, "period"].to_numpy()
        since_start = periods - start
        active = since_start >= 0

        ramp = np.zeros(len(periods))
        ramp[active] = ramp_progress(since_start[active], spec.ramp_weeks)
        effect_pct = np.zeros(len(periods)) if spec.zero_effect else spec.effect_pct * mult * ramp

        base_transactions = df.loc[mask, "transactions"].to_numpy()
        base_ticket = df.loc[mask, "avg_ticket"].to_numpy()
        base_revenue = base_transactions * base_ticket

        new_ticket = base_ticket * (1 + effect_pct)
        new_transactions = base_transactions * (1 + spec.elasticity * effect_pct)
        new_revenue = new_transactions * new_ticket

        lost = base_transactions - new_transactions
        true_effect = new_revenue - base_revenue

        df.loc[mask, "avg_ticket"] = new_ticket
        df.loc[mask, "transactions"] = new_transactions
        df.loc[mask, "revenue"] = new_revenue

        for p, te, l in zip(periods, true_effect, lost):
            if p >= overall_start:
                true_effect_by_period[int(p)] = true_effect_by_period.get(int(p), 0.0) + float(te)
            lost_by_unit_period[(unit_id, int(p))] = float(l)

    if spec.spillover_frac > 0:
        for unit_id in spec.treated_units:
            neighbor = neighbor_of(unit_id)
            if neighbor in spec.treated_units:
                continue  # only spill into untreated neighbors, so the contamination story is clean
            nmask = (df["unit_id"] == neighbor).to_numpy()
            for p in df.loc[nmask, "period"]:
                lost = lost_by_unit_period.get((unit_id, int(p)), 0.0)
                if lost > 0:
                    gain = spec.spillover_frac * lost
                    idx = df.index[(df["unit_id"] == neighbor) & (df["period"] == p)]
                    df.loc[idx, "transactions"] += gain
                    df.loc[idx, "revenue"] = df.loc[idx, "transactions"] * df.loc[idx, "avg_ticket"]

    max_period = int(df["period"].max())
    true_effect_path = [true_effect_by_period.get(p, 0.0) for p in range(overall_start, max_period + 1)]
    true_cum_effect = float(sum(true_effect_path))

    baseline_denom = baseline[
        baseline["unit_id"].isin(spec.treated_units) & (baseline["period"] >= overall_start)
    ]["revenue"].sum()
    true_pct_effect = 0.0 if baseline_denom == 0 else true_cum_effect / baseline_denom * 100

    ground_truth = GroundTruth(
        treated_units=list(spec.treated_units),
        start_period=overall_start,
        true_effect_path=true_effect_path,
        true_cum_effect=true_cum_effect,
        true_pct_effect=true_pct_effect,
    )
    return df, ground_truth
