"""Cross-sectional data for an instrumental-variables situation: coffee bean
cost shocks (a regional supplier disruption, a weather event, a shipping delay)
force Daybreak Coffee to raise prices in some metros more than others, for
reasons that have nothing to do with local demand. Price itself is endogenous —
managers also raise prices more in metros that are already running hot on
demand — so naive regression of revenue on price is confounded. Cost shocks
give a (plausibly) exogenous instrument: they move price, but don't belong in
the revenue equation directly, only through price.

Genuinely different shape again: cross-sectional (one row per metro), not a
panel over time, and the whole point is a *latent* confounder the estimator
never sees directly — the ground truth here is what "no proprietary data,
only the simulator" buys you: you get to know the thing you can never observe
in real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_METROS = 500
TRUE_PRICE_EFFECT = -850.0  # $ revenue change per $1 average-ticket price increase (demand curve slopes down)
CONFOUND_ON_PRICE = 0.004  # how much latent demand pushes price up (managers chase hot demand)
CONFOUND_ON_REVENUE = 6000.0  # how much latent demand directly drives revenue (the omitted-variable bias source)
INSTRUMENT_STRENGTH = 3.0  # how much a $1 cost shock moves price (relevance) - calibrated for first-stage F > 10,
# see docs/METHODS.md: a materially weaker instrument than this makes 2SLS unusably imprecise, not just noisier


@dataclass(frozen=True)
class IVData:
    df: pd.DataFrame  # columns: metro_id, cost_shock, price, revenue (latent_demand withheld - that's the point)
    true_price_effect: float


def generate_iv_data(n_metros: int = N_METROS, seed: int = 42) -> IVData:
    rng = np.random.default_rng(seed)

    latent_demand = rng.normal(0, 1000, n_metros)  # unobserved by any estimator - the confounder
    cost_shock = rng.normal(0, 2.0, n_metros)  # $ per-unit cost shock, independent of latent_demand by construction

    base_price = 6.5
    price = (
        base_price
        + CONFOUND_ON_PRICE * latent_demand  # confounding: price chases demand
        + INSTRUMENT_STRENGTH * cost_shock  # relevance: cost shock moves price
        + rng.normal(0, 0.3, n_metros)
    )
    price = np.maximum(price, 1.0)

    base_revenue = 60_000
    revenue = (
        base_revenue
        + CONFOUND_ON_REVENUE * latent_demand / 1000  # confounding: demand also drives revenue directly
        + TRUE_PRICE_EFFECT * (price - base_price)  # the true causal effect we want to recover
        + rng.normal(0, 4000, n_metros)
        # deliberately NO direct cost_shock term here - exclusion restriction: cost shocks
        # only affect revenue through price, never directly
    )
    revenue = np.maximum(revenue, 0.0)

    df = pd.DataFrame(
        {
            "metro_id": [f"metro_{i:03d}" for i in range(n_metros)],
            "cost_shock": cost_shock,
            "price": price,
            "revenue": revenue,
        }
    )
    return IVData(df=df, true_price_effect=TRUE_PRICE_EFFECT)
