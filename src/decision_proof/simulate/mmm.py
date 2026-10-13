"""Weekly time series for a marketing-mix situation: Daybreak Coffee runs price
changes and promo/ad spend campaigns together, and wants to know how much of
revenue each one actually drove. Ad spend doesn't work like price — it carries
over (adstock: this week's spend still nudges sales several weeks out) and it
saturates (diminishing returns: doubling spend doesn't double sales).

A single aggregate time series (company-wide weekly revenue), not a panel of
units — the last of the genuinely different data shapes this phase covers (no
unit_id at all here, just period).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_WEEKS = 104
TRUE_ADSTOCK_DECAY = 0.5  # fraction of last week's adstocked spend that carries into this week
TRUE_SATURATION_K = 8000.0  # half-saturation point: adstocked spend at which the response is half its max
TRUE_PRICE_COEF = -3000.0  # $ revenue per $1 of price change
TRUE_PROMO_MAX_RESPONSE = 25000.0  # $ revenue at full saturation of the promo response


def adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Geometric carryover: adstocked_t = spend_t + decay * adstocked_{t-1}."""
    out = np.zeros_like(spend, dtype=float)
    out[0] = spend[0]
    for t in range(1, len(spend)):
        out[t] = spend[t] + decay * out[t - 1]
    return out


def saturate(adstocked: np.ndarray, k: float) -> np.ndarray:
    """Hill-style saturation (diminishing returns), normalized to [0, 1]."""
    return adstocked / (adstocked + k)


@dataclass(frozen=True)
class MMMData:
    df: pd.DataFrame  # columns: period, price, promo_spend, revenue
    true_adstock_decay: float
    true_saturation_k: float
    true_price_coef: float
    true_promo_max_response: float


def generate_mmm_data(n_weeks: int = N_WEEKS, seed: int = 42) -> MMMData:
    rng = np.random.default_rng(seed)
    t = np.arange(1, n_weeks + 1)

    base_price = 6.5
    price = base_price + rng.choice([-0.3, 0, 0.3, 0.6], size=n_weeks, p=[0.1, 0.5, 0.25, 0.15]).cumsum() * 0.1
    price = np.clip(price, 5.5, 8.5)

    # promo spend: sparse bursts of campaign activity, not a smooth series
    promo_spend = np.zeros(n_weeks)
    n_campaigns = n_weeks // 10
    campaign_starts = rng.choice(np.arange(n_weeks - 4), size=n_campaigns, replace=False)
    for start in campaign_starts:
        duration = rng.integers(2, 5)
        promo_spend[start : start + duration] += rng.uniform(3000, 12000)

    adstocked = adstock(promo_spend, TRUE_ADSTOCK_DECAY)
    promo_response = TRUE_PROMO_MAX_RESPONSE * saturate(adstocked, TRUE_SATURATION_K)

    trend = 300 * t / n_weeks * 10
    seasonal = 4000 * np.sin(2 * np.pi * t / 52)
    baseline = 70_000 + trend + seasonal
    noise = rng.normal(0, 3500, n_weeks)

    revenue = baseline + TRUE_PRICE_COEF * (price - base_price) + promo_response + noise
    revenue = np.maximum(revenue, 0.0)

    df = pd.DataFrame({"period": t, "price": price, "promo_spend": promo_spend, "revenue": revenue})
    return MMMData(
        df=df,
        true_adstock_decay=TRUE_ADSTOCK_DECAY,
        true_saturation_k=TRUE_SATURATION_K,
        true_price_coef=TRUE_PRICE_COEF,
        true_promo_max_response=TRUE_PROMO_MAX_RESPONSE,
    )
