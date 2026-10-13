"""Customer-level data for a heterogeneous-treatment-effect situation: does the
tested price change help different customers differently, by how much they
already spend? Randomized like Q8 (so there's no confounding to also have to
deal with — this module is purely about *whose* effect is bigger, not whether
the average effect is real), but here the true treatment effect itself varies
with a customer covariate (loyalty_score), rather than being constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE_EFFECT = 1.0  # CATE at loyalty_score = 0
EFFECT_SLOPE = 0.08  # $ of additional treatment effect per point of loyalty_score


@dataclass(frozen=True)
class UpliftData:
    df: pd.DataFrame  # columns: customer_id, loyalty_score, treatment, outcome
    base_effect: float
    effect_slope: float

    def true_cate(self, loyalty_score: np.ndarray) -> np.ndarray:
        return self.base_effect + self.effect_slope * loyalty_score


def generate_uplift_data(n_customers: int = 3000, seed: int = 42) -> UpliftData:
    rng = np.random.default_rng(seed)

    loyalty_score = rng.uniform(0, 100, n_customers)
    treatment = rng.integers(0, 2, n_customers)

    baseline = 30.0 + 0.3 * loyalty_score
    true_cate = BASE_EFFECT + EFFECT_SLOPE * loyalty_score
    noise = rng.normal(0, 15, n_customers)

    outcome = baseline + true_cate * treatment + noise
    outcome = np.maximum(outcome, 0.0)

    df = pd.DataFrame(
        {
            "customer_id": [f"cust_{i:05d}" for i in range(n_customers)],
            "loyalty_score": loyalty_score,
            "treatment": treatment,
            "outcome": outcome,
        }
    )
    return UpliftData(df=df, base_effect=BASE_EFFECT, effect_slope=EFFECT_SLOPE)
