"""Customer-level data for a true randomized experiment: an in-app price test
where each customer is randomly shown one of two prices, 50/50. No confounding
by design — this is the one scenario in the repo where simple difference-in-
means is already unbiased. The interesting question isn't "is this estimate
right," it's "how much faster can we get a precise answer" — which is what
CUPED (estimators/rct.py) is for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRUE_EFFECT = 4.0  # $ average post-period spend caused by the tested price
PRE_POST_CORRELATION_SLOPE = 0.6  # how strongly pre-period spend predicts post-period spend


@dataclass(frozen=True)
class RCTData:
    df: pd.DataFrame  # columns: customer_id, pre_period_spend, treatment, post_period_spend
    true_effect: float


def generate_rct_data(n_customers: int = 2000, seed: int = 42) -> RCTData:
    rng = np.random.default_rng(seed)

    pre_period_spend = rng.gamma(shape=4.0, scale=15.0, size=n_customers)  # right-skewed, like real spend data
    treatment = rng.integers(0, 2, n_customers)  # true 50/50 random assignment

    baseline_noise = rng.normal(0, 20, n_customers)
    post_period_spend = (
        10.0
        + PRE_POST_CORRELATION_SLOPE * pre_period_spend
        + TRUE_EFFECT * treatment
        + baseline_noise
    )
    post_period_spend = np.maximum(post_period_spend, 0.0)

    df = pd.DataFrame(
        {
            "customer_id": [f"cust_{i:05d}" for i in range(n_customers)],
            "pre_period_spend": pre_period_spend,
            "treatment": treatment,
            "post_period_spend": post_period_spend,
        }
    )
    return RCTData(df=df, true_effect=TRUE_EFFECT)
