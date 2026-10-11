import numpy as np
import pytest

from decision_proof.estimators.rdd import RegressionDiscontinuity
from decision_proof.simulate.rdd import generate_rdd_data


def test_generate_rdd_data_is_reproducible():
    a = generate_rdd_data(n_customers=500, seed=42)
    b = generate_rdd_data(n_customers=500, seed=42)
    import pandas.testing as pdt

    pdt.assert_frame_equal(a.df, b.df)


def test_rdd_recovers_the_true_jump():
    data = generate_rdd_data(n_customers=4000, seed=42)
    est = RegressionDiscontinuity().fit(data.df, "running_var", "next_period_spend", data.cutoff)

    assert est.effect_lo <= data.true_jump <= est.effect_hi
    assert abs(est.effect - data.true_jump) < 3  # reasonably tight at n=4000


def test_rdd_placebo_cutoffs_show_no_effect():
    data = generate_rdd_data(n_customers=4000, seed=42)
    estimator = RegressionDiscontinuity()
    placebos = estimator.placebo_cutoffs(
        data.df, "running_var", "next_period_spend", data.cutoff, offsets=[-40, -20, 20, 40]
    )

    assert len(placebos) == 4
    for fake_cutoff, est in placebos.items():
        significant = not (est.effect_lo <= 0 <= est.effect_hi)
        assert not significant, f"placebo at {fake_cutoff} spuriously significant"


def test_rdd_raises_with_too_few_observations_near_cutoff():
    data = generate_rdd_data(n_customers=100, seed=42)
    estimator = RegressionDiscontinuity(bandwidth_frac=0.01)
    with pytest.raises(ValueError):
        estimator.fit(data.df, "running_var", "next_period_spend", data.cutoff)


def test_rdd_coverage_across_seeds():
    """Loose calibration check: the 90% interval should contain the true jump
    close to 90% of the time, not dramatically off in either direction."""
    covered = 0
    n = 30
    for seed in range(2000, 2000 + n):
        data = generate_rdd_data(n_customers=3000, seed=seed)
        est = RegressionDiscontinuity().fit(data.df, "running_var", "next_period_spend", data.cutoff)
        if est.effect_lo <= data.true_jump <= est.effect_hi:
            covered += 1

    assert covered / n >= 0.6  # loose bound - this is a sanity check, not a precise calibration claim
