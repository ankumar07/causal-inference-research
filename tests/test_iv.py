import numpy as np

from decision_proof.estimators.iv import InstrumentalVariables, naive_ols
from decision_proof.simulate.iv import generate_iv_data


def test_generate_iv_data_is_reproducible():
    a = generate_iv_data(seed=42)
    b = generate_iv_data(seed=42)
    import pandas.testing as pdt

    pdt.assert_frame_equal(a.df, b.df)


def test_naive_ols_is_confounded_and_misses_the_truth():
    """Exit check for Q7: naive OLS of revenue on price is biased by the latent
    demand confounder and should not reliably contain the true effect."""
    data = generate_iv_data(seed=42)
    naive = naive_ols(data.df, "revenue", "price")

    assert not (naive.lo <= data.true_price_effect <= naive.hi)


def test_first_stage_is_a_strong_instrument():
    data = generate_iv_data(seed=42)
    f_stat = InstrumentalVariables().first_stage_f_stat(data.df, "price", "cost_shock")
    assert f_stat > 10  # Staiger-Stock rule of thumb


def test_2sls_recovers_the_true_price_effect():
    data = generate_iv_data(seed=42)
    result = InstrumentalVariables().fit(data.df, "revenue", "price", "cost_shock")

    assert result.lo <= data.true_price_effect <= result.hi
    assert abs(result.coef - data.true_price_effect) < 200  # reasonably tight


def test_2sls_coverage_beats_naive_ols_across_seeds():
    """Exit test for Q7: across many replications, 2SLS's 90% interval should
    cover the true effect close to 90% of the time; naive OLS's shouldn't cover
    it at all, since it's confounded, not just noisier."""
    naive_covered = 0
    iv_covered = 0
    n = 30
    for seed in range(2000, 2000 + n):
        data = generate_iv_data(seed=seed)
        naive = naive_ols(data.df, "revenue", "price")
        result = InstrumentalVariables().fit(data.df, "revenue", "price", "cost_shock")
        if naive.lo <= data.true_price_effect <= naive.hi:
            naive_covered += 1
        if result.lo <= data.true_price_effect <= result.hi:
            iv_covered += 1

    assert iv_covered / n >= 0.7
    assert naive_covered / n <= 0.2


def test_naive_ols_and_2sls_disagree_in_magnitude():
    data = generate_iv_data(seed=42)
    naive = naive_ols(data.df, "revenue", "price")
    result = InstrumentalVariables().fit(data.df, "revenue", "price", "cost_shock")
    # naive consistently understates the magnitude of the (negative) true effect
    assert abs(naive.coef) < abs(result.coef)
