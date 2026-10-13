from decision_proof.estimators.rct import cuped, simple_diff_in_means, variance_reduction
from decision_proof.simulate.rct import generate_rct_data


def test_generate_rct_data_is_reproducible():
    a = generate_rct_data(seed=42)
    b = generate_rct_data(seed=42)
    import pandas.testing as pdt

    pdt.assert_frame_equal(a.df, b.df)


def test_treatment_is_roughly_balanced():
    data = generate_rct_data(n_customers=2000, seed=42)
    frac_treated = data.df["treatment"].mean()
    assert 0.4 < frac_treated < 0.6


def test_simple_diff_in_means_recovers_true_effect():
    data = generate_rct_data(seed=42)
    result = simple_diff_in_means(data.df, "post_period_spend", "treatment")
    assert result.lo <= data.true_effect <= result.hi


def test_cuped_recovers_true_effect():
    data = generate_rct_data(seed=42)
    result = cuped(data.df, "post_period_spend", "treatment", "pre_period_spend")
    assert result.lo <= data.true_effect <= result.hi


def test_cuped_narrower_than_simple_diff_in_means():
    """Exit test for Q8: CUPED should give a visibly tighter interval than plain
    difference-in-means, without needing to change the point estimate's accuracy -
    the whole value proposition is variance reduction, not bias correction."""
    data = generate_rct_data(seed=42)
    naive = simple_diff_in_means(data.df, "post_period_spend", "treatment")
    adjusted = cuped(data.df, "post_period_spend", "treatment", "pre_period_spend")

    naive_width = naive.hi - naive.lo
    adjusted_width = adjusted.hi - adjusted.lo
    assert adjusted_width < naive_width * 0.9  # at least a 10% tighter interval


def test_variance_reduction_predicts_observed_width_reduction():
    data = generate_rct_data(seed=42)
    naive = simple_diff_in_means(data.df, "post_period_spend", "treatment")
    adjusted = cuped(data.df, "post_period_spend", "treatment", "pre_period_spend")

    vr = variance_reduction(data.df, "post_period_spend", "pre_period_spend")
    expected_width_ratio = (1 - vr) ** 0.5
    observed_width_ratio = (adjusted.hi - adjusted.lo) / (naive.hi - naive.lo)
    assert abs(observed_width_ratio - expected_width_ratio) < 0.05


def test_coverage_across_seeds():
    naive_covered = cuped_covered = 0
    n = 30
    for seed in range(2000, 2000 + n):
        data = generate_rct_data(seed=seed)
        naive = simple_diff_in_means(data.df, "post_period_spend", "treatment")
        adj = cuped(data.df, "post_period_spend", "treatment", "pre_period_spend")
        if naive.lo <= data.true_effect <= naive.hi:
            naive_covered += 1
        if adj.lo <= data.true_effect <= adj.hi:
            cuped_covered += 1

    assert naive_covered / n >= 0.7
    assert cuped_covered / n >= 0.7
