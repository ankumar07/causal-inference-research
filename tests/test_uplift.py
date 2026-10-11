import numpy as np

from decision_proof.estimators.uplift import TLearner
from decision_proof.simulate.uplift import generate_uplift_data


def test_generate_uplift_data_is_reproducible():
    a = generate_uplift_data(seed=42)
    b = generate_uplift_data(seed=42)
    import pandas.testing as pdt

    pdt.assert_frame_equal(a.df, b.df)


def test_t_learner_recovers_the_true_heterogeneity_slope():
    """Exit test for Q9: the fitted CATE slope (how much the treatment effect
    grows with loyalty_score) should be close to the true slope baked into the
    simulator."""
    data = generate_uplift_data(seed=42)
    fitted = TLearner().fit(data.df, "outcome", "treatment", "loyalty_score")

    assert abs(fitted.cate_slope - data.effect_slope) < 0.05


def test_t_learner_slope_is_unbiased_across_seeds():
    slopes = []
    for seed in range(1000, 1030):
        data = generate_uplift_data(seed=seed)
        fitted = TLearner().fit(data.df, "outcome", "treatment", "loyalty_score")
        slopes.append(fitted.cate_slope)

    mean_slope = np.mean(slopes)
    assert abs(mean_slope - 0.08) < 0.02
    assert all(s > 0 for s in slopes)  # correct sign every time


def test_segment_cate_is_monotonically_increasing():
    """The business question Q9 answers: which segment is most price-sensitive.
    With a positive true slope, average CATE should increase tier over tier."""
    data = generate_uplift_data(seed=42)
    fitted = TLearner().fit(data.df, "outcome", "treatment", "loyalty_score")
    segments = fitted.segment_cate(data.df, n_segments=3)

    cates = segments["cate"].to_numpy()
    assert len(cates) == 3
    assert cates[0] < cates[1] < cates[2]


def test_predict_cate_matches_true_cate_reasonably_well():
    data = generate_uplift_data(seed=42)
    fitted = TLearner().fit(data.df, "outcome", "treatment", "loyalty_score")

    test_x = np.array([10.0, 50.0, 90.0])
    predicted = fitted.predict_cate(test_x)
    true = data.true_cate(test_x)

    assert np.all(np.abs(predicted - true) < 3.0)
