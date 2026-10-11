import numpy as np

from decision_proof.estimators.mmm import MarketingMixModel
from decision_proof.simulate.mmm import adstock, generate_mmm_data, saturate


def test_generate_mmm_data_is_reproducible():
    a = generate_mmm_data(seed=42)
    b = generate_mmm_data(seed=42)
    import pandas.testing as pdt

    pdt.assert_frame_equal(a.df, b.df)


def test_adstock_carries_over_and_decays():
    spend = np.array([100.0, 0.0, 0.0, 0.0])
    out = adstock(spend, decay=0.5)
    assert out[0] == 100.0
    assert out[1] == 50.0
    assert out[2] == 25.0
    assert out[3] == 12.5


def test_saturate_is_increasing_and_bounded():
    adstocked = np.array([0.0, 1000.0, 10_000.0, 100_000.0])
    out = saturate(adstocked, k=5000.0)
    assert out[0] == 0.0
    assert np.all(np.diff(out) > 0)  # monotonically increasing
    assert np.all(out < 1.0)  # bounded below the asymptote


def test_mmm_fit_runs_and_produces_sane_output():
    data = generate_mmm_data(seed=42)
    est = MarketingMixModel().fit(data.df, "revenue", "price", "promo_spend")

    assert 0 <= est.r2 <= 1
    assert est.decay in MarketingMixModel().decay_grid
    assert len(est.fitted_promo_response) == len(data.df)
    assert np.all(est.fitted_promo_response >= 0)


def test_mmm_recovers_price_coef_within_one_se():
    data = generate_mmm_data(seed=42)
    est = MarketingMixModel().fit(data.df, "revenue", "price", "promo_spend")
    assert abs(est.price_coef - data.true_price_coef) < 2 * est.price_coef_se


def test_mmm_recovers_promo_scale_within_one_se():
    data = generate_mmm_data(seed=42)
    est = MarketingMixModel().fit(data.df, "revenue", "price", "promo_spend")
    assert abs(est.promo_scale - data.true_promo_max_response) < 2 * est.promo_scale_se


def test_mmm_decay_grid_search_usually_finds_the_true_decay():
    matches = 0
    n = 15
    for seed in range(2000, 2000 + n):
        data = generate_mmm_data(seed=seed)
        est = MarketingMixModel().fit(data.df, "revenue", "price", "promo_spend")
        if est.decay == data.true_adstock_decay:
            matches += 1
    assert matches / n >= 0.6
