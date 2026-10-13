"""Cached compute for the Advanced Methods page — one function per Q5-Q10
technique, each a pure deterministic function safe for st.cache_data.
"""

from __future__ import annotations

import streamlit as st

SEED = 42


@st.cache_data
def staggered_did_demo() -> dict:
    from decision_proof.estimators.did import DiDEventStudy
    from decision_proof.estimators.staggered_did import StaggeredDiD
    from decision_proof.simulate.scenarios import scenario_s5_heterogeneous

    panel, decision, truth, starts = scenario_s5_heterogeneous(seed=SEED)
    naive = DiDEventStudy().fit(panel, decision)
    staggered = StaggeredDiD().fit(panel, starts)
    return {
        "true_cum_effect": truth.true_cum_effect,
        "naive_cum_effect": naive.cum_effect,
        "naive_cum_lo": naive.cum_lo,
        "naive_cum_hi": naive.cum_hi,
        "staggered_cum_effect": staggered.cum_effect,
        "staggered_cum_lo": staggered.cum_lo,
        "staggered_cum_hi": staggered.cum_hi,
        "cohorts": starts,
    }


@st.cache_data
def rdd_demo() -> dict:
    from decision_proof.estimators.rdd import RegressionDiscontinuity
    from decision_proof.simulate.rdd import generate_rdd_data

    data = generate_rdd_data(n_customers=4000, seed=SEED)
    estimator = RegressionDiscontinuity()
    est = estimator.fit(data.df, "running_var", "next_period_spend", data.cutoff)
    placebos = estimator.placebo_cutoffs(
        data.df, "running_var", "next_period_spend", data.cutoff, offsets=[-40, -20, 20, 40]
    )
    return {
        "true_jump": data.true_jump,
        "cutoff": data.cutoff,
        "effect": est.effect,
        "effect_lo": est.effect_lo,
        "effect_hi": est.effect_hi,
        "bandwidth": est.bandwidth,
        "placebos": {k: (p.effect, p.effect_lo, p.effect_hi) for k, p in placebos.items()},
        "sample_df": data.df.sample(min(600, len(data.df)), random_state=SEED),
    }


@st.cache_data
def iv_demo() -> dict:
    from decision_proof.estimators.iv import InstrumentalVariables, naive_ols
    from decision_proof.simulate.iv import generate_iv_data

    data = generate_iv_data(seed=SEED)
    naive = naive_ols(data.df, "revenue", "price")
    iv = InstrumentalVariables()
    result = iv.fit(data.df, "revenue", "price", "cost_shock")
    f_stat = iv.first_stage_f_stat(data.df, "price", "cost_shock")
    return {
        "true_effect": data.true_price_effect,
        "naive_coef": naive.coef,
        "naive_lo": naive.lo,
        "naive_hi": naive.hi,
        "iv_coef": result.coef,
        "iv_lo": result.lo,
        "iv_hi": result.hi,
        "f_stat": f_stat,
    }


@st.cache_data
def rct_cuped_demo() -> dict:
    from decision_proof.estimators.rct import cuped, simple_diff_in_means, variance_reduction
    from decision_proof.simulate.rct import generate_rct_data

    data = generate_rct_data(seed=SEED)
    naive = simple_diff_in_means(data.df, "post_period_spend", "treatment")
    adjusted = cuped(data.df, "post_period_spend", "treatment", "pre_period_spend")
    vr = variance_reduction(data.df, "post_period_spend", "pre_period_spend")
    return {
        "true_effect": data.true_effect,
        "naive_effect": naive.effect,
        "naive_lo": naive.lo,
        "naive_hi": naive.hi,
        "cuped_effect": adjusted.effect,
        "cuped_lo": adjusted.lo,
        "cuped_hi": adjusted.hi,
        "variance_reduction": vr,
    }


@st.cache_data
def uplift_demo() -> dict:
    from decision_proof.estimators.uplift import TLearner
    from decision_proof.simulate.uplift import generate_uplift_data

    data = generate_uplift_data(seed=SEED)
    fitted = TLearner().fit(data.df, "outcome", "treatment", "loyalty_score")
    segments = fitted.segment_cate(data.df, n_segments=3)
    return {
        "true_base_effect": data.base_effect,
        "true_effect_slope": data.effect_slope,
        "fitted_intercept": fitted.cate_intercept,
        "fitted_slope": fitted.cate_slope,
        "segments": segments.to_dict(orient="records"),
    }


@st.cache_data
def mmm_demo() -> dict:
    from decision_proof.estimators.mmm import MarketingMixModel
    from decision_proof.simulate.mmm import generate_mmm_data

    data = generate_mmm_data(seed=SEED)
    est = MarketingMixModel().fit(data.df, "revenue", "price", "promo_spend")
    return {
        "true_decay": data.true_adstock_decay,
        "true_k": data.true_saturation_k,
        "true_price_coef": data.true_price_coef,
        "true_promo_max": data.true_promo_max_response,
        "fit_decay": est.decay,
        "fit_k": est.saturation_k,
        "fit_price_coef": est.price_coef,
        "fit_price_coef_se": est.price_coef_se,
        "fit_promo_scale": est.promo_scale,
        "fit_promo_scale_se": est.promo_scale_se,
        "r2": est.r2,
        "period": data.df["period"].tolist(),
        "revenue": data.df["revenue"].tolist(),
        "fitted_promo_response": est.fitted_promo_response.tolist(),
    }
