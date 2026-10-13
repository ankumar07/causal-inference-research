import plotly.graph_objects as go
import streamlit as st

from decision_proof.app.methods_common import (
    iv_demo,
    mmm_demo,
    rct_cuped_demo,
    rdd_demo,
    staggered_did_demo,
    uplift_demo,
)

st.set_page_config(page_title="Advanced Methods — Decision Proof", page_icon="🧬", layout="wide")
st.title("Advanced Methods")
st.caption(
    "Each quarter's price-change batch (previous page) uses the same estimate → refute "
    "workflow throughout. Real pricing decisions don't always fit that shape — these six "
    "cover the situations that call for a different causal technique."
)

tabs = st.tabs(
    ["Staggered DiD", "Regression discontinuity", "Instrumental variables", "RCT + CUPED", "Uplift / CATE", "Marketing mix"]
)

with tabs[0]:
    st.subheader("Staggered-adoption DiD")
    st.caption("A multi-wave rollout where the effect size itself grows wave over wave.")
    with st.spinner("Fitting (wild bootstrap)..."):
        d = staggered_did_demo()
    st.warning(
        "Honest result, not a clean win: on a 120-seed comparison, naive TWFE's mean absolute "
        "bias actually beat this estimator's. See docs/METHODS.md for the full finding — "
        "reported as-is rather than tuned to look better."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("True cumulative effect", f"${d['true_cum_effect']:,.0f}")
    c2.metric("Naive TWFE", f"${d['naive_cum_effect']:,.0f}")
    c2.caption(f"[{d['naive_cum_lo']:,.0f}, {d['naive_cum_hi']:,.0f}]")
    c3.metric("Staggered-robust", f"${d['staggered_cum_effect']:,.0f}")
    c3.caption(f"[{d['staggered_cum_lo']:,.0f}, {d['staggered_cum_hi']:,.0f}]")

with tabs[1]:
    st.subheader("Regression discontinuity")
    st.caption("A loyalty-tier price threshold: does crossing it cause a jump in future spend?")
    d = rdd_demo()
    c1, c2 = st.columns(2)
    c1.metric("True jump at cutoff", f"${d['true_jump']:.2f}")
    c1.metric("Estimated jump", f"${d['effect']:.2f}")
    c1.caption(f"90% interval: \\${d['effect_lo']:.2f} to \\${d['effect_hi']:.2f}  (bandwidth ±\\${d['bandwidth']:.0f})")

    with c2:
        st.markdown("**Placebo cutoffs** (should show no effect)")
        for fake_cutoff, (eff, lo, hi) in d["placebos"].items():
            sig = not (lo <= 0 <= hi)
            st.caption(f"at {fake_cutoff:.0f}: {eff:.2f} [{lo:.2f}, {hi:.2f}] " + ("⚠️ significant" if sig else "✓ null"))

    fig = go.Figure()
    sample = d["sample_df"]
    fig.add_trace(
        go.Scatter(x=sample["running_var"], y=sample["next_period_spend"], mode="markers", marker=dict(size=4, opacity=0.4), name="customers")
    )
    fig.add_vline(x=d["cutoff"], line_dash="dot", line_color="gray")
    fig.update_layout(xaxis_title="trailing 90-day spend (running variable)", yaxis_title="next-period spend", height=350, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

with tabs[2]:
    st.subheader("Instrumental variables (2SLS)")
    st.caption("Cost shocks move price for reasons unrelated to local demand — an instrument for the confounded effect of price on revenue.")
    d = iv_demo()
    c1, c2, c3 = st.columns(3)
    c1.metric("True price effect", f"${d['true_effect']:,.0f}")
    c2.metric("Naive OLS (confounded)", f"${d['naive_coef']:,.0f}")
    c2.caption(f"[{d['naive_lo']:,.0f}, {d['naive_hi']:,.0f}]")
    c3.metric("2SLS", f"${d['iv_coef']:,.0f}")
    c3.caption(f"[{d['iv_lo']:,.0f}, {d['iv_hi']:,.0f}]")
    st.caption(f"First-stage F-statistic: {d['f_stat']:.0f} (Staiger-Stock rule of thumb wants > 10)")

with tabs[3]:
    st.subheader("RCT analysis + CUPED")
    st.caption("A true randomized price test — simple difference-in-means is already unbiased here. CUPED tightens the interval using pre-period spend.")
    d = rct_cuped_demo()
    c1, c2, c3 = st.columns(3)
    c1.metric("True effect", f"${d['true_effect']:.2f}")
    c2.metric("Difference-in-means", f"${d['naive_effect']:.2f}")
    c2.caption(f"[{d['naive_lo']:.2f}, {d['naive_hi']:.2f}]  width={d['naive_hi']-d['naive_lo']:.2f}")
    c3.metric("CUPED-adjusted", f"${d['cuped_effect']:.2f}")
    c3.caption(f"[{d['cuped_lo']:.2f}, {d['cuped_hi']:.2f}]  width={d['cuped_hi']-d['cuped_lo']:.2f}")
    width_reduction = 1 - (d["cuped_hi"] - d["cuped_lo"]) / (d["naive_hi"] - d["naive_lo"])
    st.caption(f"Variance reduction from the pre-period covariate: {d['variance_reduction']:.0%} → {width_reduction:.0%} tighter interval")

with tabs[4]:
    st.subheader("Uplift modeling / CATE (T-learner)")
    st.caption("Randomized price test again, but the treatment effect itself varies by customer loyalty — which segment is most price-sensitive?")
    d = uplift_demo()
    st.caption(
        f"Fitted CATE slope {d['fitted_slope']:.3f} vs. true {d['true_effect_slope']:.3f} "
        f"(effect $ per point of loyalty score)"
    )
    fig = go.Figure(
        go.Bar(
            x=[s["segment"] for s in d["segments"]],
            y=[s["cate"] for s in d["segments"]],
            marker_color="#1f77b4",
        )
    )
    fig.update_layout(xaxis_title="loyalty-score tier", yaxis_title="average CATE ($)", height=350, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

with tabs[5]:
    st.subheader("Marketing mix model")
    st.caption("Price + promo spend together — promo carries over (adstock) and saturates (diminishing returns).")
    d = mmm_demo()
    c1, c2 = st.columns(2)
    c1.metric("Adstock decay (grid search)", f"{d['fit_decay']:.2f}", help=f"true: {d['true_decay']:.2f}")
    c1.metric("Price coefficient", f"${d['fit_price_coef']:,.0f}")
    c1.caption(f"±{d['fit_price_coef_se']:,.0f} SE (true ${d['true_price_coef']:,.0f})")
    c2.metric("R²", f"{d['r2']:.2f}")
    c2.metric("Promo scale (max response)", f"${d['fit_promo_scale']:,.0f}")
    c2.caption(f"±{d['fit_promo_scale_se']:,.0f} SE (true ${d['true_promo_max']:,.0f})")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["period"], y=d["revenue"], mode="lines", name="revenue", line=dict(color="#1f77b4")))
    fig.add_trace(
        go.Scatter(x=d["period"], y=d["fitted_promo_response"], mode="lines", name="fitted promo contribution", line=dict(color="#ff7f0e"))
    )
    fig.update_layout(xaxis_title="week", yaxis_title="$", height=350, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
