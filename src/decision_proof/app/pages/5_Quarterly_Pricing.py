import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from decision_proof.app.quarterly_common import run_all_quarter_experiments
from decision_proof.design.sensitivity import fit_price_sensitivity_from_rows
from decision_proof.simulate.quarterly import EXPERIMENT_QUARTERS

st.set_page_config(page_title="Quarterly Pricing — Decision Proof", page_icon="📅", layout="wide")
st.title("Quarterly Pricing")
st.caption(
    "Daybreak Coffee runs a batch of price-change experiments every quarter, each at a "
    "different magnitude. Pick one quarter, or several, to see every experiment run and "
    "the price sensitivity range fitted across them."
)

with st.spinner("Running this quarter's experiments..."):
    all_rows = run_all_quarter_experiments()

quarters_available = EXPERIMENT_QUARTERS
selected = st.multiselect(
    "Quarters", options=quarters_available, default=quarters_available, format_func=lambda q: f"Q{q}"
)

rows = [r for r in all_rows if r["quarter"] in selected]
if not rows:
    st.info("Pick at least one quarter.")
    st.stop()

st.subheader("Experiments")
display = pd.DataFrame(rows)[
    [
        "quarter",
        "decision_id",
        "price_change_pct",
        "verdict",
        "pct_effect",
        "pct_lo",
        "pct_hi",
        "dominant_age_band",
        "dominant_income_tier",
    ]
].rename(
    columns={
        "price_change_pct": "price change",
        "pct_effect": "revenue effect (%)",
        "pct_lo": "90% lo (%)",
        "pct_hi": "90% hi (%)",
        "dominant_age_band": "age band",
        "dominant_income_tier": "income tier",
    }
)
display["price change"] = display["price change"].map(lambda p: f"+{p:.0%}")
st.dataframe(display, width="stretch", hide_index=True)

st.subheader("Price sensitivity")
result = fit_price_sensitivity_from_rows(rows)

col1, col2 = st.columns([1, 2])

with col1:
    if result.is_fitted:
        st.metric("Sensitivity (revenue % per 1pp price change)", f"{result.slope:.2f}")
        st.caption(f"90% interval: {result.slope_lo:.2f} to {result.slope_hi:.2f}")
        st.caption(
            f"Fit on {result.n_points} of {len(rows)} experiments — "
            f"{result.n_excluded_not_proven} excluded for coming back NOT_PROVEN."
        )
        if result.n_points < 4:
            st.warning(
                "Few surviving points — select more quarters to pool more evidence and "
                "tighten this estimate."
            )
    else:
        st.info(
            f"Only {result.n_points} experiment(s) survived refutation — need at least 2 "
            "to fit a sensitivity line. Select more quarters."
        )

with col2:
    fig = go.Figure()
    proven = [p for p in result.points if p.verdict != "NOT_PROVEN"]
    not_proven = [p for p in result.points if p.verdict == "NOT_PROVEN"]

    if proven:
        fig.add_trace(
            go.Scatter(
                x=[p.price_change_pct * 100 for p in proven],
                y=[p.effect_pct for p in proven],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[p.effect_pct_hi - p.effect_pct for p in proven],
                    arrayminus=[p.effect_pct - p.effect_pct_lo for p in proven],
                ),
                mode="markers",
                name="used in fit",
                marker=dict(color="#1f77b4", size=10),
            )
        )
    if not_proven:
        fig.add_trace(
            go.Scatter(
                x=[p.price_change_pct * 100 for p in not_proven],
                y=[p.effect_pct for p in not_proven],
                mode="markers",
                name="excluded (NOT_PROVEN)",
                marker=dict(color="lightgray", size=10, symbol="x"),
            )
        )
    if result.is_fitted:
        xs = [p.price_change_pct * 100 for p in result.points]
        x_line = [min(xs), max(xs)]
        y_line = [result.slope * x + result.intercept for x in x_line]
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="fitted sensitivity", line=dict(color="#ff7f0e")))

    fig.update_layout(
        xaxis_title="price change (%)", yaxis_title="revenue effect (%)", height=400, margin=dict(t=10)
    )
    st.plotly_chart(fig, width="stretch")
