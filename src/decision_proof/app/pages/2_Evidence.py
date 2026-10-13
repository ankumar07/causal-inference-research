import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from decision_proof.app.common import analyze_decision, get_tools

st.set_page_config(page_title="Evidence — Decision Proof", page_icon="🔎", layout="wide")
st.title("Evidence")

tools = get_tools()
decision_id = st.session_state.get("selected_decision_id")
if not decision_id or decision_id not in tools.panels:
    st.warning("Pick a decision on the Decisions page first.")
    st.page_link("pages/1_Decisions.py", label="← Back to Decisions")
    st.stop()

panel, decision = tools.panels[decision_id]
with st.spinner("Running estimators..."):
    estimates, report = analyze_decision(tools, panel, decision)

st.subheader(decision.description)
did_est = estimates["did_event_study"]
sc_est = estimates["synthetic_control"]

col1, col2 = st.columns(2)

with col1:
    st.markdown("**DiD event study** (effect by week since the decision, 90% CI)")
    hs = [p["h"] for p in did_est.path]
    ests = [p["est"] for p in did_est.path]
    los = [p["lo"] for p in did_est.path]
    his = [p["hi"] for p in did_est.path]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hs + hs[::-1], y=his + los[::-1], fill="toself", fillcolor="rgba(31,119,180,0.2)",
            line=dict(width=0), name="90% CI", showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=hs, y=ests, mode="lines+markers", name="effect", line=dict(color="#1f77b4")))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(xaxis_title="weeks since decision", yaxis_title="$ effect", height=380, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

with col2:
    st.markdown("**Synthetic control fit** (first treated unit vs. its synthetic twin)")
    unit = decision.treated_units[0]
    weights = sc_est.diagnostics["donor_weights"].get(unit, {})
    wide = panel.df.pivot(index="period", columns="unit_id", values=panel.outcome)
    actual = wide[unit]
    synthetic = sum(w * wide[d] for d, w in weights.items()) if weights else pd.Series(0, index=wide.index)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=wide.index, y=actual, mode="lines", name=f"{unit} (actual)", line=dict(color="#1f77b4")))
    fig2.add_trace(go.Scatter(x=wide.index, y=synthetic, mode="lines", name="synthetic", line=dict(color="#ff7f0e", dash="dash")))
    fig2.add_vline(x=decision.start_period, line_dash="dot", line_color="gray")
    fig2.update_layout(xaxis_title="period (week)", yaxis_title=panel.outcome, height=380, margin=dict(t=10))
    st.plotly_chart(fig2, width="stretch")

col3, col4 = st.columns(2)

with col3:
    st.markdown("**Donor weights** (first treated unit)")
    if weights:
        donors = sorted(weights.items(), key=lambda kv: -kv[1])
        fig3 = go.Figure(go.Bar(x=[w for _, w in donors], y=[d for d, _ in donors], orientation="h"))
        fig3.update_layout(xaxis_title="weight", height=max(220, 30 * len(donors)), margin=dict(t=10))
        st.plotly_chart(fig3, width="stretch")
    else:
        st.info("No donor weights recorded.")

with col4:
    st.markdown("**Placebo-in-space distribution**")
    ratios = sc_est.diagnostics["placebo_rmspe_ratios"]
    donor_ratios = [r for r in ratios["donors"].values() if r != float("inf")]
    real_ratio = ratios["treated"].get(unit)
    fig4 = go.Figure(go.Histogram(x=donor_ratios, nbinsx=15, name="placebo donors"))
    if real_ratio is not None and real_ratio != float("inf"):
        fig4.add_vline(x=real_ratio, line_color="red", line_width=2, annotation_text=f"{unit} (real)")
    fig4.update_layout(xaxis_title="post/pre RMSPE ratio", yaxis_title="count", height=320, margin=dict(t=10))
    st.plotly_chart(fig4, width="stretch")

st.subheader("Refutation checklist")
checks_df = pd.DataFrame(
    [
        {"check": c.name, "critical": c.critical, "status": c.status, "explanation": c.explanation}
        for c in report.checks
    ]
)
st.dataframe(checks_df, width="stretch", hide_index=True)
st.caption(f"Verdict: **{report.verdict}** — " + " ".join(report.reasons))
