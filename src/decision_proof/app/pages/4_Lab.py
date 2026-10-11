import os
import re

import streamlit as st

from decision_proof.app.common import get_tools

st.set_page_config(page_title="Lab — Decision Proof", page_icon="🧪", layout="wide")
st.title("Lab")

st.subheader("Calibration")
report_path = "docs/LAB_REPORT.md"
if os.path.exists(report_path):
    with open(report_path) as f:
        text = f.read()
    image_match = re.search(r"!\[.*?\]\((.+?)\)", text)
    if image_match:
        image_path = os.path.join(os.path.dirname(report_path), image_match.group(1))
        if os.path.exists(image_path):
            st.image(image_path)
        text = text[: image_match.start()] + text[image_match.end() :]
    st.markdown(text)
else:
    st.info("Run `uv run python -m decision_proof.calibrate.harness --n 200` to generate the lab report.")

st.divider()
st.subheader("Experiment designer")
tools = get_tools()
decision_id = st.session_state.get("selected_decision_id")
if decision_id and decision_id in tools.panels:
    panel, decision = tools.panels[decision_id]
    st.caption(f"Using pre-period noise from: {decision.description}")
    weeks = st.slider("Weeks of data", 4, 30, 12)
    candidate_pct = st.slider("Candidate effect (%)", 1.0, 15.0, 6.0, step=0.5)
    if st.button("Design experiment"):
        result = tools.call(
            "design_experiment",
            {"decision_id": decision_id, "candidate_effect_pct": candidate_pct, "weeks": weeks},
        )
        c1, c2 = st.columns(2)
        c1.metric("Minimum detectable effect", f"{result['minimum_detectable_effect_pct']:.1f}%")
        c2.metric(
            f"{candidate_pct:.1f}% effect detectable?",
            "Likely" if result["candidate_likely_detectable"] else "Unlikely",
        )
        st.caption(result["method"])
else:
    st.info("Pick a decision on the Decisions page first to use its historical noise for a power calculation.")

st.divider()
st.subheader("Ledger track record")
track = tools.ledger.track_record()
if track.empty:
    st.info("No graded decisions yet.")
else:
    display = track[
        ["decision_id", "realized_cum_effect", "inside_range", "interval_score", "running_share_inside"]
    ]
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        f"Share of graded decisions whose actual outcome fell inside the predicted range: "
        f"{track['running_share_inside'].iloc[-1]:.0%}"
    )
