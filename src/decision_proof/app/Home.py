import streamlit as st

from decision_proof.app.common import ensure_demo_seeded, get_tools

st.set_page_config(page_title="Decision Proof", page_icon="☕", layout="wide")

st.title("Decision Proof")
st.caption(
    "A causal verification layer for business decisions — modeled on Daybreak Coffee, "
    "a fictional multi-state drive-thru chain (structurally based on a real public chain's "
    "footprint, no proprietary data)."
)

tools = get_tools()
with st.spinner("Loading demo scenarios..."):
    ensure_demo_seeded(tools)

st.markdown(
    """
Forecasts say what will probably happen. Dashboards say what did happen. Neither says *why* —
and *why* is what a board needs before it approves the next decision. Decision Proof estimates
a decision's causal effect with several methods, tries hard to break each estimate, and only
then states a result.

> **The core principle:** an estimate that hasn't survived an attempt to refute it is not a
> result. Decision Proof is designed to say "not proven" when that is the honest answer.

### Five scenarios are pre-loaded

| Scenario | What it shows |
|---|---|
| **S1** | Price +6%, 60 weeks of history — the clean case: BOARD-READY |
| **S2** | Same decision, 12 weeks of history — short-history degradation |
| **S3** | A promo timed to a seasonal spike, no true effect — a false positive the system catches |
| **S4** | Price +10%, elastic demand, with spillover contamination — conservation-constrained SC corrects for it |
| **S5** | A staggered rollout across 6 metros over 8 weeks |

**Start on the Decisions page** (sidebar) to pick a scenario and walk the full
estimate → refute → verdict pipeline.
"""
)
