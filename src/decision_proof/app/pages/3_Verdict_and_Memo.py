import os

import streamlit as st

from decision_proof.agent.memo import MemoRejected, draft_memo
from decision_proof.app.common import analyze_decision, get_tools

st.set_page_config(page_title="Verdict & memo — Decision Proof", page_icon="✅", layout="wide")
st.title("Verdict & memo")

tools = get_tools()
decision_id = st.session_state.get("selected_decision_id")
if not decision_id or decision_id not in tools.panels:
    st.warning("Pick a decision on the Decisions page first.")
    st.page_link("pages/1_Decisions.py", label="← Back to Decisions")
    st.stop()

panel, decision = tools.panels[decision_id]
with st.spinner("Running estimators..."):
    estimates, report = analyze_decision(tools, panel, decision)
primary = estimates["synthetic_control"]

st.subheader(decision.description)

VERDICT_STYLE = {
    "BOARD_READY": ("🟢", "BOARD-READY"),
    "DIRECTIONAL": ("🟡", "DIRECTIONAL"),
    "NOT_PROVEN": ("🔴", "NOT PROVEN"),
}
icon, label = VERDICT_STYLE[report.verdict]
st.markdown(f"## {icon} {label}")
st.caption(" ".join(report.reasons))

col1, col2, col3 = st.columns(3)
col1.metric("Cumulative effect", f"${primary.cum_effect:,.0f}")
col1.caption(f"90% interval: \\${primary.cum_lo:,.0f} to \\${primary.cum_hi:,.0f}")
col2.metric("Percent effect", f"{primary.pct_effect:.1f}%")
col2.caption(f"90% interval: {primary.pct_lo:.1f}% to {primary.pct_hi:.1f}%")
if primary.lag_50 is not None:
    col3.metric("Lag to 50% / 90% of effect", f"{primary.lag_50} / {primary.lag_90} weeks")
else:
    col3.metric("Lag to 50% / 90% of effect", "n/a")

st.divider()
st.subheader("Claude memo")

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.info(
        "Set `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL`) in your environment "
        "to draft a memo. The memo is verdict-gated and every number in it is checked "
        "against the analysis above before it's shown — an invented number gets the "
        "draft rejected and retried, not shipped."
    )
else:
    if st.button("Draft memo", type="primary"):
        with st.spinner("Drafting..."):
            try:
                result = draft_memo(decision, primary, report)
                st.success(f"Grounding check passed on attempt {result.attempts}/3")
                # escape $ before display: the memo text routinely contains several dollar
                # amounts, and a bare "$...$" pair renders as LaTeX math, not literal text.
                st.write(result.text.replace("$", "\\$"))
            except MemoRejected as e:
                st.error(f"Memo rejected: {e}")
