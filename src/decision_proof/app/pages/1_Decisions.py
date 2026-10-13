import streamlit as st

from decision_proof.app.common import ensure_demo_seeded, get_tools

st.set_page_config(page_title="Decisions — Decision Proof", page_icon="📋", layout="wide")
st.title("Decisions")

tools = get_tools()
ensure_demo_seeded(tools)

df = tools.ledger.list_decisions()

if df.empty:
    st.info("No decisions logged yet.")
    st.stop()

display = df[
    ["decision_id", "description", "method", "verdict", "status", "pred_cum_lo", "pred_cum_effect", "pred_cum_hi"]
].rename(
    columns={
        "pred_cum_lo": "predicted lo ($)",
        "pred_cum_effect": "predicted effect ($)",
        "pred_cum_hi": "predicted hi ($)",
    }
)
st.dataframe(display, width="stretch", hide_index=True)

st.subheader("Analyze a decision")
decision_id = st.selectbox("Decision", df["decision_id"].tolist())
row = df[df["decision_id"] == decision_id].iloc[0]
st.caption(row["description"])

col1, col2 = st.columns(2)
with col1:
    if st.button("Evidence →", type="primary", width="stretch"):
        st.session_state.selected_decision_id = decision_id
        st.switch_page("pages/2_Evidence.py")
with col2:
    if st.button("Verdict & memo →", width="stretch"):
        st.session_state.selected_decision_id = decision_id
        st.switch_page("pages/3_Verdict_and_Memo.py")
