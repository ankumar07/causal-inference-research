import pytest

from decision_proof.agent.grounding import check_grounding, extract_numbers, grounded_values
from decision_proof.agent.memo import MemoRejected, draft_memo
from decision_proof.estimators.synth import SyntheticControl
from decision_proof.refute.verdict import refute
from decision_proof.simulate.scenarios import generate_scenario


class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self._responses.pop(0) if self._responses else self._responses[-1]
        return _FakeResponse(text)


class FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


@pytest.fixture
def s1_analysis():
    panel, decision, truth = generate_scenario("S1", seed=42)
    estimate = SyntheticControl().fit(panel, decision)
    report = refute(panel, decision, {"synthetic_control": estimate})
    return decision, estimate, report


def test_extract_numbers_handles_dollars_percents_and_negatives():
    text = "Revenue rose $1,234 (12.5%), while the prior month was -$500 ($(200) adjusted)."
    numbers = extract_numbers(text)
    assert 1234.0 in numbers
    assert 12.5 in numbers
    assert -500.0 in numbers


def test_check_grounding_passes_a_grounded_memo(s1_analysis):
    decision, estimate, report = s1_analysis
    allowed = grounded_values(decision, estimate, report)
    memo = (
        f"We estimate ${estimate.cum_effect:,.0f} in incremental revenue "
        f"(90% interval: ${estimate.cum_lo:,.0f} to ${estimate.cum_hi:,.0f}), "
        f"a {estimate.pct_effect:.1f}% lift across {len(decision.treated_units)} metro markets."
    )
    ok, ungrounded = check_grounding(memo, allowed)
    assert ok
    assert ungrounded == []


def test_check_grounding_rejects_an_invented_number(s1_analysis):
    decision, estimate, report = s1_analysis
    allowed = grounded_values(decision, estimate, report)
    memo = f"We estimate ${estimate.cum_effect:,.0f}, and expect $999,999,999 next quarter."
    ok, ungrounded = check_grounding(memo, allowed)
    assert not ok
    assert 999_999_999.0 in ungrounded


def test_draft_memo_accepts_a_grounded_response_on_first_try(s1_analysis):
    decision, estimate, report = s1_analysis
    memo_text = (
        f"The price change drove ${estimate.cum_effect:,.0f} in incremental revenue "
        f"(90% interval ${estimate.cum_lo:,.0f} to ${estimate.cum_hi:,.0f})."
    )
    client = FakeClient(responses=[memo_text])

    result = draft_memo(decision, estimate, report, model="test-model", client=client)

    assert result.attempts == 1
    assert result.text == memo_text
    assert len(client.messages.calls) == 1


def test_draft_memo_retries_then_accepts_a_corrected_response(s1_analysis):
    decision, estimate, report = s1_analysis
    bad = "We expect this to generate $42,000,000 in pure profit."
    good = (
        f"The price change drove ${estimate.cum_effect:,.0f} in incremental revenue "
        f"(90% interval ${estimate.cum_lo:,.0f} to ${estimate.cum_hi:,.0f})."
    )
    client = FakeClient(responses=[bad, good])

    result = draft_memo(decision, estimate, report, model="test-model", client=client)

    assert result.attempts == 2
    assert result.text == good
    assert len(client.messages.calls) == 2
    # the retry prompt should tell the model what went wrong
    assert "42000000" in client.messages.calls[1]["messages"][0]["content"].replace(",", "").replace(".0", "") \
        or "42,000,000" in client.messages.calls[1]["messages"][0]["content"]


def test_draft_memo_rejects_after_max_attempts_when_never_corrected(s1_analysis):
    """Exit test for B7: a memo with an invented number is rejected."""
    decision, estimate, report = s1_analysis
    always_bad = "We're confident this will generate $42,000,000 in pure profit, guaranteed."
    client = FakeClient(responses=[always_bad, always_bad, always_bad, always_bad])

    with pytest.raises(MemoRejected):
        draft_memo(decision, estimate, report, model="test-model", client=client)


def test_not_proven_uses_the_not_proven_system_prompt():
    panel, decision, truth = generate_scenario("S3", seed=42)
    estimate = SyntheticControl().fit(panel, decision)
    report = refute(panel, decision, {"synthetic_control": estimate})
    assert report.verdict == "NOT_PROVEN"

    memo = "We can't yet attribute this to the decision; more history would help settle it."
    client = FakeClient(responses=[memo])

    draft_memo(decision, estimate, report, model="test-model", client=client)

    system_prompt = client.messages.calls[0]["system"]
    assert "NOT_PROVEN" in system_prompt
    assert "can't yet attribute" in system_prompt
