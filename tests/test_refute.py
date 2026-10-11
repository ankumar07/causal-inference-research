from decision_proof.estimators.did import DiDEventStudy
from decision_proof.estimators.synth import SyntheticControl
from decision_proof.refute.verdict import refute
from decision_proof.simulate.scenarios import generate_scenario


def _estimate(name: str, seed: int = 42):
    panel, decision, truth = generate_scenario(name, seed=seed)
    estimates = {
        "did_event_study": DiDEventStudy().fit(panel, decision),
        "synthetic_control": SyntheticControl().fit(panel, decision),
    }
    return panel, decision, truth, estimates


def test_s1_is_board_ready():
    """Exit test for B3: on S1 (full history, real effect), the verdict is BOARD_READY."""
    panel, decision, truth, estimates = _estimate("S1")
    report = refute(panel, decision, estimates)

    assert report.verdict == "BOARD_READY"
    assert all(c.status != "FAIL" for c in report.checks if c.critical)
    assert len(report.reasons) > 0


def test_s3_is_not_proven():
    """Exit test for B3: on S3 (no true effect, seasonal-spike trap), the verdict is NOT_PROVEN."""
    panel, decision, truth, estimates = _estimate("S3")
    report = refute(panel, decision, estimates)

    assert report.verdict == "NOT_PROVEN"
    assert any(c.critical and c.status == "FAIL" for c in report.checks)


def test_s2_is_not_board_ready():
    """S2 (same decision as S1, only 12 weeks of pre-period history) should degrade.

    The README's build plan scripted this as DIRECTIONAL. What the suite actually
    finds is NOT_PROVEN: with 20 donors and only 12 pre-period points, synthetic
    control is underdetermined for every unit (placebo donors included), so
    placebo-in-space loses its discriminating power rather than just widening the
    interval. That's a real finding, not a tuned-to-look-good one (see CLAUDE.md) —
    flagged for a follow-up look in B4 (calibration) or B5, not silently patched here.
    """
    panel, decision, truth, estimates = _estimate("S2")
    report = refute(panel, decision, estimates)

    assert report.verdict != "BOARD_READY"


def test_verdict_report_shape():
    panel, decision, truth, estimates = _estimate("S1")
    report = refute(panel, decision, estimates)

    assert report.verdict in ("BOARD_READY", "DIRECTIONAL", "NOT_PROVEN")
    for check in report.checks:
        assert check.status in ("PASS", "WARN", "FAIL")
        assert check.explanation


def test_methods_agree_requires_overlap_and_sign():
    from decision_proof.contracts import EffectEstimate
    from decision_proof.refute.checks import methods_agree

    def est(lo, hi):
        mid = (lo + hi) / 2
        return EffectEstimate(
            method="x", cum_effect=mid, cum_lo=lo, cum_hi=hi,
            pct_effect=0, pct_lo=0, pct_hi=0, path=[], lag_50=None, lag_90=None, diagnostics={},
        )

    assert methods_agree({"a": est(10, 20), "b": est(15, 25)})
    assert not methods_agree({"a": est(10, 20), "b": est(21, 30)})
    assert not methods_agree({"a": est(-20, -10), "b": est(5, 15)})
