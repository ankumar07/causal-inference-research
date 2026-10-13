import pytest

from decision_proof.contracts import EffectEstimate, RefutationReport
from decision_proof.design.sensitivity import fit_price_sensitivity


def _estimate(pct_effect, pct_lo, pct_hi):
    return EffectEstimate(
        method="x", cum_effect=0, cum_lo=0, cum_hi=0,
        pct_effect=pct_effect, pct_lo=pct_lo, pct_hi=pct_hi,
        path=[], lag_50=None, lag_90=None, diagnostics={},
    )


def _report(verdict):
    return RefutationReport(checks=[], verdict=verdict, reasons=["test"])


def test_fit_recovers_a_clean_linear_relationship():
    # effect_pct = 0.5 * price_pct_points, noiseless, all BOARD_READY
    experiments = [
        (0.03, _estimate(1.5, 1.0, 2.0), _report("BOARD_READY"), "p3"),
        (0.06, _estimate(3.0, 2.5, 3.5), _report("BOARD_READY"), "p6"),
        (0.09, _estimate(4.5, 4.0, 5.0), _report("BOARD_READY"), "p9"),
        (0.12, _estimate(6.0, 5.5, 6.5), _report("BOARD_READY"), "p12"),
    ]
    result = fit_price_sensitivity(experiments)

    assert result.is_fitted
    assert result.slope == pytest.approx(0.5, abs=0.01)
    assert result.slope_lo <= 0.5 <= result.slope_hi
    assert result.n_points == 4
    assert result.n_excluded_not_proven == 0


def test_not_proven_experiments_are_excluded_from_the_fit_but_kept_in_points():
    experiments = [
        (0.03, _estimate(100, 50, 150), _report("NOT_PROVEN"), "p3"),  # wild outlier, should be ignored
        (0.06, _estimate(3.0, 2.5, 3.5), _report("BOARD_READY"), "p6"),
        (0.09, _estimate(4.5, 4.0, 5.0), _report("DIRECTIONAL"), "p9"),
    ]
    result = fit_price_sensitivity(experiments)

    assert result.n_points == 2
    assert result.n_excluded_not_proven == 1
    assert len(result.points) == 3
    assert result.slope < 10  # not dragged toward the excluded outlier


def test_fewer_than_two_surviving_points_returns_unfitted_result():
    experiments = [
        (0.03, _estimate(1.0, 0.5, 1.5), _report("NOT_PROVEN"), "p3"),
        (0.06, _estimate(2.0, 1.5, 2.5), _report("BOARD_READY"), "p6"),
    ]
    result = fit_price_sensitivity(experiments)

    assert not result.is_fitted
    assert result.n_points == 1


def test_no_experiments_at_all():
    result = fit_price_sensitivity([])
    assert not result.is_fitted
    assert result.points == []
