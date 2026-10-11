from decision_proof.contracts import Decision, EffectEstimate


def test_contracts_roundtrip():
    d = Decision(
        decision_id="d1",
        description="Price +6% on 4 North regions",
        treated_units=["r1", "r2", "r3", "r4"],
        start_period=61,
        outcome="revenue",
    )
    assert d.owner is None
    e = EffectEstimate(
        method="did", cum_effect=1.0, cum_lo=0.5, cum_hi=1.5,
        pct_effect=2.0, pct_lo=1.0, pct_hi=3.0,
        path=[{"h": 0, "est": 0.1, "lo": 0.0, "hi": 0.2}],
        lag_50=2, lag_90=4, diagnostics={},
    )
    assert e.cum_lo <= e.cum_effect <= e.cum_hi
