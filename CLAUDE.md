# Decision Proof: conventions

README.md is the spec.

- Python 3.11, type hints everywhere, pydantic contracts in `contracts.py`.
- Every estimator exposes `fit(panel, decision) -> EffectEstimate`.
- Simulated data only. No employer data, code, queries or schemas.
- No network calls outside `agent/`. API key via `ANTHROPIC_API_KEY`, model via `ANTHROPIC_MODEL`; never in code.
- No notebooks in `src/`.
- Tests first: write the exit test for a module, then the implementation.
- Everything is seeded and reproducible.
- Report calibration numbers as they come out; never tune to look good.
- Verdict thresholds are judgment calls: configurable and documented in `docs/METHODS.md`.
