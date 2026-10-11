# Build plan (internal)


### 2.1 Ground rules
- **Only simulated data and my own genericized papers.** No employer data, code, queries or schemas.
- Build on my own laptop, my own time and my own accounts.
- Private GitHub repo; give the CTO access on request.

### 2.2 Scope tiers (cut from the bottom, never from the top)

| Tier | Includes | Why it matters |
|---|---|---|
| **MUST** | M1 simulator, M2 DiD + synthetic control, M3 refutation, M4 verdict, M5 calibration (S1–S4), M9 Evidence and Verdict pages, M8 memo with grounding check | This is the story: estimate, then refute, then make an honest claim |
| **SHOULD** | Conservation-constrained synthetic control, M7 ledger and grading, Lab page, METHODS.md | My original method, plus the attribution and grading bullet in the JD |
| **NICE** | M6 experiment designer, S5 staggered rollout, Bayesian time-series estimator | Covers "design experiments with clients" |
| **STRETCH** | Databricks job, FastAPI, public deployment | Signals only |

### 2.3 Time blocks (about 26 focused hours)

| Block | Hours | Deliverable | Exit test |
|---|---|---|---|
| 0. Setup | 1 | Repo, uv, contracts.py, CLAUDE.md, CI with pytest | `uv run pytest` passes on empty tests |
| 1. Simulator | 3 | DGP + scenarios S1–S4 + ground truth | Plot S1: ramp visible, seeds reproducible |
| 2. Estimators | 4 | DiD event study (wild bootstrap) + synthetic control (placebo inference) | S1 intervals contain the true effect |
| 3. Refutation + verdict | 3 | All checks + verdict rules | S1 = BOARD-READY, S2 = DIRECTIONAL, S3 = NOT PROVEN |
| 4. Calibration lab | 3 | Harness + LAB_REPORT.md | Coverage table generated; **start the long run in the background before Block 5** |
| 5. Conservation-constrained SC + ledger | 3 | Contamination-aware estimator, DuckDB ledger, grading | S4: bias reduced versus plain synthetic control (whatever the lab shows) |
| 6. Claude layer | 3 | Tools, verdict-gated memo, grounding check | A memo with an invented number is rejected in a test |
| 7. Streamlit app | 4 | 4 pages, clean charts | Full demo click-through with no errors |
| 8. Polish + backup | 2 | README, METHODS.md, a recorded 5-minute walkthrough as backup | Fresh clone + quickstart works |

**Feature freeze: Monday 10 pm.** Tuesday morning is for rehearsal only, no new code. Sleep before the interview. The live causal assessment is the real test, and a tired version of me will do worse on it than a smaller demo would cost.

### 2.4 Working with Claude Code
- Put this README in the repo first and tell Claude Code to treat it as the spec.
- Build one module at a time: **tests first** (for example "write tests that S1 estimates contain the truth"), then the implementation.
- Keep `CLAUDE.md` short: Python 3.11, type hints, pydantic contracts, no notebooks in `src/`, every estimator returns `EffectEstimate`, no network calls outside `agent/`.
- Read and understand every line I merge. The JD says "you own every line," and I will be asked about the internals.

### 2.5 Demo script (5–7 minutes)
1. **Framing (30 s):** "AI recommendations stall at the board because nobody can defend the cause. This is a small version of the causal layer: estimate, refute, then claim."
2. **S1 (90 s):** Larkfield price +6%. Event study with the ramp, synthetic control fit, effect range with lag. Verdict: BOARD-READY.
3. **Where it breaks (90 s):** the same decision with 12 weeks of history drops to DIRECTIONAL, with the reasons shown. Then S3: a seasonal spike with no true effect is NOT PROVEN. *"Saying no is the feature."*
4. **The lab (60 s):** coverage table and the S4 contamination result with the conservation-constrained synthetic control. "This is how I'd hold the lab's methods to a standard."
5. **Claude memo (45 s):** the CFO paragraph, plus the grounding check rejecting an invented number.
6. **Ledger (30 s):** the predicted range now, graded when the outcome lands.
7. **Close (30 s):** what I'd do in the first 90 days at Unloq (below).

### 2.6 How the prototype maps to the JD

| JD line | Where it shows up |
|---|---|
| Map drivers, estimate effects from natural experiments, refute every estimate | M2 + M3 + M4 |
| Return effects as honest ranges, with how much and what lag | `EffectEstimate` intervals + `lag_50`/`lag_90` + M5 coverage |
| Design experiments with clients ("if we do X, here's the range") | M6 |
| Own the attribution method; grade each call against what really happened | M7 |
| Raise the bar on evaluation; write up the lab's methods | M5 + LAB_REPORT.md + METHODS.md |
| Real LLM depth: tool use beyond a demo | M8 tool use + grounding check + verdict gating |
| Explain cause and effect to a CFO | Memo + verdict reasons |
| Knows when each method breaks on short histories | S2 + window-sensitivity check + lab |

### 2.7 First-90-days talking points
1. **Weeks 1–4:** audit what clients' data actually looks like (history length, granularity, how much contamination), and set a minimum data standard for each method.
2. **Weeks 4–8:** ship the refutation and verdict layer behind the prescriptive engine, so nothing goes out to a client without passing it.
3. **Weeks 8–12:** start the decision ledger with live clients, publish the first internal methods note, and design the first client experiment.

### 2.8 Interview-day checklist
- [ ] Fresh clone + quickstart tested on a clean environment
- [ ] App runs locally; backup recording ready if the screen-share fails
- [ ] Lab report numbers memorised, especially where each method fails
- [ ] Able to explain the wild bootstrap, placebo inference and the conservation constraint without notes
- [ ] Revision done on the gaps: uplift/CATE meta-learners, MMM (adstock and saturation), power/MDE, CUPED
- [ ] Questions for the CTO ready: runway, client data reality, team, equity, publishing rights
