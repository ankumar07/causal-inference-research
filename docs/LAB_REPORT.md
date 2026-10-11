# Lab report

200 replications per scenario per method, 3000 total fits, 74s wall clock.
Regenerate with `uv run python -m decision_proof.calibrate.harness --n 200`.

Numbers below are whatever the harness produced on this run — not tuned to hit a target.

![90% interval coverage by scenario and method](lab_coverage.png)

## Coverage, bias, RMSE, interval width

| Scenario | Method | n | Avg true effect | Bias | RMSE | 90% coverage | Avg interval width | Significant rate |
|---|---|---|---|---|---|---|---|---|
| S1 | conservation_constrained_sc | 200 | $165,541 | $3,211 | $127,219 | 84% | $342,210 | 58% |
| S1 | did_event_study | 200 | $165,541 | $-26,322 | $139,174 | 84% | $547,782 | 23% |
| S1 | synthetic_control | 200 | $165,541 | $2,135 | $116,039 | 87% | $348,389 | 58% |
| S2 | conservation_constrained_sc | 200 | $165,541 | $6,248 | $225,284 | 80% | $688,177 | 32% |
| S2 | did_event_study | 200 | $165,541 | $-32,682 | $227,730 | 86% | $911,216 | 17% |
| S2 | synthetic_control | 200 | $165,541 | $4,611 | $213,280 | 86% | $713,475 | 30% |
| S3 | conservation_constrained_sc | 200 | $0 | $41,271 | $212,140 | 74% | $467,696 | 26% |
| S3 | did_event_study | 200 | $0 | $13,036 | $139,249 | 86% | $532,079 | 14% |
| S3 | synthetic_control | 200 | $0 | $47,116 | $178,850 | 76% | $470,860 | 24% |
| S4 | conservation_constrained_sc | 200 | $-260,879 | $24,305 | $153,368 | 72% | $318,156 | 71% |
| S4 | did_event_study | 200 | $-260,879 | $-119,182 | $179,622 | 99% | $637,775 | 82% |
| S4 | synthetic_control | 200 | $-260,879 | $-107,146 | $172,404 | 90% | $588,786 | 60% |
| S5 | conservation_constrained_sc | 200 | $216,371 | $23,187 | $163,208 | 74% | $438,886 | 66% |
| S5 | did_event_study | 200 | $216,371 | $3,885 | $176,805 | 91% | $652,225 | 29% |
| S5 | synthetic_control | 200 | $216,371 | $22,615 | $158,950 | 78% | $448,789 | 64% |

## Where each method breaks

**S1**: Price +6%, 60 weeks of history. The clean case.
- conservation_constrained_sc: 84% coverage (target 90%), bias $3,211, power (detected as significant) 58%
- did_event_study: 84% coverage (target 90%), bias $-26,322, power (detected as significant) 23%
- synthetic_control: 87% coverage (target 90%), bias $2,135, power (detected as significant) 58%

**S2**: Same decision, 12 weeks of history. Short-history degradation.
- conservation_constrained_sc: 80% coverage (target 90%), bias $6,248, power (detected as significant) 32%
- did_event_study: 86% coverage (target 90%), bias $-32,682, power (detected as significant) 17%
- synthetic_control: 86% coverage (target 90%), bias $4,611, power (detected as significant) 30%

**S3**: No true effect, timed to a seasonal spike. True effect is 0 — `significant_rate` here *is* the false-positive rate.
- conservation_constrained_sc: false-positive rate 26% (found a 'significant' effect when the true effect is $0)
- did_event_study: false-positive rate 14% (found a 'significant' effect when the true effect is $0)
- synthetic_control: false-positive rate 24% (found a 'significant' effect when the true effect is $0)

**S4**: Price +10%, elastic demand (net revenue-negative), with spillover into a neighboring donor. Naive synthetic control is expected to be biased here.
- conservation_constrained_sc: 72% coverage (target 90%), bias $24,305, power (detected as significant) 71%
- did_event_study: 99% coverage (target 90%), bias $-119,182, power (detected as significant) 82%
- synthetic_control: 90% coverage (target 90%), bias $-107,146, power (detected as significant) 60%

**S5**: Staggered rollout across 6 metros over 8 weeks.
- conservation_constrained_sc: 74% coverage (target 90%), bias $23,187, power (detected as significant) 66%
- did_event_study: 91% coverage (target 90%), bias $3,885, power (detected as significant) 29%
- synthetic_control: 78% coverage (target 90%), bias $22,615, power (detected as significant) 64%

