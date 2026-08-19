# Reproduction report

## Executive result

The audit verifies the paper's smoothness definition/inclusion, falsifies four
printed theorem statements with exact assumption-satisfying counterexamples,
and keeps the named-model empirical relationship blocked because no route
matches the full paper protocol.

```text
PARTIAL_C1_VERIFIED_C2_C3_C4_C5_FALSIFIED_C6_BLOCKED_HISTORICAL_SCORE_5_OF_12_NO_CURRENT_SCORE
```

## Counterexample summary

- C2: balanced zero-gap deep-linear ray has complete Hessian norm `2t²`, so no finite global `H0` exists without a rank-restoring assumption.
- C3: the complete Hessian norm `63.1359` exceeds the maximum valid printed Equation (31) RHS `4.0883`; independent methods agree within `2.14e-14`.
- C4: the class-stable fixed-step method reaches the minimizer in one step while the printed lower bound is `1.7463`; the adaptive upper bound is untouched.
- C5: the displayed Theorem 4.3 count is `−13.8629` at a valid zero-gap/epsilon case; a nonnegative iteration count cannot satisfy it.

## Claim 6 boundary

FineWeb and ImageNet32 routes do not pass the positive-slope criterion, but
their batch, sequence, precision, tokenizer, steps, and/or model protocol are
materially downscaled. The literal estimator route failed twice before science
execution because `uv` was unavailable. These are blockers, not falsifications.

## Score and publication boundary

The previous live judge result is `5/12`; forecasts in the release report are
not scores. This repository makes no current score, publication approval, or
author-endorsement claim.
