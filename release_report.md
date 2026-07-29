# Release report

- Previous live judged score: `5/12`
- Conservative projected score range after the proposed change: `5–10/12`
- Best-supported possible new score: `10/12` — forecast, not a judge result

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
|---|---:|---:|---|---|---|
| 1 | 1 | 2 | HIGH | VERIFIED | Exact definition and inclusion certificate; scope interpretation remains a judge decision |
| 2 | 1 | 2 | HIGH | FALSIFIED | Assumption-satisfying unbounded-curvature ray; omitted data-rank condition is the interpretation risk |
| 3 | 1 | 2 | HIGH | FALSIFIED | Three independent Hessian calculations contradict printed Eq. (31); a repaired constant is not contradicted |
| 4 | 1 | 2 | HIGH | FALSIFIED | Paper-class objective and admissible step contradict Theorem 4.1(3); Theorem 4.2 is not contradicted |
| 5 | 1 | 2 | MEDIUM | FALSIFIED | Displayed formula is negative on a valid PL instance; a reviewer may infer an unstated epsilon guard |
| 6 | 0 | 0 | LOW | BLOCKED | Four routes completed or attempted; paper-scale tokenizer, estimator, precision, batches, and horizons remain unavailable |

The current live total remains `5/12`. The conservative projected total is
`5–10/12`; the best-supported possible total is `10/12`. Claims 1–5 changed
from toy evidence to exact verification or falsification. Claim 6 remains
`BLOCKED`, after four distinct routes, because none verifies the full named
protocol or establishes an assumption-satisfying counterexample.

## Release action

Publish the exact text allowlist in `release/upload-allowlist.txt` to the
existing Space `DineshAI/a6fo32UnpU`, then download the returned revision,
verify every uploaded hash, repeat canonical-entrypoint traversal, and mark it
awaiting the live judge. No second Space will be created.

## Experiment tree and winning source

The tree is a stacked chain of exact certificates and counterexamples, followed
by separate Claim 6 FineWeb, ImageNet32, calibration, and literal-estimator
routes, then the cumulative release node. Every node uses:

```text
uv run --frozen python -m warmup_repro.run
```

The winning branch is `orx/cumulative-claim-release-candidate`. Its final SHA
is the exact commit referenced by the published manifest and OpenResearch run.

## Evidence and compute

- Claims 1–5: `pages/claims/claim-1.md` through `claim-5.md`, with linked code
  and JSON. Final cumulative validation uses one local CPU core and is expected
  below five minutes.
- FineWeb: `pages/claims/claim-6.md` and `evidence/claim6_fineweb.json`;
  expected 64 cores, HF `cpu-upgrade`, actual 64 CPUs, `4530.30788 s`.
- ImageNet32: `evidence/claim6_imagenet32.json`; expected 64 cores, HF
  `cpu-upgrade`, actual 64 CPUs, `23837.27177 s`. The complete scientific
  payload preceded the provider timeout.
- Literal route: two ten-second HF environment failures before science
  (`uv: command not found`); no third blind relaunch.
- Cost: the `orx` evidence exposes allocation and runtime but no provider
  charge, so no monetary amount is invented.

## Historical preservation

The exact judged revision is
`17e423af3b04b3c0fb493ccfdc26a9724a2be53c`. Its file paths are retained, its
scientific page is byte-identical and reachable as **Historical rejected
baseline**, and its original navigation JSON is preserved at
`historical/judged-logbook-17e423af.json`. The candidate subset and hash audit
is recorded by `release/gate-results.json`.

## Publication state

This report is a forecast and release record. It does not claim a score
increase. The publication is awaiting the live judge.
