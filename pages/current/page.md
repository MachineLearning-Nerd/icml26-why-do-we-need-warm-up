# Current exact claim verification

This is the canonical evaluator entrypoint for the new campaign. It supersedes
the **Historical rejected baseline** at `pages/overview/page.md` without
deleting it.

- Paper: arXiv `2510.03164`
- Fixed command: `uv run --frozen python -m warmup_repro.run`
- Environment: Python `>=3.12,<3.13`, pinned by `pyproject.toml` and `uv.lock`
- Seed: `20260729`
- Paper HTML SHA-256: `e73c02ea9d475c9f200d2719867b03b4557c2b3e1169953a12513601f0d86e47`
- Paper PDF SHA-256: `736bb3c117b8b698edbd4e0b567c5e3b673539e592d4c21da89a667114193b30`
- Previous live score: `5/12`
- Conservative forecast: `5–10/12`
- Best-supported possible score: `10/12` — forecast only

## Claim navigation

| Claim | Current page | Verdict | Core evidence |
|---|---|---|---|
| 1 | [Definition and inclusion](../claims/claim-1.md) | VERIFIED | Symbolic proof + independent Hessian/HVP check |
| 2 | [Deep-linear counterexample](../claims/claim-2.md) | FALSIFIED | Balanced zero-gap ray with curvature `2t²→∞` |
| 3 | [Equation (31) counterexample](../claims/claim-3.md) | FALSIFIED | Hessian `63.1359` > maximum printed RHS `4.0883` |
| 4 | [Theorem 4.1 counterexample](../claims/claim-4.md) | FALSIFIED | One hit < claimed lower bound `1.7463` |
| 5 | [Theorem 4.3 counterexample](../claims/claim-5.md) | FALSIFIED | Displayed upper bound `−13.8629` iterations |
| 6 | [Named-model empirical routes](../claims/claim-6.md) | BLOCKED | Exact parameter counts, material protocol deviations |

## Reproducible sources

- [Pinned project dependencies](../../pyproject.toml)
- [Exact lockfile](../../uv.lock)
- [Fixed command dispatcher](../../warmup_repro/run.py)
- [Definition and scaled checks](../../warmup_repro/certificates.py)
- [Claims 2 and 5 exact checks](../../warmup_repro/counterexamples_exact.py)
- [Claim 3 exact check](../../warmup_repro/proposition33_counterexample.py)
- [Claim 4 exact check](../../warmup_repro/theorem41_class_stable.py)
- [Claim 6 FineWeb route](../../warmup_repro/fineweb_lm.py)
- [Claim 6 ImageNet32 route](../../warmup_repro/imagenet32.py)
- [Claim 6 literal estimator route](../../warmup_repro/literal_proxy.py)

The verifier exits nonzero on a failed assumption certificate, result
inequality, independent checker, control, or historical regression.

## Release records

- [Visibility matrix](../../visibility_matrix.md)
- [Evaluator-blind review](../red-team/page.md)
- [Release report](../../release_report.md)
- [Illustrated report](../../reports/warmup_claims/report.md)

No toy result is described as full-scale. `BLOCKED` is not a pass.
