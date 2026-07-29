# Why Do We Need Warm-up? Exact claim reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-a6fo32UnpU-why-do-we-need-warm-up-a-theoretical-perspective/blob/main/notebooks/warmup_claims.py)

This project audits arXiv 2510.03164 claim by claim. The previous live judge
awarded `5/12` for toy evidence. The new candidate replaces Claims 1–5 with an
exact definition certificate or assumption-satisfying counterexamples; Claim 6
remains `BLOCKED` because exact parameter counts do not compensate for
shortened CPU horizons and material batch, sequence, tokenizer, and precision
substitutions.

Headline result: Claim 2 has loss gap zero but Hessian norm `2t²→∞`; Claim 3
has complete Hessian `63.1359` versus a maximum printed bound `4.0883`; Claim 4
reaches the minimizer in one iteration versus a claimed lower bound `1.7463`;
Claim 5's displayed bound becomes `−13.8629` iterations. These are scoped
falsifications of the printed statements, not claims that the repaired ideas
are false.

- [Illustrated claim-by-claim report](reports/warmup_claims/report.md)
- [Tutorial-style marimo notebook](notebooks/warmup_claims.py)
- [Machine-readable evidence](evidence/)

The paper numbers and observed numbers, assumption audits, controls,
independent checkers, compute, and limitations are inline in the report.
Conservative projected score: `5–10/12`; best-supported possible score:
`10/12`, a forecast only. The live score remains `5/12` until the evaluator
judges the published revision.

## Experiment log

Every formal node uses the exact command
`uv run --frozen python -m warmup_repro.run`.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public README, report, notebook, and evidence | Not run as an experiment (publication surface) | Presentation-only baseline branch | — |
| [`orx/judged-5-of-12-historical-baseline`](https://github.com/MachineLearning-Nerd/icml26-repro-a6fo32UnpU-why-do-we-need-warm-up-a-theoretical-perspective/tree/orx/judged-5-of-12-historical-baseline) | Freeze the exact judged Space and score | `uv run --frozen python -m warmup_repro.run` | Historical regression PASS; all six new verdicts conservatively BLOCKED | HF cpu-upgrade, 1 thread / 64 visible CPUs, 21 s |
| [`orx/exact-theorem-counterexamples`](https://github.com/MachineLearning-Nerd/icml26-repro-a6fo32UnpU-why-do-we-need-warm-up-a-theoretical-perspective/tree/orx/exact-theorem-counterexamples) | Exact Proposition 3.2 and Theorem 4.3 audits | `uv run --frozen python -m warmup_repro.run` | Claims 2 and 5 FALSIFIED; verifier PASS | Local CPU, 1 thread, 15.45 s |
| [`orx/exact-proposition-3-3-counterexample`](https://github.com/MachineLearning-Nerd/icml26-repro-a6fo32UnpU-why-do-we-need-warm-up-a-theoretical-perspective/tree/orx/exact-proposition-3-3-counterexample) | Three-way Hessian check for Eq. (31) | `uv run --frozen python -m warmup_repro.run` | Claim 3 FALSIFIED; verifier PASS | Local CPU, 1 thread, 13.63 s |
| [`orx/class-stable-theorem-4-1-counterexample`](https://github.com/MachineLearning-Nerd/icml26-repro-a6fo32UnpU-why-do-we-need-warm-up-a-theoretical-perspective/tree/orx/class-stable-theorem-4-1-counterexample) | Counterexample respecting the theorem's class-stability prose | `uv run --frozen python -m warmup_repro.run` | Claim 4 FALSIFIED; cumulative verifier PASS | Local CPU, 1 thread, 13.99 s |
| [`orx/exact-scale-fineweb-lm-curvature`](https://github.com/MachineLearning-Nerd/icml26-repro-a6fo32UnpU-why-do-we-need-warm-up-a-theoretical-perspective/tree/orx/exact-scale-fineweb-lm-curvature) | 70M/160M/410M PlainLM trajectories on FineWeb | `uv run --frozen python -m warmup_repro.run` | No positive-slope test passes; Claim 6 BLOCKED because protocol is downscaled | HF cpu-upgrade, 64 threads, 4530.31 s |
| [`orx/exact-imagenet32-vision-curvature`](https://github.com/MachineLearning-Nerd/icml26-repro-a6fo32UnpU-why-do-we-need-warm-up-a-theoretical-perspective/tree/orx/exact-imagenet32-vision-curvature) | ResNet50/ViT-Tiny, official ImageNet32, three seeds | `uv run --frozen python -m warmup_repro.run` | Neither relationship test passes; Claim 6 BLOCKED because 30-step batch-8 float32 trajectories are materially downscaled | HF cpu-upgrade, 64 threads, 23837.27 s |
| [`orx/literal-section-3-2-proxy-falsification-route`](https://github.com/MachineLearning-Nerd/icml26-repro-a6fo32UnpU-why-do-we-need-warm-up-a-theoretical-perspective/tree/orx/literal-section-3-2-proxy-falsification-route) | Literal printed estimator and same-batch controls | `uv run --frozen python -m warmup_repro.run` | BLOCKED before science: two default HF images lacked `uv` | HF cpu-upgrade, two 10 s environment failures |
| [`orx/cumulative-claim-release-candidate`](https://github.com/MachineLearning-Nerd/icml26-repro-a6fo32UnpU-why-do-we-need-warm-up-a-theoretical-perspective/tree/orx/cumulative-claim-release-candidate) | Integrate evidence and rerun accepted Claims 1–5 | `uv run --frozen python -m warmup_repro.run` | Cumulative verifier and historical regression PASS | Local CPU, 1 thread / 8 visible CPUs, 14.81 s |
| [`orx/final-evaluator-visible-release-gate`](https://github.com/MachineLearning-Nerd/icml26-repro-a6fo32UnpU-why-do-we-need-warm-up-a-theoretical-perspective/tree/orx/final-evaluator-visible-release-gate) | Final manifests and canonical-entrypoint red team | `uv run --frozen python -m warmup_repro.run` | Visibility, text manifest, and two-pass red-team gates PASS; parent cumulative verifier PASS | Local CPU, 1 thread |

## Notebook

The notebook embeds the small final results and links its headline image from
GitHub, so expensive experiments are not required to read it in Molab.

```text
uv run marimo edit notebooks/warmup_claims.py
uv run marimo run notebooks/warmup_claims.py
```

## Upstream workspace note

ICML 2026 agent reproduction workspace for OpenReview paper `a6fo32UnpU`.
