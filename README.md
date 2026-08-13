# Why Do We Need Warm-up? A Theoretical Perspective

Independent, claim-by-claim reproduction audit for ICML 2026 paper
[“Why Do We Need Warm-up? A Theoretical Perspective”](https://arxiv.org/abs/2510.03164).

This repository is an independent audit of the paper’s definitions, quantified
theorems, counterexamples, and named-model experiments. It is not an official
author implementation or an endorsement of every printed formula.

## Current assessment

The six audited claim groups are assessed as one `VERIFIED`, four `FALSIFIED`,
and one `BLOCKED`. The previous live judge score was `5/12`; the `5–10/12`
range is a forecast, not a new judge result.

| Claim | Paper statement tested | How this audit produces the result | Evidence and verdict |
| --- | --- | --- | --- |
| 1 | Definition 3.1 defines `(H0,H1)`-smoothness, and Proposition B.1 includes ordinary `L`-smoothness as `H1=0`. | Reconstruct the inclusion symbolically, solve the `(L0,L1)` extremal constant, and compare a complete 48-dimensional Hessian with an independent HVP power estimate. A separate control checks the printed Proposition B.2 sum constants. | Exact definition/inclusion certificate and HVP relative error `8.95e-11`. The separate B.2 constants are rejected because they can make `H0<0`. **VERIFIED for the stated definition/inclusion.** |
| 2 | Proposition 3.2(ii) claims the deep-linear MSE objective is `(H0,H1)`-smooth for all strongly balanced weights under its stated data assumptions. | Set `X=diag(1,0)`, `Y=0`, and `W1=W2=diag(0,t)`. Audit strong balancedness, loss gap, and the complete Hessian on `t=1,2,4,8,16`; replace `X` with the identity as a negative control. | Loss gap and balance residual stay zero while the exact Hessian norm is `2t²`, hence unbounded. **FALSIFIED as printed; a rank-restored statement is not tested false.** |
| 3 | Proposition 3.3(ii), Equation (31), gives explicit `(H0,H1)` constants for the two-layer cross-entropy plus L2 result under global activation bounds. | Use a one-input, one-hidden-unit, one-sample activation `phi(s)=s+(a/omega)sin(omega*s)` with a full assumption audit; compare autograd, an analytic eigensystem, and an 80-digit Hessian calculation. Set `a=0` as the negative control. | Complete Hessian norm `63.1359` exceeds the maximum valid printed RHS `4.0883`; the independent methods agree within `2.14e-14`. **FALSIFIED for the printed constants; the corrected omitted `Clinear*f*` term is not falsified.** |
| 4 | Theorem 4.1(3) and Appendix J claim a fixed-step iteration lower bound under the Equation (63) stability cap. | Use the paper’s class-stable piecewise quadratic/exponential objective, certify `C2`, strong convexity, PL, and `(H0,H1)` assumptions, then measure the first hit under an admissible fixed step. Audit the logarithm inequality and a looser-epsilon control. | The method reaches the minimizer in one step while the claimed lower bound is `1.7463`; the adaptive Theorem 4.2 upper bound is not contradicted. **FALSIFIED for Theorem 4.1(3) as printed.** |
| 5 | Theorem 4.3 presents an “after at most” PL iteration expression without an epsilon-domain guard. | Use `f(w)=w²/2` with `H0=H1=mu=1`, start at the minimizer, and set `epsilon=1`; verify all smoothness/PL assumptions. Use a positive initial gap and small epsilon as the control. | The displayed bound is `-13.8629` iterations although the minimum valid count is zero. The small-epsilon control passes. **FALSIFIED for the displayed formula without a guard.** |
| 6 | Section 3.2 claims the consecutive-stochastic-gradient local-smoothness proxy is approximately linear in loss early in training for 70M/160M/410M PlainLM and ResNet50/ViT-Tiny on ImageNet32. | Run separate FineWeb, ImageNet32, complete-Hessian proxy-calibration, and literal-estimator routes with exact parameter counts, independent norm reductions, permutation controls, and same-batch controls. Compare every route with the paper’s batch, sequence, precision, tokenizer, estimator, and horizon requirements. | No positive-slope test passes, but the LM and vision trajectories are materially downscaled; the literal route also has two pre-science `uv` environment failures. **BLOCKED, not falsified.** |

The canonical claim pages, source audits, raw JSON, checkers, controls, and
limitations are linked from [`CURRENT.md`](CURRENT.md) and
[`pages/current/page.md`](pages/current/page.md).

## What the paper is doing

The paper studies why learning-rate warm-up—starting with a small step size and
increasing it early in training—can help neural-network optimization. Its core
assumption generalizes smoothness to a curvature bound

```text
||Hessian f(w)|| <= H0 + H1 * (f(w) - f*)
```

The paper argues that curvature decreases as the loss approaches its optimum,
derives an adaptive schedule with warm-up-like behavior, proves convergence
upper/lower bounds, and tests the proposed relationship on language and vision
models. This audit checks the literal quantifiers and displayed constants
before considering repaired interpretations.

## Reproducing the evidence

All formal nodes use one pinned Python 3.12 environment and the fixed command:

```bash
uv run --frozen python -m warmup_repro.run
```

The dispatcher selects the committed stage. Claims 1–5 use symbolic
certificates, complete Hessians, analytic or high-precision checks, assumption
audits, and controls that remove the proposed contradiction mechanism. Claim 6
uses separate expensive routes because the named models and data are much
larger; a short or substituted trajectory is never promoted to a full-paper
verification.

Useful entry points:

- [Current verification entrypoint](CURRENT.md)
- [Claim-by-claim pages](pages/claims/)
- [Illustrated report](reports/warmup_claims/report.md)
- [Release report](release_report.md)
- [Visibility matrix](visibility_matrix.md)
- [Machine-readable evidence](evidence/)
- [Warm-up notebook](notebooks/warmup_claims.py)

## Branch organization

`main` is the publication surface. The descriptive `audit/*`,
`historical/*`, and `release/*` branches preserve the experiment lineage. The
complete old-to-new mapping, including the historical `orx/*` names, is in
[`branch-audit.md`](branch-audit.md).

## Scope and limitations

- Claim 1 verifies the exact definition and inclusion, while separately
  recording that the printed Proposition B.2 closure constants are defective.
- Claims 2–5 target the literal paper statements and identify assumption,
  constant, inequality-direction, or domain-guard defects. A repaired theorem
  is not treated as falsified by the counterexample to the printed theorem.
- Claim 6 uses exact parameter-count model constructions where possible, but
  does not reproduce the paper’s full tokenizer, estimator ordering, batch,
  sequence lengths, precision, or training horizons. Its honest status is
  `BLOCKED`.
- The live score remains `5/12` until an evaluator judges the published
  revision. No score increase is claimed here.

## Paper

- **Title:** Why Do We Need Warm-up? A Theoretical Perspective
- **Authors:** Foivos Alimisis, Rustem Islamov, Aurelien Lucchi
- **Paper:** [arXiv:2510.03164](https://arxiv.org/abs/2510.03164)
- **Author-hosted PDF:** [warmup.pdf](https://rustem-islamov.github.io/files/publications/Warmup/warmup.pdf)
- **Submission:** ICML 2026; arXiv v1 submitted October 3, 2025, v2 revised June 28, 2026
- **Paper identifier:** `a6fo32UnpU`

## Citation

```bibtex
@misc{alimisis2025warmup,
  title         = {Why Do We Need Warm-up? A Theoretical Perspective},
  author        = {Alimisis, Foivos and Islamov, Rustem and Lucchi, Aurelien},
  year          = {2025},
  eprint        = {2510.03164},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {ICML 2026; revised version arXiv:2510.03164v2}
}
```

## Thank you

Thank you to Foivos Alimisis, Rustem Islamov, and Aurelien Lucchi for making a
subtle optimization question concrete through a shared smoothness framework,
explicit theorem statements, and named-model experiments. The paper’s detailed
assumptions and public-facing artifacts made it possible to audit the literal
claims, find where printed statements need qualification, and preserve the
important distinction between a counterexample and an incomplete reproduction.

## Attribution

This independent audit is maintained by [MachineLearning-Nerd](https://github.com/MachineLearning-Nerd).
It is not affiliated with or endorsed by the paper’s authors.
