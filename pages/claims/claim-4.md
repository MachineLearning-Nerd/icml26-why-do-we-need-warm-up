# Claim 4 — FALSIFIED

## Exact paper statement

Theorem 4.1(3) and Appendix J claim a fixed-step iteration lower bound for the
paper's strongly-convex/PL construction when the constant step obeys Equation
(63). The imported claim is a conjunction with Theorem 4.2; falsifying
Theorem 4.1 makes that conjunction false. Theorem 4.2 is not contradicted.

## Assumption-satisfying counterexample

Use the paper's class-stable objective:

- `f(w)=w²` for `|w|≤1`;
- `f(w)=2exp(|w|−1)−1` for `|w|>1`.

Value, gradient, and Hessian match at `|w|=1`; all three residuals are zero.
The function is convex, 2-strongly-convex, 2-PL, and `(H0,H1)`-smooth with
`H0=2,H1=1`.

| Quantity | Value |
|---|---:|
| Initial point | 1.0487901641694322 |
| Initial loss | 1.1 |
| Constant step | 0.49942388769972956 |
| Eq. (63) cap | 1.9914730541896817 |
| Observed first hit at `ε=1e-6` | 1 |
| Claimed lower bound | 1.7462978859421259 |

Appendix J uses `−log(1−z)≥z` in the direction that enlarges a necessary lower
bound.

## Control and scope

At `ε=0.1`, the claimed bound is `0.3010≤1`, so the non-contradictory target
passes. Theorem 4.2's adaptive upper bound remains intact.

- [Executable source](../../warmup_repro/theorem41_class_stable.py)
- [Raw JSON](../../evidence/claim4_counterexample.json)
- Evidence SHA: `8ab79046a711d491657d6b6add8280fe533ddad3`
- Seed: `20260729`
- Fixed command: `uv run --frozen python -m warmup_repro.run`
- CPU/runtime: local, 1 selected thread / 8 visible CPUs, `13.993695 s`
