# Claim 3 — FALSIFIED

## Exact paper statement

Proposition 3.3(ii), Equation (31), supplies explicit `(H0,H1)` constants for
the two-layer CE+L2 objective under global activation bounds
`|φ(s)|≤C1|s|`, `|φ'(s)|≤C2`, `|φ''(s)|≤C3`, with positive L2 coefficients.

## Assumption-satisfying counterexample

Use one input, one hidden unit, one positive sample, `λ1=λ2=0.1`, and

`φ(s)=s+(a/ω)sin(ωs)`,

where `a=1e-4` and `ω=4238903.076676133`. Globally,
`C1=C2=1.0001` and `C3=423.89030766761334`.

| Quantity | Value |
|---|---:|
| Loss | 0.3250829739187598 |
| Rigorous lower bound on `f*` | 0.32506100278721883 |
| Maximum printed Eq. (31) RHS over every valid `f*` | 4.088329645538989 |
| Complete Hessian norm | 63.13592027816656 |
| Contradiction margin | 59.04759063262757 |

## Independent checker, control, and scope

Closed-form eigensystem: `63.13592027816658`. Eighty-digit calculation:
`63.13592027816657`. Maximum disagreement with autograd: `2.14e-14`.

Setting `a=0` is the control; its contradiction margin is `−3.5524`, so it is
rejected as intended. The reconstructed proof shows Eq. (31) drops
`Clinear·f*`. This falsifies the printed constants, not the corrected `H0`.

- [Executable source](../../warmup_repro/proposition33_counterexample.py)
- [Raw JSON](../../evidence/claim3_counterexample.json)
- Evidence SHA: `f3004d8eb2ca316d27899353d76e0ef985b1d76f`
- Seed: `20260729`
- Fixed command: `uv run --frozen python -m warmup_repro.run`
- CPU/runtime: local, 1 selected thread / 8 visible CPUs, `13.63419 s`
