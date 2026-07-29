# Claim 5 — FALSIFIED

## Exact paper statement

Theorem 4.3 states its displayed PL iteration expression as an “after at most”
upper bound without an epsilon-domain guard.

## Assumption-satisfying counterexample

For `f(w)=w²/2`, take `H0=H1=μ=1`. Then
`f''(w)=1≤1+f(w)` and `|∇f(w)|²=2f(w)`, so all printed curvature and PL
assumptions hold. At `w0=0`, `f(w0)−f*=0`, and `ε=1`:

| Quantity | Value |
|---|---:|
| Minimum valid iteration count | 0 |
| Displayed paper upper bound | −13.862943611198906 |

No nonnegative hitting time can be at most a negative number.

## Control and scope

For initial gap one and `ε=1e-6`, the adaptive method hits in 77 iterations
while the positive paper expression is `302.4473`. The corrected small-error
recurrence therefore passes. The falsification is the displayed formula without
a domain guard.

- [Executable source](../../warmup_repro/counterexamples_exact.py)
- [Raw JSON](../../evidence/claim5_counterexample.json)
- Evidence SHA: `b405297f177c119ba696b54d49701815be1617b4`
- Seed: `20260729`
- Fixed command: `uv run --frozen python -m warmup_repro.run`
- CPU/runtime: local, 1 selected thread / 8 visible CPUs, `15.444996 s`
