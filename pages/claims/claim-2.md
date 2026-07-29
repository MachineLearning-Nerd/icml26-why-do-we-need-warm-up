# Claim 2 — FALSIFIED

## Exact paper statement

Proposition 3.2(ii) asserts the deep-linear MSE objective is
`(H0,H1)`-smooth for all strongly balanced weights under its stated data
assumptions. The printed statement does not require `X` to have full row rank.

## Assumption-satisfying counterexample

Set `X=diag(1,0)`, `Y=0`, and `W1=W2=diag(0,t)`. The network is two-layer,
the MSE loss and target follow the proposition, and strong balancedness holds
with residual zero.

| t | Loss gap | Balance residual | Complete Hessian norm |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 2 |
| 2 | 0 | 0 | 8 |
| 4 | 0 | 0 | 32 |
| 8 | 0 | 0 | 128 |
| 16 | 0 | 0 | 512 |

The complete `8×8` Hessian and symbolic restriction both give `2t²`. At
`f−f*=0`, Definition 3.1 would require `2t²≤H0` for every `t`, impossible for
finite `H0`.

## Independent checker, control, and scope

Autograd and the symbolic second derivative agree exactly on the tested ray.
The control replaces `X` by the identity: rank becomes two and the same ray has
loss `t⁴` (`256` at `t=4`), removing the mechanism.

This falsifies the proposition as printed. A rank-restored proposition is not
falsified.

- [Executable source](../../warmup_repro/counterexamples_exact.py)
- [Raw JSON](../../evidence/claim2_counterexample.json)
- Evidence SHA: `b405297f177c119ba696b54d49701815be1617b4`
- Seed: `20260729`
- Fixed command: `uv run --frozen python -m warmup_repro.run`
- CPU/runtime: local, 1 selected thread / 8 visible CPUs, `15.444996 s`
