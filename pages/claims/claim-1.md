# Claim 1 — VERIFIED

## Exact paper statement

Definition 3.1 calls a twice-differentiable function `(H0,H1)`-smooth when,
for every `w`,

`||∇²f(w)|| ≤ H0 + H1(f(w)−f*)`.

Proposition B.1 includes ordinary `L`-smooth functions as the special case
`H0=L,H1=0`.

## Assumptions and certificate

The statement is definitional and universally quantified over the function
domain. The inclusion is an exact symbolic substitution, not a sample. The
separate `(L0,L1)` inclusion reconstructs its extremal constant
`ν=0.5671432904097838` with zero equation residual.

An independent numerical checker uses a complete 48-dimensional Hessian:

| Quantity | Value |
|---|---:|
| Exact spectral radius | 19.94614995750155 |
| HVP power estimate | 19.946149955715686 |
| Relative error | `8.9534e-11` |

## Control and scope

The control rejects Proposition B.2's printed sum constants: they can produce
`H0=−100` for a zero-Hessian sum. Closure survives with a corrected
shift-invariant constant. This page verifies the exact definition and inclusion,
not the defective printed closure constants.

- [Executable source](../../warmup_repro/certificates.py)
- [Raw JSON](../../evidence/claim1_definition.json)
- Evidence SHA: `52ea25684a5d1ff4733dbf9b2a4cb77fc340e549`
- Seed: `20260729`
- Fixed command: `uv run --frozen python -m warmup_repro.run`
- CPU/runtime: 1 selected thread; cumulative verifier `13.99 s` on the winning chain
