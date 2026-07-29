# Evaluator-visible evidence matrix

Traversal starts at `CURRENT.md` or `pages/current/page.md`. Every path below is
reachable from the current page without repository knowledge.

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | `pages/claims/claim-1.md` | `warmup_repro/certificates.py` | Yes | `evidence/claim1_definition.json` | Full Hessian vs HVP | Defective B.2 constants rejected | Definition 3.1 + B.1 inclusion | VERIFIED |
| 2 | `pages/claims/claim-2.md` | `warmup_repro/counterexamples_exact.py` | Yes | `evidence/claim2_counterexample.json` | Symbolic `2t²` vs complete 8×8 Hessian | Full-rank X removes mechanism | Proposition 3.2(ii), all strongly balanced W | FALSIFIED |
| 3 | `pages/claims/claim-3.md` | `warmup_repro/proposition33_counterexample.py` | Yes | `evidence/claim3_counterexample.json` | Autograd, analytic, 80-digit Hessians | Linear activation no longer contradicts | Proposition 3.3(ii), printed Eq. (31) | FALSIFIED |
| 4 | `pages/claims/claim-4.md` | `warmup_repro/theorem41_class_stable.py` | Yes | `evidence/claim4_counterexample.json` | C2 boundary + symbolic log audit | Loose epsilon target is non-contradictory | Theorem 4.1(3) under Eq. (63) cap | FALSIFIED |
| 5 | `pages/claims/claim-5.md` | `warmup_repro/counterexamples_exact.py` | Yes | `evidence/claim5_counterexample.json` | Exact quadratic identities | Small-epsilon recurrence passes | Theorem 4.3 displayed formula | FALSIFIED |
| 6 | `pages/claims/claim-6.md` | LM, vision, calibration, and literal-estimator sources | Yes | FineWeb + ImageNet32 + calibration JSON | Independent reductions and complete Hessian | Permutations + same-batch control | Section 3.2 named models and early-training quantifier | BLOCKED |

Every page also exposes the fixed command, pinned environment, seed, source
anchor, Git SHA, CPU/runtime, limitations, and scope. Claim 6 is complete as a
`BLOCKED` record; missing paper-scale capability is not represented as a pass.
