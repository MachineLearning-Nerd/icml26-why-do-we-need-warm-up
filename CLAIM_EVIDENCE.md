# Claim evidence ledger

The verdicts below apply to the literal printed claim contracts. A counterexample
to a printed statement does not automatically falsify a corrected theorem.

| Claim | Paper anchor | How evidence is produced | Control and scope boundary | Status |
| --- | --- | --- | --- | --- |
| C1 | Definition 3.1, Proposition B.1, and the printed Proposition B.2 closure | `warmup_repro/certificates.py` reconstructs the definition/inclusion and `evidence/claim1_definition.json` compares a complete Hessian with an independent HVP estimate. | Definition/inclusion pass with relative HVP error `8.95e-11`. The separate B.2 constants are rejected because they can make `H0 < 0`. | `VERIFIED_SCOPED` |
| C2 | Proposition 3.2(ii) | `warmup_repro/counterexamples_exact.py` uses `X=diag(1,0)`, `Y=0`, and balanced `W1=W2=diag(0,t)`; the complete Hessian norm is `2t²`. | Strong balancedness and zero loss gap hold while curvature is unbounded. Full-row-rank `X` is a negative control, so a rank-restored proposition is not falsified. | `FALSIFIED_SCOPED` |
| C3 | Proposition 3.3(ii), Equation (31) | `warmup_repro/proposition33_counterexample.py` checks a bounded-activation oscillatory function with autograd, analytic eigensystem, and 80-digit Hessian. | Hessian norm `63.1359` exceeds every valid printed RHS upper bound `4.0883`; setting `a=0` removes the contradiction. The corrected omitted `Clinear*f*` term is not tested false. | `FALSIFIED_SCOPED` |
| C4 | Theorem 4.1(3), Appendix J, Equations (63) and (65) | `warmup_repro/theorem41_class_stable.py` certifies the paper's class-stable objective and measures the first hit under an admissible fixed step. | One-step convergence is below the printed lower bound `1.7463`; the adaptive Theorem 4.2 upper bound is not contradicted. | `FALSIFIED_SCOPED` |
| C5 | Theorem 4.3 displayed “after at most” iteration expression | `warmup_repro/counterexamples_exact.py` uses `f(w)=w²/2`, verifies smoothness/PL assumptions, and tests the domain guard. | At zero initial gap and `epsilon=1`, the displayed count is `−13.8629`; positive-gap small-epsilon control passes. | `FALSIFIED_SCOPED` |
| C6 | Section 3.2, Figure 1, Appendix K.1 and Figure K.2 | `warmup_repro/fineweb_lm.py`, `imagenet32.py`, `proxy_calibration.py`, and `literal_proxy.py` cover exact parameter counts, vision routes, controls, and literal-estimator attempts. | The LM/vision runs are materially downscaled or protocol-deviant; literal routes failed before science execution. No full tokenizer, estimator, batch, sequence, precision, and horizon package exists. | `BLOCKED_PROTOCOL` |

## Historical score boundary

The previous live evaluator result is `5/12`. The release report's `5–10/12`
and `10/12` values are forecasts, not current scores. No local counterexample
or blocked route changes the historical result.
