# Branch audit

The old `orx/*` names are retained here only as historical provenance. Each
was renamed to describe the evidence or release role.

| Historical branch | Clean branch | Purpose |
| --- | --- | --- |
| `orx/judged-5-of-12-historical-baseline` | `historical/judged-5-of-12-baseline` | Preserve the exact judged Space revision and prior score. |
| `orx/exact-certificates-plus-dimension-sweeps` | `audit/c1-certificates-dimension-sweeps` | Build exact definition certificates and scaled checks. |
| `orx/exact-theorem-counterexamples` | `audit/c2-c5-exact-counterexamples` | Add the exact Claims 2 and 5 counterexamples. |
| `orx/symbolic-proposition-3-3-and-e-2-certificates` | `audit/c1-symbolic-certificates` | Repair the symbolic Proposition 3.3 and E.2 certificate comparison. |
| `orx/exact-proposition-3-3-counterexample` | `audit/c3-proposition33-counterexample` | Falsify the printed Proposition 3.3 constants with an assumption-satisfying Hessian check. |
| `orx/stochastic-curvature-proxy-calibration` | `audit/c6-proxy-calibration` | Calibrate the changing-minibatch curvature proxy against a complete Hessian. |
| `orx/class-stable-theorem-4-1-counterexample` | `audit/c4-class-stable-counterexample` | Test Theorem 4.1 with the paper’s class-stability construction. |
| `orx/assumption-satisfying-falsification-audit` | `audit/assumption-satisfying-falsification` | Repair the finite-ray negative control and consolidate assumption checks. |
| `orx/exact-scale-fineweb-lm-curvature` | `audit/c6-fineweb-lm-curvature` | Run exact parameter-count PlainLM routes on FineWeb. |
| `orx/exact-imagenet32-vision-curvature` | `audit/c6-imagenet32-curvature` | Run the ResNet50/ViT-Tiny ImageNet32 route. |
| `orx/literal-section-3-2-proxy-falsification-route` | `audit/c6-literal-proxy` | Implement the displayed estimator literally with same-batch and denominator controls. |
| `orx/cumulative-claim-release-candidate` | `release/cumulative-claim-candidate` | Integrate the exact Claims 1–5 evidence and Claim 6 routes. |
| `orx/final-evaluator-visible-release-gate` | `release/final-evaluator-gate` | Publish manifests, visibility gates, and the final red-team record. |

`main` is the current publication surface. Every clean branch receives this
README and branch map so an experiment checkout remains self-describing.
Superseded `orx/*` refs are deleted from the live GitHub repository after the
clean refs are published.
