# Evaluation

The claim is `FALSIFIED` only if the Hessian norm exceeds the worst-case printed RHS by at least 50 and both independent Hessian checks agree within `1e-9`. The verifier exits nonzero otherwise.

Setting `a=0` removes the high-second-derivative mechanism; this negative control must not falsify Equation (31).
