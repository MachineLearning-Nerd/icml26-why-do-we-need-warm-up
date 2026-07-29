# Method

Use `phi(s)=s+(a/omega)sin(omega*s)` with `a=1e-4`. Its three activation assumptions follow from elementary global inequalities. A coercive scalar lower bound gives a rigorous lower bound on `f*`, so the verifier maximizes the paper’s RHS over every feasible `f*` rather than estimating the optimum.

The complete 2-by-2 Hessian is evaluated three ways: autograd, a closed-form expression, and 80-digit SymPy arithmetic. The fixed command is `uv run --frozen python -m warmup_repro.run`.
