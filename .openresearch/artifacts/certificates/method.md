# Method

The verifier has three independent layers:

1. SymPy checks the exact regime inequalities, boundary matching of the lower-bound witness,
   homogeneity degrees, and the `(L0,L1)` inclusion constant.
2. PyTorch computes direct Hessian-vector products on balanced deep-linear, balanced leaky-ReLU,
   L2-regularized two-layer CE/MSE, and formal single-attention models. The largest case has more
   than 100,000 parameters. A small complete Hessian independently calibrates the HVP estimator.
3. A dense nonseparable exponential-quadratic objective has an analytic `(H0,H1)` certificate,
   convex Aiming constant one, and `mu`-PL from strong convexity. Fixed and adaptive GD use
   predeclared horizons and first-hit targets independent of theorem bounds.

