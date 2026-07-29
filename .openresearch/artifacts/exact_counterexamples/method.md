# Method

The Proposition 3.2 witness uses exact 2x2 diagonal matrices and materializes the
complete 8x8 float64 Hessian. SymPy independently differentiates the one-coordinate
restriction and proves its curvature tends to infinity while the loss gap is zero.

The Theorem 4.1 and 4.3 witnesses use the rational quadratic `f(w)=w^2/2`, whose
convexity, Aiming, PL, and `(H0,H1)` assumptions are identities. Direct GD hitting
times are compared with the exact displayed formulas. Each defect has a control
that restores the omitted rank, minimax, or epsilon-domain condition.
