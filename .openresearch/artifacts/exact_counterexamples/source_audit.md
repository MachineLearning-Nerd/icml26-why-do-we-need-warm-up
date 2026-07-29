# Exact source and quantifier audit

- Proposition 3.2(ii) quantifies over all strongly balanced weights but states only
  `d <= m`; it does not require `lambda_min(XX^T)>0`. Its proof later divides by
  `sqrt(lambda_min(XX^T))` in Eq. (11).
- Theorem 4.1 is displayed as a per-function lower bound without quantifying over a
  fixed step or a worst-case class. The primary template, Theorem 4 of
  arXiv:1905.11881, explicitly takes the supremum over initialization and objective
  in a class with fixed constants.
- Theorem 4.3 quantifies over `(H0,H1)`-smooth and `mu`-PL functions, but its displayed
  count has no guard ensuring `log(H0/(2 H1 epsilon))` yields a nonnegative total.
- Proposition B.2 gives sum constants that can make `H0` negative, contradicting
  Definition 3.1's requirement `H0 >= 0`.
