# Source audit

The paragraph after Theorem 4.1 says its fixed step is chosen to be stable over the relevant function class. The earlier unrestricted quadratic counterexample is therefore not used as the final basis.

Appendix J derives Equation (63) as its stability cap, then analyzes the strongly convex construction in Equation (65). In the quadratic phase it replaces the exact denominator `-log(1-eta*mu)` using `-log(1-z)>=z`; that replacement increases, rather than decreases, the proposed necessary iteration count.
