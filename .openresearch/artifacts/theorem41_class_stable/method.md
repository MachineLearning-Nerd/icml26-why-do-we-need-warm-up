# Method

Choose the Equation (65) parameters so value, gradient, and Hessian match at `|w|=1`: `f=w^2` inside and `f=2 exp(|w|-1)-1` outside. This objective is 2-strongly convex, 2-PL, and `(H0,H1)=(2,1)`-smooth.

At loss 1.1, choose the constant step that maps the initial point exactly to zero. The verifier checks that it is below Equation (63), evaluates the theorem’s lower bound, and symbolically checks the C2 joins.
