# Method

The verifier reconstructs the Proposition 3.3 master bound from its blockwise constants, checks a complete 9-by-9 autograd Hessian against central finite differences, and verifies the universal scalar inequality chain with SymPy. It also exhausts a four-task bounded Rademacher distribution for the Proposition E.2 Jensen and variance reductions.

The fixed command is `uv run --frozen python -m warmup_repro.run`. The expected allocation is one CPU core and every numerical library is restricted to one thread.
