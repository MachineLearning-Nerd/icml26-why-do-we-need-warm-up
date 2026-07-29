# Method

A deterministic 35-parameter classifier is small enough for a complete Hessian at every step. The verifier computes the printed stochastic proxy, same-batch and full-batch finite-difference quotients, the exact Hessian spectral norm, and the exact Hessian action along the update.

The initial complete Hessian is independently reconstructed by central differences of gradients. The fixed command is `uv run --frozen python -m warmup_repro.run`.
