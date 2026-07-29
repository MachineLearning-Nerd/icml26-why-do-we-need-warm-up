# Evaluator contract

The fixed command exits nonzero on cumulative regression failure, parameter-count
mismatch, incomplete trajectories, or disagreement between the two proxy reductions.
Scientific non-confirmation is reported as `BLOCKED`, not converted into process
failure or a false `VERIFIED` label.
