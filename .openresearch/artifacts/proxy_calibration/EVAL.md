# Evaluation

The verifier exits nonzero if the independent Hessian error exceeds `1e-7`, if the calibrated batch-noise ratio is below 100, or if the same-batch negative control fails to remove batch noise.

This is an estimator-calibration route, not full-scale Claim 6 verification.
