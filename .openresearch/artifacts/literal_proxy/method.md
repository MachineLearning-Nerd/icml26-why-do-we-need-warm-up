# Method

The route uses the exact 71,941,888-parameter 70M PlainLM architecture already
audited in the FineWeb route. At every measured step it evaluates three gradient
differences:

1. the displayed two-minibatch, two-iterate estimator;
2. a same-minibatch finite-difference control that removes batch-switch noise;
3. the one-backward approximation used in the earlier short trajectory.

Float32 and float64 reductions independently check every norm. A vector identity
checks that the literal numerator equals the sum of the same-batch curvature
change and same-iterate batch-switch change. A zero-denominator input must be
rejected. The run uses seed `20260729`, 64 requested CPU threads, and the fixed
command `uv run --frozen python -m warmup_repro.run`.
