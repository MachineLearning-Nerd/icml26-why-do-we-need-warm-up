# Evaluation contract

The executable verifier exits nonzero if the published Proposition B.2 formula does not fail on
the exact counterexample, if the independent checker disagrees, if the corrected constant fails,
or if the historical slope negative control is not accepted.

This route does not treat a label mismatch, implementation error, or violated assumption as a
falsification.

