# Method

The verifier reads the official archive central directory through HTTP byte
ranges, fetches only `train_data_batch_1`, and checks its ZIP size, CRC32,
and SHA-256. It never substitutes CIFAR or validation images.

Three deterministic 30-step trajectories are run for each cited architecture.
The exact paper proxy is reduced independently in float32 and float64. Pooled
bootstrap slopes, per-seed signs, and 500-permutation controls are predeclared.
The inherited certificate suite is rerun by the same fixed command.
