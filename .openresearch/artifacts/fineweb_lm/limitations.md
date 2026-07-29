# Limitations and deviations

- This CPU route preserves the named model scales but not the paper's 1.2B/3.2B-token
  horizons.
- It uses micro-batch 1, sequence length 16, deterministic hashed word pieces, and
  float32 CPU instead of batch 256, sequence lengths 1024/2048, the unavailable
  original tokenizer, and FP16 GPU training.
- A short exact-scale trajectory is direct but low-power evidence. The claim remains
  `BLOCKED` unless every predeclared architecture passes the bootstrap and permutation
  criteria.
