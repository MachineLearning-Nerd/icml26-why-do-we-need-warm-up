# Limitations and deviations

- Eight measured updates do not cover “much of early training.”
- Only 70M is tested in this route; 160M and 410M are not.
- Micro-batch 1 and sequence length 16 replace batch 256 and length 1024.
- Deterministic hashed tokens replace the unavailable tokenizer.
- Float32 CPU replaces mixed-precision FP16 GPU training.
- The public repository does not expose the paper's measurement code.

These are material deviations. This route cannot verify or falsify Claim 6.
