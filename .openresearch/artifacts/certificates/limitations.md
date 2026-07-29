# Limitations and deviations

- The named neural architectures are scaled to roughly 10^5 parameters, not production scale.
  Their role is to stress the proof-derived structure; they do not replace the universal proof.
- The deep-linear and leaky-ReLU data are synthetic because their propositions quantify over
  matrices rather than a named dataset.
- Theorem 4.1’s hard instance is intrinsically one-dimensional; its certificate, rather than a
  cosmetic high-dimensional embedding, is the lower-bound evidence.
- Claim 6 is deliberately left `BLOCKED` in this route.
