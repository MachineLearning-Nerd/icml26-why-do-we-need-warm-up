# Source audit

Paper: arXiv 2510.03164, Section 3.2, Figure 1, Appendix K.1 and Figure K.2.

- HTML SHA-256: `e73c02ea9d475c9f200d2719867b03b4557c2b3e1169953a12513601f0d86e47`
- PDF SHA-256: `736bb3c117b8b698edbd4e0b567c5e3b673539e592d4c21da89a667114193b30`
- Retrieved: 2026-07-29 with an explicit browser User-Agent.
- Models: 70M, 160M, and 410M PlainLM Transformer language models.
- Data: FineWeb.
- Optimizer: SGD, learning rate `1e-4`, gradient clipping `1`.
- Quantifier: “For much of early training” and all three named scales.
- Displayed estimator: `||grad f_{S_k}(w_{k+1}) - grad f_{S_{k-1}}(w_k)|| / ||w_{k+1}-w_k||`.

The cited public PlainLM repository does not contain the paper's local-smoothness
measurement instrumentation. Its exact evaluation ordering therefore cannot be
recovered from public code; this route independently implements the displayed
equation literally.
