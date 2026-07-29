# Source audit

Retrieved 2026-07-29 with User-Agent
`Mozilla/5.0 (compatible; OpenResearch-Reproduction/1.0; +https://openresearch.sh)`.

| Source | SHA-256 |
| --- | --- |
| https://ar5iv.labs.arxiv.org/html/2510.03164 | `e73c02ea9d475c9f200d2719867b03b4557c2b3e1169953a12513601f0d86e47` |
| https://arxiv.org/pdf/2510.03164 | `736bb3c117b8b698edbd4e0b567c5e3b673539e592d4c21da89a667114193b30` |

The retrieved paper is arXiv v2 dated 2026-06-28. Primary anchors:

- Definition 3.1: HTML `#S3.Thmdefinition1`.
- Proposition 3.2: HTML `#S3.Thmtheorem2` (deep linear MSE).
- Proposition D.1: leaky-ReLU extension; the imported judge label “Proposition 3.1” is not the paper anchor.
- Proposition 3.3: HTML `#S3.Thmtheorem3` (two-layer CE with L2).
- Proposition 3.4: HTML `#S3.Thmtheorem4` (single-attention-layer transformer); it is not a two-layer MSE theorem.
- Section 3.2: HTML `#S3.SS2`.
- Theorems 4.1–4.3: HTML `#S4.Thmtheorem1`, `#S4.Thmtheorem2`, and `#S4.Thmtheorem3`.

Exact assumptions and quantifiers are reconstructed in child claim contracts. This baseline records
the source identity and the two discrepancies between imported judge labels and the paper.

