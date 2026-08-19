# Primary-source audit

## Paper identity

- Title: *Why Do We Need Warm-up? A Theoretical Perspective*
- Authors: Foivos Alimisis, Rustem Islamov, and Aurelien Lucchi
- arXiv: [2510.03164](https://arxiv.org/abs/2510.03164), audited at v2
- ICML submission: `a6fo32UnpU`
- Paper HTML SHA-256: `e73c02ea9d475c9f200d2719867b03b4557c2b3e1169953a12513601f0d86e47`
- Paper PDF SHA-256: `736bb3c117b8b698edbd4e0b567c5e3b673539e592d4c21da89a667114193b30`
- Retrieved full-text SHA-256: `6bd19f5a61acf281a0a4d83738006eee2a2087711a317c87c3235143a40eccc3`

## Exact scope

- C1: Definition 3.1 and Proposition B.1 inclusion; B.2 printed constants are a separate rejected check.
- C2: Proposition 3.2(ii) for all strongly balanced weights under the printed data assumptions.
- C3: Proposition 3.3(ii) and the printed Equation (31) constants under global activation bounds.
- C4: Theorem 4.1(3)'s fixed-step lower bound under the Equation (63) cap; Theorem 4.2 is not contradicted.
- C5: Theorem 4.3's displayed iteration expression without an epsilon-domain guard.
- C6: Section 3.2's named-model local-smoothness relationship with the paper's full experimental protocol.

Every counterexample is checked against the displayed assumptions and includes
a control. Claim 6 is not falsified by downscaled experiments because those
experiments do not match the paper's complete protocol.
