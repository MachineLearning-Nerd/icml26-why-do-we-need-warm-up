# Source audit

The exact contract is Section 3.2 and Remark K.1 of arXiv:2510.03164v2:
70M, 160M, and 410M NanoGPT-derived language models on FineWeb, SGD at constant
LR `1e-4`, gradient clipping at 1, and the consecutive-stochastic-gradient
local-smoothness proxy.

The paper table appears internally inconsistent: its displayed 410M row repeats the
6-layer/512-wide 70M configuration. The cited PlainLM source at commit
`bdeaea7796cbe95d0ae0dd692957c86814542baf` and parameter-count reconstruction
resolve the configurations as 6x512 (71,941,888), 12x768 (162,186,240), and
24x1024 (411,309,056), with untied 50,280-token embeddings.
