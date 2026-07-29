# Source audit

Section 3.2, Figure 3, and Remark K.1 of arXiv:2510.03164v2 specify
ResNet50 and ViT-Tiny on ImageNet32, SGD at constant LR `1e-4`, clipping
at 1, and the consecutive-stochastic-gradient local-smoothness proxy.

The cited vision repository was audited at commit
`9d9268c3a8e1d2f051d5fd66b24f3887b69edb65`. Its ImageNet32 ResNet50
uses a 3x3 stride-1 stem and `[3,4,6,3]` bottleneck stages. Table K.2 gives
ViT-Tiny as patch 4, width 192, 12 layers, 3 heads, MLP ratio 3, class
token, and linearly increasing drop-path to 0.1.
