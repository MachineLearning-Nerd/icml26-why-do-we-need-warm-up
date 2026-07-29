# Method

The implementation reconstructs the cited PlainLM architecture: untied embeddings,
PreLN/RMSNorm blocks, RoPE, Q/K normalization, SwiGLU, depth-scaled residual
initialization, and no dropout. FineWeb rows come from the public dataset-server API
and are content-hashed. Each exact-scale model follows the paper's SGD update and
proxy for a predeclared 12-step early trajectory.

The proxy norm is independently reduced in float32 and float64. A 500-permutation
control tests whether the observed linear fit exceeds arbitrary loss/proxy pairings.
The fixed cumulative command also reruns the inherited certificate suite.
