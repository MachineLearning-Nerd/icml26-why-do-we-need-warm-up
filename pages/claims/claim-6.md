# Claim 6 — BLOCKED

## Exact paper statement

Section 3.2 reports that, for much of early training, the stochastic
local-smoothness proxy is approximately linear in stochastic training loss for
70M, 160M, and 410M PlainLM models on FineWeb and for ResNet50 and ViT-Tiny on
ImageNet32. The LM protocol uses SGD at `1e-4`, clipping at one, batch 256,
long sequences, FP16, and billion-token horizons.

## Route 1 — exact parameter-count FineWeb models

| Model | Parameters | Slope | 95% CI | Permutation p |
|---|---:|---:|---:|---:|
| 70M | 71,941,888 | −8,248 | [−37,178, 21,008] | 0.531 |
| 160M | 162,186,240 | −13,311 | [−33,825, 3,619] | 0.190 |
| 410M | 411,309,056 | −2,310 | [−67,275, 26,306] | 0.922 |

No predeclared positive-slope test passes. Float32 and float64 norm reductions
agree within `1.86e-7`.

This is not a falsification: 12 steps, micro-batch one, sequence length 16,
hashed tokens, and float32 CPU are material deviations.

- [Executable LM route](../../warmup_repro/fineweb_lm.py)
- [Raw LM JSON](../../evidence/claim6_fineweb.json)
- Run SHA: `72831108f97570f966d2f9e243154456bf8dc116`
- Run ID: `01955761-b2c2-4462-8fd8-57e6309c1de8`
- Compute: HF cpu-upgrade, 64 selected/visible CPUs, `4530.30788 s`

## Route 2 — ImageNet32 models

The exact ResNet50/ViT-Tiny route uses official ImageNet32 bytes, three seeds,
30 measured steps, bootstrap intervals, permutation controls, and independent
norm reductions.

| Model | Parameters | Pooled slope | 95% CI | R² | Permutation p |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 25,549,352 | −190,901 | [−448,990, 110,872] | 0.0139 | 0.248 |
| ViT-Tiny | 4,666,792 | 14,391 | [−21,559, 49,189] | 0.00645 | 0.455 |

Neither architecture passes the predeclared relationship criterion. The
independent reduction checker agrees within `4.16e-7`, and 500 label
permutations per seed supply the negative control. The provider marked the run
failed only after the complete `verifier: PASS` payload had printed, when its
timeout was reached.

- [Executable vision route](../../warmup_repro/imagenet32.py)
- [Raw vision JSON](../../evidence/claim6_imagenet32.json)
- Run SHA: `01426573fe559c956cf133ca8b06312b1773422e`
- Run ID: `9f00d661-9f97-4381-8a26-93aaa1baa24f`
- Compute: HF cpu-upgrade, 64 selected/visible CPUs, `23837.27177 s`
- Deviations: 30 steps per seed, batch eight, float32 CPU, first official
  training member only

## Route 3 — complete-Hessian proxy calibration

On a complete 35-parameter Hessian problem, exact curvature-loss correlation is
`0.9949`, but the changing-minibatch printed proxy correlation is `0.1947`.
The printed proxy is `2,832–28,490` versus exact Hessian norm about `0.603`;
median batch-noise/curvature-change is `18,283.97`. A same-batch control removes
the noise term, and full-batch finite differences match Hessian actions.

- [Executable calibration](../../warmup_repro/proxy_calibration.py)
- [Raw calibration JSON](../../evidence/claim6_proxy_calibration.json)

## Route 4 — dedicated literal-estimator falsification

The displayed two-iterate/two-minibatch estimator was independently implemented
with a vector-decomposition checker, same-batch control, and zero-denominator
negative control. Two HF cpu-upgrade launches failed before the code ran because
the default image did not expose `uv`. Per the two-failure repair cap, the route
was not blindly relaunched.

- [Literal estimator source](../../warmup_repro/literal_proxy.py)
- Commit: `ad6d23581a729362f17dfa5497ca2e3630a9d18d`
- Failed run IDs: `6523341e-e06e-424a-8d95-0dff541b2556`,
  `90a49ad1-129a-4330-801c-0be18970c127`
- Failure output: `uv: command not found`

## Final limitation

No route supplies the paper's full tokenizer, estimator implementation, batch,
sequence, precision, and training horizons across all five named models.
Neither verification nor valid falsification is established. Verdict:
`BLOCKED`.
