# Reproduction environment

## Pinned setup

- Python: `>=3.12,<3.13` from `pyproject.toml` and `uv.lock`
- Canonical command: `uv run --frozen python -m warmup_repro.run`
- Formal stages use deterministic seed `20260729` and one selected thread where the verifier records it.
- PyTorch and torchvision use the explicit CPU index in the lockfile.
- Claim 6's named-model routes use Hugging Face `cpu-upgrade` where recorded; the full paper-scale protocol is not claimed as reproduced.

## Reproduction boundary

Claims 1–5 use symbolic certificates, exact Hessians, analytic/high-precision
checks, and assumption-satisfying controls. Claim 6 uses separate FineWeb,
ImageNet32, calibration, and literal-estimator routes, but material deviations
in model/data scale, batch, sequence length, tokenizer, precision, horizon, or
estimator implementation prevent a verification or valid falsification.
