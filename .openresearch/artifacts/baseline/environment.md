# Environment and run contract

- Fixed command: `uv run --frozen python -m warmup_repro.run`
- Python: `3.12`
- Manager: `uv`
- Environment: repository-level `.venv`
- Lockfile: `uv.lock`
- Resolver: `uv 0.11.29`
- Locked direct versions: `datasets 4.8.5`, `marimo 0.23.15`, `matplotlib 3.11.1`,
  `numpy 2.5.1`, `sympy 1.14.0`, `torch 2.13.0+cpu`, `torchvision 0.28.0+cpu`
- Seed: `20260729`
- Baseline estimated compute: one CPU core; dependency installation has uncertain runtime, so the
  run is routed through Hugging Face `cpu-upgrade`.
