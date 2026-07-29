# Evaluator-blind review

## Review 1

Starting only from `CURRENT.md`, the reviewer opened:

1. `pages/current/page.md`;
2. all six `pages/claims/claim-*.md` pages;
3. every linked `evidence/*.json` file;
4. every linked verifier source;
5. `pyproject.toml`, `uv.lock`, `visibility_matrix.md`, and
   `release_report.md`.

The first pass could not locate machine-readable ImageNet32 evidence, complete
checker/control summaries, or final vision statistics inline. These were
treated as missing.

Fixes: added `evidence/claim6_imagenet32.json`, added the pooled slopes,
intervals, permutation tests, independent-checker error, runtime, and
deviations to Claim 6, and added aggregate checker/control outputs.

## Review 2 after fixes

The reviewer repeated the same canonical traversal without repository
knowledge. It located the current verifier before the historical page and
found, for every claim: exact statement and quantifier scope, numerical
assumption audit, executable source, fixed command, pinned environment, inline
results, raw JSON, checker, negative control, limitations, SHA, seed, and
CPU/runtime record.

Files opened in order:

1. `CURRENT.md`
2. `pages/current/page.md`
3. `pages/claims/claim-1.md`
4. `evidence/claim1_definition.json`
5. `pages/claims/claim-2.md`
6. `evidence/claim2_counterexample.json`
7. `pages/claims/claim-3.md`
8. `evidence/claim3_counterexample.json`
9. `pages/claims/claim-4.md`
10. `evidence/claim4_counterexample.json`
11. `pages/claims/claim-5.md`
12. `evidence/claim5_counterexample.json`
13. `pages/claims/claim-6.md`
14. `evidence/claim6_fineweb.json`
15. `evidence/claim6_imagenet32.json`
16. `evidence/claim6_proxy_calibration.json`
17. `warmup_repro/run.py` and the linked claim modules
18. `pyproject.toml` and `uv.lock`
19. `visibility_matrix.md`
20. `release_report.md`

Conclusion: Claims 1–5 are directly reviewable as scoped exact results. Claim
6 is directly reviewable as `BLOCKED`; the absent paper-scale capability is
explicit. No conclusion required a hidden OpenResearch log or unpublished
branch.
