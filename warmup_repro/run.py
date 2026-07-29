from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "baseline"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_historical_record(record: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if record.get("space_id") != "DineshAI/a6fo32UnpU":
        errors.append("wrong space_id")
    if record.get("sha") != "17e423af3b04b3c0fb493ccfdc26a9724a2be53c":
        errors.append("wrong judged Space SHA")
    claims = record.get("claims")
    if not isinstance(claims, list) or len(claims) != 6:
        errors.append("expected exactly six judge claims")
    else:
        verdicts = [item.get("verdict") for item in claims if isinstance(item, dict)]
        if verdicts != ["toy", "toy", "toy", "toy", "toy", "inconclusive"]:
            errors.append("unexpected historical verdict sequence")
    return errors


def git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    started = time.perf_counter()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    config = json.loads((ROOT / "warmup_repro" / "config.json").read_text())
    if config["stage"] == "proxy_calibration":
        from warmup_repro.proxy_calibration import run_proxy_calibration

        return run_proxy_calibration(config, started)

    if config["stage"] == "proposition33_counterexample":
        from warmup_repro.proposition33_counterexample import run_counterexample

        return run_counterexample(config, started)

    if config["stage"] == "proof_certificates":
        from warmup_repro.proof_certificates import run_proof_certificates

        return run_proof_certificates(config, started)

    if config["stage"] == "exact_counterexamples":
        from warmup_repro.counterexamples_exact import run_counterexamples

        return run_counterexamples(config, started)

    if config["stage"] == "certificates_and_scales":
        from warmup_repro.certificates import run_certificates

        return run_certificates(config, started)

    record_path = ARTIFACTS / "historical_verdict.json"
    record = json.loads(record_path.read_text())
    errors = validate_historical_record(record)

    expected_hash = "e01f0b81e019303024d52cf8120e868f6eb86c67a88fc9db14ecf02feb0888f9"
    observed_hash = sha256(record_path)
    if observed_hash != expected_hash:
        errors.append(f"historical verdict hash mismatch: {observed_hash}")

    manifest = [
        line
        for line in (ARTIFACTS / "judged_space_manifest.sha256").read_text().splitlines()
        if line.strip()
    ]
    if len(manifest) != 13:
        errors.append(f"protected manifest has {len(manifest)} entries, expected 13")

    corrupted = dict(record)
    corrupted["space_id"] = "wrong/space"
    negative_control_passed = bool(validate_historical_record(corrupted))
    if not negative_control_passed:
        errors.append("negative control did not fail")

    runtime = time.perf_counter() - started
    claims = [
        {
            "claim": index,
            "verdict": "BLOCKED",
            "reason": "Historical evidence is toy-scale or inconclusive; retained only as the judged control.",
        }
        for index in range(1, 7)
    ]
    result = {
        "stage": config["stage"],
        "git_sha": git_sha(),
        "seed": config["seed"],
        "historical_live_score": "5/12",
        "historical_space_sha": record["sha"],
        "historical_verdict_dataset_sha": "1b782309550dd756df4fd0f3ffe8dca8a32d5269",
        "historical_record_sha256": observed_hash,
        "protected_file_count": len(manifest),
        "negative_control": "FAILED_AS_INTENDED" if negative_control_passed else "UNEXPECTED_PASS",
        "expected_cores": config["expected_cores"],
        "actual_cpu_allocation": os.cpu_count(),
        "thread_limit": 1,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "runtime_seconds": round(runtime, 6),
        "claims": claims,
        "verifier": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    print("=== WARMUP REPRODUCTION BASELINE ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("=== END WARMUP REPRODUCTION BASELINE ===")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
