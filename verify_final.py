#!/usr/bin/env python3
"""Verify the committed claim, counterexample, blocker, and attribution contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_STATUS = "PARTIAL_C1_VERIFIED_C2_C3_C4_C5_FALSIFIED_C6_BLOCKED_HISTORICAL_SCORE_5_OF_12_NO_CURRENT_SCORE"
EXPECTED_BRANCHES = {
    "audit/assumption-satisfying-falsification",
    "audit/c1-certificates-dimension-sweeps",
    "audit/c1-symbolic-certificates",
    "audit/c2-c5-exact-counterexamples",
    "audit/c3-proposition33-counterexample",
    "audit/c4-class-stable-counterexample",
    "audit/c6-fineweb-lm-curvature",
    "audit/c6-imagenet32-curvature",
    "audit/c6-literal-proxy",
    "audit/c6-proxy-calibration",
    "historical/judged-5-of-12-baseline",
    "main",
    "release/cumulative-claim-candidate",
    "release/final-evaluator-gate",
}
EXPECTED_COMMITS = 36
CANONICAL_IDENTITY = "MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>"
CLAIM_IDS = ["C1", "C2", "C3", "C4", "C5", "C6"]
EXPECTED_CLAIM_STATUSES = {
    "C1": "VERIFIED_SCOPED",
    "C2": "FALSIFIED_SCOPED",
    "C3": "FALSIFIED_SCOPED",
    "C4": "FALSIFIED_SCOPED",
    "C5": "FALSIFIED_SCOPED",
    "C6": "BLOCKED_PROTOCOL",
}


def load(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"verification failed: {message}")


def published_branches() -> set[str]:
    remote = git("for-each-ref", "refs/remotes/origin", "--format=%(refname:short)").splitlines()
    remote = {
        name.removeprefix("origin/")
        for name in remote
        if name.startswith("origin/") and name != "origin/HEAD"
    }
    if remote:
        return remote
    return set(git("for-each-ref", "refs/heads", "--format=%(refname:short)").splitlines())


def main() -> None:
    claims = load("claims.json")
    verdicts = load("reproduction_verdicts.json")
    manifest = load("EVIDENCE_MANIFEST.json")
    state = load("AUTONOMOUS_STATE.json")
    c1 = load("evidence/claim1_definition.json")
    c2 = load("evidence/claim2_counterexample.json")
    c3 = load("evidence/claim3_counterexample.json")
    c4 = load("evidence/claim4_counterexample.json")
    c5 = load("evidence/claim5_counterexample.json")
    c6_fineweb = load("evidence/claim6_fineweb.json")
    c6_imagenet = load("evidence/claim6_imagenet32.json")
    c6_proxy = load("evidence/claim6_proxy_calibration.json")
    gate = load("release/gate-results.json")
    logbook = load("logbook.json")

    require(claims["overall_status"] == EXPECTED_STATUS, "claims overall status")
    require(state["overall_status"] == EXPECTED_STATUS, "autonomous state overall status")
    require([claim["id"] for claim in claims["claims"]] == CLAIM_IDS, "claim ordering")
    require({claim["id"]: claim["status"] for claim in claims["claims"]} == EXPECTED_CLAIM_STATUSES, "claim statuses")
    require(verdicts["claim_statuses"] == EXPECTED_CLAIM_STATUSES, "verdict statuses")

    require(all((ROOT / path).exists() for path in manifest["required_paths"]), "manifest paths")
    require(manifest["controls"]["source_pinned"], "source pin")
    require(manifest["controls"]["independent_checkers"], "independent checkers")
    require(manifest["controls"]["negative_controls_recorded"], "negative controls")
    require(manifest["controls"]["scope_limit_for_repaired_theorems"], "repaired-theorem boundary")
    require(manifest["controls"]["claim6_protocol_blocker_visible"], "Claim 6 blocker")

    require(c1["verdict"] == "VERIFIED", "Claim 1 verdict")
    require(c1["independent_checker"]["pass"] is True, "Claim 1 HVP checker")
    require(c1["independent_checker"]["relative_error"] < 1e-8, "Claim 1 HVP error")
    require(c2["verdict"] == "FALSIFIED", "Claim 2 verdict")
    require(c2["counterexample"]["rows"][-1]["hessian_norm"] == 512, "Claim 2 curvature")
    require(c2["negative_control"]["outcome"] == "COUNTEREXAMPLE_MECHANISM_REMOVED_AS_INTENDED", "Claim 2 control")
    require(c3["verdict"] == "FALSIFIED", "Claim 3 verdict")
    require(c3["result"]["complete_hessian_norm"] > c3["result"]["paper_rhs_upper_bound_over_all_valid_fstar"], "Claim 3 strict contradiction")
    require(c3["negative_control"]["outcome"] == "REJECTED_AS_INTENDED", "Claim 3 control")
    require(c4["verdict"] == "FALSIFIED", "Claim 4 verdict")
    require(c4["result"]["observed_first_hit"] < c4["result"]["paper_iteration_lower_bound"], "Claim 4 strict contradiction")
    require(c4["negative_control"]["outcome"] == "NONCONTRADICTORY_TARGET_PASSES_AS_INTENDED", "Claim 4 control")
    require(c5["verdict"] == "FALSIFIED", "Claim 5 verdict")
    require(c5["counterexample"]["paper_iteration_bound"] < 0, "Claim 5 negative iteration bound")
    require(c5["negative_control"]["outcome"] == "SMALL_EPSILON_DOMAIN_PASSES_AS_INTENDED", "Claim 5 control")
    require(c6_fineweb["verdict"] == "BLOCKED", "FineWeb boundary")
    require(c6_imagenet["verdict"] == "BLOCKED", "ImageNet boundary")
    require(c6_proxy["verdict"] == "BLOCKED", "proxy calibration boundary")

    require(gate["status"] == "PASS", "release gate")
    require(gate["candidate_json_valid"] is True, "candidate JSON")
    require(gate["cumulative_verifier"]["status"] == "PASS", "cumulative verifier")
    require(logbook["space_id"] == "DineshAI/a6fo32UnpU", "Space identity")
    require(verdicts["historical_external_result"]["score"] == "5/12", "historical score")
    require(verdicts["historical_external_result"]["current_score_claim"] is False, "current score claim")
    require(verdicts["publication"]["publication_allowed"] is False, "publication allowed")
    require(verdicts["publication"]["author_endorsement_claimed"] is False, "author endorsement")

    branches = published_branches()
    require(branches == EXPECTED_BRANCHES, "published branch set")
    require(not any(branch.startswith("orx/") for branch in branches), "legacy orx branch")
    require(int(git("rev-list", "--all", "--count")) == EXPECTED_COMMITS, "reachable commit count")
    identities = git("log", "--all", "--format=%an <%ae>\n%cn <%ce>").splitlines()
    require(identities and all(identity == CANONICAL_IDENTITY for identity in identities), "canonical commit identity")
    messages = git("log", "--all", "--format=%B")
    require("co-authored-by:" not in messages.lower(), "co-author trailer")

    print(
        "FINAL_AUDIT=VERIFIED "
        f"branches={len(branches)} commits={EXPECTED_COMMITS} "
        "claims=C1_verified_scoped,C2:C5_falsified_scoped,C6_blocked_protocol "
        "historical_score=5/12 current_score_claim=false publication_allowed=false"
    )


if __name__ == "__main__":
    main()
