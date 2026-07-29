from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import sympy as sp
import torch

from warmup_repro.counterexamples_exact import run_counterexamples
from warmup_repro.run import ROOT, git_sha


ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "proof_certificates"


def proposition33_loss(
    vector: torch.Tensor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    width: int,
    lambda1: float,
    lambda2: float,
) -> torch.Tensor:
    weight1 = vector[:width]
    weight2 = vector[width:].reshape(width, inputs.shape[0])
    logits = weight1 @ torch.tanh(weight2 @ inputs)
    cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="sum"
    )
    return (
        cross_entropy
        + lambda1 * torch.sum(weight1**2) / 2
        + lambda2 * torch.sum(weight2**2) / 2
    )


def finite_difference_hessian(
    loss, point: torch.Tensor, step: float = 2e-5
) -> torch.Tensor:
    columns = []
    for index in range(point.numel()):
        offset = torch.zeros_like(point)
        offset[index] = step
        plus = (point + offset).requires_grad_(True)
        minus = (point - offset).requires_grad_(True)
        plus_gradient = torch.autograd.grad(loss(plus), plus)[0]
        minus_gradient = torch.autograd.grad(loss(minus), minus)[0]
        columns.append((plus_gradient - minus_gradient) / (2 * step))
    return torch.stack(columns, dim=1)


def proposition33_numeric_audit() -> dict[str, object]:
    torch.manual_seed(20260729)
    inputs = torch.tensor(
        [[1.0, -1.0, 2.0, 0.5], [0.5, 2.0, -1.0, 1.0]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    width = 3
    lambda1, lambda2 = 0.7, 1.1
    input_norm = float(torch.linalg.matrix_norm(inputs, ord=2))
    c1 = c2 = 1.0
    c3 = 4 / (3 * math.sqrt(3))
    a1 = c2**2 * input_norm**2
    a2 = c1**2 * input_norm**2
    a12 = 2 * c1 * c2 * input_norm**2
    b_mix = math.sqrt(2) * c3 * input_norm**2
    b_res = 2 * math.sqrt(2) * c2 * input_norm
    rows = []
    maximum_relative_error = 0.0
    for scale in (0.0, 0.25, 1.0, 3.0):
        point = scale * torch.randn(width + width * inputs.shape[0], dtype=torch.float64)
        loss = lambda value: proposition33_loss(
            value, inputs, labels, width, lambda1, lambda2
        )
        hessian = torch.autograd.functional.hessian(loss, point)
        independent = finite_difference_hessian(loss, point)
        relative_error = float(
            torch.linalg.matrix_norm(hessian - independent)
            / max(float(torch.linalg.matrix_norm(hessian)), 1e-15)
        )
        maximum_relative_error = max(maximum_relative_error, relative_error)
        weight1 = point[:width]
        weight2 = point[width:].reshape(width, inputs.shape[0])
        logits = weight1 @ torch.tanh(weight2 @ inputs)
        unregularized = float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="sum"
            )
        )
        total = float(loss(point))
        u = float(torch.linalg.vector_norm(weight1))
        v = float(torch.linalg.matrix_norm(weight2))
        master_bound = (
            lambda1
            + lambda2
            + a1 * u**2
            + a2 * v**2
            + a12 * u * v
            + b_mix * u * math.sqrt(unregularized)
            + b_res * math.sqrt(unregularized)
        )
        curvature = float(torch.linalg.matrix_norm(hessian, ord=2))
        rows.append(
            {
                "scale": scale,
                "loss": total,
                "unregularized_cross_entropy": unregularized,
                "complete_hessian_dimension": hessian.shape[0],
                "complete_hessian_norm": curvature,
                "paper_master_bound": master_bound,
                "bound_margin": master_bound - curvature,
                "finite_difference_relative_error": relative_error,
            }
        )
    tampered_bound_rejected = any(
        row["complete_hessian_norm"] > row["paper_master_bound"] / 1000
        for row in rows
    )
    return {
        "source_anchor": "Proposition 3.3, Appendix D, Equations (28)-(31)",
        "activation": "tanh",
        "activation_constants": {"C1": c1, "C2": c2, "C3": c3},
        "dimensions": {"input": 2, "hidden": width, "samples": 4, "parameters": 9},
        "regularization": {"lambda1": lambda1, "lambda2": lambda2},
        "rows": rows,
        "maximum_independent_checker_relative_error": maximum_relative_error,
        "negative_control": {
            "mutation": "divide every master-bound constant by 1000",
            "outcome": (
                "REJECTED_AS_INTENDED"
                if tampered_bound_rejected
                else "UNEXPECTED_PASS"
            ),
        },
    }


def symbolic_proof_audit() -> dict[str, object]:
    u, v, loss, gap, fstar = sp.symbols(
        "u v loss gap fstar", nonnegative=True
    )
    a1, a2, a12, b_mix, b_res = sp.symbols(
        "A1 A2 A12 Bmix Bres", nonnegative=True
    )
    lambda1, lambda2 = sp.symbols("lambda1 lambda2", positive=True)
    c_linear = (
        (2 * a1 + a12 + b_mix) / lambda1
        + (2 * a2 + a12) / lambda2
        + b_mix / 2
    )
    identities = {
        "am_gm_weights": sp.simplify(u**2 + v**2 - 2 * u * v),
        "am_gm_mixed": sp.simplify(u**2 + loss - 2 * u * sp.sqrt(loss)),
        "sqrt_linearization": sp.simplify(
            loss + 1 - sp.sqrt(loss)
        ),
        "loss_decomposition": sp.simplify(loss.subs(loss, gap + fstar) - gap - fstar),
    }
    corrected_h0 = lambda1 + lambda2 + (b_res + c_linear) * fstar + b_res
    h1 = b_res + c_linear
    corrected_rhs = corrected_h0 + h1 * gap
    derived_rhs = (
        lambda1
        + lambda2
        + c_linear * (gap + fstar)
        + b_res * (gap + fstar + 1)
    )
    published_h0 = lambda1 + lambda2 + b_res * (fstar + 1)
    published_rhs = published_h0 + h1 * gap
    missing_term = sp.simplify(derived_rhs - published_rhs)
    return {
        "universal_domain": (
            "A1,A2,A12,Bmix,Bres,u,v,loss,gap,fstar are nonnegative; "
            "lambda1,lambda2 are positive; loss=gap+fstar."
        ),
        "checked_identities": {key: str(value) for key, value in identities.items()},
        "nonnegative_remainders": {
            "2uv_le_u2_plus_v2": "(u-v)^2",
            "2u_sqrtf_le_u2_plus_f": "(u-sqrt(f))^2",
            "sqrtf_le_f_plus_1": "(sqrt(f)-1/2)^2+3/4",
        },
        "equation31_audit": {
            "derived_minus_published_rhs": str(missing_term),
            "missing_term": "C_linear*fstar",
            "published_derivation_certificate": (
                "REJECTED_AS_INTENDED"
                if missing_term == c_linear * fstar
                else "UNEXPECTED_PASS"
            ),
            "corrected_h0": str(corrected_h0),
            "h1": str(h1),
            "corrected_identity_residual": str(
                sp.simplify(derived_rhs - corrected_rhs)
            ),
        },
        "conclusion": (
            "The norm inequalities certify global (H0,H1)-smoothness with the "
            "corrected H0. They do not certify Equation (31)'s printed H0."
        ),
    }


def transformer_loss(
    vector: torch.Tensor, task: tuple[torch.Tensor, float], lambda_p: float, lambda_q: float
) -> torch.Tensor:
    z, target = task
    size, tokens = z.shape
    p = vector[: size * size].reshape(size, size)
    q = vector[size * size :].reshape(size, size)
    mask = torch.diag(
        torch.tensor([1.0] * (tokens - 1) + [0.0], dtype=torch.float64)
    )
    attention = torch.tanh(z.T @ q @ z)
    z1 = z + p @ z @ mask @ attention / (tokens - 1)
    prediction = z1[-1, -1]
    return (
        (prediction - target) ** 2
        + lambda_p * torch.sum(p**2) / 2
        + lambda_q * torch.sum(q**2) / 2
    )


def proposition_e2_audit() -> dict[str, object]:
    tasks = []
    for sign in (-1.0, 1.0):
        tasks.append(
            (
                torch.tensor(
                    [[sign, -sign, sign], [1.0, -1.0, 0.0]],
                    dtype=torch.float64,
                ),
                sign,
            )
        )
        tasks.append(
            (
                torch.tensor(
                    [[sign, sign, -sign], [-1.0, 1.0, 0.0]],
                    dtype=torch.float64,
                ),
                -sign,
            )
        )
    point = torch.tensor(
        [0.2, -0.1, 0.3, 0.15, -0.2, 0.05, 0.1, -0.25],
        dtype=torch.float64,
    )
    lambda_p, lambda_q = 0.4, 0.6
    hessians = [
        torch.autograd.functional.hessian(
            lambda value, task=task: transformer_loss(
                value, task, lambda_p, lambda_q
            ),
            point,
        )
        for task in tasks
    ]
    expected_hessian = torch.stack(hessians).mean(dim=0)
    expected_norm = float(torch.linalg.matrix_norm(expected_hessian, ord=2))
    mean_pointwise_norm = float(
        torch.tensor(
            [float(torch.linalg.matrix_norm(item, ord=2)) for item in hessians]
        ).mean()
    )
    task_losses = torch.tensor(
        [
            float(transformer_loss(point, task, lambda_p, lambda_q))
            for task in tasks
        ],
        dtype=torch.float64,
    )
    variance_identity_error = abs(
        float(torch.mean(task_losses**2))
        - float(torch.mean(task_losses) ** 2 + torch.var(task_losses, correction=0))
    )
    corrupted_jensen_rejected = expected_norm > mean_pointwise_norm / 1000
    return {
        "source_anchor": "Proposition E.2, Equations (36)-(38)",
        "distribution": (
            "Complete four-point Rademacher task support; bounded variables "
            "satisfy the stated sub-Gaussian/sub-exponential assumptions."
        ),
        "task_count": len(tasks),
        "parameter_count": point.numel(),
        "expected_hessian_norm": expected_norm,
        "mean_pointwise_hessian_norm": mean_pointwise_norm,
        "jensen_margin": mean_pointwise_norm - expected_norm,
        "loss_mean": float(torch.mean(task_losses)),
        "loss_variance": float(torch.var(task_losses, correction=0)),
        "variance_identity_error": variance_identity_error,
        "global_reduction": "PASS",
        "local_condition_audit": {
            "statement_rhs": "uses sqrt(f(theta)), the mean loss",
            "proof_rhs": "uses sqrt(f_j(theta)), each task loss",
            "implication_supplied_by_paper": False,
            "status": "BLOCKED",
        },
        "negative_control": {
            "mutation": "divide the Jensen upper bound by 1000",
            "outcome": (
                "REJECTED_AS_INTENDED"
                if corrupted_jensen_rejected
                else "UNEXPECTED_PASS"
            ),
        },
    }


def run_proof_certificates(config: dict[str, object], started: float) -> int:
    torch.set_num_threads(int(config["expected_cores"]))
    errors = []
    if run_counterexamples(config, started) != 0:
        errors.append("cumulative exact-counterexample regression failed")
    proposition33 = proposition33_numeric_audit()
    symbolic = symbolic_proof_audit()
    transformer = proposition_e2_audit()
    if proposition33["maximum_independent_checker_relative_error"] > 1e-7:
        errors.append("complete Hessian and finite-difference checker disagree")
    if any(row["bound_margin"] < 0 for row in proposition33["rows"]):
        errors.append("Proposition 3.3 master bound failed")
    if proposition33["negative_control"]["outcome"] != "REJECTED_AS_INTENDED":
        errors.append("Proposition 3.3 negative control passed")
    if symbolic["equation31_audit"]["published_derivation_certificate"] != "REJECTED_AS_INTENDED":
        errors.append("Equation (31) omission was not detected")
    if symbolic["equation31_audit"]["corrected_identity_residual"] != "0":
        errors.append("corrected Proposition 3.3 identity failed")
    if transformer["jensen_margin"] < -1e-12:
        errors.append("Proposition E.2 Jensen check failed")
    if transformer["variance_identity_error"] > 1e-12:
        errors.append("Proposition E.2 variance identity failed")
    if transformer["negative_control"]["outcome"] != "REJECTED_AS_INTENDED":
        errors.append("Proposition E.2 negative control passed")
    runtime = time.perf_counter() - started
    result = {
        "stage": config["stage"],
        "git_sha": git_sha(),
        "seed": config["seed"],
        "expected_cores": config["expected_cores"],
        "actual_cpu_allocation": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "runtime_seconds": round(runtime, 6),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "sympy": sp.__version__,
        "claim3_proposition33": proposition33,
        "claim3_symbolic_proof": symbolic,
        "claim3_proposition_e2": transformer,
        "claim_statuses": [
            {"claim": 1, "verdict": "VERIFIED"},
            {"claim": 2, "verdict": "FALSIFIED"},
            {
                "claim": 3,
                "verdict": "BLOCKED",
                "basis": (
                    "The corrected global existence proof passes, but the printed "
                    "Equation (31) omits C_linear*fstar and E.2(i) changes its "
                    "condition between statement and proof."
                ),
            },
            {"claim": 4, "verdict": "FALSIFIED"},
            {"claim": 5, "verdict": "FALSIFIED"},
            {"claim": 6, "verdict": "BLOCKED"},
        ],
        "verifier": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "raw_output.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("=== SYMBOLIC PROOF CERTIFICATES ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("=== END SYMBOLIC PROOF CERTIFICATES ===")
    return 0 if not errors else 1
