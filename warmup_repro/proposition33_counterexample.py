from __future__ import annotations

import json
import math
import os
import platform
import sys
import time

import sympy as sp
import torch

from warmup_repro.proof_certificates import run_proof_certificates
from warmup_repro.run import git_sha


def softplus(value: float) -> float:
    return math.log1p(math.exp(value))


def evaluate_counterexample(amplitude: float) -> dict[str, object]:
    lambda1 = lambda2 = 0.1
    c1 = c2 = 1 + amplitude
    lower_bound_radius = math.log(c1 / lambda1 - 1) / c1
    point_coordinate = math.sqrt(lower_bound_radius)
    phase_index = 1_000_000
    omega = (math.pi / 2 + 2 * math.pi * phase_index) / point_coordinate
    c3 = amplitude * omega

    def activation(value: torch.Tensor) -> torch.Tensor:
        return value + (amplitude / omega) * torch.sin(omega * value)

    def loss(vector: torch.Tensor) -> torch.Tensor:
        logit = vector[0] * activation(vector[1])
        return (
            torch.nn.functional.softplus(-logit)
            + lambda1 * vector[0] ** 2 / 2
            + lambda2 * vector[1] ** 2 / 2
        )

    point = torch.tensor(
        [point_coordinate, point_coordinate], dtype=torch.float64
    )
    objective = float(loss(point))
    hessian = torch.autograd.functional.hessian(loss, point)
    curvature = float(torch.linalg.matrix_norm(hessian, ord=2))

    phi = point_coordinate + amplitude / omega
    phi_prime = 1.0
    phi_second = -amplitude * omega
    logit = point_coordinate * phi
    residual = -1 / (1 + math.exp(logit))
    sigmoid_derivative = math.exp(logit) / (1 + math.exp(logit)) ** 2
    h11 = sigmoid_derivative * phi**2 + lambda1
    h12 = (
        sigmoid_derivative * phi * point_coordinate * phi_prime
        + residual * phi_prime
    )
    h22 = (
        sigmoid_derivative * (point_coordinate * phi_prime) ** 2
        + residual * point_coordinate * phi_second
        + lambda2
    )
    analytic_radius = max(
        abs((h11 + h22 + math.sqrt((h11 - h22) ** 2 + 4 * h12**2)) / 2),
        abs((h11 + h22 - math.sqrt((h11 - h22) ** 2 + 4 * h12**2)) / 2),
    )

    digits = 80
    s = sp.Float(point_coordinate, digits)
    a = sp.Float(amplitude, digits)
    w = sp.Float(omega, digits)
    lam = sp.Rational(1, 10)
    phi_hp = s + a / w
    logit_hp = s * phi_hp
    residual_hp = -1 / (1 + sp.exp(logit_hp))
    sigmoid_derivative_hp = sp.exp(logit_hp) / (1 + sp.exp(logit_hp)) ** 2
    h11_hp = sigmoid_derivative_hp * phi_hp**2 + lam
    h12_hp = sigmoid_derivative_hp * phi_hp * s + residual_hp
    h22_hp = sigmoid_derivative_hp * s**2 + residual_hp * s * (-a * w) + lam
    discriminant_hp = sp.sqrt((h11_hp - h22_hp) ** 2 + 4 * h12_hp**2)
    high_precision_radius = float(
        max(
            abs(sp.N((h11_hp + h22_hp + discriminant_hp) / 2, digits)),
            abs(sp.N((h11_hp + h22_hp - discriminant_hp) / 2, digits)),
        )
    )

    fstar_lower_bound = (
        softplus(-c1 * lower_bound_radius)
        + lambda1 * lower_bound_radius
    )
    a1 = c2**2
    a2 = c1**2
    a12 = 2 * c1 * c2
    b_mix = math.sqrt(2) * c3
    b_res = 2 * math.sqrt(2) * c2
    c_linear = (
        (2 * a1 + a12 + b_mix) / lambda1
        + (2 * a2 + a12) / lambda2
        + b_mix / 2
    )
    h1 = b_res + c_linear
    published_rhs_upper_bound = (
        lambda1
        + lambda2
        + b_res * (fstar_lower_bound + 1)
        + h1 * (objective - fstar_lower_bound)
    )
    return {
        "source_anchor": "Proposition 3.3(ii), Equation (31)",
        "model": "one-input, one-hidden-unit, one-sample CE+L2 network",
        "labels": [1],
        "input": [1],
        "lambda1": lambda1,
        "lambda2": lambda2,
        "activation": "phi(s)=s+(a/omega)sin(omega*s)",
        "activation_parameters": {
            "a": amplitude,
            "omega": omega,
            "phase_index": phase_index,
        },
        "assumption_certificates": {
            "phi_growth": "|phi(s)| <= (1+a)|s| from |sin(x)|<=|x|",
            "phi_prime": "|phi'(s)|=|1+a*cos(omega*s)| <= 1+a",
            "phi_second": "|phi''(s)|=|a*omega*sin(omega*s)| <= a*omega",
            "C1": c1,
            "C2": c2,
            "C3": c3,
            "positive_regularization": lambda1 > 0 and lambda2 > 0,
        },
        "point": [point_coordinate, point_coordinate],
        "phase_at_point": "pi/2 + 2*pi*phase_index",
        "loss_at_point": objective,
        "fstar_lower_bound": fstar_lower_bound,
        "fstar_lower_bound_proof": (
            "Set r=(u^2+v^2)/2. Since u*phi(v)<=C1*r, "
            "f(u,v)>=softplus(-C1*r)+lambda*r. Its exact minimizer is "
            "r=log(C1/lambda-1)/C1."
        ),
        "paper_constants": {
            "Bres": b_res,
            "Bmix": b_mix,
            "Clinear": c_linear,
            "H1": h1,
        },
        "paper_rhs_upper_bound_over_all_valid_fstar": published_rhs_upper_bound,
        "complete_autograd_hessian": hessian.tolist(),
        "complete_hessian_norm": curvature,
        "analytic_hessian_norm": analytic_radius,
        "high_precision_hessian_norm": high_precision_radius,
        "autograd_analytic_error": abs(curvature - analytic_radius),
        "autograd_high_precision_error": abs(curvature - high_precision_radius),
        "contradiction_margin": curvature - published_rhs_upper_bound,
        "verdict": (
            "FALSIFIED"
            if curvature > published_rhs_upper_bound
            else "NOT_FALSIFIED"
        ),
        "scope": (
            "This contradicts the printed Equation (31) constants. It does not "
            "contradict the corrected H0 that includes Clinear*fstar."
        ),
    }


def run_counterexample(config: dict[str, object], started: float) -> int:
    torch.set_num_threads(int(config["expected_cores"]))
    errors = []
    if run_proof_certificates(config, started) != 0:
        errors.append("cumulative proof-certificate regression failed")
    counterexample = evaluate_counterexample(1e-4)
    control = evaluate_counterexample(0.0)
    if counterexample["verdict"] != "FALSIFIED":
        errors.append("Proposition 3.3 contradiction did not hold")
    if counterexample["contradiction_margin"] < 50:
        errors.append("Proposition 3.3 contradiction lacks calibrated margin")
    if counterexample["autograd_analytic_error"] > 1e-9:
        errors.append("autograd and analytic Hessians disagree")
    if counterexample["autograd_high_precision_error"] > 1e-9:
        errors.append("autograd and high-precision Hessians disagree")
    if control["verdict"] != "NOT_FALSIFIED":
        errors.append("linear-activation negative control unexpectedly falsified")
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
        "claim3_counterexample": counterexample,
        "negative_control": {
            "change": "set oscillation amplitude a=0, so phi(s)=s and C3=0",
            "contradiction_margin": control["contradiction_margin"],
            "outcome": (
                "REJECTED_AS_INTENDED"
                if control["verdict"] == "NOT_FALSIFIED"
                else "UNEXPECTED_FALSIFICATION"
            ),
        },
        "claim_statuses": [
            {"claim": 1, "verdict": "VERIFIED"},
            {"claim": 2, "verdict": "FALSIFIED"},
            {
                "claim": 3,
                "verdict": "FALSIFIED",
                "basis": (
                    "A smooth activation satisfies every Proposition 3.3 "
                    "assumption but violates its printed Equation (31) bound."
                ),
            },
            {"claim": 4, "verdict": "FALSIFIED"},
            {"claim": 5, "verdict": "FALSIFIED"},
            {"claim": 6, "verdict": "BLOCKED"},
        ],
        "verifier": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    print("=== EXACT PROPOSITION 3.3 COUNTEREXAMPLE ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("=== END EXACT PROPOSITION 3.3 COUNTEREXAMPLE ===")
    return 0 if not errors else 1
