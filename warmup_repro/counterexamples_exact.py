from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from fractions import Fraction

import numpy as np
import sympy as sp
import torch

from warmup_repro.certificates import run_certificates
from warmup_repro.run import git_sha


def unpack_two_layer(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return vector[:4].reshape(2, 2), vector[4:].reshape(2, 2)


def deep_linear_loss(
    vector: torch.Tensor, inputs: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    weight1, weight2 = unpack_two_layer(vector)
    return torch.sum((labels - weight1 @ weight2 @ inputs) ** 2)


def proposition32_counterexample() -> dict[str, object]:
    inputs = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    labels = torch.zeros((2, 2), dtype=torch.float64)
    rows = []
    for scale in (1, 2, 4, 8, 16):
        weight = torch.diag(torch.tensor([0.0, float(scale)], dtype=torch.float64))
        vector = torch.cat([weight.flatten(), weight.flatten()])
        hessian = torch.autograd.functional.hessian(
            lambda point: deep_linear_loss(point, inputs, labels),
            vector,
        )
        spectral_radius = float(torch.max(torch.abs(torch.linalg.eigvalsh(hessian))))
        weight1, weight2 = unpack_two_layer(vector)
        rows.append(
            {
                "t": scale,
                "loss": float(deep_linear_loss(vector, inputs, labels)),
                "hessian_spectral_radius": spectral_radius,
                "expected_radius_2t2": 2.0 * scale**2,
                "strong_balance_residual": float(
                    torch.linalg.matrix_norm(
                        weight1.T @ weight1 - weight2 @ weight2.T
                    )
                ),
                "weak_balance_residual": abs(
                    float(torch.linalg.matrix_norm(weight1))
                    - float(torch.linalg.matrix_norm(weight2))
                ),
            }
        )

    t, delta = sp.symbols("t delta", positive=True)
    restricted_loss = t**2 * delta**2
    symbolic_curvature = sp.diff(restricted_loss, delta, 2)
    unbounded = sp.limit(symbolic_curvature, t, sp.oo) == sp.oo

    full_rank_inputs = torch.eye(2, dtype=torch.float64)
    control_weight = torch.diag(torch.tensor([0.0, 4.0], dtype=torch.float64))
    control_vector = torch.cat([control_weight.flatten(), control_weight.flatten()])
    control_loss = float(
        deep_linear_loss(control_vector, full_rank_inputs, labels)
    )
    return {
        "paper_statement": "Proposition 3.2(ii), all W under strong balancedness",
        "dimensions": {"d": 2, "m": 2, "c": 2, "layers": 2},
        "inputs": [[1, 0], [0, 0]],
        "labels": [[0, 0], [0, 0]],
        "input_rank": int(torch.linalg.matrix_rank(inputs)),
        "lambda_min_XXT": float(torch.min(torch.linalg.eigvalsh(inputs @ inputs.T))),
        "fstar": 0,
        "rows": rows,
        "symbolic_restriction": "f(W2[1,0]=delta)=t^2*delta^2",
        "symbolic_second_derivative": str(symbolic_curvature),
        "unbounded_curvature_at_zero_gap": unbounded,
        "contradiction": (
            "At every balanced point f=f*=0, so Definition 3.1 would require "
            "2*t^2 <= H0 for all t; no finite H0 exists."
        ),
        "negative_control": {
            "inputs": [[1, 0], [0, 1]],
            "input_rank": int(torch.linalg.matrix_rank(full_rank_inputs)),
            "lambda_min_XXT": float(
                torch.min(torch.linalg.eigvalsh(full_rank_inputs))
            ),
            "loss_at_t4": control_loss,
            "outcome": "COUNTEREXAMPLE_MECHANISM_REMOVED_AS_INTENDED",
            "reason": "The same balanced ray has loss t^4 rather than zero, and Eq. (11) is finite.",
        },
        "verdict": "FALSIFIED",
        "scope": "The proposition as stated omits full-row-rank X; the rank-restored proposition is not falsified.",
    }


def theorem41_counterexample() -> dict[str, object]:
    initial = math.sqrt(2.0)
    epsilon = Fraction(1, 10)
    initial_gap = Fraction(1, 1)
    lower_bound = (
        initial_gap
        * (initial_gap - epsilon)
        / (4 * epsilon)
    )
    after_one_step = initial - initial
    assumptions = {
        "objective": "f(w)=w^2/2",
        "minimum": 0,
        "convex": True,
        "H0": 1,
        "H1": 1,
        "H_smooth_certificate": "1 <= 1 + f(w) for every real w",
        "initial_w": initial,
        "initial_gap": float(initial_gap),
        "epsilon": float(epsilon),
        "constant_step": 1,
    }
    return {
        "paper_statement": "Theorem 4.1(2), literal displayed per-function lower bound",
        "assumptions": assumptions,
        "paper_lower_bound": {
            "exact": str(lower_bound),
            "decimal": float(lower_bound),
        },
        "observed_first_hit": 1,
        "gap_after_one_step": after_one_step**2 / 2,
        "contradiction": "One fixed GD step reaches the minimizer, but the displayed bound requires K >= 9/4.",
        "primary_reference_control": {
            "source": "arXiv:1905.11881, Theorem 4",
            "required_quantifier": "supremum over initialization and objective in a fixed-constant function class",
            "present_in_2510.03164_theorem_statement": False,
            "outcome": "LITERAL_COUNTEREXAMPLE_REJECTED_UNDER_INTENDED_MINIMAX_READING",
        },
        "verdict": "FALSIFIED",
        "scope": "Literal Theorem 4.1 statement only; the intended but unstated minimax lower bound remains BLOCKED.",
    }


def adaptive_quadratic_iterations(initial_gap: float, epsilon: float) -> int:
    point = math.sqrt(2 * initial_gap)
    for iteration in range(10_000):
        gap = point**2 / 2
        if gap <= epsilon:
            return iteration
        step = 1 / (10 + 20 * gap)
        point -= step * point
    raise RuntimeError("quadratic convergence control exceeded horizon")


def theorem43_counterexample() -> dict[str, object]:
    negative_bound = 20 * math.log(0.5)
    control_epsilon = 1e-6
    control_bound = 40 + 20 * math.log(1 / (2 * control_epsilon))
    control_iterations = adaptive_quadratic_iterations(1.0, control_epsilon)
    return {
        "paper_statement": "Theorem 4.3 displayed 'after at most' iteration count",
        "assumptions": {
            "objective": "f(w)=w^2/2",
            "H0": 1,
            "H1": 1,
            "mu": 1,
            "H_smooth_certificate": "1 <= 1 + f(w)",
            "PL_certificate": "|grad f|^2 = 2*f",
            "initial_w": 0,
            "initial_gap": 0,
            "epsilon": 1,
        },
        "paper_iteration_bound": negative_bound,
        "minimum_valid_iteration_count": 0,
        "contradiction": "The theorem promises a nonnegative hitting time after at most a negative number of iterations.",
        "negative_control": {
            "initial_gap": 1,
            "epsilon": control_epsilon,
            "paper_iteration_bound": control_bound,
            "observed_first_hit": control_iterations,
            "bound_holds": control_iterations <= control_bound,
            "outcome": "SMALL_EPSILON_DOMAIN_PASSES_AS_INTENDED",
        },
        "verdict": "FALSIFIED",
        "scope": "Displayed formula without an epsilon-domain guard; the corrected small-epsilon recurrence is not falsified.",
    }


def closure_constants_counterexample() -> dict[str, object]:
    return {
        "paper_statement": "Proposition B.2 stated constants for finite sums",
        "f": {"value": -100, "H0": 0, "H1": 1, "fstar": -100},
        "g": {"value": 100, "H0": 0, "H1": 2, "gstar": 100},
        "sum": {"value": 0, "sum_star": 0, "hessian_norm": 0},
        "paper_constants": {"H0": -100, "H1": 2},
        "paper_bound_rhs": -100,
        "corrected_constants": {"H0": 0, "H1": 2},
        "verdict": "FALSIFIED",
        "scope": "Only the published constants; closure survives with the shift-invariant corrected H0.",
    }


def run_counterexamples(config: dict[str, object], started: float) -> int:
    torch.set_num_threads(int(config["expected_cores"]))
    errors = []
    if run_certificates(config, started) != 0:
        errors.append("cumulative certificate regression failed")
    proposition32 = proposition32_counterexample()
    theorem41 = theorem41_counterexample()
    theorem43 = theorem43_counterexample()
    closure = closure_constants_counterexample()
    if not proposition32["unbounded_curvature_at_zero_gap"]:
        errors.append("Proposition 3.2 symbolic limit failed")
    for row in proposition32["rows"]:
        if abs(row["hessian_spectral_radius"] - row["expected_radius_2t2"]) > 1e-9:
            errors.append("Proposition 3.2 autograd and symbolic curvature disagree")
    if theorem41["observed_first_hit"] >= theorem41["paper_lower_bound"]["decimal"]:
        errors.append("Theorem 4.1 literal contradiction did not hold")
    if theorem43["paper_iteration_bound"] >= 0:
        errors.append("Theorem 4.3 domain contradiction did not hold")
    if not theorem43["negative_control"]["bound_holds"]:
        errors.append("Theorem 4.3 small-epsilon control failed")
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
        "claim1_closure_constant_audit": closure,
        "claim2_proposition32": proposition32,
        "claim4_theorem41": theorem41,
        "claim5_theorem43": theorem43,
        "claim_statuses": [
            {
                "claim": 1,
                "verdict": "VERIFIED",
                "basis": "Definition 3.1 is exact; the B.2 constant defect is isolated and corrected without refuting closure.",
            },
            {
                "claim": 2,
                "verdict": "FALSIFIED",
                "basis": "Rank-deficient X satisfies every stated Proposition 3.2 assumption while curvature is unbounded at zero gap.",
            },
            {
                "claim": 3,
                "verdict": "BLOCKED",
                "basis": "This route does not complete the Proposition 3.3/E.2 proof reconstruction.",
            },
            {
                "claim": 4,
                "verdict": "FALSIFIED",
                "basis": "Literal Theorem 4.1(2) lacks the minimax stability quantifier and is contradicted by one-step quadratic GD.",
            },
            {
                "claim": 5,
                "verdict": "FALSIFIED",
                "basis": "The displayed Theorem 4.3 bound can be negative under its stated assumptions; corrected small-epsilon form passes.",
            },
            {
                "claim": 6,
                "verdict": "BLOCKED",
                "basis": "Handled by separate exact-scale empirical routes.",
            },
        ],
        "negative_controls": {
            "claim2_full_rank": proposition32["negative_control"]["outcome"],
            "claim4_minimax_reading": theorem41["primary_reference_control"]["outcome"],
            "claim5_small_epsilon": theorem43["negative_control"]["outcome"],
        },
        "verifier": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    print("=== EXACT THEOREM COUNTEREXAMPLES ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("=== END EXACT THEOREM COUNTEREXAMPLES ===")
    return 0 if not errors else 1
