from __future__ import annotations

import json
import math
import os
import platform
import sys
import time

import sympy as sp
import torch

from warmup_repro.proxy_calibration import run_proxy_calibration
from warmup_repro.run import git_sha


def objective(point: float) -> float:
    magnitude = abs(point)
    if magnitude <= 1:
        return point**2
    return 2 * math.exp(magnitude - 1) - 1


def gradient(point: float) -> float:
    magnitude = abs(point)
    if magnitude <= 1:
        return 2 * point
    return math.copysign(2 * math.exp(magnitude - 1), point)


def curvature(point: float) -> float:
    if abs(point) <= 1:
        return 2
    return 2 * math.exp(abs(point) - 1)


def theorem41_counterexample() -> dict[str, object]:
    h0, h1, mu = 2.0, 1.0, 2.0
    initial_loss = 1.1
    initial = 1 + math.log((initial_loss + 1) / 2)
    step = initial / (initial_loss + 1)
    stability_cap = 2 * (math.log(initial_loss) + 1) / initial_loss
    epsilon = 1e-6
    paper_lower_bound = (
        h1
        / (4 * mu)
        * initial_loss
        / (math.log(initial_loss) + 1)
        * math.log(initial_loss / epsilon)
    )
    next_point = initial - step * gradient(initial)
    control_epsilon = 0.1
    control_lower_bound = (
        h1
        / (4 * mu)
        * initial_loss
        / (math.log(initial_loss) + 1)
        * math.log(initial_loss / control_epsilon)
    )

    w = sp.symbols("w", positive=True)
    inner = w**2
    outer = 2 * sp.exp(w - 1) - 1
    boundary_residuals = [
        sp.simplify(sp.diff(inner, w, order).subs(w, 1)
                    - sp.diff(outer, w, order).subs(w, 1))
        for order in (0, 1, 2)
    ]
    z = sp.symbols("z", positive=True)
    exact_iterations = sp.log(sp.Symbol("ratio", positive=True)) / (
        -2 * sp.log(1 - z)
    )
    paper_replacement = sp.log(sp.Symbol("ratio", positive=True)) / (2 * z)
    replacement_difference_series = sp.series(
        paper_replacement - exact_iterations, z, 0, 3
    )
    return {
        "source_anchor": "Theorem 4.1(3), Appendix J, Equations (63) and (65)",
        "objective": {
            "inside": "f(w)=w^2 for |w|<=1",
            "outside": "f(w)=2*exp(|w|-1)-1 for |w|>1",
            "fstar": 0,
        },
        "assumption_certificates": {
            "C2_boundary_residuals_orders_0_1_2": [
                str(item) for item in boundary_residuals
            ],
            "convex": True,
            "strongly_convex_mu": mu,
            "PL_mu": mu,
            "H0": h0,
            "H1": h1,
            "H_smooth_inside": "2 <= 2 + f(w)",
            "H_smooth_outside": "f''(w)=f(w)+1 <= f(w)+2",
        },
        "initial_point": initial,
        "initial_loss": objective(initial),
        "constant_step": step,
        "equation63_stability_cap": stability_cap,
        "step_below_stability_cap": step <= stability_cap,
        "point_after_one_step": next_point,
        "loss_after_one_step": objective(next_point),
        "epsilon": epsilon,
        "observed_first_hit": 1,
        "paper_iteration_lower_bound": paper_lower_bound,
        "contradiction_margin_iterations": paper_lower_bound - 1,
        "proof_error": {
            "exact_quadratic_hitting_expression": str(exact_iterations),
            "paper_replacement_expression": str(paper_replacement),
            "replacement_minus_exact_series": str(replacement_difference_series),
            "reason": (
                "-log(1-z)>=z makes the exact necessary lower bound no larger "
                "than log(ratio)/(2z); Appendix J uses the latter as a lower bound."
            ),
        },
        "negative_control": {
            "epsilon": control_epsilon,
            "paper_iteration_lower_bound": control_lower_bound,
            "observed_first_hit": 1,
            "bound_holds": 1 >= control_lower_bound,
            "outcome": (
                "NONCONTRADICTORY_TARGET_PASSES_AS_INTENDED"
                if 1 >= control_lower_bound
                else "UNEXPECTED_CONTRADICTION"
            ),
        },
        "verdict": "FALSIFIED" if 1 < paper_lower_bound else "NOT_FALSIFIED",
        "scope": (
            "Part (3) of Theorem 4.1 under its Appendix J stability cap. "
            "The adaptive upper bound in Theorem 4.2 is not contradicted."
        ),
    }


def run_theorem41(config: dict[str, object], started: float) -> int:
    torch.set_num_threads(int(config["expected_cores"]))
    errors = []
    if run_proxy_calibration(config, started) != 0:
        errors.append("cumulative verifier regression failed")
    counterexample = theorem41_counterexample()
    if counterexample["verdict"] != "FALSIFIED":
        errors.append("Theorem 4.1(3) contradiction did not hold")
    if not counterexample["step_below_stability_cap"]:
        errors.append("constant step exceeds Equation (63) cap")
    if any(
        value != "0"
        for value in counterexample["assumption_certificates"][
            "C2_boundary_residuals_orders_0_1_2"
        ]
    ):
        errors.append("piecewise objective is not C2")
    if abs(counterexample["point_after_one_step"]) > 1e-14:
        errors.append("one-step hitting certificate failed")
    if (
        counterexample["negative_control"]["outcome"]
        != "NONCONTRADICTORY_TARGET_PASSES_AS_INTENDED"
    ):
        errors.append("Theorem 4.1 negative control failed")
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
        "sympy": sp.__version__,
        "claim4_counterexample": counterexample,
        "claim_statuses": [
            {"claim": 1, "verdict": "VERIFIED"},
            {"claim": 2, "verdict": "FALSIFIED"},
            {"claim": 3, "verdict": "FALSIFIED"},
            {
                "claim": 4,
                "verdict": "FALSIFIED",
                "basis": (
                    "The paper's own C2 strongly-convex construction and an "
                    "Eq.63-admissible step contradict Theorem 4.1(3)."
                ),
            },
            {"claim": 5, "verdict": "FALSIFIED"},
            {"claim": 6, "verdict": "BLOCKED"},
        ],
        "verifier": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    print("=== CLASS-STABLE THEOREM 4.1 COUNTEREXAMPLE ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("=== END CLASS-STABLE THEOREM 4.1 COUNTEREXAMPLE ===")
    return 0 if not errors else 1
