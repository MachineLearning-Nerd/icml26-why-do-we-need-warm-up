from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

from warmup_repro.run import ROOT, git_sha, validate_historical_record


ARTIFACTS = ROOT / ".openresearch" / "artifacts"


def paper_sum_constant(
    h0_f: Fraction,
    h1_f: Fraction,
    f_star: Fraction,
    h0_g: Fraction,
    h1_g: Fraction,
    g_star: Fraction,
    h_star: Fraction,
) -> tuple[Fraction, Fraction]:
    h1 = max(h1_f, h1_g)
    h0 = h0_f + h0_g + h1 * h_star - h1_f * f_star - h1_g * g_star
    return h0, h1


def corrected_sum_constant(
    h0_f: Fraction,
    h1_f: Fraction,
    f_star: Fraction,
    h0_g: Fraction,
    h1_g: Fraction,
    g_star: Fraction,
    h_star: Fraction,
) -> tuple[Fraction, Fraction]:
    h1 = max(h1_f, h1_g)
    h0 = h0_f + h0_g + h1 * (h_star - f_star - g_star)
    return h0, h1


def slope_proxy_counterexample() -> dict[str, float | bool]:
    points = np.linspace(1.0, 2.0, 30)
    values = np.exp(points * points) - 1.0
    curvatures = (4.0 * points * points + 2.0) * np.exp(points * points)
    slope = float(np.polyfit(np.log(values), np.log(curvatures), 1)[0])
    ratio_at_2 = float(curvatures[-1] / values[-1])
    w_far = 10.0
    ratio_at_10 = float(
        ((4.0 * w_far * w_far + 2.0) * math.exp(w_far * w_far))
        / (math.exp(w_far * w_far) - 1.0)
    )
    return {
        "finite_ray_loglog_slope": slope,
        "historical_threshold_accepts": slope <= 1.2,
        "curvature_to_gap_ratio_w2": ratio_at_2,
        "curvature_to_gap_ratio_w10": ratio_at_10,
        "global_affine_bound_exists": False,
    }


def run_falsification(config: dict[str, object], started: float) -> int:
    errors: list[str] = []
    historical = json.loads(
        (ARTIFACTS / "baseline" / "historical_verdict.json").read_text()
    )
    errors.extend(validate_historical_record(historical))

    f_star = Fraction(-100)
    g_star = Fraction(100)
    h_star = Fraction(0)
    paper_h0, paper_h1 = paper_sum_constant(
        Fraction(0),
        Fraction(1),
        f_star,
        Fraction(0),
        Fraction(2),
        g_star,
        h_star,
    )
    corrected_h0, corrected_h1 = corrected_sum_constant(
        Fraction(0),
        Fraction(1),
        f_star,
        Fraction(0),
        Fraction(2),
        g_star,
        h_star,
    )
    paper_bound_holds = Fraction(0) <= paper_h0
    corrected_bound_holds = Fraction(0) <= corrected_h0

    a, b, fs, gs, hs = sp.symbols("a b f_s g_s h_s", real=True)
    symbolic_paper = sp.Max(a, b) * hs - a * fs - b * gs
    symbolic_substitution = sp.simplify(
        symbolic_paper.subs({a: 1, b: 2, fs: -100, gs: 100, hs: 0})
    )
    independent_check_agrees = symbolic_substitution == paper_h0 == -100

    if paper_bound_holds:
        errors.append("published Proposition B.2 counterexample unexpectedly passed")
    if not corrected_bound_holds or corrected_h0 != 0 or corrected_h1 != 2:
        errors.append("corrected closure constant negative control failed")
    if not independent_check_agrees:
        errors.append("SymPy and exact-rational checkers disagree")

    theorem43_quadratic = {
        "objective": "f(w)=w^2/2",
        "H0": 1.0,
        "H1": 1.0,
        "mu": 1.0,
        "w0": 0.0,
        "epsilon": 1.0,
        "assumption_H_smooth": True,
        "assumption_mu_PL": True,
        "target_already_met_at_K0": True,
        "stated_iteration_bound": 20.0 * math.log(0.5),
    }
    theorem43_domain_omission = theorem43_quadratic["stated_iteration_bound"] < 0
    if not theorem43_domain_omission:
        errors.append("Theorem 4.3 missing-epsilon-domain control did not trigger")

    slope_control = slope_proxy_counterexample()
    if not slope_control["historical_threshold_accepts"]:
        errors.append("finite-ray historical proxy did not accept its negative control")

    runtime = time.perf_counter() - started
    results = {
        "stage": config["stage"],
        "git_sha": git_sha(),
        "seed": config["seed"],
        "expected_cores": config["expected_cores"],
        "actual_cpu_allocation": os.cpu_count(),
        "thread_limit": 1,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "runtime_seconds": round(runtime, 6),
        "historical_regression": "PASS" if not validate_historical_record(historical) else "FAIL",
        "claim1_closure_constant": {
            "source_anchor": "Proposition B.2 / HTML #A2.Thmtheorem2",
            "functions": "f(w)=-100 and g(w)=100 for all real w",
            "assumptions": {
                "f_minimum": -100,
                "g_minimum": 100,
                "h_minimum": 0,
                "f_constants": [0, 1],
                "g_constants": [0, 2],
                "both_H_smooth": True,
            },
            "published_constants": [float(paper_h0), float(paper_h1)],
            "published_bound_at_any_w": f"0 <= {paper_h0}",
            "published_bound_holds": paper_bound_holds,
            "independent_sympy_value": int(symbolic_substitution),
            "independent_check_agrees": independent_check_agrees,
            "corrected_constants": [float(corrected_h0), float(corrected_h1)],
            "corrected_bound_holds": corrected_bound_holds,
            "verdict": "FALSIFIED",
            "scope": "The specific Proposition B.2 constants are false; closure itself survives with the corrected shift-invariant H0.",
        },
        "historical_slope_proxy_negative_control": slope_control,
        "theorem43_epsilon_domain_audit": {
            **theorem43_quadratic,
            "missing_epsilon_restriction_detected": theorem43_domain_omission,
            "verdict": "FALSIFIED",
            "scope": "The displayed iteration bound is not valid as written for every epsilon; convergence for the intended small-epsilon regime is not falsified.",
        },
        "claim_statuses": [
            {
                "claim": 1,
                "verdict": "FALSIFIED",
                "basis": "Published closure constants have an exact assumption-satisfying counterexample; core Definition 3.1 remains a definition.",
            },
            {
                "claim": 2,
                "verdict": "BLOCKED",
                "basis": "No assumption-satisfying counterexample found by this analytical route.",
            },
            {
                "claim": 3,
                "verdict": "BLOCKED",
                "basis": "Imported proposition labels disagree with arXiv v2; no counterexample yet.",
            },
            {
                "claim": 4,
                "verdict": "BLOCKED",
                "basis": "Lower-bound quantifier requires minimax interpretation; no valid counterexample established.",
            },
            {
                "claim": 5,
                "verdict": "FALSIFIED",
                "basis": "The theorem's displayed bound permits a negative iteration count without an epsilon-domain assumption.",
            },
            {
                "claim": 6,
                "verdict": "BLOCKED",
                "basis": "This route does not run the named full-scale empirical models.",
            },
        ],
        "negative_control": {
            "published_formula": "FAILED_AS_INTENDED" if not paper_bound_holds else "UNEXPECTED_PASS",
            "corrected_formula": "PASS" if corrected_bound_holds else "FAIL",
        },
        "verifier": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    print("=== ASSUMPTION-SATISFYING FALSIFICATION AUDIT ===")
    print(json.dumps(results, indent=2, sort_keys=True))
    print("=== END ASSUMPTION-SATISFYING FALSIFICATION AUDIT ===")
    return 0 if not errors else 1
