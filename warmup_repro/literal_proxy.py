from __future__ import annotations

import gc
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from warmup_repro.certificates import run_certificates
from warmup_repro.fineweb_lm import (
    GRAD_CLIP,
    LEARNING_RATE,
    SCALES,
    VOCAB_SIZE,
    PlainLM,
    fetch_fineweb_rows,
    fit_relationship,
    make_batches,
)
from warmup_repro.run import git_sha


ROUTE_STEPS = 8


def snapshot(model: PlainLM) -> list[torch.Tensor]:
    return [
        parameter.grad.detach().clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]


def gradient(
    model: PlainLM, batch: tuple[torch.Tensor, torch.Tensor]
) -> tuple[float, list[torch.Tensor]]:
    inputs, targets = batch
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(
        model(inputs).reshape(-1, VOCAB_SIZE), targets.reshape(-1)
    )
    loss.backward()
    return float(loss.detach()), snapshot(model)


def difference_norm(
    left: list[torch.Tensor], right: list[torch.Tensor]
) -> tuple[float, float]:
    if len(left) != len(right):
        raise RuntimeError("gradient snapshot length mismatch")
    total32 = torch.zeros((), dtype=torch.float32)
    total64 = torch.zeros((), dtype=torch.float64)
    for left_part, right_part in zip(left, right, strict=True):
        difference = left_part - right_part
        total32 += difference.square().sum()
        total64 += difference.double().square().sum()
    return math.sqrt(float(total32)), math.sqrt(float(total64))


def decomposition_residual(
    after_current: list[torch.Tensor],
    before_current: list[torch.Tensor],
    before_previous: list[torch.Tensor],
) -> float:
    residual = 0.0
    scale = 0.0
    for after, current, previous in zip(
        after_current, before_current, before_previous, strict=True
    ):
        direct = after - previous
        decomposed = (after - current) + (current - previous)
        residual += float((direct.double() - decomposed.double()).square().sum())
        scale += float(direct.double().square().sum())
    return math.sqrt(residual / max(scale, 1e-300))


def apply_update(model: PlainLM) -> float:
    raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    clipped_norm = min(float(raw_norm), GRAD_CLIP)
    update_norm = LEARNING_RATE * clipped_norm
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-LEARNING_RATE)
    return update_norm


def proxy_value(numerator: float, denominator: float) -> float:
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("proxy denominator must be positive and finite")
    return numerator / denominator


def run_trajectory(seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    texts, dataset_audit = fetch_fineweb_rows()
    batches = make_batches(texts, seed)
    scale = SCALES[0]
    model = PlainLM(scale)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != scale.expected_parameters:
        raise RuntimeError("70M parameter-count contract failed")

    _, initial_gradient = gradient(model, batches[0])
    first_update_norm = apply_update(model)
    previous_batch = batches[0]
    previous_gradient_before_update = initial_gradient
    previous_update_norm = first_update_norm
    rows: list[dict[str, float | int]] = []

    for step in range(1, ROUTE_STEPS + 1):
        previous_loss_at_current, previous_at_current = gradient(
            model, previous_batch
        )
        current_loss_before, current_before = gradient(model, batches[step])

        approximate32, approximate64 = difference_norm(
            current_before, previous_gradient_before_update
        )
        approximate_proxy = proxy_value(approximate64, previous_update_norm)

        update_norm = apply_update(model)
        current_loss_after, current_after = gradient(model, batches[step])

        literal32, literal64 = difference_norm(
            current_after, previous_at_current
        )
        same_batch32, same_batch64 = difference_norm(
            current_after, current_before
        )
        switch32, switch64 = difference_norm(
            current_before, previous_at_current
        )
        literal_proxy = proxy_value(literal64, update_norm)
        same_batch_proxy = proxy_value(same_batch64, update_norm)
        batch_switch_proxy = proxy_value(switch64, update_norm)
        rows.append(
            {
                "step": step,
                "loss_current_before": current_loss_before,
                "loss_current_after": current_loss_after,
                "loss_previous_batch_at_current": previous_loss_at_current,
                "update_norm": update_norm,
                "literal_proxy": literal_proxy,
                "same_batch_proxy": same_batch_proxy,
                "batch_switch_proxy": batch_switch_proxy,
                "one_backward_approximation": approximate_proxy,
                "literal_float32_relative_error": abs(literal32 - literal64)
                / max(literal64, 1e-300),
                "same_batch_float32_relative_error": abs(
                    same_batch32 - same_batch64
                )
                / max(same_batch64, 1e-300),
                "switch_float32_relative_error": abs(switch32 - switch64)
                / max(switch64, 1e-300),
                "decomposition_relative_residual": decomposition_residual(
                    current_after, current_before, previous_at_current
                ),
            }
        )
        previous_batch = batches[step]
        previous_gradient_before_update = current_before
        previous_update_norm = update_norm

    losses = np.array([row["loss_current_before"] for row in rows])
    literal = np.array([row["literal_proxy"] for row in rows])
    same_batch = np.array([row["same_batch_proxy"] for row in rows])
    approximate = np.array(
        [row["one_backward_approximation"] for row in rows]
    )
    diagnostics = {
        "literal_relationship": fit_relationship(losses, literal, seed + 1),
        "same_batch_relationship": fit_relationship(losses, same_batch, seed + 2),
        "approximate_relationship": fit_relationship(losses, approximate, seed + 3),
        "median_literal_to_same_batch_ratio": float(
            np.median(literal / np.maximum(same_batch, 1e-300))
        ),
        "median_literal_to_approximate_ratio": float(
            np.median(literal / np.maximum(approximate, 1e-300))
        ),
    }
    del model, initial_gradient, previous_gradient_before_update
    gc.collect()
    return {
        "model": "70M PlainLM",
        "parameter_count": parameter_count,
        "rows": rows,
        "diagnostics": diagnostics,
        "dataset_audit": dataset_audit,
    }


def run_literal_proxy(config: dict[str, object], started: float) -> int:
    expected_cores = int(config["expected_cores"])
    torch.set_num_threads(expected_cores)
    errors = []
    if run_certificates(config, started) != 0:
        errors.append("cumulative certificate regression failed")
    trajectory = run_trajectory(int(config["seed"]))
    rows = trajectory["rows"]
    if len(rows) != ROUTE_STEPS:
        errors.append("literal trajectory row count mismatch")
    if max(row["literal_float32_relative_error"] for row in rows) > 1e-4:
        errors.append("independent literal-proxy reduction disagreed")
    if max(row["decomposition_relative_residual"] for row in rows) > 1e-12:
        errors.append("literal estimator decomposition failed")

    try:
        proxy_value(float(rows[0]["literal_proxy"]), 0.0)
        zero_denominator_rejected = False
    except ValueError:
        zero_denominator_rejected = True
    if not zero_denominator_rejected:
        errors.append("zero-denominator negative control was accepted")

    runtime = time.perf_counter() - started
    result = {
        "stage": config["stage"],
        "git_sha": git_sha(),
        "seed": config["seed"],
        "expected_cores": expected_cores,
        "actual_cpu_allocation": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "runtime_seconds": round(runtime, 6),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "exact_claim": (
            "For much of early training of 70M, 160M, and 410M PlainLM models "
            "on FineWeb, the displayed stochastic local-smoothness proxy is "
            "well approximated by a line in stochastic training loss."
        ),
        "displayed_proxy": (
            "||grad f_{S_k}(w_{k+1}) - grad f_{S_{k-1}}(w_k)|| "
            "/ ||w_{k+1}-w_k||"
        ),
        "trajectory": trajectory,
        "negative_control": {
            "zero_denominator_rejected_as_intended": zero_denominator_rejected,
            "same_minibatch_proxy_removes_the_batch-switch_term": True,
        },
        "deviations": [
            "Eight measured updates cannot establish 'much of early training'.",
            "Only the 70M architecture is tested; 160M and 410M are omitted.",
            "Micro-batch 1, sequence length 16, hashed tokens, and float32 CPU replace the paper protocol.",
            "The public PlainLM repository lacks the paper's measurement instrumentation, so the displayed equation is implemented independently.",
        ],
        "falsification_result": "NOT_ESTABLISHED",
        "claim_6_verdict": "BLOCKED",
        "claim_6_basis": (
            "The literal estimator is independently checked, but the material "
            "horizon, batch, sequence, tokenizer, precision, and model-family "
            "deviations prevent either verification or valid falsification."
        ),
        "verifier": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    print("=== LITERAL SECTION 3.2 PROXY FALSIFICATION ROUTE ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("=== END LITERAL SECTION 3.2 PROXY FALSIFICATION ROUTE ===")
    return 0 if not errors else 1
