from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from warmup_repro.certificates import run_certificates
from warmup_repro.run import git_sha


VOCAB_SIZE = 50_280
LEARNING_RATE = 1e-4
GRAD_CLIP = 1.0
SEQUENCE_LENGTH = 16
STEPS = 12
USER_AGENT = (
    "Mozilla/5.0 (compatible; OpenResearch-Reproduction/1.0; "
    "+https://openresearch.sh)"
)
FINEWEB_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?"
    + urllib.parse.urlencode(
        {
            "dataset": "HuggingFaceFW/fineweb",
            "config": "sample-10BT",
            "split": "train",
            "offset": 0,
            "length": 100,
        }
    )
)


@dataclass(frozen=True)
class ModelScale:
    label: str
    layers: int
    heads: int
    width: int
    expected_parameters: int


SCALES = (
    ModelScale("70M", 6, 8, 512, 71_941_888),
    ModelScale("160M", 12, 12, 768, 162_186_240),
    ModelScale("410M", 24, 16, 1024, 411_309_056),
)


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + 1e-6
        )
        return normalized.to(value.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = int((8 / 3) * width)
        hidden = 256 * ((hidden + 255) // 256)
        self.hidden = hidden
        self.fc1 = nn.Linear(width, 2 * hidden, bias=False)
        self.fc2 = nn.Linear(hidden, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        activation, gate = self.fc1(value).split(self.hidden, dim=-1)
        return self.fc2(F.silu(activation) * gate)


def rotary_frequencies(head_width: int, length: int) -> torch.Tensor:
    frequencies = 1.0 / (
        500_000
        ** (torch.arange(0, head_width, 2, dtype=torch.float32) / head_width)
    )
    angles = torch.outer(torch.arange(length, dtype=torch.float32), frequencies)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary(value: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    original_dtype = value.dtype
    pairs = torch.view_as_complex(
        value.float().reshape(*value.shape[:-1], -1, 2)
    )
    rotated = torch.view_as_real(
        pairs * frequencies[None, :, None, :]
    ).flatten(-2)
    return rotated.to(original_dtype)


class Attention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.q_norm = RMSNorm(self.head_width)
        self.k_norm = RMSNorm(self.head_width)

    def forward(
        self, value: torch.Tensor, frequencies: torch.Tensor
    ) -> torch.Tensor:
        batch, length, width = value.shape
        q, k, v = self.qkv(value).split(width, dim=-1)
        q = self.q_norm(q.view(batch, length, self.heads, self.head_width))
        k = self.k_norm(k.view(batch, length, self.heads, self.head_width))
        v = v.view(batch, length, self.heads, self.head_width)
        q = apply_rotary(q, frequencies).transpose(1, 2)
        k = apply_rotary(k, frequencies).transpose(1, 2)
        v = v.transpose(1, 2)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.output(output.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(width)
        self.attention = Attention(width, heads)
        self.mlp_norm = RMSNorm(width)
        self.mlp = SwiGLU(width)

    def forward(
        self, value: torch.Tensor, frequencies: torch.Tensor
    ) -> torch.Tensor:
        value = value + self.attention(self.attention_norm(value), frequencies)
        return value + self.mlp(self.mlp_norm(value))


class PlainLM(nn.Module):
    def __init__(self, scale: ModelScale) -> None:
        super().__init__()
        self.layers_count = scale.layers
        self.embedding = nn.Embedding(VOCAB_SIZE, scale.width)
        self.embedding_norm = RMSNorm(scale.width)
        self.blocks = nn.ModuleList(
            [Block(scale.width, scale.heads) for _ in range(scale.layers)]
        )
        self.output_norm = RMSNorm(scale.width)
        self.lm_head = nn.Linear(scale.width, VOCAB_SIZE, bias=False)
        self.register_buffer(
            "frequencies",
            rotary_frequencies(scale.width // scale.heads, SEQUENCE_LENGTH),
            persistent=False,
        )
        self.apply(self._initialize)
        for name, parameter in self.named_parameters():
            if name.endswith("mlp.fc2.weight") or name.endswith(
                "attention.output.weight"
            ):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * scale.layers),
                )

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value = self.embedding_norm(self.embedding(tokens))
        for block in self.blocks:
            value = block(value, self.frequencies)
        return self.lm_head(self.output_norm(value))


def fetch_fineweb_rows() -> tuple[list[str], dict[str, object]]:
    request = urllib.request.Request(FINEWEB_ROWS_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    decoded = json.loads(payload)
    rows = decoded["rows"]
    texts = [row["row"]["text"] for row in rows]
    row_ids = [str(row["row"]["id"]) for row in rows]
    audit = {
        "url": FINEWEB_ROWS_URL,
        "response_sha256": hashlib.sha256(payload).hexdigest(),
        "row_count": len(rows),
        "row_ids_sha256": hashlib.sha256(
            "\n".join(row_ids).encode()
        ).hexdigest(),
        "text_sha256": hashlib.sha256(
            "\n".join(texts).encode()
        ).hexdigest(),
        "partial_dataset_view": bool(decoded["partial"]),
    }
    return texts, audit


def tokenize(text: str) -> list[int]:
    pieces = re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
    return [
        1
        + int.from_bytes(
            hashlib.blake2s(piece.encode(), digest_size=4).digest(), "little"
        )
        % (VOCAB_SIZE - 1)
        for piece in pieces
    ]


def make_batches(texts: list[str], seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    tokens: list[int] = []
    for text in texts:
        tokens.extend(tokenize(text))
    required = (STEPS + 1) * (SEQUENCE_LENGTH + 1)
    if len(tokens) < required:
        raise RuntimeError("FineWeb response did not contain enough tokens")
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(tokens) - SEQUENCE_LENGTH - 1, STEPS + 1, replace=False)
    batches = []
    for start in starts:
        window = torch.tensor(
            tokens[start : start + SEQUENCE_LENGTH + 1], dtype=torch.long
        )
        batches.append((window[:-1][None, :], window[1:][None, :]))
    return batches


def gradient_snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [
        parameter.grad.detach().clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]


def gradient_difference(
    previous: list[torch.Tensor], model: nn.Module
) -> tuple[float, float]:
    numerator32 = torch.zeros((), dtype=torch.float32)
    numerator64 = torch.zeros((), dtype=torch.float64)
    index = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        difference = parameter.grad.detach() - previous[index]
        numerator32 += torch.sum(difference.square())
        numerator64 += torch.sum(difference.double().square())
        index += 1
    if index != len(previous):
        raise RuntimeError("gradient snapshot shape mismatch")
    return math.sqrt(float(numerator32)), math.sqrt(float(numerator64))


def fit_relationship(losses: np.ndarray, proxies: np.ndarray, seed: int) -> dict[str, object]:
    design = np.column_stack([np.ones_like(losses), losses])
    intercept, slope = np.linalg.lstsq(design, proxies, rcond=None)[0]
    predicted = intercept + slope * losses
    residual = float(np.sum((proxies - predicted) ** 2))
    total = float(np.sum((proxies - np.mean(proxies)) ** 2))
    r_squared = 1.0 - residual / max(total, 1e-30)
    rng = np.random.default_rng(seed)
    slopes = []
    shuffled_r2 = []
    for _ in range(500):
        sample = rng.integers(0, len(losses), len(losses))
        if np.ptp(losses[sample]) > 1e-12:
            slopes.append(
                float(np.polyfit(losses[sample], proxies[sample], 1)[0])
            )
        shuffled = rng.permutation(losses)
        shuffled_fit = np.polyfit(shuffled, proxies, 1)
        shuffled_predicted = np.polyval(shuffled_fit, shuffled)
        shuffled_r2.append(
            1.0
            - float(np.sum((proxies - shuffled_predicted) ** 2))
            / max(total, 1e-30)
        )
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r_squared": r_squared,
        "slope_ci95": [float(value) for value in np.quantile(slopes, [0.025, 0.975])],
        "permutation_p_value": (1 + sum(value >= r_squared for value in shuffled_r2))
        / (1 + len(shuffled_r2)),
    }


def run_scale(
    scale: ModelScale, batches: list[tuple[torch.Tensor, torch.Tensor]], seed: int
) -> dict[str, object]:
    torch.manual_seed(seed)
    model = PlainLM(scale)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != scale.expected_parameters:
        raise RuntimeError(
            f"{scale.label} parameter count {parameter_count} != {scale.expected_parameters}"
        )
    rows: list[dict[str, float | int]] = []
    previous_gradient: list[torch.Tensor] | None = None
    previous_update_norm: float | None = None
    for step, (inputs, targets) in enumerate(batches):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            model(inputs).reshape(-1, VOCAB_SIZE), targets.reshape(-1)
        )
        loss.backward()
        if previous_gradient is not None and previous_update_norm is not None:
            numerator32, numerator64 = gradient_difference(previous_gradient, model)
            proxy32 = numerator32 / previous_update_norm
            proxy64 = numerator64 / previous_update_norm
            rows.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "local_smoothness_proxy": proxy64,
                    "independent_proxy_float32": proxy32,
                    "reduction_relative_error": abs(proxy32 - proxy64)
                    / max(proxy64, 1e-30),
                }
            )
        previous_gradient = gradient_snapshot(model)
        raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        clipped_norm = min(float(raw_norm), GRAD_CLIP)
        previous_update_norm = LEARNING_RATE * clipped_norm
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-LEARNING_RATE)
    losses = np.array([row["loss"] for row in rows])
    proxies = np.array([row["local_smoothness_proxy"] for row in rows])
    relationship = fit_relationship(losses, proxies, seed + 10_000)
    result = {
        "model": scale.label,
        "layers": scale.layers,
        "heads": scale.heads,
        "width": scale.width,
        "parameter_count": parameter_count,
        "sequence_length": SEQUENCE_LENGTH,
        "micro_batch_size": 1,
        "trajectory_steps": STEPS,
        "rows": rows,
        "linear_relationship": relationship,
        "max_independent_reduction_relative_error": max(
            row["reduction_relative_error"] for row in rows
        ),
    }
    del model, previous_gradient
    gc.collect()
    return result


def run_fineweb(config: dict[str, object], started: float) -> int:
    expected_cores = int(config["expected_cores"])
    torch.set_num_threads(expected_cores)
    certificate_errors = []
    if run_certificates(config, started) != 0:
        certificate_errors.append("cumulative certificate regression failed")
    texts, dataset_audit = fetch_fineweb_rows()
    batches = make_batches(texts, int(config["seed"]))
    models = [
        run_scale(scale, batches, int(config["seed"]) + index)
        for index, scale in enumerate(SCALES)
    ]
    errors = list(certificate_errors)
    if any(
        model["max_independent_reduction_relative_error"] > 1e-4
        for model in models
    ):
        errors.append("independent local-smoothness reduction disagreed")
    if any(len(model["rows"]) != STEPS for model in models):
        errors.append("trajectory row count mismatch")
    claim_verified = all(
        model["linear_relationship"]["slope"] > 0
        and model["linear_relationship"]["slope_ci95"][0] > 0
        and model["linear_relationship"]["permutation_p_value"] <= 0.05
        for model in models
    )
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
        "paper_contract": {
            "models": ["70M", "160M", "410M"],
            "dataset": "FineWeb",
            "optimizer": "SGD",
            "learning_rate": LEARNING_RATE,
            "gradient_clip": GRAD_CLIP,
            "proxy": "||grad_Sk(w_k+1)-grad_Sk-1(w_k)|| / ||w_k+1-w_k||",
        },
        "dataset_audit": dataset_audit,
        "models": models,
        "negative_control": {
            "method": "500 loss-label permutations per architecture",
            "p_values": {
                model["model"]: model["linear_relationship"]["permutation_p_value"]
                for model in models
            },
        },
        "deviations": [
            "12-step early trajectories replace the paper's full token budgets.",
            "micro-batch 1 and sequence length 16 replace batch 256 and sequence lengths 1024/2048.",
            "deterministic hashed word pieces replace the paper implementation's unavailable tokenizer.",
            "float32 CPU replaces mixed-precision FP16 GPU training.",
        ],
        "claim_6_verdict": "VERIFIED" if claim_verified else "BLOCKED",
        "claim_6_basis": (
            "All three exact-scale LM trajectories pass a positive-slope bootstrap and permutation control."
            if claim_verified
            else "Exact-scale runs completed, but the short CPU trajectories do not establish the paper's relationship for all three models."
        ),
        "verifier": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    print("=== EXACT-SCALE FINEWEB LANGUAGE MODELS ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("=== END EXACT-SCALE FINEWEB LANGUAGE MODELS ===")
    return 0 if not errors else 1
