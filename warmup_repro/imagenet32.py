from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import pickle
import platform
import struct
import sys
import time
import urllib.request
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from warmup_repro.certificates import run_certificates
from warmup_repro.run import git_sha


IMAGENET32_URL = "https://image-net.org/data/downsample/Imagenet32_train.zip"
USER_AGENT = (
    "Mozilla/5.0 (compatible; OpenResearch-Reproduction/1.0; "
    "+https://openresearch.sh)"
)
LEARNING_RATE = 1e-4
GRAD_CLIP = 1.0
BATCH_SIZE = 8
STEPS = 30
SEEDS = (100, 101, 102)


def http_request(*, start: int | None = None, end: int | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(IMAGENET32_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = response.read()
        if start is not None and response.status != 206:
            raise RuntimeError("ImageNet server ignored the byte-range request")
    return payload


def content_length() -> int:
    request = urllib.request.Request(
        IMAGENET32_URL,
        headers={"User-Agent": USER_AGENT},
        method="HEAD",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def fetch_first_training_member() -> tuple[dict[str, object], dict[str, object]]:
    archive_size = content_length()
    tail_start = max(0, archive_size - 131_072)
    tail = http_request(start=tail_start, end=archive_size - 1)
    eocd_position = tail.rfind(b"PK\x05\x06")
    if eocd_position < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    eocd = struct.unpack_from("<4s4H2LH", tail, eocd_position)
    central_size = eocd[5]
    central_offset = eocd[6]
    central = http_request(
        start=central_offset,
        end=central_offset + central_size - 1,
    )
    position = 0
    member: dict[str, object] | None = None
    while position < len(central):
        fields = struct.unpack_from("<4s6H3L5H2L", central, position)
        if fields[0] != b"PK\x01\x02":
            raise RuntimeError("invalid ZIP central-directory signature")
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        name_length, extra_length, comment_length = fields[10:13]
        local_offset = fields[16]
        name_start = position + 46
        name = central[name_start : name_start + name_length].decode()
        if name.endswith("train_data_batch_1"):
            member = {
                "name": name,
                "compression": fields[4],
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "crc32": fields[7],
                "local_offset": local_offset,
            }
            break
        position = name_start + name_length + extra_length + comment_length
    if member is None:
        raise RuntimeError("train_data_batch_1 not found in official archive")
    local_offset = int(member["local_offset"])
    local_header = http_request(start=local_offset, end=local_offset + 29)
    local_fields = struct.unpack("<4s5H3L2H", local_header)
    if local_fields[0] != b"PK\x03\x04":
        raise RuntimeError("invalid ZIP local-header signature")
    name_length, extra_length = local_fields[9:11]
    data_start = local_offset + 30 + name_length + extra_length
    compressed = http_request(
        start=data_start,
        end=data_start + int(member["compressed_size"]) - 1,
    )
    if int(member["compression"]) == 8:
        uncompressed = zlib.decompress(compressed, -15)
    elif int(member["compression"]) == 0:
        uncompressed = compressed
    else:
        raise RuntimeError("unsupported ZIP compression method")
    if len(uncompressed) != int(member["uncompressed_size"]):
        raise RuntimeError("ImageNet32 member size mismatch")
    if zlib.crc32(uncompressed) != int(member["crc32"]):
        raise RuntimeError("ImageNet32 member CRC mismatch")
    entry = pickle.load(io.BytesIO(uncompressed), encoding="latin1")
    audit = {
        "url": IMAGENET32_URL,
        "archive_content_length": archive_size,
        "member": member["name"],
        "compressed_bytes_fetched": len(compressed),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed_sha256": hashlib.sha256(uncompressed).hexdigest(),
        "crc32_verified": True,
    }
    return entry, audit


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        if stride != 1 or input_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_planes,
                    self.expansion * planes,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(value)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        return F.relu(output + self.shortcut(value))


class ImageNet32ResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._layer(64, 3, 1)
        self.layer2 = self._layer(128, 4, 2)
        self.layer3 = self._layer(256, 6, 2)
        self.layer4 = self._layer(512, 3, 2)
        self.linear = nn.Linear(2048, 1000)

    def _layer(self, planes: int, blocks: int, first_stride: int) -> nn.Sequential:
        strides = [first_stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.input_planes, planes, stride))
            self.input_planes = Bottleneck.expansion * planes
        return nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(value)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = F.avg_pool2d(output, 4)
        return self.linear(output.flatten(1))


class DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return value
        keep = 1.0 - self.probability
        mask = torch.rand(
            (value.shape[0],) + (1,) * (value.ndim - 1),
            device=value.device,
        ) < keep
        return value * mask / keep


class VisionBlock(nn.Module):
    def __init__(self, drop_path: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(192)
        self.attention = nn.MultiheadAttention(
            192, 3, dropout=0.0, bias=True, batch_first=True
        )
        self.norm2 = nn.LayerNorm(192)
        self.mlp = nn.Sequential(
            nn.Linear(192, 576),
            nn.GELU(),
            nn.Linear(576, 192),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(value)
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        value = value + self.drop_path(attended)
        return value + self.drop_path(self.mlp(self.norm2(value)))


class VisionTransformerTiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, 192, 4, stride=4)
        self.class_token = nn.Parameter(torch.zeros(1, 1, 192))
        self.position = nn.Parameter(torch.zeros(1, 65, 192))
        self.blocks = nn.ModuleList(
            [VisionBlock(0.1 * index / 11) for index in range(12)]
        )
        self.norm = nn.LayerNorm(192)
        self.head = nn.Linear(192, 1000)
        self.apply(self._initialize)
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        patches = self.patch(value).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(value.shape[0], -1, -1)
        output = torch.cat([class_token, patches], dim=1) + self.position
        for block in self.blocks:
            output = block(output)
        return self.head(self.norm(output[:, 0]))


def prepare_batches(
    entry: dict[str, object], seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    data = np.asarray(entry["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels_key = "labels" if "labels" in entry else "fine_labels"
    labels = np.asarray(entry[labels_key], dtype=np.int64) - 1
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(labels), BATCH_SIZE * (STEPS + 1), replace=False)
    images = torch.from_numpy(data[indices].copy()).float() / 255.0
    targets = torch.from_numpy(labels[indices].copy()).long()
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    augmented = torch.empty_like(images)
    for index in range(len(images)):
        top = int(rng.integers(0, 9))
        left = int(rng.integers(0, 9))
        image = padded[index, :, top : top + 32, left : left + 32]
        if rng.random() < 0.5:
            image = image.flip(-1)
        augmented[index] = image
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    augmented = (augmented - mean) / std
    return [
        (
            augmented[index : index + BATCH_SIZE],
            targets[index : index + BATCH_SIZE],
        )
        for index in range(0, len(targets), BATCH_SIZE)
    ]


def gradient_snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [
        parameter.grad.detach().clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]


def gradient_difference(
    previous: list[torch.Tensor], model: nn.Module
) -> tuple[float, float]:
    total32 = torch.zeros((), dtype=torch.float32)
    total64 = torch.zeros((), dtype=torch.float64)
    index = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        difference = parameter.grad.detach() - previous[index]
        total32 += difference.square().sum()
        total64 += difference.double().square().sum()
        index += 1
    if index != len(previous):
        raise RuntimeError("gradient snapshot shape mismatch")
    return math.sqrt(float(total32)), math.sqrt(float(total64))


def linear_fit(losses: np.ndarray, proxies: np.ndarray, seed: int) -> dict[str, float | list[float]]:
    slope, intercept = np.polyfit(losses, proxies, 1)
    predicted = intercept + slope * losses
    total = float(np.sum((proxies - proxies.mean()) ** 2))
    r_squared = 1 - float(np.sum((proxies - predicted) ** 2)) / max(total, 1e-30)
    rng = np.random.default_rng(seed)
    slopes = []
    shuffled_r2 = []
    for _ in range(500):
        sample = rng.integers(0, len(losses), len(losses))
        if np.ptp(losses[sample]) > 1e-12:
            slopes.append(float(np.polyfit(losses[sample], proxies[sample], 1)[0]))
        shuffled = rng.permutation(losses)
        shuffled_model = np.polyfit(shuffled, proxies, 1)
        shuffled_prediction = np.polyval(shuffled_model, shuffled)
        shuffled_r2.append(
            1
            - float(np.sum((proxies - shuffled_prediction) ** 2))
            / max(total, 1e-30)
        )
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "slope_ci95": [float(value) for value in np.quantile(slopes, [0.025, 0.975])],
        "permutation_p_value": (1 + sum(value >= r_squared for value in shuffled_r2))
        / (1 + len(shuffled_r2)),
    }


def train_trajectory(
    model_name: str,
    factory: type[nn.Module],
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    model = factory()
    model.train()
    rows = []
    previous_gradient = None
    previous_update_norm = None
    for step, (images, targets) in enumerate(batches):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images), targets)
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
                    "reduction_relative_error": abs(proxy64 - proxy32)
                    / max(proxy64, 1e-30),
                }
            )
        previous_gradient = gradient_snapshot(model)
        raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        previous_update_norm = LEARNING_RATE * min(float(raw_norm), GRAD_CLIP)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-LEARNING_RATE)
    losses = np.array([row["loss"] for row in rows])
    proxies = np.array([row["local_smoothness_proxy"] for row in rows])
    result = {
        "model": model_name,
        "seed": seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "rows": rows,
        "linear_relationship": linear_fit(losses, proxies, seed + 20_000),
        "max_independent_reduction_relative_error": max(
            row["reduction_relative_error"] for row in rows
        ),
    }
    del model, previous_gradient
    gc.collect()
    return result


def pooled_fit(trajectories: list[dict[str, object]], seed: int) -> dict[str, object]:
    losses = np.concatenate(
        [np.array([row["loss"] for row in run["rows"]]) for run in trajectories]
    )
    proxies = np.concatenate(
        [
            np.array([row["local_smoothness_proxy"] for row in run["rows"]])
            for run in trajectories
        ]
    )
    return linear_fit(losses, proxies, seed)


def run_imagenet32(config: dict[str, object], started: float) -> int:
    expected_cores = int(config["expected_cores"])
    torch.set_num_threads(expected_cores)
    errors = []
    if run_certificates(config, started) != 0:
        errors.append("cumulative certificate regression failed")
    entry, dataset_audit = fetch_first_training_member()
    model_factories = {
        "ResNet50": ImageNet32ResNet50,
        "ViT-Tiny": VisionTransformerTiny,
    }
    trajectories = []
    for model_name, factory in model_factories.items():
        for seed in SEEDS:
            batches = prepare_batches(entry, seed)
            trajectories.append(
                train_trajectory(model_name, factory, batches, seed)
            )
    pooled = {
        model_name: pooled_fit(
            [run for run in trajectories if run["model"] == model_name],
            int(config["seed"]) + index,
        )
        for index, model_name in enumerate(model_factories)
    }
    if any(
        run["max_independent_reduction_relative_error"] > 1e-4
        for run in trajectories
    ):
        errors.append("independent local-smoothness reduction disagreed")
    if any(len(run["rows"]) != STEPS for run in trajectories):
        errors.append("trajectory row count mismatch")
    verified_models = {}
    for model_name in model_factories:
        seed_slopes = [
            run["linear_relationship"]["slope"]
            for run in trajectories
            if run["model"] == model_name
        ]
        verified_models[model_name] = (
            pooled[model_name]["slope"] > 0
            and pooled[model_name]["slope_ci95"][0] > 0
            and pooled[model_name]["permutation_p_value"] <= 0.05
            and sum(slope > 0 for slope in seed_slopes) >= 2
        )
    claim_verified = all(verified_models.values())
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
            "models": ["ResNet50", "ViT-Tiny"],
            "dataset": "ImageNet32 training split",
            "optimizer": "SGD",
            "learning_rate": LEARNING_RATE,
            "gradient_clip": GRAD_CLIP,
            "proxy": "||grad_Sk(w_k+1)-grad_Sk-1(w_k)|| / ||w_k+1-w_k||",
        },
        "dataset_audit": dataset_audit,
        "trajectories": trajectories,
        "pooled_relationships": pooled,
        "model_acceptance": verified_models,
        "negative_control": {
            "method": "500 loss-label permutations per seed and pooled architecture",
            "pooled_p_values": {
                model: relationship["permutation_p_value"]
                for model, relationship in pooled.items()
            },
        },
        "deviations": [
            "Three 30-step early trajectories replace full ImageNet32 training.",
            "Batch 8 float32 CPU replaces the paper's undisclosed batch and GPU precision.",
            "Only the first official training member is sampled; no validation images are substituted.",
        ],
        "claim_6_vision_verdict": "VERIFIED" if claim_verified else "BLOCKED",
        "claim_6_vision_basis": (
            "Both named architectures pass pooled bootstrap/permutation criteria and at least two of three seed slopes are positive."
            if claim_verified
            else "Exact architecture/data runs completed, but at least one predeclared relationship criterion was not met."
        ),
        "verifier": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    print("=== EXACT IMAGENET32 VISION MODELS ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("=== END EXACT IMAGENET32 VISION MODELS ===")
    return 0 if not errors else 1
