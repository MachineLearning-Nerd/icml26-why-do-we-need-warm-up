from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict[str, object]:
    return json.loads((ROOT / "evidence" / name).read_text())


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close()


plt.figure(figsize=(8.5, 3.2))
labels = [f"Claim {index}" for index in range(1, 7)]
statuses = ["VERIFIED", "FALSIFIED", "FALSIFIED", "FALSIFIED", "FALSIFIED", "BLOCKED"]
colors = ["#2b8a3e"] + ["#c92a2a"] * 4 + ["#868e96"]
plt.barh(labels[::-1], [1] * 6, color=colors[::-1])
for row, status in enumerate(statuses[::-1]):
    plt.text(0.03, row, status, va="center", ha="left", color="white", weight="bold")
plt.xlim(0, 1)
plt.xticks([])
plt.title("Exact campaign verdicts: five claims resolved, empirical scale claim blocked")
for spine in plt.gca().spines.values():
    spine.set_visible(False)
save("headline_verdicts.png")


claim2 = load("claim2_counterexample.json")
rows = claim2["counterexample"]["rows"]
t = np.array([row["t"] for row in rows], dtype=float)
curvature = np.array([row["hessian_norm"] for row in rows], dtype=float)
plt.figure(figsize=(6.8, 4.2))
plt.loglog(t, curvature, "o-", label="complete Hessian norm")
plt.loglog(t, 2 * t**2, "--", label=r"symbolic $2t^2$")
plt.xlabel("balanced-ray scale t")
plt.ylabel("curvature at zero loss gap")
plt.title("Claim 2: curvature is unbounded while loss gap stays zero")
plt.grid(alpha=0.25, which="both")
plt.legend()
save("claim2_unbounded_curvature.png")


claim3 = load("claim3_counterexample.json")
result3 = claim3["result"]
plt.figure(figsize=(6.8, 4.2))
values = [
    result3["paper_rhs_upper_bound_over_all_valid_fstar"],
    result3["complete_hessian_norm"],
]
bars = plt.bar(["maximum printed RHS", "complete Hessian norm"], values, color=["#748ffc", "#c92a2a"])
plt.ylabel("curvature bound")
plt.title("Claim 3: an assumption-satisfying point exceeds Eq. (31)")
for bar, value in zip(bars, values, strict=True):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}", ha="center")
save("claim3_bound_violation.png")


claim4 = load("claim4_counterexample.json")
claim5 = load("claim5_counterexample.json")
plt.figure(figsize=(8.2, 4.2))
labels45 = ["Thm. 4.1\nobserved", "Thm. 4.1\npaper lower bound", "Thm. 4.3\nminimum valid", "Thm. 4.3\npaper upper bound"]
values45 = [
    claim4["result"]["observed_first_hit"],
    claim4["result"]["paper_iteration_lower_bound"],
    claim5["counterexample"]["minimum_valid_iteration_count"],
    claim5["counterexample"]["paper_iteration_bound"],
]
bars = plt.bar(labels45, values45, color=["#2b8a3e", "#c92a2a", "#2b8a3e", "#c92a2a"])
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("iterations")
plt.title("Claims 4–5: displayed iteration statements contradict valid instances")
for bar, value in zip(bars, values45, strict=True):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + (0.25 if value >= 0 else -0.9),
        f"{value:.2f}",
        ha="center",
    )
save("theorem_iteration_contradictions.png")


claim6 = load("claim6_fineweb.json")
models = claim6["models"]
positions = np.arange(len(models))
slopes = np.array([model["slope"] for model in models])
lower = np.array([model["slope_ci95"][0] for model in models])
upper = np.array([model["slope_ci95"][1] for model in models])
plt.figure(figsize=(7.4, 4.4))
plt.errorbar(
    positions,
    slopes,
    yerr=np.vstack([slopes - lower, upper - slopes]),
    fmt="o",
    capsize=5,
    color="#364fc7",
)
plt.axhline(0, color="#c92a2a", linestyle="--", linewidth=1)
plt.xticks(positions, [model["model"] for model in models])
plt.ylabel("proxy-vs-loss slope (95% bootstrap CI)")
plt.title("Claim 6 short FineWeb routes: no predeclared positive slope passes")
plt.grid(alpha=0.25, axis="y")
save("claim6_fineweb_slopes.png")


calibration = load("claim6_proxy_calibration.json")
plt.figure(figsize=(7.4, 4.4))
names = ["exact Hessian", "printed proxy min", "printed proxy max"]
values = [
    calibration["exact_hessian_norm"],
    calibration["printed_proxy_range"][0],
    calibration["printed_proxy_range"][1],
]
plt.bar(names, values, color=["#2b8a3e", "#ffa94d", "#c92a2a"])
plt.yscale("log")
plt.ylabel("magnitude (log scale)")
plt.title("Claim 6 diagnostic: changing-minibatch noise dominates curvature")
save("claim6_proxy_noise.png")
