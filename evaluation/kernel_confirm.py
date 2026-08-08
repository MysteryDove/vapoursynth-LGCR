#!/usr/bin/env python3
"""Run the frozen supplemental Lanczos4-versus-Jinc3 kernel holdout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluation.common import degrade_scene, make_scene, measure_scene
from evaluation.kernel_study import KernelSpec, reconstruct
from evaluation.run import CORE, RESULTS, ROOT, bootstrap_ci, make_420_clip


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "kernel_study_config.json"
METHODS = (
    ("plain_jinc3", KernelSpec("jinc3", {"kernel": "jinc", "taps": 3}), "plain"),
    ("algo6_jinc3", KernelSpec("jinc3", {"kernel": "jinc", "taps": 3}), "algo6"),
    (
        "plain_lanczos4",
        KernelSpec("lanczos4", {"kernel": "lanczos", "taps": 4}),
        "plain",
    ),
    (
        "algo6_lanczos4",
        KernelSpec("lanczos4", {"kernel": "lanczos", "taps": 4}),
        "algo6",
    ),
)


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    selected = config["selected_non_jinc"]
    if selected != {
        "development_edge_mae": 0.023005181565140195,
        "kernel": "lanczos",
        "name": "lanczos4",
        "taps": 4,
    }:
        raise ValueError("kernel selection differs from the frozen Lanczos4 candidate")
    return config


def run_rows(config: dict) -> list[dict]:
    seeds = range(config["confirmation_seed_start"], config["confirmation_seed_end"] + 1)
    scenes = [make_scene(seed, "kernel_holdout") for seed in seeds]
    degradations = tuple(config["confirmation_degradations"])
    sitings = tuple(config["confirmation_sitings"])
    rows: list[dict] = []
    total = len(scenes) * len(degradations) * len(sitings)
    done = 0
    for scene in scenes:
        for degradation in degradations:
            for siting in sitings:
                y, u, v = degrade_scene(scene, degradation, siting)
                clip = make_420_clip(y, u, v, siting)
                for method, spec, mode in METHODS:
                    out_u, out_v = reconstruct(clip, spec, mode)
                    rows.append({
                        "split": "kernel_holdout",
                        "scene": scene.name,
                        "family": scene.family,
                        "condition": scene.condition,
                        "degradation": degradation,
                        "true_siting": siting,
                        "assumed_siting": siting,
                        "method": method,
                        **measure_scene(scene, out_u, out_v),
                    })
                done += 1
                if done % 32 == 0 or done == total:
                    print(f"[{done:>3}/{total}] {scene.name} D={degradation} siting={siting}")
    return rows


def scene_values(
    rows: list[dict],
    method: str,
    metric: str,
) -> tuple[list[str], np.ndarray]:
    names = sorted({row["scene"] for row in rows})
    kept: list[str] = []
    values: list[float] = []
    for name in names:
        samples = [
            float(row[metric]) for row in rows
            if row["scene"] == name and row["method"] == method
        ]
        samples = [value for value in samples if np.isfinite(value)]
        if samples:
            kept.append(name)
            values.append(float(np.mean(samples)))
    return kept, np.asarray(values, dtype=np.float64)


def paired_delta(
    rows: list[dict],
    method: str,
    reference: str,
    metric: str = "edge_mae",
) -> tuple[float, float, float, float]:
    names_a, values_a = scene_values(rows, method, metric)
    names_b, values_b = scene_values(rows, reference, metric)
    reference_by_name = dict(zip(names_b, values_b))
    delta = np.asarray(
        [value - reference_by_name[name] for name, value in zip(names_a, values_a)],
        dtype=np.float64,
    )
    low, high = bootstrap_ci(delta)
    return float(np.mean(delta)), low, high, float(np.mean(delta < 0.0))


def method_row(rows: list[dict], method: str) -> str:
    edge = scene_values(rows, method, "edge_mae")[1]
    low, high = bootstrap_ci(edge)
    artifacts = [
        float(np.mean(scene_values(rows, method, metric)[1]))
        for metric in (
            "bleed_mass",
            "phase_abs_px",
            "alias_px",
            "spread_delta_px",
            "ringing_mass",
        )
    ]
    return (
        f"| {method} | {np.mean(edge):.6f} [{low:.6f}, {high:.6f}] | "
        + " | ".join(f"{value:.6f}" for value in artifacts)
        + " |"
    )


def report_markdown(rows: list[dict]) -> str:
    lines = [
        "# Frozen Supplemental Kernel Holdout",
        "",
        f"Scenes: {len({row['scene'] for row in rows})}; raw rows: {len(rows)}.",
        "",
        "This holdout was specified after the development screen but before these",
        "seeds were evaluated. It is supplemental and does not replace the main",
        "paper's frozen Jinc3 analysis.",
        "",
        "Lower is better; transition-spread delta is best near zero.",
        "",
        "| method | edge MAE [95% CI] | bleed profile | |phase| px | alias px | spread delta px | ringing |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, _, _ in METHODS:
        lines.append(method_row(rows, method))

    lines.extend((
        "",
        "## Frozen Paired Comparisons",
        "",
        "Negative deltas favor Lanczos4 algo6. Confidence intervals resample scenes.",
        "",
        "| reference | edge MAE delta [95% CI] | scenes improved |",
        "|---|---:|---:|",
    ))
    for reference in ("algo6_jinc3", "plain_lanczos4"):
        mean, low, high, improved = paired_delta(rows, "algo6_lanczos4", reference)
        lines.append(
            f"| {reference} | {mean:+.6f} [{low:+.6f}, {high:+.6f}] | "
            f"{improved * 100:.1f}% |"
        )

    lines.extend((
        "",
        "## Lanczos4 Algo6 Minus Jinc3 Algo6 By Degradation",
        "",
        "| degradation | paired edge MAE delta [95% CI] | scenes improved |",
        "|---|---:|---:|",
    ))
    for degradation in ("box", "triangle", "bicubic", "lanczos"):
        selected = [row for row in rows if row["degradation"] == degradation]
        mean, low, high, improved = paired_delta(
            selected, "algo6_lanczos4", "algo6_jinc3"
        )
        lines.append(
            f"| {degradation} | {mean:+.6f} [{low:+.6f}, {high:+.6f}] | "
            f"{improved * 100:.1f}% |"
        )

    lines.extend((
        "",
        "## Edge MAE By Scene Condition",
        "",
        "| condition | scenes | Lanczos4 algo6 | Jinc3 algo6 | Lanczos4 plain |",
        "|---|---:|---:|---:|---:|",
    ))
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        values = [
            float(np.mean(scene_values(selected, method, "edge_mae")[1]))
            for method in ("algo6_lanczos4", "algo6_jinc3", "plain_lanczos4")
        ]
        count = len({row["scene"] for row in selected})
        lines.append(
            f"| {condition} | {count} | {values[0]:.6f} | {values[1]:.6f} | "
            f"{values[2]:.6f} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()
    config = load_config()
    CORE.std.LoadPlugin(str(ROOT / "liblgcr.so"))
    rows = run_rows(config)
    report = report_markdown(rows)
    print(report)
    if args.write_results:
        write_csv(rows, RESULTS / "kernel_holdout.csv")
        (RESULTS / "kernel_holdout.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
