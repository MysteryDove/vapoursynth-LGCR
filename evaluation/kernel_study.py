#!/usr/bin/env python3
"""Development-only screen of LGCR base kernels.

This diagnostic must not be used to replace the frozen primary method on the
existing held-out test split. A selected non-Jinc candidate needs a new holdout.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation.common import degrade_scene, generate_scenes, measure_scene
from evaluation.run import CORE, RESULTS, ROOT, bootstrap_ci, frame_uv, make_420_clip


@dataclass(frozen=True)
class KernelSpec:
    name: str
    plugin_args: dict[str, float | int | str]


KERNELS = (
    KernelSpec("bilinear", {"kernel": "bilinear"}),
    KernelSpec("bicubic_catrom", {"kernel": "bicubic", "b": 0.0, "c": 0.5}),
    KernelSpec("bicubic_sharp", {"kernel": "bicubic", "b": 0.0, "c": 0.6}),
    KernelSpec(
        "bicubic_mitchell",
        {"kernel": "bicubic", "b": 1.0 / 3.0, "c": 1.0 / 3.0},
    ),
    KernelSpec("spline16", {"kernel": "spline16"}),
    KernelSpec("spline36", {"kernel": "spline36"}),
    KernelSpec("lanczos2", {"kernel": "lanczos", "taps": 2}),
    KernelSpec("lanczos3", {"kernel": "lanczos", "taps": 3}),
    KernelSpec("lanczos4", {"kernel": "lanczos", "taps": 4}),
    KernelSpec("jinc2", {"kernel": "jinc", "taps": 2}),
    KernelSpec("jinc3", {"kernel": "jinc", "taps": 3}),
    KernelSpec("jinc4", {"kernel": "jinc", "taps": 4}),
)
DEGRADATIONS = ("box", "triangle", "bicubic")
MODES = ("plain", "algo6")


def reconstruct(clip, spec: KernelSpec, mode: str) -> tuple[np.ndarray, np.ndarray]:
    common = dict(spec.plugin_args)
    common.update({"sparse": 0, "ar": 0.0})
    if mode == "plain":
        node = CORE.lgcr.Recon(clip, algo=2, strength=0.0, **common)
    elif mode == "algo6":
        node = CORE.lgcr.Recon(clip, algo=6, strength=0.8, **common)
    else:
        raise ValueError(mode)
    return frame_uv(node)


def run_rows() -> list[dict]:
    scenes = generate_scenes("dev")
    rows: list[dict] = []
    total = len(scenes) * len(DEGRADATIONS)
    done = 0
    for scene in scenes:
        for degradation in DEGRADATIONS:
            y, u, v = degrade_scene(scene, degradation, "left")
            clip = make_420_clip(y, u, v, "left")
            for spec in KERNELS:
                for mode in MODES:
                    out_u, out_v = reconstruct(clip, spec, mode)
                    rows.append({
                        "split": "development",
                        "scene": scene.name,
                        "family": scene.family,
                        "condition": scene.condition,
                        "degradation": degradation,
                        "true_siting": "left",
                        "base_kernel": spec.name,
                        "mode": mode,
                        **measure_scene(scene, out_u, out_v),
                    })
            done += 1
            print(f"[{done:>2}/{total}] {scene.name} D={degradation}")
    return rows


def scene_values(
    rows: list[dict],
    kernel: str,
    mode: str,
    metric: str,
    degradation: str | None = None,
) -> tuple[list[str], np.ndarray]:
    selected = [
        row for row in rows
        if row["base_kernel"] == kernel
        and row["mode"] == mode
        and (degradation is None or row["degradation"] == degradation)
    ]
    names = sorted({row["scene"] for row in selected})
    kept: list[str] = []
    values: list[float] = []
    for name in names:
        samples = [float(row[metric]) for row in selected if row["scene"] == name]
        samples = [value for value in samples if np.isfinite(value)]
        if samples:
            kept.append(name)
            values.append(float(np.mean(samples)))
    return kept, np.asarray(values, dtype=np.float64)


def paired_kernel_delta(rows: list[dict], kernel: str) -> tuple[float, float, float, float]:
    names_plain, plain = scene_values(rows, kernel, "plain", "edge_mae")
    names_algo6, algo6 = scene_values(rows, kernel, "algo6", "edge_mae")
    plain_by_name = dict(zip(names_plain, plain))
    delta = np.asarray(
        [value - plain_by_name[name] for name, value in zip(names_algo6, algo6)],
        dtype=np.float64,
    )
    low, high = bootstrap_ci(delta)
    return float(np.mean(delta)), low, high, float(np.mean(delta < 0.0))


def report_markdown(rows: list[dict]) -> str:
    lines = [
        "# Development-Only LGCR Base-Kernel Screen",
        "",
        f"Scenes: {len({row['scene'] for row in rows})}; raw rows: {len(rows)}.",
        "",
        "**Status: EXPLORATORY SCREEN.** This uses the existing development split only.",
        "It cannot replace Jinc3 in the frozen primary analysis, and the existing held-out",
        "split must not be used to confirm a newly selected kernel.",
        "",
        "All LGCR runs use strength 0.8, algo6, dense evaluation, and zero hull margin.",
        "Deltas are algo6 minus the signal-only reconstruction with the same base kernel.",
        "Confidence intervals resample scenes; negative values favor algo6.",
        "",
        "## Overall Edge Error",
        "",
        "| base kernel | plain edge MAE | algo6 edge MAE | algo6-minus-plain [95% CI] | scenes improved |",
        "|---|---:|---:|---:|---:|",
    ]
    algo6_scores: dict[str, float] = {}
    for spec in KERNELS:
        plain = scene_values(rows, spec.name, "plain", "edge_mae")[1]
        algo6 = scene_values(rows, spec.name, "algo6", "edge_mae")[1]
        algo6_scores[spec.name] = float(np.mean(algo6))
        delta, low, high, improved = paired_kernel_delta(rows, spec.name)
        lines.append(
            f"| {spec.name} | {np.mean(plain):.6f} | {np.mean(algo6):.6f} | "
            f"{delta:+.6f} [{low:+.6f}, {high:+.6f}] | {improved * 100:.1f}% |"
        )

    non_jinc = [spec.name for spec in KERNELS if not spec.name.startswith("jinc")]
    best_non_jinc = min(non_jinc, key=algo6_scores.get)
    best_overall = min(algo6_scores, key=algo6_scores.get)
    lines.extend((
        "",
        "## Screening Decisions",
        "",
        f"- Lowest development algo6 edge MAE overall: `{best_overall}` ({algo6_scores[best_overall]:.6f}).",
        f"- Lowest development non-Jinc algo6 edge MAE: `{best_non_jinc}` ({algo6_scores[best_non_jinc]:.6f}).",
        "- Any confirmatory comparison must freeze the non-Jinc candidate above and use new scenes.",
        "",
        "## Algo6 Edge MAE By Degradation",
        "",
        "| base kernel | box | triangle | bicubic |",
        "|---|---:|---:|---:|",
    ))
    for spec in KERNELS:
        values = [
            float(np.mean(scene_values(
                rows, spec.name, "algo6", "edge_mae", degradation
            )[1]))
            for degradation in DEGRADATIONS
        ]
        lines.append(
            f"| {spec.name} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |"
        )

    lines.extend((
        "",
        "## Algo6 Boundary Artifacts",
        "",
        "| base kernel | bleed profile | |phase| px | alias px | spread delta px | ringing |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for spec in KERNELS:
        values = [
            float(np.mean(scene_values(rows, spec.name, "algo6", metric)[1]))
            for metric in (
                "bleed_mass",
                "phase_abs_px",
                "alias_px",
                "spread_delta_px",
                "ringing_mass",
            )
        ]
        lines.append(
            f"| {spec.name} | " + " | ".join(f"{value:.6f}" for value in values) + " |"
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
    CORE.std.LoadPlugin(str(ROOT / "liblgcr.so"))
    rows = run_rows()
    report = report_markdown(rows)
    print(report)
    if args.write_results:
        write_csv(rows, RESULTS / "kernel_study.csv")
        (RESULTS / "kernel_study.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
