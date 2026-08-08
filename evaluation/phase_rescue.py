#!/usr/bin/env python3
"""Exploratory diagnostic for LGCR's separable-kernel phase rescue."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from evaluation.common import degrade_scene, generate_scenes
from evaluation.run import CORE, RESULTS, ROOT, bootstrap_ci, frame_uv, make_420_clip


DEGRADATIONS = ("box", "triangle", "bicubic", "lanczos")


def run_rows() -> list[dict]:
    scenes = [scene for scene in generate_scenes("test") if scene.condition == "coedge"]
    rows: list[dict] = []
    for scene in scenes:
        h, w = scene.shape
        yy, xx = np.mgrid[0:h, 0:w]
        edge_band = (
            (np.abs(xx - scene.edge_x[:, None]) <= 4.0)
            & (yy >= 6)
            & (yy < h - 6)
        )
        phase_masks = {
            "exact": edge_band & ((xx % 2) == 0),
            "half": edge_band & ((xx % 2) == 1),
        }
        for degradation in DEGRADATIONS:
            y, u, v = degrade_scene(scene, degradation, "left")
            clip = make_420_clip(y, u, v, "left")
            outputs = []
            for rescue in (0.0, 1.0):
                node = CORE.lgcr.Recon(
                    clip,
                    kernel="lanczos",
                    taps=3,
                    algo=2,
                    strength=0.8,
                    rescue=rescue,
                    stretch=0.0,
                    ridge=0,
                    ms=0.0,
                    sparse=0,
                    ar=0.0,
                )
                outputs.append(frame_uv(node))

            errors = [
                0.5 * (np.abs(out_u - scene.u) + np.abs(out_v - scene.v))
                for out_u, out_v in outputs
            ]
            output_delta = np.maximum(
                np.abs(outputs[1][0] - outputs[0][0]),
                np.abs(outputs[1][1] - outputs[0][1]),
            )
            for phase, mask in phase_masks.items():
                off = float(np.mean(errors[0][mask]))
                on = float(np.mean(errors[1][mask]))
                rows.append({
                    "scene": scene.name,
                    "family": scene.family,
                    "degradation": degradation,
                    "true_siting": "left",
                    "reconstruction_kernel": "lanczos3",
                    "x_phase": phase,
                    "rescue_off_edge_mae": off,
                    "rescue_on_edge_mae": on,
                    "delta": on - off,
                    "max_output_delta": float(np.max(output_delta[mask])),
                })
    return rows


def _scene_means(rows: list[dict], phase: str, key: str) -> np.ndarray:
    values = []
    for scene in sorted({row["scene"] for row in rows}):
        selected = [float(row[key]) for row in rows
                    if row["scene"] == scene and row["x_phase"] == phase]
        values.append(float(np.mean(selected)))
    return np.asarray(values, dtype=np.float64)


def report_markdown(rows: list[dict]) -> str:
    scene_count = len({row["scene"] for row in rows})
    lines = [
        "# Targeted Phase-Rescue Diagnostic (Exploratory)",
        "",
        f"Strict co-edge scenes: {scene_count}; raw rows: {len(rows)}.",
        "",
        "This post-hoc mechanism diagnostic is not a confirmatory benchmark. It",
        "uses left-sited 4:2:0, separable Lanczos3 reconstruction, and disables",
        "anisotropy, ridge, and mutual-structure gates. At the horizontal exact",
        "phase, a separable interpolating kernel is a delta; at half phase the",
        "rescue term is zero by construction. Deltas are rescue-on minus rescue-off,",
        "so negative values favor rescue. Confidence intervals resample scenes.",
        "",
        "| horizontal source phase | rescue off MAE | rescue on MAE | paired delta [95% CI] | scenes improved | max output change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for phase in ("exact", "half"):
        off = _scene_means(rows, phase, "rescue_off_edge_mae")
        on = _scene_means(rows, phase, "rescue_on_edge_mae")
        delta = _scene_means(rows, phase, "delta")
        lo, hi = bootstrap_ci(delta)
        max_change = max(float(row["max_output_delta"])
                         for row in rows if row["x_phase"] == phase)
        lines.append(
            f"| {phase} | {np.mean(off):.6f} | {np.mean(on):.6f} "
            f"| {np.mean(delta):+.6f} [{lo:+.6f}, {hi:+.6f}] "
            f"| {np.mean(delta < 0) * 100:.1f}% | {max_change:.8f} |"
        )

    lines.extend((
        "", "## Exact-Phase Delta by Degradation", "",
        "| degradation | rescue-on minus rescue-off MAE |",
        "|---|---:|",
    ))
    for degradation in DEGRADATIONS:
        selected = [float(row["delta"]) for row in rows
                    if row["x_phase"] == "exact" and row["degradation"] == degradation]
        lines.append(f"| {degradation} | {np.mean(selected):+.6f} |")
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
        write_csv(rows, RESULTS / "phase_rescue.csv")
        (RESULTS / "phase_rescue.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
