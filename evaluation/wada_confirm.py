#!/usr/bin/env python3
"""Run the frozen Wada EJBF versus LGCR supplemental holdout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluation.baselines import wada_ejbf
from evaluation.common import degrade_scene, make_scene, measure_scene, reconstruct_plane
from evaluation.kernel_confirm import paired_delta, scene_values
from evaluation.kernel_study import KernelSpec, reconstruct
from evaluation.run import CORE, RESULTS, ROOT, bootstrap_ci, frame_uv, make_420_clip


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "wada_study_config.json"
JINC3 = KernelSpec("jinc3", {"kernel": "jinc", "taps": 3})
LANCZOS4 = KernelSpec("lanczos4", {"kernel": "lanczos", "taps": 4})
METHOD_LABELS = {
    "plain_bilinear": "Plain bilinear",
    "wada_ejbf": "Wada EJBF",
    "plain_lanczos4": "Plain Lanczos4",
    "algo6_lanczos4": "Algo6 Lanczos4",
    "algo6_jinc3": "Algo6 Jinc3",
    "wada_plus_lgcr": "Wada + LGCR correction",
}


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    expected = {
        "bootstrap_repetitions": 4000,
        "bootstrap_seed": 20260808,
        "protocol_version": 1,
        "seed_start": 4000,
        "seed_end": 4063,
        "degradations": ["box", "triangle", "bicubic", "lanczos"],
        "sitings": ["left", "center"],
        "primary_endpoint": "edge_mae",
        "primary_comparison": {
            "method": "wada_ejbf",
            "reference": "algo6_lanczos4",
        },
        "wada_ejbf": {
            "radius": 12,
            "sigma_chroma": 0.1,
            "sigma_luma": 0.03,
            "sigma_spatial": 4.0,
        },
        "hybrid": {
            "correction_ar": -1.0,
            "correction_kernel": "lanczos",
            "correction_taps": 4,
            "hull_radius": 2,
        },
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Wada study config changed for {key}")
    if config.get("methods") != list(METHOD_LABELS):
        raise ValueError("Wada study method order changed")
    return config


def _local_hull(plane: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Return replicated-boundary local extrema on the chroma grid."""
    if radius < 0:
        raise ValueError("hull radius must be non-negative")
    h, w = plane.shape
    padded = np.pad(plane.astype(np.float64), radius, mode="edge")
    samples = [
        padded[dy:dy + h, dx:dx + w]
        for dy in range(2 * radius + 1)
        for dx in range(2 * radius + 1)
    ]
    return np.minimum.reduce(samples), np.maximum.reduce(samples)


def _raw_lgcr_correction(clip, config: dict) -> tuple[np.ndarray, np.ndarray]:
    common = {
        "kernel": config["correction_kernel"],
        "taps": int(config["correction_taps"]),
        "sparse": 0,
        "ar": float(config["correction_ar"]),
    }
    plain_u, plain_v = frame_uv(
        CORE.lgcr.Recon(clip, algo=2, strength=0.0, **common)
    )
    guided_u, guided_v = frame_uv(
        CORE.lgcr.Recon(clip, algo=6, strength=0.8, **common)
    )
    return guided_u - plain_u, guided_v - plain_v


def _hybrid(
    wada_u: np.ndarray,
    wada_v: np.ndarray,
    correction_u: np.ndarray,
    correction_v: np.ndarray,
    low_u: np.ndarray,
    low_v: np.ndarray,
    siting: str,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    outputs: list[np.ndarray] = []
    for base, correction, low in (
        (wada_u, correction_u, low_u),
        (wada_v, correction_v, low_v),
    ):
        low_min, low_max = _local_hull(low, radius)
        hull_min = reconstruct_plane(low_min, base.shape, "bilinear", siting)
        hull_max = reconstruct_plane(low_max, base.shape, "bilinear", siting)
        lo = np.minimum(base, hull_min)
        hi = np.maximum(base, hull_max)
        outputs.append(np.clip(base + correction, lo, hi).astype(np.float32))
    return outputs[0], outputs[1]


def run_rows(config: dict) -> list[dict]:
    seeds = range(config["seed_start"], config["seed_end"] + 1)
    scenes = [make_scene(seed, "wada_holdout") for seed in seeds]
    rows: list[dict] = []
    total = len(scenes) * len(config["degradations"]) * len(config["sitings"])
    done = 0
    for scene in scenes:
        for degradation in config["degradations"]:
            for siting in config["sitings"]:
                y, u, v = degrade_scene(scene, degradation, siting)
                clip = make_420_clip(y, u, v, siting)
                plain_bilinear = (
                    reconstruct_plane(u, y.shape, "bilinear", siting),
                    reconstruct_plane(v, y.shape, "bilinear", siting),
                )
                wada = wada_ejbf(y, u, v, siting, **config["wada_ejbf"])
                plain_lanczos4 = reconstruct(clip, LANCZOS4, "plain")
                algo6_lanczos4 = reconstruct(clip, LANCZOS4, "algo6")
                algo6_jinc3 = reconstruct(clip, JINC3, "algo6")
                correction = _raw_lgcr_correction(clip, config["hybrid"])
                hybrid = _hybrid(
                    *wada,
                    *correction,
                    u,
                    v,
                    siting,
                    int(config["hybrid"]["hull_radius"]),
                )
                outputs = {
                    "plain_bilinear": plain_bilinear,
                    "wada_ejbf": wada,
                    "plain_lanczos4": plain_lanczos4,
                    "algo6_lanczos4": algo6_lanczos4,
                    "algo6_jinc3": algo6_jinc3,
                    "wada_plus_lgcr": hybrid,
                }
                for method in config["methods"]:
                    out_u, out_v = outputs[method]
                    rows.append({
                        "split": "wada_holdout",
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
                if done % 16 == 0 or done == total:
                    print(
                        f"[{done:>3}/{total}] {scene.name} "
                        f"D={degradation} siting={siting}"
                    )
    return rows


def _method_row(rows: list[dict], method: str) -> str:
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
        f"| {METHOD_LABELS[method]} | {np.mean(edge):.6f} "
        f"[{low:.6f}, {high:.6f}] | "
        + " | ".join(f"{value:.6f}" for value in artifacts)
        + " |"
    )


def _comparison_row(rows: list[dict], method: str, reference: str) -> str:
    mean, low, high, improved = paired_delta(rows, method, reference)
    return (
        f"| {METHOD_LABELS[method]} minus {METHOD_LABELS[reference]} | "
        f"{mean:+.6f} [{low:+.6f}, {high:+.6f}] | "
        f"{improved * 100:.1f}% |"
    )


def report_markdown(rows: list[dict], config: dict) -> str:
    lines = [
        "# Frozen Supplemental Wada EJBF Holdout",
        "",
        f"Scenes: {len({row['scene'] for row in rows})}; raw rows: {len(rows)}.",
        "",
        "This third synthetic split was frozen after the main and base-kernel",
        "results were known but before these seeds were evaluated. The Wada",
        "comparison is primary within this supplemental study; the hybrid is",
        "exploratory. Neither changes the paper's original primary analysis.",
        "",
        "Lower is better; transition-spread delta is best near zero.",
        "",
        "| method | edge MAE [95% CI] | bleed profile | |phase| px | alias px | spread delta px | ringing |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in config["methods"]:
        lines.append(_method_row(rows, method))

    lines.extend((
        "",
        "## Paired Scene-Level Comparisons",
        "",
        "Negative deltas favor the first method. Confidence intervals resample scenes.",
        "",
        "| comparison | edge MAE delta [95% CI] | scenes improved |",
        "|---|---:|---:|",
    ))
    for method, reference in (
        ("wada_ejbf", "algo6_lanczos4"),
        ("wada_ejbf", "plain_bilinear"),
        ("wada_ejbf", "algo6_jinc3"),
        ("wada_plus_lgcr", "wada_ejbf"),
        ("wada_plus_lgcr", "algo6_lanczos4"),
    ):
        lines.append(_comparison_row(rows, method, reference))

    lines.extend((
        "",
        "## Wada EJBF Minus Algo6 Lanczos4 By Degradation",
        "",
        "| degradation | Wada EJBF | Algo6 Lanczos4 | paired delta [95% CI] | scenes improved |",
        "|---|---:|---:|---:|---:|",
    ))
    for degradation in config["degradations"]:
        selected = [row for row in rows if row["degradation"] == degradation]
        wada = float(np.mean(scene_values(selected, "wada_ejbf", "edge_mae")[1]))
        lgcr = float(np.mean(scene_values(selected, "algo6_lanczos4", "edge_mae")[1]))
        mean, low, high, improved = paired_delta(
            selected, "wada_ejbf", "algo6_lanczos4"
        )
        lines.append(
            f"| {degradation} | {wada:.6f} | {lgcr:.6f} | "
            f"{mean:+.6f} [{low:+.6f}, {high:+.6f}] | "
            f"{improved * 100:.1f}% |"
        )

    lines.extend((
        "",
        "## Edge MAE By Scene Condition",
        "",
        "| condition | scenes | Wada EJBF | Algo6 Lanczos4 | Wada + LGCR | Plain Lanczos4 |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        values = [
            float(np.mean(scene_values(selected, method, "edge_mae")[1]))
            for method in (
                "wada_ejbf",
                "algo6_lanczos4",
                "wada_plus_lgcr",
                "plain_lanczos4",
            )
        ]
        count = len({row["scene"] for row in selected})
        lines.append(
            f"| {condition} | {count} | {values[0]:.6f} | {values[1]:.6f} | "
            f"{values[2]:.6f} | {values[3]:.6f} |"
        )

    lines.extend((
        "",
        "The Wada implementation follows the published equation and normalized",
        "parameters but has not been validated bit-for-bit against author code.",
        "The hybrid is a diagnostic composition, not a plugin mode or a selected method.",
    ))
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
    report = report_markdown(rows, config)
    print(report)
    if args.write_results:
        write_csv(rows, RESULTS / "wada_holdout.csv")
        (RESULTS / "wada_holdout.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
