#!/usr/bin/env python3
"""Run development tuning or the held-out LGCR paper benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import vapoursynth as vs

from evaluation.baselines import (
    galosh_form_upsample,
    guided_filter_upsample,
    joint_bilateral,
    korhonen_upsample,
    wada_ejbf,
)
from evaluation.common import SITING, degrade_scene, generate_scenes, measure_scene


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
RESULTS = HERE / "results"
CORE = vs.core
CORE.num_threads = 8
F420 = CORE.query_video_format(vs.YUV, vs.FLOAT, 32, 1, 1)
F444 = CORE.query_video_format(vs.YUV, vs.FLOAT, 32, 0, 0)


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def make_420_clip(y: np.ndarray, u: np.ndarray, v: np.ndarray, assumed_siting: str):
    h, w = y.shape
    blank = CORE.std.BlankClip(width=w, height=h, format=F420, length=1)
    code = SITING[assumed_siting][2]

    def fill(n, f):
        del n
        f = f.copy()
        np.copyto(np.asarray(f[0]), y)
        np.copyto(np.asarray(f[1]), u)
        np.copyto(np.asarray(f[2]), v)
        f.props["_ChromaLocation"] = code
        return f

    return blank.std.ModifyFrame(blank, fill)


def frame_uv(node) -> tuple[np.ndarray, np.ndarray]:
    frame = node.get_frame(0)
    return np.asarray(frame[1]).copy(), np.asarray(frame[2]).copy()


def zimg_reconstruct(clip, method: str, assumed_siting: str):
    code = SITING[assumed_siting][2]
    common = dict(format=F444, chromaloc_in=code, dither_type="none")
    if method == "zimg_bilinear":
        return frame_uv(CORE.resize.Bilinear(clip, **common))
    if method == "zimg_bicubic":
        return frame_uv(CORE.resize.Bicubic(clip, filter_param_a=0.0, filter_param_b=0.5, **common))
    if method == "zimg_lanczos3":
        return frame_uv(CORE.resize.Lanczos(clip, filter_param_a=3.0, **common))
    if method == "zimg_spline36":
        return frame_uv(CORE.resize.Spline36(clip, **common))
    raise ValueError(method)


def lgcr_reconstruct(clip, method: str):
    common = dict(kernel="jinc", taps=3, sparse=0, ar=0.0)
    if method == "lgcr_plain_jinc":
        return frame_uv(CORE.lgcr.Recon(clip, strength=0.0, algo=2, **common))
    if method == "lgcr_algo2":
        return frame_uv(CORE.lgcr.Recon(clip, strength=0.8, algo=2, **common))
    if method == "lgcr_algo4":
        return frame_uv(CORE.lgcr.Recon(clip, strength=0.8, algo=4, **common))
    if method == "lgcr_algo6":
        return frame_uv(CORE.lgcr.Recon(clip, strength=0.8, algo=6, **common))
    if method == "ablate_algo6_ungated":
        return frame_uv(CORE.lgcr.Recon(
            clip, strength=0.8, algo=6, qgate=0.0, ms=0.0, **common))
    if method == "ablate_algo6_plus_q":
        return frame_uv(CORE.lgcr.Recon(
            clip, strength=0.8, algo=6, qgate=1.0, ms=0.0, **common))
    if method == "ablate_similarity":
        return frame_uv(CORE.lgcr.Recon(
            clip, strength=0.8, algo=2, rescue=0.0, stretch=0.0,
            ridge=0, ms=0.0, **common))
    if method == "ablate_plus_rescue":
        return frame_uv(CORE.lgcr.Recon(
            clip, strength=0.8, algo=2, rescue=1.0, stretch=0.0,
            ridge=0, ms=0.0, **common))
    if method == "ablate_plus_anisotropy":
        return frame_uv(CORE.lgcr.Recon(
            clip, strength=0.8, algo=2, rescue=1.0, stretch=1.0,
            ridge=0, ms=0.0, **common))
    raise ValueError(method)


SIGNAL_METHODS = (
    "zimg_bilinear",
    "zimg_bicubic",
    "zimg_lanczos3",
    "zimg_spline36",
)
ANALYTIC_METHODS = ("jbu", "guided_filter", "wada_ejbf", "korhonen", "galosh_form")
LGCR_METHODS = ("lgcr_plain_jinc", "lgcr_algo2", "lgcr_algo4", "lgcr_algo6")
ABLATION_METHODS = (
    "lgcr_plain_jinc",
    "ablate_similarity",
    "ablate_plus_rescue",
    "ablate_plus_anisotropy",
    "lgcr_algo2",
    "ablate_algo6_ungated",
    "ablate_algo6_plus_q",
    "lgcr_algo6",
)
MAIN_METHODS = SIGNAL_METHODS + ANALYTIC_METHODS + LGCR_METHODS


def analytic_reconstruct(method: str, y, u, v, siting: str, config: dict):
    if method == "jbu":
        return joint_bilateral(y, u, v, siting, **config["jbu"])
    if method == "guided_filter":
        return guided_filter_upsample(y, u, v, siting, **config["guided_filter"])
    if method == "wada_ejbf":
        return wada_ejbf(y, u, v, siting, **config["wada_ejbf"])
    if method == "korhonen":
        return korhonen_upsample(y, u, v, siting, **config["korhonen"])
    if method == "galosh_form":
        return galosh_form_upsample(y, u, v, siting, **config["galosh_form"])
    raise ValueError(method)


def run_rows(
    split: str,
    config: dict,
    methods: tuple[str, ...],
    quick: bool = False,
    siting_mismatch: bool = False,
) -> list[dict]:
    scenes = generate_scenes(split)
    if quick:
        scenes = scenes[:4]
    degradations = ("box", "triangle", "bicubic") if split == "dev" else ("box", "triangle", "bicubic", "lanczos")
    true_sitings = ("left",) if split == "dev" else ("left", "center")
    if siting_mismatch:
        degradations = ("box",)
        true_sitings = ("left", "center", "topleft")
        scenes = scenes[:8]

    rows: list[dict] = []
    total = len(scenes) * len(degradations) * len(true_sitings)
    done = 0
    for scene in scenes:
        for degradation in degradations:
            for true_siting in true_sitings:
                assumed_sitings = (true_siting,)
                if siting_mismatch:
                    assumed_sitings = ("left", "center", "topleft")
                y, u, v = degrade_scene(scene, degradation, true_siting)
                for assumed_siting in assumed_sitings:
                    clip = make_420_clip(y, u, v, assumed_siting)
                    for method in methods:
                        if method in SIGNAL_METHODS:
                            out_u, out_v = zimg_reconstruct(clip, method, assumed_siting)
                        elif method in ANALYTIC_METHODS:
                            out_u, out_v = analytic_reconstruct(method, y, u, v, assumed_siting, config)
                        else:
                            out_u, out_v = lgcr_reconstruct(clip, method)
                        metrics = measure_scene(scene, out_u, out_v)
                        rows.append({
                            "split": split,
                            "scene": scene.name,
                            "family": scene.family,
                            "condition": scene.condition,
                            "degradation": degradation,
                            "true_siting": true_siting,
                            "assumed_siting": assumed_siting,
                            "method": method,
                            **metrics,
                        })
                done += 1
                if done % 8 == 0 or done == total:
                    print(f"[{done:>3}/{total}] {scene.name} D={degradation} siting={true_siting}")
    return rows


def tune_analytic(config: dict, quick: bool = False) -> dict:
    scenes = generate_scenes("dev")
    if quick:
        scenes = scenes[:4]
    observations = []
    for scene in scenes:
        for degradation in ("box", "triangle", "bicubic"):
            y, u, v = degrade_scene(scene, degradation, "left")
            observations.append((scene, y, u, v))

    grids: dict[str, list[dict]] = {
        "jbu": [
            {"radius": 2, "sigma_spatial": spatial, "sigma_range": range_sigma}
            for spatial in (1.5, 2.5, 4.0)
            for range_sigma in (0.015, 0.03, 0.05, 0.08)
        ],
        "guided_filter": [
            {"radius": radius, "eps": eps}
            for radius in (1, 2, 3)
            for eps in (2.5e-5, 1e-4, 4e-4)
        ],
        "galosh_form": [
            {"sigma_range": range_sigma, "anti_ringing": True}
            for range_sigma in (0.015, 0.03, 0.05, 0.08)
        ],
    }
    tuned = dict(config)
    for method, candidates in grids.items():
        scored = []
        for candidate in candidates:
            values = []
            for scene, y, u, v in observations:
                local = dict(config)
                local[method] = candidate
                out_u, out_v = analytic_reconstruct(method, y, u, v, "left", local)
                values.append(measure_scene(scene, out_u, out_v)["edge_mae"])
            scored.append((float(np.mean(values)), candidate))
        scored.sort(key=lambda item: item[0])
        tuned[method] = scored[0][1]
        print(f"{method}: edge_mae={scored[0][0]:.6f} params={scored[0][1]}")
    return tuned


def _scene_values(rows: list[dict], method: str, metric: str) -> tuple[list[str], np.ndarray]:
    scene_names = sorted({row["scene"] for row in rows})
    values = []
    kept = []
    for scene in scene_names:
        vals = [float(row[metric]) for row in rows if row["scene"] == scene and row["method"] == method]
        vals = [value for value in vals if np.isfinite(value)]
        if vals:
            kept.append(scene)
            values.append(float(np.mean(vals)))
    return kept, np.asarray(values, dtype=np.float64)


def bootstrap_ci(values: np.ndarray, seed: int = 20260808, reps: int = 4000) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def paired_delta(rows: list[dict], method: str, reference: str, metric: str):
    names_a, values_a = _scene_values(rows, method, metric)
    names_b, values_b = _scene_values(rows, reference, metric)
    map_b = dict(zip(names_b, values_b))
    pairs = [(value, map_b[name]) for name, value in zip(names_a, values_a) if name in map_b]
    delta = np.asarray([a - b for a, b in pairs], dtype=np.float64)
    lo, hi = bootstrap_ci(delta)
    return float(np.mean(delta)), lo, hi, float(np.mean(delta < 0))


def select_development_methods(rows: list[dict], config: dict) -> dict:
    tuned = dict(config)
    external = SIGNAL_METHODS + ANALYTIC_METHODS
    external_scores = {method: float(np.mean(_scene_values(rows, method, "edge_mae")[1])) for method in external}
    lgcr_scores = {method: float(np.mean(_scene_values(rows, method, "edge_mae")[1])) for method in LGCR_METHODS[1:]}
    tuned["primary_external"] = min(external_scores, key=external_scores.get)
    tuned["primary_lgcr"] = min(lgcr_scores, key=lgcr_scores.get)
    print(f"primary_external={tuned['primary_external']} edge_mae={external_scores[tuned['primary_external']]:.6f}")
    print(f"primary_lgcr={tuned['primary_lgcr']} edge_mae={lgcr_scores[tuned['primary_lgcr']]:.6f}")
    return tuned


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_markdown(rows: list[dict], config: dict, title: str) -> str:
    methods = tuple(dict.fromkeys(row["method"] for row in rows))
    lines = [f"# {title}", "", f"Scenes: {len(set(row['scene'] for row in rows))}; raw rows: {len(rows)}.", ""]
    lines.extend((
        "The table reports scene-weighted means. Profile metrics exclude scenes",
        "without a chroma transition. Lower is better except that transition-spread",
        "delta is signed; zero matches the ground-truth width. Edge-MAE confidence",
        "intervals resample scenes and retain all conditions for each sampled scene.", "",
        "## Reconstruction Error", "",
        "| method | edge MAE [scene-bootstrap 95% CI] | full MAE | PSNR dB | smooth MAE |",
        "|---|---:|---:|---:|---:|",
    ))
    for method in methods:
        edge_values = _scene_values(rows, method, "edge_mae")[1]
        edge_mean = float(np.mean(edge_values))
        edge_ci = bootstrap_ci(edge_values)
        means = [float(np.mean(_scene_values(rows, method, metric)[1]))
                 for metric in ("mae", "psnr", "smooth_mae")]
        lines.append(
            f"| {method} | {edge_mean:.6f} [{edge_ci[0]:.6f}, {edge_ci[1]:.6f}] "
            f"| {means[0]:.6f} | {means[1]:.3f} | {means[2]:.6f} |"
        )

    lines.extend((
        "", "## Boundary Artifact Measures", "",
        "| method | bleed profile error | |phase| px | alias px | spread delta px | ringing |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for method in methods:
        means = [float(np.mean(_scene_values(rows, method, metric)[1]))
                 for metric in ("bleed_mass", "phase_abs_px", "alias_px", "spread_delta_px", "ringing_mass")]
        lines.append(f"| {method} | " + " | ".join(f"{value:.6f}" for value in means) + " |")

    primary = config["primary_lgcr"]
    lines.extend(("", f"## Paired Scene-Level Deltas for `{primary}`", ""))
    lines.append("Negative MAE deltas favor LGCR. Confidence intervals resample scenes.")
    lines.extend(("", "| reference | edge MAE delta [95% CI] | smooth MAE delta [95% CI] | scenes improved |", "|---|---:|---:|---:|"))
    references = [reference for reference in (config["primary_external"], "lgcr_plain_jinc") if reference in methods]
    for reference in references:
        edge = paired_delta(rows, primary, reference, "edge_mae")
        smooth = paired_delta(rows, primary, reference, "smooth_mae")
        lines.append(
            f"| {reference} | {edge[0]:+.6f} [{edge[1]:+.6f}, {edge[2]:+.6f}] "
            f"| {smooth[0]:+.6f} [{smooth[1]:+.6f}, {smooth[2]:+.6f}] | {edge[3] * 100:.1f}% |"
        )

    if config["primary_external"] in methods:
        lines.extend(("", "## Primary Result by Actual Degradation", "", "| degradation | primary edge MAE | external edge MAE | plain edge MAE |", "|---|---:|---:|---:|"))
        for degradation in sorted({row["degradation"] for row in rows}):
            subset = [row for row in rows if row["degradation"] == degradation]
            values = []
            for method in (primary, config["primary_external"], "lgcr_plain_jinc"):
                values.append(float(np.mean(_scene_values(subset, method, "edge_mae")[1])))
            lines.append(f"| {degradation} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |")

        lines.extend(("", "## Primary Result by Scene Condition", "", "| condition | scenes | primary edge MAE | external edge MAE | plain edge MAE |", "|---|---:|---:|---:|---:|"))
        for condition in sorted({row["condition"] for row in rows}):
            subset = [row for row in rows if row["condition"] == condition]
            values = []
            for method in (primary, config["primary_external"], "lgcr_plain_jinc"):
                values.append(float(np.mean(_scene_values(subset, method, "edge_mae")[1])))
            scene_count = len({row["scene"] for row in subset})
            lines.append(
                f"| {condition} | {scene_count} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |"
            )

    if any(row["true_siting"] != row["assumed_siting"] for row in rows):
        lines.extend(("", "## Siting Assumption Intervention", ""))
        lines.extend((
            "| method | matched edge MAE | mismatched edge MAE | matched |phase| px | mismatched |phase| px |",
            "|---|---:|---:|---:|---:|",
        ))
        for method in methods:
            matched = [row for row in rows if row["method"] == method and row["true_siting"] == row["assumed_siting"]]
            mismatched = [row for row in rows if row["method"] == method and row["true_siting"] != row["assumed_siting"]]
            me = float(np.nanmean([row["edge_mae"] for row in matched]))
            xe = float(np.nanmean([row["edge_mae"] for row in mismatched]))
            mp = float(np.nanmean([row["phase_abs_px"] for row in matched]))
            xp = float(np.nanmean([row["phase_abs_px"] for row in mismatched]))
            lines.append(f"| {method} | {me:.6f} | {xe:.6f} | {mp:.6f} | {xp:.6f} |")

    plain = {(row["scene"], row["degradation"], row["true_siting"], row["assumed_siting"]): row
             for row in rows if row["method"] == "lgcr_plain_jinc"}
    regrets = []
    for row in rows:
        if row["method"] != primary:
            continue
        key = (row["scene"], row["degradation"], row["true_siting"], row["assumed_siting"])
        regrets.append((float(row["edge_mae"]) - float(plain[key]["edge_mae"]), row))
    lines.extend(("", "## Largest Edge-MAE Regrets vs Plain Jinc", ""))
    for delta, row in sorted(regrets, key=lambda item: item[0], reverse=True)[:5]:
        lines.append(
            f"- `{row['scene']}` ({row['condition']}, D={row['degradation']}, "
            f"siting={row['true_siting']}): {delta:+.6f}"
        )
    lines.extend(("", "These controlled synthetic results do not establish prevalence or subjective benefit on released animation."))
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="tune analytic baselines and select methods on development scenes")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--siting-mismatch", action="store_true")
    parser.add_argument("--write-results", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    CORE.std.LoadPlugin(str(ROOT / "liblgcr.so"))
    config = load_config()
    if args.tune:
        config = tune_analytic(config, args.quick)
        development = run_rows("dev", config, MAIN_METHODS, args.quick)
        config = select_development_methods(development, config)
        if args.write_results:
            with CONFIG_PATH.open("w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2, sort_keys=True)
                handle.write("\n")
            write_csv(development, RESULTS / "development.csv")
            (RESULTS / "development.md").write_text(
                report_markdown(development, config, "LGCR Development Results"), encoding="utf-8")
        return

    methods = ABLATION_METHODS if args.ablation else MAIN_METHODS
    rows = run_rows(args.split, config, methods, args.quick, args.siting_mismatch)
    title = "LGCR Siting-Mismatch Results" if args.siting_mismatch else (
        "LGCR Mechanism Ablation" if args.ablation else "LGCR Held-Out Controlled Synthetic Benchmark")
    report = report_markdown(rows, config, title)
    print(report)
    if args.write_results:
        suffix = "siting" if args.siting_mismatch else ("ablation" if args.ablation else args.split)
        write_csv(rows, RESULTS / f"{suffix}.csv")
        (RESULTS / f"{suffix}.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
