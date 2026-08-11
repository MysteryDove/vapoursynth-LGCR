#!/usr/bin/env python3
"""Independent paired study of the optional BM refinement stage."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np
import vapoursynth as vs

from evaluation.common import (
    SITING,
    Scene,
    degrade_scene,
    downsample_plane,
    make_scene,
    measure_scene,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
CORE = vs.core
CORE.num_threads = 8
F420 = CORE.query_video_format(vs.YUV, vs.FLOAT, 32, 1, 1)

RECON_SEEDS = range(3000, 3064)
TEMPORAL_SEEDS = range(4000, 4032)
DEGRADATIONS = ("box", "triangle", "bicubic", "lanczos")
SITINGS = ("left", "center")
RECON_OBSERVATIONS = ("clean", "q10", "q10_noise")
TEMPORAL_OBSERVATIONS = ("clean", "q10_noise")
MOTIONS = (0, 1, 2, 3)
METRICS = (
    "mae",
    "psnr",
    "edge_mae",
    "smooth_mae",
    "phase_px",
    "phase_abs_px",
    "alias_px",
    "spread_delta_px",
    "bleed_mass",
    "ringing_mass",
)
BASES = (
    ("algo2_lanczos3", {
        "kernel": "lanczos", "taps": 3, "algo": 2,
        "strength": 0.8, "sparse": 0, "ar": 0.0,
    }),
    ("algo6_jinc3", {
        "kernel": "jinc", "taps": 3, "algo": 6,
        "strength": 0.8, "sparse": 0, "ar": 0.0,
    }),
    ("algo6_lanczos4", {
        "kernel": "lanczos", "taps": 4, "algo": 6,
        "strength": 0.8, "sparse": 0, "ar": 0.0,
    }),
)


def make_clip(ys: list[np.ndarray], us: list[np.ndarray], vs_: list[np.ndarray],
              siting: str):
    height, width = ys[0].shape
    blank = CORE.std.BlankClip(
        width=width, height=height, format=F420, length=len(ys))
    location = SITING[siting][2]

    def fill(n, f):
        f = f.copy()
        np.copyto(np.asarray(f[0]), ys[n])
        np.copyto(np.asarray(f[1]), us[n])
        np.copyto(np.asarray(f[2]), vs_[n])
        f.props["_ChromaLocation"] = location
        return f

    return blank.std.ModifyFrame(blank, fill)


def frame_uv(node, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    frame = node.get_frame(index)
    return np.asarray(frame[1]).copy(), np.asarray(frame[2]).copy()


def noisy_chroma(u: np.ndarray, v: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    out_u = np.clip(u + rng.normal(0.0, 0.006, u.shape), -0.5, 0.5)
    out_v = np.clip(v + rng.normal(0.0, 0.006, v.shape), -0.5, 0.5)
    return out_u.astype(np.float32), out_v.astype(np.float32)


def observe_recon(scene: Scene, seed: int, degradation: str, siting: str,
                  observation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bits = 0 if observation == "clean" else 10
    y, u, v = degrade_scene(scene, degradation, siting, bits=bits)
    if observation == "q10_noise":
        degradation_index = DEGRADATIONS.index(degradation)
        siting_index = SITINGS.index(siting)
        noise_seed = 10_000_000 + seed * 100 + degradation_index * 10 + siting_index
        u, v = noisy_chroma(u, v, noise_seed)
    return y, u, v


def recon_rows(seeds: range) -> list[dict]:
    rows: list[dict] = []
    total = len(seeds) * len(DEGRADATIONS) * len(SITINGS) * len(RECON_OBSERVATIONS)
    done = 0
    for seed in seeds:
        scene = make_scene(seed, "bm_holdout")
        for degradation in DEGRADATIONS:
            for siting in SITINGS:
                for observation in RECON_OBSERVATIONS:
                    y, u, v = observe_recon(
                        scene, seed, degradation, siting, observation)
                    clip = make_clip([y], [u], [v], siting)
                    for base, options in BASES:
                        for bm in (False, True):
                            out_u, out_v = frame_uv(
                                CORE.lgcr.Recon(clip, bm=bm, **options))
                            rows.append({
                                "study": "Recon",
                                "scene": scene.name,
                                "seed": seed,
                                "family": scene.family,
                                "condition": scene.condition,
                                "degradation": degradation,
                                "true_siting": siting,
                                "observation": observation,
                                "motion": "",
                                "base": base,
                                "bm": int(bm),
                                **measure_scene(scene, out_u, out_v),
                            })
                    done += 1
                    if done % 32 == 0 or done == total:
                        print(f"Recon [{done:>4}/{total}] seed={seed} "
                              f"D={degradation} siting={siting} obs={observation}",
                              flush=True)
    return rows


def quantize_yuv(y: np.ndarray, u: np.ndarray, v: np.ndarray):
    levels = 1023.0
    yq = np.rint(np.clip(y, 0.0, 1.0) * levels) / levels
    uq = np.rint(np.clip(u + 0.5, 0.0, 1.0) * levels) / levels - 0.5
    vq = np.rint(np.clip(v + 0.5, 0.0, 1.0) * levels) / levels - 0.5
    return yq.astype(np.float32), uq.astype(np.float32), vq.astype(np.float32)


def shift_plane(plane: np.ndarray, dx: int) -> np.ndarray:
    indices = np.clip(np.arange(plane.shape[1]) - dx, 0, plane.shape[1] - 1)
    return plane[:, indices].copy()


def temporal_clip(scene: Scene, seed: int, motion: int, observation: str):
    ys: list[np.ndarray] = []
    us: list[np.ndarray] = []
    vs_: list[np.ndarray] = []
    for frame_index, dx in enumerate((-motion, 0, motion)):
        y = shift_plane(scene.y, dx)
        u = downsample_plane(shift_plane(scene.u, dx), "triangle", "left")
        v = downsample_plane(shift_plane(scene.v, dx), "triangle", "left")
        if observation == "q10_noise":
            y, u, v = quantize_yuv(y, u, v)
            noise_seed = 20_000_000 + seed * 100 + motion * 10 + frame_index
            u, v = noisy_chroma(u, v, noise_seed)
        ys.append(y.astype(np.float32))
        us.append(u.astype(np.float32))
        vs_.append(v.astype(np.float32))
    return make_clip(ys, us, vs_, "left")


def temporal_rows(seeds: range) -> list[dict]:
    rows: list[dict] = []
    total = len(seeds) * len(MOTIONS) * len(TEMPORAL_OBSERVATIONS)
    done = 0
    for seed in seeds:
        scene = make_scene(seed, "bm_temporal")
        for motion in MOTIONS:
            for observation in TEMPORAL_OBSERVATIONS:
                clip = temporal_clip(scene, seed, motion, observation)
                for bm in (False, True):
                    out_u, out_v = frame_uv(CORE.lgcr.TRecon(
                        clip, strength=0.8, trad=1, bm=bm), 1)
                    rows.append({
                        "study": "TRecon",
                        "scene": scene.name,
                        "seed": seed,
                        "family": scene.family,
                        "condition": scene.condition,
                        "degradation": "triangle",
                        "true_siting": "left",
                        "observation": observation,
                        "motion": motion,
                        "base": "trecon_lanczos3",
                        "bm": int(bm),
                        **measure_scene(scene, out_u, out_v),
                    })
                done += 1
                if done % 16 == 0 or done == total:
                    print(f"TRecon [{done:>3}/{total}] seed={seed} "
                          f"motion={motion} obs={observation}", flush=True)
    return rows


def matches(row: dict, filters: dict) -> bool:
    return all(row[key] == value for key, value in filters.items())


def paired_scene_values(rows: list[dict], metric: str, **filters):
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        if not matches(row, filters):
            continue
        value = float(row[metric])
        if not np.isfinite(value):
            continue
        grouped.setdefault(row["scene"], {0: [], 1: []})[int(row["bm"])].append(value)
    off: list[float] = []
    on: list[float] = []
    names: list[str] = []
    for scene in sorted(grouped):
        values = grouped[scene]
        if values[0] and values[1]:
            names.append(scene)
            off.append(float(np.mean(values[0])))
            on.append(float(np.mean(values[1])))
    return names, np.asarray(off), np.asarray(on)


def bootstrap_ci(values: np.ndarray, reps: int = 4000):
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(20260811)
    indices = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def summary(rows: list[dict], metric: str, **filters) -> dict[str, float]:
    _, off, on = paired_scene_values(rows, metric, **filters)
    delta = on - off
    lo, hi = bootstrap_ci(delta)
    return {
        "off": float(np.mean(off)),
        "on": float(np.mean(on)),
        "delta": float(np.mean(delta)),
        "lo": lo,
        "hi": hi,
        "improved": float(np.mean(delta < 0.0)),
        "n": len(delta),
    }


def validate_rows(rows: list[dict], expected: int) -> None:
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} raw rows, found {len(rows)}")
    pairs: dict[tuple, set[int]] = {}
    for row in rows:
        for metric in ("mae", "psnr", "edge_mae", "smooth_mae"):
            if not np.isfinite(float(row[metric])):
                raise RuntimeError(f"non-finite {metric}: {row}")
        key = tuple(row[field] for field in (
            "study", "scene", "degradation", "true_siting", "observation",
            "motion", "base",
        ))
        pairs.setdefault(key, set()).add(int(row["bm"]))
    incomplete = [key for key, values in pairs.items() if values != {0, 1}]
    if incomplete:
        raise RuntimeError(f"incomplete BM pairs: {incomplete[:3]}")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def effect_cell(result: dict) -> str:
    return (f"{result['delta']:+.6f} "
            f"[{result['lo']:+.6f}, {result['hi']:+.6f}]")


def report_markdown(rows: list[dict], elapsed: float) -> str:
    lines = [
        "# Optional BM Refinement Study",
        "",
        "This independent study follows `evaluation/bm_study_protocol.md`. All",
        "effects are paired `bm=True` minus `bm=False`; negative error deltas",
        "favor BM. The study does not amend the frozen paper benchmark.",
        "",
        f"Raw rows: {len(rows):,}; wall time: {elapsed:.1f} seconds.",
        "",
        "## Recon Primary Results",
        "",
        "| base | observation | edge MAE off | edge MAE on | paired delta [95% CI] | scenes improved | full MAE delta | smooth MAE delta | PSNR delta dB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for base, _ in BASES:
        for observation in RECON_OBSERVATIONS:
            filters = {"study": "Recon", "base": base, "observation": observation}
            edge = summary(rows, "edge_mae", **filters)
            full = summary(rows, "mae", **filters)
            smooth = summary(rows, "smooth_mae", **filters)
            psnr = summary(rows, "psnr", **filters)
            lines.append(
                f"| {base} | {observation} | {edge['off']:.6f} | {edge['on']:.6f} "
                f"| {effect_cell(edge)} | {100 * edge['improved']:.1f}% "
                f"| {full['delta']:+.6f} | {smooth['delta']:+.6f} "
                f"| {psnr['delta']:+.3f} |"
            )

    lines.extend((
        "",
        "## Recon Boundary Effects",
        "",
        "| base | observation | bleed delta | phase delta px | alias delta px | spread delta px | ringing delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ))
    for base, _ in BASES:
        for observation in RECON_OBSERVATIONS:
            filters = {"study": "Recon", "base": base, "observation": observation}
            values = [summary(rows, metric, **filters)["delta"] for metric in (
                "bleed_mass", "phase_abs_px", "alias_px", "spread_delta_px",
                "ringing_mass",
            )]
            lines.append(f"| {base} | {observation} | " +
                         " | ".join(f"{value:+.6f}" for value in values) + " |")

    lines.extend((
        "",
        "## Recon Edge-MAE Delta By Degradation",
        "",
        "| base | observation | degradation | paired delta [95% CI] | scenes improved |",
        "|---|---|---|---:|---:|",
    ))
    for base, _ in BASES:
        for observation in RECON_OBSERVATIONS:
            for degradation in DEGRADATIONS:
                result = summary(rows, "edge_mae", study="Recon", base=base,
                                 observation=observation, degradation=degradation)
                lines.append(f"| {base} | {observation} | {degradation} "
                             f"| {effect_cell(result)} | {100 * result['improved']:.1f}% |")

    lines.extend((
        "",
        "## Recon Edge-MAE Delta By Scene Condition",
        "",
        "| base | observation | condition | paired delta [95% CI] | scenes improved |",
        "|---|---|---|---:|---:|",
    ))
    conditions = sorted({row["condition"] for row in rows if row["study"] == "Recon"})
    for base, _ in BASES:
        for observation in RECON_OBSERVATIONS:
            for condition in conditions:
                result = summary(rows, "edge_mae", study="Recon", base=base,
                                 observation=observation, condition=condition)
                lines.append(f"| {base} | {observation} | {condition} "
                             f"| {effect_cell(result)} | {100 * result['improved']:.1f}% |")

    lines.extend((
        "",
        "## TRecon Primary Results",
        "",
        "| observation | motion px/frame | edge MAE off | edge MAE on | paired delta [95% CI] | scenes improved | full MAE delta | smooth MAE delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for observation in TEMPORAL_OBSERVATIONS:
        for motion in MOTIONS:
            filters = {"study": "TRecon", "observation": observation, "motion": motion}
            edge = summary(rows, "edge_mae", **filters)
            full = summary(rows, "mae", **filters)
            smooth = summary(rows, "smooth_mae", **filters)
            lines.append(
                f"| {observation} | {motion} | {edge['off']:.6f} | {edge['on']:.6f} "
                f"| {effect_cell(edge)} | {100 * edge['improved']:.1f}% "
                f"| {full['delta']:+.6f} | {smooth['delta']:+.6f} |"
            )

    lines.extend(("", "## Largest Edge-MAE Regressions", ""))
    pairs: dict[tuple, dict[int, dict]] = {}
    key_fields = (
        "study", "scene", "degradation", "true_siting", "observation",
        "motion", "base",
    )
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        pairs.setdefault(key, {})[int(row["bm"])] = row
    regrets = []
    for pair in pairs.values():
        if 0 in pair and 1 in pair:
            regrets.append((float(pair[1]["edge_mae"]) - float(pair[0]["edge_mae"]), pair[1]))
    for delta, row in sorted(regrets, reverse=True, key=lambda item: item[0])[:5]:
        lines.append(
            f"- `{row['scene']}` ({row['study']}, {row['base']}, "
            f"obs={row['observation']}, D={row['degradation']}, "
            f"siting={row['true_siting']}, motion={row['motion']}): {delta:+.6f}"
        )

    lines.extend((
        "",
        "The `q10_noise` condition uses synthetic independent chroma noise and",
        "is not a codec model. These results do not establish subjective benefit",
        "or expected prevalence on released video.",
        "",
    ))
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--write-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CORE.std.LoadPlugin(str(ROOT / "liblgcr.so"))
    recon_seeds = range(3000, 3004) if args.quick else RECON_SEEDS
    temporal_seeds = range(4000, 4004) if args.quick else TEMPORAL_SEEDS
    started = time.perf_counter()
    rows = recon_rows(recon_seeds) + temporal_rows(temporal_seeds)
    expected = (
        len(recon_seeds) * len(DEGRADATIONS) * len(SITINGS) *
        len(RECON_OBSERVATIONS) * len(BASES) * 2 +
        len(temporal_seeds) * len(MOTIONS) * len(TEMPORAL_OBSERVATIONS) * 2
    )
    validate_rows(rows, expected)
    elapsed = time.perf_counter() - started
    report = report_markdown(rows, elapsed)
    print(report)
    if args.write_results:
        if args.quick:
            raise RuntimeError("refusing to overwrite frozen results from --quick")
        write_csv(rows, RESULTS / "bm_study.csv")
        (RESULTS / "bm_study.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
