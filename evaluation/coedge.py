#!/usr/bin/env python3
"""Validate corpus manifests and measure luma/chroma edge co-location."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REQUIRED_FIELDS = (
    "id", "domain", "split", "path", "sha256", "license", "source_url",
    "work_id", "shot_id",
)
Y_THRESHOLD = 0.04
C_THRESHOLD = 0.04
COEDGE_RADIUS = 1
MIN_WORKS_PER_DOMAIN = 10
MIN_SHOTS_PER_DOMAIN = 30
DEFAULT_MANIFESTS = (
    HERE / "corpora" / "animation.csv",
    HERE / "corpora" / "natural.csv",
)


def selected_manifests(requested: list[Path] | None) -> list[Path]:
    return list(DEFAULT_MANIFESTS if requested is None else requested)


def rgb_to_yuv709(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    u = (b - y) / 1.8556
    v = (r - y) / 1.5748
    return y, u, v


def _gradient(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded = np.pad(plane.astype(np.float64), 1, mode="edge")
    gx = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    gy = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    return gx, gy, np.hypot(gx, gy)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(mask, radius, mode="constant")
    out = np.zeros_like(mask, dtype=bool)
    width = 2 * radius + 1
    for dy in range(width):
        for dx in range(width):
            out |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return out


def edge_statistics(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> dict[str, float]:
    if y.shape != u.shape or y.shape != v.shape:
        raise ValueError("Y, U, and V must have identical shapes")
    yx, yy, ym = _gradient(y)
    ux, uy, _ = _gradient(u)
    vx, vy, _ = _gradient(v)
    cm = np.sqrt(ux * ux + uy * uy + vx * vx + vy * vy)
    y_edge = ym >= Y_THRESHOLD
    c_edge = cm >= C_THRESHOLD
    y_near = _dilate(y_edge, COEDGE_RADIUS)
    c_near = _dilate(c_edge, COEDGE_RADIUS)
    colocated = y_edge & c_edge
    denominator = ym * cm
    direction = np.sqrt(
        (yx * ux + yy * uy) ** 2 + (yx * vx + yy * vy) ** 2
    ) / np.maximum(denominator, 1e-12)
    return {
        "luma_edge_density": float(np.mean(y_edge)),
        "chroma_edge_density": float(np.mean(c_edge)),
        "chroma_near_luma": float(np.mean(y_near[c_edge])) if np.any(c_edge) else float("nan"),
        "luma_near_chroma": float(np.mean(c_near[y_edge])) if np.any(y_edge) else float("nan"),
        "direction_agreement": float(np.mean(direction[colocated])) if np.any(colocated) else float("nan"),
    }


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(field not in reader.fieldnames for field in REQUIRED_FIELDS):
            raise ValueError(f"{path}: manifest header does not match the corpus contract")
        rows = list(reader)
    for line, row in enumerate(rows, start=2):
        missing = [field for field in REQUIRED_FIELDS if not row.get(field, "").strip()]
        if missing:
            raise ValueError(f"{path}:{line}: missing {', '.join(missing)}")
        if row["domain"] not in ("animation", "natural"):
            raise ValueError(f"{path}:{line}: invalid domain {row['domain']!r}")
        if row["split"] not in ("pilot", "test"):
            raise ValueError(f"{path}:{line}: invalid split {row['split']!r}")
        media = (path.parent / row["path"]).resolve()
        if not media.is_file():
            raise FileNotFoundError(f"{path}:{line}: media not found: {media}")
        digest = hashlib.sha256(media.read_bytes()).hexdigest()
        if digest.lower() != row["sha256"].lower():
            raise ValueError(f"{path}:{line}: SHA-256 mismatch for {media}")
        row["resolved_path"] = str(media)
    return rows


def evaluate(manifests: list[Path]) -> list[dict]:
    output = []
    ids: set[str] = set()
    shots: set[tuple[str, str]] = set()
    for manifest in manifests:
        for row in read_manifest(manifest):
            if row["id"] in ids:
                raise ValueError(f"duplicate corpus id: {row['id']}")
            shot = (row["work_id"], row["shot_id"])
            if shot in shots:
                raise ValueError(f"duplicate work/shot pair: {shot}")
            ids.add(row["id"])
            shots.add(shot)
            rgb = np.asarray(Image.open(row["resolved_path"]).convert("RGB"))
            y, u, v = rgb_to_yuv709(rgb)
            output.append({
                "id": row["id"],
                "domain": row["domain"],
                "split": row["split"],
                "work_id": row["work_id"],
                "shot_id": row["shot_id"],
                "width": rgb.shape[1],
                "height": rgb.shape[0],
                **edge_statistics(y, u, v),
            })
    return output


def _work_values(rows: list[dict], domain: str, metric: str) -> np.ndarray:
    values = []
    for work in sorted({row["work_id"] for row in rows if row["domain"] == domain}):
        samples = [float(row[metric]) for row in rows
                   if row["domain"] == domain and row["work_id"] == work]
        samples = [value for value in samples if np.isfinite(value)]
        if samples:
            values.append(float(np.mean(samples)))
    return np.asarray(values, dtype=np.float64)


def _bootstrap_ci(values: np.ndarray, seed: int = 20260808) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(4000, len(values)))
    return tuple(float(value) for value in np.quantile(values[indices].mean(axis=1), (0.025, 0.975)))


def _bootstrap_difference(
    animation: np.ndarray,
    natural: np.ndarray,
    seed: int = 20260808,
) -> tuple[float, float]:
    if not len(animation) or not len(natural):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    animation_indices = rng.integers(0, len(animation), size=(4000, len(animation)))
    natural_indices = rng.integers(0, len(natural), size=(4000, len(natural)))
    difference = (
        animation[animation_indices].mean(axis=1)
        - natural[natural_indices].mean(axis=1)
    )
    return tuple(float(value) for value in np.quantile(difference, (0.025, 0.975)))


def _domain_counts(rows: list[dict], domain: str) -> tuple[int, int]:
    subset = [row for row in rows if row["domain"] == domain]
    return len({row["work_id"] for row in subset}), len(subset)


def corpus_complete(rows: list[dict]) -> bool:
    test_rows = [row for row in rows if row["split"] == "test"]
    return all(
        _domain_counts(test_rows, domain)[0] >= MIN_WORKS_PER_DOMAIN
        and _domain_counts(test_rows, domain)[1] >= MIN_SHOTS_PER_DOMAIN
        for domain in ("animation", "natural")
    )


def report_markdown(rows: list[dict]) -> str:
    lines = ["# Animation/Natural Domain Co-Edge Validation", ""]
    if not rows:
        lines.extend((
            "**Status: INCOMPLETE.** The selected manifests contain no admitted media.",
            "No claim about animation-domain co-edge prevalence is supported yet.",
            "See `evaluation/corpora/README.md` for admission and sampling rules.",
        ))
        return "\n".join(lines) + "\n"

    pilot_rows = [row for row in rows if row["split"] == "pilot"]
    test_rows = [row for row in rows if row["split"] == "test"]
    complete = corpus_complete(rows)
    if complete:
        lines.append(
            "**Status: COMPLETE FOR THE FROZEN CORPUS CONTRACT.** Both test domains meet "
            "the minimum work and shot counts."
        )
    else:
        lines.append(
            "**Status: INCOMPLETE.** One or both test domains are below the required "
            f"{MIN_WORKS_PER_DOMAIN} works and {MIN_SHOTS_PER_DOMAIN} shots."
        )
        lines.append("Any values below are descriptive and do not support a domain claim.")

    lines.extend((
        "",
        f"Admitted rows: {len(pilot_rows)} pilot and {len(test_rows)} test. Pilot rows are excluded from all tables and intervals below.",
        "",
        f"Fixed thresholds: Y={Y_THRESHOLD:.2f}, C={C_THRESHOLD:.2f}; co-edge radius={COEDGE_RADIUS} px.",
        "Means and confidence intervals use works as the independent unit.",
        "",
        "| domain | works | test images | luma-edge density [95% CI] | chroma-edge density [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ))
    for domain in ("animation", "natural"):
        work_count, image_count = _domain_counts(test_rows, domain)
        cells = []
        for metric in ("luma_edge_density", "chroma_edge_density"):
            values = _work_values(test_rows, domain, metric)
            lo, hi = _bootstrap_ci(values)
            cells.append(f"{np.mean(values):.4f} [{lo:.4f}, {hi:.4f}]" if len(values) else "n/a")
        lines.append(f"| {domain} | {work_count} | {image_count} | " + " | ".join(cells) + " |")

    lines.extend((
        "",
        "| domain | chroma near luma [95% CI] | luma near chroma [95% CI] | direction agreement [95% CI] |",
        "|---|---:|---:|---:|",
    ))
    for domain in ("animation", "natural"):
        cells = []
        for metric in ("chroma_near_luma", "luma_near_chroma", "direction_agreement"):
            values = _work_values(test_rows, domain, metric)
            lo, hi = _bootstrap_ci(values)
            cells.append(f"{np.mean(values):.4f} [{lo:.4f}, {hi:.4f}]" if len(values) else "n/a")
        lines.append(f"| {domain} | " + " | ".join(cells) + " |")

    lines.extend((
        "",
        "## Animation Minus Natural Contrasts",
        "",
        "Positive values indicate a larger animation-domain mean. Each bootstrap sample resamples works independently within each domain.",
        "",
        "| metric | difference [95% CI] |",
        "|---|---:|",
    ))
    for metric in ("chroma_near_luma", "luma_near_chroma", "direction_agreement"):
        animation = _work_values(test_rows, "animation", metric)
        natural = _work_values(test_rows, "natural", metric)
        lo, hi = _bootstrap_difference(animation, natural)
        if len(animation) and len(natural):
            difference = float(np.mean(animation) - np.mean(natural))
            value = f"{difference:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        else:
            value = "n/a"
        lines.append(f"| {metric} | {value} |")
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict], path: Path) -> None:
    fields = (
        "id", "domain", "split", "work_id", "shot_id", "width", "height",
        "luma_edge_density", "chroma_edge_density", "chroma_near_luma",
        "luma_near_chroma", "direction_agreement",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", action="append", type=Path,
        help="manifest to evaluate; repeat for multiple files (defaults to both tracked manifests)",
    )
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    manifests = selected_manifests(args.manifest)
    rows = evaluate(manifests)
    if args.require_data and not rows:
        raise SystemExit("corpus manifests are empty")
    if args.require_complete and not corpus_complete(rows):
        raise SystemExit(
            f"test corpus does not meet {MIN_WORKS_PER_DOMAIN} works and "
            f"{MIN_SHOTS_PER_DOMAIN} shots per domain"
        )
    report = report_markdown(rows)
    print(report)
    if args.write_results:
        write_csv(rows, RESULTS / "domain_coedge.csv")
        (RESULTS / "domain_coedge.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
