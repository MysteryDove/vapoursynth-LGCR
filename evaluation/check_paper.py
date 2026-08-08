#!/usr/bin/env python3
"""Check that the working paper agrees with the tracked raw evaluation rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"
PAPER = ROOT / "paper" / "paper.md"
BIBLIOGRAPHY = ROOT / "paper" / "references.bib"
SEED = 20260808
REPS = 4000


def read_rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def subset(rows: list[dict[str, str]], **conditions: str) -> list[dict[str, str]]:
    return [
        row for row in rows
        if all(row[key] == value for key, value in conditions.items())
    ]


def scene_values(
    rows: list[dict[str, str]],
    method: str,
    metric: str,
) -> tuple[list[str], np.ndarray]:
    names = sorted({row["scene"] for row in rows})
    kept: list[str] = []
    values: list[float] = []
    for name in names:
        selected = [
            float(row[metric]) for row in rows
            if row["scene"] == name and row["method"] == method
        ]
        selected = [value for value in selected if np.isfinite(value)]
        if selected:
            kept.append(name)
            values.append(float(np.mean(selected)))
    return kept, np.asarray(values, dtype=np.float64)


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(values), size=(REPS, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def paired_delta(
    rows: list[dict[str, str]],
    method: str,
    reference: str,
    metric: str = "edge_mae",
) -> tuple[float, float, float, float]:
    names_a, values_a = scene_values(rows, method, metric)
    names_b, values_b = scene_values(rows, reference, metric)
    by_name_b = dict(zip(names_b, values_b))
    delta = np.asarray(
        [value - by_name_b[name] for name, value in zip(names_a, values_a)],
        dtype=np.float64,
    )
    low, high = bootstrap_ci(delta)
    return float(np.mean(delta)), low, high, float(np.mean(delta < 0.0))


def direct_mean(rows: list[dict[str, str]], method: str, metric: str) -> float:
    values = [float(row[metric]) for row in rows if row["method"] == method]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values))


def paper_table_row(
    rows: list[dict[str, str]],
    method: str,
    label: str,
) -> str:
    edge = scene_values(rows, method, "edge_mae")[1]
    low, high = bootstrap_ci(edge)
    artifact = [
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
        f"| {label} | {np.mean(edge):.6f} [{low:.6f}, {high:.6f}] | "
        + " | ".join(f"{value:.6f}" for value in artifact)
        + " |"
    )


def main() -> None:
    paper = PAPER.read_text(encoding="utf-8")
    paper_without_emphasis = re.sub(r"(?m)^> ?", "", paper.replace("**", ""))
    compact_paper = " ".join(paper_without_emphasis.split())
    bibliography = BIBLIOGRAPHY.read_text(encoding="utf-8")
    failures: list[str] = []

    def require(fragment: str, label: str, compact: bool = False) -> None:
        haystack = compact_paper if compact else paper_without_emphasis
        if fragment not in haystack:
            failures.append(f"{label}: expected {fragment!r}")

    with (ROOT / "evaluation" / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("protocol_version") != 2:
        failures.append("config: protocol_version is not 2")
    if config.get("primary_external") != "wada_ejbf":
        failures.append("config: frozen external comparator is not wada_ejbf")
    if config.get("primary_lgcr") != "lgcr_algo6":
        failures.append("config: frozen LGCR method is not lgcr_algo6")

    development = read_rows("development.csv")
    test = read_rows("test.csv")
    ablation = read_rows("ablation.csv")
    siting = read_rows("siting.csv")
    phase = read_rows("phase_rescue.csv")

    dev_wada = float(np.mean(scene_values(development, "wada_ejbf", "edge_mae")[1]))
    dev_algo6 = float(np.mean(scene_values(development, "lgcr_algo6", "edge_mae")[1]))
    require(
        f"Wada EJBF (`{dev_wada:.6f}`) becomes the external comparator; "
        f"`lgcr_algo6` (`{dev_algo6:.6f}`) becomes the primary LGCR method.",
        "development selections",
        compact=True,
    )

    scene_count = len({row["scene"] for row in test})
    require(
        f"The held-out benchmark contains {scene_count} scenes and {len(test):,} raw method-condition rows.",
        "held-out size",
        compact=True,
    )

    table_methods = {
        "zimg_bilinear": "zimg Bilinear",
        "zimg_bicubic": "zimg Bicubic",
        "zimg_lanczos3": "zimg Lanczos3",
        "zimg_spline36": "zimg Spline36",
        "jbu": "JBU",
        "guided_filter": "Guided filter",
        "wada_ejbf": "Wada EJBF",
        "korhonen": "Korhonen",
        "galosh_form": "GALOSH-form",
        "lgcr_plain_jinc": "Plain Jinc3",
        "lgcr_algo2": "LGCR algo2",
        "lgcr_algo4": "LGCR algo4",
        "lgcr_algo6": "LGCR algo6 (primary)",
    }
    for method, label in table_methods.items():
        require(paper_table_row(test, method, label), f"held-out table row {method}")

    primary_pairs = (
        ("lgcr_algo6", "wada_ejbf", "algo6-minus-Wada"),
        ("lgcr_algo6", "lgcr_plain_jinc", "algo6-minus-plain"),
        ("wada_ejbf", "lgcr_plain_jinc", "Wada-minus-plain"),
        ("lgcr_algo4", "wada_ejbf", "algo4-minus-Wada"),
    )
    for method, reference, label in primary_pairs:
        mean, low, high, _ = paired_delta(test, method, reference)
        require(
            f"{mean:+.6f} [95% CI {low:+.6f}, {high:+.6f}]",
            label,
            compact=True,
        )

    degradation_labels = {
        "box": "Box",
        "triangle": "Triangle",
        "bicubic": "Bicubic",
        "lanczos": "Lanczos (unseen)",
    }
    for degradation, label in degradation_labels.items():
        selected = subset(test, degradation=degradation)
        algo6 = float(np.mean(scene_values(selected, "lgcr_algo6", "edge_mae")[1]))
        wada = float(np.mean(scene_values(selected, "wada_ejbf", "edge_mae")[1]))
        mean, low, high, improved = paired_delta(selected, "lgcr_algo6", "wada_ejbf")
        row = (
            f"| {label} | {algo6:.6f} | {wada:.6f} | "
            f"{mean:+.6f} [{low:+.6f}, {high:+.6f}] | {improved * 100:.1f}% |"
        )
        require(row, f"degradation row {degradation}")

    condition_labels = {
        "coedge": "Co-edge",
        "isoluminant": "Isoluminant",
        "luma_only": "Luma only",
        "misaligned": "Misaligned",
        "ridge": "Ridge",
        "soft_chroma": "Soft chroma",
    }
    for condition, label in condition_labels.items():
        selected = subset(test, condition=condition)
        count = len({row["scene"] for row in selected})
        values = [
            float(np.mean(scene_values(selected, method, "edge_mae")[1]))
            for method in ("lgcr_algo6", "wada_ejbf", "lgcr_plain_jinc")
        ]
        row = (
            f"| {label} | {count} | {values[0]:.6f} | {values[1]:.6f} | "
            f"{values[2]:.6f} |"
        )
        require(row, f"condition row {condition}")

    for method, label in (
        ("lgcr_plain_jinc", "Plain Jinc"),
        ("wada_ejbf", "Wada EJBF"),
        ("lgcr_algo6", "LGCR algo6"),
    ):
        matched = [
            row for row in siting
            if row["method"] == method and row["true_siting"] == row["assumed_siting"]
        ]
        mismatched = [
            row for row in siting
            if row["method"] == method and row["true_siting"] != row["assumed_siting"]
        ]
        row = (
            f"| {label} | {direct_mean(matched, method, 'edge_mae'):.6f} | "
            f"{direct_mean(mismatched, method, 'edge_mae'):.6f} | "
            f"{direct_mean(matched, method, 'phase_abs_px'):.6f} | "
            f"{direct_mean(mismatched, method, 'phase_abs_px'):.6f} |"
        )
        require(row, f"siting row {method}")

    ablation_labels = {
        "lgcr_plain_jinc": "Plain Jinc3",
        "ablate_similarity": "Similarity only",
        "ablate_plus_rescue": "+ phase rescue",
        "ablate_plus_anisotropy": "+ anisotropy",
        "lgcr_algo2": "Full algo2",
        "ablate_algo6_ungated": "Algo6 ungated",
        "ablate_algo6_plus_q": "Algo6 + affine credibility",
        "lgcr_algo6": "Full algo6 (+ mutual structure)",
    }
    for method, label in ablation_labels.items():
        value = float(np.mean(scene_values(ablation, method, "edge_mae")[1]))
        require(f"| {label} | {value:.6f} |", f"ablation row {method}")

    phase_exact = [row for row in phase if row["x_phase"] == "exact"]
    phase_half = [row for row in phase if row["x_phase"] == "half"]
    exact_off = np.asarray([float(row["rescue_off_edge_mae"]) for row in phase_exact])
    exact_on = np.asarray([float(row["rescue_on_edge_mae"]) for row in phase_exact])
    exact_delta_names = sorted({row["scene"] for row in phase_exact})
    exact_delta = np.asarray([
        np.mean([float(row["delta"]) for row in phase_exact if row["scene"] == name])
        for name in exact_delta_names
    ])
    low, high = bootstrap_ci(exact_delta)
    require(
        f"from {np.mean(exact_off):.6f} to {np.mean(exact_on):.6f}, a paired delta of "
        f"{np.mean(exact_delta):+.6f} [95% CI {low:+.6f}, {high:+.6f}]",
        "phase-rescue exact phase",
        compact=True,
    )
    max_half = max(float(row["max_output_delta"]) for row in phase_half)
    if max_half != 0.0:
        failures.append(f"phase-rescue half phase: expected zero output change, found {max_half}")
    require(
        "At half phase, the rescue term is zero by construction",
        "phase-rescue half phase",
        compact=True,
    )

    required_caveats = (
        "This draft does not support a claim about prevalence or subjective benefit on released animation.",
        "No real animation evidence.",
        "No codec in the primary forward model.",
        "Formula-level baselines.",
        "This work does not claim the first color-bleeding reducer",
    )
    for caveat in required_caveats:
        require(caveat, f"required caveat {caveat}", compact=True)

    doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
    body, reference_section = paper.split("## References", maxsplit=1)
    body_dois = {doi.rstrip(".,);").lower() for doi in doi_pattern.findall(body)}
    reference_dois = {
        doi.rstrip(".,);").lower()
        for doi in doi_pattern.findall(reference_section)
    }
    missing_references = sorted(body_dois - reference_dois)
    if missing_references:
        failures.append(
            "reference list: missing body DOI(s): " + ", ".join(missing_references)
        )
    paper_dois = {doi.rstrip(".,);").lower() for doi in doi_pattern.findall(paper)}
    bibliography_dois = {
        doi.rstrip(".,);").lower() for doi in doi_pattern.findall(bibliography)
    }
    missing_dois = sorted(paper_dois - bibliography_dois)
    if missing_dois:
        failures.append("bibliography: missing paper DOI(s): " + ", ".join(missing_dois))

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(f"paper consistency check failed with {len(failures)} issue(s)")
    print("paper/result consistency checks: OK")


if __name__ == "__main__":
    main()
