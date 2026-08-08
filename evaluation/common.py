"""Synthetic animation scenes, the 4:2:0 forward model, and metrics.

The evaluation keeps the true degradation and sampling grid separate from the
grid assumed by a reconstruction method. Pixel centers use integer luma
coordinates. For 2x subsampling, a chroma sample is located at
``2 * index + 0.5 + shift``. Thus H.273 left siting has ``shift_x=-0.5`` and
center siting has ``shift_x=0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


SITING = {
    "left": (-0.5, 0.0, 0),
    "center": (0.0, 0.0, 1),
    "topleft": (-0.5, -0.5, 2),
    "top": (0.0, -0.5, 3),
    "bottomleft": (-0.5, 0.5, 4),
    "bottom": (0.0, 0.5, 5),
}


@dataclass(frozen=True)
class Scene:
    name: str
    split: str
    family: str
    condition: str
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    alpha: np.ndarray
    edge_x: np.ndarray
    side_u: tuple[float, float]
    side_v: tuple[float, float]

    @property
    def shape(self) -> tuple[int, int]:
        return self.y.shape


def rgb_to_yuv709(rgb: Iterable[float]) -> tuple[float, float, float]:
    r, g, b = (float(value) for value in rgb)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    u = (b - y) / 1.8556
    v = (r - y) / 1.5748
    return y, u, v


PALETTE = (
    ((0.92, 0.08, 0.12), (0.07, 0.16, 0.92)),
    ((0.96, 0.28, 0.58), (0.05, 0.70, 0.78)),
    ((0.94, 0.72, 0.08), (0.18, 0.08, 0.62)),
    ((0.08, 0.74, 0.24), (0.88, 0.12, 0.08)),
    ((0.85, 0.18, 0.84), (0.08, 0.68, 0.34)),
    ((0.95, 0.45, 0.08), (0.10, 0.24, 0.88)),
)


def _alpha_from_edge(x: np.ndarray, edge_x: np.ndarray, width: float) -> np.ndarray:
    if width <= 0:
        return (x >= edge_x[:, None]).astype(np.float64)
    return np.clip(0.5 + (x - edge_x[:, None]) / width, 0.0, 1.0)


def make_scene(seed: int, split: str, size: int = 96) -> Scene:
    """Create a deterministic, single-boundary animation-style scene."""
    if size % 2:
        raise ValueError("scene size must be even")
    rng = np.random.default_rng(seed)
    h = w = size
    yy = np.arange(h, dtype=np.float64)
    xx = np.arange(w, dtype=np.float64)[None, :]

    family = ("straight", "sine", "chevron")[seed % 3]
    slope = np.tan(np.deg2rad(rng.uniform(-32.0, 32.0)))
    phase = rng.uniform(-0.45, 0.45)
    center = 0.5 * (w - 1) + phase
    base = center + slope * (yy - 0.5 * (h - 1))
    if family == "sine":
        period = rng.uniform(26.0, 44.0)
        base += rng.uniform(1.5, 3.5) * np.sin(2.0 * np.pi * yy / period + rng.uniform(0, 2 * np.pi))
    elif family == "chevron":
        period = rng.uniform(28.0, 42.0)
        q = np.mod(yy + rng.uniform(0, period), period) / period
        base += rng.uniform(1.0, 3.0) * (4.0 * np.abs(q - 0.5) - 1.0)

    conditions = (
        "coedge", "coedge", "coedge", "soft_chroma",
        "misaligned", "isoluminant", "ridge", "luma_only",
    )
    condition = conditions[seed % len(conditions)]
    rgb0, rgb1 = PALETTE[seed % len(PALETTE)]
    y0, u0, v0 = rgb_to_yuv709(rgb0)
    y1, u1, v1 = rgb_to_yuv709(rgb1)

    if condition == "isoluminant":
        ym = 0.5 * (y0 + y1)
        y0 = y1 = ym
    if condition == "luma_only":
        u1, v1 = u0, v0

    luma_width = rng.uniform(0.65, 1.15)
    chroma_width = luma_width
    chroma_edge = base.copy()
    if condition == "soft_chroma":
        chroma_width = rng.uniform(4.0, 7.0)
    elif condition == "misaligned":
        chroma_edge += rng.choice((-1.0, 1.0)) * rng.uniform(0.75, 1.25)

    ay = _alpha_from_edge(xx, base, luma_width)
    ac = _alpha_from_edge(xx, chroma_edge, chroma_width)
    y = y0 + (y1 - y0) * ay
    u = u0 + (u1 - u0) * ac
    v = v0 + (v1 - v0) * ac

    if condition == "ridge":
        distance = np.abs(xx - base[:, None])
        ink = np.clip(1.0 - distance / rng.uniform(1.2, 2.0), 0.0, 1.0)
        y = y * (1.0 - ink) + 0.04 * ink

    if seed % 5 == 0:
        y = np.clip(y + rng.normal(0.0, 0.0025, y.shape), 0.0, 1.0)

    return Scene(
        name=f"{split}_{seed:05d}_{family}_{condition}",
        split=split,
        family=family,
        condition=condition,
        y=y.astype(np.float32),
        u=u.astype(np.float32),
        v=v.astype(np.float32),
        alpha=ac.astype(np.float32),
        edge_x=chroma_edge.astype(np.float64),
        side_u=(u0, u1),
        side_v=(v0, v1),
    )


def generate_scenes(split: str, size: int = 96) -> list[Scene]:
    if split == "dev":
        seeds = range(100, 112)
    elif split == "test":
        seeds = range(1000, 1032)
    else:
        raise ValueError("split must be 'dev' or 'test'")
    return [make_scene(seed, split, size) for seed in seeds]


def _sinc(x: np.ndarray) -> np.ndarray:
    return np.sinc(x)


def _cubic(x: np.ndarray, b: float = 0.0, c: float = 0.6) -> np.ndarray:
    ax = np.abs(x)
    out = np.zeros_like(ax, dtype=np.float64)
    m1 = ax < 1.0
    z = ax[m1]
    out[m1] = ((12 - 9 * b - 6 * c) * z**3 + (-18 + 12 * b + 6 * c) * z**2 + (6 - 2 * b)) / 6
    m2 = (ax >= 1.0) & (ax < 2.0)
    z = ax[m2]
    out[m2] = ((-b - 6 * c) * z**3 + (6 * b + 30 * c) * z**2 + (-12 * b - 48 * c) * z + (8 * b + 24 * c)) / 6
    return out


def kernel_values(kind: str, distance: np.ndarray) -> np.ndarray:
    ad = np.abs(distance)
    if kind == "nearest":
        return (ad < 0.5).astype(np.float64)
    if kind == "bilinear" or kind == "triangle":
        return np.maximum(0.0, 1.0 - ad)
    if kind == "bicubic":
        return _cubic(distance)
    if kind == "lanczos":
        return np.where(ad < 3.0, _sinc(distance) * _sinc(distance / 3.0), 0.0)
    raise ValueError(f"unknown kernel: {kind}")


def downsample_matrix(n: int, kind: str, shift: float) -> np.ndarray:
    """Return a matrix mapping n full-resolution samples to n/2 samples."""
    if n % 2:
        raise ValueError("4:2:0 evaluation requires even dimensions")
    positions = (np.arange(n // 2) + 0.5) * 2.0 - 0.5 + shift
    source = np.arange(n, dtype=np.float64)[None, :]
    if kind == "box":
        weights = ((source >= positions[:, None] - 1.0) &
                   (source < positions[:, None] + 1.0)).astype(np.float64)
    else:
        weights = kernel_values(kind, (source - positions[:, None]) / 2.0)
    sums = weights.sum(axis=1, keepdims=True)
    if np.any(np.abs(sums) < 1e-12):
        raise RuntimeError(f"empty degradation support for {kind}")
    return weights / sums


def downsample_plane(plane: np.ndarray, kind: str, siting: str) -> np.ndarray:
    sx, sy, _ = SITING[siting]
    wy = downsample_matrix(plane.shape[0], kind, sy)
    wx = downsample_matrix(plane.shape[1], kind, sx)
    return (wy @ plane.astype(np.float64) @ wx.T).astype(np.float32)


def degrade_scene(scene: Scene, kind: str, siting: str, bits: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = scene.y.copy()
    u = downsample_plane(scene.u, kind, siting)
    v = downsample_plane(scene.v, kind, siting)
    if bits:
        levels = float((1 << bits) - 1)
        y = (np.rint(np.clip(y, 0, 1) * levels) / levels).astype(np.float32)
        u = (np.rint(np.clip(u + 0.5, 0, 1) * levels) / levels - 0.5).astype(np.float32)
        v = (np.rint(np.clip(v + 0.5, 0, 1) * levels) / levels - 0.5).astype(np.float32)
    return y, u, v


def reconstruction_matrix(n_low: int, n_full: int, kind: str, shift: float) -> np.ndarray:
    source_positions = (np.arange(n_low) + 0.5) * (n_full / n_low) - 0.5 + shift
    output_positions = np.arange(n_full, dtype=np.float64)[:, None]
    scale = n_full / n_low
    distance = (output_positions - source_positions[None, :]) / scale
    if kind == "nearest":
        nearest = np.argmin(np.abs(distance), axis=1)
        weights = np.zeros_like(distance)
        weights[np.arange(n_full), nearest] = 1.0
    else:
        weights = kernel_values(kind, distance)
    sums = weights.sum(axis=1, keepdims=True)
    good = np.abs(sums[:, 0]) >= 1e-12
    weights[good] /= sums[good]
    if not np.all(good):
        nearest = np.argmin(np.abs(distance[~good]), axis=1)
        weights[~good] = 0.0
        weights[np.flatnonzero(~good), nearest] = 1.0
    return weights


def reconstruct_plane(plane: np.ndarray, shape: tuple[int, int], kind: str, siting: str) -> np.ndarray:
    sx, sy, _ = SITING[siting]
    wy = reconstruction_matrix(plane.shape[0], shape[0], kind, sy)
    wx = reconstruction_matrix(plane.shape[1], shape[1], kind, sx)
    return (wy @ plane.astype(np.float64) @ wx.T).astype(np.float32)


def _crossing(profile: np.ndarray, level: float, expected: float) -> float | None:
    values = np.asarray(profile, dtype=np.float64) - level
    candidates: list[float] = []
    for i in np.flatnonzero(values[:-1] * values[1:] <= 0):
        den = values[i + 1] - values[i]
        if abs(den) < 1e-12:
            candidates.append(i + 0.5)
        else:
            candidates.append(i - values[i] / den)
    if not candidates:
        return None
    return min(candidates, key=lambda value: abs(value - expected))


def profile_metrics(scene: Scene, out_u: np.ndarray, out_v: np.ndarray) -> dict[str, float]:
    du = scene.side_u[1] - scene.side_u[0]
    dv = scene.side_v[1] - scene.side_v[0]
    norm2 = du * du + dv * dv
    if norm2 < 1e-10:
        return {key: float("nan") for key in (
            "phase_px", "phase_abs_px", "alias_px", "spread_delta_px",
            "bleed_mass", "ringing_mass",
        )}

    alpha = ((out_u - scene.side_u[0]) * du + (out_v - scene.side_v[0]) * dv) / norm2
    phases: list[float] = []
    spreads: list[float] = []
    gt_spreads: list[float] = []
    bleed: list[float] = []
    ringing: list[float] = []
    h, w = scene.shape
    for y in range(6, h - 6):
        expected = float(scene.edge_x[y])
        if expected < 10 or expected > w - 11:
            continue
        pred = alpha[y]
        truth = scene.alpha[y]
        p50 = _crossing(pred, 0.5, expected)
        g50 = _crossing(truth, 0.5, expected)
        p10 = _crossing(pred, 0.1, expected - 1.0)
        p90 = _crossing(pred, 0.9, expected + 1.0)
        g10 = _crossing(truth, 0.1, expected - 1.0)
        g90 = _crossing(truth, 0.9, expected + 1.0)
        if p50 is not None and g50 is not None:
            phases.append(p50 - g50)
        if None not in (p10, p90, g10, g90) and p90 >= p10 and g90 >= g10:
            spreads.append(p90 - p10)
            gt_spreads.append(g90 - g10)
        lo = max(0, int(np.floor(expected)) - 7)
        hi = min(w, int(np.ceil(expected)) + 8)
        bleed.append(float(np.mean(np.abs(pred[lo:hi] - truth[lo:hi]))))
        ringing.append(float(np.mean(np.maximum(-pred[lo:hi], 0.0) + np.maximum(pred[lo:hi] - 1.0, 0.0))))

    if not phases:
        phase = phase_abs = alias = float("nan")
    else:
        phase_values = np.asarray(phases)
        phase = float(np.mean(phase_values))
        phase_abs = float(abs(phase))
        alias = float(np.sqrt(np.mean(np.diff(phase_values) ** 2) / 2.0)) if len(phases) > 1 else 0.0
    spread_delta = float(np.mean(spreads) - np.mean(gt_spreads)) if spreads else float("nan")
    return {
        "phase_px": phase,
        "phase_abs_px": phase_abs,
        "alias_px": alias,
        "spread_delta_px": spread_delta,
        "bleed_mass": float(np.mean(bleed)) if bleed else float("nan"),
        "ringing_mass": float(np.mean(ringing)) if ringing else float("nan"),
    }


def measure_scene(scene: Scene, out_u: np.ndarray, out_v: np.ndarray) -> dict[str, float]:
    if out_u.shape != scene.shape or out_v.shape != scene.shape:
        raise ValueError("reconstruction shape differs from ground truth")
    err_u = out_u.astype(np.float64) - scene.u
    err_v = out_v.astype(np.float64) - scene.v
    abs_error = 0.5 * (np.abs(err_u) + np.abs(err_v))
    mse = 0.5 * (err_u * err_u + err_v * err_v)
    h, w = scene.shape
    yy, xx = np.mgrid[0:h, 0:w]
    distance = np.abs(xx - scene.edge_x[:, None])
    interior = (xx >= 6) & (xx < w - 6) & (yy >= 6) & (yy < h - 6)
    edge = interior & (distance <= 4.0)
    smooth = interior & (distance >= 10.0)
    result = {
        "mae": float(np.mean(abs_error[interior])),
        "psnr": float(-10.0 * np.log10(max(float(np.mean(mse[interior])), 1e-15))),
        "edge_mae": float(np.mean(abs_error[edge])),
        "smooth_mae": float(np.mean(abs_error[smooth])),
    }
    result.update(profile_metrics(scene, out_u, out_v))
    return result

