"""Formula-level analytic baselines used by the paper evaluation."""

from __future__ import annotations

import numpy as np

from evaluation.common import SITING, downsample_plane, reconstruct_plane


def _grid_mapping(shape: tuple[int, int], low_shape: tuple[int, int], siting: str):
    h, w = shape
    lh, lw = low_shape
    if h != 2 * lh or w != 2 * lw:
        raise ValueError("analytic baselines currently implement exact 2x reconstruction")
    sx, sy, _ = SITING[siting]
    yy, xx = np.mgrid[0:h, 0:w]
    scx = (xx + 0.5 - sx) / 2.0 - 0.5
    scy = (yy + 0.5 - sy) / 2.0 - 0.5
    return yy, xx, scy, scx


def _bilinear_sample(plane: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, plane.shape[1] - 1.000001)
    y = np.clip(y, 0.0, plane.shape[0] - 1.000001)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    return (
        plane[y0, x0] * (1 - fx) * (1 - fy)
        + plane[y0, x0 + 1] * fx * (1 - fy)
        + plane[y0 + 1, x0] * (1 - fx) * fy
        + plane[y0 + 1, x0 + 1] * fx * fy
    )


def _guide_at_chroma_samples(y: np.ndarray, low_shape: tuple[int, int], siting: str) -> np.ndarray:
    sx, sy, _ = SITING[siting]
    cy, cx = np.mgrid[0:low_shape[0], 0:low_shape[1]]
    lx = (cx + 0.5) * 2.0 - 0.5 + sx
    ly = (cy + 0.5) * 2.0 - 0.5 + sy
    return _bilinear_sample(y.astype(np.float64), lx, ly)


def joint_bilateral(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    siting: str,
    sigma_spatial: float = 2.5,
    sigma_range: float = 0.05,
    radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical Gaussian joint bilateral upsampling on the source grid."""
    yy, xx, scy, scx = _grid_mapping(y.shape, u.shape, siting)
    sx, sy, _ = SITING[siting]
    guide_low = _guide_at_chroma_samples(y, u.shape, siting)
    bx = np.floor(scx).astype(np.int64)
    by = np.floor(scy).astype(np.int64)
    acc_u = np.zeros_like(y, dtype=np.float64)
    acc_v = np.zeros_like(y, dtype=np.float64)
    total = np.zeros_like(y, dtype=np.float64)
    inv_s = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_r = 1.0 / (2.0 * sigma_range * sigma_range)
    for oy in range(-radius, radius + 1):
        iy = np.clip(by + oy, 0, u.shape[0] - 1)
        py = (iy + 0.5) * 2.0 - 0.5 + sy
        for ox in range(-radius, radius + 1):
            ix = np.clip(bx + ox, 0, u.shape[1] - 1)
            px = (ix + 0.5) * 2.0 - 0.5 + sx
            spatial = np.exp(-((px - xx) ** 2 + (py - yy) ** 2) * inv_s)
            delta = guide_low[iy, ix] - y
            weight = spatial * np.exp(-(delta * delta) * inv_r)
            acc_u += weight * u[iy, ix]
            acc_v += weight * v[iy, ix]
            total += weight
    nearest_x = np.clip(np.rint(scx).astype(np.int64), 0, u.shape[1] - 1)
    nearest_y = np.clip(np.rint(scy).astype(np.int64), 0, u.shape[0] - 1)
    good = total > 1e-12
    out_u = np.where(good, acc_u / np.maximum(total, 1e-12), u[nearest_y, nearest_x])
    out_v = np.where(good, acc_v / np.maximum(total, 1e-12), v[nearest_y, nearest_x])
    return out_u.astype(np.float32), out_v.astype(np.float32)


def _edge_shift(plane: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a plane with replicated boundaries."""
    iy = np.clip(np.arange(plane.shape[0]) + dy, 0, plane.shape[0] - 1)
    ix = np.clip(np.arange(plane.shape[1]) + dx, 0, plane.shape[1] - 1)
    return plane[np.ix_(iy, ix)]


def wada_ejbf(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    siting: str,
    sigma_spatial: float = 4.0,
    sigma_luma: float = 0.03,
    sigma_chroma: float = 0.10,
    radius: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Wada et al.'s extended joint bilateral filter (EJBF).

    The decoded chroma is first bilinearly expanded, then both chroma planes
    and luma jointly determine the Gaussian range weight (Eq. 10 in Wada et
    al., MTA 2015). The published normalized-range parameters are used by
    default. ``radius=12`` truncates the spatial Gaussian at three sigma.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if min(sigma_spatial, sigma_luma, sigma_chroma) <= 0:
        raise ValueError("EJBF sigmas must be positive")

    guide = y.astype(np.float64)
    base_u = reconstruct_plane(u, y.shape, "bilinear", siting).astype(np.float64)
    base_v = reconstruct_plane(v, y.shape, "bilinear", siting).astype(np.float64)
    acc_u = np.zeros_like(guide)
    acc_v = np.zeros_like(guide)
    total = np.zeros_like(guide)
    inv_s = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_y = 1.0 / (2.0 * sigma_luma * sigma_luma)
    inv_c = 1.0 / (2.0 * sigma_chroma * sigma_chroma)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yq = _edge_shift(guide, dy, dx)
            uq = _edge_shift(base_u, dy, dx)
            vq = _edge_shift(base_v, dy, dx)
            exponent = -(dx * dx + dy * dy) * inv_s
            exponent -= (yq - guide) ** 2 * inv_y
            exponent -= ((uq - base_u) ** 2 + (vq - base_v) ** 2) * inv_c
            weight = np.exp(exponent)
            acc_u += weight * uq
            acc_v += weight * vq
            total += weight
    out_u = acc_u / np.maximum(total, 1e-15)
    out_v = acc_v / np.maximum(total, 1e-15)
    return out_u.astype(np.float32), out_v.astype(np.float32)


def _box_mean(a: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return a.astype(np.float64)
    padded = np.pad(a.astype(np.float64), radius, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    width = 2 * radius + 1
    sums = (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    )
    return sums / float(width * width)


def guided_filter_upsample(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    siting: str,
    radius: int = 2,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """He et al. guided-filter upsampling with low-res affine coefficients."""
    guide = downsample_plane(y, "box", siting).astype(np.float64)
    mean_i = _box_mean(guide, radius)
    var_i = _box_mean(guide * guide, radius) - mean_i * mean_i
    outputs = []
    for chroma in (u.astype(np.float64), v.astype(np.float64)):
        mean_p = _box_mean(chroma, radius)
        cov_ip = _box_mean(guide * chroma, radius) - mean_i * mean_p
        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i
        mean_a = _box_mean(a, radius)
        mean_b = _box_mean(b, radius)
        ah = reconstruct_plane(mean_a, y.shape, "bilinear", siting)
        bh = reconstruct_plane(mean_b, y.shape, "bilinear", siting)
        outputs.append((ah * y + bh).astype(np.float32))
    return outputs[0], outputs[1]


def korhonen_upsample(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    siting: str,
    theta: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized 2x form of Korhonen's four-candidate luma-MSD rule."""
    _, _, scy, scx = _grid_mapping(y.shape, u.shape, siting)
    x0 = np.floor(scx).astype(np.int64)
    y0 = np.floor(scy).astype(np.int64)
    candidate_x = np.stack((x0, x0, x0 + 1, x0 + 1))
    candidate_y = np.stack((y0, y0 + 1, y0, y0 + 1))
    candidate_x = np.clip(candidate_x, 0, u.shape[1] - 1)
    candidate_y = np.clip(candidate_y, 0, u.shape[0] - 1)

    mean_y = downsample_plane(y, "box", siting).astype(np.float64)
    mean_y2 = downsample_plane(y * y, "box", siting).astype(np.float64)
    target = y.astype(np.float64)[None, :, :]
    msd = target * target - 2.0 * target * mean_y[candidate_y, candidate_x] + mean_y2[candidate_y, candidate_x]
    msd = np.maximum(msd, 0.0)
    span = msd.max(axis=0) - msd.min(axis=0)
    alpha = np.minimum(span / theta, 1.0)

    order = np.argsort(msd, axis=0)
    sorted_msd = np.take_along_axis(msd, order, axis=0)
    inverse_rank = np.zeros_like(msd)
    np.put_along_axis(inverse_rank, order, sorted_msd[::-1], axis=0)

    dx = np.abs(scx[None, :, :] - candidate_x)
    dy = np.abs(scy[None, :, :] - candidate_y)
    distance_weight = np.maximum(0.0, 1.0 - dx) * np.maximum(0.0, 1.0 - dy)
    weights = np.maximum(inverse_rank, 1e-15) ** alpha[None, :, :]
    weights *= np.maximum(distance_weight, 1e-15) ** (1.0 - alpha[None, :, :])
    weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-15)
    out_u = np.sum(weights * u[candidate_y, candidate_x], axis=0)
    out_v = np.sum(weights * v[candidate_y, candidate_x], axis=0)
    return out_u.astype(np.float32), out_v.astype(np.float32)


def _bessel_j1(z: np.ndarray) -> np.ndarray:
    """Power-series J1, accurate for the GALOSH support |z| <= 3*pi."""
    z = np.asarray(z, dtype=np.float64)
    half = 0.5 * z
    term = half.copy()
    total = term.copy()
    for k in range(1, 36):
        term *= -(half * half) / (k * (k + 1.0))
        total += term
    return total


def _jinc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    pix = np.pi * x
    return np.where(np.abs(x) < 1e-8, 1.0, 2.0 * _bessel_j1(pix) / np.where(pix == 0, 1.0, pix))


def galosh_form_upsample(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    siting: str,
    sigma_range: float = 0.05,
    anti_ringing: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """GALOSH-form signed EWA Jinc-Jinc-3 joint bilateral reconstruction.

    This generalizes GALOSH's top-left 2x grid to H.273 siting while preserving
    its signed radial kernel, shared luma range weights, signed denominator
    guard, and nearest-2x2 hull clamp.
    """
    yy, xx, scy, scx = _grid_mapping(y.shape, u.shape, siting)
    sx, sy, _ = SITING[siting]
    guide_low = _guide_at_chroma_samples(y, u.shape, siting)
    bx = np.floor(scx).astype(np.int64)
    by = np.floor(scy).astype(np.int64)
    acc_u = np.zeros_like(y, dtype=np.float64)
    acc_v = np.zeros_like(y, dtype=np.float64)
    total = np.zeros_like(y, dtype=np.float64)
    inv_r = 1.0 / (2.0 * sigma_range * sigma_range)
    for oy in range(-2, 3):
        iy = np.clip(by + oy, 0, u.shape[0] - 1)
        py = (iy + 0.5) * 2.0 - 0.5 + sy
        for ox in range(-2, 3):
            ix = np.clip(bx + ox, 0, u.shape[1] - 1)
            px = (ix + 0.5) * 2.0 - 0.5 + sx
            radius = np.hypot(px - xx, py - yy)
            spatial = np.where(radius < 3.0, _jinc(radius) * _jinc(radius / 3.0), 0.0)
            delta = guide_low[iy, ix] - y
            weight = spatial * np.exp(-(delta * delta) * inv_r)
            acc_u += weight * u[iy, ix]
            acc_v += weight * v[iy, ix]
            total += weight

    safe = np.where(np.abs(total) > 1e-6, total, np.where(total < 0, -1e-6, 1e-6))
    out_u = acc_u / safe
    out_v = acc_v / safe
    nx0 = np.clip(np.floor(scx).astype(np.int64), 0, u.shape[1] - 1)
    nx1 = np.clip(nx0 + 1, 0, u.shape[1] - 1)
    ny0 = np.clip(np.floor(scy).astype(np.int64), 0, u.shape[0] - 1)
    ny1 = np.clip(ny0 + 1, 0, u.shape[0] - 1)
    nearest_u = np.stack((u[ny0, nx0], u[ny0, nx1], u[ny1, nx0], u[ny1, nx1]))
    nearest_v = np.stack((v[ny0, nx0], v[ny0, nx1], v[ny1, nx0], v[ny1, nx1]))
    if anti_ringing:
        out_u = np.clip(out_u, nearest_u.min(axis=0), nearest_u.max(axis=0))
        out_v = np.clip(out_v, nearest_v.min(axis=0), nearest_v.max(axis=0))
    return out_u.astype(np.float32), out_v.astype(np.float32)
