"""Scalar NumPy reference for lgcr.Downsample.

This intentionally favors a direct formula transcription over speed. Tests use
small arrays to pin the polyphase baseline, direction/gate math, and quality-2
candidate decision independently from the C++ row-ring implementation.
"""

from __future__ import annotations

import math

import numpy as np


_DIRECTIONS = (
    (1.0, 0.0),
    (0.9238795, 0.3826834),
    (0.7071068, 0.7071068),
    (0.3826834, 0.9238795),
    (0.0, 1.0),
    (-0.3826834, 0.9238795),
    (-0.7071068, 0.7071068),
    (-0.9238795, 0.3826834),
)

_LOCATIONS = {
    "left": (-0.5, 0.0),
    "center": (0.0, 0.0),
    "topleft": (-0.5, -0.5),
    "top": (0.0, -0.5),
    "bottomleft": (-0.5, 0.5),
    "bottom": (0.0, 0.5),
}


def _at(plane: np.ndarray, x: int, y: int) -> float:
    y = min(max(y, 0), plane.shape[0] - 1)
    x = min(max(x, 0), plane.shape[1] - 1)
    return float(plane[y, x])


def _sinc(x: float) -> float:
    if abs(x) < 1e-8:
        return 1.0
    x *= math.pi
    return math.sin(x) / x


def _kernel(kernel: str, x: float) -> float:
    x = abs(x)
    if kernel == "lanczos3":
        return _sinc(x) * _sinc(x / 3.0) if x < 3.0 else 0.0
    if x < 1.0:
        return ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0
    if x < 2.0:
        z = x - 1.0
        return ((-6.0 / 11.0 * z + 270.0 / 209.0) * z - 156.0 / 209.0) * z
    if x < 3.0:
        z = x - 2.0
        return ((1.0 / 11.0 * z - 45.0 / 209.0) * z + 26.0 / 209.0) * z
    return 0.0


def _weights(size: int, kernel: str, shift: float) -> tuple[np.ndarray, np.ndarray]:
    outputs = size // 2
    if kernel == "binomial":
        integer_phase = abs(abs(shift) - 0.5) < 1e-6
        taps = (np.array([1, 4, 6, 4, 1], np.float32) / np.float32(16)
                if integer_phase else
                np.array([1, 3, 3, 1], np.float32) / np.float32(8))
        starts = np.empty(outputs, np.int32)
        for i in range(outputs):
            center = 2.0 * i + 0.5 + shift
            starts[i] = round(center) - 2 if integer_phase else math.floor(center) - 1
        return starts, np.repeat(taps[None, :], outputs, axis=0)

    support = 6.0
    taps = 12
    starts = np.empty(outputs, np.int32)
    table = np.empty((outputs, taps), np.float32)
    for i in range(outputs):
        center = 2.0 * i + 0.5 + shift
        first = math.ceil(center - support)
        first = min(max(first, -taps + 1), size - 1)
        if first + taps > size:
            first = max(-taps + 1, size - taps)
        starts[i] = first
        raw = [_kernel(kernel, (first + tap - center) * 0.5) for tap in range(taps)]
        row = np.asarray(raw, np.float32)
        total = sum(raw)
        if abs(total) > 1e-9:
            row /= np.float32(total)
        table[i] = row
    return starts, table


def _baseline(plane: np.ndarray, kernel: str, shift_x: float,
              shift_y: float) -> np.ndarray:
    sx, wx = _weights(plane.shape[1], kernel, shift_x)
    sy, wy = _weights(plane.shape[0], kernel, shift_y)
    horizontal = np.empty((plane.shape[0], plane.shape[1] // 2), np.float32)
    for y in range(plane.shape[0]):
        for x in range(horizontal.shape[1]):
            value = np.float32(0)
            for tap, weight in enumerate(wx[x]):
                value = np.float32(value + weight * _at(plane, int(sx[x]) + tap, y))
            horizontal[y, x] = value
    output = np.empty((plane.shape[0] // 2, plane.shape[1] // 2), np.float32)
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            value = np.float32(0)
            for tap, weight in enumerate(wy[y]):
                value = np.float32(value + weight * _at(horizontal, x, int(sy[y]) + tap))
            output[y, x] = value
    return output


def _classify(y_plane: np.ndarray, u_plane: np.ndarray, v_plane: np.ndarray,
              cx: float, cy: float, quality: int) -> tuple[int, int, float, float, bool]:
    ix, iy = math.floor(cx + 0.5), math.floor(cy + 0.5)
    if quality == 0:
        tl, tc, tr = (_at(y_plane, ix + x, iy - 1) for x in (-1, 0, 1))
        ml, mr = _at(y_plane, ix - 1, iy), _at(y_plane, ix + 1, iy)
        bl, bc, br = (_at(y_plane, ix + x, iy + 1) for x in (-1, 0, 1))
        center_gx = (tr + 2 * mr + br - tl - 2 * ml - bl) * 0.125
        center_gy = (bl + 2 * bc + br - tl - 2 * tc - tr) * 0.125
        jxx, jxy, jyy = center_gx**2, center_gx * center_gy, center_gy**2
    else:
        jxx = jxy = jyy = 0.0
        center_gx = center_gy = 0.0
        for k in (-1, 0, 1):
            gx = 0.5 * (_at(y_plane, ix + 1, iy + k) - _at(y_plane, ix - 1, iy + k))
            gy = 0.5 * (_at(y_plane, ix, iy + k + 1) - _at(y_plane, ix, iy + k - 1))
            jxx, jxy, jyy = jxx + gx * gx, jxy + gx * gy, jyy + gy * gy
            if k == 0:
                center_gx, center_gy = gx, gy
    trace = jxx + jyy
    if trace < 1e-6:
        return 0, 1, 0.0, 0.0, True
    coherence2 = ((jxx - jyy)**2 + 4 * jxy * jxy) / (trace * trace + 1e-12)
    responses = []
    for direction in range(0, 8, 2 if quality == 0 else 1):
        nx, ny = _DIRECTIONS[direction]
        response = nx * nx * jxx + 2 * nx * ny * jxy + ny * ny * jyy
        responses.append((response, direction))
    responses.sort(reverse=True)
    best, primary = responses[0]
    second, secondary = responses[1]
    mix = 0.0
    if quality and best > 1e-12:
        ratio = min(max(second / best, 0.0), 1.0)
        mix = 0.5 * min(max((ratio - 0.8535534) / (1.0 - 0.8535534), 0.0), 1.0)
    ugx = 0.5 * (_at(u_plane, ix + 1, iy) - _at(u_plane, ix - 1, iy))
    ugy = 0.5 * (_at(u_plane, ix, iy + 1) - _at(u_plane, ix, iy - 1))
    vgx = 0.5 * (_at(v_plane, ix + 1, iy) - _at(v_plane, ix - 1, iy))
    vgy = 0.5 * (_at(v_plane, ix, iy + 1) - _at(v_plane, ix, iy - 1))
    chroma_energy = ugx**2 + ugy**2 + vgx**2 + vgy**2
    nx, ny = _DIRECTIONS[primary]
    aligned = (ugx * nx + ugy * ny)**2 + (vgx * nx + vgy * ny)**2
    step_x, step_y = round(nx), round(ny)
    curvature_u = (_at(u_plane, ix + step_x, iy + step_y) - 2 * _at(u_plane, ix, iy) +
                   _at(u_plane, ix - step_x, iy - step_y))
    curvature_v = (_at(v_plane, ix + step_x, iy + step_y) - 2 * _at(v_plane, ix, iy) +
                   _at(v_plane, ix - step_x, iy - step_y))
    curvature_energy = curvature_u**2 + curvature_v**2
    sharpness = curvature_energy / (curvature_energy + chroma_energy + 1e-12)
    center_energy = center_gx**2 + center_gy**2
    gate = (trace / (trace + 3e-5) * coherence2 / (coherence2 + 0.08) *
            chroma_energy / (chroma_energy + 2e-5) *
            aligned / (chroma_energy + 1e-12) *
            center_energy / (center_energy + 1e-5) * sharpness)
    if quality == 0:
        gate *= 0.985
    elif quality == 2:
        gate = min(1.0, gate * 1.01)
    if sharpness < (0.20 if quality == 0 else 0.12):
        gate = 0.0
    isotropic = coherence2 < (0.20 if quality == 0 else 0.10)
    return primary, secondary, mix, min(max(gate, 0.0), 1.0), isotropic


def _guided_values(y_plane: np.ndarray, u_plane: np.ndarray, v_plane: np.ndarray,
                   cx: float, cy: float, envelopes: list[tuple[int, float, bool]]) -> list[tuple[float, float]]:
    max_radius = max(radius for radius, _, _ in envelopes)
    ix, iy = math.floor(cx), math.floor(cy)
    fx, fy = cx - ix, cy - iy
    a, b = _at(y_plane, ix, iy), _at(y_plane, ix + 1, iy)
    c, d = _at(y_plane, ix, iy + 1), _at(y_plane, ix + 1, iy + 1)
    center_luma = a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy
    sums = [[0.0, 0.0, 0.0] for _ in envelopes]
    for sy in range(math.ceil(cy - max_radius), math.floor(cy + max_radius) + 1):
        for sx in range(math.ceil(cx - max_radius), math.floor(cx + max_radius) + 1):
            dx, dy = sx - cx, sy - cy
            delta = _at(y_plane, sx, sy) - center_luma
            rw = 2.25e-4 / (2.25e-4 + delta * delta)
            for index, (radius, anisotropy, isotropic) in enumerate(envelopes):
                if abs(dx) > radius or abs(dy) > radius:
                    continue
                direction = anisotropy if isinstance(anisotropy, tuple) else None
                if isotropic:
                    sw = 1.0 / (1.0 + 0.42 * (dx * dx + dy * dy))
                else:
                    direction_index, amount = direction
                    nx, ny = _DIRECTIONS[direction_index]
                    normal = dx * nx + dy * ny
                    tangent = -dx * ny + dy * nx
                    sw = 1.0 / (1.0 + (0.9 + amount) * normal**2 + 0.16 * tangent**2)
                weight = rw * sw
                sums[index][0] += weight
                sums[index][1] += weight * _at(u_plane, sx, sy)
                sums[index][2] += weight * _at(v_plane, sx, sy)
    return [(su / weight, sv / weight) for weight, su, sv in sums]


def _pixel_candidates(y_plane: np.ndarray, u_plane: np.ndarray, v_plane: np.ndarray,
                      cx: float, cy: float, quality: int) -> tuple[list[tuple[float, float]], float]:
    primary, secondary, mix, gate, isotropic = _classify(
        y_plane, u_plane, v_plane, cx, cy, quality)
    if gate <= 1e-4:
        gate = 0.0
    if quality == 0:
        values = _guided_values(y_plane, u_plane, v_plane, cx, cy,
                                [(2, (primary, 1.8), isotropic)])
        return values, gate
    if quality == 1:
        values = _guided_values(y_plane, u_plane, v_plane, cx, cy, [
            (3, (primary, 2.2), isotropic), (3, (secondary, 2.2), isotropic)])
        return [tuple(a + mix * (b - a) for a, b in zip(values[0], values[1]))], gate
    values = _guided_values(y_plane, u_plane, v_plane, cx, cy, [
        (3, (primary, 2.2), isotropic), (3, (secondary, 2.2), isotropic),
        (4, (primary, 2.7), isotropic), (4, (secondary, 2.7), isotropic),
        (4, (primary, 0.0), True),
    ])
    soft = tuple(a + mix * (b - a) for a, b in zip(values[0], values[1]))
    return [soft, values[2], values[3], values[4]], gate


def downsample_reference(y_plane: np.ndarray, u_plane: np.ndarray,
                         v_plane: np.ndarray, quality: int = 0,
                         kernel: str = "spline36", strength: float = 1.0,
                         loc: str = "left") -> tuple[np.ndarray, np.ndarray]:
    """Return normalized float U/V reference arrays on the 4:2:0 grid."""
    shift_x, shift_y = _LOCATIONS[loc]
    base_u = _baseline(u_plane, kernel, shift_x, shift_y)
    base_v = _baseline(v_plane, kernel, shift_x, shift_y)
    if strength == 0:
        return base_u, base_v

    height, width = base_u.shape
    candidates = np.empty((height, width, 4, 2), np.float32)
    gates = np.empty((height, width), np.float32)
    count = 1 if quality < 2 else 4
    for y in range(height):
        cy = 2.0 * y + 0.5 + shift_y
        for x in range(width):
            cx = 2.0 * x + 0.5 + shift_x
            values, gate = _pixel_candidates(y_plane, u_plane, v_plane, cx, cy, quality)
            gates[y, x] = gate
            for candidate in range(count):
                candidates[y, x, candidate] = values[candidate]

    if quality < 2:
        output_u = base_u + np.float32(strength) * gates * (candidates[:, :, 0, 0] - base_u)
        output_v = base_v + np.float32(strength) * gates * (candidates[:, :, 0, 1] - base_v)
        return output_u.astype(np.float32), output_v.astype(np.float32)

    output_u = np.empty_like(base_u)
    output_v = np.empty_like(base_v)
    full_u = base_u[:, :, None] + gates[:, :, None] * (candidates[:, :, :, 0] - base_u[:, :, None])
    full_v = base_v[:, :, None] + gates[:, :, None] * (candidates[:, :, :, 1] - base_v[:, :, None])
    all_u = full_u
    all_v = full_v
    for y in range(height):
        for x in range(width):
            scores = np.zeros(4, np.float64)
            for hy in (2 * y, 2 * y + 1):
                for hx in (2 * x, 2 * x + 1):
                    gx = 0.5 * (_at(y_plane, hx + 1, hy) - _at(y_plane, hx - 1, hy))
                    gy = 0.5 * (_at(y_plane, hx, hy + 1) - _at(y_plane, hx, hy - 1))
                    weight = 0.125 + (gx * gx + gy * gy) / (gx * gx + gy * gy + 1e-5)
                    lx, ly = (hx - 0.5 - shift_x) * 0.5, (hy - 0.5 - shift_y) * 0.5
                    x0, y0 = math.floor(lx), math.floor(ly)
                    fx, fy = lx - x0, ly - y0
                    for candidate in range(4):
                        a = _at(all_u[:, :, candidate], x0, y0)
                        b = _at(all_u[:, :, candidate], x0 + 1, y0)
                        c = _at(all_u[:, :, candidate], x0, y0 + 1)
                        d = _at(all_u[:, :, candidate], x0 + 1, y0 + 1)
                        ru = a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy
                        a = _at(all_v[:, :, candidate], x0, y0)
                        b = _at(all_v[:, :, candidate], x0 + 1, y0)
                        c = _at(all_v[:, :, candidate], x0, y0 + 1)
                        d = _at(all_v[:, :, candidate], x0 + 1, y0 + 1)
                        rv = a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy
                        scores[candidate] += weight * ((ru - _at(u_plane, hx, hy))**2 +
                                                       (rv - _at(v_plane, hx, hy))**2)
            best = 0
            for candidate in range(1, 4):
                if scores[candidate] < scores[best]:
                    best = candidate
            output_u[y, x] = base_u[y, x] + strength * gates[y, x] * (
                candidates[y, x, best, 0] - base_u[y, x])
            output_v[y, x] = base_v[y, x] + strength * gates[y, x] * (
                candidates[y, x, best, 1] - base_v[y, x])
    return output_u, output_v
