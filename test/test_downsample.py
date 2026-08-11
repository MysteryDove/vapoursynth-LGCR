#!/usr/bin/env python3
"""Correctness and regression coverage for lgcr.Downsample."""

from __future__ import annotations

import os
import sys

import numpy as np
import vapoursynth as vs


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from downsample_reference import downsample_reference  # noqa: E402


core = vs.core
core.num_threads = 2
core.std.LoadPlugin(os.environ.get("LGCR_PLUGIN", os.path.join(ROOT, "liblgcr.so")))

F444 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 0, 0)
LOCATIONS = {
    "left": (0, -0.5, 0.0),
    "center": (1, 0.0, 0.0),
    "topleft": (2, -0.5, -0.5),
    "top": (3, 0.0, -0.5),
    "bottomleft": (4, -0.5, 0.5),
    "bottom": (5, 0.0, 0.5),
}


def make_float_clip(planes: tuple[np.ndarray, np.ndarray, np.ndarray], length: int = 1):
    height, width = planes[0].shape
    blank = core.std.BlankClip(width=width, height=height, format=F444, length=length)

    def fill(n, f):
        result = f.copy()
        for plane in range(3):
            np.copyto(np.asarray(result[plane]), planes[plane])
        return result

    return blank.std.ModifyFrame(blank, fill)


def arrays(frame):
    return [np.asarray(frame[plane]).copy() for plane in range(3)]


def expect_error(call, label: str):
    try:
        node = call()
        if hasattr(node, "get_frame"):
            node.get_frame(0)
    except vs.Error:
        return
    raise AssertionError(f"invalid Downsample input was accepted: {label}")


def test_api_location_props_and_plane_sharing():
    source = core.std.BlankClip(width=32, height=24, format=F444, length=1,
                                color=[0.25, 0.10, -0.15])
    source_frame = source.get_frame(0)
    source_y = np.asarray(source_frame[0])
    for name, (location, _, _) in LOCATIONS.items():
        frame = core.lgcr.Downsample(source, loc=name).get_frame(0)
        assert frame.width == 32 and frame.height == 24
        assert frame.format.subsampling_w == 1 and frame.format.subsampling_h == 1
        assert frame.format.sample_type == vs.FLOAT and frame.format.bits_per_sample == 32
        assert frame.props["_ChromaLocation"] == location
        output_y = np.asarray(frame[0])
        assert np.array_equal(source_y, output_y), f"Y changed for loc={name}"
        assert np.shares_memory(source_y, output_y), f"Y was copied for loc={name}"

    inherited = core.std.SetFrameProp(source, prop="LGCRDownsampleTest", intval=73)
    inherited_frame = core.lgcr.Downsample(inherited).get_frame(0)
    assert inherited_frame.props["LGCRDownsampleTest"] == 73
    assert inherited_frame.props["_ChromaLocation"] == 0
    print("Downsample API, properties, and Y-plane sharing: OK")


def test_constant_and_sample_types():
    formats = [
        core.query_video_format(vs.YUV, vs.INTEGER, bits, 0, 0)
        for bits in (8, 10, 16)
    ] + [F444]
    for fmt in formats:
        if fmt.sample_type == vs.FLOAT:
            color = [0.375, 0.125, -0.20]
        else:
            maximum = (1 << fmt.bits_per_sample) - 1
            color = [round(0.375 * maximum), round(0.625 * maximum),
                     round(0.30 * maximum)]
        clip = core.std.BlankClip(width=34, height=26, format=fmt, length=1,
                                  color=color)
        source = clip.get_frame(0)
        for kernel in ("spline36", "lanczos3", "binomial"):
            for quality in (0, 1, 2):
                frame = core.lgcr.Downsample(
                    clip, kernel=kernel, quality=quality).get_frame(0)
                assert frame.format.sample_type == fmt.sample_type
                assert frame.format.bits_per_sample == fmt.bits_per_sample
                assert np.array_equal(np.asarray(source[0]), np.asarray(frame[0]))
                for plane in (1, 2):
                    values = np.asarray(frame[plane])
                    expected = color[plane]
                    tolerance = 2e-7 if fmt.sample_type == vs.FLOAT else 0
                    assert float(np.max(np.abs(values.astype(np.float64) - expected))) <= tolerance, (
                        f"constant changed: bits={fmt.bits_per_sample}, kernel={kernel}, "
                        f"quality={quality}, plane={plane}")
    print("Downsample constants and sample types: OK")


def test_strength_baseline_and_continuity():
    height, width = 24, 32
    yy, xx = np.mgrid[:height, :width]
    planes = (
        (0.2 + 0.5 * (xx > yy + 3)).astype(np.float32),
        (-0.25 + 0.5 * (xx > yy + 3) + 0.01 * np.sin(xx)).astype(np.float32),
        (0.20 - 0.4 * (xx > yy + 3) + 0.01 * np.cos(yy)).astype(np.float32),
    )
    clip = make_float_clip(planes)
    for kernel in ("spline36", "lanczos3", "binomial"):
        baseline = arrays(core.lgcr.Downsample(
            clip, quality=0, kernel=kernel, strength=0).get_frame(0))
        for quality in (1, 2):
            candidate = arrays(core.lgcr.Downsample(
                clip, quality=quality, kernel=kernel, strength=0).get_frame(0))
            for plane in range(3):
                assert np.array_equal(baseline[plane], candidate[plane]), (
                    f"strength=0 changed with quality={quality}, kernel={kernel}")
        near_zero = arrays(core.lgcr.Downsample(
            clip, quality=2, kernel=kernel, strength=1e-5).get_frame(0))
        delta = max(float(np.max(np.abs(near_zero[p] - baseline[p]))) for p in (1, 2))
        assert delta <= 1.1e-5, f"strength is discontinuous near zero: {delta}"
    print("Downsample baseline identity and strength continuity: OK")


def test_numpy_reference():
    height, width = 16, 20
    yy, xx = np.mgrid[:height, :width]
    side = xx > yy + 2
    planes = (
        (0.2 + 0.5 * side + 0.01 * np.sin(xx)).astype(np.float32),
        (-0.2 + 0.4 * side).astype(np.float32),
        (0.2 - 0.35 * side).astype(np.float32),
    )
    clip = make_float_clip(planes)
    for location in LOCATIONS:
        for kernel in ("spline36", "lanczos3", "binomial"):
            for quality in (0, 1, 2):
                frame = core.lgcr.Downsample(
                    clip, loc=location, kernel=kernel,
                    quality=quality).get_frame(0)
                ref_u, ref_v = downsample_reference(
                    *planes, loc=location, kernel=kernel, quality=quality)
                error = max(
                    float(np.max(np.abs(np.asarray(frame[1]) - ref_u))),
                    float(np.max(np.abs(np.asarray(frame[2]) - ref_v))),
                )
                # The high-mode hard candidate decision can differ at a near
                # tie because NumPy scores in float64 while C++ accumulates
                # float32. Both selected candidates remain formula-equivalent.
                tolerance = 2e-5 if quality == 2 else 2e-6
                assert error <= tolerance, (
                    f"NumPy mismatch: loc={location}, kernel={kernel}, "
                    f"quality={quality}, error={error}")
    print("Downsample NumPy scalar reference: OK")


def test_impulse_siting():
    height = width = 32
    impulse_x = impulse_y = 16
    y_plane = np.full((height, width), 0.4, np.float32)
    u_plane = np.zeros((height, width), np.float32)
    v_plane = np.zeros((height, width), np.float32)
    u_plane[impulse_y, impulse_x] = 1.0
    v_plane[impulse_y, impulse_x] = -1.0
    clip = make_float_clip((y_plane, u_plane, v_plane))
    for location, (_, shift_x, shift_y) in LOCATIONS.items():
        frame = core.lgcr.Downsample(
            clip, loc=location, kernel="binomial", strength=0).get_frame(0)
        impulse = np.asarray(frame[1]).astype(np.float64)
        yy, xx = np.mgrid[:impulse.shape[0], :impulse.shape[1]]
        total = impulse.sum()
        centroid_x = float((impulse * xx).sum() / total)
        centroid_y = float((impulse * yy).sum() / total)
        expected_x = (impulse_x - 0.5 - shift_x) / 2.0
        expected_y = (impulse_y - 0.5 - shift_y) / 2.0
        assert abs(centroid_x - expected_x) <= 1e-6
        assert abs(centroid_y - expected_y) <= 1e-6
    print("Downsample six-location impulse phase: OK")


def hull_overshoot(value: float, low: float, high: float) -> float:
    return max(low - value, value - high, 0.0)


def test_hull_and_fallbacks():
    height, width = 32, 40
    yy, xx = np.mgrid[:height, :width]
    side = xx > yy + 4
    y_plane = (0.2 + 0.55 * side).astype(np.float32)
    u_plane = (-0.30 + 0.55 * side + 0.015 * np.sin(xx * 0.8)).astype(np.float32)
    v_plane = (0.25 - 0.50 * side + 0.012 * np.cos(yy * 0.7)).astype(np.float32)
    clip = make_float_clip((y_plane, u_plane, v_plane))
    for quality in (0, 1, 2):
        baseline = core.lgcr.Downsample(clip, quality=quality, strength=0).get_frame(0)
        guided = core.lgcr.Downsample(clip, quality=quality, strength=1).get_frame(0)
        radius = quality + 2
        for oy in range(height // 2):
            cy = 2 * oy + 0.5
            ys = range(math_ceil(cy - radius), math_floor(cy + radius) + 1)
            for ox in range(width // 2):
                cx = 2 * ox
                xs = range(math_ceil(cx - radius), math_floor(cx + radius) + 1)
                for plane, source in ((1, u_plane), (2, v_plane)):
                    samples = [source[min(max(y, 0), height - 1),
                                      min(max(x, 0), width - 1)]
                               for y in ys for x in xs]
                    low, high = float(min(samples)), float(max(samples))
                    before = hull_overshoot(float(np.asarray(baseline[plane])[oy, ox]), low, high)
                    after = hull_overshoot(float(np.asarray(guided[plane])[oy, ox]), low, high)
                    assert after <= before + 2e-6, (
                        f"guided correction increased hull overshoot at q={quality}")

    isoluminant = make_float_clip((
        np.full((height, width), 0.4, np.float32),
        (0.15 * np.sin(xx * 0.25)).astype(np.float32),
        (0.15 * np.cos(yy * 0.25)).astype(np.float32),
    ))
    luma_only = make_float_clip((
        y_plane,
        np.full((height, width), 0.12, np.float32),
        np.full((height, width), -0.08, np.float32),
    ))
    for case in (isoluminant, luma_only):
        for quality in (0, 1, 2):
            baseline = arrays(core.lgcr.Downsample(
                case, quality=quality, strength=0).get_frame(0))
            guided = arrays(core.lgcr.Downsample(
                case, quality=quality, strength=1).get_frame(0))
            for plane in (1, 2):
                assert np.array_equal(baseline[plane], guided[plane]), (
                    f"fallback changed chroma for quality={quality}")

    soft_planes = (
        (0.2 + 0.5 * (xx > width // 2)).astype(np.float32),
        (0.18 * np.tanh((xx - width // 2) / 6.0)).astype(np.float32),
        (-0.16 * np.tanh((xx - width // 2) / 6.0)).astype(np.float32),
    )
    mismatch_planes = (
        soft_planes[0],
        (-0.25 + 0.5 * (yy > height // 2)).astype(np.float32),
        (0.25 - 0.5 * (yy > height // 2)).astype(np.float32),
    )

    def roundtrip_mae(node, truth):
        errors = []
        for upsampled in (
                node.resize.Bilinear(format=F444),
                node.resize.Spline36(format=F444),
                core.lgcr.Recon(node)):
            frame = upsampled.get_frame(0)
            errors.append(np.mean([
                np.mean(np.abs(np.asarray(frame[plane]) - truth[plane]))
                for plane in (1, 2)
            ]))
        return float(np.mean(errors))

    for label, truth in (("soft-chroma", soft_planes),
                         ("direction-mismatch", mismatch_planes)):
        case = make_float_clip(truth)
        baseline_error = roundtrip_mae(
            core.lgcr.Downsample(case, strength=0), truth)
        for quality in (0, 1, 2):
            guided_error = roundtrip_mae(
                core.lgcr.Downsample(case, quality=quality), truth)
            assert guided_error <= baseline_error * 1.01 + 1e-8, (
                f"{label} MAE regressed at quality={quality}: "
                f"{baseline_error} -> {guided_error}")
    print("Downsample hull constraint and fallback gates: OK")


def math_ceil(value):
    return int(np.ceil(value))


def math_floor(value):
    return int(np.floor(value))


def test_quality_ordering():
    height, width = 96, 128
    yy, xx = np.mgrid[:height, :width]
    side = (xx > width // 2) | ((yy < height // 2) & (xx > yy + 18))
    texture = 0.008 * np.sin(xx * 0.11) * np.cos(yy * 0.07)
    planes = (
        (0.22 + 0.42 * side + texture).astype(np.float32),
        (-0.28 + 0.54 * side + 0.3 * texture).astype(np.float32),
        (0.30 - 0.57 * side - 0.2 * texture).astype(np.float32),
    )
    clip = make_float_clip(planes)
    edge = np.zeros((height, width), bool)
    edge[:, width // 2 - 4:width // 2 + 5] = True
    diagonal_distance = np.abs(xx - yy - 18)
    edge |= (yy < height // 2) & (diagonal_distance <= 4)

    def average_edge_mae(node):
        upsamplers = (
            node.resize.Bilinear(format=F444),
            node.resize.Spline36(format=F444),
            core.lgcr.Recon(node),
        )
        errors = []
        for upsampled in upsamplers:
            frame = upsampled.get_frame(0)
            errors.append(np.mean([
                np.mean(np.abs(np.asarray(frame[plane]) - planes[plane])[edge])
                for plane in (1, 2)
            ]))
        return float(np.mean(errors))

    for kernel in ("spline36", "lanczos3", "binomial"):
        baseline = average_edge_mae(core.lgcr.Downsample(
            clip, kernel=kernel, strength=0))
        quality_errors = [average_edge_mae(core.lgcr.Downsample(
            clip, kernel=kernel, quality=q)) for q in (0, 1, 2)]
        assert all(error < baseline for error in quality_errors), (
            f"guided quality did not beat {kernel} baseline: "
            f"{quality_errors} vs {baseline}")
        assert quality_errors[1] <= quality_errors[0]
        assert quality_errors[2] <= quality_errors[1]
        print(f"Downsample {kernel} average edge MAE: base={baseline:.7f}, "
              f"q0={quality_errors[0]:.7f}, q1={quality_errors[1]:.7f}, "
              f"q2={quality_errors[2]:.7f}")


def test_boundaries_and_invalid_parameters():
    for width, height in ((2, 2), (4, 2), (2, 6), (6, 4), (18, 14)):
        clip = core.std.BlankClip(width=width, height=height, format=F444, length=1)
        for quality in (0, 1, 2):
            frame = core.lgcr.Downsample(clip, quality=quality).get_frame(0)
            for plane in range(3):
                assert np.isfinite(np.asarray(frame[plane])).all()

    odd = core.std.BlankClip(width=17, height=14, format=F444, length=1)
    f420 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 1, 1)
    subsampled = core.std.BlankClip(width=16, height=16, format=f420, length=1)
    rgb = core.std.BlankClip(width=16, height=16, format=vs.RGBS, length=1)
    valid = core.std.BlankClip(width=16, height=16, format=F444, length=1)
    expect_error(lambda: core.lgcr.Downsample(odd), "odd dimensions")
    expect_error(lambda: core.lgcr.Downsample(subsampled), "YUV420 input")
    expect_error(lambda: core.lgcr.Downsample(rgb), "RGB input")
    for quality in (-1, 3):
        expect_error(lambda quality=quality: core.lgcr.Downsample(
            valid, quality=quality), f"quality={quality}")
    for strength in (-0.01, 1.01, float("nan")):
        expect_error(lambda strength=strength: core.lgcr.Downsample(
            valid, strength=strength), f"strength={strength}")
    expect_error(lambda: core.lgcr.Downsample(valid, kernel="lanczos"), "kernel")
    expect_error(lambda: core.lgcr.Downsample(valid, loc="middle"), "loc")
    print("Downsample boundary sizes and invalid parameters: OK")


def test_avx2_scalar_consistency():
    scalar_path = os.environ.get(
        "LGCR_PLUGIN_SCALAR", os.path.join(ROOT, "liblgcr_scalar.so"))
    core.std.LoadPlugin(scalar_path)
    height, width = 34, 46
    yy, xx = np.mgrid[:height, :width]
    side = xx * height > yy * width
    clip = make_float_clip((
        (0.2 + 0.5 * side + 0.01 * np.sin(xx)).astype(np.float32),
        (-0.3 + 0.55 * side).astype(np.float32),
        (0.25 - 0.5 * side).astype(np.float32),
    ))
    maximum_error = 0.0
    for kernel in ("spline36", "lanczos3", "binomial"):
        for quality in (0, 1, 2):
            native = core.lgcr.Downsample(
                clip, kernel=kernel, quality=quality).get_frame(0)
            scalar = core.lgcr_scalar.Downsample(
                clip, kernel=kernel, quality=quality).get_frame(0)
            for plane in range(3):
                maximum_error = max(maximum_error, float(np.max(np.abs(
                    np.asarray(native[plane]) - np.asarray(scalar[plane])))))
    assert maximum_error <= 2e-6, f"AVX2/scalar mismatch: {maximum_error}"
    print(f"Downsample AVX2/scalar max error: {maximum_error:.3e}")


test_api_location_props_and_plane_sharing()
test_constant_and_sample_types()
test_strength_baseline_and_continuity()
test_numpy_reference()
test_impulse_siting()
test_hull_and_fallbacks()
test_quality_ordering()
test_boundaries_and_invalid_parameters()
test_avx2_scalar_consistency()
print("OK")
