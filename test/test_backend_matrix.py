#!/usr/bin/env python3
"""Geometry, kernel specialization, and CPU backend matrix regressions."""
import os

import numpy as np
import vapoursynth as vs


core = vs.core
core.num_threads = 1
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
core.std.LoadPlugin(os.environ.get("LGCR_PLUGIN", os.path.join(ROOT, "liblgcr.so")))
core.std.LoadPlugin(os.environ.get(
    "LGCR_PLUGIN_SCALAR", os.path.join(ROOT, "liblgcr_scalar.so")))

F444 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 0, 0)
F422 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 1, 0)
F420 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 1, 1)
KERNELS = (
    ("bilinear", {}),
    ("bicubic", {}),
    ("spline16", {}),
    ("spline36", {}),
    ("lanczos3", {"kernel": "lanczos", "taps": 3}),
    ("lanczos4", {"kernel": "lanczos", "taps": 4}),
    ("jinc3", {"kernel": "jinc", "taps": 3}),
)


def kernel_args(name, options):
    result = dict(options)
    result.setdefault("kernel", name)
    return result


def assert_finite(frame, label):
    for plane in range(3):
        assert np.isfinite(np.asarray(frame[plane])).all(), f"{label}: plane {plane}"


def max_diff(left, right):
    return max(float(np.max(np.abs(
        np.asarray(left[plane]).astype(np.float64) -
        np.asarray(right[plane]).astype(np.float64)))) for plane in range(3))


def edge_clip(width, height, fmt):
    blank = core.std.BlankClip(width=width, height=height, format=F444, length=1)
    yy, xx = np.mgrid[0:height, 0:width]
    split = (xx >= width // 2).astype(np.float32)
    y = (0.2 + 0.3 * split + 0.01 * np.sin(yy * 0.4)).astype(np.float32)
    u = (-0.25 + 0.5 * split).astype(np.float32)
    v = (0.30 - 0.55 * split).astype(np.float32)

    def fill(n, f):
        result = f.copy()
        np.copyto(np.asarray(result[0]), y)
        np.copyto(np.asarray(result[1]), u)
        np.copyto(np.asarray(result[2]), v)
        return result

    clip = blank.std.ModifyFrame(blank, fill)
    return clip if fmt == F444 else clip.resize.Bilinear(format=fmt)


def test_degenerate_and_boundaries():
    cases = ((1, 1, F444), (1, 17, F444), (17, 1, F444),
             (2, 1, F422), (2, 2, F420), (16, 12, F420))
    for width, height, fmt in cases:
        clip = core.std.BlankClip(width=width, height=height, format=fmt, length=1,
                                  color=[0.4, 0.1, -0.1])
        for name, options in KERNELS:
            args = kernel_args(name, options)
            frame = core.lgcr.Recon(clip, algo=2, sparse=0, **args).get_frame(0)
            assert_finite(frame, f"{width}x{height}/{name}")
    print("degenerate axes and kernel boundaries: OK")


def test_formats_and_siting():
    for fmt, label in ((F420, "420"), (F422, "422"), (F444, "444")):
        source = edge_clip(48, 32, fmt)
        for location in (0, 1, 2, 3, 4, 5):
            located = source.std.SetFrameProps(_ChromaLocation=location)
            frame = core.lgcr.Recon(located, kernel="spline36", algo=4).get_frame(0)
            assert_finite(frame, f"{label}/location={location}")
    print("420/422/444 and H.273 siting: OK")


def test_strength_and_sparse():
    source = edge_clip(64, 48, F420)
    for name, options in KERNELS:
        args = kernel_args(name, options)
        zero = core.lgcr.Recon(source, strength=0.0, sparse=0, **args).get_frame(0)
        tiny = core.lgcr.Recon(source, strength=1e-6, sparse=0, **args).get_frame(0)
        full = core.lgcr.Recon(source, strength=1.0, sparse=0, **args).get_frame(0)
        tiny_step = max_diff(zero, tiny)
        total = max_diff(zero, full)
        assert tiny_step <= total * 3e-6 + 2e-7, (
            f"strength discontinuity for {name}: tiny={tiny_step}, total={total}")

        for algo in (2, 4, 6):
            dense = core.lgcr.Recon(
                source, strength=0.8, algo=algo, sparse=0, **args).get_frame(0)
            sparse = core.lgcr.Recon(
                source, strength=0.8, algo=algo, sparse=1, **args).get_frame(0)
            assert max_diff(dense, sparse) <= 1e-6, (
                f"sparse mismatch for algo={algo}/{name}")
    print("strength continuity and sparse equivalence: OK")


def test_algo4_sparse_ramp():
    width, height = 128, 64
    blank = core.std.BlankClip(width=width, height=height, format=F444, length=1)
    yy, xx = np.mgrid[0:height, 0:width]
    transition = np.clip((xx - 60) / 7.0, 0, 1)
    planes = (
        (0.30 + 0.05 * transition).astype(np.float32),
        (-0.15 + 0.55 * transition).astype(np.float32),
        (0.40 - 0.60 * transition).astype(np.float32),
    )

    def fill(n, f):
        result = f.copy()
        for plane in range(3):
            np.copyto(np.asarray(result[plane]), planes[plane])
        return result

    source = blank.std.ModifyFrame(blank, fill).resize.Bilinear(format=F420)
    dense = core.lgcr.Recon(source, kernel="jinc", taps=3,
                            algo=4, sparse=0).get_frame(0)
    sparse = core.lgcr.Recon(source, kernel="jinc", taps=3,
                             algo=4, sparse=1).get_frame(0)
    assert max_diff(dense, sparse) <= 1e-6, "algo4 sparse ramp mismatch"
    print("algo4 weak-ramp sparse equivalence: OK")


def test_algo6_sparse_affine_roi():
    source = edge_clip(256, 96, F420)
    options = dict(kernel="lanczos", taps=3, algo=6, strength=0.8)
    dense = core.lgcr.Recon(source, sparse=0, **options).get_frame(0)
    sparse = core.lgcr.Recon(source, sparse=1, **options).get_frame(0)
    assert max_diff(dense, sparse) <= 1e-6, "algo6 sparse affine ROI mismatch"
    print("algo6 sparse affine ROI equivalence: OK")


def test_scalar_avx_matrix():
    for fmt, label in ((F420, "420"), (F422, "422"), (F444, "444")):
        source = edge_clip(48, 32, fmt)
        for algo in (4, 6):
            for name, options in (KERNELS[0], KERNELS[4], KERNELS[-1]):
                args = kernel_args(name, options)
                avx = core.lgcr.Recon(source, algo=algo, sparse=0, **args).get_frame(0)
                scalar = core.lgcr_scalar.Recon(
                    source, algo=algo, sparse=0, **args).get_frame(0)
                error = max_diff(avx, scalar)
                assert error <= 1e-5, (
                    f"AVX2/scalar mismatch {label}/{algo}/{name}: {error}")
    print("algo4/6 scalar-vs-AVX2 matrix: OK")


def test_scaled_separable_cache():
    source = edge_clip(48, 32, F420)
    for width, height in ((13, 7), (96, 64), (31, 7)):
        for algo in (2, 6):
            args = dict(width=width, height=height, kernel="lanczos", taps=4,
                        algo=algo, sparse=1, strength=0.8)
            avx = core.lgcr.Recon(source, **args).get_frame(0)
            scalar = core.lgcr_scalar.Recon(source, **args).get_frame(0)
            assert_finite(avx, f"scaled separable {width}x{height}/algo={algo}")
            error = max_diff(avx, scalar)
            assert error <= 1e-5, (
                f"scaled separable AVX2/scalar mismatch "
                f"{width}x{height}/algo={algo}: {error}")
    print("scaled separable row cache: OK")


def test_bilinear_fast_and_fallback():
    cases = (
        (50, 34, F420, ((50, 34), (97, 65), (31, 19))),
        (50, 33, F422, ((50, 33), (97, 63), (31, 17))),
        (17, 13, F444, ((17, 13), (33, 25), (9, 7))),
        (1, 17, F444, ((1, 33), (1, 9))),
        (17, 1, F444, ((33, 1), (9, 1))),
    )
    for width, height, fmt, outputs in cases:
        source = edge_clip(width, height, fmt)
        for location in (0, 1, 2, 3, 4, 5):
            located = source.std.SetFrameProps(_ChromaLocation=location)
            for out_width, out_height in outputs:
                args = dict(width=out_width, height=out_height,
                            kernel="bilinear", strength=0.0, sparse=0)
                avx = core.lgcr.Recon(located, **args).get_frame(0)
                scalar = core.lgcr_scalar.Recon(located, **args).get_frame(0)
                error = max_diff(avx, scalar)
                assert error <= 1e-6, (
                    f"bilinear mismatch {width}x{height}->{out_width}x{out_height} "
                    f"location={location}: {error}")
    print("bilinear 2x fast path and scalar fallbacks: OK")


test_degenerate_and_boundaries()
test_formats_and_siting()
test_strength_and_sparse()
test_algo4_sparse_ramp()
test_algo6_sparse_affine_roi()
test_scalar_avx_matrix()
test_scaled_separable_cache()
test_bilinear_fast_and_fallback()
print("OK")
