#!/usr/bin/env python3
"""Compare the checked-in candidate against the independent baseline build.

The matrix is deliberately small enough for ``make check`` while exercising
the format, kernel, resize, odd-size, and sparse/dense branches that are most
likely to diverge when a CPU stage is changed.
"""
import os

import numpy as np
import vapoursynth as vs


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
core = vs.core
core.num_threads = 1
core.std.LoadPlugin(os.environ.get("LGCR_PLUGIN", os.path.join(ROOT, "liblgcr.so")))
baseline_path = os.environ.get("LGCR_PLUGIN_BASELINE", os.path.join(ROOT, "liblgcr_baseline.so"))
if not os.path.exists(baseline_path):
    raise SystemExit("baseline plugin is missing; run 'make baseline-plugin' first")
core.std.LoadPlugin(baseline_path)
candidate = getattr(core, os.environ.get("LGCR_CANDIDATE_NAMESPACE", "lgcr"))
baseline = getattr(core, os.environ.get("LGCR_BASELINE_NAMESPACE", "lgcr_baseline"))

W, H = 35, 27
YY, XX = np.mgrid[0:H, 0:W]
edge = (XX >= 17).astype(np.float32)
diagonal = (XX * H > YY * W).astype(np.float32)
texture = (0.02 * np.sin(XX * 0.37) * np.cos(YY * 0.29)).astype(np.float32)
planes = (
    (0.17 + 0.31 * edge + 0.04 * diagonal + texture).astype(np.float32),
    (-0.23 + 0.52 * edge + 0.3 * texture).astype(np.float32),
    (0.29 - 0.57 * edge - 0.3 * texture).astype(np.float32),
)


def make_source(fmt, width, height):
    fmt444 = core.query_video_format(vs.YUV, fmt.sample_type,
                                     fmt.bits_per_sample, 0, 0)
    blank = core.std.BlankClip(width=width, height=height, format=fmt444, length=1)
    scale = float((1 << fmt.bits_per_sample) - 1)

    def fill(n, f):
        result = f.copy()
        for plane, values in enumerate(planes):
            values = values[:height, :width]
            target = np.asarray(result[plane])
            if fmt.sample_type == vs.FLOAT:
                np.copyto(target, values)
            else:
                np.copyto(target, np.rint(np.clip(values, 0.0, 1.0) * scale).astype(target.dtype))
        return result

    source444 = blank.std.ModifyFrame(blank, fill)
    return source444 if fmt.subsampling_w == 0 and fmt.subsampling_h == 0 else source444.resize.Bilinear(format=fmt)


def max_diff(left, right):
    scale = 1.0 if left.format.sample_type == vs.FLOAT else float(
        (1 << left.format.bits_per_sample) - 1)
    return max(float(np.max(np.abs(np.asarray(left[p], dtype=np.float64) -
                                  np.asarray(right[p], dtype=np.float64))) / scale)
               for p in range(3))


def max_raw_diff(left, right):
    return max(float(np.max(np.abs(np.asarray(left[p], dtype=np.int64) -
                                   np.asarray(right[p], dtype=np.int64))))
               for p in range(3))


FORMATS = (
    (vs.FLOAT, 32, 0, 0, "444-float", 35, 27),
    (vs.FLOAT, 32, 1, 0, "422-float", 34, 27),
    (vs.FLOAT, 32, 1, 1, "420-float", 34, 26),
    (vs.INTEGER, 8, 0, 0, "444-8", 35, 27),
    (vs.INTEGER, 10, 1, 0, "422-10", 34, 27),
    (vs.INTEGER, 12, 1, 1, "420-12", 34, 26),
    (vs.INTEGER, 16, 1, 1, "420-16", 34, 26),
)
KERNELS = (
    ("bilinear", {}),
    ("bicubic", {}),
    ("spline16", {}),
    ("spline36", {}),
    ("lanczos3", {"kernel": "lanczos", "taps": 3}),
    ("lanczos4", {"kernel": "lanczos", "taps": 4}),
    ("jinc3", {"kernel": "jinc", "taps": 3}),
)

cases = 0
worst_float = (0.0, None)
worst_integer = (0.0, None)
for sample_type, bits, sw, sh, label, width, height in FORMATS:
    fmt = core.query_video_format(vs.YUV, sample_type, bits, sw, sh)
    source = make_source(fmt, width, height)
    for kernel_name, kernel_options in KERNELS:
        options = dict(kernel_options)
        options.setdefault("kernel", kernel_name)
        for algo in (2, 4, 6):
            for sparse in (0, 1):
                resize = (width - 4, height - (2 if sh else 0))
                for out_size in ((width, height), resize):
                    kwargs = dict(options, width=out_size[0], height=out_size[1],
                                   algo=algo, sparse=sparse, strength=0.8)
                    candidate_frame = candidate.Recon(source, **kwargs).get_frame(0)
                    reference = baseline.Recon(source, **kwargs).get_frame(0)
                    error = (max_diff(candidate_frame, reference)
                             if sample_type == vs.FLOAT
                             else max_raw_diff(candidate_frame, reference))
                    cases += 1
                    domain_worst = worst_float if sample_type == vs.FLOAT else worst_integer
                    if error > domain_worst[0]:
                        domain_worst = (error, (label, kernel_name, algo, sparse, out_size))
                        if sample_type == vs.FLOAT:
                            worst_float = domain_worst
                        else:
                            worst_integer = domain_worst
                    limit = 1e-5 if sample_type == vs.FLOAT else 0.0
                    if error > limit:
                        raise AssertionError(
                            f"baseline mismatch {label}/{kernel_name}/algo={algo}/"
                            f"sparse={sparse}/{out_size}: {error}")

# Collaborative filtering moves and reallocates its output Plane. It must keep
# the owning scratch fallback even when all other float outputs write directly.
for sample_type, bits, sw, sh, label, width, height in (
        FORMATS[0], FORMATS[-1]):
    fmt = core.query_video_format(vs.YUV, sample_type, bits, sw, sh)
    source = make_source(fmt, width, height)
    for algo in (2, 6):
        kwargs = dict(kernel="lanczos", taps=3, width=width, height=height,
                      algo=algo, sparse=1, strength=0.8, bm=1)
        candidate_frame = candidate.Recon(source, **kwargs).get_frame(0)
        reference = baseline.Recon(source, **kwargs).get_frame(0)
        error = (max_diff(candidate_frame, reference)
                 if sample_type == vs.FLOAT
                 else max_raw_diff(candidate_frame, reference))
        cases += 1
        limit = 1e-5 if sample_type == vs.FLOAT else 0.0
        if error > limit:
            raise AssertionError(f"baseline mismatch {label}/algo={algo}/bm=1: {error}")

print(f"baseline numerical matrix: {cases} cases, "
      f"float max={worst_float[0]:.3e} at {worst_float[1]}, "
      f"integer max={worst_integer[0]:.3e} at {worst_integer[1]}")
