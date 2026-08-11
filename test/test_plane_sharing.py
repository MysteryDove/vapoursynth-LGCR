#!/usr/bin/env python3
"""Verify that same-size Recon retains the source luma allocation."""
import os

import numpy as np
import vapoursynth as vs


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LGCR_PROFILE", "1")
core = vs.core
core.num_threads = 1
core.std.LoadPlugin(os.environ.get("LGCR_PLUGIN", os.path.join(ROOT, "liblgcr.so")))


def make_source(sample_type, bits, subsampling_w, subsampling_h, width, height):
    fmt = core.query_video_format(vs.YUV, sample_type, bits,
                                  subsampling_w, subsampling_h)
    source = core.std.BlankClip(width=width, height=height, format=fmt,
                                length=1, keep=True)

    def fill(n, f):
        result = f.copy()
        for plane in range(3):
            values = np.asarray(result[plane])
            yy, xx = np.indices(values.shape)
            if sample_type == vs.FLOAT:
                values[:] = (0.1 * plane + xx * 0.003 + yy * 0.005).astype(np.float32)
            else:
                peak = (1 << bits) - 1
                values[:] = ((xx * 17 + yy * 29 + plane * 101) % peak).astype(values.dtype)
        return result

    return source.std.ModifyFrame(source, fill)


formats = (
    (vs.INTEGER, 8, 1, 1, 34, 26, "420-8"),
    (vs.INTEGER, 10, 1, 1, 34, 26, "420-10"),
    (vs.INTEGER, 12, 1, 0, 34, 27, "422-12"),
    (vs.INTEGER, 16, 0, 0, 35, 27, "444-16"),
    (vs.FLOAT, 32, 1, 1, 34, 26, "420-float"),
    (vs.FLOAT, 32, 1, 0, 34, 27, "422-float"),
    (vs.FLOAT, 32, 0, 0, 35, 27, "444-float"),
)

cases = 0
for sample_type, bits, sw, sh, width, height, label in formats:
    source = make_source(sample_type, bits, sw, sh, width, height)
    source_frame = source.get_frame(0)
    source_y = np.asarray(source_frame[0])
    for algo in (2, 4, 6):
        result = core.lgcr.Recon(source, width=width, height=height,
                                 algo=algo, sparse=1).get_frame(0)
        result_y = np.asarray(result[0])
        if result_y.__array_interface__["data"][0] != source_y.__array_interface__["data"][0]:
            raise AssertionError(f"luma plane not shared: {label}/algo={algo}")
        if result_y.tobytes() != source_y.tobytes():
            raise AssertionError(f"luma plane changed: {label}/algo={algo}")
        if "_ChromaLocation" in result.props:
            raise AssertionError(f"stale chroma location: {label}/algo={algo}")
        expected_input = (0 if sample_type == vs.FLOAT else
                          width * height +
                          2 * ((width + (1 << sw) - 1) >> sw) *
                          ((height + (1 << sh) - 1) >> sh))
        expected_output = 0 if sample_type == vs.FLOAT else 2 * width * height
        if result.props["_LGCR_input_conversion_pixels"] != expected_input:
            raise AssertionError(f"input work mismatch: {label}/algo={algo}")
        if result.props["_LGCR_output_conversion_pixels"] != expected_output:
            raise AssertionError(f"output work mismatch: {label}/algo={algo}")
        if result.props["_LGCR_luma_resample_pixels"] != 0:
            raise AssertionError(f"same-size luma resample counted: {label}/algo={algo}")
        cases += 1

print(f"same-size luma sharing: {cases} cases")
