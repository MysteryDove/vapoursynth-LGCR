#!/usr/bin/env python3
"""Safety and correctness regressions for Recon and TRecon."""
import os
import sys

import numpy as np
import vapoursynth as vs

core = vs.core
core.num_threads = 2
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
core.std.LoadPlugin(os.environ.get("LGCR_PLUGIN", os.path.join(ROOT, "liblgcr.so")))

F444 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 0, 0)
F420 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 1, 1)


def arrays(frame):
    return [np.asarray(frame[p]).copy() for p in range(3)]


def assert_finite(frame, label):
    for p in range(3):
        assert np.isfinite(np.asarray(frame[p])).all(), f"{label}: non-finite plane {p}"


def test_degenerate_axes():
    for width, height in ((2, 2), (2, 64), (64, 2)):
        clip = core.std.BlankClip(width=width, height=height, format=F420, length=2,
                                  color=[0.4, 0.1, -0.1])
        assert_finite(core.lgcr.Recon(clip).get_frame(0), f"Recon {width}x{height}")
        assert_finite(core.lgcr.TRecon(clip).get_frame(0), f"TRecon {width}x{height}")
    print("degenerate 420 axes: OK")


def test_invalid_taps():
    clip = core.std.BlankClip(width=16, height=16, format=F420, length=1)
    for kernel in ("lanczos", "jinc"):
        for taps in (-1, 0, 65):
            try:
                core.lgcr.Recon(clip, kernel=kernel, taps=taps).get_frame(0)
            except vs.Error as exc:
                assert "1..64" in str(exc), str(exc)
            else:
                raise AssertionError(f"{kernel} taps={taps} was accepted")
    assert_finite(core.lgcr.Recon(clip, kernel="lanczos", taps=3).get_frame(0),
                  "host after invalid taps")
    print("invalid taps are recoverable: OK")


def make_420(width, height, length, fill):
    blank = core.std.BlankClip(width=width, height=height, format=F420, length=length)

    def modify(n, f):
        f = f.copy()
        fill(n, f)
        return f

    return blank.std.ModifyFrame(blank, modify)


def test_temporal_chroma_gate():
    width = height = 64
    yy, xx = np.mgrid[0:height, 0:width]
    y = (0.35 + 0.08 * np.sin(xx * 0.31) * np.cos(yy * 0.27)).astype(np.float32)

    def fill(n, frame):
        np.copyto(np.asarray(frame[0]), y)
        np.asarray(frame[1])[:] = 0.0 if n == 0 else 0.2
        np.asarray(frame[2])[:] = 0.0 if n == 0 else -0.2

    clip = make_420(width, height, 2, fill)
    single = core.lgcr.Recon(clip, strength=0.8, algo=2, sparse=0).get_frame(0)
    temporal = core.lgcr.TRecon(clip, strength=0.8, trad=3).get_frame(0)
    for p in range(3):
        assert np.array_equal(np.asarray(single[p]), np.asarray(temporal[p])), (
            f"chroma-inconsistent neighbor affected plane {p}")
    print("temporal block chroma gate: OK")


def test_boundary_deduplication():
    rng = np.random.default_rng(19)
    width = height = 64
    y = rng.random((height, width), dtype=np.float32)
    u = rng.uniform(-0.3, 0.3, (height // 2, width // 2)).astype(np.float32)
    v = rng.uniform(-0.3, 0.3, (height // 2, width // 2)).astype(np.float32)

    def fill(n, frame):
        np.copyto(np.asarray(frame[0]), y)
        np.copyto(np.asarray(frame[1]), u)
        np.copyto(np.asarray(frame[2]), v)

    clip = make_420(width, height, 2, fill)
    for n in (0, 1):
        reference = arrays(core.lgcr.TRecon(clip, trad=1).get_frame(n))
        for trad in (2, 3):
            candidate = core.lgcr.TRecon(clip, trad=trad).get_frame(n)
            for p in range(3):
                assert np.array_equal(reference[p], np.asarray(candidate[p])), (
                    f"boundary frame {n} changed with trad={trad}, plane {p}")
    print("temporal boundary deduplication: OK")


def test_444_identity():
    rng = np.random.default_rng(23)
    width, height = 48, 32
    expected = [rng.uniform(-0.4 if p else 0.0, 0.4 if p else 1.0,
                            (height, width)).astype(np.float32) for p in range(3)]
    blank = core.std.BlankClip(width=width, height=height, format=F444, length=1)

    def fill(n, f):
        f = f.copy()
        for p in range(3):
            np.copyto(np.asarray(f[p]), expected[p])
        return f

    clip = blank.std.ModifyFrame(blank, fill)
    output = core.lgcr.TRecon(clip, strength=0).get_frame(0)
    for p in range(3):
        assert np.array_equal(expected[p], np.asarray(output[p])), f"444 identity plane {p}"
    print("TRecon 444 strength=0 identity: OK")


test_degenerate_axes()
test_invalid_taps()
if "--asan" not in sys.argv:
    test_temporal_chroma_gate()
    test_boundary_deduplication()
    test_444_identity()
print("OK")
