#!/usr/bin/env python3
"""Regression coverage for the optional collaborative chroma stage."""

import os

import numpy as np
import vapoursynth as vs


core = vs.core
core.num_threads = 2
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
core.std.LoadPlugin(os.environ.get("LGCR_PLUGIN", os.path.join(ROOT, "liblgcr.so")))

F444 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 0, 0)


def arrays(frame):
    return [np.asarray(frame[p]).copy() for p in range(3)]


def make_noisy_clip(width=64, height=64, length=3):
    rng = np.random.default_rng(107)
    noise_u = rng.normal(0.0, 0.006, (height, width)).astype(np.float32)
    noise_v = rng.normal(0.0, 0.006, (height, width)).astype(np.float32)
    blank = core.std.BlankClip(width=width, height=height, format=F444, length=length)

    def fill(n, f):
        f = f.copy()
        np.asarray(f[0])[:] = 0.45
        np.copyto(np.asarray(f[1]), 0.08 + noise_u)
        np.copyto(np.asarray(f[2]), -0.06 + noise_v)
        return f

    return blank.std.ModifyFrame(blank, fill)


def chroma_variance(frame):
    return float(np.mean([np.var(np.asarray(frame[p]).astype(np.float64))
                          for p in (1, 2)]))


def test_default_off_and_denoising():
    clip = make_noisy_clip()
    implicit = arrays(core.lgcr.Recon(clip, strength=0).get_frame(1))
    disabled = arrays(core.lgcr.Recon(clip, strength=0, bm=False).get_frame(1))
    enabled_frame = core.lgcr.Recon(clip, strength=0, bm=True).get_frame(1)

    for p in range(3):
        assert np.array_equal(implicit[p], disabled[p]), (
            f"Recon default changed with explicit bm=False on plane {p}")
        assert np.isfinite(np.asarray(enabled_frame[p])).all(), (
            f"Recon bm=True produced non-finite plane {p}")
    assert np.array_equal(implicit[0], np.asarray(enabled_frame[0])), (
        "collaborative chroma stage modified luma")

    before = float(np.mean([np.var(disabled[p].astype(np.float64)) for p in (1, 2)]))
    after = chroma_variance(enabled_frame)
    assert after < 0.85 * before, f"BM stage did not reduce chroma noise: {before} -> {after}"
    print(f"Recon BM noise variance: {before:.8f} -> {after:.8f}")


def test_constant_preservation():
    clip = core.std.BlankClip(width=40, height=32, format=F444, length=1,
                              color=[0.4, 0.1, -0.1])
    disabled = arrays(core.lgcr.Recon(clip, strength=0, bm=False).get_frame(0))
    enabled = arrays(core.lgcr.Recon(clip, strength=0, bm=True).get_frame(0))
    error = max(float(np.max(np.abs(disabled[p] - enabled[p]))) for p in (1, 2))
    assert error <= 1e-6, f"BM stage shifted constant chroma by {error}"
    print("BM constant preservation: OK")


def test_trecon_switch():
    clip = make_noisy_clip()
    implicit = arrays(core.lgcr.TRecon(clip, strength=0, trad=1).get_frame(1))
    disabled = arrays(core.lgcr.TRecon(clip, strength=0, trad=1, bm=False).get_frame(1))
    enabled_frame = core.lgcr.TRecon(clip, strength=0, trad=1, bm=True).get_frame(1)

    for p in range(3):
        assert np.array_equal(implicit[p], disabled[p]), (
            f"TRecon default changed with explicit bm=False on plane {p}")
        assert np.isfinite(np.asarray(enabled_frame[p])).all(), (
            f"TRecon bm=True produced non-finite plane {p}")
    assert chroma_variance(enabled_frame) < 0.85 * float(np.mean([
        np.var(disabled[p].astype(np.float64)) for p in (1, 2)
    ])), "TRecon bm=True did not activate collaborative filtering"
    print("TRecon BM switch: OK")


def test_edge_preservation():
    width = height = 64
    yy, xx = np.mgrid[0:height, 0:width]
    side = xx >= width // 2
    expected = (
        np.where(side, 0.58, 0.24).astype(np.float32),
        np.where(side, 0.22, -0.18).astype(np.float32),
        np.where(side, -0.20, 0.16).astype(np.float32),
    )
    blank = core.std.BlankClip(width=width, height=height, format=F444, length=1)

    def fill(n, f):
        f = f.copy()
        for p in range(3):
            np.copyto(np.asarray(f[p]), expected[p])
        return f

    gt = blank.std.ModifyFrame(blank, fill)
    f420 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 1, 1)
    source = gt.resize.Bilinear(format=f420)
    disabled = core.lgcr.Recon(source, algo=2, sparse=0, bm=False).get_frame(0)
    enabled = core.lgcr.Recon(source, algo=2, sparse=0, bm=True).get_frame(0)
    band = np.abs(xx - width // 2) <= 4

    def edge_mae(frame):
        return float(np.mean([
            np.abs(np.asarray(frame[p]).astype(np.float64) - expected[p])[band].mean()
            for p in (1, 2)
        ]))

    before = edge_mae(disabled)
    after = edge_mae(enabled)
    assert after <= before + 1e-5, f"BM stage regressed a clean co-edge: {before} -> {after}"
    print(f"BM co-edge MAE: {before:.8f} -> {after:.8f}")


def test_scaled_and_integer_paths():
    clip = make_noisy_clip(width=48, height=32, length=1)
    scaled = core.lgcr.Recon(clip, width=80, height=56, bm=True).get_frame(0)
    assert scaled.width == 80 and scaled.height == 56, "BM resize dimensions changed"
    for p in range(3):
        assert np.isfinite(np.asarray(scaled[p])).all(), (
            f"scaled BM output produced non-finite plane {p}")

    f420_8 = core.query_video_format(vs.YUV, vs.INTEGER, 8, 1, 1)
    integer = core.std.BlankClip(width=48, height=32, format=f420_8, length=3,
                                 color=[64, 128, 128])
    for name, node in (
            ("Recon", core.lgcr.Recon(integer, bm=True)),
            ("TRecon", core.lgcr.TRecon(integer, bm=True))):
        frame = node.get_frame(1)
        values = [int(np.asarray(frame[p])[16, 24]) for p in range(3)]
        assert all(abs(value - expected) <= 1
                   for value, expected in zip(values, (64, 128, 128))), (
            f"{name} integer BM path changed a constant: {values}")
    print("BM scaled and integer paths: OK")


test_default_off_and_denoising()
test_constant_preservation()
test_trecon_switch()
test_edge_preservation()
test_scaled_and_integer_paths()
print("OK")
