#!/usr/bin/env python3
"""Focused correctness regressions for algo=6 and continuous gate controls."""
import os

import numpy as np
import vapoursynth as vs

core = vs.core
core.num_threads = 4

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
core.std.LoadPlugin(os.environ.get("LGCR_PLUGIN", os.path.join(ROOT, "liblgcr.so")))
core.std.LoadPlugin(os.environ.get(
    "LGCR_PLUGIN_SCALAR", os.path.join(ROOT, "liblgcr_scalar.so")))

W, H = 128, 128
F444 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 0, 0)
F420 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 1, 1)
YY, XX = np.mgrid[0:H, 0:W]


def make_clip(y, u, v):
    blank = core.std.BlankClip(width=W, height=H, format=F444, length=1)

    def fill(n, f):
        f = f.copy()
        np.copyto(np.asarray(f[0]), y)
        np.copyto(np.asarray(f[1]), u)
        np.copyto(np.asarray(f[2]), v)
        return f

    return blank.std.ModifyFrame(blank, fill)


def edge_scene(alias="none", soft=0, shift=0):
    ty = np.clip(XX - 64 + 0.5, 0, 1)
    if soft:
        tc = np.clip((XX - (64 + shift) + soft / 2) / soft, 0, 1)
    else:
        tc = np.clip(XX - (64 + shift) + 0.5, 0, 1)
    alt = np.zeros((H, W), np.float64)
    if alias == "x":
        alt = 0.10 * np.where((XX & 1) == 0, 1.0, -1.0)
    elif alias == "y":
        alt = 0.10 * np.where((YY & 1) == 0, 1.0, -1.0)
    elif alias == "xy":
        alt = 0.10 * np.where(((XX + YY) & 1) == 0, 1.0, -1.0)
    y = (0.30 + 0.05 * ty + alt).astype(np.float32)
    u = (-0.15 + 0.55 * tc).astype(np.float32)
    v = (0.40 - 0.60 * tc).astype(np.float32)
    band = (np.abs(XX - (64 + shift)) <= 4) & (YY > 6) & (YY < H - 6)
    return y, u, v, band


def chroma_mae(frame, gt, mask):
    return float(np.mean([
        np.abs(np.asarray(frame[p]).astype(np.float64) - np.asarray(gt[p]))[mask].mean()
        for p in (1, 2)
    ]))


def max_chroma_diff(a, b):
    return float(max(np.max(np.abs(np.asarray(a[p]) - np.asarray(b[p]))) for p in (1, 2)))


def run_alias_regressions():
    for scale in (1, 2):
        results = {}
        for alias in ("none", "x", "y", "xy"):
            y, u, v, band = edge_scene(alias=alias)
            gt = make_clip(y, u, v)
            src = gt.resize.Bilinear(format=F420)
            ow, oh = W * scale, H * scale
            if scale == 1:
                fgt = gt.get_frame(0)
                out_band = band
            else:
                fgt = gt.resize.Spline36(width=ow, height=oh).get_frame(0)
                out_band = np.kron(band, np.ones((2, 2), dtype=bool))
            plain = core.lgcr.Recon(src, width=ow, height=oh, kernel="jinc", taps=3,
                                    strength=0).get_frame(0)
            detail = core.lgcr.Recon(src, width=ow, height=oh, kernel="jinc", taps=3,
                                     strength=0.8, algo=6).get_frame(0)
            results[alias] = (chroma_mae(plain, fgt, out_band),
                              chroma_mae(detail, fgt, out_band))

        clean = results["none"][1]
        for alias, (plain_err, detail_err) in results.items():
            assert detail_err <= plain_err + 1e-6, (
                f"algo6 regressed vs plain at scale={scale}, alias={alias}: "
                f"{detail_err} > {plain_err}")
            assert abs(detail_err - clean) < 0.002, (
                f"axis-Nyquist suppression depends on alias orientation at scale={scale}, "
                f"alias={alias}: {detail_err} vs clean {clean}")
        print(f"alias scale={scale} " + " ".join(
            f"{key}:plain={value[0]:.5f}/a6={value[1]:.5f}"
            for key, value in results.items()))


def run_gate_regressions():
    y, u, v, band = edge_scene(soft=7)
    gt = make_clip(y, u, v)
    src = gt.resize.Bilinear(format=F420)
    fgt = gt.get_frame(0)

    ms_errors = []
    for ms in (0.0, 0.25, 0.5, 0.75, 1.0):
        frame = core.lgcr.Recon(src, kernel="jinc", taps=3, strength=0.8,
                                algo=6, qgate=0, ms=ms).get_frame(0)
        ms_errors.append(chroma_mae(frame, fgt, band))
    assert all(a > b + 1e-5 for a, b in zip(ms_errors, ms_errors[1:])), (
        f"ms must continuously reduce harmful detail transfer: {ms_errors}")
    print("ms interpolation: " + " ".join(f"{v:.7f}" for v in ms_errors))

    q0 = core.lgcr.Recon(src, kernel="jinc", taps=3, strength=0.8,
                         algo=6, qgate=0, ms=0).get_frame(0)
    q1 = core.lgcr.Recon(src, kernel="jinc", taps=3, strength=0.8,
                         algo=6, qgate=1, ms=0).get_frame(0)
    qdiff = max_chroma_diff(q0, q1)
    assert qdiff > 1e-4, f"qgate parameter appears ineffective: max diff {qdiff}"
    print(f"qgate 0-vs-1 max chroma diff: {qdiff:.7f}")

    plain = core.lgcr.Recon(src, kernel="jinc", taps=3, strength=0,
                            algo=6).get_frame(0)
    tiny = core.lgcr.Recon(src, kernel="jinc", taps=3, strength=1e-6,
                           algo=6).get_frame(0)
    for p in range(3):
        assert np.array_equal(np.asarray(plain[p]), np.asarray(tiny[p])), (
            f"strength=1e-6 changed plane {p}")
    print("strength 0-vs-1e-6: bit-identical")

    avx = core.lgcr.Recon(src, kernel="jinc", taps=3, strength=0.8,
                          algo=6, qgate=0.6, ms=0.4).get_frame(0)
    scalar = core.lgcr_scalar.Recon(src, kernel="jinc", taps=3, strength=0.8,
                                    algo=6, qgate=0.6, ms=0.4).get_frame(0)
    max_diff = max(float(np.max(np.abs(np.asarray(avx[p]) - np.asarray(scalar[p]))))
                   for p in range(3))
    assert max_diff <= 1e-5, f"AVX2/scalar algo6 mismatch: {max_diff}"
    print(f"AVX2-vs-scalar algo6 max diff: {max_diff:.2e}")

    for algo in (3, 4):
        errors = []
        frames = []
        for ms in (0.0, 0.5, 1.0):
            frame = core.lgcr.Recon(src, kernel="jinc", taps=3, strength=0.8,
                                    algo=algo, ms=ms).get_frame(0)
            frames.append(frame)
            errors.append(chroma_mae(frame, fgt, band))
        assert max_chroma_diff(frames[0], frames[1]) > 1e-5
        assert max_chroma_diff(frames[1], frames[2]) > 1e-5
        assert errors[0] > errors[1] > errors[2], (
            f"algo{algo} ms must be a continuous gate on the soft-edge case: {errors}")
        print(f"algo{algo} ms interpolation: "
              + " ".join(f"{value:.7f}" for value in errors))


run_alias_regressions()
run_gate_regressions()
print("OK")
