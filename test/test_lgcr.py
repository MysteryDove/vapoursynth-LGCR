#!/usr/bin/env python3
"""Smoke + ablation test for the LGCR VapourSynth plugin.

Builds a synthetic 4:4:4 float clip with sharp color edges (a vertical edge
and a diagonal edge), downsamples to 4:2:0 (this is where chroma bleeding is
introduced), then reconstructs with LGCR at strength=0 (pure kernel) and
strength=0.8 (guided) and measures chroma error vs the ground truth.

Also cross-checks the AVX2 build against the scalar build (must be identical
within float rounding).
"""
import os

import vapoursynth as vs

core = vs.core
core.num_threads = 4

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

W, H = 128, 128
FMT444 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 0, 0)
FMT420 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 1, 1)

# Two strongly saturated colors (float chroma is 0-centered)
C1 = (0.30, -0.15, 0.40)   # red-ish
C2 = (0.25, 0.40, -0.20)   # blue-ish


def make_gt(n, f):
    f = f.copy()
    yp, up, vp = f[0], f[1], f[2]
    for j in range(H):
        for i in range(W):
            # vertical edge at x=64 in bottom half, diagonal edge in top half
            if j < H // 2:
                c = C1 if i < j else C2          # 45-degree edge
            else:
                c = C1 if i < W // 2 else C2     # vertical edge
            yp[j, i] = c[0]
            up[j, i] = c[1]
            vp[j, i] = c[2]
    return f


blank = core.std.BlankClip(width=W, height=H, format=FMT444, length=1, fpsnum=24)
gt = blank.std.ModifyFrame(blank, make_gt)

# Degrade to 4:2:0 (chroma bleed introduced here), as a decoder would see it
src420 = gt.resize.Bilinear(format=FMT420)

core.std.LoadPlugin(os.path.join(ROOT, "liblgcr.so"))
plain = core.lgcr.Recon(src420, kernel="lanczos", taps=3, strength=0.0, ar=-1.0)
guided = core.lgcr.Recon(src420, kernel="lanczos", taps=3, strength=0.8)

fgt = gt.get_frame(0)
fplain = plain.get_frame(0)
fguided = guided.get_frame(0)

assert fplain.width == W and fplain.height == H, "output size mismatch"
assert fplain.format.subsampling_w == 0 and fplain.format.subsampling_h == 0, "output must be 444"


def band_mae(fa, fb, plane, edge_pred):
    a, b = fa[plane], fb[plane]
    tot, cnt = 0.0, 0
    for j in range(H):
        for i in range(W):
            if edge_pred(i, j):
                tot += abs(a[j, i] - b[j, i])
                cnt += 1
    return tot / cnt if cnt else 0.0


def near_vertical_edge(i, j):
    return H // 2 + 4 <= j < H - 4 and abs(i - W // 2) <= 3 and i != W // 2


def near_diagonal_edge(i, j):
    return 4 <= j < H // 2 - 4 and 0 < abs(i - j) <= 2


def flat_region(i, j):
    # well inside color 2, away from both edges
    return j >= H // 2 + 8 and i >= W // 2 + 8


failed = False
for plane, name in ((1, "U"), (2, "V")):
    e_vert_p = band_mae(fplain, fgt, plane, near_vertical_edge)
    e_vert_g = band_mae(fguided, fgt, plane, near_vertical_edge)
    e_diag_p = band_mae(fplain, fgt, plane, near_diagonal_edge)
    e_diag_g = band_mae(fguided, fgt, plane, near_diagonal_edge)
    e_flat_p = band_mae(fplain, fgt, plane, flat_region)
    e_flat_g = band_mae(fguided, fgt, plane, flat_region)
    ok_v = e_vert_g < e_vert_p
    ok_d = e_diag_g < e_diag_p
    failed = failed or not ok_v or not ok_d
    print(f"[{name}] vertical-edge MAE  plain={e_vert_p:.4f}  guided={e_vert_g:.4f}  "
          f"({'BETTER' if ok_v else 'WORSE'})")
    print(f"[{name}] diagonal-edge MAE  plain={e_diag_p:.4f}  guided={e_diag_g:.4f}  "
          f"({'BETTER' if ok_d else 'WORSE'})")
    print(f"[{name}] flat-region MAE    plain={e_flat_p:.4f}  guided={e_flat_g:.4f}")

# Luma must be identical between plain and guided (guidance touches chroma only)
yp, yg = fplain[0], fguided[0]
maxdy = max(abs(yp[j, i] - yg[j, i]) for j in range(H) for i in range(W))
print(f"luma plain-vs-guided max diff: {maxdy:.2e} (should be 0)")
failed = failed or maxdy != 0.0

# Cross-check scalar build vs AVX2 build
core.std.LoadPlugin(os.path.join(ROOT, "liblgcr_scalar.so"))
scalar = core.lgcr_scalar.Recon(src420, kernel="lanczos", taps=3, strength=0.8)
fscalar = scalar.get_frame(0)
maxdc = 0.0
for p in (0, 1, 2):
    a, b = fguided[p], fscalar[p]
    for j in range(H):
        for i in range(W):
            maxdc = max(maxdc, abs(a[j, i] - b[j, i]))
print(f"AVX2-vs-scalar max diff: {maxdc:.2e} (should be < 1e-5)")
failed = failed or maxdc > 1e-5

print("FAILED" if failed else "OK")
raise SystemExit(1 if failed else 0)
