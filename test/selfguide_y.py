#!/usr/bin/env python3
"""Y self-guidance ablation: treat the LUMA plane as if it were subsampled.

Chain:  Y0 (full res) -> 2x downsample -> Y1
        Y1 lives in the chroma planes of a YUV420 clip whose Y plane is Y0,
        so LGCR's luma guide IS the data being reconstructed (perfect
        luma/data correlation — the theoretical upper bound of the method).

Outputs (grayscale Y + RGB composite + zoomed comparison strips):
  out_yself_guided.png / out_yself_rgb.png
  yself_head.png / yself_coat.png   (left=orig Y, mid=plain jinc3, right=self-guided)
"""
import numpy as np
from PIL import Image

import vapoursynth as vs

core = vs.core
core.num_threads = 8
core.std.LoadPlugin("/home/owen/dev/bsflab/liblgcr.so")

arr = np.asarray(Image.open("/home/owen/下载/src.png").convert("RGB")).astype(np.float64)
r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
y0f = 0.299 * r + 0.587 * g + 0.114 * b
cb = np.clip((b - y0f) * 0.564 + 128, 0, 255).astype(np.uint8)
cr = np.clip((r - y0f) * 0.713 + 128, 0, 255).astype(np.uint8)
Y0 = np.clip(y0f, 0, 255).astype(np.uint8)
H, W = Y0.shape
HW, HH = W // 2, H // 2

blank444 = core.std.BlankClip(width=W, height=H, format=vs.YUV444P8, length=1)


def make_clip(yp, up, vp):
    def fill(n, f):
        f = f.copy()
        np.copyto(np.asarray(f[0]), yp)
        np.copyto(np.asarray(f[1]), up)
        np.copyto(np.asarray(f[2]), vp)
        return f
    return blank444.std.ModifyFrame(blank444, fill)


# Y1: 2x downsampled luma (the "subsampled luma" to reconstruct from)
neutral = np.full((H, W), 128, np.uint8)
y1clip = make_clip(Y0, neutral, neutral).resize.Spline36(width=HW, height=HH)
Y1 = np.asarray(y1clip.get_frame(0)[0])  # (HH, HW)

# YUV420 clip: Y = Y0 (guide), U = V = Y1 (data)
blank420 = core.std.BlankClip(width=W, height=H, format=vs.YUV420P8, length=1)


def fill420(n, f):
    f = f.copy()
    np.copyto(np.asarray(f[0]), Y0)
    np.copyto(np.asarray(f[1]), Y1)
    np.copyto(np.asarray(f[2]), Y1)
    return f


src420 = blank420.std.ModifyFrame(blank420, fill420)

guided = core.lgcr.Recon(src420, kernel="jinc", taps=3, strength=0.8).get_frame(0)
plain = core.lgcr.Recon(src420, kernel="jinc", taps=3, strength=0.0).get_frame(0)

Yg = np.asarray(guided[1])  # reconstructed luma is in the chroma planes
Yp = np.asarray(plain[1])
Y0f = Y0.astype(np.float64)

mae_g = np.abs(Yg.astype(np.float64) - Y0f).mean()
mae_p = np.abs(Yp.astype(np.float64) - Y0f).mean()
print(f"luma reconstruction MAE vs original:  plain={mae_p:.3f}  self-guided={mae_g:.3f}")

# edge-band metric (3x3 luma range > 8)
ya = Y0f
mx = np.maximum(np.maximum(ya, np.roll(ya, 1, 0)), np.roll(ya, 1, 1))
mx = np.maximum(mx, np.maximum(np.roll(ya, -1, 0), np.roll(ya, -1, 1)))
mn = np.minimum(np.minimum(ya, np.roll(ya, 1, 0)), np.roll(ya, 1, 1))
mn = np.minimum(mn, np.minimum(np.roll(ya, -1, 0), np.roll(ya, -1, 1)))
em = (mx - mn) > 8
print(f"  edge band:   plain={np.abs(Yp.astype(np.float64)-Y0f)[em].mean():.3f}  "
      f"self-guided={np.abs(Yg.astype(np.float64)-Y0f)[em].mean():.3f}")
print(f"  smooth band: plain={np.abs(Yp.astype(np.float64)-Y0f)[~em].mean():.3f}  "
      f"self-guided={np.abs(Yg.astype(np.float64)-Y0f)[~em].mean():.3f}")

# round-trip: downsample the reconstruction, compare against Y1
rt = make_clip(Yg, neutral, neutral).resize.Spline36(width=HW, height=HH).get_frame(0)
rt_plain = make_clip(Yp, neutral, neutral).resize.Spline36(width=HW, height=HH).get_frame(0)
Y1f = Y1.astype(np.float64)
print(f"round-trip vs Y1: plain={np.abs(np.asarray(rt_plain[0]).astype(np.float64)-Y1f).mean():.3f}  "
      f"self-guided={np.abs(np.asarray(rt[0]).astype(np.float64)-Y1f).mean():.3f}")

# ---- outputs ----
Image.fromarray(Yg, "L").save("out_yself_guided.png")

rgb = np.clip(np.stack([Yg.astype(np.float64) + 1.403 * (cr.astype(np.float64) - 128),
                        Yg.astype(np.float64) - 0.714 * (cr.astype(np.float64) - 128)
                        - 0.344 * (cb.astype(np.float64) - 128),
                        Yg.astype(np.float64) + 1.773 * (cb.astype(np.float64) - 128)],
                       axis=-1), 0, 255).astype(np.uint8)
Image.fromarray(rgb, "RGB").save("out_yself_rgb.png")

s, zoom = 260, 4
for tag, (x0, y0) in {"head": (960, 540), "coat": (1000, 700)}.items():
    strip = Image.new("L", (s * zoom * 3 + 16, s * zoom), 32)
    for k, plane in enumerate((Y0, Yp, Yg)):
        c = Image.fromarray(plane[y0:y0 + s, x0:x0 + s], "L").resize((s * zoom, s * zoom), Image.NEAREST)
        strip.paste(c, (k * (s * zoom + 8), 0))
    strip.save(f"yself_{tag}.png")
    print(f"yself_{tag}.png  (left=original Y, mid=plain jinc3, right=self-guided)")
