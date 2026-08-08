#!/usr/bin/env python3
"""Run LGCR on a still image: PNG -> (degrade to 4:2:0) -> LGCR reconstruct -> PNG.

Usage:
  python3 recon_image.py IN.png OUT.png [kernel] [strength] [scale]

Default: jinc3, strength=0.8, native size (pure chroma reconstruction).
The 4:2:0 degradation simulates what a decoder hands the renderer; LGCR then
rebuilds 4:4:4 chroma guided by luma. RGB<->YUV is BT.601 full-range, done in
numpy so the roundtrip is self-consistent.
"""
import sys

import numpy as np
from PIL import Image

import vapoursynth as vs

core = vs.core
core.num_threads = 8


def rgb_to_yuv444(arr):
    r = arr[..., 0].astype(np.float64)
    g = arr[..., 1].astype(np.float64)
    b = arr[..., 2].astype(np.float64)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 128.0
    cr = (r - y) * 0.713 + 128.0
    return (np.clip(y, 0, 255).astype(np.uint8),
            np.clip(cb, 0, 255).astype(np.uint8),
            np.clip(cr, 0, 255).astype(np.uint8))


def yuv444_to_rgb(y, cb, cr):
    yy = y.astype(np.float64)
    cb = cb.astype(np.float64) - 128.0
    cr = cr.astype(np.float64) - 128.0
    r = yy + 1.403 * cr
    g = yy - 0.714 * cr - 0.344 * cb
    b = yy + 1.773 * cb
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


def png_to_clip(path):
    arr = np.asarray(Image.open(path).convert("RGB"))
    y, cb, cr = rgb_to_yuv444(arr)
    h, w = y.shape
    blank = core.std.BlankClip(width=w, height=h, format=vs.YUV444P8, length=1, fpsnum=24)

    def fill(n, f):
        f = f.copy()
        np.copyto(np.asarray(f[0]), y)
        np.copyto(np.asarray(f[1]), cb)
        np.copyto(np.asarray(f[2]), cr)
        return f

    return blank.std.ModifyFrame(blank, fill)


def clip_to_png(clip, path):
    f = clip.get_frame(0)
    rgb = yuv444_to_rgb(np.asarray(f[0]), np.asarray(f[1]), np.asarray(f[2]))
    Image.fromarray(rgb, "RGB").save(path)


def main():
    inp, out = sys.argv[1], sys.argv[2]
    kernel = sys.argv[3] if len(sys.argv) > 3 else "jinc"
    strength = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
    scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

    core.std.LoadPlugin("/home/owen/dev/bsflab/liblgcr.so")

    src = png_to_clip(inp)
    # degrade to 4:2:0 (this is where chroma bleed/fringing is introduced)
    src420 = src.resize.Bilinear(format=vs.YUV420P8)

    w = int(round(src.width * scale)) & ~1
    h = int(round(src.height * scale)) & ~1
    kwargs = dict(kernel=kernel, strength=strength)
    if kernel in ("lanczos", "jinc"):
        kwargs["taps"] = 3
    rec = core.lgcr.Recon(src420, width=w, height=h, **kwargs)

    clip_to_png(rec, out)
    print(f"wrote {out}  ({w}x{h}, kernel={kernel}, strength={strength})")


if __name__ == "__main__":
    main()
