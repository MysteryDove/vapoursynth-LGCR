#!/usr/bin/env python3
"""Evaluate retained Recon algorithms on a self-consistent real-image 420 round trip."""
import argparse
import hashlib
import os

import numpy as np
from PIL import Image
import vapoursynth as vs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def rgb_to_yuv444(arr):
    r = arr[..., 0].astype(np.float64)
    g = arr[..., 1].astype(np.float64)
    b = arr[..., 2].astype(np.float64)
    y = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)
    cb = (b - y) * 0.564 + 128.0
    cr = (r - y) * 0.713 + 128.0
    return (y, np.clip(cb, 0, 255).astype(np.uint8),
            np.clip(cr, 0, 255).astype(np.uint8))


def make_clip(core, y, u, v):
    h, w = y.shape
    blank = core.std.BlankClip(width=w, height=h, format=vs.YUV444P8, length=1)

    def fill(n, f):
        f = f.copy()
        np.copyto(np.asarray(f[0]), y)
        np.copyto(np.asarray(f[1]), u)
        np.copyto(np.asarray(f[2]), v)
        return f

    return blank.std.ModifyFrame(blank, fill)


def region_masks(y):
    """Exact gradient thresholds used by the established README regression."""
    yf = y.astype(np.float64)
    gx = np.abs(np.diff(yf, axis=1, append=yf[:, -1:]))
    gy = np.abs(np.diff(yf, axis=0, append=yf[-1:, :]))
    magnitude = gx + gy
    return magnitude > 30, magnitude < 3


def evaluate(input_path):
    core = vs.core
    core.num_threads = 8
    core.std.LoadPlugin(os.environ.get("LGCR_PLUGIN", os.path.join(ROOT, "liblgcr.so")))

    rgb = np.asarray(Image.open(input_path).convert("RGB"))
    y, u, v = rgb_to_yuv444(rgb)
    gt = make_clip(core, y, u, v)
    src420 = gt.resize.Bilinear(format=vs.YUV420P8)
    edge, smooth = region_masks(y)
    masks = {"full": np.ones_like(edge), "edge": edge, "smooth": smooth}
    configs = {"plain": dict(strength=0.0)}
    configs.update({f"algo{algo}": dict(strength=0.8, algo=algo)
                    for algo in (2, 4, 6)})

    rows = {}
    for tag, kwargs in configs.items():
        frame = core.lgcr.Recon(src420, kernel="jinc", taps=3, **kwargs).get_frame(0)
        errors = [np.abs(np.asarray(frame[p]).astype(np.float64) - ref.astype(np.float64))
                  for p, ref in ((1, u), (2, v))]
        rows[tag] = {name: float(np.mean([error[mask].mean() for error in errors]))
                     for name, mask in masks.items()}
    return rgb.shape[1], rgb.shape[0], float(edge.mean()), float(smooth.mean()), rows


def render_report(input_path, width, height, edge_fraction, smooth_fraction, rows):
    with open(input_path, "rb") as source:
        digest = hashlib.sha256(source.read()).hexdigest()
    tags = list(rows)
    lines = [
        f"source: {os.path.basename(input_path)} {width}x{height}",
        f"sha256: {digest}",
        f"edge fraction: {edge_fraction:.6f}",
        f"smooth fraction: {smooth_fraction:.6f}",
        "",
        f"{'region':<10}" + "".join(f"{tag:>10}" for tag in tags),
        "-" * (10 + 10 * len(tags)),
    ]
    for region in ("full", "edge", "smooth"):
        lines.append(f"{region:<10}" + "".join(f"{rows[tag][region]:>10.3f}" for tag in tags))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="reference RGB image")
    args = parser.parse_args()
    width, height, edge_fraction, smooth_fraction, rows = evaluate(args.input)
    report = render_report(args.input, width, height, edge_fraction, smooth_fraction, rows)
    print(report)
    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "real_image_latest.md"), "w") as output:
        output.write("# LGCR real-image evaluation\n\n```text\n")
        output.write(report)
        output.write("\n```\n")


if __name__ == "__main__":
    main()
