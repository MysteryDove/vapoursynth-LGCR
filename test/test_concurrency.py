#!/usr/bin/env python3
"""Determinism smoke test for independent Recon frame requests."""
import hashlib
import os

import numpy as np
import vapoursynth as vs


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
core = vs.core
core.std.LoadPlugin(os.environ.get("LGCR_PLUGIN", os.path.join(ROOT, "liblgcr.so")))

F444 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 0, 0)
WIDTH, HEIGHT = 192, 108
yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
planes = (
    (0.15 + 0.35 * (xx > WIDTH // 2) + 0.01 * np.sin(yy * 0.2)).astype(np.float32),
    (-0.25 + 0.55 * (xx > WIDTH // 2)).astype(np.float32),
    (0.30 - 0.60 * (xx > WIDTH // 2)).astype(np.float32),
)
blank = core.std.BlankClip(width=WIDTH, height=HEIGHT, format=F444, length=1)


def fill(n, f):
    frame = f.copy()
    for plane in range(3):
        np.copyto(np.asarray(frame[plane]), planes[plane])
    return frame


source = blank.std.ModifyFrame([blank], fill).resize.Bilinear(format=
    core.query_video_format(vs.YUV, vs.FLOAT, 32, 1, 1))
source = source.std.Loop()


def digest(frame):
    h = hashlib.sha256()
    for plane in range(3):
        h.update(np.asarray(frame[plane]).tobytes())
    return h.digest()


for threads in (1, 8, 16):
    core.num_threads = threads
    for algo in (2, 4, 6):
        node = core.lgcr.Recon(source, kernel="lanczos", taps=3,
                                strength=0.8, algo=algo, sparse=1)
        futures = [node.get_frame_async(n) for n in range(24)]
        frames = [future.result() for future in futures]
        checksums = [digest(frame) for frame in frames]
        assert len(set(checksums)) == 1, (threads, algo)
        # A second request batch must be byte-identical even after the first
        # batch has populated geometry and frame caches.
        assert checksums == [digest(node.get_frame(n)) for n in range(24)]
print("concurrent Recon determinism: OK")
