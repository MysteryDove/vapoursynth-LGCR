#!/usr/bin/env python3
"""LGCR formal evaluation battery.

Covers the scenarios the v1.x series was found wanting in, plus the classic
wins. Every case: build a 444 ground truth, degrade to 420, reconstruct,
measure chroma MAE in the relevant band.

Cases:
  hard_v / hard_d45  - luma & chroma strictly co-edged hard steps (the win case)
  isoluminant        - chroma edge with NO luma edge (no information for guide)
  misalign4          - chroma edge shifted 4 luma px from the luma edge
  hardL_softC        - hard luma edge, genuinely soft (7px) chroma transition
  ridge_line         - luma ridge (line art) + soft chroma blend
  ramp               - pure linear ramp (luma and chroma)
  texture            - chroma texture, flat luma
  noise              - hard edge + sensor-ish noise (robustness)
  upscale2x          - same hard edge, but 420 -> 2x upscale (scaler mode)
  nullspace_x/y/xy   - targeted axis/checker Nyquist aliases in luma only
  temporal           - two frames, edge moves 1 luma px: aligned error variance

Usage: python3 test/battery.py [--all] [--check] [--write-results]

--all runs algo 1..6 (default: plain+algo2), --check enables the quality gates,
and result files are only updated when --write-results is explicit.
"""
import math
import os
import sys

import numpy as np
import vapoursynth as vs

core = vs.core
core.num_threads = 8
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
core.std.LoadPlugin(os.path.join(ROOT, "liblgcr.so"))

W, H = 128, 128
F444 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 0, 0)
F420 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 1, 1)
BLANK = core.std.BlankClip(width=W, height=H, format=F444, length=1)


def make_clip(Yf, Uf, Vf, length=1):
    blank = core.std.BlankClip(width=W, height=H, format=F444, length=length)
    def fill(n, f):
        f = f.copy()
        np.copyto(np.asarray(f[0]), Yf[min(n, len(Yf) - 1)])
        np.copyto(np.asarray(f[1]), Uf[min(n, len(Uf) - 1)])
        np.copyto(np.asarray(f[2]), Vf[min(n, len(Vf) - 1)])
        return f
    return blank.std.ModifyFrame(blank, fill)


def degrade(clip):
    return clip.resize.Bilinear(format=F420)


def mae_band(out, gt, plane, mask):
    a = np.asarray(out[plane]).astype(np.float64)
    b = np.asarray(gt[plane]).astype(np.float64)
    return float(np.abs(a - b)[mask].mean())


def run_case(name, Yf, Uf, Vf, mask, algos, scale=1.0):
    """Yf/Uf/Vf: float32 GT planes (H,W). Returns {tag: mae}."""
    clip = make_clip([Yf], [Uf], [Vf])
    src420 = degrade(clip)
    fgt = clip.get_frame(0)
    ow, oh = int(W * scale), int(H * scale)
    if scale != 1.0:
        fgt = clip.resize.Spline36(width=ow, height=oh).get_frame(0)
        mask = np.kron(mask, np.ones((2, 2))) > 0  # 2x nearest expansion of mask
    res = {}
    for tag, kw in algos.items():
        if scale != 1.0 and kw.get("algo") == 5:
            res[tag] = float("nan")  # algo5 is same-size only; plugin rejects scaling
            continue
        out = core.lgcr.Recon(src420, width=ow, height=oh, kernel="jinc", taps=3, **kw).get_frame(0)
        res[tag] = (mae_band(out, fgt, 1, mask) + mae_band(out, fgt, 2, mask)) / 2
    return res


def grids():
    yy, xx = np.mgrid[0:H, 0:W]
    return yy, xx


def build_cases():
    yy, xx = grids()
    cases = {}

    def edge_planes(xedg, soft=0, cshift=0):
        # luma hard edge at xedg; chroma edge at xedg+cshift, optionally soft
        tY = np.clip(xx - xedg + 0.5, 0, 1)
        if soft:
            tC = np.clip((xx - (xedg + cshift) + soft / 2) / soft, 0, 1)
        else:
            tC = np.clip(xx - (xedg + cshift) + 0.5, 0, 1)
        Y = (0.30 + 0.05 * tY).astype(np.float32)  # small luma step (hard case for guide)
        U = (-0.15 + 0.55 * tC).astype(np.float32)
        V = (0.40 - 0.60 * tC).astype(np.float32)
        return Y, U, V

    band_v = (np.abs(xx - 64) <= 3) & (xx != 64) & (yy > 4) & (yy < H - 4)

    Y, U, V = edge_planes(64)
    cases["hard_v"] = (Y, U, V, band_v, 1.0)

    t = np.clip(xx - yy + 0.5, 0, 1)
    Y = (0.30 + 0.05 * t).astype(np.float32)
    U = (-0.15 + 0.55 * t).astype(np.float32)
    V = (0.40 - 0.60 * t).astype(np.float32)
    band_d = (np.abs(xx - yy) <= 2) & (xx != yy) & (yy > 4) & (yy < H - 4)
    cases["hard_d45"] = (Y, U, V, band_d, 1.0)

    # isoluminant: no luma edge at all
    Y = np.full((H, W), 0.35, np.float32)
    tC = np.clip(xx - 64 + 0.5, 0, 1)
    U = (-0.15 + 0.55 * tC).astype(np.float32)
    V = (0.40 - 0.60 * tC).astype(np.float32)
    cases["isoluminant"] = (Y, U, V, band_v, 1.0)

    Y, U, V = edge_planes(64, cshift=4)
    cases["misalign4"] = (Y, U, V, (np.abs(xx - 66) <= 5) & (yy > 4) & (yy < H - 4), 1.0)

    # Half-chroma-px Y/C phase offset (edge at 65, luma at 64): the degraded
    # chroma edge lands BETWEEN samples; snapping to the luma edge then
    # misplaces the reconstruction by 1 luma px. Found by eval_gates.py:
    # the worst single failure of the ungated snap (delta +0.026).
    Y, U, V = edge_planes(64, cshift=1)
    cases["misalign1"] = (Y, U, V, (np.abs(xx - 65) <= 4) & (yy > 4) & (yy < H - 4), 1.0)

    # Targeted Nyquist counterexamples: luma = step + axis/checker alternating
    # texture, chroma = step only. These are not a claim to span an unknown
    # encoder's entire null space.
    t = np.clip(xx - 64 + 0.5, 0, 1)
    for label, alt in (
        ("nullspace_x", 0.10 * np.where((xx % 2) == 0, 1.0, -1.0)),
        ("nullspace_y", 0.10 * np.where((yy % 2) == 0, 1.0, -1.0)),
        ("nullspace_xy", 0.10 * np.where(((xx + yy) % 2) == 0, 1.0, -1.0)),
    ):
        Y = (0.30 + 0.05 * t + alt).astype(np.float32)
        U = (-0.15 + 0.55 * t).astype(np.float32)
        V = (0.40 - 0.60 * t).astype(np.float32)
        cases[label] = (Y, U, V, band_v, 1.0)

    Y, U, V = edge_planes(64, soft=7)
    cases["hardL_softC"] = (Y, U, V, band_v, 1.0)

    # ridge line art: 2px dark luma line, chroma soft blend toward line color
    dist = np.abs(xx - 64)
    Y = np.where(dist <= 1, 0.35, 0.70).astype(np.float32)
    tC = np.clip((3 - dist) / 3, 0, 1)
    U = (-0.05 + 0.10 * tC).astype(np.float32)
    V = np.full((H, W), 0.10, np.float32)
    cases["ridge_line"] = (Y, U, V, (dist <= 5) & (yy > 4) & (yy < H - 4), 1.0)

    t = np.clip((xx - 60) / 7.0, 0, 1)
    Y = (0.30 + 0.05 * t).astype(np.float32)
    U = (-0.15 + 0.55 * t).astype(np.float32)
    V = (0.40 - 0.60 * t).astype(np.float32)
    cases["ramp"] = (Y, U, V, (xx >= 58) & (xx < 70) & (yy > 4) & (yy < H - 4), 1.0)

    Y = np.full((H, W), 0.5, np.float32)
    U = (0.15 * np.sin(xx * 0.9) * np.sin(yy * 0.7)).astype(np.float32)
    V = np.zeros((H, W), np.float32)
    cases["texture"] = (Y, U, V, (xx >= 8) & (xx < 120) & (yy >= 8) & (yy < 120), 1.0)

    rng = np.random.default_rng(7)
    Y, U, V = edge_planes(64)
    Y = np.clip(Y + rng.normal(0, 0.008, (H, W)), 0, 1).astype(np.float32)
    U = np.clip(U + rng.normal(0, 0.004, (H, W)), -0.5, 0.5).astype(np.float32)
    cases["noise"] = (Y, U, V, band_v, 1.0)

    Y, U, V = edge_planes(64)
    cases["upscale2x"] = (Y, U, V, band_v, 2.0)

    return cases


def run_temporal(algos):
    """Aligned edge-band error variance: edge moves 1 luma px between two
    frames; warp frame 1's per-pixel chroma error back by the known motion
    and measure the variance of the aligned errors (flicker/phase-recovery
    metric). The previous version measured inter-frame diff in the STATIC
    region, which is trivially zero and says nothing (external review)."""
    yy, xx = grids()
    Ys, Us, Vs = [], [], []
    for k in range(2):
        tY = np.clip(xx - (64 + k) + 0.5, 0, 1)
        tC = np.clip(xx - (64 + k) + 0.5, 0, 1)
        Ys.append((0.30 + 0.05 * tY).astype(np.float32))
        Us.append((-0.15 + 0.55 * tC).astype(np.float32))
        Vs.append((0.40 - 0.60 * tC).astype(np.float32))
    clip = make_clip(Ys, Us, Vs, length=2)
    src420 = degrade(clip)
    band = (np.abs(xx - 64) <= 4) & (yy > 4) & (yy < H - 4)
    band0 = band & (xx < W - 1)  # after the -1px back-warp
    out = {}
    for tag, kw in algos.items():
        node = core.lgcr.Recon(src420, kernel="jinc", taps=3, **kw)
        f0, f1 = node.get_frame(0), node.get_frame(1)
        g0, g1 = clip.get_frame(0), clip.get_frame(1)
        plane_vars = []
        for p in (1, 2):
            e0 = np.asarray(f0[p]).astype(np.float64) - np.asarray(g0[p]).astype(np.float64)
            e1 = np.asarray(f1[p]).astype(np.float64) - np.asarray(g1[p]).astype(np.float64)
            e1w = np.roll(e1, -1, axis=1)  # edge moved +1px: warp back
            temporal_var = np.var(np.stack((e0, e1w)), axis=0)
            plane_vars.append(temporal_var[band0].mean())
        out[tag] = float(np.mean(plane_vars))
    return out


def run_trecon_cases():
    """TRecon: odd-pel motion (true phase diversity), even-pel motion (same
    phase -> must NOT gain), static noise (averaging), single-frame isolation."""
    yy, xx = np.mgrid[0:H, 0:W]

    def edge_planes(xedg, noise=0.0, seed=0):
        t = np.clip(xx - xedg + 0.5, 0, 1)
        Y = (0.30 + 0.05 * t).astype(np.float32)
        U = (-0.15 + 0.55 * t).astype(np.float32)
        V = (0.40 - 0.60 * t).astype(np.float32)
        if noise:
            rng = np.random.default_rng(seed)
            Y = np.clip(Y + rng.normal(0, noise, (H, W)), 0, 1).astype(np.float32)
            U = np.clip(U + rng.normal(0, noise, (H, W)), -0.5, 0.5).astype(np.float32)
        return Y, U, V

    band = (np.abs(xx - 64) <= 4) & (yy > 4) & (yy < H - 4)
    out = {}
    for label, frames in (
        ("t_move1", [edge_planes(63), edge_planes(64), edge_planes(65)]),
        ("t_move2", [edge_planes(62), edge_planes(64), edge_planes(66)]),
        ("t_move3", [edge_planes(61), edge_planes(64), edge_planes(67)]),
        ("t_static_noise", [edge_planes(64, noise=0.01, seed=s) for s in (1, 2, 3)]),
    ):
        gt = make_clip([f[0] for f in frames], [f[1] for f in frames],
                       [f[2] for f in frames], length=len(frames))
        src420 = degrade(gt)
        fgt = gt.get_frame(1)
        row = {}
        for tag, node in (("plain", core.lgcr.Recon(src420, strength=0.0)),
                          ("algo2", core.lgcr.Recon(src420, strength=0.8, algo=2)),
                          ("trecon", core.lgcr.TRecon(src420, strength=0.8, trad=1))):
            f = node.get_frame(1)
            row[tag] = float(np.mean([np.abs(np.asarray(f[p]).astype(np.float64) -
                                            np.asarray(fgt[p]).astype(np.float64))[band].mean()
                                      for p in (1, 2)]))
        out[label] = row

    # Isolation: a single-frame clip has NO temporal information; TRecon must
    # reduce to Recon (boundary frames clamped to n are skipped, not used as
    # duplicate taps). Reports max abs chroma difference between the two.
    Y, U, V = edge_planes(64)
    gt = make_clip([Y], [U], [V], length=1)
    src420 = degrade(gt)
    fr = core.lgcr.Recon(src420, strength=0.8, algo=2).get_frame(0)
    ft = core.lgcr.TRecon(src420, strength=0.8, trad=1).get_frame(0)
    out["t_single"] = {"isolation_maxdiff": float(max(
        np.abs(np.asarray(fr[p]) - np.asarray(ft[p])).max() for p in (1, 2)))}

    # Flat-luma ME ambiguity: three IDENTICAL frames, isoluminant chroma
    # texture. Motion is unobservable; TRecon must not drag texture around.
    Y = np.full((H, W), 0.5, np.float32)
    U = (0.15 * np.sin(xx * 0.9) * np.sin(yy * 0.7)).astype(np.float32)
    V = (0.10 * np.cos(xx * 0.5)).astype(np.float32)
    gt = make_clip([Y] * 3, [U] * 3, [V] * 3, length=3)
    src420 = degrade(gt)
    fgt = gt.get_frame(1)
    row = {}
    for tag, node in (("algo2", core.lgcr.Recon(src420, strength=0.8, algo=2, sparse=0)),
                      ("trecon", core.lgcr.TRecon(src420, strength=0.8, trad=1))):
        f = node.get_frame(1)
        row[tag] = float(np.mean([np.abs(np.asarray(f[p]).astype(np.float64) -
                                        np.asarray(fgt[p]).astype(np.float64)).mean()
                                  for p in (1, 2)]))
    out["t_flat_amb"] = row
    return out


def run_int8_case():
    """Integer path: constant YUV420P8 (Y,U,V)=(64,128,128) must round-trip
    through Recon and TRecon without blowing up (regression: TRecon once
    normalized int input by 2^bits-1 instead of its reciprocal -> all 255)."""
    F420_8 = core.query_video_format(vs.YUV, vs.INTEGER, 8, 1, 1)
    blank = core.std.BlankClip(width=64, height=64, format=F420_8, length=3,
                               color=[64, 128, 128])
    res = {}
    for tag, node in (("recon8", core.lgcr.Recon(blank)),
                      ("trecon8", core.lgcr.TRecon(blank))):
        f = node.get_frame(1)
        res[tag] = [int(np.asarray(f[p][32, 32])) for p in range(3)]
    ok = all(abs(v - t) <= 1 for vals in res.values()
             for v, t in zip(vals, (64, 128, 128)))
    return res, ok


def check_gates(case_results, temporal_results, int8_ok):
    def pair(name):
        row = case_results[name]
        return row["plain"], row["algo2"]

    assert int8_ok, "integer Recon/TRecon round-trip failed"
    assert temporal_results["t_single"]["isolation_maxdiff"] <= 1e-6, (
        f"single-frame TRecon isolation failed: {temporal_results['t_single']}")
    assert abs(temporal_results["t_move2"]["trecon"] -
               temporal_results["t_move2"]["algo2"]) <= 1e-6, (
        f"even-pixel motion must not add temporal gain: {temporal_results['t_move2']}")
    assert temporal_results["t_flat_amb"]["trecon"] <= (
        temporal_results["t_flat_amb"]["algo2"] + 1e-6), (
        f"flat-luma motion ambiguity regressed: {temporal_results['t_flat_amb']}")
    assert temporal_results["t_static_noise"]["trecon"] < (
        temporal_results["t_static_noise"]["algo2"]), (
        f"TRecon must improve static noise: {temporal_results['t_static_noise']}")

    for name in ("hard_v", "hard_d45", "upscale2x"):
        plain, algo2 = pair(name)
        assert algo2 <= plain + 1e-6, f"{name}: algo2 {algo2} > plain {plain}"
    for name in ("isoluminant", "misalign4", "texture"):
        plain, algo2 = pair(name)
        assert abs(algo2 - plain) <= 1e-6, (
            f"{name}: algo2 differs from plain by {algo2 - plain}")
    for name in ("hardL_softC", "ridge_line", "ramp", "noise"):
        plain, algo2 = pair(name)
        assert algo2 <= plain + 0.001, (
            f"{name}: algo2 regression {algo2 - plain} exceeds 0.001")


def main():
    algos = {"plain": dict(strength=0.0), "algo2": dict(strength=0.8, algo=2)}
    if "--all" in sys.argv:
        for a in (1, 3, 4, 5, 6):
            algos[f"algo{a}"] = dict(strength=0.8, algo=a)

    cases = build_cases()
    tags = list(algos.keys())
    case_results = {}
    lines = []
    header = f"{'case':<13}" + "".join(f"{t:>10}" for t in tags)
    lines.append(header)
    lines.append("-" * len(header))
    for name, (Y, U, V, mask, scale) in cases.items():
        res = run_case(name, Y, U, V, mask, algos, scale)
        case_results[name] = res
        def fmt(v):
            return f"{v:>10.5f}" if v == v else f"{'n/a':>10}"
        lines.append(f"{name:<13}" + "".join(fmt(res[t]) for t in tags))
    res = run_temporal(algos)
    lines.append(f"{'temporal':<13}" + "".join(f"{res[t]:>10.5f}" for t in tags)
                 + "   (aligned edge-band error variance; lower=better)")
    tr = run_trecon_cases()
    lines.append("")
    lines.append("temporal cases (plain / algo2 / trecon):")
    for label, row in tr.items():
        if label == "t_single":
            lines.append(f"{label:<13} isolation maxdiff = {row['isolation_maxdiff']:.6f}"
                         "  (must be ~0: single frame has no temporal information)")
        elif label == "t_flat_amb":
            lines.append(f"{label:<13}" + "".join(f"{row[t]:>10.5f}" for t in ("algo2", "trecon"))
                         + "   (identical frames, unobservable motion: trecon ~= algo2 required)")
        else:
            lines.append(f"{label:<13}" + "".join(f"{row[t]:>10.5f}" for t in ("plain", "algo2", "trecon")))
    res8, ok8 = run_int8_case()
    lines.append("")
    lines.append(f"int8 constant (64,128,128): {res8}  {'OK' if ok8 else 'FAIL'}")
    report = "\n".join(lines)
    print(report)
    if "--check" in sys.argv:
        check_gates(case_results, tr, ok8)
        print("battery gates: OK")
    if "--write-results" in sys.argv:
        results_dir = os.path.join(HERE, "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "latest.md"), "w") as f:
            f.write("# LGCR battery results\n\n```\n" + report + "\n```\n")


if __name__ == "__main__":
    main()
