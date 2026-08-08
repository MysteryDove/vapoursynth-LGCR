#!/usr/bin/env python3
"""Gate/statistic evaluation for LGCR — classifier metrics, separate from the
reconstruction battery.

Idea (external review): evaluate each candidate gate/statistic as a CLASSIFIER
whose job is to predict "does enabling the guided mechanism beat the plain
kernel on this sample", not to match a human hard/soft semantic.

  label = 1  if  bandMAE(guided ungated) < bandMAE(plain) - eps   (mechanism helps)
  label = 0  if  bandMAE(guided ungated) > bandMAE(plain) + eps   (mechanism hurts)
  near-zero deltas are excluded (no effect either way)

Statistics under test (all recomputed here in numpy from the SAME degraded
planes the plugin sees):
  cedge_proxy - legacy max-projection width fade proxy (not pixel-exact plugin output)
  width_eq   - corrected rotation-equivariant normal-profile width baseline
  ms         - in-plugin mutual-structure gate (participation width + centroid)
  q_box/q_tri/q_bic - affine credibility with candidate degradation kernels
  q_min      - pixelwise min over candidate kernels
  q_prod     - production qMean * (qmin/qmax) * chroma significance
  q_prod_ms  - mean of the per-pixel production q * ms composite

Outputs: AUC per statistic + risk/coverage table
(risk = mean signed MAE delta vs plain among accepted samples; negative = win).
Every generated source is evaluated under box, triangle, and bicubic ACTUAL
4:2:0 degradations; these are independent of the three candidate q models.

Usage: python3 test/eval_gates.py
"""
import io
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
F420 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 1, 1)

YY, XX = np.mgrid[0:H, 0:W]


# ---------------------------------------------------------------------------
# sample generation
# ---------------------------------------------------------------------------

def make_420_clip(Yf, Uf, Vf):
    """Construct the decoder-visible source without asking zimg to choose D."""
    blank = core.std.BlankClip(width=W, height=H, format=F420, length=1)
    def fill(n, f):
        f = f.copy()
        np.copyto(np.asarray(f[0]), Yf)
        np.copyto(np.asarray(f[1]), Uf)
        np.copyto(np.asarray(f[2]), Vf)
        return f
    return blank.std.ModifyFrame(blank, fill)


def edge_sample(soft=0, shift=0, angle=0.0, contrast=0.55, luma_step=0.05,
                noise=0.0, seed=7):
    """Hard/soft co-edged or misaligned edge. angle in degrees (0 = vertical)."""
    rad = np.deg2rad(angle)
    # Signed normal coordinate in luma pixels; all orientations and shifts now
    # have the same physical scale and pass through the frame center.
    tcoord = (XX - 64.0) * np.cos(rad) + (YY - 64.0) * np.sin(rad) + 64.0
    e = 64.0
    tY = np.clip(tcoord - e + 0.5, 0, 1)
    if soft:
        tC = np.clip((tcoord - (e + shift) + soft / 2) / soft, 0, 1)
    else:
        tC = np.clip(tcoord - (e + shift) + 0.5, 0, 1)
    Y = (0.30 + luma_step * tY).astype(np.float32)
    U = (-0.15 + contrast * tC).astype(np.float32)
    V = (0.40 - (contrast + 0.05) * tC).astype(np.float32)
    if noise:
        rng = np.random.default_rng(seed)
        Y = np.clip(Y + rng.normal(0, noise, (H, W)), 0, 1).astype(np.float32)
    band = (np.abs(tcoord - e) <= 3) & (tcoord != e) & (YY > 6) & (YY < H - 6)
    return Y, U, V, band


def gen_samples():
    S = []
    # soft=1 is algebraically identical to soft=0 on this sampled ramp, so it
    # must not be counted as an independent observation.
    for soft in (0, 2, 3, 5, 7):
        for shift in (0, 0.5, 1, 2, 4):  # luma px; 0.5/1 = sub-chroma-px phase
            for ang in (0.0, 22.5, 45.0):
                for c in (0.55, 0.20):
                    S.append((f"edge_s{soft}_m{shift}_a{int(ang)}_c{c}",
                              *edge_sample(soft, shift, ang, c)))
    # isoluminant: no luma edge
    t = np.clip(XX - 64 + 0.5, 0, 1)
    Y = np.full((H, W), 0.35, np.float32)
    S.append(("isoluminant", Y, (-0.15 + 0.55 * t).astype(np.float32),
              (0.40 - 0.60 * t).astype(np.float32),
              (np.abs(XX - 64) <= 3) & (YY > 6) & (YY < H - 6)))
    # Axis/checker Nyquist counterexamples. These are targeted aliases, not a
    # claim to span the full null space of an unknown encoder.
    for label, alt in (
        ("nullspace_x", 0.10 * np.where((XX % 2) == 0, 1.0, -1.0)),
        ("nullspace_y", 0.10 * np.where((YY % 2) == 0, 1.0, -1.0)),
        ("nullspace_xy", 0.10 * np.where(((XX + YY) % 2) == 0, 1.0, -1.0)),
    ):
        Yn = (0.30 + 0.05 * t + alt).astype(np.float32)
        S.append((label, Yn, (-0.15 + 0.55 * t).astype(np.float32),
                  (0.40 - 0.60 * t).astype(np.float32),
                  (np.abs(XX - 64) <= 3) & (YY > 6) & (YY < H - 6)))
    # ridge line art
    dist = np.abs(XX - 64)
    Yr = np.where(dist <= 1, 0.35, 0.70).astype(np.float32)
    tC = np.clip((3 - dist) / 3, 0, 1)
    S.append(("ridge", Yr, (-0.05 + 0.10 * tC).astype(np.float32),
              np.full((H, W), 0.10, np.float32),
              (dist <= 5) & (YY > 6) & (YY < H - 6)))
    # ramp
    t7 = np.clip((XX - 60) / 7.0, 0, 1)
    S.append(("ramp", (0.30 + 0.05 * t7).astype(np.float32),
              (-0.15 + 0.55 * t7).astype(np.float32),
              (0.40 - 0.60 * t7).astype(np.float32),
              (XX >= 56) & (XX < 72) & (YY > 6) & (YY < H - 6)))
    return S


# ---------------------------------------------------------------------------
# numpy versions of the degradation + statistics
# ---------------------------------------------------------------------------

def sited_box(Y):
    """Footprint box to the 420 chroma grid (left siting: luma {2cx-1, 2cx})."""
    out = np.zeros((H // 2, W // 2))
    for cy in range(H // 2):
        for cx in range(W // 2):
            ys = np.clip((2 * cy, 2 * cy + 1), 0, H - 1)
            xs = np.clip((2 * cx - 1, 2 * cx), 0, W - 1)
            out[cy, cx] = Y[np.ix_(ys, xs)].mean()
    return out


def _axis_weights(n_src, kind, sit):
    """Per-output-sample (index, weight) lists for 2x downscale, sited."""
    out = []
    for c in range(n_src // 2):
        pos = (c + 0.5) * 2 - 0.5 + sit  # luma units
        ws = {}
        if kind == "box":
            idx = range(int(np.ceil(pos - 1)), int(np.ceil(pos + 1)))
            for i in idx:
                ws[i] = ws.get(i, 0.0) + 0.5
        else:
            sup = 2.0 if kind == "tri" else 4.0
            for i in range(int(np.floor(pos - sup)), int(np.ceil(pos + sup)) + 1):
                d = abs(i - pos) / 2.0  # source units -> kernel units
                if kind == "tri":
                    w = max(0.0, 1.0 - d)
                else:  # bicubic b=0 c=0.6 (Keys)
                    b_, c_ = 0.0, 0.6
                    if d < 1:
                        w = ((12 - 9 * b_ - 6 * c_) * d**3 + (-18 + 12 * b_ + 6 * c_) * d**2 + (6 - 2 * b_)) / 6
                    elif d < 2:
                        w = ((-b_ - 6 * c_) * d**3 + (6 * b_ + 30 * c_) * d**2 + (-12 * b_ - 48 * c_) * d + (8 * b_ + 24 * c_)) / 6
                    else:
                        w = 0.0
                if w:
                    ws[i] = ws.get(i, 0.0) + w
        tot = sum(ws.values())
        out.append([(min(max(i, 0), n_src - 1), w / tot) for i, w in ws.items()])
    return out


def degrade_kernel(Y, kind):
    """Candidate matched degradation D^(Y): 2x separable, left siting."""
    wx = _axis_weights(W, kind, -0.5)
    wy = _axis_weights(H, kind, 0.0)
    tmp = np.zeros((H, W // 2))
    for cx, iws in enumerate(wx):
        for i, w in iws:
            tmp[:, cx] += Y[:, i] * w
    out = np.zeros((H // 2, W // 2))
    for cy, iws in enumerate(wy):
        for j, w in iws:
            out[cy, :] += tmp[j, :] * w
    return out


def sobel(p):
    P = np.pad(p, 1, mode="edge")
    gx = (P[0:-2, 2:] + 2 * P[1:-1, 2:] + P[2:, 2:] - P[0:-2, 0:-2] - 2 * P[1:-1, 0:-2] - P[2:, 0:-2]) / 8
    gy = (P[2:, 0:-2] + 2 * P[2:, 1:-1] + P[2:, 2:] - P[0:-2, 0:-2] - 2 * P[0:-2, 1:-1] - P[0:-2, 2:]) / 8
    return gx, gy


def box3(a):
    P = np.pad(a, 1, mode="edge")
    h, w = a.shape
    out = np.zeros_like(a)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            out += P[1 + dj:1 + dj + h, 1 + di:1 + di + w]
    return out / 9


def bil(p, x, y):
    x = np.clip(x, 0, p.shape[1] - 1.001)
    y = np.clip(y, 0, p.shape[0] - 1.001)
    x0 = x.astype(int)
    y0 = y.astype(int)
    fx = x - x0
    fy = y - y0
    return (p[y0, x0] * (1 - fx) * (1 - fy) + p[y0, x0 + 1] * fx * (1 - fy)
            + p[y0 + 1, x0] * (1 - fx) * fy + p[y0 + 1, x0 + 1] * fx * fy)


def normals_from(Yc):
    gx, gy = sobel(Yc)
    sxx, sxy, syy = box3(gx * gx), box3(gx * gy), box3(gy * gy)
    theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    return np.cos(theta), np.sin(theta), sxx + syy


def band_chroma(band):
    """Full-res bool band -> chroma-res bool (any full-res px in footprint)."""
    b = np.zeros((H // 2, W // 2), bool)
    for cy in range(H // 2):
        for cx in range(W // 2):
            y0, y1 = max(0, 2 * cy - 1), min(H, 2 * cy + 2)
            x0, x1 = max(0, 2 * cx - 1), min(W, 2 * cx + 2)
            b[cy, cx] = band[y0:y1, x0:x1].any()
    return b


# --- statistic: current in-plugin cedge (max-projection width fade) ---------

def stat_cedge_proxy(U, V, nxm, nym, bcm):
    ch, cw = U.shape
    vals = []
    for cy, cx in np.argwhere(bcm):
        x0, x1 = max(0, cx - 3), min(cw, cx + 4)
        y0, y1 = max(0, cy - 3), min(ch, cy + 4)
        rc = max(np.ptp(U[y0:y1, x0:x1]), np.ptp(V[y0:y1, x0:x1]))
        if rc < 1e-6:
            vals.append(1.0)  # mirrors the current plugin's pass-through default
            continue
        anx, any_ = abs(nxm[cy, cx]), abs(nym[cy, cx])
        gmax = 0.0
        for j in range(y0, y1):
            for i in range(x0, x1):
                i1 = min(i + 1, cw - 1)
                j1 = min(j + 1, ch - 1)
                gu = max(abs(U[j, i1] - U[j, i]) * anx,
                         abs(U[j1, i] - U[j, i]) * any_)
                gv = max(abs(V[j, i1] - V[j, i]) * anx,
                         abs(V[j1, i] - V[j, i]) * any_)
                gmax = max(gmax, gu, gv)
        wc = rc / (gmax + 1e-6)
        vals.append(np.clip((2.2 - wc) / 0.7, 0, 1))
    return float(np.mean(vals)) if vals else 0.0


# --- statistic: rotation-equivariant normal-profile width -------------------

def stat_cedge_eq(U, V, nxm, nym, bcm):
    vals = []
    ks = np.arange(-3, 4)
    for cy, cx in np.argwhere(bcm):
        nx, ny = nxm[cy, cx], nym[cy, cx]
        px, py = cx + ks * nx, cy + ks * ny
        pu = bil(U, px, py)
        pv = bil(V, px, py)
        # Project onto the endpoint chroma-change direction. Equal endpoints
        # (ridge/nonmonotonic profile) are absence of evidence, not a hard edge.
        du = pu[-1] - pu[0]
        dv = pv[-1] - pv[0]
        nrm = np.hypot(du, dv)
        if nrm < 1e-6:
            vals.append(0.0)
            continue
        s = (pu * du + pv * dv) / nrm
        lo, hi = s.min(), s.max()
        rng = hi - lo
        total_variation = np.abs(np.diff(s)).sum()
        if rng < 1e-6 or total_variation < 1e-6:
            vals.append(0.0)
            continue

        def crossing(level):
            for k in range(6):
                if (s[k] - level) * (s[k + 1] - level) <= 0 and s[k + 1] != s[k]:
                    return k + (level - s[k]) / (s[k + 1] - s[k])
            return None

        c10 = crossing(lo + 0.1 * rng)
        c90 = crossing(lo + 0.9 * rng)
        if c10 is None or c90 is None or c90 <= c10:
            vals.append(0.0)
            continue
        width_fade = np.clip((2.2 - (c90 - c10)) / 0.7, 0, 1)
        monotonicity = abs(s[-1] - s[0]) / (total_variation + 1e-12)
        vals.append(width_fade * np.clip(monotonicity, 0, 1))
    return float(np.mean(vals)) if vals else 0.0


# --- statistic: ms gate (participation width + centroid, mirrors maps.cpp) --

def ms_map(Yc, U, V, mask=None):
    gx, gy = sobel(Yc)
    ux, uy = sobel(U)
    vx, vy = sobel(V)
    nxm, nym, _ = normals_from(Yc)
    ch, cw = Yc.shape
    gate = np.zeros((ch, cw))
    ks = np.arange(-3, 4, dtype=float)
    coords = np.argwhere(mask) if mask is not None else np.argwhere(np.ones_like(gate, bool))
    for cy, cx in coords:
        nx, ny = nxm[cy, cx], nym[cy, cx]
        px, py = cx + ks * nx, cy + ks * ny
        gY = np.abs(bil(gx, px, py) * nx + bil(gy, px, py) * ny)
        au = bil(ux, px, py) * nx + bil(uy, px, py) * ny
        av = bil(vx, px, py) * nx + bil(vy, px, py) * ny
        gC = np.sqrt(au * au + av * av)
        sumY, sumC = gY.sum(), gC.sum()
        if sumY < 1e-9 or sumC < 1e-9:
            continue
        partY = sumY * sumY / (gY ** 2).sum()
        partC = sumC * sumC / (gC ** 2).sum()
        ratio = partC / partY
        dcent = abs((ks * gY).sum() / sumY - (ks * gC).sum() / sumC)
        tw = np.clip((1.6 - ratio) / 0.4, 0, 1)
        tp = np.clip((1.0 - dcent) / 0.5, 0, 1)
        gate[cy, cx] = (tw * tw * (3 - 2 * tw)) * (tp * tp * (3 - 2 * tp))
    return gate


# --- statistic: affine credibility q with candidate kernels -----------------

def local_mean(a, radius=2):
    """Clipped-window population mean, matching buildAffineMaps()."""
    h, w = a.shape
    p = np.pad(a, radius, mode="constant")
    sums = np.zeros_like(a, dtype=np.float64)
    counts = np.zeros_like(a, dtype=np.float64)
    ones = np.pad(np.ones_like(a, dtype=np.float64), radius, mode="constant")
    for dj in range(2 * radius + 1):
        for di in range(2 * radius + 1):
            sums += p[dj:dj + h, di:di + w]
            counts += ones[dj:dj + h, di:di + w]
    return sums / counts


def q_maps(Y, U, V, eps=0.005 ** 2, eps_sig=(0.25 * 0.01) ** 2):
    """Return every per-pixel component of the production algo=6 q gate."""
    mean_u, mean_v = local_mean(U), local_mean(V)
    var_u = np.maximum(0.0, local_mean(U * U) - mean_u * mean_u)
    var_v = np.maximum(0.0, local_mean(V * V) - mean_v * mean_v)
    var_c = var_u + var_v

    candidates = []
    for kind in ("box", "tri", "bic"):
        yc = degrade_kernel(Y, kind)
        mean_y = local_mean(yc)
        var_y = np.maximum(0.0, local_mean(yc * yc) - mean_y * mean_y)
        cov_u = local_mean(yc * U) - mean_y * mean_u
        cov_v = local_mean(yc * V) - mean_y * mean_v
        q = (cov_u * cov_u + cov_v * cov_v) / ((var_y + eps) * (var_c + eps))
        candidates.append(np.clip(q, 0, 1))

    stack = np.stack(candidates)
    q_min = stack.min(axis=0)
    q_max = stack.max(axis=0)
    q_mean = stack.mean(axis=0)
    stability = np.divide(q_min, q_max, out=np.zeros_like(q_min), where=q_max > 1e-9)
    significance = var_c / (var_c + eps_sig)
    q_prod = q_mean * stability * significance
    return {
        "q_box": candidates[0],
        "q_tri": candidates[1],
        "q_bic": candidates[2],
        "q_min": q_min,
        "q_mean": q_mean,
        "stability": stability,
        "significance": significance,
        "q_prod": q_prod,
    }


# ---------------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------------

def auc_for(rows, score, label):
    pos = np.asarray([r[score] for r in rows if r[label] < 0])
    neg = np.asarray([r[score] for r in rows if r[label] > 0])
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def coverage_result(rows, score, label, target):
    values = np.sort([r[score] for r in rows])[::-1]
    rank = max(1, int(np.ceil(target * len(values)))) - 1
    threshold = values[rank]
    accepted = [r for r in rows if r[score] >= threshold]
    risk = float(np.mean([r[label] for r in accepted]))
    return len(accepted) / len(rows), risk


def report_classifier(rows, label, stats, title, eps=5e-4):
    labeled = [r for r in rows if abs(r[label]) > eps]
    pos = sum(r[label] < 0 for r in labeled)
    neg = sum(r[label] > 0 for r in labeled)
    print(f"\n--- {title} ---")
    print(f"{len(rows)} observations, {len(labeled)} labeled "
          f"({pos} benefit / {neg} harm), {len(rows) - len(labeled)} neutral excluded; "
          f"mean delta={np.mean([r[label] for r in rows]):+.5f}")
    print(f"{'statistic':<12} {'AUC':>6}   risk at target coverage "
          "(target->actual:risk; lower is better)")
    for score in stats:
        line = f"{score:<12} {auc_for(labeled, score, label):>6.3f}   "
        for target in (0.05, 0.10, 0.20, 0.50, 1.0):
            actual, risk = coverage_result(labeled, score, label, target)
            line += f"{int(target * 100):>3}%->{actual * 100:>4.1f}%:{risk:+.4f}  "
        print(line)

    print("per-actual-D AUC:")
    print(f"{'statistic':<12}" + "".join(f"{kind:>10}" for kind in ("box", "tri", "bic")))
    for score in stats:
        line = f"{score:<12}"
        for kind in ("box", "tri", "bic"):
            subset = [r for r in labeled if r["actual_d"] == kind]
            line += f"{auc_for(subset, score, label):>10.3f}"
        print(line)


def main():
    samples = gen_samples()
    rows = []
    actual_kernels = ("box", "tri", "bic")
    for sample_index, (name, Y, U, V, band) in enumerate(samples, 1):
        y64 = Y.astype(np.float64)
        u64 = U.astype(np.float64)
        v64 = V.astype(np.float64)
        yc_box = sited_box(y64)
        nxm, nym, _ = normals_from(yc_box)
        bcm = band_chroma(band)

        for actual_d in actual_kernels:
            u420 = degrade_kernel(u64, actual_d)
            v420 = degrade_kernel(v64, actual_d)
            src420 = make_420_clip(Y, u420.astype(np.float32), v420.astype(np.float32))
            maes = {}
            for tag, kw in (
                ("plain", dict(strength=0.0)),
                ("algo2_ungated", dict(strength=0.8, algo=2, ms=0, cedge=0)),
                ("algo6_ungated", dict(strength=0.8, algo=6, qgate=0, ms=0)),
                ("algo6_full", dict(strength=0.8, algo=6, qgate=1, ms=1)),
            ):
                f = core.lgcr.Recon(src420, kernel="jinc", taps=3, **kw).get_frame(0)
                maes[tag] = float((np.abs(np.asarray(f[1]).astype(np.float64) - u64)[band].mean()
                                   + np.abs(np.asarray(f[2]).astype(np.float64) - v64)[band].mean()) / 2)

            msm = ms_map(yc_box, u420, v420, bcm)
            qm = q_maps(y64, u420, v420)
            means = {key: float(value[bcm].mean()) for key, value in qm.items()}
            rows.append({
                "name": name,
                "actual_d": actual_d,
                "delta2": maes["algo2_ungated"] - maes["plain"],
                "delta6": maes["algo6_ungated"] - maes["plain"],
                "gain6": maes["algo6_full"] - maes["plain"],
                "cedge_proxy": stat_cedge_proxy(u420, v420, nxm, nym, bcm),
                "width_eq": stat_cedge_eq(u420, v420, nxm, nym, bcm),
                "ms": float(msm[bcm].mean()),
                "q_prod_ms": float((qm["q_prod"] * msm)[bcm].mean()),
                **means,
            })
        if sample_index % 25 == 0 or sample_index == len(samples):
            print(f"[{sample_index:>3}/{len(samples)}] {name}")

    print(f"\n{len(samples)} unique samples x {len(actual_kernels)} actual degradations "
          f"= {len(rows)} observations")
    report_classifier(
        rows, "delta2", ("cedge_proxy", "width_eq", "ms"),
        "label: ungated algo=2 mechanism vs plain",
    )
    report_classifier(
        rows, "delta6", ("q_box", "q_min", "q_prod", "ms", "q_prod_ms"),
        "label: q/ms-independent algo=6 detail transfer vs plain",
    )
    print("\nfull algo=6 mean delta vs plain by actual D:")
    for kind in actual_kernels:
        vals = [r["gain6"] for r in rows if r["actual_d"] == kind]
        print(f"  {kind:<4} {np.mean(vals):+.5f}")
    print(f"  all  {np.mean([r['gain6'] for r in rows]):+.5f}")
    full_labeled = [r for r in rows if abs(r["gain6"]) > 5e-4]
    print(f"full algo=6 labeled outcomes: "
          f"{sum(r['gain6'] < 0 for r in full_labeled)} benefit / "
          f"{sum(r['gain6'] > 0 for r in full_labeled)} harm / "
          f"{len(rows) - len(full_labeled)} neutral")
    print("worst full algo=6 residuals:")
    for row in sorted(rows, key=lambda r: r["gain6"], reverse=True)[:8]:
        print(f"  {row['name']:<26} D={row['actual_d']:<3} "
              f"delta={row['gain6']:+.5f} q*ms={row['q_prod_ms']:.3f}")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


if __name__ == "__main__":
    capture = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, capture)
    try:
        main()
    finally:
        sys.stdout = original_stdout
    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "eval_gates_latest.md"), "w") as report:
        report.write("# LGCR gate evaluation\n\n```text\n")
        report.write("\n".join(line.rstrip() for line in capture.getvalue().splitlines()))
        report.write("\n")
        report.write("```\n")
