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
  cedge_cur  - current in-plugin width fade (max-projection, per-plane max)
  cedge_eq   - rotation-equivariant normal-profile 10-90% width version
  ms         - in-plugin mutual-structure gate (participation width + centroid)
  q_box/q_tri/q_bic - affine credibility with candidate degradation kernels
  q_min      - min over candidate kernels (unknown-D stability)
  qxms       - q_min x ms (the composite the review argues for)

Outputs: AUC per statistic + risk/coverage table
(risk = mean signed MAE delta vs plain among accepted samples; negative = win).

Usage: python3 test/eval_gates.py
"""
import numpy as np
import vapoursynth as vs

core = vs.core
core.num_threads = 8
core.std.LoadPlugin("/home/owen/dev/bsflab/liblgcr.so")

W, H = 128, 128
F444 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 0, 0)
F420 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 1, 1)

YY, XX = np.mgrid[0:H, 0:W]


# ---------------------------------------------------------------------------
# sample generation
# ---------------------------------------------------------------------------

def make_clip(Yf, Uf, Vf):
    blank = core.std.BlankClip(width=W, height=H, format=F444, length=1)
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
    if angle == 0.0:
        tcoord = XX
    elif angle == 45.0:
        tcoord = (XX + YY) / 2.0
    else:  # 22.5
        tcoord = XX * 0.924 + YY * 0.383
    e = 64.0 if angle == 0.0 else 64.0
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
    for soft in (0, 1, 2, 3, 5, 7):
        for shift in (0, 0.5, 1, 2, 4):  # luma px; 0.5/1 = sub-chroma-px phase
            for ang in (0.0, 45.0):
                for c in (0.55, 0.20):
                    S.append((f"edge_s{soft}_m{shift}_a{int(ang)}_c{c}",
                              *edge_sample(soft, shift, ang, c)))
    # isoluminant: no luma edge
    t = np.clip(XX - 64 + 0.5, 0, 1)
    Y = np.full((H, W), 0.35, np.float32)
    S.append(("isoluminant", Y, (-0.15 + 0.55 * t).astype(np.float32),
              (0.40 - 0.60 * t).astype(np.float32),
              (np.abs(XX - 64) <= 3) & (YY > 6) & (YY < H - 6)))
    # null-space texture: luma step + alternating texture in the null space of
    # the 2x box (and triangle, bicubic) degradation; chroma step only.
    alt = 0.10 * np.where((XX % 2) == 0, 1.0, -1.0)
    Yn = (0.30 + 0.05 * t + alt).astype(np.float32)
    S.append(("nullspace", Yn, (-0.15 + 0.55 * t).astype(np.float32),
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

def degrade(clip):
    return clip.resize.Bilinear(format=F420)


def sited_box(Y):
    """Footprint box to the 420 chroma grid (left siting: luma {2cx-1, 2cx})."""
    P = np.pad(Y, ((1, 1), (1, 1)), mode="edge")
    out = np.zeros((H // 2, W // 2))
    for cy in range(H // 2):
        for cx in range(W // 2):
            out[cy, cx] = P[2 * cy:2 * cy + 2, 2 * cx:2 * cx + 2].mean()
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
            b[cy, cx] = band[2 * cy - 1:2 * cy + 2, 2 * cx - 1:2 * cx + 2].any()
    return b


# --- statistic: current in-plugin cedge (max-projection width fade) ---------

def stat_cedge_cur(U, V, nxm, nym, bcm):
    ch, cw = U.shape
    val = np.zeros((ch, cw))
    for cy in range(ch):
        for cx in range(cw):
            x0, x1 = max(0, cx - 3), min(cw, cx + 4)
            y0, y1 = max(0, cy - 3), min(ch, cy + 4)
            rc = max(np.ptp(U[y0:y1, x0:x1]), np.ptp(V[y0:y1, x0:x1]))
            if rc < 1e-6:
                val[cy, cx] = 1.0
                continue
            anx, any_ = abs(nxm[cy, cx]), abs(nym[cy, cx])
            gmax = 0.0
            for j in range(y0, y1):
                for i in range(x0, x1):
                    i1 = min(i + 1, cw - 1)
                    j1 = min(j + 1, ch - 1)
                    gu = max(abs(U[j, i1] - U[j, i]) * anx, abs(U[j1, i] - U[j, i]) * any_)
                    gv = max(abs(V[j, i1] - V[j, i]) * anx, abs(V[j1, i] - V[j, i]) * any_)
                    gmax = max(gmax, gu, gv)
            wc = rc / (gmax + 1e-6)
            val[cy, cx] = np.clip((2.2 - wc) / 0.7, 0, 1)
    return val[bcm].mean()


# --- statistic: rotation-equivariant normal-profile width -------------------

def stat_cedge_eq(U, V, nxm, nym, bcm):
    ch, cw = U.shape
    val = np.zeros((ch, cw))
    cyy, cxx = np.mgrid[0:ch, 0:cw].astype(float)
    ks = np.arange(-3, 4)
    for cy in range(ch):
        for cx in range(cw):
            if not bcm[cy, cx]:
                continue
            nx, ny = nxm[cy, cx], nym[cy, cx]
            px, py = cx + ks * nx, cy + ks * ny
            pu = bil(U, px, py)
            pv = bil(V, px, py)
            # project onto the dominant chroma-change direction
            du = pu[-1] - pu[0]
            dv = pv[-1] - pv[0]
            nrm = np.hypot(du, dv)
            if nrm < 1e-6:
                val[cy, cx] = 1.0
                continue
            s = (pu * du + pv * dv) / nrm
            lo, hi = s.min(), s.max()
            rng = hi - lo
            if rng < 1e-6:
                val[cy, cx] = 1.0
                continue
            # 10-90% rise distance along the profile (linear crossings)
            def crossing(level):
                for k in range(6):
                    if (s[k] - level) * (s[k + 1] - level) <= 0 and s[k + 1] != s[k]:
                        return k + (level - s[k]) / (s[k + 1] - s[k])
                return None
            c10 = crossing(lo + 0.1 * rng)
            c90 = crossing(lo + 0.9 * rng)
            if c10 is None or c90 is None or c90 <= c10:
                w = 1.0  # monotonic step within window: hard
            else:
                w = c90 - c10
            val[cy, cx] = np.clip((2.2 - w) / 0.7, 0, 1)
    return val[bcm].mean() if bcm.any() else 1.0


# --- statistic: ms gate (participation width + centroid, mirrors maps.cpp) --

def ms_map(Yc, U, V):
    gx, gy = sobel(Yc)
    ux, uy = sobel(U)
    vx, vy = sobel(V)
    nxm, nym, _ = normals_from(Yc)
    ch, cw = Yc.shape
    gate = np.zeros((ch, cw))
    cyy, cxx = np.mgrid[0:ch, 0:cw].astype(float)
    ks = np.arange(-3, 4, dtype=float)
    for cy in range(ch):
        for cx in range(cw):
            nx, ny = nxm[cy, cx], nym[cy, cx]
            px, py = cx + ks * nx, cy + ks * ny
            gY = np.abs(bil(gx, px, py) * nx + bil(gy, px, py) * ny)
            au = bil(ux, px, py) * nx + bil(uy, px, py) * ny
            av = bil(vx, px, py) * nx + bil(vy, px, py) * ny
            gC = np.sqrt(au * au + av * av)
            sumY, sumC = gY.sum(), gC.sum()
            if sumY < 1e-9 or sumC < 1e-9:
                gate[cy, cx] = 0.0
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

def q_map(Y, U, V, kind, eps=1e-4):
    Yc = degrade_kernel(Y, kind)
    ch, cw = Yc.shape
    q = np.zeros((ch, cw))
    r = 2
    for cy in range(ch):
        for cx in range(cw):
            y0, y1 = max(0, cy - r), min(ch, cy + r + 1)
            x0, x1 = max(0, cx - r), min(cw, cx + r + 1)
            yw = Yc[y0:y1, x0:x1].ravel()
            uw = U[y0:y1, x0:x1].ravel()
            vw = V[y0:y1, x0:x1].ravel()
            vy_ = yw.var()
            covU = np.cov(yw, uw)[0, 1]
            covV = np.cov(yw, vw)[0, 1]
            q[cy, cx] = (covU ** 2 + covV ** 2) / ((vy_ + eps) * (uw.var() + vw.var() + eps))
    return np.clip(q, 0, 1)


# ---------------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------------

def main():
    samples = gen_samples()
    rows = []
    for name, Y, U, V, band in samples:
        clip = make_clip(Y, U, V)
        src420 = degrade(clip)
        fgt = clip.get_frame(0)
        gtU = np.asarray(fgt[1]).astype(np.float64)
        gtV = np.asarray(fgt[2]).astype(np.float64)
        maes = {}
        for tag, kw in (("plain", dict(strength=0.0)),
                        ("ungated", dict(strength=0.8, algo=2, ms=0, cedge=0))):
            f = core.lgcr.Recon(src420, kernel="jinc", taps=3, **kw).get_frame(0)
            maes[tag] = (np.abs(np.asarray(f[1]).astype(np.float64) - gtU)[band].mean()
                         + np.abs(np.asarray(f[2]).astype(np.float64) - gtV)[band].mean()) / 2
        delta = maes["ungated"] - maes["plain"]

        # statistics from the same planes the plugin sees
        fs = src420.get_frame(0)
        U420 = np.asarray(fs[1]).astype(np.float64)
        V420 = np.asarray(fs[2]).astype(np.float64)
        Yc = sited_box(Y.astype(np.float64))
        nxm, nym, _ = normals_from(Yc)
        bcm = band_chroma(band)
        ms = ms_map(Yc, U420, V420)[bcm].mean()
        qb = q_map(Y.astype(np.float64), U420, V420, "box")[bcm].mean()
        qt = q_map(Y.astype(np.float64), U420, V420, "tri")[bcm].mean()
        qc = q_map(Y.astype(np.float64), U420, V420, "bic")[bcm].mean()
        rows.append(dict(name=name, delta=delta,
                         cedge_cur=stat_cedge_cur(U420, V420, nxm, nym, bcm),
                         cedge_eq=stat_cedge_eq(U420, V420, nxm, nym, bcm),
                         ms=ms, q_box=qb, q_tri=qt, q_bic=qc,
                         q_min=min(qb, qt, qc)))
        print(f"{name:<26} delta={delta:+.5f} ms={ms:.3f} q_min={rows[-1]['q_min']:.3f}")

    # classifier evaluation
    eps = 5e-4
    labeled = [r for r in rows if abs(r["delta"]) > eps]
    pos = [r for r in labeled if r["delta"] < 0]  # mechanism helps
    neg = [r for r in labeled if r["delta"] > 0]
    print(f"\n{len(rows)} samples, {len(labeled)} labeled "
          f"({len(pos)} benefit / {len(neg)} harm), {len(rows) - len(labeled)} neutral excluded")

    stats = ("cedge_cur", "cedge_eq", "ms", "q_box", "q_min")
    print(f"\n{'statistic':<10} {'AUC':>6}   risk/coverage (risk = mean signed delta, lower=better)")
    for s in stats:
        p = np.array([r[s] for r in pos])
        n = np.array([r[s] for r in neg])
        if len(p) and len(n):
            auc = float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())
        else:
            auc = float("nan")
        line = f"{s:<10} {auc:>6.3f}   "
        for cov in (0.5, 0.75, 1.0):
            thr = np.sort([r[s] for r in labeled])[::-1]
            k = max(1, int(round(cov * len(thr)))) - 1
            t = thr[k]
            acc = [r for r in labeled if r[s] >= t]
            risk = np.mean([r["delta"] for r in acc])
            line += f"cov{int(cov*100)}%:{risk:+.4f}  "
        print(line)


if __name__ == "__main__":
    main()
