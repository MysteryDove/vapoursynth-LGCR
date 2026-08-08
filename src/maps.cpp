#include "lgcr.h"

namespace lgcr {

// BilinAxis + luma structure maps

BilinAxis buildBilinAxis(const std::vector<float> &pos, int srcN) {
    BilinAxis a;
    a.n = static_cast<int>(pos.size());
    a.i0.resize(a.n);
    a.f.resize(a.n);
    if (srcN == 1) {
        std::fill(a.i0.begin(), a.i0.end(), 0);
        std::fill(a.f.begin(), a.f.end(), 0.0f);
        return a;
    }
    for (int i = 0; i < a.n; ++i) {
        float x = pos[i];
        int j = static_cast<int>(std::floor(x));
        float fr = x - j;
        if (j < 0) { j = 0; fr = 0.0f; }
        if (j > srcN - 2) { j = srcN - 2; fr = 1.0f; }
        a.i0[i] = j;
        a.f[i] = fr;
    }
    return a;
}


GuideMaps buildGuideMaps(const Plane &structY, const Plane &lcY, int cw, int ch,
                                double rw, double rh, double shiftX, double shiftY, bool needDb) {
    const Plane &y = structY; // structure tensor / db built in OUTPUT space
    GuideMaps m;
    m.jxx = Plane(y.w, y.h);
    m.jxy = Plane(y.w, y.h);
    m.jyy = Plane(y.w, y.h);
    if (needDb)
        m.db = Plane(y.w, y.h);
    m.lc = Plane(cw, ch);

    // Full-resolution Sobel gradients
    Plane gx(y.w, y.h), gy(y.w, y.h);
    for (int j = 0; j < y.h; ++j) {
        for (int i = 0; i < y.w; ++i) {
            const float tl = y.at(i - 1, j - 1), t = y.at(i, j - 1), tr = y.at(i + 1, j - 1);
            const float l = y.at(i - 1, j), r = y.at(i + 1, j);
            const float bl = y.at(i - 1, j + 1), b = y.at(i, j + 1), br = y.at(i + 1, j + 1);
            gx.at(i, j) = (tr + 2 * r + br - tl - 2 * l - bl) * 0.125f;
            gy.at(i, j) = (bl + 2 * b + br - tl - 2 * t - tr) * 0.125f;

            if (needDb) {
                // algo=1 baseline: median |4-neighbor diff|. ~0 along an
                // axis-aligned step, = slope on a ramp. (Known limitation:
                // overestimates on diagonal steps — fixed in algo=2.)
                const float c0 = y.at(i, j);
                float diffs[4] = {
                    std::fabs(r - c0), std::fabs(l - c0),
                    std::fabs(b - c0), std::fabs(t - c0),
                };
                std::sort(diffs, diffs + 4);
                m.db.at(i, j) = (diffs[1] + diffs[2]) * 0.5f;
            }
        }
    }
    // Structure tensor: 3x3 box-smoothed outer products. The dominant
    // eigenvector gives the edge normal; coherence (lambda1-lambda2)/
    // (lambda1+lambda2) is a noise-robust edge-direction confidence.
    for (int j = 0; j < y.h; ++j) {
        for (int i = 0; i < y.w; ++i) {
            float sxx = 0, sxy = 0, syy = 0;
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) {
                    const float a = gx.at(i + di, j + dj), b = gy.at(i + di, j + dj);
                    sxx += a * a; sxy += a * b; syy += b * b;
                }
            m.jxx.at(i, j) = sxx / 9.0f;
            m.jxy.at(i, j) = sxy / 9.0f;
            m.jyy.at(i, j) = syy / 9.0f;
        }
    }

    m.lc = buildLcMap(lcY, cw, ch, rw, rh, shiftX, shiftY);
    return m;
}

// Sparse trust mask over the OUTPUT grid: active where the structure-tensor
// energy exceeds the noise floor, dilated to cover the full support window.
std::vector<uint8_t> buildTrustMask(const GuideMaps &gm, int outW, int outH,
                                    double sigma, int dilateRadius) {
    std::vector<uint8_t> mask(size_t(outW) * outH, 0);
    const float eth = float(sigma * sigma); // ~ luma step R >= sigma is worth guiding
    for (int y = 0; y < outH; ++y)
        for (int x = 0; x < outW; ++x)
            if (gm.jxx.at(x, y) + gm.jyy.at(x, y) > eth)
                mask[size_t(y) * outW + x] = 1;
    // separable dilation (sliding-window max)
    std::vector<uint8_t> tmp(size_t(outW) * outH, 0);
    const int r = dilateRadius;
    for (int y = 0; y < outH; ++y)
        for (int x = 0; x < outW; ++x) {
            uint8_t m = 0;
            for (int i = std::max(0, x - r); i <= std::min(outW - 1, x + r); ++i)
                m |= mask[size_t(y) * outW + i];
            tmp[size_t(y) * outW + x] = m;
        }
    for (int y = 0; y < outH; ++y)
        for (int x = 0; x < outW; ++x) {
            uint8_t m = 0;
            for (int j = std::max(0, y - r); j <= std::min(outH - 1, y + r); ++j)
                m |= tmp[size_t(j) * outW + x];
            mask[size_t(y) * outW + x] = m;
        }
    return mask;
}

// Footprint-averaged luma at each chroma sample (used per-frame by TRecon).
Plane buildLcMap(const Plane &lcY, int cw, int ch, double rw, double rh,
                 double shiftX, double shiftY) {
    Plane lc(cw, ch);
    // Chroma-res luma level: average of the luma footprint of each chroma sample
    for (int cy = 0; cy < ch; ++cy) {
        for (int cx = 0; cx < cw; ++cx) {
            // Sited luma position of this chroma sample
            const double lx = (cx + 0.5) * rw - 0.5 + shiftX;
            const double ly = (cy + 0.5) * rh - 0.5 + shiftY;
            // Footprint: luma samples whose centers fall in [l - r/2, l + r/2)
            int x0 = static_cast<int>(std::ceil(lx - 0.5 * rw));
            int x1 = static_cast<int>(std::ceil(lx + 0.5 * rw)); // exclusive
            int y0 = static_cast<int>(std::ceil(ly - 0.5 * rh));
            int y1 = static_cast<int>(std::ceil(ly + 0.5 * rh));
            x0 = std::clamp(x0, 0, lcY.w - 1);
            x1 = std::clamp(x1, x0 + 1, lcY.w);
            y0 = std::clamp(y0, 0, lcY.h - 1);
            y1 = std::clamp(y1, y0 + 1, lcY.h);

            double sumL = 0;
            for (int j = y0; j < y1; ++j)
                for (int i = x0; i < x1; ++i)
                    sumL += lcY.at(i, j);
            lc.at(cx, cy) = static_cast<float>(sumL / (double(x1 - x0) * (y1 - y0)));
        }
    }
    return lc;
}

// Candidate encoder degradation D^(Y): luma resampled to the chroma grid with
// a specific kernel (0=box footprint, 1=triangle/bilinear, 2=bicubic b0c0.6),
// honoring siting. Used by the affine-credibility machinery (algo=6): the
// true encoder kernel is unknown, so statistics are computed per candidate
// and only accepted when they agree (multi-kernel stability).
Plane buildYcMap(const Plane &Y, int cw, int ch, double rw, double rh,
                 double shiftX, double shiftY, int kind) {
    if (kind == 0)
        return buildLcMap(Y, cw, ch, rw, rh, shiftX, shiftY);
    const double sup = (kind == 1) ? 1.0 : 2.0; // kernel support in chroma units
    Plane out(cw, ch);
    for (int cy = 0; cy < ch; ++cy)
        for (int cx = 0; cx < cw; ++cx) {
            const double lx = (cx + 0.5) * rw - 0.5 + shiftX;
            const double ly = (cy + 0.5) * rh - 0.5 + shiftY;
            const int x0 = int(std::floor(lx - sup * rw)), x1 = int(std::ceil(lx + sup * rw));
            const int y0 = int(std::floor(ly - sup * rh)), y1 = int(std::ceil(ly + sup * rh));
            double sum = 0, wsum = 0;
            for (int j = y0; j <= y1; ++j) {
                const double dy = std::fabs(j - ly) / rh;
                double wy;
                if (kind == 1)
                    wy = std::max(0.0, 1.0 - dy);
                else
                    wy = kernelEval(Kernel::Bicubic, dy, 0.0, 0.6);
                if (wy == 0.0)
                    continue;
                for (int i = x0; i <= x1; ++i) {
                    const double dx = std::fabs(i - lx) / rw;
                    double wx;
                    if (kind == 1)
                        wx = std::max(0.0, 1.0 - dx);
                    else
                        wx = kernelEval(Kernel::Bicubic, dx, 0.0, 0.6);
                    if (wx == 0.0)
                        continue;
                    const double w = wx * wy;
                    sum += w * Y.at(i, j);
                    wsum += w;
                }
            }
            out.at(cx, cy) = static_cast<float>(wsum > 0.0 ? sum / wsum : Y.at(int(lx), int(ly)));
        }
    return out;
}

// ---------------------------------------------------------------------------
// Mutual-structure co-edge gate (review item 5)
//
// The luma guide may only transfer high-frequency detail into chroma where
// the chroma planes CONFIRM a co-located edge. Both sides' gradient-magnitude
// profiles are measured along the luma edge normal over +/-3 chroma px, and
// compared in shape:
//
//   width:  participation ratio p = (sum g)^2 / sum(g^2)  (~number of taps
//           the edge is spread over). Chroma much WIDER than luma (soft
//           blend under a hard luma edge, the hardL_softC failure) must not
//           receive luma high frequencies. The other direction (chroma
//           sharper than luma) is harmless and passes.
//   phase:  |centroid(gY) - centroid(gC)| in chroma px. A chroma edge
//           shifted from the luma edge (encoder misalignment) is not
//           co-located; transferring would invent an edge at the wrong place.
//
// The luma side uses the lc map (footprint box = the candidate 420 encoder
// filter), so the comparison happens at the resolution where chroma
// information actually exists. Measured separation on the battery:
// hard_v/d45/ramp ratio=1.00 dcent=0; hardL_softC ratio=1.8; misalign4
// dcent=2.0.
Plane buildMutualGate(const Plane &lc, const Plane &U, const Plane &V, double sigma) {
    (void)sigma; // reserved for a future noise-adaptive profile floor
    const int w = lc.w, h = lc.h;
    Plane gx(w, h), gy(w, h), ux(w, h), uy(w, h), vx(w, h), vy(w, h);
    auto sobel = [](const Plane &p, Plane &ox, Plane &oy) {
        for (int j = 0; j < p.h; ++j)
            for (int i = 0; i < p.w; ++i) {
                const float tl = p.at(i - 1, j - 1), t = p.at(i, j - 1), tr = p.at(i + 1, j - 1);
                const float l = p.at(i - 1, j), r = p.at(i + 1, j);
                const float bl = p.at(i - 1, j + 1), b = p.at(i, j + 1), br = p.at(i + 1, j + 1);
                ox.at(i, j) = (tr + 2 * r + br - tl - 2 * l - bl) * 0.125f;
                oy.at(i, j) = (bl + 2 * b + br - tl - 2 * t - tr) * 0.125f;
            }
    };
    sobel(lc, gx, gy);
    sobel(U, ux, uy);
    sobel(V, vx, vy);

    Plane gate(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // luma edge normal from the 3x3-smoothed structure tensor
            float sxx = 0, sxy = 0, syy = 0;
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) {
                    const float a = gx.at(x + di, y + dj);
                    const float b = gy.at(x + di, y + dj);
                    sxx += a * a;
                    sxy += a * b;
                    syy += b * b;
                }
            const float jdiff = sxx - syy, jsum = sxx + syy;
            if (jsum < 1e-10f) {
                gate.at(x, y) = 0.0f;
                continue;
            }
            const float theta = 0.5f * std::atan2(2.0f * sxy, jdiff);
            const float nx = std::cos(theta), ny = std::sin(theta);

            float sumY = 0, sqY = 0, momY = 0;
            float sumC = 0, sqC = 0, momC = 0;
            for (int k = -3; k <= 3; ++k) {
                const double px = x + k * nx, py = y + k * ny;
                const float gY = std::fabs(bilinear(gx, px, py) * nx + bilinear(gy, px, py) * ny);
                const float au = bilinear(ux, px, py) * nx + bilinear(uy, px, py) * ny;
                const float av = bilinear(vx, px, py) * nx + bilinear(vy, px, py) * ny;
                const float gC = std::sqrt(au * au + av * av);
                sumY += gY; sqY += gY * gY; momY += k * gY;
                sumC += gC; sqC += gC * gC; momC += k * gC;
            }
            // (No absolute energy gates: on flat luma the guide is inert
            // anyway, and on flat chroma every weighting gives the same
            // average — an energy floor can only CAP legitimate weak edges.)
            if (sumY < 1e-9f || sumC < 1e-9f) {
                gate.at(x, y) = 0.0f;
                continue;
            }
            const float partY = sumY * sumY / (sqY + 1e-12f);
            const float partC = sumC * sumC / (sqC + 1e-12f);
            const float ratio = partC / (partY + 1e-6f);
            const float dcent = std::fabs(momY / sumY - momC / sumC);
            const float tw = std::clamp((1.6f - ratio) / 0.4f, 0.0f, 1.0f);
            const float tp = std::clamp((1.0f - dcent) / 0.5f, 0.0f, 1.0f);
            gate.at(x, y) = (tw * tw * (3.0f - 2.0f * tw)) * (tp * tp * (3.0f - 2.0f * tp));
        }
    }
    return gate;
}

} // namespace lgcr
