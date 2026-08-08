#include "lgcr.h"

namespace lgcr {

// BilinAxis + luma structure maps

BilinAxis buildBilinAxis(const std::vector<float> &pos, int srcN) {
    BilinAxis a;
    a.n = static_cast<int>(pos.size());
    a.i0.resize(a.n);
    a.f.resize(a.n);
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

} // namespace lgcr
