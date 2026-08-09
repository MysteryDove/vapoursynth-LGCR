#include "lgcr.h"

namespace lgcr {

// BilinAxis + luma structure maps

namespace {

void sobelPixel(const Plane &src, int x, int y, float &gx, float &gy) {
    const float tl = src.at(x - 1, y - 1);
    const float t = src.at(x, y - 1);
    const float tr = src.at(x + 1, y - 1);
    const float l = src.at(x - 1, y);
    const float r = src.at(x + 1, y);
    const float bl = src.at(x - 1, y + 1);
    const float b = src.at(x, y + 1);
    const float br = src.at(x + 1, y + 1);
    gx = (tr + 2.0f * r + br - tl - 2.0f * l - bl) * 0.125f;
    gy = (bl + 2.0f * b + br - tl - 2.0f * t - tr) * 0.125f;
}

template <class Backend>
void sobelRow(const Plane &src, int y, float *gx, float *gy) {
    if (src.w < 3 || src.h < 3 || y == 0 || y + 1 == src.h) {
        for (int x = 0; x < src.w; ++x)
            sobelPixel(src, x, y, gx[x], gy[x]);
        return;
    }

    sobelPixel(src, 0, y, gx[0], gy[0]);
    const auto two = Backend::set1(2.0f);
    const auto scale = Backend::set1(0.125f);
    const float *top = src.row(y - 1);
    const float *mid = src.row(y);
    const float *bot = src.row(y + 1);
    int x = 1;
    for (; x + Backend::lanes <= src.w - 1; x += Backend::lanes) {
        auto vx = Backend::add(Backend::load(top + x + 1),
                               Backend::mul(two, Backend::load(mid + x + 1)));
        vx = Backend::add(vx, Backend::load(bot + x + 1));
        vx = Backend::sub(vx, Backend::load(top + x - 1));
        vx = Backend::sub(vx, Backend::mul(two, Backend::load(mid + x - 1)));
        vx = Backend::sub(vx, Backend::load(bot + x - 1));

        auto vy = Backend::add(Backend::load(bot + x - 1),
                               Backend::mul(two, Backend::load(bot + x)));
        vy = Backend::add(vy, Backend::load(bot + x + 1));
        vy = Backend::sub(vy, Backend::load(top + x - 1));
        vy = Backend::sub(vy, Backend::mul(two, Backend::load(top + x)));
        vy = Backend::sub(vy, Backend::load(top + x + 1));
        Backend::store(gx + x, Backend::mul(vx, scale));
        Backend::store(gy + x, Backend::mul(vy, scale));
    }
    for (; x + 1 < src.w; ++x) {
        gx[x] = (top[x + 1] + 2.0f * mid[x + 1] + bot[x + 1]
               - top[x - 1] - 2.0f * mid[x - 1] - bot[x - 1]) * 0.125f;
        gy[x] = (bot[x - 1] + 2.0f * bot[x] + bot[x + 1]
               - top[x - 1] - 2.0f * top[x] - top[x + 1]) * 0.125f;
    }
    sobelPixel(src, src.w - 1, y, gx[src.w - 1], gy[src.w - 1]);
}

void sobelPlane(const Plane &src, Plane &gx, Plane &gy) {
    for (int y = 0; y < src.h; ++y)
        sobelRow<NativeBackend>(src, y, gx.row(y), gy.row(y));
}

template <class Backend>
void sobelTensor3x3(const Plane &src, Plane &jxx, Plane &jxy, Plane &jyy,
                    PipelineMetrics *metrics) {
    const int w = src.w, h = src.h;
    std::array<std::vector<float>, 3> gxRows, gyRows;
    std::array<int, 3> rowIds{{-1, -1, -1}};
    for (int slot = 0; slot < 3; ++slot) {
        gxRows[slot].resize(w);
        gyRows[slot].resize(w);
    }
    std::vector<float> verticalXX(w), verticalXY(w), verticalYY(w);
    const auto scale = Backend::set1(1.0f / 9.0f);

    auto ensureSobelRow = [&](int sourceY) {
        const int slot = sourceY % 3;
        if (rowIds[slot] != sourceY) {
            ScopedCpuTimer timer(metrics, CpuProfileSlot::GuideSobel);
            sobelRow<Backend>(src, sourceY, gxRows[slot].data(), gyRows[slot].data());
            rowIds[slot] = sourceY;
        }
        return slot;
    };

    for (int y = 0; y < h; ++y) {
        const int slot0 = ensureSobelRow(std::max(0, y - 1));
        const int slot1 = ensureSobelRow(y);
        const int slot2 = ensureSobelRow(std::min(h - 1, y + 1));
        const float *gx0 = gxRows[slot0].data(), *gx1 = gxRows[slot1].data();
        const float *gx2 = gxRows[slot2].data();
        const float *gy0 = gyRows[slot0].data(), *gy1 = gyRows[slot1].data();
        const float *gy2 = gyRows[slot2].data();
        ScopedCpuTimer timer(metrics, CpuProfileSlot::GuideTensor);

        int x = 0;
        for (; x + Backend::lanes <= w; x += Backend::lanes) {
            const auto ax0 = Backend::load(gx0 + x);
            const auto ax1 = Backend::load(gx1 + x);
            const auto ax2 = Backend::load(gx2 + x);
            const auto ay0 = Backend::load(gy0 + x);
            const auto ay1 = Backend::load(gy1 + x);
            const auto ay2 = Backend::load(gy2 + x);
            Backend::store(verticalXX.data() + x,
                Backend::add(Backend::add(Backend::mul(ax0, ax0),
                                          Backend::mul(ax1, ax1)),
                             Backend::mul(ax2, ax2)));
            Backend::store(verticalXY.data() + x,
                Backend::add(Backend::add(Backend::mul(ax0, ay0),
                                          Backend::mul(ax1, ay1)),
                             Backend::mul(ax2, ay2)));
            Backend::store(verticalYY.data() + x,
                Backend::add(Backend::add(Backend::mul(ay0, ay0),
                                          Backend::mul(ay1, ay1)),
                             Backend::mul(ay2, ay2)));
        }
        for (; x < w; ++x) {
            verticalXX[x] = gx0[x] * gx0[x] + gx1[x] * gx1[x] + gx2[x] * gx2[x];
            verticalXY[x] = gx0[x] * gy0[x] + gx1[x] * gy1[x] + gx2[x] * gy2[x];
            verticalYY[x] = gy0[x] * gy0[x] + gy1[x] * gy1[x] + gy2[x] * gy2[x];
        }

        float *outXX = jxx.row(y), *outXY = jxy.row(y), *outYY = jyy.row(y);
        const int rightOfFirst = std::min(1, w - 1);
        outXX[0] = (2.0f * verticalXX[0] + verticalXX[rightOfFirst]) / 9.0f;
        outXY[0] = (2.0f * verticalXY[0] + verticalXY[rightOfFirst]) / 9.0f;
        outYY[0] = (2.0f * verticalYY[0] + verticalYY[rightOfFirst]) / 9.0f;

        x = 1;
        for (; x + Backend::lanes <= w - 1; x += Backend::lanes) {
            Backend::store(outXX + x, Backend::mul(scale,
                Backend::add(Backend::add(Backend::load(verticalXX.data() + x - 1),
                                          Backend::load(verticalXX.data() + x)),
                             Backend::load(verticalXX.data() + x + 1))));
            Backend::store(outXY + x, Backend::mul(scale,
                Backend::add(Backend::add(Backend::load(verticalXY.data() + x - 1),
                                          Backend::load(verticalXY.data() + x)),
                             Backend::load(verticalXY.data() + x + 1))));
            Backend::store(outYY + x, Backend::mul(scale,
                Backend::add(Backend::add(Backend::load(verticalYY.data() + x - 1),
                                          Backend::load(verticalYY.data() + x)),
                             Backend::load(verticalYY.data() + x + 1))));
        }
        for (; x + 1 < w; ++x) {
            outXX[x] = (verticalXX[x - 1] + verticalXX[x] + verticalXX[x + 1]) / 9.0f;
            outXY[x] = (verticalXY[x - 1] + verticalXY[x] + verticalXY[x + 1]) / 9.0f;
            outYY[x] = (verticalYY[x - 1] + verticalYY[x] + verticalYY[x + 1]) / 9.0f;
        }
        if (w > 1) {
            outXX[w - 1] = (verticalXX[w - 2] + 2.0f * verticalXX[w - 1]) / 9.0f;
            outXY[w - 1] = (verticalXY[w - 2] + 2.0f * verticalXY[w - 1]) / 9.0f;
            outYY[w - 1] = (verticalYY[w - 2] + 2.0f * verticalYY[w - 1]) / 9.0f;
        }
    }
}

struct BilinearPoint {
    int x0, x1, y0, y1;
    float fx, fy;
};

BilinearPoint bilinearPoint(int width, int height, double x, double y) {
    const int rawX = static_cast<int>(std::floor(x));
    const int rawY = static_cast<int>(std::floor(y));
    return {
        std::clamp(rawX, 0, width - 1),
        std::clamp(rawX + 1, 0, width - 1),
        std::clamp(rawY, 0, height - 1),
        std::clamp(rawY + 1, 0, height - 1),
        static_cast<float>(x - rawX),
        static_cast<float>(y - rawY),
    };
}

float sampleProjected(const Plane &xPlane, const Plane &yPlane,
                      const BilinearPoint &point, float nx, float ny) {
    const float *xt = xPlane.row(point.y0), *xb = xPlane.row(point.y1);
    const float *yt = yPlane.row(point.y0), *yb = yPlane.row(point.y1);
    const float a = xt[point.x0] * nx + yt[point.x0] * ny;
    const float b = xt[point.x1] * nx + yt[point.x1] * ny;
    const float c = xb[point.x0] * nx + yb[point.x0] * ny;
    const float d = xb[point.x1] * nx + yb[point.x1] * ny;
    return a + (b - a) * point.fx + (c - a) * point.fy +
           (a - b - c + d) * point.fx * point.fy;
}

} // namespace

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

    // Discover the actual two-phase mapping after clamping. This covers all
    // siting modes and avoids assuming a particular centered phase. Only a
    // contiguous AVX2-safe interior is described; borders retain scalar rules.
    auto is2xBlock = [&](int begin) {
        if (begin < 0 || begin + 8 > a.n)
            return false;
        const int base = a.i0[begin];
        if (base < 0 || base + 7 >= srcN)
            return false;
        for (int lane = 0; lane < 8; ++lane) {
            const int relative = a.i0[begin + lane] - base;
            if (relative < 0 || relative + 1 > 7)
                return false;
            if (lane >= 2 &&
                (a.i0[begin + lane] != a.i0[begin + lane - 2] + 1 ||
                 std::fabs(a.f[begin + lane] - a.f[begin + lane - 2]) > 1e-3f))
                return false;
        }
        return true;
    };
    int bestBegin = 0, bestEnd = 0;
    for (int begin = 0; begin + 8 <= a.n; ++begin) {
        if (!is2xBlock(begin))
            continue;
        int end = begin;
        while (is2xBlock(end))
            end += 8;
        if (end - begin > bestEnd - bestBegin) {
            bestBegin = begin;
            bestEnd = end;
        }
    }
    if (bestEnd > bestBegin) {
        a.phase2x.enabled = true;
        a.phase2x.simdBegin = bestBegin;
        a.phase2x.simdEnd = bestEnd;
        a.phase2x.fraction = {{a.f[bestBegin], a.f[bestBegin + 1]}};
    }
    return a;
}


GuideMaps buildGuideMaps(const Plane &structY, const Plane &lcY, int cw, int ch,
                         double rw, double rh, double shiftX, double shiftY,
                         PipelineMetrics *metrics) {
    const Plane &y = structY; // structure tensor / db built in OUTPUT space
    GuideMaps m;
    m.jxx = Plane(y.w, y.h);
    m.jxy = Plane(y.w, y.h);
    m.jyy = Plane(y.w, y.h);

    // The fused implementation retains only three Sobel rows. Float storage
    // points and the clamped border calculation match the former two-pass path.
    sobelTensor3x3<NativeBackend>(y, m.jxx, m.jxy, m.jyy, metrics);

    {
        ScopedCpuTimer timer(metrics, CpuProfileSlot::GuideLcMap);
        m.lc = buildLcMap(lcY, cw, ch, rw, rh, shiftX, shiftY);
    }
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
    // Separable binary dilation with a rolling population count: O(pixels)
    // regardless of support radius.
    std::vector<uint8_t> tmp(size_t(outW) * outH, 0);
    const int r = dilateRadius;
    for (int y = 0; y < outH; ++y) {
        const uint8_t *source = mask.data() + size_t(y) * outW;
        uint8_t *target = tmp.data() + size_t(y) * outW;
        int active = 0;
        for (int x = 0; x <= std::min(outW - 1, r); ++x)
            active += source[x] != 0;
        for (int x = 0; x < outW; ++x) {
            target[x] = active != 0;
            const int remove = x - r;
            const int add = x + r + 1;
            if (remove >= 0)
                active -= source[remove] != 0;
            if (add < outW)
                active += source[add] != 0;
        }
    }
    std::vector<int> columnCounts(outW, 0);
    for (int y = 0; y <= std::min(outH - 1, r); ++y) {
        const uint8_t *row = tmp.data() + size_t(y) * outW;
        for (int x = 0; x < outW; ++x)
            columnCounts[x] += row[x] != 0;
    }
    for (int y = 0; y < outH; ++y) {
        uint8_t *target = mask.data() + size_t(y) * outW;
        for (int x = 0; x < outW; ++x)
            target[x] = columnCounts[x] != 0;
        const int remove = y - r;
        const int add = y + r + 1;
        const uint8_t *removeRow = remove >= 0
            ? tmp.data() + size_t(remove) * outW : nullptr;
        const uint8_t *addRow = add < outH
            ? tmp.data() + size_t(add) * outW : nullptr;
        for (int x = 0; x < outW; ++x) {
            if (removeRow)
                columnCounts[x] -= removeRow[x] != 0;
            if (addRow)
                columnCounts[x] += addRow[x] != 0;
        }
    }
    return mask;
}

// Footprint-averaged luma at each chroma sample (used per-frame by TRecon).
Plane buildLcMap(const Plane &lcY, int cw, int ch, double rw, double rh,
                 double shiftX, double shiftY) {
    Plane lc(cw, ch);
    std::vector<int> xBegin(cw), xEnd(cw), yBegin(ch), yEnd(ch);
    for (int cx = 0; cx < cw; ++cx) {
        const double lx = (cx + 0.5) * rw - 0.5 + shiftX;
        int first = static_cast<int>(std::ceil(lx - 0.5 * rw));
        int last = static_cast<int>(std::ceil(lx + 0.5 * rw));
        first = std::clamp(first, 0, lcY.w - 1);
        last = std::clamp(last, first + 1, lcY.w);
        xBegin[cx] = first;
        xEnd[cx] = last;
    }
    for (int cy = 0; cy < ch; ++cy) {
        const double ly = (cy + 0.5) * rh - 0.5 + shiftY;
        int first = static_cast<int>(std::ceil(ly - 0.5 * rh));
        int last = static_cast<int>(std::ceil(ly + 0.5 * rh));
        first = std::clamp(first, 0, lcY.h - 1);
        last = std::clamp(last, first + 1, lcY.h);
        yBegin[cy] = first;
        yEnd[cy] = last;
    }

    // Chroma-res luma level: average of the luma footprint of each chroma sample
    for (int cy = 0; cy < ch; ++cy) {
        int cx = 0;
#ifdef __AVX2__
        if (yEnd[cy] - yBegin[cy] == 2) {
            const float *row0 = lcY.row(yBegin[cy]);
            const float *row1 = lcY.row(yBegin[cy] + 1);
            const __m256 quarter = _mm256_set1_ps(0.25f);
            const __m256i order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
            for (; cx + 8 <= cw; ) {
                const int base = xBegin[cx];
                bool contiguousPairs = base >= 0 && base + 15 < lcY.w;
                for (int lane = 0; lane < 8 && contiguousPairs; ++lane)
                    contiguousPairs = xBegin[cx + lane] == base + 2 * lane &&
                                      xEnd[cx + lane] == base + 2 * lane + 2;
                if (!contiguousPairs) {
                    double sumL = 0.0;
                    for (int j = yBegin[cy]; j < yEnd[cy]; ++j)
                        for (int i = xBegin[cx]; i < xEnd[cx]; ++i)
                            sumL += lcY.row(j)[i];
                    lc.row(cy)[cx] = static_cast<float>(sumL /
                        (double(xEnd[cx] - xBegin[cx]) *
                         (yEnd[cy] - yBegin[cy])));
                    ++cx;
                    continue;
                }
                const __m256 lo = _mm256_add_ps(
                    _mm256_loadu_ps(row0 + base), _mm256_loadu_ps(row1 + base));
                const __m256 hi = _mm256_add_ps(
                    _mm256_loadu_ps(row0 + base + 8),
                    _mm256_loadu_ps(row1 + base + 8));
                const __m256 pairs = _mm256_permutevar8x32_ps(
                    _mm256_hadd_ps(lo, hi), order);
                _mm256_storeu_ps(lc.row(cy) + cx, _mm256_mul_ps(pairs, quarter));
                cx += 8;
            }
        }
#endif
        for (; cx < cw; ++cx) {
            double sumL = 0;
            for (int j = yBegin[cy]; j < yEnd[cy]; ++j) {
                const float *source = lcY.row(j);
                for (int i = xBegin[cx]; i < xEnd[cx]; ++i)
                    sumL += source[i];
            }
            lc.row(cy)[cx] = static_cast<float>(
                sumL / (double(xEnd[cx] - xBegin[cx]) * (yEnd[cy] - yBegin[cy])));
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
    auto maps = buildYcMaps(nullptr, Y, cw, ch, rw, rh, shiftX, shiftY, nullptr);
    return std::move(maps[std::clamp(kind, 0, 2)]);
}

std::array<Plane, 3> buildYcMaps(const LGCRData *owner, const Plane &Y, int cw, int ch,
                                 double rw, double rh,
                                 double shiftX, double shiftY,
                                 PipelineMetrics *metrics) {
    std::array<Plane, 3> out;
    {
        ScopedCpuTimer timer(metrics, CpuProfileSlot::AffineCandidateBox);
        out[0] = buildLcMap(Y, cw, ch, rw, rh, shiftX, shiftY);
    }

    Plane tmp(cw, Y.h);
    auto resampleCandidate = [&](Kernel kernel, double p1, double p2,
                                 double support, Plane &dst) {
        LGCRData candidate = owner ? *owner : LGCRData{};
        candidate.kernel = kernel;
        candidate.kp1 = p1;
        candidate.kp2 = p2;
        candidate.support = support;
        candidate.radial = false;
        const auto horizontal = cachedWeights(&candidate, Y.w, cw, shiftX);
        const auto vertical = cachedWeights(&candidate, Y.h, ch, shiftY);
        dst = Plane(cw, ch);
        resampleH(Y, tmp, *horizontal);
        resampleV(tmp, dst, *vertical);
    };

    {
        ScopedCpuTimer timer(metrics, CpuProfileSlot::AffineCandidateBilinear);
        resampleCandidate(Kernel::Bilinear, 0.0, 0.0, 1.0, out[1]);
    }
    {
        ScopedCpuTimer timer(metrics, CpuProfileSlot::AffineCandidateBicubic);
        resampleCandidate(Kernel::Bicubic, 0.0, 0.6, 2.0, out[2]);
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
Plane buildMutualGate(const Plane &lc, const Plane &U, const Plane &V, double sigma,
                      const uint8_t *activeMask, int activeStride) {
    (void)sigma; // reserved for a future noise-adaptive profile floor
    const int w = lc.w, h = lc.h;
    Plane gx(w, h), gy(w, h), ux(w, h), uy(w, h), vx(w, h), vy(w, h);
    sobelPlane(lc, gx, gy);
    sobelPlane(U, ux, uy);
    sobelPlane(V, vx, vy);

    Plane gate(w, h);
    if (activeMask)
        std::fill(gate.px.begin(), gate.px.end(), 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (activeMask && activeMask[size_t(y) * activeStride + x] == 0)
                continue;
            // luma edge normal from the 3x3-smoothed structure tensor
            float sxx = 0, sxy = 0, syy = 0;
            if (x > 0 && x + 1 < w && y > 0 && y + 1 < h) {
                for (int dy = -1; dy <= 1; ++dy) {
                    const float *rx = gx.row(y + dy) + x - 1;
                    const float *ry = gy.row(y + dy) + x - 1;
                    for (int dx = 0; dx < 3; ++dx) {
                        const float a = rx[dx], b = ry[dx];
                        sxx += a * a; sxy += a * b; syy += b * b;
                    }
                }
            } else {
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const float a = gx.at(x + dx, y + dy);
                        const float b = gy.at(x + dx, y + dy);
                        sxx += a * a; sxy += a * b; syy += b * b;
                    }
            }
            const float jsum = sxx + syy;
            if (jsum < 1e-10f) {
                gate.at(x, y) = 0.0f;
                continue;
            }
            const TensorDirection direction = principalTensorDirection(sxx, sxy, syy);
            const float nx = direction.nx, ny = direction.ny;

            float sumY = 0, sqY = 0, momY = 0;
            float sumC = 0, sqC = 0, momC = 0;
            for (int k = -3; k <= 3; ++k) {
                const double px = x + k * nx, py = y + k * ny;
                const BilinearPoint point = bilinearPoint(w, h, px, py);
                const float gY = std::fabs(sampleProjected(gx, gy, point, nx, ny));
                const float au = sampleProjected(ux, uy, point, nx, ny);
                const float av = sampleProjected(vx, vy, point, nx, ny);
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
