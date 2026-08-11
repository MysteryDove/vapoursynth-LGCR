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

template <class Backend>
void sobelTensor3x3(const Plane &src, Plane &jxx, Plane &jxy, Plane &jyy,
                    PipelineMetrics *metrics, uint8_t *trustSeed,
                    float trustThreshold) {
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
        const auto tensorStart = std::chrono::steady_clock::now();

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
        if (metrics) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - tensorStart).count();
            metrics->add(CpuProfileSlot::GuideTensor,
                         static_cast<uint64_t>(elapsed));
        }
        if (trustSeed) {
            ScopedCpuTimer seedTimer(metrics, CpuProfileSlot::TrustSeed);
            uint8_t *seed = trustSeed + size_t(y) * w;
            for (int sx = 0; sx < w; ++sx)
                seed[sx] = outXX[sx] + outYY[sx] > trustThreshold;
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
                         PipelineMetrics *metrics, double trustSigma,
                         FrameScratchAllocator *scratch) {
    const Plane &y = structY; // structure tensor / db built in OUTPUT space
    GuideMaps m;
    m.jxx = scratchPlane(scratch, y.w, y.h);
    m.jxy = scratchPlane(scratch, y.w, y.h);
    m.jyy = scratchPlane(scratch, y.w, y.h);

    // The fused implementation retains only three Sobel rows. Float storage
    // points and the clamped border calculation match the former two-pass path.
    if (trustSigma >= 0.0)
        m.trustSeed.resize(size_t(y.w) * y.h);
    sobelTensor3x3<NativeBackend>(
        y, m.jxx, m.jxy, m.jyy, metrics,
        m.trustSeed.empty() ? nullptr : m.trustSeed.data(),
        float(trustSigma * trustSigma));

    {
        ScopedCpuTimer timer(metrics, CpuProfileSlot::GuideLcMap);
        m.lc = buildLcMap(lcY, cw, ch, rw, rh, shiftX, shiftY, scratch);
    }
    return m;
}

// Sparse trust mask over the OUTPUT grid: active where the structure-tensor
// energy exceeds the noise floor, dilated to cover the full support window.
std::vector<uint8_t> buildTrustMask(const GuideMaps &gm, int outW, int outH,
                                    double sigma, int dilateRadius,
                                    PipelineMetrics *metrics) {
    std::vector<uint8_t> mask;
    const float eth = float(sigma * sigma); // ~ luma step R >= sigma is worth guiding
    if (gm.trustSeed.size() == size_t(outW) * outH) {
        mask = gm.trustSeed;
    } else {
        ScopedCpuTimer timer(metrics, CpuProfileSlot::TrustSeed);
        mask.assign(size_t(outW) * outH, 0);
        for (int y = 0; y < outH; ++y)
            for (int x = 0; x < outW; ++x)
                if (gm.jxx.at(x, y) + gm.jyy.at(x, y) > eth)
                    mask[size_t(y) * outW + x] = 1;
    }
    // Separable binary dilation with a rolling population count: O(pixels)
    // regardless of support radius.
    ScopedCpuTimer dilateTimer(metrics, CpuProfileSlot::TrustDilate);
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

SparseWorkset buildSparseWorkset(std::vector<uint8_t> mask, int maskWidth,
                                 int maskHeight, int outputWidth, int outputHeight,
                                 const ChromaAxis &ax, const ChromaAxis &ay,
                                 int chromaWidth, int chromaHeight) {
    SparseWorkset workset;
    workset.maskWidth = maskWidth;
    workset.maskHeight = maskHeight;
    workset.outputWidth = outputWidth;
    workset.outputHeight = outputHeight;
    workset.chromaWidth = chromaWidth;
    workset.chromaHeight = chromaHeight;
    workset.mask = std::move(mask);
    workset.outputRowOffsets.resize(size_t(outputHeight) + 1);
    workset.outputIndexRowOffsets.resize(size_t(outputHeight) + 1);

    std::vector<int> maskX(outputWidth);
    for (int x = 0; x < outputWidth; ++x)
        maskX[x] = std::min(maskWidth - 1,
                            int(int64_t(x) * maskWidth / outputWidth));

    for (int y = 0; y < outputHeight; ++y) {
        workset.outputRowOffsets[y] = workset.outputSpans.size();
        workset.outputIndexRowOffsets[y] = workset.outputIndices.size();
        const int maskY = std::min(maskHeight - 1,
                                   int(int64_t(y) * maskHeight / outputHeight));
        const uint8_t *maskRow = workset.mask.data() + size_t(maskY) * maskWidth;
        int x = 0;
        while (x < outputWidth) {
            while (x < outputWidth && maskRow[maskX[x]] == 0)
                ++x;
            const int begin = x;
            while (x < outputWidth && maskRow[maskX[x]] != 0) {
                workset.outputIndices.push_back(uint32_t(size_t(y) * outputWidth + x));
                ++x;
            }
            if (begin < x)
                workset.outputSpans.push_back({begin, x});
        }
    }
    workset.outputRowOffsets[outputHeight] = workset.outputSpans.size();
    workset.outputIndexRowOffsets[outputHeight] = workset.outputIndices.size();
    workset.activeOutputPixels = workset.outputIndices.size();

    workset.chromaMask.assign(size_t(chromaWidth) * chromaHeight, 0);
    for (int y = 0; y < outputHeight; ++y) {
        const int cy0 = ay.chromaBilin.i0[y];
        const int cy1 = std::min(cy0 + 1, chromaHeight - 1);
        uint8_t *active0 = workset.chromaMask.data() + size_t(cy0) * chromaWidth;
        uint8_t *active1 = workset.chromaMask.data() + size_t(cy1) * chromaWidth;
        for (size_t spanIndex = workset.outputRowOffsets[y];
             spanIndex < workset.outputRowOffsets[y + 1]; ++spanIndex) {
            const SparseSpan span = workset.outputSpans[spanIndex];
            for (int x = span.begin; x < span.end; ++x) {
                const int cx0 = ax.chromaBilin.i0[x];
                const int cx1 = std::min(cx0 + 1, chromaWidth - 1);
                active0[cx0] = active0[cx1] = 1;
                active1[cx0] = active1[cx1] = 1;
            }
        }
    }

    workset.chromaRowOffsets.resize(size_t(chromaHeight) + 1);
    for (int y = 0; y < chromaHeight; ++y) {
        workset.chromaRowOffsets[y] = workset.chromaSpans.size();
        const uint8_t *row = workset.chromaMask.data() + size_t(y) * chromaWidth;
        int x = 0;
        while (x < chromaWidth) {
            while (x < chromaWidth && row[x] == 0)
                ++x;
            const int begin = x;
            while (x < chromaWidth && row[x] != 0) {
                ++workset.activeChromaPixels;
                ++x;
            }
            if (begin < x)
                workset.chromaSpans.push_back({begin, x});
        }
    }
    workset.chromaRowOffsets[chromaHeight] = workset.chromaSpans.size();
    return workset;
}

// Footprint-averaged luma at each chroma sample (used per-frame by TRecon).
Plane buildLcMap(const Plane &lcY, int cw, int ch, double rw, double rh,
                 double shiftX, double shiftY,
                 FrameScratchAllocator *scratch) {
    Plane lc = scratchPlane(scratch, cw, ch);
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

    auto filtered = buildYcFilteredMaps(
        owner, Y, cw, ch, rw, rh, shiftX, shiftY, metrics);
    out[1] = std::move(filtered[0]);
    out[2] = std::move(filtered[1]);
    return out;
}

static std::array<Plane, 2> buildYcFiltered420Pair(
    const LGCRData *owner, const Plane &Y, int cw, int ch,
    double shiftX, double shiftY, FrameScratchAllocator *scratch) {
    LGCRData bilinear = owner ? *owner : LGCRData{};
    bilinear.kernel = Kernel::Bilinear;
    bilinear.kp1 = bilinear.kp2 = 0.0;
    bilinear.support = 1.0;
    bilinear.radial = false;
    LGCRData bicubic = owner ? *owner : LGCRData{};
    bicubic.kernel = Kernel::Bicubic;
    bicubic.kp1 = 0.0;
    bicubic.kp2 = 0.6;
    bicubic.support = 2.0;
    bicubic.radial = false;

    const auto hx0 = cachedWeights(&bilinear, Y.w, cw, shiftX);
    const auto hy0 = cachedWeights(&bilinear, Y.h, ch, shiftY);
    const auto hx1 = cachedWeights(&bicubic, Y.w, cw, shiftX);
    const auto hy1 = cachedWeights(&bicubic, Y.h, ch, shiftY);
    std::array<Plane, 2> out{{scratchPlane(scratch, cw, ch),
                              scratchPlane(scratch, cw, ch)}};

    struct RowCache {
        Plane values;
        std::vector<int> ids;
        RowCache(int width, int rows, FrameScratchAllocator *scratch)
            : values(scratchPlane(scratch, width, rows)), ids(rows, -1) {}
    } cache0(cw, std::max(1, hx0->sup), scratch),
      cache1(cw, std::max(1, hx1->sup), scratch);

    auto horizontalRow = [&](const WeightTable &table, RowCache &cache,
                             int sourceY) -> const float * {
        sourceY = std::clamp(sourceY, 0, Y.h - 1);
        const int slot = sourceY % cache.values.h;
        if (cache.ids[slot] == sourceY)
            return cache.values.row(slot);
        const float *source = Y.row(sourceY);
        float *target = cache.values.row(slot);
        for (int x = 0; x < cw; ++x) {
            const float *weights = table.w.data() + size_t(x) * table.sup;
            const int start = table.start[x];
            double value = 0.0;
            int tap = 0;
            if (start >= 0 && start + table.sup <= Y.w) {
#ifdef __AVX2__
                __m256 accumulated = _mm256_setzero_ps();
                for (; tap + 8 <= table.sup; tap += 8)
                    accumulated = _mm256_fmadd_ps(
                        _mm256_loadu_ps(weights + tap),
                        _mm256_loadu_ps(source + start + tap), accumulated);
                value = NativeBackend::horizontalSum(accumulated);
#endif
                for (; tap < table.sup; ++tap)
                    value += double(weights[tap]) * source[start + tap];
            } else {
                for (; tap < table.sup; ++tap)
                    value += double(weights[tap]) *
                        source[std::clamp(start + tap, 0, Y.w - 1)];
            }
            target[x] = static_cast<float>(value);
        }
        cache.ids[slot] = sourceY;
        return target;
    };

    std::vector<const float *> rows0(hy0->sup), rows1(hy1->sup);
    for (int y = 0; y < ch; ++y) {
        const float *weights0 = hy0->w.data() + size_t(y) * hy0->sup;
        const float *weights1 = hy1->w.data() + size_t(y) * hy1->sup;
        for (int tap = 0; tap < hy0->sup; ++tap)
            rows0[tap] = horizontalRow(*hx0, cache0, hy0->start[y] + tap);
        for (int tap = 0; tap < hy1->sup; ++tap)
            rows1[tap] = horizontalRow(*hx1, cache1, hy1->start[y] + tap);

        float *target0 = out[0].row(y), *target1 = out[1].row(y);
        int x = 0;
#ifdef __AVX2__
        for (; x + 8 <= cw; x += 8) {
            __m256 value0 = _mm256_setzero_ps();
            __m256 value1 = _mm256_setzero_ps();
            for (int tap = 0; tap < hy0->sup; ++tap)
                value0 = _mm256_fmadd_ps(
                    _mm256_set1_ps(weights0[tap]),
                    _mm256_loadu_ps(rows0[tap] + x), value0);
            for (int tap = 0; tap < hy1->sup; ++tap)
                value1 = _mm256_fmadd_ps(
                    _mm256_set1_ps(weights1[tap]),
                    _mm256_loadu_ps(rows1[tap] + x), value1);
            _mm256_storeu_ps(target0 + x, value0);
            _mm256_storeu_ps(target1 + x, value1);
        }
#endif
        for (; x < cw; ++x) {
            double value0 = 0.0, value1 = 0.0;
            for (int tap = 0; tap < hy0->sup; ++tap)
                value0 += double(weights0[tap]) * rows0[tap][x];
            for (int tap = 0; tap < hy1->sup; ++tap)
                value1 += double(weights1[tap]) * rows1[tap][x];
            target0[x] = static_cast<float>(value0);
            target1[x] = static_cast<float>(value1);
        }
    }
    return out;
}

std::array<Plane, 2> buildYcFilteredMaps(
    const LGCRData *owner, const Plane &Y, int cw, int ch, double rw, double rh,
    double shiftX, double shiftY, PipelineMetrics *metrics,
    FrameScratchAllocator *scratch) {
    if (Y.w == 2 * cw && Y.h == 2 * ch &&
        std::fabs(rw - 2.0) < 1e-12 && std::fabs(rh - 2.0) < 1e-12) {
        const auto start = std::chrono::steady_clock::now();
        auto out = buildYcFiltered420Pair(
            owner, Y, cw, ch, shiftX, shiftY, scratch);
        if (metrics) {
            const uint64_t elapsed = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start).count());
            metrics->add(CpuProfileSlot::AffineCandidateBilinear, elapsed / 2);
            metrics->add(CpuProfileSlot::AffineCandidateBicubic,
                         elapsed - elapsed / 2);
        }
        return out;
    }
    std::array<Plane, 2> out;

    Plane tmp = scratchPlane(scratch, cw, Y.h);
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
        dst = scratchPlane(scratch, cw, ch);
        resampleH(Y, tmp, *horizontal);
        resampleV(tmp, dst, *vertical);
    };

    {
        ScopedCpuTimer timer(metrics, CpuProfileSlot::AffineCandidateBilinear);
        resampleCandidate(Kernel::Bilinear, 0.0, 0.0, 1.0, out[0]);
    }
    {
        ScopedCpuTimer timer(metrics, CpuProfileSlot::AffineCandidateBicubic);
        resampleCandidate(Kernel::Bicubic, 0.0, 0.6, 2.0, out[1]);
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
static std::vector<uint8_t> dilateMask(const uint8_t *source, int stride,
                                       int width, int height, int radius) {
    std::vector<uint8_t> horizontal(size_t(width) * height, 0);
    std::vector<uint8_t> result(size_t(width) * height, 0);
    for (int y = 0; y < height; ++y) {
        const uint8_t *row = source + size_t(y) * stride;
        uint8_t *target = horizontal.data() + size_t(y) * width;
        int count = 0;
        for (int x = 0; x <= std::min(width - 1, radius); ++x)
            count += row[x] != 0;
        for (int x = 0; x < width; ++x) {
            target[x] = count != 0;
            if (x >= radius)
                count -= row[x - radius] != 0;
            if (x + radius + 1 < width)
                count += row[x + radius + 1] != 0;
        }
    }
    std::vector<int> counts(width, 0);
    for (int y = 0; y <= std::min(height - 1, radius); ++y)
        for (int x = 0; x < width; ++x)
            counts[x] += horizontal[size_t(y) * width + x] != 0;
    for (int y = 0; y < height; ++y) {
        uint8_t *target = result.data() + size_t(y) * width;
        for (int x = 0; x < width; ++x)
            target[x] = counts[x] != 0;
        if (y >= radius) {
            const uint8_t *row = horizontal.data() + size_t(y - radius) * width;
            for (int x = 0; x < width; ++x)
                counts[x] -= row[x] != 0;
        }
        if (y + radius + 1 < height) {
            const uint8_t *row = horizontal.data() + size_t(y + radius + 1) * width;
            for (int x = 0; x < width; ++x)
                counts[x] += row[x] != 0;
        }
    }
    return result;
}

namespace {

class SobelRowRing {
public:
    static constexpr int halo = 4;

    SobelRowRing(const Plane &source, bool dense,
                 const std::vector<uint8_t> *active,
                 FrameScratchAllocator *scratch)
        : source_(source), dense_(dense), active_(active),
          ringRows_(std::min(source.h, 2 * halo + 1)),
          gx_(scratchPlane(scratch, source.w, ringRows_)),
          gy_(scratchPlane(scratch, source.w, ringRows_)),
          rowIds_(ringRows_, -1) {}

    void ensureHalo(int centerY) {
        const int begin = std::max(0, centerY - halo);
        const int end = std::min(source_.h - 1, centerY + halo);
        for (int sourceY = begin; sourceY <= end; ++sourceY)
            ensure(sourceY);
    }

    const float *xRow(int sourceY) const {
        return gx_.row(sourceY % ringRows_);
    }
    const float *yRow(int sourceY) const {
        return gy_.row(sourceY % ringRows_);
    }
    float xAt(int x, int y) const {
        x = std::clamp(x, 0, source_.w - 1);
        y = std::clamp(y, 0, source_.h - 1);
        return xRow(y)[x];
    }
    float yAt(int x, int y) const {
        x = std::clamp(x, 0, source_.w - 1);
        y = std::clamp(y, 0, source_.h - 1);
        return yRow(y)[x];
    }
    float sampleProjected(const BilinearPoint &point, float nx, float ny) const {
        const float *xt = xRow(point.y0), *xb = xRow(point.y1);
        const float *yt = yRow(point.y0), *yb = yRow(point.y1);
        const float a = xt[point.x0] * nx + yt[point.x0] * ny;
        const float b = xt[point.x1] * nx + yt[point.x1] * ny;
        const float c = xb[point.x0] * nx + yb[point.x0] * ny;
        const float d = xb[point.x1] * nx + yb[point.x1] * ny;
        return a + (b - a) * point.fx + (c - a) * point.fy +
               (a - b - c + d) * point.fx * point.fy;
    }

private:
    void ensure(int sourceY) {
        const int slot = sourceY % ringRows_;
        if (rowIds_[slot] == sourceY)
            return;
        float *gx = gx_.row(slot);
        float *gy = gy_.row(slot);
        if (dense_) {
            sobelRow<NativeBackend>(source_, sourceY, gx, gy);
        } else {
            std::fill_n(gx, source_.w, 0.0f);
            std::fill_n(gy, source_.w, 0.0f);
            const uint8_t *activeRow =
                active_->data() + size_t(sourceY) * source_.w;
            int x = 0;
            while (x < source_.w) {
                while (x < source_.w && activeRow[x] == 0)
                    ++x;
                while (x < source_.w && activeRow[x] != 0) {
                    sobelPixel(source_, x, sourceY, gx[x], gy[x]);
                    ++x;
                }
            }
        }
        rowIds_[slot] = sourceY;
    }

    const Plane &source_;
    bool dense_;
    const std::vector<uint8_t> *active_;
    int ringRows_;
    Plane gx_;
    Plane gy_;
    std::vector<int> rowIds_;
};

} // namespace

Plane buildMutualGate(const Plane &lc, const Plane &U, const Plane &V, double sigma,
                      const uint8_t *activeMask, int activeStride,
                      const SparseWorkset *workset, PipelineMetrics *metrics,
                      FrameScratchAllocator *scratch) {
    (void)sigma; // reserved for a future noise-adaptive profile floor
    const int w = lc.w, h = lc.h;
    const bool sharedWorkset = workset && workset->chromaWidth == w &&
        workset->chromaHeight == h && activeMask;
    const bool dense = !activeMask || (sharedWorkset && workset->chromaDenseFallback());
    // A gate sample reaches three pixels along the edge normal and Sobel reaches
    // one pixel farther. Keep only that four-pixel halo instead of six full
    // gradient planes.
    const std::vector<uint8_t> gradientMask = dense
        ? std::vector<uint8_t>{}
        : dilateMask(activeMask, activeStride, w, h, SobelRowRing::halo);
    SobelRowRing lumaGradients(lc, dense, &gradientMask, scratch);
    SobelRowRing uGradients(U, dense, &gradientMask, scratch);
    SobelRowRing vGradients(V, dense, &gradientMask, scratch);

    Plane gate = scratchPlane(scratch, w, h);
    if (activeMask && !dense)
        gate.fill(0.0f);
    auto processPixel = [&](int x, int y) {
            // luma edge normal from the 3x3-smoothed structure tensor
            float sxx = 0, sxy = 0, syy = 0;
            if (x > 0 && x + 1 < w && y > 0 && y + 1 < h) {
                for (int dy = -1; dy <= 1; ++dy) {
                    const float *rx = lumaGradients.xRow(y + dy) + x - 1;
                    const float *ry = lumaGradients.yRow(y + dy) + x - 1;
                    for (int dx = 0; dx < 3; ++dx) {
                        const float a = rx[dx], b = ry[dx];
                        sxx += a * a; sxy += a * b; syy += b * b;
                    }
                }
            } else {
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const float a = lumaGradients.xAt(x + dx, y + dy);
                        const float b = lumaGradients.yAt(x + dx, y + dy);
                        sxx += a * a; sxy += a * b; syy += b * b;
                    }
            }
            const float jsum = sxx + syy;
            if (jsum < 1e-10f) {
                gate.at(x, y) = 0.0f;
                return;
            }
            const TensorDirection direction = principalTensorDirection(sxx, sxy, syy);
            const float nx = direction.nx, ny = direction.ny;

            float sumY = 0, sqY = 0, momY = 0;
            float sumC = 0, sqC = 0, momC = 0;
            for (int k = -3; k <= 3; ++k) {
                const double px = x + k * nx, py = y + k * ny;
                const BilinearPoint point = bilinearPoint(w, h, px, py);
                const float gY = std::fabs(
                    lumaGradients.sampleProjected(point, nx, ny));
                const float au = uGradients.sampleProjected(point, nx, ny);
                const float av = vGradients.sampleProjected(point, nx, ny);
                const float gC = std::sqrt(au * au + av * av);
                sumY += gY; sqY += gY * gY; momY += k * gY;
                sumC += gC; sqC += gC * gC; momC += k * gC;
            }
            // (No absolute energy gates: on flat luma the guide is inert
            // anyway, and on flat chroma every weighting gives the same
            // average — an energy floor can only CAP legitimate weak edges.)
            if (sumY < 1e-9f || sumC < 1e-9f) {
                gate.at(x, y) = 0.0f;
                return;
            }
            const float partY = sumY * sumY / (sqY + 1e-12f);
            const float partC = sumC * sumC / (sqC + 1e-12f);
            const float ratio = partC / (partY + 1e-6f);
            const float dcent = std::fabs(momY / sumY - momC / sumC);
            const float tw = std::clamp((1.6f - ratio) / 0.4f, 0.0f, 1.0f);
            const float tp = std::clamp((1.0f - dcent) / 0.5f, 0.0f, 1.0f);
            gate.at(x, y) = (tw * tw * (3.0f - 2.0f * tw)) * (tp * tp * (3.0f - 2.0f * tp));
    };
    using Clock = std::chrono::steady_clock;
    uint64_t gradientNanoseconds = 0;
    uint64_t gateNanoseconds = 0;
    for (int y = 0; y < h; ++y) {
        const auto gradientStart = metrics ? Clock::now() : Clock::time_point{};
        lumaGradients.ensureHalo(y);
        uGradients.ensureHalo(y);
        vGradients.ensureHalo(y);
        if (metrics)
            gradientNanoseconds += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now() - gradientStart).count());

        const auto gateStart = metrics ? Clock::now() : Clock::time_point{};
        if (!dense && sharedWorkset) {
            for (size_t spanIndex = workset->chromaRowOffsets[y];
                 spanIndex < workset->chromaRowOffsets[y + 1]; ++spanIndex) {
                const SparseSpan span = workset->chromaSpans[spanIndex];
                for (int x = span.begin; x < span.end; ++x)
                    processPixel(x, y);
            }
        } else {
            for (int x = 0; x < w; ++x) {
                if (activeMask && !dense &&
                    activeMask[size_t(y) * activeStride + x] == 0)
                    continue;
                processPixel(x, y);
            }
        }
        if (metrics)
            gateNanoseconds += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now() - gateStart).count());
    }
    if (metrics) {
        metrics->add(CpuProfileSlot::MutualGradients, gradientNanoseconds);
        metrics->add(CpuProfileSlot::MutualGate, gateNanoseconds);
    }
    return gate;
}

} // namespace lgcr
