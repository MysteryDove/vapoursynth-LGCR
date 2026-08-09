#include "lgcr.h"

#include <array>
#include <deque>

namespace lgcr {

// LGF-backed selector / sharpen / temporal support / detail transfer

namespace {

template <size_t N, class Sample, class Consume>
void forEachWindowSum(int width, int height, int radius,
                      Sample sample, Consume consume) {
    using Values = std::array<double, N>;
    std::vector<Values> columns(static_cast<size_t>(width));
    const int cachedRowCount = std::max(1, 2 * radius + 2);
    std::vector<Values> cachedRows(static_cast<size_t>(cachedRowCount) * width);
    int currentTop = 0;
    int currentBottom = -1;

    auto addRow = [&](int y) {
        Values values{};
        for (int x = 0; x < width; ++x) {
            values.fill(0.0);
            sample(x, y, values);
            cachedRows[size_t(y % cachedRowCount) * width + x] = values;
            for (size_t k = 0; k < N; ++k)
                columns[x][k] += values[k];
        }
    };
    auto removeRow = [&](int y) {
        const Values *values = cachedRows.data() + size_t(y % cachedRowCount) * width;
        for (int x = 0; x < width; ++x)
            for (size_t k = 0; k < N; ++k)
                columns[x][k] -= values[x][k];
    };

    for (int y = 0; y < height; ++y) {
        const int top = std::max(0, y - radius);
        const int bottom = std::min(height - 1, y + radius);
        while (currentBottom < bottom)
            addRow(++currentBottom);
        while (currentTop < top)
            removeRow(currentTop++);

        Values total{};
        int right = std::min(width - 1, radius);
        for (int x = 0; x <= right; ++x)
            for (size_t k = 0; k < N; ++k)
                total[k] += columns[x][k];

        for (int x = 0; x < width; ++x) {
            const int left = std::max(0, x - radius);
            right = std::min(width - 1, x + radius);
            const int count = (right - left + 1) * (bottom - top + 1);
            consume(x, y, count, total);
            const int remove = x - radius;
            const int add = x + radius + 1;
            if (remove >= 0)
                for (size_t k = 0; k < N; ++k)
                    total[k] -= columns[remove][k];
            if (add < width)
                for (size_t k = 0; k < N; ++k)
                    total[k] += columns[add][k];
        }
    }
}

template <class Backend>
void minMax5(const Plane &src, Plane &minimum, Plane &maximum) {
    Plane horizontalMin(src.w, src.h), horizontalMax(src.w, src.h);
    for (int y = 0; y < src.h; ++y) {
        const float *source = src.row(y);
        float *rowMin = horizontalMin.row(y);
        float *rowMax = horizontalMax.row(y);
        int x = 0;
        for (; x < std::min(2, src.w); ++x) {
            float lo = source[std::max(0, x - 2)];
            float hi = lo;
            for (int dx = -1; dx <= 2; ++dx) {
                const float value = source[std::clamp(x + dx, 0, src.w - 1)];
                lo = std::min(lo, value);
                hi = std::max(hi, value);
            }
            rowMin[x] = lo;
            rowMax[x] = hi;
        }
        for (; x + Backend::lanes <= src.w - 2; x += Backend::lanes) {
            auto lo = Backend::load(source + x - 2);
            auto hi = lo;
            for (int dx = -1; dx <= 2; ++dx) {
                const auto value = Backend::load(source + x + dx);
                lo = Backend::min(lo, value);
                hi = Backend::max(hi, value);
            }
            Backend::store(rowMin + x, lo);
            Backend::store(rowMax + x, hi);
        }
        for (; x < src.w; ++x) {
            float lo = source[std::max(0, x - 2)];
            float hi = lo;
            for (int dx = -1; dx <= 2; ++dx) {
                const float value = source[std::clamp(x + dx, 0, src.w - 1)];
                lo = std::min(lo, value);
                hi = std::max(hi, value);
            }
            rowMin[x] = lo;
            rowMax[x] = hi;
        }
    }

    for (int y = 0; y < src.h; ++y) {
        float *rowMin = minimum.row(y);
        float *rowMax = maximum.row(y);
        int x = 0;
        for (; x + Backend::lanes <= src.w; x += Backend::lanes) {
            auto lo = Backend::load(horizontalMin.row(std::max(0, y - 2)) + x);
            auto hi = Backend::load(horizontalMax.row(std::max(0, y - 2)) + x);
            for (int dy = -1; dy <= 2; ++dy) {
                const int sourceY = std::clamp(y + dy, 0, src.h - 1);
                lo = Backend::min(lo, Backend::load(horizontalMin.row(sourceY) + x));
                hi = Backend::max(hi, Backend::load(horizontalMax.row(sourceY) + x));
            }
            Backend::store(rowMin + x, lo);
            Backend::store(rowMax + x, hi);
        }
        for (; x < src.w; ++x) {
            float lo = horizontalMin.at(x, y - 2);
            float hi = horizontalMax.at(x, y - 2);
            for (int dy = -1; dy <= 2; ++dy) {
                lo = std::min(lo, horizontalMin.at(x, y + dy));
                hi = std::max(hi, horizontalMax.at(x, y + dy));
            }
            rowMin[x] = lo;
            rowMax[x] = hi;
        }
    }
}

#ifdef __AVX2__
template <class Backend>
void minMax5Pair(const Plane &u, const Plane &v,
                 Plane &minimumU, Plane &maximumU,
                 Plane &minimumV, Plane &maximumV) {
    const int width = u.w, height = u.h;
    constexpr int ringRows = 5;
    std::array<std::vector<float>, 4> ring;
    for (auto &buffer : ring)
        buffer.resize(size_t(ringRows) * width);
    std::array<int, ringRows> rowIds{{-1, -1, -1, -1, -1}};

    auto horizontalRow = [&](int sourceY) {
        const int slot = sourceY % ringRows;
        if (rowIds[slot] == sourceY)
            return slot;
        const float *sourceU = u.row(sourceY), *sourceV = v.row(sourceY);
        float *loU = ring[0].data() + size_t(slot) * width;
        float *hiU = ring[1].data() + size_t(slot) * width;
        float *loV = ring[2].data() + size_t(slot) * width;
        float *hiV = ring[3].data() + size_t(slot) * width;
        int x = 0;
        for (; x < std::min(2, width); ++x) {
            float ulo = sourceU[std::max(0, x - 2)], uhi = ulo;
            float vlo = sourceV[std::max(0, x - 2)], vhi = vlo;
            for (int dx = -1; dx <= 2; ++dx) {
                const int sx = std::clamp(x + dx, 0, width - 1);
                ulo = std::min(ulo, sourceU[sx]); uhi = std::max(uhi, sourceU[sx]);
                vlo = std::min(vlo, sourceV[sx]); vhi = std::max(vhi, sourceV[sx]);
            }
            loU[x] = ulo; hiU[x] = uhi; loV[x] = vlo; hiV[x] = vhi;
        }
        for (; x + Backend::lanes <= width - 2; x += Backend::lanes) {
            auto ulo = Backend::load(sourceU + x - 2), uhi = ulo;
            auto vlo = Backend::load(sourceV + x - 2), vhi = vlo;
            for (int dx = -1; dx <= 2; ++dx) {
                const auto uv = Backend::load(sourceU + x + dx);
                const auto vv = Backend::load(sourceV + x + dx);
                ulo = Backend::min(ulo, uv); uhi = Backend::max(uhi, uv);
                vlo = Backend::min(vlo, vv); vhi = Backend::max(vhi, vv);
            }
            Backend::store(loU + x, ulo); Backend::store(hiU + x, uhi);
            Backend::store(loV + x, vlo); Backend::store(hiV + x, vhi);
        }
        for (; x < width; ++x) {
            float ulo = sourceU[std::max(0, x - 2)], uhi = ulo;
            float vlo = sourceV[std::max(0, x - 2)], vhi = vlo;
            for (int dx = -1; dx <= 2; ++dx) {
                const int sx = std::clamp(x + dx, 0, width - 1);
                ulo = std::min(ulo, sourceU[sx]); uhi = std::max(uhi, sourceU[sx]);
                vlo = std::min(vlo, sourceV[sx]); vhi = std::max(vhi, sourceV[sx]);
            }
            loU[x] = ulo; hiU[x] = uhi; loV[x] = vlo; hiV[x] = vhi;
        }
        rowIds[slot] = sourceY;
        return slot;
    };

    for (int y = 0; y < height; ++y) {
        std::array<int, 5> slots;
        for (int dy = -2; dy <= 2; ++dy)
            slots[dy + 2] = horizontalRow(std::clamp(y + dy, 0, height - 1));
        float *outLoU = minimumU.row(y), *outHiU = maximumU.row(y);
        float *outLoV = minimumV.row(y), *outHiV = maximumV.row(y);
        int x = 0;
        for (; x + Backend::lanes <= width; x += Backend::lanes) {
            auto ulo = Backend::load(ring[0].data() + size_t(slots[0]) * width + x);
            auto uhi = Backend::load(ring[1].data() + size_t(slots[0]) * width + x);
            auto vlo = Backend::load(ring[2].data() + size_t(slots[0]) * width + x);
            auto vhi = Backend::load(ring[3].data() + size_t(slots[0]) * width + x);
            for (int row = 1; row < 5; ++row) {
                ulo = Backend::min(ulo, Backend::load(
                    ring[0].data() + size_t(slots[row]) * width + x));
                uhi = Backend::max(uhi, Backend::load(
                    ring[1].data() + size_t(slots[row]) * width + x));
                vlo = Backend::min(vlo, Backend::load(
                    ring[2].data() + size_t(slots[row]) * width + x));
                vhi = Backend::max(vhi, Backend::load(
                    ring[3].data() + size_t(slots[row]) * width + x));
            }
            Backend::store(outLoU + x, ulo); Backend::store(outHiU + x, uhi);
            Backend::store(outLoV + x, vlo); Backend::store(outHiV + x, vhi);
        }
        for (; x < width; ++x) {
            float ulo = ring[0][size_t(slots[0]) * width + x];
            float uhi = ring[1][size_t(slots[0]) * width + x];
            float vlo = ring[2][size_t(slots[0]) * width + x];
            float vhi = ring[3][size_t(slots[0]) * width + x];
            for (int row = 1; row < 5; ++row) {
                ulo = std::min(ulo, ring[0][size_t(slots[row]) * width + x]);
                uhi = std::max(uhi, ring[1][size_t(slots[row]) * width + x]);
                vlo = std::min(vlo, ring[2][size_t(slots[row]) * width + x]);
                vhi = std::max(vhi, ring[3][size_t(slots[row]) * width + x]);
            }
            outLoU[x] = ulo; outHiU[x] = uhi; outLoV[x] = vlo; outHiV[x] = vhi;
        }
    }
}
#endif

void slidingMinMax(const Plane &src, int radius, Plane &minimum, Plane &maximum) {
    if (radius == 2) {
        minMax5<NativeBackend>(src, minimum, maximum);
        return;
    }

    Plane horizontalMin(src.w, src.h), horizontalMax(src.w, src.h);
    for (int y = 0; y < src.h; ++y) {
        std::deque<int> minQueue, maxQueue;
        int next = 0;
        const float *row = src.row(y);
        for (int x = 0; x < src.w; ++x) {
            const int wantedRight = std::min(src.w - 1, x + radius);
            while (next <= wantedRight) {
                while (!minQueue.empty() && row[minQueue.back()] >= row[next])
                    minQueue.pop_back();
                while (!maxQueue.empty() && row[maxQueue.back()] <= row[next])
                    maxQueue.pop_back();
                minQueue.push_back(next);
                maxQueue.push_back(next);
                ++next;
            }
            const int wantedLeft = x - radius;
            while (!minQueue.empty() && minQueue.front() < wantedLeft)
                minQueue.pop_front();
            while (!maxQueue.empty() && maxQueue.front() < wantedLeft)
                maxQueue.pop_front();
            horizontalMin.row(y)[x] = row[minQueue.front()];
            horizontalMax.row(y)[x] = row[maxQueue.front()];
        }
    }

    for (int x = 0; x < src.w; ++x) {
        std::deque<int> minQueue, maxQueue;
        int next = 0;
        for (int y = 0; y < src.h; ++y) {
            const int wantedBottom = std::min(src.h - 1, y + radius);
            while (next <= wantedBottom) {
                while (!minQueue.empty() &&
                       horizontalMin.row(minQueue.back())[x] >= horizontalMin.row(next)[x])
                    minQueue.pop_back();
                while (!maxQueue.empty() &&
                       horizontalMax.row(maxQueue.back())[x] <= horizontalMax.row(next)[x])
                    maxQueue.pop_back();
                minQueue.push_back(next);
                maxQueue.push_back(next);
                ++next;
            }
            const int wantedTop = y - radius;
            while (!minQueue.empty() && minQueue.front() < wantedTop)
                minQueue.pop_front();
            while (!maxQueue.empty() && maxQueue.front() < wantedTop)
                maxQueue.pop_front();
            minimum.row(y)[x] = horizontalMin.row(minQueue.front())[x];
            maximum.row(y)[x] = horizontalMax.row(maxQueue.front())[x];
        }
    }
}

Plane pointSampledLuma(const Plane &Y, int cw, int ch, double rw, double rh,
                       double shiftX, double shiftY) {
    Plane result(cw, ch);
    std::vector<float> xPosition(cw), yPosition(ch);
    for (int x = 0; x < cw; ++x)
        xPosition[x] = float((x + 0.5) * rw - 0.5 + shiftX);
    for (int y = 0; y < ch; ++y)
        yPosition[y] = float((y + 0.5) * rh - 0.5 + shiftY);
    const BilinAxis xAxis = buildBilinAxis(xPosition, Y.w);
    const BilinAxis yAxis = buildBilinAxis(yPosition, Y.h);
    for (int y = 0; y < ch; ++y)
        for (int x = 0; x < cw; ++x)
            result.row(y)[x] = bilinearFast(
                Y, xAxis.i0[x], xAxis.f[x], yAxis.i0[y], yAxis.f[y]);
    return result;
}

} // namespace

void buildLGF(const Plane &Y, int cw, int ch, double rw, double rh,
                     double shiftX, double shiftY, const Plane &C, int radius, double eps,
                     Plane &a, Plane &b, Plane &conf, bool cedge) {
    // Luma level per chroma sample = POINT SAMPLE at the sited position (not
    // the footprint average used by the sim path). The regression needs Y at
    // exactly the position C refers to; a misaligned aperture biases the
    // slope by a * offset * slope, which dominates on gradients.
    Plane ls = pointSampledLuma(Y, cw, ch, rw, rh, shiftX, shiftY);
    Plane cmin, cmax;
    if (cedge) {
        cmin = Plane(cw, ch);
        cmax = Plane(cw, ch);
        slidingMinMax(C, radius, cmin, cmax);
    }

    forEachWindowSum<4>(cw, ch, radius,
        [&](int x, int y, std::array<double, 4> &values) {
            const double ly = ls.row(y)[x];
            const double cc = C.row(y)[x];
            values = { ly, ly * ly, cc, ly * cc };
        },
        [&](int x, int y, int n, const std::array<double, 4> &sum) {
            const double meanY = sum[0] / n;
            const double meanC = sum[2] / n;
            const double cov = sum[3] / n - meanY * meanC;
            const double var = sum[1] / n - meanY * meanY;
            const double av = cov / (var + eps);
            a.row(y)[x] = static_cast<float>(av);
            b.row(y)[x] = static_cast<float>(meanC - av * meanY);
            float cf = static_cast<float>(var / (var + eps));
            if (cedge) {
                const float lgx = (ls.at(x + 1, y) - ls.at(x - 1, y)) * 0.5f;
                const float lgy = (ls.at(x, y + 1) - ls.at(x, y - 1)) * 0.5f;
                const float gl = std::hypot(lgx, lgy);
                const float range = cmax.row(y)[x] - cmin.row(y)[x];
                if (gl > 1e-9f && range > 1e-6f) {
                    const float cgx = (C.at(x + 1, y) - C.at(x - 1, y)) * 0.5f;
                    const float cgy = (C.at(x, y + 1) - C.at(x, y - 1)) * 0.5f;
                    const float gradN = std::fabs(cgx * lgx / gl + cgy * lgy / gl);
                    const float wc = range / (gradN + 1e-6f);
                    cf *= std::clamp((2.2f - wc) / 0.7f, 0.0f, 1.0f);
                }
            }
            conf.row(y)[x] = cf;
        });
}

void buildLGFPair(const Plane &Y, int cw, int ch, double rw, double rh,
                  double shiftX, double shiftY, const Plane &U, const Plane &V,
                  int radius, double eps,
                  Plane &aU, Plane &bU, Plane &confU,
                  Plane &aV, Plane &bV, Plane &confV, bool cedge) {
    Plane ls = pointSampledLuma(Y, cw, ch, rw, rh, shiftX, shiftY);
    Plane minU, maxU, minV, maxV;
    if (cedge) {
        minU = Plane(cw, ch); maxU = Plane(cw, ch);
        minV = Plane(cw, ch); maxV = Plane(cw, ch);
        slidingMinMax(U, radius, minU, maxU);
        slidingMinMax(V, radius, minV, maxV);
    }

    forEachWindowSum<6>(cw, ch, radius,
        [&](int x, int y, std::array<double, 6> &values) {
            const double ly = ls.row(y)[x];
            const double u = U.row(y)[x], v = V.row(y)[x];
            values = { ly, ly * ly, u, ly * u, v, ly * v };
        },
        [&](int x, int y, int n, const std::array<double, 6> &sum) {
            const double meanY = sum[0] / n;
            const double var = sum[1] / n - meanY * meanY;
            const double meanU = sum[2] / n, meanV = sum[4] / n;
            const double slopeU = (sum[3] / n - meanY * meanU) / (var + eps);
            const double slopeV = (sum[5] / n - meanY * meanV) / (var + eps);
            aU.row(y)[x] = static_cast<float>(slopeU);
            bU.row(y)[x] = static_cast<float>(meanU - slopeU * meanY);
            aV.row(y)[x] = static_cast<float>(slopeV);
            bV.row(y)[x] = static_cast<float>(meanV - slopeV * meanY);
            float baseConfidence = static_cast<float>(var / (var + eps));
            float confidenceU = baseConfidence, confidenceV = baseConfidence;

            if (cedge) {
                const float lgx = (ls.at(x + 1, y) - ls.at(x - 1, y)) * 0.5f;
                const float lgy = (ls.at(x, y + 1) - ls.at(x, y - 1)) * 0.5f;
                const float gl = std::hypot(lgx, lgy);
                auto fade = [&](const Plane &C, const Plane &minimum,
                                const Plane &maximum) {
                    const float range = maximum.row(y)[x] - minimum.row(y)[x];
                    if (gl <= 1e-9f || range <= 1e-6f)
                        return 1.0f;
                    const float cgx = (C.at(x + 1, y) - C.at(x - 1, y)) * 0.5f;
                    const float cgy = (C.at(x, y + 1) - C.at(x, y - 1)) * 0.5f;
                    const float gradN = std::fabs(cgx * lgx / gl + cgy * lgy / gl);
                    const float width = range / (gradN + 1e-6f);
                    return std::clamp((2.2f - width) / 0.7f, 0.0f, 1.0f);
                };
                confidenceU *= fade(U, minU, maxU);
                confidenceV *= fade(V, minV, maxV);
            }
            confU.row(y)[x] = confidenceU;
            confV.row(y)[x] = confidenceV;
        });
}

// algo=4: per-pixel mechanism selector.
//   out = plain + lam * (w2 * (guided - plain) + w3 * conf * (lgf - plain))
// w2/w3 come from the guided pass (hard-edge-ness x axis/diagonal split).
template <class Backend>
void blendSelectorImpl(Plane &outU, Plane &outV,
                          const Plane &gU, const Plane &gV,
                          const Plane &aU, const Plane &bU, const Plane &aV, const Plane &bV,
                          const Plane &confU, const Plane &confV, const Plane &Y,
                          const ChromaAxis &ax, const ChromaAxis &ay,
                          const Plane &w2, const Plane &w3,
                          int cw, int ch, float lam) {
    [[maybe_unused]] constexpr int lanes = Backend::lanes;
    (void)cw;
    (void)ch;
    const BilinAxis &bcx = ax.chromaBilin;
    const BilinAxis &bcy = ay.chromaBilin;
    const BilinAxis &blx = ax.lumaBilin;
    const BilinAxis &bly = ay.lumaBilin;
    for (int oy = 0; oy < outU.h; ++oy) {
        float *ru = outU.row(oy);
        float *rv = outV.row(oy);
        const float *gur = gU.row(oy);
        const float *gvr = gV.row(oy);
        const float *w2r = w2.row(oy);
        const float *w3r = w3.row(oy);
        const int cyi = bcy.i0[oy], lyi = bly.i0[oy];
        const float cyf = bcy.f[oy], lyf = bly.f[oy];
        for (int ox = 0; ox < outU.w; ++ox) {
            const float L0 = bilinearFast(Y, blx.i0[ox], blx.f[ox], lyi, lyf);
            const int cxi = bcx.i0[ox];
            const float cxf = bcx.f[ox];
            const float w2v = w2r[ox] * lam;
            const float w3v = w3r[ox] * lam;
            if (w2v < 1e-4f && w3v < 1e-4f)
                continue;
            const float au = bilinearFast(aU, cxi, cxf, cyi, cyf);
            const float bu = bilinearFast(bU, cxi, cxf, cyi, cyf);
            const float av = bilinearFast(aV, cxi, cxf, cyi, cyf);
            const float bv = bilinearFast(bV, cxi, cxf, cyi, cyf);
            const float cu = bilinearFast(confU, cxi, cxf, cyi, cyf);
            const float cv = bilinearFast(confV, cxi, cxf, cyi, cyf);
            ru[ox] += w2v * (gur[ox] - ru[ox]) + w3v * cu * (au * L0 + bu - ru[ox]);
            rv[ox] += w2v * (gvr[ox] - rv[ox]) + w3v * cv * (av * L0 + bv - rv[ox]);
        }
    }
}

void blendSelector(Plane &outU, Plane &outV,
                   const Plane &gU, const Plane &gV,
                   const Plane &aU, const Plane &bU, const Plane &aV, const Plane &bV,
                   const Plane &confU, const Plane &confV, const Plane &Y,
                   const ChromaAxis &ax, const ChromaAxis &ay,
                   const Plane &w2, const Plane &w3,
                   int cw, int ch, float lam) {
    blendSelectorImpl<NativeBackend>(outU, outV, gU, gV, aU, bU, aV, bV,
                                     confU, confV, Y, ax, ay, w2, w3,
                                     cw, ch, lam);
}

void sharpenPlane(Plane &p, const Plane *guideLc, const Plane &guideY,
                         double rw, double rh, double shiftX, double shiftY,
                         const SharpenData &d) {
    // guideLc: chroma-res luma footprint map (nullptr for luma self-guide)
    const int w = p.w, h = p.h;
    Plane src = p; // copy
    const float invG2 = 1.0f / float(d.gspatial * d.gspatial);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float c0 = src.at(x, y);
            // reference luma at this sample
            float L0;
            if (guideLc) {
                L0 = bilinear(guideY, (x + 0.5) * rw - 0.5 + shiftX,
                                      (y + 0.5) * rh - 0.5 + shiftY);
            } else {
                L0 = c0;
            }
            float maxAbsDL = 0.0f, hullMin = 1e30f, hullMax = -1e30f;
            for (int j = -2; j <= 2; ++j)
                for (int i = -2; i <= 2; ++i) {
                    if (guideLc)
                        maxAbsDL = std::max(maxAbsDL,
                                            std::fabs(guideLc->at(x + i, y + j) - L0));
                    else
                        maxAbsDL = std::max(maxAbsDL,
                                            std::fabs(src.at(x + i, y + j) - L0));
                    hullMin = std::min(hullMin, src.at(x + i, y + j));
                    hullMax = std::max(hullMax, src.at(x + i, y + j));
                }
            // sigma floor from the local first-order slope (median |4-neighbor
            // diff|): on smooth slopes this lets curvature taps pass the gate;
            // on hard steps the median is ~0 and the gate stays strict.
            const float c0l = src.at(x, y);
            float nd[4] = {
                std::fabs(src.at(x + 1, y) - c0l), std::fabs(src.at(x - 1, y) - c0l),
                std::fabs(src.at(x, y + 1) - c0l), std::fabs(src.at(x, y - 1) - c0l),
            };
            std::sort(nd, nd + 4);
            const float slopeMed = (nd[1] + nd[2]) * 0.5f;
            const float sig = std::max(std::max(float(d.sigma), float(d.sratio) * maxAbsDL),
                                       3.0f * slopeMed);
            const float invS2 = 1.0f / (sig * sig);
            float hp = 0.0f, wsum = 0.0f;
            for (int j = -2; j <= 2; ++j)
                for (int i = -2; i <= 2; ++i) {
                    if (i == 0 && j == 0)
                        continue;
                    const float dL = (guideLc ? guideLc->at(x + i, y + j)
                                              : src.at(x + i, y + j)) - L0;
                    const float sim = 1.0f / (1.0f + dL * dL * invS2);
                    const float sp = 1.0f / (1.0f + float(i * i + j * j) * invG2);
                    const float n = sp * sim;
                    hp += n * (c0 - src.at(x + i, y + j));
                    wsum += n;
                }
            float out = c0;
            if (wsum > 1e-6f)
                out = c0 + float(d.alpha) * hp / wsum;
            if (d.arMargin >= 0.0)
                out = std::clamp(out, hullMin - float(d.arMargin), hullMax + float(d.arMargin));
            p.at(x, y) = out;
        }
    }
}

// Back-projection data consistency: one cheap IBP-style iteration.
// residual = C_src - box2x2(C_out);  C_out += bp * residual (per 2x2 block)
void backProject(Plane &cOut, const Plane &cSrc, float bp) {
    const int cw = cSrc.w, ch = cSrc.h;
    for (int cy = 0; cy < ch; ++cy)
        for (int cx = 0; cx < cw; ++cx) {
            const int x0 = 2 * cx, y0 = 2 * cy;
            const float p00 = cOut.at(x0, y0), p10 = cOut.at(x0 + 1, y0);
            const float p01 = cOut.at(x0, y0 + 1), p11 = cOut.at(x0 + 1, y0 + 1);
            const float r = cSrc.at(cx, cy) - 0.25f * (p00 + p10 + p01 + p11);
            const float dcorr = bp * r;
            cOut.at(x0, y0) += dcorr;
            cOut.at(x0 + 1, y0) += dcorr;
            cOut.at(x0, y0 + 1) += dcorr;
            cOut.at(x0 + 1, y0 + 1) += dcorr;
        }
}

// ---------------------------------------------------------------------------
// TRecon support: integer-pel luma block matching
// ---------------------------------------------------------------------------

void blockMatch(const Plane &cur, const Plane &nbr, int blockSize, int search,
                float tsad, std::vector<int16_t> &mvx, std::vector<int16_t> &mvy,
                std::vector<float> &conf, int &bw, int &bh) {
    bw = (cur.w + blockSize - 1) / blockSize;
    bh = (cur.h + blockSize - 1) / blockSize;
    mvx.assign(size_t(bw) * bh, 0);
    mvy.assign(size_t(bw) * bh, 0);
    conf.assign(size_t(bw) * bh, 0.0f);
    const float invT2 = 1.0f / (tsad * tsad);

    for (int by = 0; by < bh; ++by)
        for (int bx = 0; bx < bw; ++bx) {
            const int x0 = bx * blockSize, y0 = by * blockSize;
            const int bw_ = std::min(blockSize, cur.w - x0);
            const int bh_ = std::min(blockSize, cur.h - y0);
            float best = 1e30f, sad0 = 1e30f;
            double sum = 0.0, sum2 = 0.0;
            int bdx = 0, bdy = 0;
            for (int j = 0; j < bh_; j += 2)
                for (int i = 0; i < bw_; i += 2) {
                    const float v = cur.at(x0 + i, y0 + j);
                    sum += v;
                    sum2 += double(v) * v;
                }
            const float npix = float((bw_ + 1) / 2 * ((bh_ + 1) / 2));
            for (int dy = -search; dy <= search; ++dy)
                for (int dx = -search; dx <= search; ++dx) {
                    if (x0 + dx < 0 || y0 + dy < 0 ||
                        x0 + dx + bw_ > cur.w || y0 + dy + bh_ > cur.h)
                        continue;
                    float sad = 0;
                    for (int j = 0; j < bh_; j += 2)
                        for (int i = 0; i < bw_; i += 2)
                            sad += std::fabs(cur.at(x0 + i, y0 + j) -
                                             nbr.at(x0 + dx + i, y0 + dy + j));
                    if (dx == 0 && dy == 0)
                        sad0 = sad;
                    // Equal SAD: prefer the smaller motion (zero-motion
                    // bias). On a straight edge the tangential component is
                    // unobservable (aperture problem) but harmless — the
                    // fetched samples are identical along iso-contours.
                    const int mnorm = std::abs(dx) + std::abs(dy);
                    const int bnorm = std::abs(bdx) + std::abs(bdy);
                    if (sad < best || (sad == best && mnorm < bnorm)) {
                        best = sad;
                        bdx = dx;
                        bdy = dy;
                    }
                }
            // Zero-motion snap: accept a nonzero vector only when it beats
            // staying put by a real margin. In noisy flat blocks the search
            // otherwise mines a random small vector out of the noise.
            if ((bdx != 0 || bdy != 0) && sad0 - best <= 0.10f * tsad * npix) {
                bdx = 0;
                bdy = 0;
                best = sad0;
            }
            const size_t idx = size_t(by) * bw + bx;
            mvx[idx] = int16_t(bdx);
            mvy[idx] = int16_t(bdy);
            const float sadPx = best / npix;
            // match quality x observability. Motion is only observable where
            // the LUMA block has texture: a flat block matches everywhere
            // (margin-based uniqueness fails differently: it also fires on
            // straight edges via the aperture ambiguity). The earlier
            // first-scan tie rule picked a far corner at FULL confidence and
            // dragged chroma texture across the frame (0.012 -> 0.059).
            const float var = float(std::max(0.0, sum2 / npix - (sum / npix) * (sum / npix)));
            const float matchQ = 1.0f / (1.0f + sadPx * sadPx * invT2);
            const float uniq = var / (var + 0.25f * tsad * tsad);
            conf[idx] = matchQ * uniq;
        }
}

void applyChromaConsistency(const Plane &curU, const Plane &curV,
                            const Plane &nbrU, const Plane &nbrV,
                            int lumaW, int lumaH, int blockSize,
                            const std::vector<int16_t> &mvx,
                            const std::vector<int16_t> &mvy,
                            int bw, int bh, std::vector<float> &conf) {
    const double rw = double(lumaW) / curU.w;
    const double rh = double(lumaH) / curU.h;
    std::vector<float> delta;
    delta.reserve(size_t(std::ceil(blockSize / rw)) *
                  size_t(std::ceil(blockSize / rh)));

    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            const size_t bi = size_t(by) * bw + bx;
            if (conf[bi] <= 0.0f)
                continue;
            const int lx0 = bx * blockSize;
            const int ly0 = by * blockSize;
            const int lx1 = std::min(lumaW, lx0 + blockSize);
            const int ly1 = std::min(lumaH, ly0 + blockSize);
            const int cx0 = std::clamp(int(std::floor(lx0 / rw)), 0, curU.w - 1);
            const int cy0 = std::clamp(int(std::floor(ly0 / rh)), 0, curU.h - 1);
            const int cx1 = std::clamp(int(std::ceil(lx1 / rw)), cx0 + 1, curU.w);
            const int cy1 = std::clamp(int(std::ceil(ly1 / rh)), cy0 + 1, curU.h);
            const double mx = mvx[bi] / rw;
            const double my = mvy[bi] / rh;

            delta.clear();
            for (int cy = cy0; cy < cy1; ++cy) {
                for (int cx = cx0; cx < cx1; ++cx) {
                    const int nx = std::clamp(int(std::lround(cx + mx)), 0, nbrU.w - 1);
                    const int ny = std::clamp(int(std::lround(cy + my)), 0, nbrU.h - 1);
                    delta.push_back(std::max(std::fabs(curU.at(cx, cy) - nbrU.at(nx, ny)),
                                             std::fabs(curV.at(cx, cy) - nbrV.at(nx, ny))));
                }
            }
            const size_t mid = delta.size() / 2;
            std::nth_element(delta.begin(), delta.begin() + mid, delta.end());
            float median = delta[mid];
            if ((delta.size() & 1) == 0) {
                const float lower = *std::max_element(delta.begin(), delta.begin() + mid);
                median = 0.5f * (lower + median);
            }
            const float ratio = median / 0.04f;
            const float ratio2 = ratio * ratio;
            const float cconf = 1.0f / (1.0f + ratio2 * ratio2);
            conf[bi] *= cconf;
        }
    }
}

// ---------------------------------------------------------------------------
// algo=6: constrained detail transfer
// ---------------------------------------------------------------------------

static float median3(float a, float b, float c) {
    return a + b + c - std::min({a, b, c}) - std::max({a, b, c});
}

#ifdef __AVX2__
void affineFixed5Avx2(const Plane &cb, const Plane &cr,
                      const std::array<Plane, 3> &yc, double eps, double epsSig,
                      AffineMaps &m) {
    constexpr int momentCount = 16;
    constexpr int ringRows = 5;
    const int width = cb.w, height = cb.h;
    std::vector<double> columns(size_t(momentCount) * width, 0.0);
    std::vector<double> ring(size_t(momentCount) * ringRows * width);
    int currentTop = 0, currentBottom = -1;

    auto ringPlane = [&](int moment, int row) {
        return ring.data() + (size_t(moment) * ringRows + row % ringRows) * width;
    };
    auto column = [&](int moment) {
        return columns.data() + size_t(moment) * width;
    };
    auto addRow = [&](int y) {
        for (int x = 0; x < width; ++x) {
            const double u = cb.row(y)[x], v = cr.row(y)[x];
            ringPlane(0, y)[x] = u; ringPlane(1, y)[x] = u * u;
            ringPlane(2, y)[x] = v; ringPlane(3, y)[x] = v * v;
            for (int k = 0; k < 3; ++k) {
                const double ly = yc[k].row(y)[x];
                const int base = 4 + 4 * k;
                ringPlane(base, y)[x] = ly;
                ringPlane(base + 1, y)[x] = ly * ly;
                ringPlane(base + 2, y)[x] = ly * u;
                ringPlane(base + 3, y)[x] = ly * v;
            }
            m.degradedLuma.row(y)[x] = median3(
                yc[0].row(y)[x], yc[1].row(y)[x], yc[2].row(y)[x]);
        }
        for (int k = 0; k < momentCount; ++k) {
            double *dst = column(k);
            const double *src = ringPlane(k, y);
            int x = 0;
            for (; x + 4 <= width; x += 4)
                _mm256_storeu_pd(dst + x, _mm256_add_pd(
                    _mm256_loadu_pd(dst + x), _mm256_loadu_pd(src + x)));
            for (; x < width; ++x)
                dst[x] += src[x];
        }
    };
    auto removeRow = [&](int y) {
        for (int k = 0; k < momentCount; ++k) {
            double *dst = column(k);
            const double *src = ringPlane(k, y);
            int x = 0;
            for (; x + 4 <= width; x += 4)
                _mm256_storeu_pd(dst + x, _mm256_sub_pd(
                    _mm256_loadu_pd(dst + x), _mm256_loadu_pd(src + x)));
            for (; x < width; ++x)
                dst[x] -= src[x];
        }
    };

    auto consume = [&](int x, int y, int count, const double *sum) {
        const double inverseCount = 1.0 / count;
        const double meanU = sum[0] * inverseCount;
        const double meanV = sum[2] * inverseCount;
        const double varU = std::max(0.0, sum[1] * inverseCount - meanU * meanU);
        const double varV = std::max(0.0, sum[3] * inverseCount - meanV * meanV);
        const float rangeU = m.mxU.row(y)[x] - m.mnU.row(y)[x];
        const float rangeV = m.mxV.row(y)[x] - m.mnV.row(y)[x];
        m.rng.row(y)[x] = std::max(rangeU, rangeV);

        double qk[3], aUk[3], aVk[3];
        for (int k = 0; k < 3; ++k) {
            const int base = 4 + 4 * k;
            const double meanY = sum[base] * inverseCount;
            const double varY = std::max(
                0.0, sum[base + 1] * inverseCount - meanY * meanY);
            const double covU = sum[base + 2] * inverseCount - meanY * meanU;
            const double covV = sum[base + 3] * inverseCount - meanY * meanV;
            aUk[k] = std::clamp(covU / (varY + eps), -16.0, 16.0);
            aVk[k] = std::clamp(covV / (varY + eps), -16.0, 16.0);
            qk[k] = std::clamp((covU * covU + covV * covV) /
                                   ((varY + eps) * (varU + varV + eps)),
                               0.0, 1.0);
        }
        const double qsum = qk[0] + qk[1] + qk[2];
        const double qmax = std::max({qk[0], qk[1], qk[2]});
        const double qmin = std::min({qk[0], qk[1], qk[2]});
        m.aU.row(y)[x] = median3(float(aUk[0]), float(aUk[1]), float(aUk[2]));
        m.aV.row(y)[x] = median3(float(aVk[0]), float(aVk[1]), float(aVk[2]));
        const double stability = qmax > 1e-9 ? qmin / qmax : 0.0;
        const double chromaSignal = (varU + varV) / (varU + varV + epsSig);
        m.g.row(y)[x] = float((qsum / 3.0) * stability * chromaSignal);
    };

    for (int y = 0; y < height; ++y) {
        const int top = std::max(0, y - 2);
        const int bottom = std::min(height - 1, y + 2);
        while (currentTop < top)
            removeRow(currentTop++);
        while (currentBottom < bottom)
            addRow(++currentBottom);
        const int verticalCount = bottom - top + 1;

        alignas(32) double sums[momentCount][4];
        int x = 0;
        for (; x < std::min(2, width); ++x) {
            double one[momentCount]{};
            for (int sx = 0; sx <= std::min(width - 1, x + 2); ++sx)
                for (int k = 0; k < momentCount; ++k)
                    one[k] += column(k)[sx];
            consume(x, y, (std::min(width - 1, x + 2) + 1) * verticalCount, one);
        }
        for (; x + 3 <= width - 3; x += 4) {
            for (int k = 0; k < momentCount; ++k) {
                const double *c = column(k) + x;
                __m256d value = _mm256_add_pd(
                    _mm256_loadu_pd(c - 2), _mm256_loadu_pd(c - 1));
                value = _mm256_add_pd(value, _mm256_loadu_pd(c));
                value = _mm256_add_pd(value, _mm256_loadu_pd(c + 1));
                value = _mm256_add_pd(value, _mm256_loadu_pd(c + 2));
                _mm256_store_pd(sums[k], value);
            }
            for (int lane = 0; lane < 4; ++lane) {
                double one[momentCount];
                for (int k = 0; k < momentCount; ++k)
                    one[k] = sums[k][lane];
                consume(x + lane, y, 5 * verticalCount, one);
            }
        }
        for (; x < width; ++x) {
            double one[momentCount]{};
            const int left = std::max(0, x - 2);
            const int right = std::min(width - 1, x + 2);
            for (int sx = left; sx <= right; ++sx)
                for (int k = 0; k < momentCount; ++k)
                    one[k] += column(k)[sx];
            consume(x, y, (right - left + 1) * verticalCount, one);
        }
    }
}
#endif

static Plane suppressSourceNyquist(Plane cur, double rw, double rh) {
    auto levelsFor = [](double ratio) {
        int levels = 0;
        while (ratio > 1.5 && levels < 4) {
            ratio *= 0.5;
            ++levels;
        }
        return levels;
    };
    const int xLevels = levelsFor(rw);
    const int yLevels = levelsFor(rh);
    if (xLevels == 0 && yLevels == 0)
        return cur;

    Plane next(cur.w, cur.h);
    for (int pass = 0; pass < xLevels; ++pass) {
        const int step = 1 << pass;
        for (int y = 0; y < cur.h; ++y)
            for (int x = 0; x < cur.w; ++x)
                next.at(x, y) = 0.25f * cur.at(x - step, y) + 0.5f * cur.at(x, y)
                              + 0.25f * cur.at(x + step, y);
        std::swap(cur.px, next.px);
    }
    for (int pass = 0; pass < yLevels; ++pass) {
        const int step = 1 << pass;
        for (int y = 0; y < cur.h; ++y)
            for (int x = 0; x < cur.w; ++x)
                next.at(x, y) = 0.25f * cur.at(x, y - step) + 0.5f * cur.at(x, y)
                              + 0.25f * cur.at(x, y + step);
        std::swap(cur.px, next.px);
    }
    return cur;
}

static Plane resampleDetailToOutput(Plane src, const LGCRData &d) {
    if (src.w == d.outW && src.h == d.outH)
        return src;
    Plane out(d.outW, d.outH);
    if (d.radial) {
        resampleRadial(src, out, d);
    } else {
        const auto th = cachedWeights(&d, src.w, d.outW, 0.0);
        const auto tv = cachedWeights(&d, src.h, d.outH, 0.0);
        Plane tmp(d.outW, src.h);
        resampleH(src, tmp, *th);
        resampleV(tmp, out, *tv);
    }
    return out;
}

AffineMaps buildAffineMaps(const LGCRData *d, const Plane &y, const Plane &cb,
                           const Plane &cr, int cw, int ch, double rw, double rh,
                           PipelineMetrics *metrics) {
    AffineMaps m;
    m.aU = Plane(cw, ch);
    m.aV = Plane(cw, ch);
    m.g = Plane(cw, ch);
    m.mnU = Plane(cw, ch);
    m.mxU = Plane(cw, ch);
    m.mnV = Plane(cw, ch);
    m.mxV = Plane(cw, ch);
    m.rng = Plane(cw, ch);

    const double eps = d->reg * d->reg;
    const double epsSig = (0.25 * d->sigma) * (0.25 * d->sigma);
    // Candidate encoder degradations (unknown D: accept only stable conclusions)
    const std::array<Plane, 3> yc = buildYcMaps(
        d, y, cw, ch, rw, rh, d->shiftX, d->shiftY, metrics);
    const int r = 2;
#ifdef __AVX2__
    {
        const auto start = std::chrono::steady_clock::now();
        minMax5Pair<NativeBackend>(cb, cr, m.mnU, m.mxU, m.mnV, m.mxV);
        if (metrics) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count();
            metrics->add(CpuProfileSlot::AffineMinMaxU, uint64_t(elapsed / 2));
            metrics->add(CpuProfileSlot::AffineMinMaxV,
                         uint64_t(elapsed - elapsed / 2));
        }
    }
#else
    {
        ScopedCpuTimer timer(metrics, CpuProfileSlot::AffineMinMaxU);
        slidingMinMax(cb, r, m.mnU, m.mxU);
    }
    {
        ScopedCpuTimer timer(metrics, CpuProfileSlot::AffineMinMaxV);
        slidingMinMax(cr, r, m.mnV, m.mxV);
    }
#endif
    m.degradedLuma = Plane(cw, ch);

    // U/V moments and all three degradation candidates share one rolling
    // window traversal. Working storage is O(width), not O(frame size).
    bool specialized = false;
#ifdef __AVX2__
    if (cw >= 8) {
        ScopedCpuTimer timer(metrics, CpuProfileSlot::AffineWindowMoments);
        affineFixed5Avx2(cb, cr, yc, eps, epsSig, m);
        specialized = true;
    }
#endif
    if (!specialized) {
        ScopedCpuTimer timer(metrics, CpuProfileSlot::AffineWindowMoments);
        forEachWindowSum<16>(cw, ch, r,
        [&](int x, int py, std::array<double, 16> &values) {
            const double u = cb.row(py)[x], v = cr.row(py)[x];
            values[0] = u; values[1] = u * u;
            values[2] = v; values[3] = v * v;
            for (int k = 0; k < 3; ++k) {
                const double ly = yc[k].row(py)[x];
                const int base = 4 + 4 * k;
                values[base] = ly;
                values[base + 1] = ly * ly;
                values[base + 2] = ly * u;
                values[base + 3] = ly * v;
            }
        },
        [&](int cx, int cy, int n, const std::array<double, 16> &sum) {
            const double meanU = sum[0] / n, meanV = sum[2] / n;
            const double varU = std::max(0.0, sum[1] / n - meanU * meanU);
            const double varV = std::max(0.0, sum[3] / n - meanV * meanV);
            const float rangeU = m.mxU.row(cy)[cx] - m.mnU.row(cy)[cx];
            const float rangeV = m.mxV.row(cy)[cx] - m.mnV.row(cy)[cx];
            m.rng.row(cy)[cx] = std::max(rangeU, rangeV);

            double qk[3], aUk[3], aVk[3];
            for (int k = 0; k < 3; ++k) {
                const int base = 4 + 4 * k;
                const double meanY = sum[base] / n;
                const double varY = std::max(
                    0.0, sum[base + 1] / n - meanY * meanY);
                const double covU = sum[base + 2] / n - meanY * meanU;
                const double covV = sum[base + 3] / n - meanY * meanV;
                aUk[k] = std::clamp(covU / (varY + eps), -16.0, 16.0);
                aVk[k] = std::clamp(covV / (varY + eps), -16.0, 16.0);
                qk[k] = std::clamp((covU * covU + covV * covV) /
                                       ((varY + eps) * (varU + varV + eps)),
                                   0.0, 1.0);
            }
            const double qsum = qk[0] + qk[1] + qk[2];
            const double qmax = std::max({qk[0], qk[1], qk[2]});
            const double qmin = std::min({qk[0], qk[1], qk[2]});
            m.aU.row(cy)[cx] = median3(float(aUk[0]), float(aUk[1]), float(aUk[2]));
            m.aV.row(cy)[cx] = median3(float(aVk[0]), float(aVk[1]), float(aVk[2]));
            const double qMean = qsum / 3.0;
            const double stability = qmax > 1e-9 ? qmin / qmax : 0.0;
            const double chromaSignal = (varU + varV) / (varU + varV + epsSig);
            m.g.row(cy)[cx] = float(qMean * stability * chromaSignal);
            });
    }
    // Build a kernel-independent matched guide by taking the per-sample median
    // of the candidate degradations. q/stability decide whether to trust the
    // resulting correction, but do not choose its direction or base signal.
    if (!specialized) {
        ScopedCpuTimer timer(metrics, CpuProfileSlot::AffineFinalizeMedian);
        for (int cy = 0; cy < ch; ++cy)
            for (int cx = 0; cx < cw; ++cx)
                m.degradedLuma.row(cy)[cx] = median3(
                    yc[0].row(cy)[cx], yc[1].row(cy)[cx], yc[2].row(cy)[cx]);
    }
    return m;
}

void buildDetailMap(const LGCRData *d, const Plane &y, int cw, int ch,
                    double rw, double rh, AffineMaps &m) {
    // Reconstruct matched luma to the SOURCE luma grid first. The unobservable
    // frequency locations are defined on this grid and must not move when the
    // user requests a different output size.
    LGCRData sourceD = *d;
    sourceD.outW = y.w;
    sourceD.outH = y.h;
    Plane sourceDetail(y.w, y.h);
    plainPlane(&sourceD, m.degradedLuma, y.w, y.h, cw, ch, sourceDetail);
    for (int py = 0; py < y.h; ++py)
        for (int px = 0; px < y.w; ++px)
            sourceDetail.at(px, py) = y.at(px, py) - sourceDetail.at(px, py);
    sourceDetail = suppressSourceNyquist(std::move(sourceDetail), rw, rh);
    m.detail = resampleDetailToOutput(std::move(sourceDetail), *d);
    m.degradedLuma = Plane();
}

template <class Backend>
void detailTransferImpl(const LGCRData *d, Plane &outU, Plane &outV,
                        const AffineMaps &af, const GuideMaps &gm,
                        const ChromaAxis &ax, const ChromaAxis &ay,
                        const uint8_t *mask, int maskW, int maskH) {
    [[maybe_unused]] constexpr int lanes = Backend::lanes;
    const int ow = outU.w, oh = outU.h;
    const Plane &dl = af.detail;
    const BilinAxis &bcx = ax.chromaBilin;
    const BilinAxis &bcy = ay.chromaBilin;
    const BilinAxis &blx = ax.lumaBilin;
    const BilinAxis &bly = ay.lumaBilin;
    const bool useMs = d->ms > 0.0 && gm.ms.w > 0;
    const float msStrength = float(d->ms);
    const float qStrength = float(d->qgate);
    const float strength = float(d->strength);
    const float ar = float(std::max(0.0, d->arMargin));
    const bool clampHull = d->arMargin >= 0.0;

    const bool directMask = mask && maskW == ow && maskH == oh;
    std::vector<int> maskX;
    if (mask && !directMask) {
        maskX.resize(ow);
        for (int ox = 0; ox < ow; ++ox)
            maskX[ox] = std::min(maskW - 1, int(int64_t(ox) * maskW / ow));
    }

    for (int oy = 0; oy < oh; ++oy) {
        float *ru = outU.row(oy);
        float *rv = outV.row(oy);
        const int cyi = bcy.i0[oy], lyi = bly.i0[oy];
        const float cyf = bcy.f[oy], lyf = bly.f[oy];
        const uint8_t *maskRow = nullptr;
        if (mask) {
            const int my = directMask ? oy
                : std::min(maskH - 1, int(int64_t(oy) * maskH / oh));
            maskRow = mask + size_t(my) * maskW;
        }
        for (int ox = 0; ox < ow; ++ox) {
            if (maskRow && maskRow[directMask ? ox : maskX[ox]] == 0)
                continue;
            const float jxx = bilinearFast(
                gm.jxx, blx.i0[ox], blx.f[ox], lyi, lyf);
            const float jxy = bilinearFast(
                gm.jxy, blx.i0[ox], blx.f[ox], lyi, lyf);
            const float jyy = bilinearFast(
                gm.jyy, blx.i0[ox], blx.f[ox], lyi, lyf);
            const TensorDirection direction = principalTensorDirection(jxx, jxy, jyy);
            const int cxi = bcx.i0[ox];
            const float cxf = bcx.f[ox];
            const float q = bilinearFast(af.g, cxi, cxf, cyi, cyf);
            float g = (1.0f - qStrength * (1.0f - q)) * strength;
            if (useMs) {
                const float ms = bilinearFast(gm.ms, cxi, cxf, cyi, cyf);
                g *= 1.0f - msStrength * (1.0f - ms);
            }
            const float au = bilinearFast(af.aU, cxi, cxf, cyi, cyf);
            const float av = bilinearFast(af.aV, cxi, cxf, cyi, cyf);
            const float coherence = direction.coherence;
            const float tx = -direction.ny, ty = direction.nx;
            // 1D restriction: attenuate detail along the edge tangent before
            // it can be injected into chroma.
            const float dc = dl.row(oy)[ox];
            const float s = 1.0f + float(d->stretch) * coherence;
            const float dtp = bilinear(dl, ox + tx * s, oy + ty * s);
            const float dtm = bilinear(dl, ox - tx * s, oy - ty * s);
            const float dt = 0.25f * dtp + 0.5f * dc + 0.25f * dtm;
            const float det = dc + (dt - dc) * coherence;
            // magnitude cap: correction may not exceed half the local chroma range
            const float cap = 0.5f * bilinearFast(af.rng, cxi, cxf, cyi, cyf);
            float cu = std::clamp(g * au * det, -cap, cap);
            float cv = std::clamp(g * av * det, -cap, cap);
            float ou = ru[ox] + cu;
            float ov = rv[ox] + cv;
            if (clampHull) {
                const float baseU = ru[ox], baseV = rv[ox];
                const float loU = std::min(baseU, bilinearFast(af.mnU, cxi, cxf, cyi, cyf) - ar);
                const float hiU = std::max(baseU, bilinearFast(af.mxU, cxi, cxf, cyi, cyf) + ar);
                const float loV = std::min(baseV, bilinearFast(af.mnV, cxi, cxf, cyi, cyf) - ar);
                const float hiV = std::max(baseV, bilinearFast(af.mxV, cxi, cxf, cyi, cyf) + ar);
                ou = std::clamp(ou, loU, hiU);
                ov = std::clamp(ov, loV, hiV);
            }
            ru[ox] = ou;
            rv[ox] = ov;
        }
    }
}

void detailTransfer(const LGCRData *d, Plane &outU, Plane &outV,
                    const AffineMaps &af, const GuideMaps &gm,
                    const ChromaAxis &ax, const ChromaAxis &ay,
                    const uint8_t *mask, int maskW, int maskH) {
    detailTransferImpl<NativeBackend>(d, outU, outV, af, gm, ax, ay,
                                      mask, maskW, maskH);
}

} // namespace lgcr
