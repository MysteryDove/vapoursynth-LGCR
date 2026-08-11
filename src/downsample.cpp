#include "lgcr.h"

#include <chrono>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace lgcr {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int highCandidateCount = 4;
constexpr float guidedActivationThreshold = 1.0e-4f;

bool downsampleProfilingEnabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("LGCR_PROFILE");
        return value && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void addElapsed(PipelineMetrics *metrics, Stage stage, Clock::time_point start) {
    if (!metrics)
        return;
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - start).count();
    metrics->add(stage, static_cast<uint64_t>(elapsed));
}

void writeDownsampleProfile(VSFrame *frame, const PipelineMetrics &metrics,
                            uint64_t totalNs, const VSAPI *vsapi) {
    VSMap *props = vsapi->getFramePropertiesRW(frame);
    for (size_t i = 0; i < stageCount; ++i) {
        const Stage stage = static_cast<Stage>(i);
        const std::string prefix = "_LGCR_" + std::string(stageName(stage));
        vsapi->mapSetFloat(props, (prefix + "_us").c_str(),
                           metrics.nanoseconds[i] / 1000.0, maReplace);
        vsapi->mapSetInt(props, (prefix + "_pixels").c_str(),
                         static_cast<int64_t>(metrics.pixels[i]), maReplace);
        vsapi->mapSetInt(props, (prefix + "_taps").c_str(),
                         static_cast<int64_t>(metrics.taps[i]), maReplace);
    }
    vsapi->mapSetFloat(props, "_LGCR_total_us", totalNs / 1000.0, maReplace);
    vsapi->mapSetInt(props, "_LGCR_output_pixels",
                     static_cast<int64_t>(metrics.outputPixels), maReplace);
    vsapi->mapSetInt(props, "_LGCR_taps_visited",
                     static_cast<int64_t>(metrics.tapsVisited), maReplace);
}

WeightTable buildBinomialWeights(int srcN, float shift) {
    WeightTable table;
    table.n = srcN / 2;
    const bool integerPhase = std::fabs(std::fabs(shift) - 0.5f) < 1e-6f;
    table.sup = integerPhase ? 5 : 4;
    table.start.resize(table.n);
    table.w.resize(size_t(table.n) * table.sup);

    static constexpr float integerWeights[5] = {
        1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f,
        4.0f / 16.0f, 1.0f / 16.0f
    };
    static constexpr float halfWeights[4] = {
        1.0f / 8.0f, 3.0f / 8.0f, 3.0f / 8.0f, 1.0f / 8.0f
    };

    for (int i = 0; i < table.n; ++i) {
        const float center = 2.0f * i + 0.5f + shift;
        if (integerPhase) {
            const int centerIndex = static_cast<int>(std::lround(center));
            table.start[i] = centerIndex - 2;
            std::copy(std::begin(integerWeights), std::end(integerWeights),
                      table.w.begin() + size_t(i) * table.sup);
        } else {
            table.start[i] = static_cast<int>(std::floor(center)) - 1;
            std::copy(std::begin(halfWeights), std::end(halfWeights),
                      table.w.begin() + size_t(i) * table.sup);
        }
    }
    return table;
}

WeightTable buildDownsampleWeights(int srcN, DownsampleKernel kernel, float shift) {
    if (kernel == DownsampleKernel::Binomial)
        return buildBinomialWeights(srcN, shift);
    const Kernel shared = kernel == DownsampleKernel::Spline36
        ? Kernel::Spline36 : Kernel::Lanczos;
    return buildWeights(srcN, srcN / 2, shared, 3.0, 0.0, 3.0, shift);
}

template <typename T>
struct SourcePlane {
    const uint8_t *data = nullptr;
    ptrdiff_t stride = 0;
    int w = 0;
    int h = 0;
    float scale = 1.0f;
    float offset = 0.0f;

    float at(int x, int y) const {
        x = std::clamp(x, 0, w - 1);
        y = std::clamp(y, 0, h - 1);
        const T *row = reinterpret_cast<const T *>(data + ptrdiff_t(y) * stride);
        return static_cast<float>(row[x]) * scale + offset;
    }
};

template <typename T>
struct OutputPlane {
    uint8_t *data = nullptr;
    ptrdiff_t stride = 0;
    int maxValue = 0;

    void writeRow(int y, const std::vector<float> &values) const {
        T *row = reinterpret_cast<T *>(data + ptrdiff_t(y) * stride);
        if constexpr (std::is_same_v<T, float>) {
            std::copy(values.begin(), values.end(), row);
        } else {
            const float scale = static_cast<float>(maxValue);
            const float offset = 0.5f * scale;
            for (size_t x = 0; x < values.size(); ++x) {
                const int value = static_cast<int>(values[x] * scale + offset + 0.5f);
                row[x] = static_cast<T>(std::clamp(value, 0, maxValue));
            }
        }
    }
};

template <typename T>
class BaselineRing {
public:
    BaselineRing(const SourcePlane<T> &u, const SourcePlane<T> &v,
                 const WeightTable &horizontal, const WeightTable &vertical)
        : u_(u), v_(v), horizontal_(horizontal), vertical_(vertical),
          capacity_(std::max(1, vertical.sup)), tags_(capacity_, -1),
          rowsU_(size_t(capacity_) * horizontal.n),
          rowsV_(size_t(capacity_) * horizontal.n) {}

    void makeRow(int outputY, std::vector<float> &baseU,
                 std::vector<float> &baseV) {
        const int width = horizontal_.n;
        std::fill(baseU.begin(), baseU.end(), 0.0f);
        std::fill(baseV.begin(), baseV.end(), 0.0f);
        const float *wy = vertical_.w.data() + size_t(outputY) * vertical_.sup;

        for (int tap = 0; tap < vertical_.sup; ++tap) {
            const int sy = std::clamp(vertical_.start[outputY] + tap, 0, u_.h - 1);
            const int slot = ensureRow(sy);
            const float *rowU = rowsU_.data() + size_t(slot) * width;
            const float *rowV = rowsV_.data() + size_t(slot) * width;
            const float weight = wy[tap];
            int x = 0;
#ifdef __AVX2__
            const __m256 vw = _mm256_set1_ps(weight);
            for (; x + 8 <= width; x += 8) {
                __m256 bu = _mm256_loadu_ps(baseU.data() + x);
                __m256 bv = _mm256_loadu_ps(baseV.data() + x);
                bu = _mm256_fmadd_ps(vw, _mm256_loadu_ps(rowU + x), bu);
                bv = _mm256_fmadd_ps(vw, _mm256_loadu_ps(rowV + x), bv);
                _mm256_storeu_ps(baseU.data() + x, bu);
                _mm256_storeu_ps(baseV.data() + x, bv);
            }
#endif
            for (; x < width; ++x) {
                baseU[x] += weight * rowU[x];
                baseV[x] += weight * rowV[x];
            }
        }
    }

private:
    int ensureRow(int sourceY) {
        const int slot = sourceY % capacity_;
        if (tags_[slot] == sourceY)
            return slot;
        tags_[slot] = sourceY;
        float *rowU = rowsU_.data() + size_t(slot) * horizontal_.n;
        float *rowV = rowsV_.data() + size_t(slot) * horizontal_.n;
        for (int x = 0; x < horizontal_.n; ++x) {
            const int start = horizontal_.start[x];
            const float *weights = horizontal_.w.data() + size_t(x) * horizontal_.sup;
#ifdef __AVX2__
            if (start >= 0 && start + horizontal_.sup <= u_.w) {
                const T *urow = reinterpret_cast<const T *>(
                    u_.data + ptrdiff_t(sourceY) * u_.stride);
                const T *vrow = reinterpret_cast<const T *>(
                    v_.data + ptrdiff_t(sourceY) * v_.stride);
                __m256 sumU = _mm256_setzero_ps();
                __m256 sumV = _mm256_setzero_ps();
                int tap = 0;
                for (; tap + 8 <= horizontal_.sup; tap += 8) {
                    const __m256 vw = _mm256_loadu_ps(weights + tap);
                    sumU = _mm256_fmadd_ps(vw, loadRaw8(urow + start + tap), sumU);
                    sumV = _mm256_fmadd_ps(vw, loadRaw8(vrow + start + tap), sumV);
                }
                float scalarU = NativeBackend::horizontalSum(sumU);
                float scalarV = NativeBackend::horizontalSum(sumV);
                for (; tap < horizontal_.sup; ++tap) {
                    scalarU += weights[tap] * rawValue(urow[start + tap]);
                    scalarV += weights[tap] * rawValue(vrow[start + tap]);
                }
                rowU[x] = scalarU * u_.scale + u_.offset;
                rowV[x] = scalarV * v_.scale + v_.offset;
                continue;
            }
#endif
            float sumU = 0.0f;
            float sumV = 0.0f;
            for (int tap = 0; tap < horizontal_.sup; ++tap) {
                const int sx = start + tap;
                sumU += weights[tap] * u_.at(sx, sourceY);
                sumV += weights[tap] * v_.at(sx, sourceY);
            }
            rowU[x] = sumU;
            rowV[x] = sumV;
        }
        return slot;
    }

    static float rawValue(float value) { return value; }
    static float rawValue(uint8_t value) { return static_cast<float>(value); }
    static float rawValue(uint16_t value) { return static_cast<float>(value); }

#ifdef __AVX2__
    static __m256 loadRaw8(const float *p) { return _mm256_loadu_ps(p); }
    static __m256 loadRaw8(const uint8_t *p) {
        const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p));
        return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
    }
    static __m256 loadRaw8(const uint16_t *p) {
        const __m128i words = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
        return _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(words));
    }
#endif

    const SourcePlane<T> &u_;
    const SourcePlane<T> &v_;
    const WeightTable &horizontal_;
    const WeightTable &vertical_;
    int capacity_ = 0;
    std::vector<int> tags_;
    std::vector<float> rowsU_;
    std::vector<float> rowsV_;
};

struct Direction {
    float x;
    float y;
};

constexpr std::array<Direction, 8> directions = {{
    { 1.0f, 0.0f },
    { 0.9238795f, 0.3826834f },
    { 0.7071068f, 0.7071068f },
    { 0.3826834f, 0.9238795f },
    { 0.0f, 1.0f },
    { -0.3826834f, 0.9238795f },
    { -0.7071068f, 0.7071068f },
    { -0.9238795f, 0.3826834f },
}};

struct LocalStats {
    int primary = 0;
    int secondary = 1;
    float secondaryMix = 0.0f;
    float gate = 0.0f;
    bool isotropic = true;
};

template <typename T>
LocalStats classify(const SourcePlane<T> &yPlane, const SourcePlane<T> &uPlane,
                    const SourcePlane<T> &vPlane, float centerX, float centerY,
                    int quality) {
    const int ix = static_cast<int>(std::floor(centerX + 0.5f));
    const int iy = static_cast<int>(std::floor(centerY + 0.5f));
    float jxx = 0.0f;
    float jxy = 0.0f;
    float jyy = 0.0f;
    float centerGx = 0.0f;
    float centerGy = 0.0f;
    if (quality == 0) {
        const float tl = yPlane.at(ix - 1, iy - 1);
        const float tc = yPlane.at(ix, iy - 1);
        const float tr = yPlane.at(ix + 1, iy - 1);
        const float ml = yPlane.at(ix - 1, iy);
        const float mr = yPlane.at(ix + 1, iy);
        const float bl = yPlane.at(ix - 1, iy + 1);
        const float bc = yPlane.at(ix, iy + 1);
        const float br = yPlane.at(ix + 1, iy + 1);
        centerGx = (tr + 2.0f * mr + br - tl - 2.0f * ml - bl) * 0.125f;
        centerGy = (bl + 2.0f * bc + br - tl - 2.0f * tc - tr) * 0.125f;
        jxx = centerGx * centerGx;
        jxy = centerGx * centerGy;
        jyy = centerGy * centerGy;
    } else {
        for (int k = -1; k <= 1; ++k) {
            const float gx = 0.5f * (yPlane.at(ix + 1, iy + k) -
                                     yPlane.at(ix - 1, iy + k));
            const float gy = 0.5f * (yPlane.at(ix, iy + k + 1) -
                                     yPlane.at(ix, iy + k - 1));
            jxx += gx * gx;
            jxy += gx * gy;
            jyy += gy * gy;
            if (k == 0) {
                centerGx = gx;
                centerGy = gy;
            }
        }
    }

    const float trace = jxx + jyy;
    if (trace < 1.0e-6f)
        return {};
    const float difference = jxx - jyy;
    const float discriminant2 = difference * difference + 4.0f * jxy * jxy;
    const float coherence2 = discriminant2 / (trace * trace + 1e-12f);

    LocalStats stats;
    float best = -1.0f;
    float second = -1.0f;
    const int step = quality == 0 ? 2 : 1;
    for (int d = 0; d < 8; d += step) {
        const Direction n = directions[d];
        const float response = n.x * n.x * jxx +
            2.0f * n.x * n.y * jxy + n.y * n.y * jyy;
        if (response > best) {
            second = best;
            stats.secondary = stats.primary;
            best = response;
            stats.primary = d;
        } else if (response > second) {
            second = response;
            stats.secondary = d;
        }
    }
    stats.isotropic = coherence2 < (quality == 0 ? 0.20f : 0.10f);
    if (quality > 0 && best > 1e-12f) {
        constexpr float adjacentAtExact = 0.8535534f;
        const float ratio = std::clamp(second / best, 0.0f, 1.0f);
        stats.secondaryMix = 0.5f * std::clamp(
            (ratio - adjacentAtExact) / (1.0f - adjacentAtExact), 0.0f, 1.0f);
    }

    const float ugx = 0.5f * (uPlane.at(ix + 1, iy) - uPlane.at(ix - 1, iy));
    const float ugy = 0.5f * (uPlane.at(ix, iy + 1) - uPlane.at(ix, iy - 1));
    const float vgx = 0.5f * (vPlane.at(ix + 1, iy) - vPlane.at(ix - 1, iy));
    const float vgy = 0.5f * (vPlane.at(ix, iy + 1) - vPlane.at(ix, iy - 1));
    const float chromaEnergy = ugx * ugx + ugy * ugy + vgx * vgx + vgy * vgy;
    const Direction normal = directions[stats.primary];
    const float un = ugx * normal.x + ugy * normal.y;
    const float vn = vgx * normal.x + vgy * normal.y;
    const float alignedEnergy = un * un + vn * vn;
    const int normalStepX = static_cast<int>(std::lround(normal.x));
    const int normalStepY = static_cast<int>(std::lround(normal.y));
    const float centerU = uPlane.at(ix, iy);
    const float centerV = vPlane.at(ix, iy);
    const float curvatureU = uPlane.at(ix + normalStepX, iy + normalStepY) -
        2.0f * centerU + uPlane.at(ix - normalStepX, iy - normalStepY);
    const float curvatureV = vPlane.at(ix + normalStepX, iy + normalStepY) -
        2.0f * centerV + vPlane.at(ix - normalStepX, iy - normalStepY);
    const float curvatureEnergy = curvatureU * curvatureU + curvatureV * curvatureV;
    const float sharpnessGate = curvatureEnergy /
        (curvatureEnergy + chromaEnergy + 1.0e-12f);

    // All comparisons and gates use squared quantities. These thresholds are
    // frozen in normalized sample units and intentionally remain internal.
    const float lumaGate = trace / (trace + 3.0e-5f);
    const float confidenceGate = coherence2 / (coherence2 + 0.08f);
    const float chromaGate = chromaEnergy / (chromaEnergy + 2.0e-5f);
    const float directionGate = alignedEnergy / (chromaEnergy + 1.0e-12f);
    const float centerEnergy = centerGx * centerGx + centerGy * centerGy;
    const float positionGate = centerEnergy / (centerEnergy + 1.0e-5f);
    stats.gate = std::clamp(lumaGate * confidenceGate * chromaGate *
                            directionGate * positionGate * sharpnessGate,
                            0.0f, 1.0f);
    if (quality == 0)
        stats.gate *= 0.985f;
    else if (quality == 2)
        stats.gate = std::min(1.0f, stats.gate * 1.01f);
    if (sharpnessGate < (quality == 0 ? 0.20f : 0.12f))
        stats.gate = 0.0f;
    return stats;
}

struct Accumulator {
    float weight = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
};

struct Envelope {
    Direction normal{};
    int radius = 0;
    float anisotropy = 0.0f;
    bool isotropic = false;
};

float spatialWeight(const Envelope &envelope, float dx, float dy) {
    const float radius2 = dx * dx + dy * dy;
    if (envelope.isotropic)
        return 1.0f / (1.0f + 0.42f * radius2);
    const float normalDistance = dx * envelope.normal.x + dy * envelope.normal.y;
    const float tangentDistance = -dx * envelope.normal.y + dy * envelope.normal.x;
    return 1.0f / (1.0f + (0.9f + envelope.anisotropy) *
        normalDistance * normalDistance + 0.16f * tangentDistance * tangentDistance);
}

template <typename T, size_t N>
std::array<Accumulator, N> accumulateGuided(
    const SourcePlane<T> &yPlane, const SourcePlane<T> &uPlane,
    const SourcePlane<T> &vPlane, float centerX, float centerY,
    const std::array<Envelope, N> &envelopes) {
    std::array<Accumulator, N> accum{};
    int maxRadius = 0;
    for (const Envelope &envelope : envelopes)
        maxRadius = std::max(maxRadius, envelope.radius);
    const int x0 = static_cast<int>(std::ceil(centerX - maxRadius));
    const int x1 = static_cast<int>(std::floor(centerX + maxRadius));
    const int y0 = static_cast<int>(std::ceil(centerY - maxRadius));
    const int y1 = static_cast<int>(std::floor(centerY + maxRadius));

    const int cx0 = static_cast<int>(std::floor(centerX));
    const int cy0 = static_cast<int>(std::floor(centerY));
    const float fx = centerX - cx0;
    const float fy = centerY - cy0;
    const float ya = yPlane.at(cx0, cy0);
    const float yb = yPlane.at(cx0 + 1, cy0);
    const float yc = yPlane.at(cx0, cy0 + 1);
    const float yd = yPlane.at(cx0 + 1, cy0 + 1);
    const float centerLuma = ya + (yb - ya) * fx + (yc - ya) * fy +
        (ya - yb - yc + yd) * fx * fy;
    constexpr float rangeScale2 = 2.25e-4f;

    for (int sy = y0; sy <= y1; ++sy) {
        for (int sx = x0; sx <= x1; ++sx) {
            const float dx = sx - centerX;
            const float dy = sy - centerY;
            const float sourceLuma = yPlane.at(sx, sy);
            const float delta = sourceLuma - centerLuma;
            const float rangeWeight = rangeScale2 / (rangeScale2 + delta * delta);
            const float sourceU = uPlane.at(sx, sy);
            const float sourceV = vPlane.at(sx, sy);
            for (size_t i = 0; i < N; ++i) {
                if (std::fabs(dx) > envelopes[i].radius ||
                    std::fabs(dy) > envelopes[i].radius)
                    continue;
                const float weight = rangeWeight * spatialWeight(envelopes[i], dx, dy);
                accum[i].weight += weight;
                accum[i].u += weight * sourceU;
                accum[i].v += weight * sourceV;
            }
        }
    }
    return accum;
}

struct GuidedPixel {
    float u = 0.0f;
    float v = 0.0f;
    float gate = 0.0f;
};

float normalizedValue(float sum, float weight, float fallback) {
    return weight > 1e-12f ? sum / weight : fallback;
}

template <typename T>
GuidedPixel makeGuidedPixel(const SourcePlane<T> &yPlane,
                            const SourcePlane<T> &uPlane,
                            const SourcePlane<T> &vPlane,
                            float centerX, float centerY, int quality,
                            float baseU, float baseV) {
    const LocalStats stats = classify(yPlane, uPlane, vPlane,
                                      centerX, centerY, quality);
    GuidedPixel result{ baseU, baseV, stats.gate };
    if (stats.gate <= guidedActivationThreshold) {
        result.gate = 0.0f;
        return result;
    }

    if (quality == 0) {
        const std::array<Envelope, 1> envelopes = {{
            { directions[stats.primary], 2, 1.8f, stats.isotropic }
        }};
        const auto accum = accumulateGuided(yPlane, uPlane, vPlane,
                                            centerX, centerY, envelopes);
        result.u = normalizedValue(accum[0].u, accum[0].weight, baseU);
        result.v = normalizedValue(accum[0].v, accum[0].weight, baseV);
        return result;
    }

    const std::array<Envelope, 2> envelopes = {{
        { directions[stats.primary], 3, 2.2f, stats.isotropic },
        { directions[stats.secondary], 3, 2.2f, stats.isotropic },
    }};
    const auto accum = accumulateGuided(yPlane, uPlane, vPlane,
                                        centerX, centerY, envelopes);
    const float u0 = normalizedValue(accum[0].u, accum[0].weight, baseU);
    const float v0 = normalizedValue(accum[0].v, accum[0].weight, baseV);
    const float u1 = normalizedValue(accum[1].u, accum[1].weight, baseU);
    const float v1 = normalizedValue(accum[1].v, accum[1].weight, baseV);
    result.u = u0 + stats.secondaryMix * (u1 - u0);
    result.v = v0 + stats.secondaryMix * (v1 - v0);
    return result;
}

struct CandidateRow {
    int tag = -1;
    std::vector<float> baseU;
    std::vector<float> baseV;
    std::vector<float> gate;
    std::array<std::vector<float>, highCandidateCount> u;
    std::array<std::vector<float>, highCandidateCount> v;

    explicit CandidateRow(int width = 0)
        : baseU(width), baseV(width), gate(width) {
        for (int i = 0; i < highCandidateCount; ++i) {
            u[i].resize(width);
            v[i].resize(width);
        }
    }
};

template <typename T>
void generateCandidateRow(CandidateRow &row, int outputY,
                          const SourcePlane<T> &yPlane,
                          const SourcePlane<T> &uPlane,
                          const SourcePlane<T> &vPlane,
                          float shiftX, float shiftY) {
    row.tag = outputY;
    const float centerY = 2.0f * outputY + 0.5f + shiftY;
    for (size_t x = 0; x < row.baseU.size(); ++x) {
        const float centerX = 2.0f * static_cast<float>(x) + 0.5f + shiftX;
        const LocalStats stats = classify(yPlane, uPlane, vPlane,
                                          centerX, centerY, 2);
        row.gate[x] = stats.gate > guidedActivationThreshold ? stats.gate : 0.0f;
        if (row.gate[x] == 0.0f) {
            for (int candidate = 0; candidate < highCandidateCount; ++candidate) {
                row.u[candidate][x] = row.baseU[x];
                row.v[candidate][x] = row.baseV[x];
            }
            continue;
        }

        const std::array<Envelope, 5> envelopes = {{
            { directions[stats.primary], 3, 2.2f, stats.isotropic },
            { directions[stats.secondary], 3, 2.2f, stats.isotropic },
            { directions[stats.primary], 4, 2.7f, stats.isotropic },
            { directions[stats.secondary], 4, 2.7f, stats.isotropic },
            { directions[stats.primary], 4, 0.0f, true },
        }};
        const auto accum = accumulateGuided(yPlane, uPlane, vPlane,
                                            centerX, centerY, envelopes);
        const float u0 = normalizedValue(accum[0].u, accum[0].weight, row.baseU[x]);
        const float v0 = normalizedValue(accum[0].v, accum[0].weight, row.baseV[x]);
        const float u1 = normalizedValue(accum[1].u, accum[1].weight, row.baseU[x]);
        const float v1 = normalizedValue(accum[1].v, accum[1].weight, row.baseV[x]);
        row.u[0][x] = u0 + stats.secondaryMix * (u1 - u0);
        row.v[0][x] = v0 + stats.secondaryMix * (v1 - v0);
        row.u[1][x] = normalizedValue(accum[2].u, accum[2].weight, row.baseU[x]);
        row.v[1][x] = normalizedValue(accum[2].v, accum[2].weight, row.baseV[x]);
        row.u[2][x] = normalizedValue(accum[3].u, accum[3].weight, row.baseU[x]);
        row.v[2][x] = normalizedValue(accum[3].v, accum[3].weight, row.baseV[x]);
        row.u[3][x] = normalizedValue(accum[4].u, accum[4].weight, row.baseU[x]);
        row.v[3][x] = normalizedValue(accum[4].v, accum[4].weight, row.baseV[x]);
    }
}

const CandidateRow &candidateRowAt(const std::array<CandidateRow, 3> &ring,
                                   int y, int height) {
    y = std::clamp(y, 0, height - 1);
    return ring[y % 3];
}

template <typename T>
void scoreCandidateRow(const std::array<CandidateRow, 3> &ring, int outputY,
                       const SourcePlane<T> &yPlane,
                       const SourcePlane<T> &uPlane,
                       const SourcePlane<T> &vPlane,
                       float shiftX, float shiftY, float strength,
                       std::vector<float> &outputU, std::vector<float> &outputV) {
    const int width = static_cast<int>(outputU.size());
    const int height = yPlane.h / 2;
    const CandidateRow &center = candidateRowAt(ring, outputY, height);
    for (int x = 0; x < width; ++x) {
        if (center.gate[x] == 0.0f) {
            outputU[x] = center.baseU[x];
            outputV[x] = center.baseV[x];
            continue;
        }
        std::array<float, highCandidateCount> scores{};
        for (int hy = 2 * outputY; hy <= 2 * outputY + 1; ++hy) {
            for (int hx = 2 * x; hx <= 2 * x + 1; ++hx) {
                const float gx = 0.5f * (yPlane.at(hx + 1, hy) -
                                         yPlane.at(hx - 1, hy));
                const float gy = 0.5f * (yPlane.at(hx, hy + 1) -
                                         yPlane.at(hx, hy - 1));
                const float edge2 = gx * gx + gy * gy;
                const float edgeWeight = 0.125f + edge2 / (edge2 + 1.0e-5f);
                const float lowX = (hx - 0.5f - shiftX) * 0.5f;
                const float lowY = (hy - 0.5f - shiftY) * 0.5f;
                const float sourceU = uPlane.at(hx, hy);
                const float sourceV = vPlane.at(hx, hy);
                const int rawX0 = static_cast<int>(std::floor(lowX));
                const int rawY0 = static_cast<int>(std::floor(lowY));
                const int x0 = std::clamp(rawX0, 0, width - 1);
                const int x1 = std::clamp(rawX0 + 1, 0, width - 1);
                const CandidateRow &row0 = candidateRowAt(ring, rawY0, height);
                const CandidateRow &row1 = candidateRowAt(ring, rawY0 + 1, height);
                const float fx = lowX - rawX0;
                const float fy = lowY - rawY0;
                const float weights[4] = {
                    (1.0f - fx) * (1.0f - fy), fx * (1.0f - fy),
                    (1.0f - fx) * fy, fx * fy
                };
                const CandidateRow *rows[4] = { &row0, &row0, &row1, &row1 };
                const int columns[4] = { x0, x1, x0, x1 };
                for (int candidate = 0; candidate < highCandidateCount; ++candidate) {
                    float reconstructedU = 0.0f;
                    float reconstructedV = 0.0f;
                    for (int corner = 0; corner < 4; ++corner) {
                        const CandidateRow &sampleRow = *rows[corner];
                        const int column = columns[corner];
                        float valueU = sampleRow.baseU[column];
                        float valueV = sampleRow.baseV[column];
                        valueU += sampleRow.gate[column] *
                            (sampleRow.u[candidate][column] - valueU);
                        valueV += sampleRow.gate[column] *
                            (sampleRow.v[candidate][column] - valueV);
                        reconstructedU += weights[corner] * valueU;
                        reconstructedV += weights[corner] * valueV;
                    }
                    const float du = reconstructedU - sourceU;
                    const float dv = reconstructedV - sourceV;
                    scores[candidate] += edgeWeight * (du * du + dv * dv);
                }
            }
        }
        int best = 0;
        for (int candidate = 1; candidate < highCandidateCount; ++candidate)
            if (scores[candidate] < scores[best])
                best = candidate;
        const float baseU = center.baseU[x];
        const float baseV = center.baseV[x];
        outputU[x] = baseU + strength * center.gate[x] *
            (center.u[best][x] - baseU);
        outputV[x] = baseV + strength * center.gate[x] *
            (center.v[best][x] - baseV);
    }
}

template <typename T>
void processFrame(const DownsampleData &d, const VSFrame *src, VSFrame *dst,
                  PipelineMetrics *metrics, const VSAPI *vsapi) {
    const int width = d.viIn->width;
    const int height = d.viIn->height;
    const int chromaWidth = width / 2;
    const int chromaHeight = height / 2;
    const bool isFloat = std::is_same_v<T, float>;
    const int maxValue = isFloat ? 0 : (1 << d.viIn->format.bitsPerSample) - 1;
    const float inputScale = isFloat ? 1.0f : 1.0f / maxValue;
    const float chromaOffset = isFloat ? 0.0f : -0.5f;

    const SourcePlane<T> yPlane{
        vsapi->getReadPtr(src, 0), vsapi->getStride(src, 0),
        width, height, inputScale, 0.0f
    };
    const SourcePlane<T> uPlane{
        vsapi->getReadPtr(src, 1), vsapi->getStride(src, 1),
        width, height, inputScale, chromaOffset
    };
    const SourcePlane<T> vPlane{
        vsapi->getReadPtr(src, 2), vsapi->getStride(src, 2),
        width, height, inputScale, chromaOffset
    };
    const OutputPlane<T> outputU{
        vsapi->getWritePtr(dst, 1), vsapi->getStride(dst, 1), maxValue
    };
    const OutputPlane<T> outputV{
        vsapi->getWritePtr(dst, 2), vsapi->getStride(dst, 2), maxValue
    };

    BaselineRing<T> baseline(uPlane, vPlane, d.xWeights, d.yWeights);
    std::vector<float> baseU(chromaWidth), baseV(chromaWidth);
    std::vector<float> resultU(chromaWidth), resultV(chromaWidth);

    if (d.strength == 0.0f || d.quality < 2) {
        for (int y = 0; y < chromaHeight; ++y) {
            auto start = Clock::now();
            baseline.makeRow(y, baseU, baseV);
            addElapsed(metrics, Stage::DownsampleBase, start);

            if (d.strength == 0.0f) {
                resultU = baseU;
                resultV = baseV;
            } else {
                start = Clock::now();
                const float centerY = 2.0f * y + 0.5f + d.shiftY;
                for (int x = 0; x < chromaWidth; ++x) {
                    const float centerX = 2.0f * x + 0.5f + d.shiftX;
                    const GuidedPixel guided = makeGuidedPixel(
                        yPlane, uPlane, vPlane, centerX, centerY,
                        d.quality, baseU[x], baseV[x]);
                    resultU[x] = baseU[x] + d.strength * guided.gate *
                        (guided.u - baseU[x]);
                    resultV[x] = baseV[x] + d.strength * guided.gate *
                        (guided.v - baseV[x]);
                }
                addElapsed(metrics, Stage::DownsampleGuide, start);
            }

            start = Clock::now();
            outputU.writeRow(y, resultU);
            outputV.writeRow(y, resultV);
            addElapsed(metrics, Stage::DownsampleOutput, start);
        }
    } else {
        std::array<CandidateRow, 3> ring = {
            CandidateRow(chromaWidth), CandidateRow(chromaWidth), CandidateRow(chromaWidth)
        };
        auto generate = [&](int y) {
            CandidateRow &row = ring[y % 3];
            auto start = Clock::now();
            baseline.makeRow(y, row.baseU, row.baseV);
            addElapsed(metrics, Stage::DownsampleBase, start);
            start = Clock::now();
            generateCandidateRow(row, y, yPlane, uPlane, vPlane,
                                 d.shiftX, d.shiftY);
            addElapsed(metrics, Stage::DownsampleGuide, start);
        };
        auto scoreAndWrite = [&](int y) {
            auto start = Clock::now();
            scoreCandidateRow(ring, y, yPlane, uPlane, vPlane,
                              d.shiftX, d.shiftY, d.strength, resultU, resultV);
            addElapsed(metrics, Stage::DownsampleCandidateScore, start);
            start = Clock::now();
            outputU.writeRow(y, resultU);
            outputV.writeRow(y, resultV);
            addElapsed(metrics, Stage::DownsampleOutput, start);
        };

        for (int y = 0; y < chromaHeight; ++y) {
            generate(y);
            if (y > 0)
                scoreAndWrite(y - 1);
        }
        scoreAndWrite(chromaHeight - 1);
    }

    if (metrics) {
        const uint64_t pixels = 2 * uint64_t(chromaWidth) * chromaHeight;
        const uint64_t baseTaps = 2 * (
            uint64_t(height) * chromaWidth * d.xWeights.sup +
            uint64_t(chromaHeight) * chromaWidth * d.yWeights.sup);
        metrics->outputPixels = uint64_t(width) * height + pixels;
        metrics->tapsVisited += baseTaps;
        metrics->addWork(Stage::DownsampleBase, pixels, baseTaps);
        metrics->addWork(Stage::DownsampleOutput, pixels);
        if (d.strength > 0.0f) {
            const int diameter = 2 * (d.quality + 2) + 1;
            const int candidates = d.quality == 0 ? 1 : d.quality == 1 ? 2 : 5;
            metrics->addWork(Stage::DownsampleGuide, pixels,
                             uint64_t(chromaWidth) * chromaHeight *
                                 diameter * diameter * candidates);
        }
        if (d.quality == 2 && d.strength > 0.0f)
            metrics->addWork(Stage::DownsampleCandidateScore, pixels,
                             uint64_t(chromaWidth) * chromaHeight * 16);
    }
}

const VSFrame *VS_CC downsampleGetFrame(int n, int activationReason,
                                       void *instanceData, void **,
                                       VSFrameContext *frameCtx, VSCore *core,
                                       const VSAPI *vsapi) {
    DownsampleData *d = static_cast<DownsampleData *>(instanceData);
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady)
        return nullptr;

    const auto frameStart = Clock::now();
    PipelineMetrics frameMetrics;
    PipelineMetrics *metrics = downsampleProfilingEnabled() ? &frameMetrics : nullptr;
    const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
    const VSFrame *planeSources[3] = { src, nullptr, nullptr };
    const int sourcePlanes[3] = { 0, 0, 0 };
    VSFrame *dst = vsapi->newVideoFrame2(&d->viOut.format, d->viOut.width,
                                        d->viOut.height, planeSources,
                                        sourcePlanes, src, core);
    VSMap *props = vsapi->getFramePropertiesRW(dst);
    vsapi->mapSetInt(props, "_ChromaLocation", d->chromaLocation, maReplace);

    const VSVideoFormat *format = vsapi->getVideoFrameFormat(src);
    if (format->sampleType == stFloat)
        processFrame<float>(*d, src, dst, metrics, vsapi);
    else if (format->bytesPerSample == 1)
        processFrame<uint8_t>(*d, src, dst, metrics, vsapi);
    else
        processFrame<uint16_t>(*d, src, dst, metrics, vsapi);

    if (metrics) {
        const auto totalNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - frameStart).count();
        writeDownsampleProfile(dst, *metrics, static_cast<uint64_t>(totalNs), vsapi);
    }
    vsapi->freeFrame(src);
    return dst;
}

void VS_CC downsampleFree(void *instanceData, VSCore *, const VSAPI *vsapi) {
    DownsampleData *d = static_cast<DownsampleData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

void VS_CC downsampleCreate(const VSMap *in, VSMap *out, void *, VSCore *core,
                            const VSAPI *vsapi) {
    auto d = std::make_unique<DownsampleData>();
    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->viIn = vsapi->getVideoInfo(d->node);
    const VSVideoFormat *format = &d->viIn->format;

    auto fail = [&](const char *message) {
        vsapi->mapSetError(out, message);
        vsapi->freeNode(d->node);
    };

    const bool integerFormat = format->sampleType == stInteger &&
        format->bitsPerSample >= 8 && format->bitsPerSample <= 16;
    const bool floatFormat = format->sampleType == stFloat &&
        format->bitsPerSample == 32;
    if (format->colorFamily != cfYUV || format->numPlanes != 3 ||
        format->subSamplingW != 0 || format->subSamplingH != 0 ||
        (!integerFormat && !floatFormat)) {
        fail("Downsample: input must be planar YUV444 (8-16 bit integer or float32)");
        return;
    }
    if (d->viIn->width <= 0 || d->viIn->height <= 0 ||
        (d->viIn->width & 1) != 0 || (d->viIn->height & 1) != 0) {
        fail("Downsample: constant even input width and height are required");
        return;
    }

    int err = 0;
    const int64_t quality = vsapi->mapGetIntSaturated(in, "quality", 0, &err);
    d->quality = err ? 0 : static_cast<int>(quality);
    if (d->quality < 0 || d->quality > 2) {
        fail("Downsample: quality must be 0, 1, or 2");
        return;
    }

    const char *kernelName = vsapi->mapGetData(in, "kernel", 0, &err);
    const std::string kernel = err ? "spline36" : kernelName;
    if (kernel == "spline36")
        d->kernel = DownsampleKernel::Spline36;
    else if (kernel == "lanczos3")
        d->kernel = DownsampleKernel::Lanczos3;
    else if (kernel == "binomial")
        d->kernel = DownsampleKernel::Binomial;
    else {
        fail("Downsample: kernel must be \"spline36\", \"lanczos3\", or \"binomial\"");
        return;
    }

    const double strength = vsapi->mapGetFloatSaturated(in, "strength", 0, &err);
    d->strength = err ? 1.0f : static_cast<float>(strength);
    if (!std::isfinite(d->strength) || d->strength < 0.0f || d->strength > 1.0f) {
        fail("Downsample: strength must be in the range 0..1");
        return;
    }

    const char *locationName = vsapi->mapGetData(in, "loc", 0, &err);
    const std::string location = err ? "left" : locationName;
    if (location == "left") {
        d->shiftX = -0.5f; d->shiftY = 0.0f; d->chromaLocation = 0;
    } else if (location == "center") {
        d->shiftX = 0.0f; d->shiftY = 0.0f; d->chromaLocation = 1;
    } else if (location == "topleft") {
        d->shiftX = -0.5f; d->shiftY = -0.5f; d->chromaLocation = 2;
    } else if (location == "top") {
        d->shiftX = 0.0f; d->shiftY = -0.5f; d->chromaLocation = 3;
    } else if (location == "bottomleft") {
        d->shiftX = -0.5f; d->shiftY = 0.5f; d->chromaLocation = 4;
    } else if (location == "bottom") {
        d->shiftX = 0.0f; d->shiftY = 0.5f; d->chromaLocation = 5;
    } else {
        fail("Downsample: loc must be left, center, topleft, top, bottomleft, or bottom");
        return;
    }

    d->xWeights = buildDownsampleWeights(d->viIn->width, d->kernel, d->shiftX);
    d->yWeights = buildDownsampleWeights(d->viIn->height, d->kernel, d->shiftY);
    d->viOut = *d->viIn;
    if (!vsapi->queryVideoFormat(&d->viOut.format, cfYUV, format->sampleType,
                                 format->bitsPerSample, 1, 1, core)) {
        fail("Downsample: failed to query the YUV420 output format");
        return;
    }

    DownsampleData *data = d.release();
    const VSFilterDependency dependencies[] = { { data->node, rpGeneral } };
    vsapi->createVideoFilter(out, "Downsample", &data->viOut,
                             downsampleGetFrame, downsampleFree,
                             fmParallelRequests, dependencies, 1, data, core);
}

} // namespace

void registerDownsample(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->registerFunction(
        "Downsample",
        "clip:vnode;quality:int:opt;kernel:data:opt;strength:float:opt;loc:data:opt",
        "clip:vnode", downsampleCreate, nullptr, plugin);
}

} // namespace lgcr
