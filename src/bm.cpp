#include "lgcr.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace lgcr {

namespace {

constexpr int patchSize = 8;
constexpr int blockStep = 8;
constexpr int searchRadius = 8;
constexpr int searchStep = 4;
constexpr int maxGroup = 4;
constexpr float invSqrt2 = 0.7071067811865475f;

struct PatchMatch {
    float cost = std::numeric_limits<float>::infinity();
    int x = 0;
    int y = 0;
};

void insertMatch(std::array<PatchMatch, maxGroup> &matches, int &count,
                 const PatchMatch &candidate) {
    int position = std::min(count, maxGroup - 1);
    if (count == maxGroup && candidate.cost >= matches[maxGroup - 1].cost)
        return;
    if (count < maxGroup)
        ++count;
    while (position > 0 && candidate.cost < matches[position - 1].cost) {
        if (position < maxGroup)
            matches[position] = matches[position - 1];
        --position;
    }
    matches[position] = candidate;
}

void haarForward(float *values, int count) {
    std::array<float, patchSize> temporary{};
    for (int length = count; length > 1; length /= 2) {
        const int half = length / 2;
        for (int i = 0; i < half; ++i) {
            const float a = values[2 * i];
            const float b = values[2 * i + 1];
            temporary[i] = (a + b) * invSqrt2;
            temporary[half + i] = (a - b) * invSqrt2;
        }
        std::copy_n(temporary.data(), length, values);
    }
}

void haarInverse(float *values, int count) {
    std::array<float, patchSize> temporary{};
    for (int length = 2; length <= count; length *= 2) {
        const int half = length / 2;
        for (int i = 0; i < half; ++i) {
            const float a = values[i];
            const float b = values[half + i];
            temporary[2 * i] = (a + b) * invSqrt2;
            temporary[2 * i + 1] = (a - b) * invSqrt2;
        }
        std::copy_n(temporary.data(), length, values);
    }
}

#ifndef __AVX2__
void haar2DForward(float *patch) {
    std::array<float, patchSize> column{};
    for (int y = 0; y < patchSize; ++y)
        haarForward(patch + y * patchSize, patchSize);
    for (int x = 0; x < patchSize; ++x) {
        for (int y = 0; y < patchSize; ++y)
            column[y] = patch[y * patchSize + x];
        haarForward(column.data(), patchSize);
        for (int y = 0; y < patchSize; ++y)
            patch[y * patchSize + x] = column[y];
    }
}

void haar2DInverse(float *patch) {
    std::array<float, patchSize> column{};
    for (int x = 0; x < patchSize; ++x) {
        for (int y = 0; y < patchSize; ++y)
            column[y] = patch[y * patchSize + x];
        haarInverse(column.data(), patchSize);
        for (int y = 0; y < patchSize; ++y)
            patch[y * patchSize + x] = column[y];
    }
    for (int y = 0; y < patchSize; ++y)
        haarInverse(patch + y * patchSize, patchSize);
}
#endif

#ifdef __AVX2__
inline __m256 loadChromaGroup(const float *u, const float *v, int offset) {
    const __m128 uValues = _mm_loadu_ps(u + offset * maxGroup);
    const __m128 vValues = _mm_loadu_ps(v + offset * maxGroup);
    return _mm256_insertf128_ps(_mm256_castps128_ps256(uValues), vValues, 1);
}

inline void storeChromaGroup(float *u, float *v, int offset, __m256 values) {
    _mm_storeu_ps(u + offset * maxGroup, _mm256_castps256_ps128(values));
    _mm_storeu_ps(v + offset * maxGroup, _mm256_extractf128_ps(values, 1));
}

void haarLanesForward(__m256 *values) {
    __m256 temporary[patchSize];
    const __m256 scale = _mm256_set1_ps(invSqrt2);
    for (int length = patchSize; length > 1; length /= 2) {
        const int half = length / 2;
        for (int i = 0; i < half; ++i) {
            const __m256 a = values[2 * i];
            const __m256 b = values[2 * i + 1];
            temporary[i] = _mm256_mul_ps(_mm256_add_ps(a, b), scale);
            temporary[half + i] = _mm256_mul_ps(_mm256_sub_ps(a, b), scale);
        }
        for (int i = 0; i < length; ++i)
            values[i] = temporary[i];
    }
}

void haarLanesInverse(__m256 *values) {
    __m256 temporary[patchSize];
    const __m256 scale = _mm256_set1_ps(invSqrt2);
    for (int length = 2; length <= patchSize; length *= 2) {
        const int half = length / 2;
        for (int i = 0; i < half; ++i) {
            const __m256 a = values[i];
            const __m256 b = values[half + i];
            temporary[2 * i] = _mm256_mul_ps(_mm256_add_ps(a, b), scale);
            temporary[2 * i + 1] = _mm256_mul_ps(_mm256_sub_ps(a, b), scale);
        }
        for (int i = 0; i < length; ++i)
            values[i] = temporary[i];
    }
}
#endif

void haar2DForwardGroups(float *u, float *v, int groupCount) {
#ifdef __AVX2__
    (void)groupCount;
    __m256 values[patchSize];
    for (int y = 0; y < patchSize; ++y) {
        for (int x = 0; x < patchSize; ++x)
            values[x] = loadChromaGroup(u, v, y * patchSize + x);
        haarLanesForward(values);
        for (int x = 0; x < patchSize; ++x)
            storeChromaGroup(u, v, y * patchSize + x, values[x]);
    }
    for (int x = 0; x < patchSize; ++x) {
        for (int y = 0; y < patchSize; ++y)
            values[y] = loadChromaGroup(u, v, y * patchSize + x);
        haarLanesForward(values);
        for (int y = 0; y < patchSize; ++y)
            storeChromaGroup(u, v, y * patchSize + x, values[y]);
    }
#else
    std::array<float, patchSize * patchSize> patch{};
    for (float *plane : { u, v }) {
        for (int k = 0; k < groupCount; ++k) {
            for (int offset = 0; offset < patchSize * patchSize; ++offset)
                patch[offset] = plane[offset * maxGroup + k];
            haar2DForward(patch.data());
            for (int offset = 0; offset < patchSize * patchSize; ++offset)
                plane[offset * maxGroup + k] = patch[offset];
        }
    }
#endif
}

void haar2DInverseGroups(float *u, float *v, int groupCount) {
#ifdef __AVX2__
    (void)groupCount;
    __m256 values[patchSize];
    for (int x = 0; x < patchSize; ++x) {
        for (int y = 0; y < patchSize; ++y)
            values[y] = loadChromaGroup(u, v, y * patchSize + x);
        haarLanesInverse(values);
        for (int y = 0; y < patchSize; ++y)
            storeChromaGroup(u, v, y * patchSize + x, values[y]);
    }
    for (int y = 0; y < patchSize; ++y) {
        for (int x = 0; x < patchSize; ++x)
            values[x] = loadChromaGroup(u, v, y * patchSize + x);
        haarLanesInverse(values);
        for (int x = 0; x < patchSize; ++x)
            storeChromaGroup(u, v, y * patchSize + x, values[x]);
    }
#else
    std::array<float, patchSize * patchSize> patch{};
    for (float *plane : { u, v }) {
        for (int k = 0; k < groupCount; ++k) {
            for (int offset = 0; offset < patchSize * patchSize; ++offset)
                patch[offset] = plane[offset * maxGroup + k];
            haar2DInverse(patch.data());
            for (int offset = 0; offset < patchSize * patchSize; ++offset)
                plane[offset * maxGroup + k] = patch[offset];
        }
    }
#endif
}

int groupSizeFor(int eligible) {
    if (eligible >= 4) return 4;
    if (eligible >= 2) return 2;
    return 1;
}

std::vector<int> blockStarts(int extent) {
    std::vector<int> starts;
    if (extent < patchSize)
        return starts;
    const int last = extent - patchSize;
    for (int position = 0; position <= last; position += blockStep)
        starts.push_back(position);
    if (starts.empty() || starts.back() != last)
        starts.push_back(last);
    return starts;
}

float patchCost(const Plane &guide, const Plane &u, const Plane &v,
                int ax, int ay, int bx, int by) {
    float cost = 0.0f;
    int samples = 0;
    for (int py = 0; py < patchSize; py += 2) {
        const float *ga = guide.row(ay + py) + ax;
        const float *gb = guide.row(by + py) + bx;
        const float *ua = u.row(ay + py) + ax;
        const float *ub = u.row(by + py) + bx;
        const float *va = v.row(ay + py) + ax;
        const float *vb = v.row(by + py) + bx;
        for (int px = 0; px < patchSize; px += 2) {
            const float dg = ga[px] - gb[px];
            const float du = ua[px] - ub[px];
            const float dv = va[px] - vb[px];
            cost += dg * dg + 0.25f * (du * du + dv * dv);
            ++samples;
        }
    }
    return cost / samples;
}

} // namespace

void collaborativeChromaFilter(const Plane &guide, Plane &u, Plane &v,
                               float guideSigma, PipelineMetrics *metrics) {
    if (u.w != v.w || u.h != v.h || u.w < patchSize || u.h < patchSize)
        return;

    Plane resizedGuide;
    const Plane *matchGuide = &guide;
    if (guide.w != u.w || guide.h != u.h) {
        resizedGuide = Plane(u.w, u.h);
        const double sx = double(guide.w) / u.w;
        const double sy = double(guide.h) / u.h;
        for (int y = 0; y < u.h; ++y) {
            float *row = resizedGuide.row(y);
            const double gy = (y + 0.5) * sy - 0.5;
            for (int x = 0; x < u.w; ++x)
                row[x] = bilinear(guide, (x + 0.5) * sx - 0.5, gy);
        }
        matchGuide = &resizedGuide;
    }

    Plane sourceU = std::move(u);
    Plane sourceV = std::move(v);
    u = Plane(sourceU.w, sourceU.h);
    v = Plane(sourceV.w, sourceV.h);
    Plane weights(sourceU.w, sourceU.h);
    u.fill(0.0f);
    v.fill(0.0f);
    weights.fill(0.0f);

    const std::vector<int> startsX = blockStarts(sourceU.w);
    const std::vector<int> startsY = blockStarts(sourceU.h);
    const float sigma = std::clamp(0.6f * guideSigma, 0.002f, 0.012f);
    const float hardThreshold = 2.7f * sigma;
    const float matchLimit = std::max(4e-5f, 4.0f * guideSigma * guideSigma);
    constexpr std::array<float, patchSize> window = {
        0.25f, 0.50f, 0.75f, 1.0f, 1.0f, 0.75f, 0.50f, 0.25f
    };
    constexpr float blend = 0.65f;
    uint64_t groups = 0;
    uint64_t comparedPatches = 0;

    for (const int ay : startsY) {
        for (const int ax : startsX) {
            std::array<PatchMatch, maxGroup> matches{};
            int matchCount = 0;
            std::array<int, 25> candidateX{};
            std::array<int, 25> candidateY{};
            int candidateCount = 0;

            for (int dy = -searchRadius; dy <= searchRadius; dy += searchStep) {
                for (int dx = -searchRadius; dx <= searchRadius; dx += searchStep) {
                    const int bx = std::clamp(ax + dx, 0, sourceU.w - patchSize);
                    const int by = std::clamp(ay + dy, 0, sourceU.h - patchSize);
                    bool duplicate = false;
                    for (int i = 0; i < candidateCount; ++i) {
                        if (candidateX[i] == bx && candidateY[i] == by) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (duplicate)
                        continue;
                    candidateX[candidateCount] = bx;
                    candidateY[candidateCount] = by;
                    ++candidateCount;
                    const float cost = bx == ax && by == ay
                        ? 0.0f
                        : patchCost(*matchGuide, sourceU, sourceV, ax, ay, bx, by);
                    insertMatch(matches, matchCount, { cost, bx, by });
                    ++comparedPatches;
                }
            }

            int eligible = 0;
            while (eligible < matchCount && matches[eligible].cost <= matchLimit)
                ++eligible;
            const int groupCount = groupSizeFor(eligible);
            if (groupCount < 2)
                continue;

            std::array<float, maxGroup * patchSize * patchSize> filteredU{};
            std::array<float, maxGroup * patchSize * patchSize> filteredV{};
            std::array<float, patchSize * patchSize> minU{};
            std::array<float, patchSize * patchSize> maxU{};
            std::array<float, patchSize * patchSize> minV{};
            std::array<float, patchSize * patchSize> maxV{};
            minU.fill(std::numeric_limits<float>::infinity());
            maxU.fill(-std::numeric_limits<float>::infinity());
            minV.fill(std::numeric_limits<float>::infinity());
            maxV.fill(-std::numeric_limits<float>::infinity());

            for (int k = 0; k < groupCount; ++k) {
                for (int py = 0; py < patchSize; ++py) {
                    const float *sourceURow = sourceU.row(matches[k].y + py) + matches[k].x;
                    const float *sourceVRow = sourceV.row(matches[k].y + py) + matches[k].x;
                    for (int px = 0; px < patchSize; ++px) {
                        const int offset = py * patchSize + px;
                        const int grouped = offset * maxGroup + k;
                        const float valueU = sourceURow[px];
                        const float valueV = sourceVRow[px];
                        filteredU[grouped] = valueU;
                        filteredV[grouped] = valueV;
                        minU[offset] = std::min(minU[offset], valueU);
                        maxU[offset] = std::max(maxU[offset], valueU);
                        minV[offset] = std::min(minV[offset], valueV);
                        maxV[offset] = std::max(maxV[offset], valueV);
                    }
                }
            }
            haar2DForwardGroups(filteredU.data(), filteredV.data(), groupCount);

            int retained = 0;
            for (int coefficient = 0; coefficient < patchSize * patchSize; ++coefficient) {
                float *valuesU = filteredU.data() + coefficient * maxGroup;
                float *valuesV = filteredV.data() + coefficient * maxGroup;
                haarForward(valuesU, groupCount);
                haarForward(valuesV, groupCount);
                for (int k = 0; k < groupCount; ++k) {
                    const bool preserveDC = coefficient == 0 && k == 0;
                    if (!preserveDC && std::fabs(valuesU[k]) < hardThreshold)
                        valuesU[k] = 0.0f;
                    else
                        ++retained;
                    if (!preserveDC && std::fabs(valuesV[k]) < hardThreshold)
                        valuesV[k] = 0.0f;
                    else
                        ++retained;
                }
                haarInverse(valuesU, groupCount);
                haarInverse(valuesV, groupCount);
            }

            haar2DInverseGroups(filteredU.data(), filteredV.data(), groupCount);
            for (int k = 0; k < groupCount; ++k) {
                for (int offset = 0; offset < patchSize * patchSize; ++offset) {
                    const int grouped = offset * maxGroup + k;
                    filteredU[grouped] = std::clamp(
                        filteredU[grouped], minU[offset], maxU[offset]);
                    filteredV[grouped] = std::clamp(
                        filteredV[grouped], minV[offset], maxV[offset]);
                }
            }

            const float groupWeight = 1.0f / (1.0f + retained);
            std::array<float, patchSize * patchSize> patchWeights{};
            for (int py = 0; py < patchSize; ++py) {
                for (int px = 0; px < patchSize; ++px) {
                    const int offset = py * patchSize + px;
                    patchWeights[offset] = groupWeight * window[px] * window[py];
                }
            }
            for (int k = 0; k < groupCount; ++k) {
                for (int py = 0; py < patchSize; ++py) {
                    float *sumU = u.row(matches[k].y + py) + matches[k].x;
                    float *sumV = v.row(matches[k].y + py) + matches[k].x;
                    float *sumW = weights.row(matches[k].y + py) + matches[k].x;
                    for (int px = 0; px < patchSize; ++px) {
                        const int offset = py * patchSize + px;
                        const float weight = patchWeights[offset];
                        const int grouped = offset * maxGroup + k;
                        sumU[px] += weight * filteredU[grouped];
                        sumV[px] += weight * filteredV[grouped];
                        sumW[px] += weight;
                    }
                }
            }
            ++groups;
        }
    }

    for (int y = 0; y < u.h; ++y) {
        float *outU = u.row(y);
        float *outV = v.row(y);
        const float *srcU = sourceU.row(y);
        const float *srcV = sourceV.row(y);
        const float *weight = weights.row(y);
        for (int x = 0; x < u.w; ++x) {
            if (weight[x] > 0.0f) {
                const float filteredU = outU[x] / weight[x];
                const float filteredV = outV[x] / weight[x];
                outU[x] = srcU[x] + blend * (filteredU - srcU[x]);
                outV[x] = srcV[x] + blend * (filteredV - srcV[x]);
            } else {
                outU[x] = srcU[x];
                outV[x] = srcV[x];
            }
        }
    }

    if (metrics)
        metrics->addWork(Stage::ApplyCollaborativeFilter,
                         uint64_t(u.w) * u.h, comparedPatches + groups);
}

} // namespace lgcr
