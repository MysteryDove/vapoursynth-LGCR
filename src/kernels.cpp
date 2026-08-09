#include "lgcr.h"

namespace lgcr {

namespace {

uint64_t doubleBits(double value) {
    uint64_t bits;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline void hashCombine(size_t &seed, uint64_t value) {
    seed ^= std::hash<uint64_t>{}(value) + 0x9e3779b97f4a7c15ULL +
            (seed << 6) + (seed >> 2);
}

struct WeightKey {
    int srcN, dstN;
    Kernel kernel;
    uint64_t kp1, kp2, support, shift;

    bool operator==(const WeightKey &other) const {
        return srcN == other.srcN && dstN == other.dstN && kernel == other.kernel &&
               kp1 == other.kp1 && kp2 == other.kp2 && support == other.support &&
               shift == other.shift;
    }
};

struct WeightKeyHash {
    size_t operator()(const WeightKey &key) const {
        size_t seed = 0;
        hashCombine(seed, static_cast<uint64_t>(key.srcN));
        hashCombine(seed, static_cast<uint64_t>(key.dstN));
        hashCombine(seed, static_cast<uint64_t>(key.kernel));
        hashCombine(seed, key.kp1); hashCombine(seed, key.kp2);
        hashCombine(seed, key.support); hashCombine(seed, key.shift);
        return seed;
    }
};

struct AxisKey {
    int srcLumaN, dstLumaN;
    Kernel kernel;
    uint64_t ratio, sitShift, kp1, kp2, support;
    bool radial;

    bool operator==(const AxisKey &other) const {
        return srcLumaN == other.srcLumaN && dstLumaN == other.dstLumaN &&
               kernel == other.kernel && ratio == other.ratio &&
               sitShift == other.sitShift && kp1 == other.kp1 && kp2 == other.kp2 &&
               support == other.support && radial == other.radial;
    }
};

struct AxisKeyHash {
    size_t operator()(const AxisKey &key) const {
        size_t seed = 0;
        hashCombine(seed, static_cast<uint64_t>(key.srcLumaN));
        hashCombine(seed, static_cast<uint64_t>(key.dstLumaN));
        hashCombine(seed, static_cast<uint64_t>(key.kernel));
        hashCombine(seed, key.ratio); hashCombine(seed, key.sitShift);
        hashCombine(seed, key.kp1); hashCombine(seed, key.kp2);
        hashCombine(seed, key.support); hashCombine(seed, key.radial);
        return seed;
    }
};

struct RadialKey {
    const ChromaAxis *xAxis;
    const ChromaAxis *yAxis;
    int sourceWidth;
    int sourceHeight;

    bool operator==(const RadialKey &other) const {
        return xAxis == other.xAxis && yAxis == other.yAxis &&
               sourceWidth == other.sourceWidth && sourceHeight == other.sourceHeight;
    }
};

struct RadialKeyHash {
    size_t operator()(const RadialKey &key) const {
        size_t seed = std::hash<const void *>{}(key.xAxis);
        hashCombine(seed, reinterpret_cast<uintptr_t>(key.yAxis));
        hashCombine(seed, static_cast<uint64_t>(key.sourceWidth));
        hashCombine(seed, static_cast<uint64_t>(key.sourceHeight));
        return seed;
    }
};

} // namespace

struct GeometryCache {
    std::mutex mutex;
    std::unordered_map<WeightKey, std::shared_ptr<const WeightTable>, WeightKeyHash> weights;
    std::unordered_map<AxisKey, std::shared_ptr<const ChromaAxis>, AxisKeyHash> axes;
    std::unordered_map<RadialKey, std::shared_ptr<const RadialWeightTable>,
                       RadialKeyHash> radialWeights;
};

std::shared_ptr<GeometryCache> makeGeometryCache() {
    return std::make_shared<GeometryCache>();
}

static std::shared_ptr<GeometryCache> cacheFor(const LGCRData *d) {
    // Every LGCRData constructs an instance-local cache before frame requests
    // can run. Copies used for per-frame siting intentionally share it.
    return d->geometryCache ? d->geometryCache : makeGeometryCache();
}

static WeightKey weightKey(const LGCRData *d, int srcN, int dstN, double shift) {
    return { srcN, dstN, d->kernel, doubleBits(d->kp1), doubleBits(d->kp2),
             doubleBits(d->support), doubleBits(shift) };
}

std::shared_ptr<const WeightTable> cachedWeights(const LGCRData *d, int srcN,
                                                 int dstN, double shift) {
    auto cache = cacheFor(d);
    const WeightKey key = weightKey(d, srcN, dstN, shift);
    {
        std::lock_guard<std::mutex> lock(cache->mutex);
        const auto it = cache->weights.find(key);
        if (it != cache->weights.end())
            return it->second;
    }
    auto result = std::make_shared<const WeightTable>(
        buildWeights(srcN, dstN, d->kernel, d->kp1, d->kp2, d->support, shift));
    std::lock_guard<std::mutex> lock(cache->mutex);
    return cache->weights.emplace(key, result).first->second;
}

// weight tables, resamplers, chroma axis

WeightTable buildWeights(int srcN, int dstN, Kernel k, double kp1, double kp2,
                                double support, double shift /* source units */) {
    WeightTable t;
    t.n = dstN;
    const double scale = double(srcN) / dstN;
    // widen support when downscaling (approximate anti-aliasing)
    const double sup = (scale > 1.0) ? support * scale : support;
    const double invScale = (scale > 1.0) ? 1.0 / scale : 1.0;
    t.sup = std::max(1, 2 * static_cast<int>(std::ceil(sup)));
    t.start.resize(dstN);
    t.w.assign(size_t(dstN) * t.sup, 0.0f);

    for (int i = 0; i < dstN; ++i) {
        const double center = (i + 0.5) * scale - 0.5 + shift;
        int first = static_cast<int>(std::ceil(center - sup));
        first = std::clamp(first, -(t.sup) + 1, srcN - 1); // allow some out-of-range, clamped later
        if (first + t.sup > srcN)
            first = std::max(-(t.sup) + 1, srcN - t.sup); // may go negative; handled by clamped loads
        t.start[i] = first;
        double sum = 0.0;
        for (int j = 0; j < t.sup; ++j) {
            const double d = (first + j) - center;
            const double w = kernelEval(k, d * invScale, kp1, kp2);
            t.w[size_t(i) * t.sup + j] = static_cast<float>(w);
            sum += w;
        }
        if (std::fabs(sum) > 1e-9) {
            for (int j = 0; j < t.sup; ++j)
                t.w[size_t(i) * t.sup + j] /= static_cast<float>(sum);
        }
    }
    return t;
}

namespace {

template <class Backend, int Taps>
void resampleHFixed(const Plane &src, Plane &dst, const WeightTable &t) {
    for (int y = 0; y < src.h; ++y) {
        const float *srcRow = src.row(y);
        float *dstRow = dst.row(y);
        int x = 0;
        for (; x < t.n; ++x) {
            const float *wp = &t.w[size_t(x) * t.sup];
            const int s = t.start[x];
            if (s >= 0 && s + t.sup <= src.w) {
                double acc = 0.0;
                int j = 0;
#ifdef __AVX2__
                if constexpr (Backend::lanes == 8 && (Taps == 0 || Taps >= 8)) {
                    typename Backend::Vec vAcc = Backend::zero();
                    const int count = Taps ? Taps : t.sup;
                    for (; j + Backend::lanes <= count; j += Backend::lanes)
                        vAcc = Backend::fmadd(Backend::load(wp + j),
                                             Backend::load(srcRow + s + j), vAcc);
                    acc = Backend::horizontalSum(vAcc);
                }
#endif
                const int count = Taps ? Taps : t.sup;
                for (; j < count; ++j)
                    acc += double(wp[j]) * srcRow[s + j];
                dstRow[x] = static_cast<float>(acc);
            } else {
                double acc = 0.0;
                const int count = Taps ? Taps : t.sup;
                for (int j = 0; j < count; ++j)
                    acc += double(wp[j]) * src.at(s + j, y);
                dstRow[x] = static_cast<float>(acc);
            }
        }
    }
}

template <class Backend, int Taps>
void resampleVFixed(const Plane &src, Plane &dst, const WeightTable &t) {
    for (int y = 0; y < t.n; ++y) {
        const float *wp = &t.w[size_t(y) * t.sup];
        const int s = t.start[y];
        float *dstRow = dst.row(y);
        int x = 0;
#ifdef __AVX2__
        if constexpr (Backend::lanes == 8) {
            for (; x + Backend::lanes <= src.w; x += Backend::lanes) {
                typename Backend::Vec vAcc = Backend::zero();
                const int count = Taps ? Taps : t.sup;
                for (int j = 0; j < count; ++j) {
                    const int sy = std::clamp(s + j, 0, src.h - 1);
                    vAcc = Backend::fmadd(Backend::set1(wp[j]),
                                          Backend::load(src.row(sy) + x), vAcc);
                }
                Backend::store(dstRow + x, vAcc);
            }
        }
#endif
        for (; x < src.w; ++x) {
            double acc = 0.0;
            const int count = Taps ? Taps : t.sup;
            for (int j = 0; j < count; ++j)
                acc += double(wp[j]) * src.at(x, s + j);
            dstRow[x] = static_cast<float>(acc);
        }
    }
}

template <class Backend>
void resampleHImpl(const Plane &src, Plane &dst, const WeightTable &t) {
    switch (t.sup) {
    case 2: resampleHFixed<Backend, 2>(src, dst, t); break;
    case 4: resampleHFixed<Backend, 4>(src, dst, t); break;
    case 6: resampleHFixed<Backend, 6>(src, dst, t); break;
    case 8: resampleHFixed<Backend, 8>(src, dst, t); break;
    default: resampleHFixed<Backend, 0>(src, dst, t); break;
    }
}

template <class Backend>
void resampleVImpl(const Plane &src, Plane &dst, const WeightTable &t) {
    switch (t.sup) {
    case 2: resampleVFixed<Backend, 2>(src, dst, t); break;
    case 4: resampleVFixed<Backend, 4>(src, dst, t); break;
    case 6: resampleVFixed<Backend, 6>(src, dst, t); break;
    case 8: resampleVFixed<Backend, 8>(src, dst, t); break;
    default: resampleVFixed<Backend, 0>(src, dst, t); break;
    }
}

} // namespace

// Interior windows use direct pointers. Only the narrow boundary band uses
// clamped scalar loads, preserving the existing edge-extension semantics.
void resampleH(const Plane &src, Plane &dst, const WeightTable &t) {
    resampleHImpl<NativeBackend>(src, dst, t);
}

void resampleV(const Plane &src, Plane &dst, const WeightTable &t) {
    resampleVImpl<NativeBackend>(src, dst, t);
}


// Full 2D resample with a radially symmetric kernel (jinc). Non-separable by
// nature; used for the luma plane when kernel=jinc.
void resampleRadial(const Plane &src, Plane &dst, const LGCRData &d) {
    const int n = static_cast<int>(d.lut.size());
    const float *lut = d.lut.data();
    const double sx = double(src.w) / dst.w;
    const double sy = double(src.h) / dst.h;
    const double widen = std::max(1.0, std::max(sx, sy));   // downscale AA
    const float invW = float(1.0 / widen);
    const int radius = static_cast<int>(std::ceil(d.support * widen));
    const int sup = 2 * radius;

    for (int oy = 0; oy < dst.h; ++oy) {
        const double cy = (oy + 0.5) * sy - 0.5;
        const int y0 = static_cast<int>(std::ceil(cy - radius));
        float *dstRow = dst.row(oy);
        for (int ox = 0; ox < dst.w; ++ox) {
            const double cx = (ox + 0.5) * sx - 0.5;
            const int x0 = static_cast<int>(std::ceil(cx - radius));
            double acc = 0.0, wsum = 0.0;
            for (int j = 0; j < sup; ++j) {
                const int ty = std::clamp(y0 + j, 0, src.h - 1);
                const float dy = float((y0 + j - cy) * invW);
                const float *srcRow = src.row(ty);
                int i = 0;
                if (x0 >= 0 && x0 + sup <= src.w) {
#ifdef __AVX2__
                    __m256 vAcc = _mm256_setzero_ps();
                    __m256 vSum = _mm256_setzero_ps();
                    const __m256 vDy = _mm256_set1_ps(dy);
                    const __m256 vStep = _mm256_set1_ps(invW);
                    const __m256 vLutS = _mm256_set1_ps(d.lutScale);
                    const __m256 vDx0 = _mm256_fmadd_ps(
                        _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), vStep,
                        _mm256_set1_ps(float((x0 - cx) * invW)));
                    for (; i + 8 <= sup; i += 8) {
                        const __m256 vDx = _mm256_add_ps(vDx0, _mm256_set1_ps(float(i) * invW));
                        const __m256 r2 = _mm256_fmadd_ps(vDx, vDx, _mm256_mul_ps(vDy, vDy));
                        const __m256 idx = _mm256_mul_ps(_mm256_sqrt_ps(r2), vLutS);
                        const __m256 w = lutLookup8(lut, n, idx);
                        const __m256 vC = _mm256_loadu_ps(srcRow + x0 + i);
                        vAcc = _mm256_fmadd_ps(w, vC, vAcc);
                        vSum = _mm256_add_ps(vSum, w);
                    }
                    alignas(32) float buf[8];
                    _mm256_store_ps(buf, vAcc); for (int q = 0; q < 8; ++q) acc += buf[q];
                    _mm256_store_ps(buf, vSum); for (int q = 0; q < 8; ++q) wsum += buf[q];
#endif
                }
                for (; i < sup; ++i) {
                    const int tx = std::clamp(x0 + i, 0, src.w - 1);
                    const float dx = float((x0 + i - cx) * invW);
                    const float w = lutLookup(lut, n, std::sqrt(dx * dx + dy * dy) * d.lutScale);
                    acc += double(w) * srcRow[tx];
                    wsum += w;
                }
            }
            dstRow[ox] = static_cast<float>(wsum > 1e-6 ? acc / wsum : 0.0);
        }
    }
}


static ChromaAxis buildChromaAxisUncached(int srcLumaN, int dstLumaN, double r,
                                          double sitShift, const LGCRData *d) {
    ChromaAxis a;
    a.n = dstLumaN;
    // radial kernels widen their support on downscale (anisotropic EWA-style AA)
    const double scaleC = (double(srcLumaN) / r) / dstLumaN;
    const double widenC = d->radial ? std::max(1.0, scaleC) : 1.0;
    a.sup = std::max(1, 2 * static_cast<int>(std::ceil(d->support * widenC)));
    a.start.resize(a.n);
    a.pos.resize(a.n);
    a.lpos.resize(a.n);
    a.w.assign(size_t(a.n) * a.sup, 0.0f);
    a.am.assign(size_t(a.n) * a.sup, 0.0f);
    a.weightSum.assign(a.n, 0.0f);
    a.absoluteWeightSum.assign(a.n, 0.0f);
    a.activeTaps.assign(a.n, 0);
    const int srcChromaN = (srcLumaN + static_cast<int>(r) - 1) / static_cast<int>(r);
    const double scale = double(srcLumaN) / dstLumaN;

    for (int i = 0; i < a.n; ++i) {
        // output sample -> source luma coords -> source chroma coords
        const double lx = (i + 0.5) * scale - 0.5;
        const double sc = (lx + 0.5 - sitShift) / r - 0.5;
        a.pos[i] = static_cast<float>(sc);
        a.lpos[i] = static_cast<float>(lx);
        int first = static_cast<int>(std::ceil(sc - d->support * widenC));
        first = std::clamp(first, 0, std::max(0, srcChromaN - a.sup));
        a.start[i] = first;
        // Logical support can be narrower at an edge phase. Inactive taps do
        // not participate in range, hull, or rescue statistics.
        for (int j = 0; j < a.sup; ++j)
            if (std::fabs((first + j) - sc) < d->support * widenC) {
                a.am[size_t(i) * a.sup + j] = 1.0f;
                ++a.activeTaps[i];
            }
        if (d->radial)
            continue; // base weights come from the profile LUT at runtime
        double sum = 0.0;
        for (int j = 0; j < a.sup; ++j) {
            const double dist = (first + j) - sc;
            const double w = kernelEval(d->kernel, dist, d->kp1, d->kp2);
            a.w[size_t(i) * a.sup + j] = static_cast<float>(w);
            sum += w;
        }
        if (std::fabs(sum) > 1e-9)
            for (int j = 0; j < a.sup; ++j)
                a.w[size_t(i) * a.sup + j] /= static_cast<float>(sum);
        for (int j = 0; j < a.sup; ++j) {
            if (a.am[size_t(i) * a.sup + j] == 0.0f)
                continue;
            const float weight = a.w[size_t(i) * a.sup + j];
            a.weightSum[i] += weight;
            a.absoluteWeightSum[i] += std::fabs(weight);
        }
    }
    a.tapWeights.resize(size_t(a.sup) * a.n);
    a.tapActivity.resize(size_t(a.sup) * a.n);
    for (int tap = 0; tap < a.sup; ++tap)
        for (int i = 0; i < a.n; ++i) {
            const float active = a.am[size_t(i) * a.sup + tap];
            a.tapActivity[size_t(tap) * a.n + i] = active;
            a.tapWeights[size_t(tap) * a.n + i] =
                active * a.w[size_t(i) * a.sup + tap];
        }
    a.chromaBilin = buildBilinAxis(a.pos, srcChromaN);
    a.lumaBilin = buildBilinAxis(a.lpos, srcLumaN);
    return a;
}

ChromaAxis buildChromaAxis(int srcLumaN, int dstLumaN, double r, double sitShift,
                           const LGCRData *d) {
    return *cachedChromaAxis(d, srcLumaN, dstLumaN, r, sitShift);
}

std::shared_ptr<const ChromaAxis> cachedChromaAxis(const LGCRData *d,
                                                   int srcLumaN, int dstLumaN,
                                                   double r, double sitShift) {
    auto cache = cacheFor(d);
    const AxisKey key{ srcLumaN, dstLumaN, d->kernel, doubleBits(r),
                       doubleBits(sitShift), doubleBits(d->kp1), doubleBits(d->kp2),
                       doubleBits(d->support), d->radial };
    {
        std::lock_guard<std::mutex> lock(cache->mutex);
        const auto it = cache->axes.find(key);
        if (it != cache->axes.end())
            return it->second;
    }
    auto result = std::make_shared<const ChromaAxis>(
        buildChromaAxisUncached(srcLumaN, dstLumaN, r, sitShift, d));
    std::lock_guard<std::mutex> lock(cache->mutex);
    return cache->axes.emplace(key, result).first->second;
}

namespace {

struct PhaseKey {
    uint32_t offsetBits;
    uint64_t activity;

    bool operator==(const PhaseKey &other) const {
        return offsetBits == other.offsetBits && activity == other.activity;
    }
};

struct PhaseKeyHash {
    size_t operator()(const PhaseKey &key) const {
        size_t seed = key.offsetBits;
        hashCombine(seed, key.activity);
        return seed;
    }
};

bool collectPhases(const ChromaAxis &axis, std::vector<uint16_t> &indices,
                   std::vector<float> &offsets, std::vector<uint64_t> &activity) {
    if (axis.sup > 64)
        return false;
    std::unordered_map<PhaseKey, uint16_t, PhaseKeyHash> unique;
    indices.resize(axis.n);
    for (int x = 0; x < axis.n; ++x) {
        const float offset = axis.start[x] - axis.pos[x];
        uint32_t bits;
        std::memcpy(&bits, &offset, sizeof(bits));
        uint64_t mask = 0;
        for (int tap = 0; tap < axis.sup; ++tap)
            if (axis.am[size_t(x) * axis.sup + tap] != 0.0f)
                mask |= uint64_t{1} << tap;
        const PhaseKey key{ bits, mask };
        auto [it, inserted] = unique.emplace(key, static_cast<uint16_t>(unique.size()));
        if (inserted) {
            if (unique.size() > 64)
                return false;
            offsets.push_back(offset);
            activity.push_back(mask);
        }
        indices[x] = it->second;
    }
    return true;
}

std::shared_ptr<const RadialWeightTable> buildRadialWeights(
    const LGCRData &d, const ChromaAxis &ax, const ChromaAxis &ay,
    int sourceWidth, int sourceHeight) {
    auto table = std::make_shared<RadialWeightTable>();
    table->supportX = ax.sup;
    table->supportY = ay.sup;
    std::vector<float> xOffsets, yOffsets;
    std::vector<uint64_t> xActivity, yActivity;
    if (!collectPhases(ax, table->xPhase, xOffsets, xActivity) ||
        !collectPhases(ay, table->yPhase, yOffsets, yActivity))
        return table;
    const size_t phasePairs = xOffsets.size() * yOffsets.size();
    const size_t weightsPerPhase = size_t(ax.sup) * ay.sup;
    if (phasePairs > 1024 || phasePairs * weightsPerPhase > 262144)
        return table;

    table->phaseCountX = static_cast<int>(xOffsets.size());
    table->phaseCountY = static_cast<int>(yOffsets.size());
    table->weights.resize(phasePairs * weightsPerPhase);
    table->weightSum.resize(phasePairs);
    table->absoluteWeightSum.resize(phasePairs);
    const float invX = float(std::min(1.0, double(ax.n) / sourceWidth));
    const float invY = float(std::min(1.0, double(ay.n) / sourceHeight));
    for (int py = 0; py < table->phaseCountY; ++py) {
        for (int px = 0; px < table->phaseCountX; ++px) {
            float *weights = table->weights.data() +
                (size_t(py) * table->phaseCountX + px) * weightsPerPhase;
            for (int j = 0; j < ay.sup; ++j) {
                for (int i = 0; i < ax.sup; ++i) {
                    if ((xActivity[px] & (uint64_t{1} << i)) == 0 ||
                        (yActivity[py] & (uint64_t{1} << j)) == 0)
                        continue;
                    const float dx = (xOffsets[px] + i) * invX;
                    const float dy = (yOffsets[py] + j) * invY;
                    weights[size_t(j) * ax.sup + i] = lutLookup(
                        d.lut.data(), static_cast<int>(d.lut.size()),
                        std::sqrt(dx * dx + dy * dy) * d.lutScale);
                }
            }
            const size_t phase = size_t(py) * table->phaseCountX + px;
            double sum = 0.0, absoluteSum = 0.0;
            for (size_t i = 0; i < weightsPerPhase; ++i) {
                sum += weights[i];
                absoluteSum += std::fabs(weights[i]);
            }
            table->weightSum[phase] = sum;
            table->absoluteWeightSum[phase] = absoluteSum;
        }
    }
    return table;
}

} // namespace

std::shared_ptr<const RadialWeightTable> cachedRadialWeights(
    const LGCRData *d, const ChromaAxis &ax, const ChromaAxis &ay,
    int sourceWidth, int sourceHeight) {
    auto cache = cacheFor(d);
    const RadialKey key{ &ax, &ay, sourceWidth, sourceHeight };
    {
        std::lock_guard<std::mutex> lock(cache->mutex);
        const auto it = cache->radialWeights.find(key);
        if (it != cache->radialWeights.end())
            return it->second;
    }
    auto result = buildRadialWeights(*d, ax, ay, sourceWidth, sourceHeight);
    std::lock_guard<std::mutex> lock(cache->mutex);
    return cache->radialWeights.emplace(key, result).first->second;
}

} // namespace lgcr
