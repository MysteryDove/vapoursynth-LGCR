#include "lgcr.h"

namespace lgcr {

// weight tables, resamplers, chroma axis

WeightTable buildWeights(int srcN, int dstN, Kernel k, double kp1, double kp2,
                                double support, double shift /* source units */) {
    WeightTable t;
    t.n = dstN;
    const double scale = double(srcN) / dstN;
    // widen support when downscaling (approximate anti-aliasing)
    const double sup = (scale > 1.0) ? support * scale : support;
    const double invScale = (scale > 1.0) ? 1.0 / scale : 1.0;
    t.sup = ((2 * static_cast<int>(std::ceil(sup)) + 7) / 8) * 8;
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

// Horizontal pass. Interior pixels (full window in range) use direct SIMD
// loads; a narrow edge band falls back to clamped scalar loads.
void resampleH(const Plane &src, Plane &dst, const WeightTable &t) {
    for (int y = 0; y < src.h; ++y) {
        const float *srcRow = src.row(y);
        float *dstRow = dst.row(y);
        for (int x = 0; x < t.n; ++x) {
            const float *wp = &t.w[size_t(x) * t.sup];
            const int s = t.start[x];
            if (s >= 0 && s + t.sup <= src.w) {
                double acc = 0.0;
                int j = 0;
#ifdef __AVX2__
                __m256 vAcc = _mm256_setzero_ps();
                for (; j + 8 <= t.sup; j += 8)
                    vAcc = _mm256_fmadd_ps(_mm256_loadu_ps(wp + j),
                                           _mm256_loadu_ps(srcRow + s + j), vAcc);
                alignas(32) float buf[8];
                _mm256_store_ps(buf, vAcc);
                for (int q = 0; q < 8; ++q) acc += buf[q];
#endif
                for (; j < t.sup; ++j)
                    acc += double(wp[j]) * srcRow[s + j];
                dstRow[x] = static_cast<float>(acc);
            } else {
                double acc = 0.0;
                for (int j = 0; j < t.sup; ++j)
                    acc += double(wp[j]) * src.at(s + j, y);
                dstRow[x] = static_cast<float>(acc);
            }
        }
    }
}

// Vertical pass: vectorized across x (contiguous), scalar row clamp per tap.
void resampleV(const Plane &src, Plane &dst, const WeightTable &t) {
    for (int y = 0; y < t.n; ++y) {
        const float *wp = &t.w[size_t(y) * t.sup];
        const int s = t.start[y];
        float *dstRow = dst.row(y);
        int x = 0;
#ifdef __AVX2__
        for (; x + 8 <= src.w; x += 8) {
            __m256 vAcc = _mm256_setzero_ps();
            for (int j = 0; j < t.sup; ++j) {
                const int sy = std::clamp(s + j, 0, src.h - 1);
                vAcc = _mm256_fmadd_ps(_mm256_set1_ps(wp[j]),
                                       _mm256_loadu_ps(src.row(sy) + x), vAcc);
            }
            _mm256_storeu_ps(dstRow + x, vAcc);
        }
#endif
        for (; x < src.w; ++x) {
            double acc = 0.0;
            for (int j = 0; j < t.sup; ++j)
                acc += double(wp[j]) * src.at(x, s + j);
            dstRow[x] = static_cast<float>(acc);
        }
    }
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


ChromaAxis buildChromaAxis(int srcLumaN, int dstLumaN, double r, double sitShift,
                                  const LGCRData *d) {
    ChromaAxis a;
    a.n = dstLumaN;
    // radial kernels widen their support on downscale (anisotropic EWA-style AA)
    const double scaleC = (double(srcLumaN) / r) / dstLumaN;
    const double widenC = d->radial ? std::max(1.0, scaleC) : 1.0;
    a.sup = ((2 * static_cast<int>(std::ceil(d->support * widenC)) + 7) / 8) * 8;
    a.start.resize(a.n);
    a.pos.resize(a.n);
    a.lpos.resize(a.n);
    a.w.assign(size_t(a.n) * a.sup, 0.0f);
    a.am.assign(size_t(a.n) * a.sup, 0.0f);
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
        // logical support: taps inside the kernel window (padding taps are
        // masked out so they never touch range/hull/rescue statistics)
        for (int j = 0; j < a.sup; ++j)
            if (std::fabs((first + j) - sc) < d->support * widenC)
                a.am[size_t(i) * a.sup + j] = 1.0f;
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
    }
    return a;
}

} // namespace lgcr
