// Compile-time SIMD vocabulary shared by CPU pipeline stages.
#pragma once

#include <algorithm>
#include <cmath>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace lgcr {

struct ScalarBackend {
    using Vec = float;
    using Mask = bool;
    static constexpr int lanes = 1;

    static Vec load(const float *p) { return *p; }
    static void store(float *p, Vec value) { *p = value; }
    static Vec set1(float value) { return value; }
    static Vec zero() { return 0.0f; }
    static Vec add(Vec a, Vec b) { return a + b; }
    static Vec sub(Vec a, Vec b) { return a - b; }
    static Vec mul(Vec a, Vec b) { return a * b; }
    static Vec fmadd(Vec a, Vec b, Vec c) { return a * b + c; }
    static Vec abs(Vec value) { return std::fabs(value); }
    static Vec min(Vec a, Vec b) { return std::min(a, b); }
    static Vec max(Vec a, Vec b) { return std::max(a, b); }
    static Vec sqrt(Vec value) { return std::sqrt(value); }
    static Vec rcp(Vec value) { return 1.0f / value; }
    static Mask greater(Vec a, Vec b) { return a > b; }
    static Vec select(Mask mask, Vec yes, Vec no) { return mask ? yes : no; }
    static float horizontalSum(Vec value) { return value; }
    static float horizontalMin(Vec value) { return value; }
    static float horizontalMax(Vec value) { return value; }
};

#ifdef __AVX2__
struct AVX2Backend {
    using Vec = __m256;
    using Mask = __m256;
    static constexpr int lanes = 8;

    static Vec load(const float *p) { return _mm256_loadu_ps(p); }
    static void store(float *p, Vec value) { _mm256_storeu_ps(p, value); }
    static Vec set1(float value) { return _mm256_set1_ps(value); }
    static Vec zero() { return _mm256_setzero_ps(); }
    static Vec add(Vec a, Vec b) { return _mm256_add_ps(a, b); }
    static Vec sub(Vec a, Vec b) { return _mm256_sub_ps(a, b); }
    static Vec mul(Vec a, Vec b) { return _mm256_mul_ps(a, b); }
    static Vec fmadd(Vec a, Vec b, Vec c) { return _mm256_fmadd_ps(a, b, c); }
    static Vec abs(Vec value) {
        return _mm256_and_ps(value,
            _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
    }
    static Vec min(Vec a, Vec b) { return _mm256_min_ps(a, b); }
    static Vec max(Vec a, Vec b) { return _mm256_max_ps(a, b); }
    static Vec sqrt(Vec value) { return _mm256_sqrt_ps(value); }
    static Vec rcp(Vec value) {
        const Vec estimate = _mm256_rcp_ps(value);
        return _mm256_mul_ps(estimate,
            _mm256_fnmadd_ps(value, estimate, _mm256_set1_ps(2.0f)));
    }
    static Mask greater(Vec a, Vec b) {
        return _mm256_cmp_ps(a, b, _CMP_GT_OQ);
    }
    static Vec select(Mask mask, Vec yes, Vec no) {
        return _mm256_blendv_ps(no, yes, mask);
    }
    static float horizontalSum(Vec value) {
        const __m128 low = _mm256_castps256_ps128(value);
        const __m128 high = _mm256_extractf128_ps(value, 1);
        __m128 sum = _mm_add_ps(low, high);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }
    static float horizontalMin(Vec value) {
        __m128 reduced = _mm_min_ps(_mm256_castps256_ps128(value),
                                   _mm256_extractf128_ps(value, 1));
        reduced = _mm_min_ps(reduced, _mm_movehl_ps(reduced, reduced));
        reduced = _mm_min_ss(reduced, _mm_shuffle_ps(reduced, reduced, 1));
        return _mm_cvtss_f32(reduced);
    }
    static float horizontalMax(Vec value) {
        __m128 reduced = _mm_max_ps(_mm256_castps256_ps128(value),
                                   _mm256_extractf128_ps(value, 1));
        reduced = _mm_max_ps(reduced, _mm_movehl_ps(reduced, reduced));
        reduced = _mm_max_ss(reduced, _mm_shuffle_ps(reduced, reduced, 1));
        return _mm_cvtss_f32(reduced);
    }
};
using NativeBackend = AVX2Backend;
#else
using NativeBackend = ScalarBackend;
#endif

} // namespace lgcr
