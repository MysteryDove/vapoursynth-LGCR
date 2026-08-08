// LGCR — Luma-Guided Chroma Reconstruction: shared declarations.
// See plugin.cpp for the pipeline overview and README.md for the algorithm
// documentation and the design-decision history.
#pragma once

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifndef LGCR_SUFFIX
#define LGCR_SUFFIX ""
#endif

namespace lgcr {

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

enum class Kernel { Bilinear, Bicubic, Lanczos, Spline16, Spline36, Jinc };

// Jinc is radially symmetric and evaluated through a 1D profile LUT.
inline bool kernelIsRadial(Kernel k) { return k == Kernel::Jinc; }

inline double sinc(double x) {
    if (std::fabs(x) < 1e-8)
        return 1.0;
    x *= M_PI;
    return std::sin(x) / x;
}

inline double kernelEval(Kernel k, double x, double p1, double p2) {
    x = std::fabs(x);
    switch (k) {
    case Kernel::Bilinear:
        return x < 1.0 ? 1.0 - x : 0.0;
    case Kernel::Bicubic: {
        // Keys family, p1 = B, p2 = C
        const double b = p1, c = p2;
        if (x < 1.0)
            return ((12 - 9 * b - 6 * c) * x * x * x + (-18 + 12 * b + 6 * c) * x * x + (6 - 2 * b)) / 6.0;
        if (x < 2.0)
            return ((-b - 6 * c) * x * x * x + (6 * b + 30 * c) * x * x + (-12 * b - 48 * c) * x + (8 * b + 24 * c)) / 6.0;
        return 0.0;
    }
    case Kernel::Lanczos:
        return x < p1 ? sinc(x) * sinc(x / p1) : 0.0;
    case Kernel::Spline16:
        // fmtconv coefficients
        if (x < 1.0)
            return ((x - 9.0 / 5.0) * x - 1.0 / 5.0) * x + 1.0;
        if (x < 2.0)
            return ((-1.0 / 3.0 * (x - 1.0) + 4.0 / 5.0) * (x - 1.0) - 7.0 / 15.0) * (x - 1.0);
        return 0.0;
    case Kernel::Spline36:
        if (x < 1.0)
            return ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0;
        if (x < 2.0)
            return ((-6.0 / 11.0 * (x - 1.0) + 270.0 / 209.0) * (x - 1.0) - 156.0 / 209.0) * (x - 1.0);
        if (x < 3.0)
            return ((1.0 / 11.0 * (x - 2.0) - 45.0 / 209.0) * (x - 2.0) + 26.0 / 209.0) * (x - 2.0);
        return 0.0;
    case Kernel::Jinc:
        break; // handled via LUT (see jincWindowed / LGCRData::lut)
    }
    return 0.0;
}

// Normalized jinc: 2*J1(pi*x)/(pi*x), windowed by a sinc lobe (jinc-lanczos)
inline double jincWindowed(double x, double taps) {
    x = std::fabs(x);
    if (x >= taps)
        return 0.0;
    const double j = (x < 1e-8) ? 1.0
                                : 2.0 * std::cyl_bessel_j(1.0, M_PI * x) / (M_PI * x);
    return j * sinc(x / taps);
}

inline double kernelSupport(Kernel k, double p1) {
    switch (k) {
    case Kernel::Bilinear: return 1.0;
    case Kernel::Bicubic:  return 2.0;
    case Kernel::Lanczos:  return p1;
    case Kernel::Spline16: return 2.0;
    case Kernel::Spline36: return 3.0;
    case Kernel::Jinc:     return p1;
    }
    return 1.0;
}

// ---------------------------------------------------------------------------
// Filter data
// ---------------------------------------------------------------------------

struct LGCRData {
    VSNode *node = nullptr;
    const VSVideoInfo *viIn = nullptr;
    VSVideoInfo viOut{};

    int outW = 0, outH = 0;          // output (luma) dimensions
    Kernel kernel = Kernel::Lanczos;
    double kp1 = 3.0, kp2 = 0.6;     // lanczos taps / bicubic b,c
    int taps = 6;                    // support width in taps (even)
    double support = 3.0;
    bool radial = false;             // jinc: 2D radially symmetric kernel
    std::vector<float> lut;          // radial kernel profile (jinc-lanczos)
    float lutScale = 0.0f;           // LUT entries per unit distance

    double strength = 0.8;           // lambda0
    double sigma = 0.01;             // luminance similarity sigma floor (normalized)
    double sratio = 0.15;            // adaptive sigma = max(sigma, sratio * window luma range)
    double sdb = 3.0;                // sigma multiplier (x window range) on ramps vs steps
    double reg = 0.005;              // algo=4/6 regression regularization (normalized units)
    double stretch = 1.0;            // along-edge support stretch (0 = isotropic)
    double gsigma = 2.5;             // guided-bump width sigma (luma px)
    double rescue = 1.0;             // phase-zero additive rescue scale [0,1]
    int algo = 2;                    // 2=guided selector, 4=hybrid selector, 6=detail transfer
    bool ridge = true;               // thin-line (ridge) detection (algo 2/4)
    bool cedge = false;              // EXPERIMENTAL: wide-chroma-transition fade (off, see README)
    double ms = 1.0;                 // mutual-structure gate strength (algo 2/4/6, TRecon)
    double qgate = 1.0;              // algo=6 affine-credibility gate strength [0,1]
    bool sparse = true;              // sparse correction: guide only near luma structure
    bool locSet = false;             // loc param explicitly given (else read _ChromaLocation)
    double bp = 0.0;                 // back-projection data-consistency gain (420 same-size)
    int trad = 1;                    // TRecon: temporal radius (frames each side)
    int tsearch = 6;                 // TRecon: ME search range (luma px, integer)
    double tsad = 0.02;              // TRecon: ME SAD-per-pixel confidence scale
    double arMargin = 0.0;           // anti-ringing hull margin (<0 disables)
    double shiftX = -0.5;            // src chroma siting, luma units ("left")
    double shiftY = 0.0;             // vertical siting (from _ChromaLocation / loc)
};

// ---------------------------------------------------------------------------
// Small float plane helper
// ---------------------------------------------------------------------------

struct Plane {
    int w = 0, h = 0, stride = 0;
    std::vector<float> px;

    Plane() = default;
    Plane(int w_, int h_) : w(w_), h(h_), stride(w_) { px.resize(size_t(stride) * h); }
    float *row(int y) { return px.data() + size_t(y) * stride; }
    const float *row(int y) const { return px.data() + size_t(y) * stride; }
    float &at(int x, int y) {
        x = std::clamp(x, 0, w - 1);
        y = std::clamp(y, 0, h - 1);
        return px[size_t(y) * stride + x];
    }
    float at(int x, int y) const {
        x = std::clamp(x, 0, w - 1);
        y = std::clamp(y, 0, h - 1);
        return px[size_t(y) * stride + x];
    }
};

// Bilinear sample of a plane at fractional position (coords in sample units)
inline float bilinear(const Plane &p, double x, double y) {
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float fx = static_cast<float>(x - x0);
    const float fy = static_cast<float>(y - y0);
    const float a = p.at(x0, y0), b = p.at(x0 + 1, y0);
    const float c = p.at(x0, y0 + 1), d = p.at(x0 + 1, y0 + 1);
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}

// Precomputed bilinear sampling axis: clamped index + fraction per output
// sample, so per-pixel guide sampling needs no branches or clamps.
struct BilinAxis {
    int n = 0;
    std::vector<int> i0;
    std::vector<float> f;
};

BilinAxis buildBilinAxis(const std::vector<float> &pos, int srcN);

// Fast bilinear: 4 direct loads, indices pre-clamped via BilinAxis
inline float bilinearFast(const Plane &p, int x0, float fx, int y0, float fy) {
    if (p.w == 1) {
        const float a = p.row(y0)[0];
        if (p.h == 1)
            return a;
        const float c = p.row(y0 + 1)[0];
        return a + (c - a) * fy;
    }
    const float *r0 = p.row(y0) + x0;
    if (p.h == 1)
        return r0[0] + (r0[1] - r0[0]) * fx;
    const float *r1 = p.row(y0 + 1) + x0;
    const float a = r0[0], b = r0[1], c = r1[0], d = r1[1];
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}

// ---------------------------------------------------------------------------
// Luma structure maps (maps.cpp)
// ---------------------------------------------------------------------------

struct GuideMaps {
    Plane jxx;  // full-res structure tensor components (3x3 box-smoothed
    Plane jxy;  //   outer products of the Sobel gradient). Orientation and
    Plane jyy;  //   coherence come from the eigensystem, not raw gradients.
    Plane lc;   // chroma-res luma level of each chroma sample (footprint average)
    Plane ms;   // chroma-res mutual-structure co-edge gate [0,1] (empty if off)
};

GuideMaps buildGuideMaps(const Plane &structY, const Plane &lcY, int cw, int ch,
                         double rw, double rh, double shiftX, double shiftY);

// Mutual-structure co-edge gate at chroma res (maps.cpp). rho-correlation of
// luma/chroma gradient profiles along the luma edge normal; 1 = confirmed
// co-edge (direction + position + width agree), 0 = no chroma-side evidence.
Plane buildMutualGate(const Plane &lc, const Plane &U, const Plane &V, double sigma);

// Footprint-averaged luma at each chroma sample (standalone, for TRecon).
Plane buildLcMap(const Plane &lcY, int cw, int ch, double rw, double rh,
                 double shiftX, double shiftY);

// Candidate encoder degradation D^(Y) to the chroma grid:
// kind 0 = box footprint (== buildLcMap), 1 = triangle/bilinear, 2 = bicubic.
Plane buildYcMap(const Plane &Y, int cw, int ch, double rw, double rh,
                 double shiftX, double shiftY, int kind);

// Sparse trust mask: 1 where the output pixel's support window may touch a
// luma structure worth guiding by (tensor energy over threshold), dilated by
// the support radius. 0 = plain kernel is provably sufficient.
std::vector<uint8_t> buildTrustMask(const GuideMaps &gm, int outW, int outH,
                                    double sigma, int dilateRadius);

// ---------------------------------------------------------------------------
// Resampling (kernels.cpp)
// ---------------------------------------------------------------------------

struct WeightTable {
    int n = 0;              // number of output samples
    int sup = 0;            // taps per sample (fixed, padded to multiple of 8)
    std::vector<int> start; // first source tap index per output sample
    std::vector<float> w;   // n * sup weights, normalized
};

WeightTable buildWeights(int srcN, int dstN, Kernel k, double kp1, double kp2,
                         double support, double shift /* source units */);

void resampleH(const Plane &src, Plane &dst, const WeightTable &t);
void resampleV(const Plane &src, Plane &dst, const WeightTable &t);

// Radial (jinc) profile LUT lookups
inline float lutLookup(const float *lut, int n, float idx) {
    int i0 = static_cast<int>(idx);
    i0 = std::clamp(i0, 0, n - 2);
    const float fr = std::min(idx - float(i0), 1.0f);
    return lut[i0] + (lut[i0 + 1] - lut[i0]) * fr;
}

#ifdef __AVX2__
inline __m256 lutLookup8(const float *lut, int n, __m256 idx) {
    __m256i i0 = _mm256_cvttps_epi32(idx);
    i0 = _mm256_max_epi32(i0, _mm256_setzero_si256());
    i0 = _mm256_min_epi32(i0, _mm256_set1_epi32(n - 2));
    const __m256 fi0 = _mm256_cvtepi32_ps(i0);
    const __m256 fr = _mm256_min_ps(_mm256_sub_ps(idx, fi0), _mm256_set1_ps(1.0f));
    const __m256i one = _mm256_set1_epi32(1);
    const __m256 a = _mm256_i32gather_ps(lut, i0, 4);
    const __m256 b = _mm256_i32gather_ps(lut, _mm256_add_epi32(i0, one), 4);
    return _mm256_fmadd_ps(_mm256_sub_ps(b, a), fr, a);
}

// Fast reciprocal: rcp + one Newton-Raphson step (~22-bit precision, fine for weights)
inline __m256 rcpNR(__m256 x) {
    const __m256 r = _mm256_rcp_ps(x);
    return _mm256_mul_ps(r, _mm256_fnmadd_ps(x, r, _mm256_set1_ps(2.0f)));
}
#endif

// Full 2D resample with a radially symmetric kernel (jinc), for luma.
void resampleRadial(const Plane &src, Plane &dst, const LGCRData &d);

// ---------------------------------------------------------------------------
// Guided chroma reconstruction (recon.cpp)
// ---------------------------------------------------------------------------

struct TemporalNbr;

struct ChromaJob {
    const Plane *srcU, *srcV;  // source chroma (chroma res)
    const Plane *srcY;         // source luma (full res, for tap-level stats)
    const GuideMaps *gm;       // guide maps (built in output space)
    Plane *dstU, *dstV;        // output chroma (output luma res)
    int srcLumaW, srcLumaH;    // source luma dimensions
    double rw, rh;             // luma/chroma ratio (subsample factors)
    double shiftX, shiftY;     // src chroma siting, luma units
    const LGCRData *d;
    Plane *selW2 = nullptr;    // selector output: per-pixel weight for the
    Plane *selW3 = nullptr;    //   sim path (w2) and the LGF path (w3)
    const uint8_t *mask = nullptr; // sparse trust mask (SOURCE luma res, row-major);
    int maskW = 0, maskH = 0;      //   mask dimensions (indexed by mapped coords)
    const Plane *plainU = nullptr; //   pixels with mask==0 keep plainU/plainV
    const Plane *plainV = nullptr; //   (dst must be pre-filled with them)
    const std::vector<TemporalNbr> *nbrs = nullptr; // temporal taps (TRecon)
};

// One motion-compensated neighbor frame for TRecon. Motion is integer-pel
// block matching on luma; each neighbor contributes its actual chroma
// samples (phase diversity: odd luma-pel motion shifts the 420 phase).
struct TemporalNbr {
    const Plane *U, *V;          // neighbor chroma (chroma res)
    const Plane *lc;             // neighbor footprint luma map (chroma res)
    const std::vector<int16_t> *mvx, *mvy; // block motion field (luma px)
    const std::vector<float> *tconf;       // per-block match confidence [0,1]
    int bw = 0, bh = 0;          // block grid dimensions
    int block = 16;              // block size in luma px
};

// Per-output-x precomputed data: chroma source position, base weights, tap start
struct ChromaAxis {
    int n = 0;
    int sup = 0;               // padded to multiple of 8
    std::vector<int> start;    // first tap (clamped-safe range handled at load)
    std::vector<float> w;      // base kernel weights, n * sup
    std::vector<float> am;     // tap activity mask, n * sup: 1.0 inside the logical
                               //   kernel support, 0.0 for SIMD padding taps
    std::vector<float> pos;    // source center position per output sample (chroma units)
    std::vector<float> lpos;   // source center position per output sample (luma units)
};

ChromaAxis buildChromaAxis(int srcLumaN, int dstLumaN, double r, double sitShift,
                           const LGCRData *d);

void reconstructChroma(const ChromaJob &job);

// Plain (unguided) chroma reconstruction: separable fast path for separable
// kernels, 2D radial path for jinc. Base for algo 4/6 and the strength=0
// A/B reference.
void plainChroma(const LGCRData *d, const Plane &cb, const Plane &cr,
                 const Plane &y, const GuideMaps &gm,
                 int sw, int sh, int cw, int ch,
                 Plane &cOutU, Plane &cOutV);

// ---------------------------------------------------------------------------
// Alternative algorithms (algos.cpp)
// ---------------------------------------------------------------------------

// Internal LGF coefficient planes for the algo=4 selector branch.
struct LGFMaps {
    Plane aU, bU, confU, aV, bV, confV;
    LGFMaps(int cw, int ch) : aU(cw, ch), bU(cw, ch), confU(cw, ch),
                              aV(cw, ch), bV(cw, ch), confV(cw, ch) {}
};

void buildLGF(const Plane &Y, int cw, int ch, double rw, double rh,
              double shiftX, double shiftY, const Plane &C, int radius, double eps,
              Plane &a, Plane &b, Plane &conf, bool cedge);

LGFMaps buildLGFMaps(const LGCRData *d, const Plane &y, const Plane &cb,
                     const Plane &cr, int cw, int ch, double rw, double rh);

void blendSelector(Plane &outU, Plane &outV,
                   const Plane &gU, const Plane &gV,
                   const Plane &aU, const Plane &bU, const Plane &aV, const Plane &bV,
                   const Plane &confU, const Plane &confV, const Plane &Y,
                   const ChromaAxis &ax, const ChromaAxis &ay,
                   const Plane &w2, const Plane &w3,
                   int cw, int ch, float lam);

void backProject(Plane &cOut, const Plane &cSrc, float bp);

// ---------------------------------------------------------------------------
// algo=6: constrained detail transfer (maps + application)
//
//   C0 = P(C420)            plain kernel base (low freq / color reference)
//   Yc = consensus_k D_k(Y), Yb = P(Yc)
//   C1 = C0 + g * a * R(lp_source(Y - Yb))
//
// a is the median chroma-grid regression slope over candidate degradation
// kernels {box, triangle, bicubic}; g composites the joint-U/V
// affine credibility q, multi-kernel stability, chroma significance, and
// (at application time) the mutual-structure gate and strength. The detail
// field is binomially low-passed on the SOURCE luma grid along each subsampled
// axis before output scaling. Each dyadic stage uses its own stride, rejecting
// that stage's axis-alias frequency; this is deliberately not claimed to span
// the full null space of an unknown encoder.
struct AffineMaps {
    Plane aU, aV;      // regression slope per chroma sample (chroma res)
    Plane g;           // composite confidence before ms/strength (chroma res)
    Plane detail;      // constrained luma detail, resampled to output space
    Plane mnU, mxU, mnV, mxV; // 5x5 window hull per plane (chroma res)
    Plane rng;         // max(U,V) window range, for the magnitude cap
    AffineMaps() = default;
};

AffineMaps buildAffineMaps(const LGCRData *d, const Plane &y, const Plane &cb,
                           const Plane &cr, int cw, int ch, double rw, double rh);

void detailTransfer(const LGCRData *d, Plane &outU, Plane &outV,
                    const AffineMaps &af, const GuideMaps &gm,
                    const ChromaAxis &ax, const ChromaAxis &ay,
                    const uint8_t *mask, int maskW, int maskH);

// Integer-pel luma block matching (TRecon). Fills mvx/mvy/conf on a
// blockSize grid; conf = matchQuality x observability, where observability
// comes from the luma block variance (flat blocks have no observable motion
// -> conf ~ 0). Equal-SAD ties and near-ties resolve toward zero motion.
void blockMatch(const Plane &cur, const Plane &nbr, int blockSize, int search,
                float tsad, std::vector<int16_t> &mvx, std::vector<int16_t> &mvy,
                std::vector<float> &conf, int &bw, int &bh);

// Fold a robust, joint-U/V consistency score into the luma match confidence.
// Motion vectors are in luma pixels and confidence is on the same block grid.
void applyChromaConsistency(const Plane &curU, const Plane &curV,
                            const Plane &nbrU, const Plane &nbrV,
                            int lumaW, int lumaH, int blockSize,
                            const std::vector<int16_t> &mvx,
                            const std::vector<int16_t> &mvy,
                            int bw, int bh, std::vector<float> &conf);

// ---------------------------------------------------------------------------
// Sharpen: gated Laplacian (bilateral USM)
// ---------------------------------------------------------------------------

struct SharpenData {
    VSNode *node = nullptr;
    const VSVideoInfo *vi = nullptr;
    double alpha = 0.3;
    double sigma = 0.01;
    double sratio = 0.15;
    double gspatial = 1.2; // spatial bump sigma, in plane pixels
    double arMargin = 0.0;
};

void sharpenPlane(Plane &p, const Plane *guideLc, const Plane &guideY,
                  double rw, double rh, double shiftX, double shiftY,
                  const SharpenData &d);

// ---------------------------------------------------------------------------
// Format conversion helpers (templates — header)
// ---------------------------------------------------------------------------

template <typename T>
void planeToFloat(const uint8_t *src, ptrdiff_t srcStride, Plane &dst, int w, int h,
                  double scale, double offset) {
    for (int y = 0; y < h; ++y) {
        const T *s = reinterpret_cast<const T *>(src + srcStride * y);
        float *drow = dst.row(y);
        for (int x = 0; x < w; ++x)
            drow[x] = static_cast<float>(s[x] * scale + offset);
    }
}

template <typename T>
void floatToPlane(const Plane &src, uint8_t *dst, ptrdiff_t dstStride, int w, int h,
                  double scale, double offset, T lo, T hi) {
    for (int y = 0; y < h; ++y) {
        T *drow = reinterpret_cast<T *>(dst + dstStride * y);
        const float *s = src.row(y);
        if constexpr (std::is_same_v<T, float>) {
            for (int x = 0; x < w; ++x)
                drow[x] = static_cast<float>(s[x] * scale + offset);
        } else {
            for (int x = 0; x < w; ++x) {
                const int v = static_cast<int>(s[x] * float(scale) + float(offset) + 0.5f);
                drow[x] = static_cast<T>(std::clamp(v, int(lo), int(hi)));
            }
        }
    }
}

} // namespace lgcr
