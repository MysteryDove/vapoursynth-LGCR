#include "lgcr.h"

namespace lgcr {

// guided chroma reconstruction + plain base

#ifdef __AVX2__
static inline __m256i activeLaneMask(int count) {
    return _mm256_cmpgt_epi32(_mm256_set1_epi32(count),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
}
#endif

template <class Backend>
void reconstructChromaImpl(const ChromaJob &job) {
    const LGCRData &d = *job.d;
    const Plane &U = *job.srcU;
    const Plane &V = *job.srcV;
    const GuideMaps &gm = *job.gm;
    Plane &outU = *job.dstU;
    Plane &outV = *job.dstV;

    const auto axOwner = cachedChromaAxis(&d, job.srcLumaW, outU.w, job.rw, job.shiftX);
    const auto ayOwner = cachedChromaAxis(&d, job.srcLumaH, outU.h, job.rh, job.shiftY);
    const ChromaAxis &ax = *axOwner;
    const ChromaAxis &ay = *ayOwner;

    const bool guided = d.strength > 0.0;
    // Guide in SOURCE space: L0 and the structure probes use the sharp source
    // luma at the mapped (fractional) position. An output-space guide was
    // tried and REJECTED on the battery: after resampling, the luma edge is
    // itself smeared, and discriminating source taps against a smeared
    // reference weakened the snap (upscale2x 0.032 -> 0.044). At same size
    // the two are identical anyway.
    const Plane &oY = *job.srcY;
    const BilinAxis &bx = ax.lumaBilin;
    const BilinAxis &by = ay.lumaBilin;
    // Mutual-structure gate map lives at CHROMA res: sample at (scx, scy).
    const bool useMs = guided && d.ms > 0.0 && gm.ms.w > 0;
    const BilinAxis mbxFallback = useMs && gm.ms.w != U.w
        ? buildBilinAxis(ax.pos, gm.ms.w) : BilinAxis();
    const BilinAxis mbyFallback = useMs && gm.ms.h != U.h
        ? buildBilinAxis(ay.pos, gm.ms.h) : BilinAxis();
    const BilinAxis &mbx = gm.ms.w == U.w ? ax.chromaBilin : mbxFallback;
    const BilinAxis &mby = gm.ms.h == U.h ? ay.chromaBilin : mbyFallback;
    const float msStrength = float(d.ms);
    // probe distance: one chroma tap spacing, in source luma px
    const float spOut = float(std::max(job.rw, job.rh));

    const float invGsigma2 = 1.0f / float(d.gsigma * d.gsigma);
    const float strength = float(d.strength);
    const bool radial = d.radial;
    const int lutN = radial ? int(d.lut.size()) : 0;
    const float *lut = radial ? d.lut.data() : nullptr;
    const auto radialOwner = radial
        ? cachedRadialWeights(&d, ax, ay, U.w, U.h)
        : std::shared_ptr<const RadialWeightTable>{};
    // per-axis downscale AA factors for radial base weights (chroma units)
    const float invWxC = float(std::min(1.0, double(outU.w) / U.w));
    const float invWyC = float(std::min(1.0, double(outU.h) / U.h));
    const bool clampHull = d.arMargin >= 0.0;
    const float margin = float(std::max(0.0, d.arMargin));
    // SIMD uses masked tails for fixed 2/4/6-tap kernels. The logical tables
    // remain exact-sized; only valid source lanes are touched.
    [[maybe_unused]] const bool simdOK = Backend::lanes > 1 &&
        U.w >= ax.sup;

    std::vector<uint16_t> activeX, activeY;
    if (job.metrics) {
        const uint64_t pixels = uint64_t(outU.w) * outU.h;
        job.metrics->outputPixels += pixels;
        job.metrics->addWork(job.metricStage, pixels);
        activeX.assign(outU.w, 0);
        activeY.assign(outU.h, 0);
        for (int x = 0; x < outU.w; ++x)
            for (int i = 0; i < ax.sup; ++i)
                activeX[x] += ax.am[size_t(x) * ax.sup + i] != 0.0f;
        for (int y = 0; y < outU.h; ++y)
            for (int j = 0; j < ay.sup; ++j)
                activeY[y] += ay.am[size_t(y) * ay.sup + j] != 0.0f;
    }

    struct GuidePixel {
        float luma = 0.0f;
        float nx = 0.0f, ny = 0.0f;
        float coherence = 0.0f;
        float minDiffNormal = 0.0f;
        float ridgeReversal = 0.0f;
        float phaseGate = 0.0f;
        float mutualFade = 1.0f;
        bool hasDirection = false;
    };
    std::vector<GuidePixel> guideRow(outU.w);
    const bool sharedWorkset = job.workset &&
        job.workset->outputWidth == outU.w && job.workset->outputHeight == outU.h;
    const bool directSparseMask = job.mask &&
        job.maskW == outU.w && job.maskH == outU.h;
    std::vector<int> sparseMaskX, sparseMaskY;
    if (job.mask && !directSparseMask) {
        sparseMaskX.resize(outU.w);
        sparseMaskY.resize(outU.h);
        for (int ox = 0; ox < outU.w; ++ox)
            sparseMaskX[ox] = std::min(
                job.maskW - 1, int(int64_t(ox) * job.maskW / outU.w));
        for (int oy = 0; oy < outU.h; ++oy)
            sparseMaskY[oy] = std::min(
                job.maskH - 1, int(int64_t(oy) * job.maskH / outU.h));
    }
    auto sparseActive = [&](int ox, int oy) {
        if (!job.mask)
            return true;
        const int mx = directSparseMask ? ox : sparseMaskX[ox];
        const int my = directSparseMask ? oy : sparseMaskY[oy];
        return job.mask[size_t(my) * job.maskW + mx] != 0;
    };
    std::vector<SparseSpan> activeSpans;
    std::vector<size_t> rowSpanOffsets;
    if (job.mask && !sharedWorkset) {
        ScopedCpuTimer timer(job.metrics, CpuProfileSlot::GuidedActiveRows);
        rowSpanOffsets.resize(size_t(outU.h) + 1);
        for (int oy = 0; oy < outU.h; ++oy) {
            rowSpanOffsets[oy] = activeSpans.size();
            int ox = 0;
            while (ox < outU.w) {
                while (ox < outU.w && !sparseActive(ox, oy))
                    ++ox;
                const int begin = ox;
                while (ox < outU.w && sparseActive(ox, oy))
                    ++ox;
                if (begin < ox)
                    activeSpans.push_back({begin, ox});
            }
        }
        rowSpanOffsets[outU.h] = activeSpans.size();
    }
    auto visitActive = [&](int y, auto &&function) {
        if (sharedWorkset) {
            for (size_t spanIndex = job.workset->outputRowOffsets[y];
                 spanIndex < job.workset->outputRowOffsets[y + 1]; ++spanIndex) {
                const SparseSpan span = job.workset->outputSpans[spanIndex];
                for (int x = span.begin; x < span.end; ++x)
                    function(x);
            }
        } else if (job.mask) {
            for (size_t spanIndex = rowSpanOffsets[y];
                 spanIndex < rowSpanOffsets[y + 1]; ++spanIndex)
                for (int x = activeSpans[spanIndex].begin;
                     x < activeSpans[spanIndex].end; ++x)
                    function(x);
        } else {
            for (int x = 0; x < outU.w; ++x)
                function(x);
        }
    };
    auto prepareGuideRow = [&](int oy) {
        if (!guided)
            return;
        const int byi = by.i0[oy];
        const float byf = by.f[oy];
        auto preparePixel = [&](int ox) {
            GuidePixel &meta = guideRow[ox];
            meta = GuidePixel{};
            const int bxi = bx.i0[ox];
            const float bxf = bx.f[ox];
            meta.luma = bilinearFast(oY, bxi, bxf, byi, byf);
            const float jxx = bilinearFast(gm.jxx, bxi, bxf, byi, byf);
            const float jxy = bilinearFast(gm.jxy, bxi, bxf, byi, byf);
            const float jyy = bilinearFast(gm.jyy, bxi, bxf, byi, byf);
            const float jsum = jxx + jyy;
            const TensorDirection direction = principalTensorDirection(jxx, jxy, jyy);
            meta.coherence = direction.coherence;
            meta.nx = direction.nx;
            meta.ny = direction.ny;
            meta.hasDirection = jsum > 1e-6f;

            const float lx = ax.lpos[ox], ly = ay.lpos[oy];
            if (meta.hasDirection) {
                const float yA = bilinear(oY, lx - spOut * meta.nx,
                                          ly - spOut * meta.ny);
                const float yB = bilinear(oY, lx + spOut * meta.nx,
                                          ly + spOut * meta.ny);
                meta.minDiffNormal = std::min(std::fabs(yA - meta.luma),
                                              std::fabs(yB - meta.luma));
            }
            if (d.ridge && meta.hasDirection) {
                const float dl = 1.5f * spOut;
                const float ym2 = bilinear(oY, lx - 2 * dl * meta.nx,
                                            ly - 2 * dl * meta.ny);
                const float ym1 = bilinear(oY, lx - dl * meta.nx,
                                            ly - dl * meta.ny);
                const float yp1 = bilinear(oY, lx + dl * meta.nx,
                                            ly + dl * meta.ny);
                const float yp2 = bilinear(oY, lx + 2 * dl * meta.nx,
                                            ly + 2 * dl * meta.ny);
                const float difference[4] = {
                    ym1 - ym2, meta.luma - ym1, yp1 - meta.luma, yp2 - yp1
                };
                for (int q = 0; q < 3; ++q) {
                    const float a = difference[q], b = difference[q + 1];
                    if (a > 0.0f && b < 0.0f)
                        meta.ridgeReversal = std::max(
                            meta.ridgeReversal, std::min(a, -b));
                    if (a < 0.0f && b > 0.0f)
                        meta.ridgeReversal = std::max(
                            meta.ridgeReversal, std::min(-a, b));
                }
            }
            const float fx = ax.pos[ox] - std::floor(ax.pos[ox]);
            const float fy = ay.pos[oy] - std::floor(ay.pos[oy]);
            const float phx = std::min(fx, 1.0f - fx);
            const float phy = std::min(fy, 1.0f - fy);
            meta.phaseGate = (1.0f - 2.0f * phx) * (1.0f - 2.0f * phy);
            if (useMs) {
                const float gate = bilinearFast(gm.ms, mbx.i0[ox], mbx.f[ox],
                                                 mby.i0[oy], mby.f[oy]);
                meta.mutualFade = 1.0f - msStrength * (1.0f - gate);
            }
        };
        visitActive(oy, preparePixel);
    };

    for (int oy = 0; oy < outU.h; ++oy) {
        const int ty0 = ay.start[oy];
        const float *wyp = &ay.w[size_t(oy) * ay.sup];
        const float scy = ay.pos[oy];
        {
            ScopedCpuTimer timer(job.metrics, CpuProfileSlot::GuidedMetadata);
            prepareGuideRow(oy);
        }

        float *dstRowU = outU.row(oy);
        float *dstRowV = outV.row(oy);
        size_t compressedCursor = (job.compressedSelector || job.plainHull) && sharedWorkset
            ? job.workset->outputIndexRowOffsets[oy] : 0;

        ScopedCpuTimer tapTimer(job.metrics, CpuProfileSlot::GuidedTapAccumulation);
        auto processPixel = [&](int ox) {
            const size_t compressedIndex = (job.compressedSelector || job.plainHull)
                ? compressedCursor++ : 0;
            const int tx0 = ax.start[ox];
            const float *wxp = &ax.w[size_t(ox) * ax.sup];
            const float scx = ax.pos[ox];
            const float *radialBase = radialOwner ? radialOwner->at(ox, oy) : nullptr;

            const GuidePixel &meta = guideRow[ox];
            const float L0 = meta.luma;
            const float nxv = meta.nx, nyv = meta.ny;
            const float minDiffN = meta.minDiffNormal;
            const float coherence = meta.coherence;
            const bool hasDir = meta.hasDirection;
            const float ridgeRev = meta.ridgeReversal;
            float ssRamp = 0.0f;
            if (job.metrics) {
                const uint64_t taps = uint64_t(activeX[ox]) * activeY[oy];
                job.metrics->tapsVisited += taps;
                job.metrics->taps[static_cast<size_t>(job.metricStage)] += taps;
            }
            // Pass 1 (guided only): window luma range -> adaptive sigma;
            // chroma hull of both planes for anti-ringing.
            float maxAbsDL = 0.0f;
            float hullMinU = 1e30f, hullMaxU = -1e30f;
            float hullMinV = 1e30f, hullMaxV = -1e30f;
            if (job.plainHull) {
                hullMinU = job.plainHull->minimumU[compressedIndex];
                hullMaxU = job.plainHull->maximumU[compressedIndex];
                hullMinV = job.plainHull->minimumV[compressedIndex];
                hullMaxV = job.plainHull->maximumV[compressedIndex];
                const float *amRow = &ay.am[size_t(oy) * ay.sup];
                const float *amCol = &ax.am[size_t(ox) * ax.sup];
                for (int j = 0; j < ay.sup; ++j) {
                    if (amRow[j] == 0.0f)
                        continue;
                    const int ty = std::clamp(ty0 + j, 0, U.h - 1);
                    const float *lcRow = gm.lc.row(ty);
                    int i = 0;
#ifdef __AVX2__
                    if (simdOK) {
                        __m256 maximum = _mm256_setzero_ps();
                        const __m256 luma = _mm256_set1_ps(L0);
                        const __m256 absolute = _mm256_castsi256_ps(
                            _mm256_set1_epi32(0x7fffffff));
                        for (; i < ax.sup; i += 8) {
                            const __m256i laneMask = activeLaneMask(
                                std::min(8, ax.sup - i));
                            const __m256 active = _mm256_maskload_ps(amCol + i, laneMask);
                            const __m256 difference = _mm256_sub_ps(
                                _mm256_maskload_ps(lcRow + tx0 + i, laneMask), luma);
                            maximum = _mm256_max_ps(maximum, _mm256_and_ps(
                                _mm256_mul_ps(difference, active), absolute));
                        }
                        alignas(32) float values[8];
                        _mm256_store_ps(values, maximum);
                        for (float value : values)
                            maxAbsDL = std::max(maxAbsDL, value);
                    }
#endif
                    for (; i < ax.sup; ++i) {
                        if (amCol[i] == 0.0f)
                            continue;
                        const int tx = std::clamp(tx0 + i, 0, U.w - 1);
                        maxAbsDL = std::max(maxAbsDL, std::fabs(lcRow[tx] - L0));
                    }
                }
            } else {
                const float *amRow = &ay.am[size_t(oy) * ay.sup];
                const float *amCol = &ax.am[size_t(ox) * ax.sup];
                for (int j = 0; j < ay.sup; ++j) {
                    if (amRow[j] == 0.0f)
                        continue; // outside the logical kernel window
                    const int ty = std::clamp(ty0 + j, 0, U.h - 1);
                    const float *lcRow = guided ? gm.lc.row(ty) : nullptr;
                    const float *uRow = U.row(ty);
                    const float *vRow = V.row(ty);
                    int i = 0;
#ifdef __AVX2__
                    if (simdOK) {
                        __m256 vMaxD = _mm256_setzero_ps();
                        __m256 vMinU = _mm256_set1_ps(1e30f), vMaxU = _mm256_set1_ps(-1e30f);
                        __m256 vMinV = _mm256_set1_ps(1e30f), vMaxV = _mm256_set1_ps(-1e30f);
                        const __m256 vL0 = _mm256_set1_ps(L0);
                        const __m256 vAbs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
                        const __m256 vInf = _mm256_set1_ps(1e30f);
                        const __m256 vNInf = _mm256_set1_ps(-1e30f);
                        const __m256 vZero = _mm256_setzero_ps();
                        for (; i < ax.sup; i += 8) {
                            const __m256i laneMask = activeLaneMask(std::min(8, ax.sup - i));
                            const int tx = tx0 + i;
                            const __m256 vM = _mm256_maskload_ps(amCol + i, laneMask);
                            const __m256 vActive = _mm256_cmp_ps(vM, vZero, _CMP_GT_OQ);
                            if (guided) {
                                const __m256 dL = _mm256_sub_ps(
                                    _mm256_maskload_ps(lcRow + tx, laneMask), vL0);
                                vMaxD = _mm256_max_ps(vMaxD,
                                    _mm256_and_ps(_mm256_mul_ps(dL, vM), vAbs));
                            }
                            const __m256 vU = _mm256_maskload_ps(uRow + tx, laneMask);
                            const __m256 vV = _mm256_maskload_ps(vRow + tx, laneMask);
                            vMinU = _mm256_min_ps(vMinU, _mm256_blendv_ps(vInf, vU, vActive));
                            vMaxU = _mm256_max_ps(vMaxU, _mm256_blendv_ps(vNInf, vU, vActive));
                            vMinV = _mm256_min_ps(vMinV, _mm256_blendv_ps(vInf, vV, vActive));
                            vMaxV = _mm256_max_ps(vMaxV, _mm256_blendv_ps(vNInf, vV, vActive));
                        }
                        alignas(32) float buf[8];
                        _mm256_store_ps(buf, vMaxD); for (int q = 0; q < 8; ++q) maxAbsDL = std::max(maxAbsDL, buf[q]);
                        _mm256_store_ps(buf, vMinU); for (int q = 0; q < 8; ++q) hullMinU = std::min(hullMinU, buf[q]);
                        _mm256_store_ps(buf, vMaxU); for (int q = 0; q < 8; ++q) hullMaxU = std::max(hullMaxU, buf[q]);
                        _mm256_store_ps(buf, vMinV); for (int q = 0; q < 8; ++q) hullMinV = std::min(hullMinV, buf[q]);
                        _mm256_store_ps(buf, vMaxV); for (int q = 0; q < 8; ++q) hullMaxV = std::max(hullMaxV, buf[q]);
                    }
#endif
                    for (; i < ax.sup; ++i) {
                        if (amCol[i] == 0.0f)
                            continue;
                        const int tx = std::clamp(tx0 + i, 0, U.w - 1);
                        if (guided)
                            maxAbsDL = std::max(maxAbsDL, std::fabs(lcRow[tx] - L0));
                        hullMinU = std::min(hullMinU, uRow[tx]);
                        hullMaxU = std::max(hullMaxU, uRow[tx]);
                        hullMinV = std::min(hullMinV, vRow[tx]);
                        hullMaxV = std::max(hullMaxV, vRow[tx]);
                    }
                }
            }

            // Adaptive similarity knee: snap whenever the window contains a
            // real luma step, stay conservative on flat/noisy luma.
            // Sigma multiplier: sratio on hard steps, sdb (much larger,
            // kills discrimination) on ramps, smoothly blended.
            const float t2 = std::clamp(minDiffN / (0.1f * maxAbsDL + 1e-6f), 0.0f, 1.0f);
            ssRamp = t2 * t2 * (3.0f - 2.0f * t2); // smoothstep; 0 = hard step, 1 = ramp
            const float sigMult = float(d.sratio) + (float(d.sdb) - float(d.sratio)) * ssRamp;
            const float sigLoc = std::max(float(d.sigma), sigMult * maxAbsDL);
            const float invSigma2 = 1.0f / (sigLoc * sigLoc);
            // Ridge fade: 1 = full guidance, 0 = plain kernel (thin line)
            const float ridgeFade =
                1.0f - std::clamp(2.0f * ridgeRev / (maxAbsDL + 1e-6f), 0.0f, 1.0f);

            // Chroma-edge-presence fade: estimate the source chroma
            // transition width along the edge normal, W_C = range / max
            // along-normal adjacent-pair slope. A hard chroma edge destroyed
            // by subsampling measures ~1-1.5 chroma px (downsample blur
            // floor) -> keep guidance; a genuinely soft chroma blend (line
            // art) measures >= 2 -> the source chroma is telling the truth,
            // respect it and fall back to the plain kernel.
            float cedgeFade = 1.0f;
            if (d.cedge && guided && hasDir) {
                const float rcMax = std::max(hullMaxU - hullMinU, hullMaxV - hullMinV);
                if (rcMax > 1e-6f) {
                    float gmax = 0.0f;
                    const float anx = std::fabs(nxv), any = std::fabs(nyv);
                    for (int j = 0; j < ay.sup; ++j) {
                        const int ty = std::clamp(ty0 + j, 0, U.h - 1);
                        for (int i = 0; i < ax.sup; ++i) {
                            const int tx = std::clamp(tx0 + i, 0, U.w - 1);
                            const int tx1 = std::min(tx + 1, U.w - 1);
                            const int ty1 = std::min(ty + 1, U.h - 1);
                            const float gu = std::max(std::fabs(U.at(tx1, ty) - U.at(tx, ty)) * anx,
                                                      std::fabs(U.at(tx, ty1) - U.at(tx, ty)) * any);
                            const float gv = std::max(std::fabs(V.at(tx1, ty) - V.at(tx, ty)) * anx,
                                                      std::fabs(V.at(tx, ty1) - V.at(tx, ty)) * any);
                            gmax = std::max(gmax, std::max(gu, gv));
                        }
                    }
                    const float wc = rcMax / (gmax + 1e-6f); // chroma px
                    cedgeFade = std::clamp((2.2f - wc) / 0.7f, 0.0f, 1.0f);
                }
            }
            // Mutual-structure co-edge fade: the chroma planes must confirm
            // an edge co-located (direction/position/width) with the luma
            // edge, otherwise luma high-frequency transfer is gated off and
            // the plain kernel's (truthful) softness is kept. This is what
            // fixes the hardL_softC residual: a soft chroma blend under a
            // hard luma edge fails the profile-correlation test.
            const float msFade = meta.mutualFade;
            // strength scales the sim discrimination itself: lambda -> 0
            // must degrade continuously to the plain kernel
            const float guideFade = ridgeFade * cedgeFade * msFade * strength;

            // Phase-0 rescue: at subpixel phase ~0 the base kernel is a delta
            // and returns the coincident source sample unchanged. That sample
            // is exactly right on smooth content but corrupted when its
            // footprint straddles a luma edge — detectable via its own sim.
            // Only then do we add the (positive, blurring) bump kernel to
            // rebuild the sample from same-side neighbors.
            float rescue = 0.0f;
            if (guided) {
                const int ncx = std::clamp(int(std::lround(scx)), 0, gm.lc.w - 1);
                const int ncy = std::clamp(int(std::lround(scy)), 0, gm.lc.h - 1);
                const float dL0 = gm.lc.at(ncx, ncy) - L0;
                const float sim0 = 1.0f / (1.0f + dL0 * dL0 * invSigma2);
                const float pg = meta.phaseGate;
                rescue = float(d.rescue) * guideFade * pg * pg * (1.0f - sim0);
                // Per-kernel-family rescue gating: the "phase 0 = delta"
                // assumption only holds for separable kernels. jinc(1)=0.18,
                // so at phase 0 jinc already draws from neighbors and needs
                // proportionally less rescue. kappa = wmax^2 / sum(w^2) is 1
                // for a delta and smaller for spread kernels.
                if (rescue > 0.0f) {
                    float kappa;
                    if (!radial) {
                        float mxX = 0, ssX = 0, mxY = 0, ssY = 0;
                        for (int i = 0; i < ax.sup; ++i)
                            if (ax.am[size_t(ox) * ax.sup + i] > 0.0f) {
                                const float w = wxp[i];
                                mxX = std::max(mxX, std::fabs(w));
                                ssX += w * w;
                            }
                        for (int j = 0; j < ay.sup; ++j)
                            if (ay.am[size_t(oy) * ay.sup + j] > 0.0f) {
                                const float w = wyp[j];
                                mxY = std::max(mxY, std::fabs(w));
                                ssY += w * w;
                            }
                        kappa = (mxX * mxX * mxY * mxY) / (ssX * ssY + 1e-12f);
                    } else {
                        float mx = 0, ss = 0;
                        for (int j = 0; j < ay.sup; ++j) {
                            if (ay.am[size_t(oy) * ay.sup + j] == 0.0f) continue;
                            const float dyc = (ty0 + j - scy) * invWyC;
                            for (int i = 0; i < ax.sup; ++i) {
                                if (ax.am[size_t(ox) * ax.sup + i] == 0.0f) continue;
                                const float dxc = (tx0 + i - scx) * invWxC;
                                const float w = radialBase
                                    ? radialBase[size_t(j) * ax.sup + i]
                                    : lutLookup(lut, lutN,
                                        std::sqrt(dxc * dxc + dyc * dyc) * d.lutScale);
                                mx = std::max(mx, std::fabs(w));
                                ss += w * w;
                            }
                        }
                        kappa = (mx * mx) / (ss + 1e-12f);
                    }
                    rescue *= kappa * kappa;
                }
            }
            // Along-edge support stretch (EWA-style anisotropy)
            const float aniso = 1.0f + float(d.stretch) * coherence;
            const float invGsigma2Par = invGsigma2 / (aniso * aniso);

            double accU = 0.0, accV = 0.0, wsum = 0.0, sumAbsW = 0.0;

#ifdef __AVX2__
            if (simdOK) {
                __m256 vAccU = _mm256_setzero_ps();
                __m256 vAccV = _mm256_setzero_ps();
                __m256 vSum = _mm256_setzero_ps();
                __m256 vSumAbs = _mm256_setzero_ps();
                const __m256 vL0 = _mm256_set1_ps(L0);
                const __m256 vRescue = _mm256_set1_ps(rescue);
                const __m256 vRidge = _mm256_set1_ps(guideFade);
                const __m256 vOne = _mm256_set1_ps(1.0f);
                const __m256 vIs2 = _mm256_set1_ps(invSigma2);
                const __m256 vIg2 = _mm256_set1_ps(invGsigma2);
                const __m256 vIg2p = _mm256_set1_ps(invGsigma2Par);
                const __m256 vNx = _mm256_set1_ps(nxv);
                const __m256 vNy = _mm256_set1_ps(nyv);
                const __m256 vStep = _mm256_set1_ps(float(job.rw));
                const __m256 vDxBase = _mm256_fmadd_ps(
                    _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), vStep,
                    _mm256_set1_ps((tx0 - scx) * float(job.rw)));

                const float *amRow2 = &ay.am[size_t(oy) * ay.sup];
                const float *amCol2 = &ax.am[size_t(ox) * ax.sup];
                for (int j = 0; j < ay.sup; ++j) {
                    if (amRow2[j] == 0.0f)
                        continue;
                    const int ty = std::clamp(ty0 + j, 0, U.h - 1);
                    const float wy = wyp[j];
                    const float dy = (ty0 + j - scy) * float(job.rh); // luma units
                    const float *lcRow = guided ? gm.lc.row(ty) : nullptr;
                    const float *uRow = U.row(ty);
                    const float *vRow = V.row(ty);
                    const __m256 vWy = _mm256_set1_ps(wy);
                    const __m256 vDy = _mm256_set1_ps(dy);

                    for (int i = 0; i < ax.sup; i += 8) {
                        const __m256i laneMask = activeLaneMask(std::min(8, ax.sup - i));
                        const int tx = tx0 + i; // guaranteed in-range by start clamping
                        const __m256 vAm = _mm256_maskload_ps(amCol2 + i, laneMask);
                        const __m256 vU = _mm256_maskload_ps(uRow + tx, laneMask);
                        const __m256 vV = _mm256_maskload_ps(vRow + tx, laneMask);
                        __m256 vWx;
                        if (radialBase) {
                            vWx = _mm256_maskload_ps(
                                radialBase + size_t(j) * ax.sup + i, laneMask);
                        } else if (radial) {
                            // 2D radial base weight from the jinc LUT
                            const __m256 vDxc = _mm256_mul_ps(
                                _mm256_add_ps(vDxBase, _mm256_set1_ps(float(i) * float(job.rw))),
                                _mm256_set1_ps(float(1.0 / job.rw) * invWxC));
                            const __m256 dyc = _mm256_set1_ps(dy * float(1.0 / job.rh) * invWyC);
                            const __m256 r2 = _mm256_fmadd_ps(vDxc, vDxc, _mm256_mul_ps(dyc, dyc));
                            vWx = lutLookup8(lut, lutN,
                                _mm256_mul_ps(_mm256_sqrt_ps(r2), _mm256_set1_ps(d.lutScale)));
                        } else {
                            vWx = _mm256_mul_ps(
                                _mm256_maskload_ps(wxp + i, laneMask), vWy);
                        }

                        __m256 wG;
                        if (guided) {
                            const __m256 vLc = _mm256_maskload_ps(lcRow + tx, laneMask);
                            const __m256 vDx = _mm256_add_ps(vDxBase,
                                _mm256_set1_ps(float(i) * float(job.rw)));
                            // sim = 1 / (1 + dL^2 * invSigma2)
                            const __m256 dL = _mm256_sub_ps(vLc, vL0);
                            const __m256 simRaw = rcpNR(
                                _mm256_fmadd_ps(_mm256_mul_ps(dL, dL), vIs2, vOne));
                            // fade discrimination on thin lines: sim -> 1
                            const __m256 sim = _mm256_fnmadd_ps(vRidge,
                                _mm256_sub_ps(vOne, simRaw), vOne);
                            // anisotropic bump: stretched along the edge tangent
                            // t=(-ny,nx), never across it
                            const __m256 dp = _mm256_fmadd_ps(vDx, vNx, _mm256_mul_ps(vDy, vNy));
                            const __m256 dt = _mm256_fnmadd_ps(vDx, vNy, _mm256_mul_ps(vDy, vNx));
                            const __m256 m2 = _mm256_fmadd_ps(_mm256_mul_ps(dp, dp), vIg2,
                                _mm256_mul_ps(_mm256_mul_ps(dt, dt), vIg2p));
                            const __m256 bump = rcpNR(_mm256_add_ps(vOne, m2));
                            // W' = sim * (W + rescue*bump)
                            wG = _mm256_mul_ps(sim,
                                _mm256_fmadd_ps(vRescue, bump, vWx));
                        } else {
                            wG = vWx;
                        }

                        wG = _mm256_mul_ps(wG, vAm); // mask inactive/tail lanes
                        vAccU = _mm256_fmadd_ps(wG, vU, vAccU);
                        vAccV = _mm256_fmadd_ps(wG, vV, vAccV);
                        vSum = _mm256_add_ps(vSum, wG);
                        vSumAbs = _mm256_add_ps(vSumAbs,
                            _mm256_and_ps(wG, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff))));
                    }
                }
                alignas(32) float buf[8];
                _mm256_store_ps(buf, vAccU); for (int q = 0; q < 8; ++q) accU += buf[q];
                _mm256_store_ps(buf, vAccV); for (int q = 0; q < 8; ++q) accV += buf[q];
                _mm256_store_ps(buf, vSum);  for (int q = 0; q < 8; ++q) wsum += buf[q];
                _mm256_store_ps(buf, vSumAbs); for (int q = 0; q < 8; ++q) sumAbsW += buf[q];
            } else
#endif
            {
                const float *amRow2 = &ay.am[size_t(oy) * ay.sup];
                const float *amCol2 = &ax.am[size_t(ox) * ax.sup];
                for (int j = 0; j < ay.sup; ++j) {
                    if (amRow2[j] == 0.0f)
                        continue;
                    const int ty = std::clamp(ty0 + j, 0, U.h - 1);
                    const float wy = wyp[j];
                    const float dy = (ty0 + j - scy) * float(job.rh); // luma units
                    const float *lcRow = guided ? gm.lc.row(ty) : nullptr;
                    const float *uRow = U.row(ty);
                    const float *vRow = V.row(ty);
                    for (int i = 0; i < ax.sup; ++i) {
                        if (amCol2[i] == 0.0f)
                            continue;
                        const int tx = std::clamp(tx0 + i, 0, U.w - 1);
                        float wx;
                        if (radialBase) {
                            wx = radialBase[size_t(j) * ax.sup + i];
                        } else if (radial) {
                            const float dxc = (tx0 + i - scx) * invWxC; // chroma units, AA-scaled
                            const float dyc = (ty0 + j - scy) * invWyC;
                            wx = lutLookup(lut, lutN, std::sqrt(dxc * dxc + dyc * dyc) * d.lutScale);
                        } else {
                            wx = wxp[i] * wy;
                        }
                        float w;
                        if (guided) {
                            const float dx = (tx0 + i - scx) * float(job.rw); // luma units
                            const float dL = lcRow[tx] - L0;
                            const float dp = dx * nxv + dy * nyv;   // across edge
                            const float dt = -dx * nyv + dy * nxv;  // along edge
                            const float simRaw = 1.0f / (1.0f + dL * dL * invSigma2);
                            const float sim = 1.0f - guideFade * (1.0f - simRaw);
                            const float bump = 1.0f / (1.0f + dp * dp * invGsigma2 +
                                                       dt * dt * invGsigma2Par);
                            w = sim * (wx + rescue * bump);
                        } else {
                            w = wx;
                        }
                        accU += double(w) * uRow[tx];
                        accV += double(w) * vRow[tx];
                        wsum += w;
                        sumAbsW += std::fabs(w);
                    }
                }
            }

            // Temporal taps (TRecon): each motion-compensated neighbor frame
            // contributes its ACTUAL chroma sample nearest the mapped
            // position. The neighbor sample at (tnx,tny) holds current-frame
            // content at (tnx - mx/rw, tny - my/rh) chroma px; its weight is
            // the BASE KERNEL evaluated at that true offset — the tap is just
            // an off-lattice spatial sample, modulated by sim and the ME
            // block confidence. Phase novelty gate: motion by an ODD number
            // of luma px shifts the 420 sampling phase by half a chroma px
            // (genuinely new information); integer chroma-pel motion
            // re-samples the same lattice and only dilutes the signed base
            // kernel, so it is gated to zero for moving blocks. (An earlier
            // version keyed novelty on the rounding residual ddx, which
            // rewarded exactly the redundant even-pel case.)
            if (guided && job.nbrs && !job.nbrs->empty()) {
                const int lx = std::clamp(int(ax.lpos[ox]), 0, job.srcLumaW - 1);
                const int ly = std::clamp(int(ay.lpos[oy]), 0, job.srcLumaH - 1);
                const float frw = float(job.rw), frh = float(job.rh);
                for (const TemporalNbr &nb : *job.nbrs) {
                    const int bxx = std::min(nb.bw - 1, lx / nb.block);
                    const int byy = std::min(nb.bh - 1, ly / nb.block);
                    const size_t bi = size_t(byy) * nb.bw + bxx;
                    const float tconf = (*nb.tconf)[bi];
                    if (tconf < 0.05f)
                        continue;
                    const float mx = float((*nb.mvx)[bi]);
                    const float my = float((*nb.mvy)[bi]);
                    // nearest neighbor chroma sample to the mapped position
                    const int tnx = std::clamp(int(std::lround(scx + mx / frw)),
                                               0, nb.U->w - 1);
                    const int tny = std::clamp(int(std::lround(scy + my / frh)),
                                               0, nb.U->h - 1);
                    // offset of the content it represents from the target,
                    // in current-frame luma px (<= half a chroma px after
                    // the nearest-sample rounding above)
                    const float ddx = (tnx - scx) * frw - mx;
                    const float ddy = (tny - scy) * frh - my;
                    const bool staticBlk0 = (std::fabs(mx) + std::fabs(my) <= 1.0f) && tconf > 0.5f;
                    float dL;
                    if (staticBlk0) {
                        // Static averaging: same content, level noise is what
                        // we average away -> lenient footprint-level test.
                        dL = nb.lc->at(tnx, tny) - L0;
                    } else {
                        // Moving block: EXACT footprint-straddle test against
                        // the sharp SOURCE luma. The tap is a box average over
                        // the motion-compensated footprint; if ANY source luma
                        // sample inside that box differs from L0, the tap
                        // straddles a luma edge and its value is a cross-side
                        // mixture. A level test against the footprint-average
                        // lc cannot see a partial straddle (7% straddle moved
                        // lc by only 0.0125 with sigma=0.01) — this caused the
                        // odd-pel-motion regression.
                        const Plane &sY = *job.srcY;
                        const double lx = (tnx + 0.5) * job.rw - 0.5 + job.shiftX - mx;
                        const double ly = (tny + 0.5) * job.rh - 0.5 + job.shiftY - my;
                        const int x0 = int(std::ceil(lx - 0.5 * job.rw));
                        const int x1 = int(std::ceil(lx + 0.5 * job.rw));
                        const int y0 = int(std::ceil(ly - 0.5 * job.rh));
                        const int y1 = int(std::ceil(ly + 0.5 * job.rh));
                        float md = 0.0f;
                        for (int jj = y0; jj < y1; ++jj)
                            for (int ii = x0; ii < x1; ++ii)
                                md = std::max(md, std::fabs(sY.at(ii, jj) - L0));
                        dL = md;
                    }
                    const float sigT = staticBlk0 ? sigLoc : 0.5f * sigLoc; // sigLoc >= sigma > 0
                    const float simRaw = 1.0f / (1.0f + dL * dL / (sigT * sigT));
                    // Moving taps are all-or-nothing additives: use simRaw
                    // directly (the guideFade blend toward 1 exists for the
                    // spatial kernel's gradual strength semantics; here it
                    // would let half-rejected taps through at ~0.3 weight).
                    const float sim = staticBlk0 ? 1.0f - guideFade * (1.0f - simRaw) : simRaw;
                    // base-kernel weight at the true (post-rounding) offset
                    float kw;
                    if (radial) {
                        const float dxc = ddx / frw * invWxC;
                        const float dyc = ddy / frh * invWyC;
                        kw = lutLookup(lut, lutN,
                            std::sqrt(dxc * dxc + dyc * dyc) * d.lutScale);
                    } else {
                        kw = float(kernelEval(d.kernel, std::fabs(ddx) / job.rw * invWxC,
                                              d.kp1, d.kp2) *
                                   kernelEval(d.kernel, std::fabs(ddy) / job.rh * invWyC,
                                              d.kp1, d.kp2));
                    }
                    // Static blocks: redundant taps are pure temporal
                    // averaging (denoise) -> keep full weight. Moving blocks
                    // require phase NOVELTY: the fractional part of the
                    // motion in chroma units must be ~1/2 (odd luma-pel),
                    // otherwise the tap re-samples the existing lattice.
                    const bool staticBlk = staticBlk0;
                    float nw = 1.0f;
                    if (!staticBlk) {
                        const float fx = mx / frw, fy = my / frh;
                        const float px = fx - std::floor(fx); // [0,1)
                        const float py = fy - std::floor(fy);
                        nw = std::max(std::fabs(std::sin(float(M_PI) * px)),
                                      std::fabs(std::sin(float(M_PI) * py)));
                        // Transitional (ramp-like) luma: the footprint-luma
                        // sim cannot assign a side there (the half-value edge
                        // column averages both sides), so a motion-compensated
                        // tap cannot be validated -> drop it. Taps fire only
                        // where the guide can discriminate (hard steps) or on
                        // static blocks (pure averaging).
                        nw *= 1.0f - ssRamp;
                    }
                    const float w = tconf * sim * kw * nw;
                    accU += double(w) * nb.U->at(tnx, tny);
                    accV += double(w) * nb.V->at(tnx, tny);
                    wsum += w;
                    sumAbsW += std::fabs(w);
                    hullMinU = std::min(hullMinU, nb.U->at(tnx, tny));
                    hullMaxU = std::max(hullMaxU, nb.U->at(tnx, tny));
                    hullMinV = std::min(hullMinV, nb.V->at(tnx, tny));
                    hullMaxV = std::max(hullMaxV, nb.V->at(tnx, tny));
                }
            }

            double valueU, valueV;
            // signed kernels: reject pathological near-cancellation
            if (wsum > 1e-6 && wsum >= 0.05 * sumAbsW) {
                valueU = accU / wsum;
                valueV = accV / wsum;
            } else {
                valueU = bilinear(U, scx, scy);
                valueV = bilinear(V, scx, scy);
            }

            if (clampHull) {
                valueU = std::clamp(valueU, double(hullMinU - margin), double(hullMaxU + margin));
                valueV = std::clamp(valueV, double(hullMinV - margin), double(hullMaxV + margin));
            }

            if (job.selectorMaps || job.selectorMetadata) {
                // selector weights: hard-edge-ness times axis/diagonal split.
                // diag = 2|nx*ny|: 0 on axis-aligned edges, 1 at 45 degrees
                const float diag = hasDir ? std::min(1.0f, 2.0f * std::fabs(nxv * nyv)) : 0.0f;
                const float hedgy = (1.0f - ssRamp) * guideFade;
                const float w2 = hedgy * (1.0f - diag) * job.selectorStrength;
                const float w3 = hedgy * diag * job.selectorStrength;
                if (job.compressedSelector) {
                    const float baseU = job.plainU->row(oy)[ox];
                    const float baseV = job.plainV->row(oy)[ox];
                    if (w2 < 1e-4f && w3 < 1e-4f) {
                        job.compressedSelector->guidedDeltaU[compressedIndex] = 0.0f;
                        job.compressedSelector->guidedDeltaV[compressedIndex] = 0.0f;
                        job.compressedSelector->w3[compressedIndex] = 0.0f;
                    } else {
                        job.compressedSelector->guidedDeltaU[compressedIndex] =
                            hedgy * (1.0f - diag) * (float(valueU) - baseU);
                        job.compressedSelector->guidedDeltaV[compressedIndex] =
                            hedgy * (1.0f - diag) * (float(valueV) - baseV);
                        job.compressedSelector->w3[compressedIndex] = hedgy * diag;
                    }
                } else if (job.selectorW2 && job.selectorW3) {
                    // Algo4's first pass only emits routing metadata. LGF is
                    // built after this pass, so inactive ROI never allocates
                    // or evaluates its regression in the common sparse case.
                    job.selectorW2->row(oy)[ox] =
                        hedgy * (1.0f - diag);
                    job.selectorW3->row(oy)[ox] = hedgy * diag;
                } else if (job.selectorMaps && (w2 >= 1e-4f || w3 >= 1e-4f)) {
                    const LGFMaps &lgf = *job.selectorMaps;
                    const int cx = ax.chromaBilin.i0[ox];
                    const float cxf = ax.chromaBilin.f[ox];
                    const int cy = ay.chromaBilin.i0[oy];
                    const float cyf = ay.chromaBilin.f[oy];
                    const float baseU = dstRowU[ox], baseV = dstRowV[ox];
                    const float au = bilinearFast(lgf.aU, cx, cxf, cy, cyf);
                    const float bu = bilinearFast(lgf.bU, cx, cxf, cy, cyf);
                    const float av = bilinearFast(lgf.aV, cx, cxf, cy, cyf);
                    const float bv = bilinearFast(lgf.bV, cx, cxf, cy, cyf);
                    const float cu = bilinearFast(lgf.confU, cx, cxf, cy, cyf);
                    const float cv = bilinearFast(lgf.confV, cx, cxf, cy, cyf);
                    valueU = baseU + w2 * (float(valueU) - baseU)
                                   + w3 * cu * (au * L0 + bu - baseU);
                    valueV = baseV + w2 * (float(valueV) - baseV)
                                   + w3 * cv * (av * L0 + bv - baseV);
                } else {
                    valueU = dstRowU[ox];
                    valueV = dstRowV[ox];
                }
            }
            if (!job.compressedSelector) {
                dstRowU[ox] = static_cast<float>(valueU);
                dstRowV[ox] = static_cast<float>(valueV);
            }
        };
        visitActive(oy, processPixel);
    }
}

void reconstructChroma(const ChromaJob &job) {
    reconstructChromaImpl<NativeBackend>(job);
}


namespace {

template <class Backend, bool Dual>
void plainChromaImpl(const LGCRData &d, const Plane &U, const Plane &V,
                     Plane &outU, Plane &outV, const ChromaAxis &ax,
                     const ChromaAxis &ay, const RadialWeightTable *radialTable,
                     PipelineMetrics *metrics) {
    const bool radial = d.radial;
    const int lutN = radial ? static_cast<int>(d.lut.size()) : 0;
    const float *lut = radial ? d.lut.data() : nullptr;
    const float invWxC = float(std::min(1.0, double(outU.w) / U.w));
    const float invWyC = float(std::min(1.0, double(outU.h) / U.h));
    const bool clampHull = d.arMargin >= 0.0;
    const float margin = float(std::max(0.0, d.arMargin));
    const bool vectorWindow = Backend::lanes > 1 && ax.sup >= 6 && U.w >= ax.sup;

    std::vector<uint16_t> activeX, activeY;
    if (metrics) {
        const uint64_t pixels = uint64_t(outU.w) * outU.h;
        metrics->outputPixels += pixels;
        metrics->addWork(Stage::BuildBaseChroma, pixels);
        activeX.assign(outU.w, 0);
        activeY.assign(outU.h, 0);
        for (int x = 0; x < outU.w; ++x)
            for (int i = 0; i < ax.sup; ++i)
                activeX[x] += ax.am[size_t(x) * ax.sup + i] != 0.0f;
        for (int y = 0; y < outU.h; ++y)
            for (int j = 0; j < ay.sup; ++j)
                activeY[y] += ay.am[size_t(y) * ay.sup + j] != 0.0f;
    }

    for (int oy = 0; oy < outU.h; ++oy) {
        const int ty0 = ay.start[oy];
        const float *wyp = &ay.w[size_t(oy) * ay.sup];
        const float *amRow = &ay.am[size_t(oy) * ay.sup];
        const float scy = ay.pos[oy];
        float *dstU = outU.row(oy);
        float *dstV = Dual ? outV.row(oy) : nullptr;

        for (int ox = 0; ox < outU.w; ++ox) {
            const int tx0 = ax.start[ox];
            const float *wxp = &ax.w[size_t(ox) * ax.sup];
            const float *amCol = &ax.am[size_t(ox) * ax.sup];
            const float scx = ax.pos[ox];
            const float *cachedWeights = radialTable ? radialTable->at(ox, oy) : nullptr;
            [[maybe_unused]] const bool cachedNormalization = cachedWeights &&
                !radialTable->weightSum.empty();
            if (metrics) {
                const uint64_t taps = uint64_t(activeX[ox]) * activeY[oy];
                metrics->tapsVisited += taps;
                metrics->taps[static_cast<size_t>(Stage::BuildBaseChroma)] += taps;
            }

            double accU = 0.0, accV = 0.0, wsum = 0.0, sumAbsW = 0.0;
            float hullMinU = 1e30f, hullMaxU = -1e30f;
            float hullMinV = 1e30f, hullMaxV = -1e30f;

#ifdef __AVX2__
            if constexpr (Backend::lanes == 8) {
                if (vectorWindow) {
                    typename Backend::Vec vAccU = Backend::zero();
                    typename Backend::Vec vAccV = Backend::zero();
                    typename Backend::Vec vSum = Backend::zero();
                    typename Backend::Vec vSumAbs = Backend::zero();
                    typename Backend::Vec vMinU = Backend::set1(1e30f);
                    typename Backend::Vec vMaxU = Backend::set1(-1e30f);
                    typename Backend::Vec vMinV = Backend::set1(1e30f);
                    typename Backend::Vec vMaxV = Backend::set1(-1e30f);
                    const auto zero = Backend::zero();
                    for (int j = 0; j < ay.sup; ++j) {
                        if (amRow[j] == 0.0f)
                            continue;
                        const int ty = std::clamp(ty0 + j, 0, U.h - 1);
                        const float *uRow = U.row(ty);
                        const float *vRow = Dual ? V.row(ty) : nullptr;
                        for (int i = 0; i < ax.sup; i += Backend::lanes) {
                            const __m256i laneMask = activeLaneMask(
                                std::min(Backend::lanes, ax.sup - i));
                            const auto activity = _mm256_maskload_ps(amCol + i, laneMask);
                            const auto active = Backend::greater(activity, zero);
                            const auto vu = _mm256_maskload_ps(uRow + tx0 + i, laneMask);
                            typename Backend::Vec weight;
                            if (cachedWeights) {
                                weight = _mm256_maskload_ps(
                                    cachedWeights + size_t(j) * ax.sup + i, laneMask);
                            } else if (radial) {
                                const __m256 indices = _mm256_setr_ps(
                                    float(i), float(i + 1), float(i + 2), float(i + 3),
                                    float(i + 4), float(i + 5), float(i + 6), float(i + 7));
                                const __m256 dx = _mm256_mul_ps(
                                    _mm256_sub_ps(_mm256_add_ps(_mm256_set1_ps(float(tx0)), indices),
                                                  _mm256_set1_ps(scx)),
                                    _mm256_set1_ps(invWxC));
                                const float dy = (ty0 + j - scy) * invWyC;
                                const __m256 r2 = _mm256_fmadd_ps(
                                    dx, dx, _mm256_set1_ps(dy * dy));
                                weight = lutLookup8(lut, lutN,
                                    _mm256_mul_ps(_mm256_sqrt_ps(r2),
                                                  _mm256_set1_ps(d.lutScale)));
                            } else {
                                weight = Backend::mul(_mm256_maskload_ps(wxp + i, laneMask),
                                                      Backend::set1(wyp[j]));
                            }
                            if (!cachedWeights)
                                weight = Backend::mul(weight, activity);
                            vAccU = Backend::fmadd(weight, vu, vAccU);
                            if constexpr (Dual) {
                                const auto vv = _mm256_maskload_ps(vRow + tx0 + i, laneMask);
                                vAccV = Backend::fmadd(weight, vv, vAccV);
                                vMinV = Backend::min(vMinV,
                                    Backend::select(active, vv, Backend::set1(1e30f)));
                                vMaxV = Backend::max(vMaxV,
                                    Backend::select(active, vv, Backend::set1(-1e30f)));
                            }
                            if (!cachedNormalization) {
                                vSum = Backend::add(vSum, weight);
                                vSumAbs = Backend::add(vSumAbs, Backend::abs(weight));
                            }
                            vMinU = Backend::min(vMinU,
                                Backend::select(active, vu, Backend::set1(1e30f)));
                            vMaxU = Backend::max(vMaxU,
                                Backend::select(active, vu, Backend::set1(-1e30f)));
                        }
                    }
                    accU = Backend::horizontalSum(vAccU);
                    if constexpr (Dual)
                        accV = Backend::horizontalSum(vAccV);
                    if (cachedNormalization) {
                        const size_t phase = radialTable->phaseAt(ox, oy);
                        wsum = radialTable->weightSum[phase];
                        sumAbsW = radialTable->absoluteWeightSum[phase];
                    } else {
                        wsum = Backend::horizontalSum(vSum);
                        sumAbsW = Backend::horizontalSum(vSumAbs);
                    }
                    hullMinU = Backend::horizontalMin(vMinU);
                    hullMaxU = Backend::horizontalMax(vMaxU);
                    if constexpr (Dual) {
                        hullMinV = Backend::horizontalMin(vMinV);
                        hullMaxV = Backend::horizontalMax(vMaxV);
                    }
                }
            }
#endif
            if (!vectorWindow) {
                for (int j = 0; j < ay.sup; ++j) {
                    if (amRow[j] == 0.0f)
                        continue;
                    const int ty = std::clamp(ty0 + j, 0, U.h - 1);
                    const float *uRow = U.row(ty);
                    const float *vRow = Dual ? V.row(ty) : nullptr;
                    for (int i = 0; i < ax.sup; ++i) {
                        if (amCol[i] == 0.0f)
                            continue;
                        const int tx = std::clamp(tx0 + i, 0, U.w - 1);
                        float weight;
                        if (cachedWeights) {
                            weight = cachedWeights[size_t(j) * ax.sup + i];
                        } else if (radial) {
                            const float dx = (tx0 + i - scx) * invWxC;
                            const float dy = (ty0 + j - scy) * invWyC;
                            weight = lutLookup(lut, lutN,
                                std::sqrt(dx * dx + dy * dy) * d.lutScale);
                        } else {
                            weight = wxp[i] * wyp[j];
                        }
                        const float u = uRow[tx];
                        accU += double(weight) * u;
                        wsum += weight;
                        sumAbsW += std::fabs(weight);
                        hullMinU = std::min(hullMinU, u); hullMaxU = std::max(hullMaxU, u);
                        if constexpr (Dual) {
                            const float v = vRow[tx];
                            accV += double(weight) * v;
                            hullMinV = std::min(hullMinV, v);
                            hullMaxV = std::max(hullMaxV, v);
                        }
                    }
                }
            }

            double valueU, valueV = 0.0;
            if (wsum > 1e-6 && wsum >= 0.05 * sumAbsW) {
                valueU = accU / wsum;
                if constexpr (Dual)
                    valueV = accV / wsum;
            } else {
                valueU = bilinear(U, scx, scy);
                if constexpr (Dual)
                    valueV = bilinear(V, scx, scy);
            }
            if (clampHull) {
                valueU = std::clamp(valueU, double(hullMinU - margin),
                                    double(hullMaxU + margin));
                if constexpr (Dual)
                    valueV = std::clamp(valueV, double(hullMinV - margin),
                                        double(hullMaxV + margin));
            }
            dstU[ox] = static_cast<float>(valueU);
            if constexpr (Dual)
                dstV[ox] = static_cast<float>(valueV);
        }
    }
}

template <class Backend, bool Dual>
void plainChromaBilinear(const Plane &U, const Plane &V,
                         Plane *outU, Plane *outV, int outputWidth,
                         int outputHeight, const ChromaAxis &ax,
                         const ChromaAxis &ay, PipelineMetrics *metrics,
                         PlainPlaneRowSink rowSink = nullptr,
                         void *rowSinkContext = nullptr) {
    const BilinAxis &bx = ax.chromaBilin;
    const BilinAxis &by = ay.chromaBilin;
    Plane sinkRowU, sinkRowV;
    if (rowSink) {
        sinkRowU = Plane(outputWidth, 1);
        if constexpr (Dual)
            sinkRowV = Plane(outputWidth, 1);
    }
    for (int oy = 0; oy < outputHeight; ++oy) {
        const int y0 = by.i0[oy];
        const float fy = by.f[oy];
        float *du = rowSink ? sinkRowU.row(0) : outU->row(oy);
        float *dv = nullptr;
        if constexpr (Dual)
            dv = rowSink ? sinkRowV.row(0) : outV->row(oy);
        int ox = 0;
#ifdef __AVX2__
        if constexpr (Backend::lanes == 8) {
            if (U.w >= 2) {
                const int y1 = U.h == 1 ? y0 : y0 + 1;
                const float *u0 = U.row(y0), *u1 = U.row(y1);
                const float *v0 = Dual ? V.row(y0) : nullptr;
                const float *v1 = Dual ? V.row(y1) : nullptr;
                const __m256 vfy = _mm256_set1_ps(fy);
                const __m256i one = _mm256_set1_epi32(1);
                if (bx.phase2x.enabled) {
                    for (; ox < bx.phase2x.simdBegin; ++ox) {
                        du[ox] = bilinearFast(U, bx.i0[ox], bx.f[ox], y0, fy);
                        if constexpr (Dual)
                            dv[ox] = bilinearFast(V, bx.i0[ox], bx.f[ox], y0, fy);
                    }
                    for (; ox < bx.phase2x.simdEnd; ox += 8) {
                        const int base = bx.i0[ox];
                        const __m256i indices = _mm256_sub_epi32(
                            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
                                bx.i0.data() + ox)), _mm256_set1_epi32(base));
                        const __m256 ixFrac = _mm256_loadu_ps(bx.f.data() + ox);
                        const __m256 u0Samples = _mm256_loadu_ps(u0 + base);
                        const __m256 u1Samples = _mm256_loadu_ps(u1 + base);
                        const __m256 ua = _mm256_permutevar8x32_ps(u0Samples, indices);
                        const __m256 ub = _mm256_permutevar8x32_ps(
                            u0Samples, _mm256_add_epi32(indices, one));
                        const __m256 uc = _mm256_permutevar8x32_ps(u1Samples, indices);
                        const __m256 ud = _mm256_permutevar8x32_ps(
                            u1Samples, _mm256_add_epi32(indices, one));
                        const __m256 ut = _mm256_fmadd_ps(ixFrac, _mm256_sub_ps(ub, ua), ua);
                        const __m256 ubase = _mm256_fmadd_ps(ixFrac, _mm256_sub_ps(ud, uc), uc);
                        _mm256_storeu_ps(du + ox,
                            _mm256_fmadd_ps(vfy, _mm256_sub_ps(ubase, ut), ut));
                        if constexpr (Dual) {
                            const __m256 v0Samples = _mm256_loadu_ps(v0 + base);
                            const __m256 v1Samples = _mm256_loadu_ps(v1 + base);
                            const __m256 va = _mm256_permutevar8x32_ps(v0Samples, indices);
                            const __m256 vb = _mm256_permutevar8x32_ps(
                                v0Samples, _mm256_add_epi32(indices, one));
                            const __m256 vc = _mm256_permutevar8x32_ps(v1Samples, indices);
                            const __m256 vd = _mm256_permutevar8x32_ps(
                                v1Samples, _mm256_add_epi32(indices, one));
                            const __m256 vt = _mm256_fmadd_ps(ixFrac, _mm256_sub_ps(vb, va), va);
                            const __m256 vbase = _mm256_fmadd_ps(ixFrac, _mm256_sub_ps(vd, vc), vc);
                            _mm256_storeu_ps(dv + ox,
                                _mm256_fmadd_ps(vfy, _mm256_sub_ps(vbase, vt), vt));
                        }
                    }
                } else for (; ox + 8 <= outputWidth; ox += 8) {
                    const __m256i ix = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(bx.i0.data() + ox));
                    const __m256 ixFrac = _mm256_loadu_ps(bx.f.data() + ox);
                    const __m256 ua = _mm256_i32gather_ps(u0, ix, 4);
                    const __m256 ub = _mm256_i32gather_ps(u0, _mm256_add_epi32(ix, one), 4);
                    const __m256 uc = _mm256_i32gather_ps(u1, ix, 4);
                    const __m256 ud = _mm256_i32gather_ps(u1, _mm256_add_epi32(ix, one), 4);
                    const __m256 ut = _mm256_fmadd_ps(ixFrac, _mm256_sub_ps(ub, ua), ua);
                    const __m256 ubase = _mm256_fmadd_ps(ixFrac, _mm256_sub_ps(ud, uc), uc);
                    _mm256_storeu_ps(du + ox,
                        _mm256_fmadd_ps(vfy, _mm256_sub_ps(ubase, ut), ut));
                    if constexpr (Dual) {
                        const __m256 va = _mm256_i32gather_ps(v0, ix, 4);
                        const __m256 vb = _mm256_i32gather_ps(v0, _mm256_add_epi32(ix, one), 4);
                        const __m256 vc = _mm256_i32gather_ps(v1, ix, 4);
                        const __m256 vd = _mm256_i32gather_ps(v1, _mm256_add_epi32(ix, one), 4);
                        const __m256 vt = _mm256_fmadd_ps(ixFrac, _mm256_sub_ps(vb, va), va);
                        const __m256 vbase = _mm256_fmadd_ps(ixFrac, _mm256_sub_ps(vd, vc), vc);
                        _mm256_storeu_ps(dv + ox,
                            _mm256_fmadd_ps(vfy, _mm256_sub_ps(vbase, vt), vt));
                    }
                }
            }
        }
#endif
        for (; ox < outputWidth; ++ox) {
            du[ox] = bilinearFast(U, bx.i0[ox], bx.f[ox], y0, fy);
            if constexpr (Dual)
                dv[ox] = bilinearFast(V, bx.i0[ox], bx.f[ox], y0, fy);
        }
        if (rowSink)
            rowSink(rowSinkContext, oy, du);
    }

    if (metrics) {
        uint64_t activeX = 0, activeY = 0;
        for (uint16_t count : ax.activeTaps)
            activeX += count;
        for (uint16_t count : ay.activeTaps)
            activeY += count;
        const uint64_t pixels = uint64_t(outputWidth) * outputHeight;
        const uint64_t taps = activeX * activeY;
        metrics->outputPixels += pixels;
        metrics->tapsVisited += taps;
        metrics->addWork(Stage::BuildBaseChroma, pixels, taps);
    }
}

template <class Backend, bool Dual>
void plainChromaSeparable(const LGCRData &d, const Plane &U, const Plane &V,
                          Plane *outU, Plane *outV, int outputWidth,
                          int outputHeight, const ChromaAxis &ax,
                          const ChromaAxis &ay, PipelineMetrics *metrics,
                          CompressedChromaHull *hull,
                          const SparseWorkset *workset,
                          PlainPlaneRowSink rowSink = nullptr,
                          void *rowSinkContext = nullptr) {
    const auto functionStart = std::chrono::steady_clock::now();
    uint64_t horizontalNs = 0;
    const int horizontalRows = std::min(U.h, std::max(1, ay.sup));
    Plane horizontalU(outputWidth, horizontalRows), horizontalV;
    Plane horizontalMinU(outputWidth, horizontalRows), horizontalMaxU(outputWidth, horizontalRows);
    Plane horizontalMinV, horizontalMaxV;
    if constexpr (Dual) {
        horizontalV = Plane(outputWidth, horizontalRows);
        horizontalMinV = Plane(outputWidth, horizontalRows);
        horizontalMaxV = Plane(outputWidth, horizontalRows);
    }
    std::vector<int> cachedSourceRows(horizontalRows, -1);
    uint64_t horizontalRowsComputed = 0;
    const std::vector<float> &sumX = ax.weightSum;
    const std::vector<float> &sumAbsX = ax.absoluteWeightSum;
    const std::vector<float> &sumY = ay.weightSum;
    const std::vector<float> &sumAbsY = ay.absoluteWeightSum;

    auto computeHorizontalRow = [&](int sourceY) {
        const auto horizontalStart = std::chrono::steady_clock::now();
        const int slot = sourceY % horizontalRows;
        const float *sourceU = U.row(sourceY);
        const float *sourceV = Dual ? V.row(sourceY) : nullptr;
        int ox = 0;
#ifdef __AVX2__
        if constexpr (Backend::lanes == 8) {
            const __m256 zero = _mm256_setzero_ps();
            const __m256 positiveInfinity = _mm256_set1_ps(1e30f);
            const __m256 negativeInfinity = _mm256_set1_ps(-1e30f);
            const __m256i zeroIndex = _mm256_setzero_si256();
            const __m256i lastIndex = _mm256_set1_epi32(U.w - 1);
            for (; ox + 8 <= outputWidth; ox += 8) {
                const __m256i starts = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(ax.start.data() + ox));
                __m256 valueU = zero, valueV = zero;
                __m256 minimumU = positiveInfinity, maximumU = negativeInfinity;
                __m256 minimumV = positiveInfinity, maximumV = negativeInfinity;
                for (int tap = 0; tap < ax.sup; ++tap) {
                    __m256i indices = _mm256_add_epi32(
                        starts, _mm256_set1_epi32(tap));
                    indices = _mm256_max_epi32(zeroIndex,
                        _mm256_min_epi32(lastIndex, indices));
                    const __m256 active = _mm256_loadu_ps(
                        ax.tapActivity.data() + size_t(tap) * ax.n + ox);
                    const __m256 activeMask = _mm256_cmp_ps(
                        active, zero, _CMP_GT_OQ);
                    const __m256 weights = _mm256_loadu_ps(
                        ax.tapWeights.data() + size_t(tap) * ax.n + ox);
                    const __m256 samplesU = _mm256_i32gather_ps(sourceU, indices, 4);
                    valueU = _mm256_fmadd_ps(weights, samplesU, valueU);
                    minimumU = _mm256_min_ps(minimumU,
                        _mm256_blendv_ps(positiveInfinity, samplesU, activeMask));
                    maximumU = _mm256_max_ps(maximumU,
                        _mm256_blendv_ps(negativeInfinity, samplesU, activeMask));
                    if constexpr (Dual) {
                        const __m256 samplesV = _mm256_i32gather_ps(sourceV, indices, 4);
                        valueV = _mm256_fmadd_ps(weights, samplesV, valueV);
                        minimumV = _mm256_min_ps(minimumV,
                            _mm256_blendv_ps(positiveInfinity, samplesV, activeMask));
                        maximumV = _mm256_max_ps(maximumV,
                            _mm256_blendv_ps(negativeInfinity, samplesV, activeMask));
                    }
                }
                _mm256_storeu_ps(horizontalU.row(slot) + ox, valueU);
                _mm256_storeu_ps(horizontalMinU.row(slot) + ox, minimumU);
                _mm256_storeu_ps(horizontalMaxU.row(slot) + ox, maximumU);
                if constexpr (Dual) {
                    _mm256_storeu_ps(horizontalV.row(slot) + ox, valueV);
                    _mm256_storeu_ps(horizontalMinV.row(slot) + ox, minimumV);
                    _mm256_storeu_ps(horizontalMaxV.row(slot) + ox, maximumV);
                }
            }
        }
#endif
        for (; ox < outputWidth; ++ox) {
            const int start = ax.start[ox];
            const float *weights = &ax.w[size_t(ox) * ax.sup];
            const float *activity = &ax.am[size_t(ox) * ax.sup];
            double valueU = 0.0, valueV = 0.0;
            float minimumU = 1e30f, maximumU = -1e30f;
            float minimumV = 1e30f, maximumV = -1e30f;
            for (int i = 0; i < ax.sup; ++i) {
                if (activity[i] == 0.0f)
                    continue;
                const int x = std::clamp(start + i, 0, U.w - 1);
                const float u = sourceU[x];
                valueU += double(weights[i]) * u;
                minimumU = std::min(minimumU, u); maximumU = std::max(maximumU, u);
                if constexpr (Dual) {
                    const float v = sourceV[x];
                    valueV += double(weights[i]) * v;
                    minimumV = std::min(minimumV, v);
                    maximumV = std::max(maximumV, v);
                }
            }
            horizontalU.row(slot)[ox] = static_cast<float>(valueU);
            horizontalMinU.row(slot)[ox] = minimumU;
            horizontalMaxU.row(slot)[ox] = maximumU;
            if constexpr (Dual) {
                horizontalV.row(slot)[ox] = static_cast<float>(valueV);
                horizontalMinV.row(slot)[ox] = minimumV;
                horizontalMaxV.row(slot)[ox] = maximumV;
            }
        }
        cachedSourceRows[slot] = sourceY;
        ++horizontalRowsComputed;
        if (metrics) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - horizontalStart).count();
            horizontalNs += static_cast<uint64_t>(elapsed);
        }
    };

    const bool clampHull = d.arMargin >= 0.0;
    const float margin = float(std::max(0.0, d.arMargin));
    Plane sinkRowU, sinkRowV;
    if (rowSink) {
        sinkRowU = Plane(outputWidth, 1);
        if constexpr (Dual)
            sinkRowV = Plane(outputWidth, 1);
    }
    for (int oy = 0; oy < outputHeight; ++oy) {
        float *outputRowU = rowSink ? sinkRowU.row(0) : outU->row(oy);
        float *outputRowV = nullptr;
        if constexpr (Dual)
            outputRowV = rowSink ? sinkRowV.row(0) : outV->row(oy);
        const int start = ay.start[oy];
        const float *weights = &ay.w[size_t(oy) * ay.sup];
        const float *activity = &ay.am[size_t(oy) * ay.sup];
        for (int j = 0; j < ay.sup; ++j) {
            if (activity[j] == 0.0f)
                continue;
            const int sourceY = std::clamp(start + j, 0, U.h - 1);
            const int slot = sourceY % horizontalRows;
            if (cachedSourceRows[slot] != sourceY)
                computeHorizontalRow(sourceY);
        }
        int ox = 0;
        size_t hullIndex = hull && workset
            ? workset->outputIndexRowOffsets[oy] : 0;
        const size_t hullEnd = hull && workset
            ? workset->outputIndexRowOffsets[oy + 1] : 0;
        auto activeHullX = [&]() {
            return int(workset->outputIndices[hullIndex] % unsigned(outputWidth));
        };
#ifdef __AVX2__
        if constexpr (Backend::lanes == 8) {
            const __m256 rowSum = _mm256_set1_ps(sumY[oy]);
            const __m256 rowAbs = _mm256_set1_ps(sumAbsY[oy]);
            const __m256 epsilon = _mm256_set1_ps(1e-6f);
            const __m256 cancellation = _mm256_set1_ps(0.05f);
            const __m256 marginVec = _mm256_set1_ps(margin);
            for (; ox + 8 <= outputWidth; ox += 8) {
                const __m256 weightSum = _mm256_mul_ps(
                    _mm256_loadu_ps(sumX.data() + ox), rowSum);
                const __m256 absoluteSum = _mm256_mul_ps(
                    _mm256_loadu_ps(sumAbsX.data() + ox), rowAbs);
                const __m256 valid = _mm256_and_ps(
                    _mm256_cmp_ps(weightSum, epsilon, _CMP_GT_OQ),
                    _mm256_cmp_ps(weightSum,
                                  _mm256_mul_ps(cancellation, absoluteSum), _CMP_GE_OQ));
                if (_mm256_movemask_ps(valid) != 0xff)
                    break;

                __m256 valueU = _mm256_setzero_ps();
                __m256 valueV = _mm256_setzero_ps();
                __m256 minimumU = _mm256_set1_ps(1e30f);
                __m256 maximumU = _mm256_set1_ps(-1e30f);
                __m256 minimumV = _mm256_set1_ps(1e30f);
                __m256 maximumV = _mm256_set1_ps(-1e30f);
                for (int j = 0; j < ay.sup; ++j) {
                    if (activity[j] == 0.0f)
                        continue;
                    const int y = std::clamp(start + j, 0, U.h - 1);
                    const int slot = y % horizontalRows;
                    const __m256 weight = _mm256_set1_ps(weights[j]);
                    valueU = _mm256_fmadd_ps(
                        weight, _mm256_loadu_ps(horizontalU.row(slot) + ox), valueU);
                    minimumU = _mm256_min_ps(
                        minimumU, _mm256_loadu_ps(horizontalMinU.row(slot) + ox));
                    maximumU = _mm256_max_ps(
                        maximumU, _mm256_loadu_ps(horizontalMaxU.row(slot) + ox));
                    if constexpr (Dual) {
                        valueV = _mm256_fmadd_ps(
                            weight, _mm256_loadu_ps(horizontalV.row(slot) + ox), valueV);
                        minimumV = _mm256_min_ps(
                            minimumV, _mm256_loadu_ps(horizontalMinV.row(slot) + ox));
                        maximumV = _mm256_max_ps(
                            maximumV, _mm256_loadu_ps(horizontalMaxV.row(slot) + ox));
                    }
                }
                if (hullIndex < hullEnd && activeHullX() < ox + 8) {
                    alignas(32) float minU[8], maxU[8], minV[8], maxV[8];
                    _mm256_store_ps(minU, minimumU);
                    _mm256_store_ps(maxU, maximumU);
                    if constexpr (Dual) {
                        _mm256_store_ps(minV, minimumV);
                        _mm256_store_ps(maxV, maximumV);
                    }
                    while (hullIndex < hullEnd && activeHullX() < ox + 8) {
                        const int lane = activeHullX() - ox;
                        hull->minimumU[hullIndex] = minU[lane];
                        hull->maximumU[hullIndex] = maxU[lane];
                        if constexpr (Dual) {
                            hull->minimumV[hullIndex] = minV[lane];
                            hull->maximumV[hullIndex] = maxV[lane];
                        }
                        ++hullIndex;
                    }
                }
                valueU = _mm256_div_ps(valueU, weightSum);
                if (clampHull) {
                    valueU = _mm256_max_ps(_mm256_sub_ps(minimumU, marginVec),
                                           _mm256_min_ps(valueU,
                                               _mm256_add_ps(maximumU, marginVec)));
                }
                _mm256_storeu_ps(outputRowU + ox, valueU);
                if constexpr (Dual) {
                    valueV = _mm256_div_ps(valueV, weightSum);
                    if (clampHull) {
                        valueV = _mm256_max_ps(_mm256_sub_ps(minimumV, marginVec),
                                               _mm256_min_ps(valueV,
                                                   _mm256_add_ps(maximumV, marginVec)));
                    }
                    _mm256_storeu_ps(outputRowV + ox, valueV);
                }
            }
        }
#endif
        for (; ox < outputWidth; ++ox) {
            double valueU = 0.0, valueV = 0.0;
            float minimumU = 1e30f, maximumU = -1e30f;
            float minimumV = 1e30f, maximumV = -1e30f;
            for (int j = 0; j < ay.sup; ++j) {
                if (activity[j] == 0.0f)
                    continue;
                const int y = std::clamp(start + j, 0, U.h - 1);
                const int slot = y % horizontalRows;
                valueU += double(weights[j]) * horizontalU.row(slot)[ox];
                minimumU = std::min(minimumU, horizontalMinU.row(slot)[ox]);
                maximumU = std::max(maximumU, horizontalMaxU.row(slot)[ox]);
                if constexpr (Dual) {
                    valueV += double(weights[j]) * horizontalV.row(slot)[ox];
                    minimumV = std::min(minimumV, horizontalMinV.row(slot)[ox]);
                    maximumV = std::max(maximumV, horizontalMaxV.row(slot)[ox]);
                }
            }
            const double weightSum = double(sumX[ox]) * sumY[oy];
            const double absoluteSum = double(sumAbsX[ox]) * sumAbsY[oy];
            if (weightSum > 1e-6 && weightSum >= 0.05 * absoluteSum) {
                valueU /= weightSum;
                if constexpr (Dual)
                    valueV /= weightSum;
            } else {
                valueU = bilinear(U, ax.pos[ox], ay.pos[oy]);
                if constexpr (Dual)
                    valueV = bilinear(V, ax.pos[ox], ay.pos[oy]);
            }
            if (clampHull) {
                valueU = std::clamp(valueU, double(minimumU - margin),
                                    double(maximumU + margin));
                if constexpr (Dual)
                    valueV = std::clamp(valueV, double(minimumV - margin),
                                    double(maximumV + margin));
            }
            if (hullIndex < hullEnd && activeHullX() == ox) {
                hull->minimumU[hullIndex] = minimumU;
                hull->maximumU[hullIndex] = maximumU;
                if constexpr (Dual) {
                    hull->minimumV[hullIndex] = minimumV;
                    hull->maximumV[hullIndex] = maximumV;
                }
                ++hullIndex;
            }
            outputRowU[ox] = static_cast<float>(valueU);
            if constexpr (Dual)
                outputRowV[ox] = static_cast<float>(valueV);
        }
        if (rowSink)
            rowSink(rowSinkContext, oy, outputRowU);
    }

    if (metrics) {
        const auto totalNs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - functionStart).count());
        metrics->add(CpuProfileSlot::PlainHorizontal, horizontalNs);
        metrics->add(CpuProfileSlot::PlainVertical,
                     totalNs > horizontalNs ? totalNs - horizontalNs : 0);
        const uint64_t pixels = uint64_t(outputWidth) * outputHeight;
        uint64_t activeHorizontal = 0, activeVertical = 0;
        for (uint16_t count : ax.activeTaps)
            activeHorizontal += count;
        for (uint16_t count : ay.activeTaps)
            activeVertical += count;
        const uint64_t taps = horizontalRowsComputed * activeHorizontal +
                              uint64_t(outputWidth) * activeVertical;
        metrics->outputPixels += pixels;
        metrics->tapsVisited += taps;
        metrics->addWork(Stage::BuildBaseChroma, pixels, taps);
    }
}

} // namespace

// Plain and guided are explicit paths, but share the exact same cached axis
// weights, signed-kernel normalization, boundary extension, and hull clamp.
// Consequently strength=0 remains continuous with strength approaching zero.
void plainChroma(const LGCRData *d, const Plane &cb, const Plane &cr,
                        const Plane &y, const GuideMaps &gm,
                        int sw, int sh, int cw, int ch,
                        Plane &cOutU, Plane &cOutV,
                        PipelineMetrics *metrics,
                        CompressedChromaHull *hull,
                        const SparseWorkset *workset) {
    (void)y;
    (void)gm;
    const double rw = double(sw) / cw, rh = double(sh) / ch;
    const auto ax = cachedChromaAxis(d, sw, cOutU.w, rw, d->shiftX);
    const auto ay = cachedChromaAxis(d, sh, cOutU.h, rh, d->shiftY);
    if (d->radial) {
        const auto radialWeights = cachedRadialWeights(d, *ax, *ay, cb.w, cb.h);
        plainChromaImpl<NativeBackend, true>(*d, cb, cr, cOutU, cOutV,
                                             *ax, *ay, radialWeights.get(), metrics);
    } else if (d->kernel == Kernel::Bilinear)
        plainChromaBilinear<NativeBackend, true>(
            cb, cr, &cOutU, &cOutV, cOutU.w, cOutU.h, *ax, *ay, metrics);
    else
        plainChromaSeparable<NativeBackend, true>(
            *d, cb, cr, &cOutU, &cOutV, cOutU.w, cOutU.h,
            *ax, *ay, metrics, hull, workset);
}

void plainPlane(const LGCRData *d, const Plane &src,
                int sw, int sh, int cw, int ch, Plane &dst) {
    const double rw = double(sw) / cw, rh = double(sh) / ch;
    const auto ax = cachedChromaAxis(d, sw, dst.w, rw, d->shiftX);
    const auto ay = cachedChromaAxis(d, sh, dst.h, rh, d->shiftY);
    if (d->radial) {
        const auto radialWeights = cachedRadialWeights(d, *ax, *ay, src.w, src.h);
        plainChromaImpl<NativeBackend, false>(*d, src, src, dst, dst,
                                              *ax, *ay, radialWeights.get(), nullptr);
    } else if (d->kernel == Kernel::Bilinear)
        plainChromaBilinear<NativeBackend, false>(
            src, src, &dst, &dst, dst.w, dst.h, *ax, *ay, nullptr);
    else
        plainChromaSeparable<NativeBackend, false>(
            *d, src, src, &dst, &dst, dst.w, dst.h,
            *ax, *ay, nullptr, nullptr, nullptr);
}

void plainPlaneRows(const LGCRData *d, const Plane &src,
                    int sw, int sh, int cw, int ch,
                    PlainPlaneRowSink sink, void *context) {
    const double rw = double(sw) / cw, rh = double(sh) / ch;
    const auto ax = cachedChromaAxis(d, sw, sw, rw, d->shiftX);
    const auto ay = cachedChromaAxis(d, sh, sh, rh, d->shiftY);
    if (d->kernel == Kernel::Bilinear)
        plainChromaBilinear<NativeBackend, false>(
            src, src, nullptr, nullptr, sw, sh, *ax, *ay, nullptr,
            sink, context);
    else
        plainChromaSeparable<NativeBackend, false>(
            *d, src, src, nullptr, nullptr, sw, sh,
            *ax, *ay, nullptr, nullptr, nullptr, sink, context);
}

// LGF coefficient planes for the internal algo=4 selector branch.


LGFMaps buildLGFMaps(const LGCRData *d, const Plane &y, const Plane &cb,
                     const Plane &cr, int cw, int ch, double rw, double rh,
                     PipelineMetrics *metrics, const uint8_t *roiMask,
                     int roiStride, FrameScratchAllocator *scratch) {
    LGFMaps m;
    m.aU = scratchPlane(scratch, cw, ch);
    m.bU = scratchPlane(scratch, cw, ch);
    m.confU = scratchPlane(scratch, cw, ch);
    m.aV = scratchPlane(scratch, cw, ch);
    m.bV = scratchPlane(scratch, cw, ch);
    m.confV = scratchPlane(scratch, cw, ch);
    if (roiMask) {
        m.aU.fill(0.0f);
        m.bU.fill(0.0f);
        m.confU.fill(0.0f);
        m.aV.fill(0.0f);
        m.bV.fill(0.0f);
        m.confV.fill(0.0f);
    }
    // buildLGFPair lives in algos.cpp so both chroma planes share luma samples
    // and a single rolling-window traversal.
    {
        ScopedCpuTimer timer(metrics, CpuProfileSlot::LGFMoments);
        buildLGFPair(y, cw, ch, rw, rh, d->shiftX, d->shiftY, cb, cr, 2,
                     d->reg * d->reg, m.aU, m.bU, m.confU,
                     m.aV, m.bV, m.confV, d->cedge, roiMask, roiStride);
    }
    return m;
}

} // namespace lgcr
