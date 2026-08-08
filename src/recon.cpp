#include "lgcr.h"

namespace lgcr {

// guided chroma reconstruction + plain base

void reconstructChroma(const ChromaJob &job) {
    const LGCRData &d = *job.d;
    const Plane &U = *job.srcU;
    const Plane &V = *job.srcV;
    const GuideMaps &gm = *job.gm;
    Plane &outU = *job.dstU;
    Plane &outV = *job.dstV;

    const ChromaAxis ax = buildChromaAxis(job.srcLumaW, outU.w, job.rw, job.shiftX, &d);
    const ChromaAxis ay = buildChromaAxis(job.srcLumaH, outU.h, job.rh, job.shiftY, &d);

    const bool guided = d.strength > 0.0;
    // Guide in SOURCE space: L0 and the structure probes use the sharp source
    // luma at the mapped (fractional) position. An output-space guide was
    // tried and REJECTED on the battery: after resampling, the luma edge is
    // itself smeared, and discriminating source taps against a smeared
    // reference weakened the snap (upscale2x 0.032 -> 0.044). At same size
    // the two are identical anyway.
    const Plane &oY = *job.srcY;
    const BilinAxis bx = guided ? buildBilinAxis(ax.lpos, oY.w) : BilinAxis();
    const BilinAxis by = guided ? buildBilinAxis(ay.lpos, oY.h) : BilinAxis();
    // probe distance: one chroma tap spacing, in source luma px
    const float spOut = float(std::max(job.rw, job.rh));

    const float invGsigma2 = 1.0f / float(d.gsigma * d.gsigma);
    const float strength = float(d.strength);
    const bool radial = d.radial;
    const int lutN = radial ? int(d.lut.size()) : 0;
    const float *lut = radial ? d.lut.data() : nullptr;
    // per-axis downscale AA factors for radial base weights (chroma units)
    const float invWxC = float(std::min(1.0, double(outU.w) / U.w));
    const float invWyC = float(std::min(1.0, double(outU.h) / U.h));
    const bool clampHull = d.arMargin >= 0.0;
    const float margin = float(std::max(0.0, d.arMargin));
    // SIMD path requires the full padded window to be inside the plane
    const bool simdOK = U.w >= ax.sup;

    for (int oy = 0; oy < outU.h; ++oy) {
        const int ty0 = ay.start[oy];
        const float *wyp = &ay.w[size_t(oy) * ay.sup];
        const float scy = ay.pos[oy];

        float *dstRowU = outU.row(oy);
        float *dstRowV = outV.row(oy);

        for (int ox = 0; ox < outU.w; ++ox) {
            const int tx0 = ax.start[ox];
            const float *wxp = &ax.w[size_t(ox) * ax.sup];
            const float scx = ax.pos[ox];

            const bool algo2 = d.algo >= 2;
            float L0 = 0.0f, nxv = 0.0f, nyv = 0.0f, minDiffN = 0.0f, dbC = 0.0f;
            float coherence = 0.0f, ssRamp = 0.0f;
            bool hasDir = false;
            float ridgeRev = 0.0f; // sign-reversal depth along the gradient
            // Subpixel phase: distance to nearest source sample, [0, 0.5]
            float phx = 0.0f, phy = 0.0f;
            // Sparse early-out: plain kernel is provably sufficient here.
            // Mask is at source luma res; map output coords back.
            if (job.mask && !job.mask[size_t(std::min(job.maskH - 1, oy * job.maskH / outU.h)) * job.maskW
                                      + std::min(job.maskW - 1, ox * job.maskW / outU.w)]) {
                dstRowU[ox] = job.plainU->row(oy)[ox];
                dstRowV[ox] = job.plainV->row(oy)[ox];
                continue;
            }
            if (guided) {
                // Guide values at the reconstruction center, from the source
                // luma at the exact (fractional) mapped position — never from
                // chroma-res averages, which blend across the very edges we
                // are trying to snap to.
                const int bxi = bx.i0[ox], byi = by.i0[oy];
                const float bxf = bx.f[ox], byf = by.f[oy];
                L0 = bilinearFast(oY, bxi, bxf, byi, byf);
                const float jxx = bilinearFast(gm.jxx, bxi, bxf, byi, byf);
                const float jxy = bilinearFast(gm.jxy, bxi, bxf, byi, byf);
                const float jyy = bilinearFast(gm.jyy, bxi, bxf, byi, byf);
                const float jdiff = jxx - jyy, jsum = jxx + jyy;
                // coherence in [0,1]: 1 = clean single orientation
                coherence = std::sqrt(jdiff * jdiff + 4.0f * jxy * jxy) / (jsum + 1e-12f);
                // dominant eigenvector = edge normal (sign-free: all uses are
                // quadratic or symmetric probes)
                const float theta = 0.5f * std::atan2(2.0f * jxy, jdiff);
                nxv = std::cos(theta);
                nyv = std::sin(theta);
                hasDir = jsum > 1e-6f;

                if (!algo2) {
                    // v1.2: slope floor from the precomputed db map
                    dbC = bilinearFast(gm.db, bxi, bxf, byi, byf);
                }

                // Step/ramp discriminator: min |diff| to the two luma
                // samples one chroma spacing away along the gradient normal.
                // On a step (any orientation) one of them is same-side (~0);
                // on a ramp both equal the per-tap slope.
                if (algo2 && hasDir) {
                    const float lcx1 = ax.lpos[ox], lcy1 = ay.lpos[oy];
                    const float yA = bilinear(oY, lcx1 - spOut * nxv, lcy1 - spOut * nyv);
                    const float yB = bilinear(oY, lcx1 + spOut * nxv, lcy1 + spOut * nyv);
                    minDiffN = std::min(std::fabs(yA - L0), std::fabs(yB - L0));
                }

                // Ridge detection: probe the luma profile along the gradient
                // normal at -2d,-d,0,+d,+2d. A step or ramp is monotonic; a
                // thin line (line art) produces a sign reversal. Guidance is
                // faded out on thin lines: there the true chroma is a soft
                // blend, and "luma has an edge" does not imply "chroma has
                // an edge".
                if (algo2 && d.ridge && hasDir) {
                    const float dl = 1.5f * spOut; // span +/-3 chroma-tap spacings
                    const float lcx2 = ax.lpos[ox], lcy2 = ay.lpos[oy];
                    const float ym2 = bilinear(oY, lcx2 - 2 * dl * nxv, lcy2 - 2 * dl * nyv);
                    const float ym1 = bilinear(oY, lcx2 - dl * nxv, lcy2 - dl * nyv);
                    const float yp1 = bilinear(oY, lcx2 + dl * nxv, lcy2 + dl * nyv);
                    const float yp2 = bilinear(oY, lcx2 + 2 * dl * nxv, lcy2 + 2 * dl * nyv);
                    const float dd[4] = { ym1 - ym2, L0 - ym1, yp1 - L0, yp2 - yp1 };
                    float rev = 0.0f;
                    for (int q = 0; q < 3; ++q) {
                        const float a = dd[q], b = dd[q + 1];
                        if (a > 0.0f && b < 0.0f) rev = std::max(rev, std::min(a, -b));
                        if (a < 0.0f && b > 0.0f) rev = std::max(rev, std::min(-a, b));
                    }
                    ridgeRev = rev;
                }

                const float fx = scx - std::floor(scx);
                const float fy = scy - std::floor(scy);
                phx = std::min(fx, 1.0f - fx);
                phy = std::min(fy, 1.0f - fy);
            }

            // Pass 1 (guided only): window luma range -> adaptive sigma;
            // chroma hull of both planes for anti-ringing.
            float maxAbsDL = 0.0f;
            float hullMinU = 1e30f, hullMaxU = -1e30f;
            float hullMinV = 1e30f, hullMaxV = -1e30f;
            {
                const float *amRow = &ay.am[size_t(oy) * ay.sup];
                const float *amCol = &ax.am[size_t(ox) * ax.sup];
                for (int j = 0; j < ay.sup; ++j) {
                    if (amRow[j] == 0.0f)
                        continue; // SIMD-padding row: never part of the logical window
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
                        for (; i + 8 <= ax.sup; i += 8) {
                            const int tx = tx0 + i;
                            const __m256 vM = _mm256_loadu_ps(amCol + i);
                            const __m256 vActive = _mm256_cmp_ps(vM, vZero, _CMP_GT_OQ);
                            if (guided) {
                                const __m256 dL = _mm256_sub_ps(_mm256_loadu_ps(lcRow + tx), vL0);
                                vMaxD = _mm256_max_ps(vMaxD,
                                    _mm256_and_ps(_mm256_mul_ps(dL, vM), vAbs));
                            }
                            const __m256 vU = _mm256_loadu_ps(uRow + tx);
                            const __m256 vV = _mm256_loadu_ps(vRow + tx);
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
            float sigLoc;
            if (algo2) {
                // sigma multiplier: sratio on hard steps, sdb (much larger,
                // kills discrimination) on ramps — smoothly blended
                const float t2 = std::clamp(minDiffN / (0.1f * maxAbsDL + 1e-6f), 0.0f, 1.0f);
                ssRamp = t2 * t2 * (3.0f - 2.0f * t2); // smoothstep; 0 = hard step, 1 = ramp
                const float sigMult = float(d.sratio) + (float(d.sdb) - float(d.sratio)) * ssRamp;
                sigLoc = std::max(float(d.sigma), sigMult * maxAbsDL);
            } else {
                // v1.2: three-way floor with the median-neighbor slope baseline
                sigLoc = std::max(std::max(float(d.sigma), float(d.sratio) * maxAbsDL),
                                  float(d.sdb) * dbC * float(std::max(job.rw, job.rh)));
            }
            const float invSigma2 = 1.0f / (sigLoc * sigLoc);
            // Ridge fade: 1 = full guidance, 0 = plain kernel (thin line)
            const float ridgeFade = algo2
                ? 1.0f - std::clamp(2.0f * ridgeRev / (maxAbsDL + 1e-6f), 0.0f, 1.0f)
                : 1.0f;

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
            // strength scales the sim discrimination itself: lambda -> 0
            // must degrade continuously to the plain kernel
            const float guideFade = ridgeFade * cedgeFade * strength;

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
                const float pg = (1.0f - 2.0f * phx) * (1.0f - 2.0f * phy);
                rescue = guideFade * pg * pg * (1.0f - sim0); // guideFade carries strength
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
                                const float w = lutLookup(lut, lutN,
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

                    for (int i = 0; i + 8 <= ax.sup; i += 8) {
                        const int tx = tx0 + i; // guaranteed in-range by start clamping
                        const __m256 vAm = _mm256_loadu_ps(amCol2 + i);
                        const __m256 vU = _mm256_loadu_ps(uRow + tx);
                        const __m256 vV = _mm256_loadu_ps(vRow + tx);
                        __m256 vWx;
                        if (radial) {
                            // 2D radial base weight from the jinc LUT
                            const __m256 vDxc = _mm256_mul_ps(
                                _mm256_add_ps(vDxBase, _mm256_set1_ps(float(i) * float(job.rw))),
                                _mm256_set1_ps(float(1.0 / job.rw) * invWxC));
                            const __m256 dyc = _mm256_set1_ps(dy * float(1.0 / job.rh) * invWyC);
                            const __m256 r2 = _mm256_fmadd_ps(vDxc, vDxc, _mm256_mul_ps(dyc, dyc));
                            vWx = lutLookup8(lut, lutN,
                                _mm256_mul_ps(_mm256_sqrt_ps(r2), _mm256_set1_ps(d.lutScale)));
                        } else {
                            vWx = _mm256_mul_ps(_mm256_loadu_ps(wxp + i), vWy);
                        }

                        __m256 wG;
                        if (guided) {
                            const __m256 vLc = _mm256_loadu_ps(lcRow + tx);
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

                        wG = _mm256_mul_ps(wG, vAm); // mask SIMD-padding taps
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
                        if (radial) {
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

            dstRowU[ox] = static_cast<float>(valueU);
            dstRowV[ox] = static_cast<float>(valueV);

            if (job.selW2) {
                // selector weights: hard-edge-ness times axis/diagonal split.
                // diag = 2|nx*ny|: 0 on axis-aligned edges, 1 at 45 degrees
                const float diag = hasDir ? std::min(1.0f, 2.0f * std::fabs(nxv * nyv)) : 0.0f;
                const float hedgy = (1.0f - ssRamp) * guideFade;
                job.selW2->at(ox, oy) = hedgy * (1.0f - diag);
                job.selW3->at(ox, oy) = hedgy * diag;
            }
        }
    }
}


// Plain (unguided) chroma reconstruction: separable fast path for separable
// kernels, 2D radial path for jinc. Base for algo 3/4/5 and the strength=0
// A/B reference.
void plainChroma(const LGCRData *d, const Plane &cb, const Plane &cr,
                        const Plane &y, const GuideMaps &gm,
                        int sw, int sh, int cw, int ch,
                        Plane &cOutU, Plane &cOutV) {
    const double rw = double(sw) / cw, rh = double(sh) / ch;
    if (!d->radial) {
        // Chroma siting shift in chroma units = -shift/r
        WeightTable th = buildWeights(cw, d->outW, d->kernel, d->kp1, d->kp2,
                                      d->support, -d->shiftX / rw);
        WeightTable tv = buildWeights(ch, d->outH, d->kernel, d->kp1, d->kp2,
                                      d->support, -d->shiftY / rh);
        Plane tmpU(d->outW, ch), tmpV(d->outW, ch);
        resampleH(cb, tmpU, th);
        resampleH(cr, tmpV, th);
        resampleV(tmpU, cOutU, tv);
        resampleV(tmpV, cOutV, tv);
    } else {
        LGCRData d0 = *d;
        d0.strength = 0.0; // plain 2D radial pass
        ChromaJob job;
        job.srcU = &cb; job.srcV = &cr; job.srcY = &y; job.gm = &gm;
        job.dstU = &cOutU; job.dstV = &cOutV;
        job.srcLumaW = sw; job.srcLumaH = sh;
        job.rw = rw; job.rh = rh; job.shiftX = d->shiftX; job.shiftY = 0.0;
        job.d = &d0;
        reconstructChroma(job);
    }
}

// LGF coefficient planes for both chroma planes (algo 3/4)


LGFMaps buildLGFMaps(const LGCRData *d, const Plane &y, const Plane &cb,
                            const Plane &cr, int cw, int ch, double rw, double rh) {
    LGFMaps m(cw, ch);
    buildLGF(y, cw, ch, rw, rh, d->shiftX, d->shiftY, cb, 2, d->reg * d->reg, m.aU, m.bU, m.confU, d->cedge);
    buildLGF(y, cw, ch, rw, rh, d->shiftX, d->shiftY, cr, 2, d->reg * d->reg, m.aV, m.bV, m.confV, d->cedge);
    return m;
}

} // namespace lgcr
