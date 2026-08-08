#include "lgcr.h"

using namespace lgcr;

static const VSFrame *VS_CC sharpenGetFrame(int n, int activationReason, void *instanceData,
                                            void **, VSFrameContext *frameCtx, VSCore *core,
                                            const VSAPI *vsapi) {
    SharpenData *d = static_cast<SharpenData *>(instanceData);
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady)
        return nullptr;

    const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
    const VSVideoFormat *fmt = vsapi->getVideoFrameFormat(src);
    const int sw = vsapi->getFrameWidth(src, 0);
    const int sh = vsapi->getFrameHeight(src, 0);
    const int cw = vsapi->getFrameWidth(src, 1);
    const int ch = vsapi->getFrameHeight(src, 1);
    const bool isFloat = fmt->sampleType == stFloat;
    const double yScale = isFloat ? 1.0 : 1.0 / double((1 << fmt->bitsPerSample) - 1);

    Plane y(sw, sh), cb(cw, ch), cr(cw, ch);
    for (int p = 0; p < 3; ++p) {
        Plane &dst = (p == 0) ? y : (p == 1) ? cb : cr;
        const double off = (p == 0 || isFloat) ? 0.0 : -0.5;
        if (isFloat)
            planeToFloat<float>(vsapi->getReadPtr(src, p), vsapi->getStride(src, p), dst, dst.w, dst.h, 1.0, 0.0);
        else if (fmt->bytesPerSample == 1)
            planeToFloat<uint8_t>(vsapi->getReadPtr(src, p), vsapi->getStride(src, p), dst, dst.w, dst.h, yScale, off);
        else
            planeToFloat<uint16_t>(vsapi->getReadPtr(src, p), vsapi->getStride(src, p), dst, dst.w, dst.h, yScale, off);
    }

    // luma self-guide
    sharpenPlane(y, nullptr, y, 1.0, 1.0, 0.0, 0.0, *d);
    // chroma, luma-guided (footprint luma map at chroma res)
    if (cw != sw || ch != sh) {
        const double rw = double(sw) / cw, rh = double(sh) / ch;
        // shiftX=-0.5: MPEG-2 left siting when subsampled
        GuideMaps gm = buildGuideMaps(y, y, cw, ch, rw, rh, -0.5, 0.0, false);
        sharpenPlane(cb, &gm.lc, y, rw, rh, -0.5, 0.0, *d);
        sharpenPlane(cr, &gm.lc, y, rw, rh, -0.5, 0.0, *d);
    } else {
        sharpenPlane(cb, nullptr, cb, 1.0, 1.0, 0.0, 0.0, *d);
        sharpenPlane(cr, nullptr, cr, 1.0, 1.0, 0.0, 0.0, *d);
    }

    VSFrame *dst = vsapi->newVideoFrame(&d->vi->format, sw, sh, src, core);
    for (int p = 0; p < 3; ++p) {
        Plane &s = (p == 0) ? y : (p == 1) ? cb : cr;
        const int pw = vsapi->getFrameWidth(dst, p), ph = vsapi->getFrameHeight(dst, p);
        const double outScale = isFloat ? 1.0 : double((1 << fmt->bitsPerSample) - 1);
        const double outOff = (p == 0 || isFloat) ? 0.0 : 0.5 * outScale;
        if (isFloat)
            floatToPlane<float>(s, vsapi->getWritePtr(dst, p), vsapi->getStride(dst, p), pw, ph, 1.0, 0.0, -1e30f, 1e30f);
        else if (fmt->bytesPerSample == 1)
            floatToPlane<uint8_t>(s, vsapi->getWritePtr(dst, p), vsapi->getStride(dst, p), pw, ph, outScale, outOff, 0, 255);
        else
            floatToPlane<uint16_t>(s, vsapi->getWritePtr(dst, p), vsapi->getStride(dst, p), pw, ph, outScale, outOff,
                                   0, uint16_t((1 << fmt->bitsPerSample) - 1));
    }
    vsapi->freeFrame(src);
    return dst;
}

static void VS_CC sharpenFree(void *instanceData, VSCore *, const VSAPI *vsapi) {
    SharpenData *d = static_cast<SharpenData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

static void VS_CC sharpenCreate(const VSMap *in, VSMap *out, void *, VSCore *core,
                                const VSAPI *vsapi) {
    auto d = std::make_unique<SharpenData>();
    int err = 0;
    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);
    const VSVideoFormat *fmt = &d->vi->format;
    if (fmt->colorFamily != cfYUV || (fmt->sampleType == stFloat && fmt->bitsPerSample != 32)) {
        vsapi->mapSetError(out, "LGCR Sharpen: planar YUV only");
        vsapi->freeNode(d->node);
        return;
    }
    d->alpha = vsapi->mapGetFloatSaturated(in, "alpha", 0, &err); if (err) d->alpha = 0.3;
    d->sigma = vsapi->mapGetFloatSaturated(in, "sigma", 0, &err); if (err) d->sigma = 0.01;
    d->sratio = vsapi->mapGetFloatSaturated(in, "sratio", 0, &err); if (err) d->sratio = 0.15;
    d->gspatial = vsapi->mapGetFloatSaturated(in, "gspatial", 0, &err); if (err) d->gspatial = 1.2;
    d->arMargin = vsapi->mapGetFloatSaturated(in, "ar", 0, &err); if (err) d->arMargin = 0.0;
    if (d->sigma <= 0.0 || d->sratio <= 0.0 || d->gspatial <= 0.0) {
        vsapi->mapSetError(out, "LGCR Sharpen: sigma/sratio/gspatial must be > 0");
        vsapi->freeNode(d->node);
        return;
    }

    SharpenData *data = d.release();
    VSFilterDependency deps[] = { { data->node, rpGeneral } };
    vsapi->createVideoFilter(out, "Sharpen", data->vi, sharpenGetFrame, sharpenFree,
                             fmParallelRequests, deps, 1, data, core);
}


// ---------------------------------------------------------------------------
// VapourSynth glue
// ---------------------------------------------------------------------------

static const VSFrame *VS_CC lgcrGetFrame(int n, int activationReason, void *instanceData,
                                         void **, VSFrameContext *frameCtx, VSCore *core,
                                         const VSAPI *vsapi) {
    LGCRData *d = static_cast<LGCRData *>(instanceData);
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady)
        return nullptr;

    const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
    const VSVideoFormat *fmt = vsapi->getVideoFrameFormat(src);

    // H.273 chroma siting from frame props, unless loc= was given explicitly.
    // 0=left 1=center 2=topleft 3=top 4=bottomleft 5=bottom
    // Only applies on the subsampled axis: on 4:4:4 input the prop is
    // meaningless and must not re-introduce a shift (it made 444->444
    // non-identity even at strength=0).
    LGCRData dloc = *d;
    if (!dloc.locSet) {
        int perr = 0;
        const int64_t cl = vsapi->mapGetInt(vsapi->getFramePropertiesRO(src),
                                            "_ChromaLocation", 0, &perr);
        if (!perr) {
            if (fmt->subSamplingW > 0)
                dloc.shiftX = (cl == 0 || cl == 2 || cl == 4) ? -0.5 : 0.0;
            if (fmt->subSamplingH > 0)
                dloc.shiftY = (cl == 2 || cl == 3) ? -0.5 : (cl == 4 || cl == 5) ? 0.5 : 0.0;
        }
    }
    d = &dloc;

    const int sw = vsapi->getFrameWidth(src, 0);
    const int sh = vsapi->getFrameHeight(src, 0);
    const int cw = vsapi->getFrameWidth(src, 1);
    const int ch = vsapi->getFrameHeight(src, 1);

    // Extract planes to normalized float (chroma zero-centered)
    const bool isFloat = fmt->sampleType == stFloat;
    const double yScale = isFloat ? 1.0 : 1.0 / double((1 << fmt->bitsPerSample) - 1);
    const double cOffset = isFloat ? 0.0 : -0.5; // int chroma: 0.5 neutral -> 0

    Plane y(sw, sh), cb(cw, ch), cr(cw, ch);
    if (isFloat) {
        planeToFloat<float>(vsapi->getReadPtr(src, 0), vsapi->getStride(src, 0), y, sw, sh, 1.0, 0.0);
        planeToFloat<float>(vsapi->getReadPtr(src, 1), vsapi->getStride(src, 1), cb, cw, ch, 1.0, 0.0);
        planeToFloat<float>(vsapi->getReadPtr(src, 2), vsapi->getStride(src, 2), cr, cw, ch, 1.0, 0.0);
    } else if (fmt->bytesPerSample == 1) {
        planeToFloat<uint8_t>(vsapi->getReadPtr(src, 0), vsapi->getStride(src, 0), y, sw, sh, yScale, 0.0);
        planeToFloat<uint8_t>(vsapi->getReadPtr(src, 1), vsapi->getStride(src, 1), cb, cw, ch, yScale, cOffset);
        planeToFloat<uint8_t>(vsapi->getReadPtr(src, 2), vsapi->getStride(src, 2), cr, cw, ch, yScale, cOffset);
    } else {
        planeToFloat<uint16_t>(vsapi->getReadPtr(src, 0), vsapi->getStride(src, 0), y, sw, sh, yScale, 0.0);
        planeToFloat<uint16_t>(vsapi->getReadPtr(src, 1), vsapi->getStride(src, 1), cb, cw, ch, yScale, cOffset);
        planeToFloat<uint16_t>(vsapi->getReadPtr(src, 2), vsapi->getStride(src, 2), cr, cw, ch, yScale, cOffset);
    }

    const double rw = double(sw) / cw;
    const double rh = double(sh) / ch;

    // Output frame
    VSFrame *dst = vsapi->newVideoFrame(&d->viOut.format, d->outW, d->outH, src, core);
    // output is 4:4:4: the chroma siting prop no longer applies
    vsapi->mapDeleteKey(vsapi->getFramePropertiesRW(dst), "_ChromaLocation");

    // Luma: same size -> verbatim copy (a same-size jinc pass would LOW-PASS:
    // jinc(1)=0.18 != 0); scaled -> separable kernel or true 2D radial.
    Plane yOut(d->outW, d->outH);
    {
        if (d->outW == sw && d->outH == sh) {
            yOut = y;
        } else if (d->radial) {
            resampleRadial(y, yOut, *d);
        } else {
            WeightTable th = buildWeights(sw, d->outW, d->kernel, d->kp1, d->kp2, d->support, 0.0);
            WeightTable tv = buildWeights(sh, d->outH, d->kernel, d->kp1, d->kp2, d->support, 0.0);
            Plane tmp(d->outW, sh);
            resampleH(y, tmp, th);
            resampleV(tmp, yOut, tv);
        }

        if (isFloat)
            floatToPlane<float>(yOut, vsapi->getWritePtr(dst, 0), vsapi->getStride(dst, 0),
                                d->outW, d->outH, 1.0, 0.0, -1e30f, 1e30f);
        else if (fmt->bytesPerSample == 1)
            floatToPlane<uint8_t>(yOut, vsapi->getWritePtr(dst, 0), vsapi->getStride(dst, 0),
                                  d->outW, d->outH, double((1 << fmt->bitsPerSample) - 1), 0.0, 0, 255);
        else
            floatToPlane<uint16_t>(yOut, vsapi->getWritePtr(dst, 0), vsapi->getStride(dst, 0),
                                   d->outW, d->outH, double((1 << fmt->bitsPerSample) - 1), 0.0,
                                   0, uint16_t((1 << fmt->bitsPerSample) - 1));
    }

    // Guide maps from the SOURCE luma (see recon.cpp for why output-space
    // was tried and rejected); Lc footprint also from the source plane.
    GuideMaps gm;
    if (d->strength > 0.0) {
        gm = buildGuideMaps(y, y, cw, ch, rw, rh, d->shiftX, d->shiftY, d->algo == 1);
        if (d->ms > 0.0 && d->algo >= 2 && d->algo <= 4)
            gm.ms = buildMutualGate(gm.lc, cb, cr, d->sigma);
    }

    // Chroma: guided reconstruction (output 444 grid == output luma grid).
    // Both planes are processed in one pass — guide weights depend only on
    // luma and geometry, so U and V share them.
    Plane cOutU(d->outW, d->outH), cOutV(d->outW, d->outH);
    if (d->algo == 5) {
        // NEDI-lite: plain base, then covariance-adaptive correction.
        // Same-size only (the 4-tap covariance scheme is defined for 2x).
        plainChroma(d, cb, cr, y, gm, sw, sh, cw, ch, cOutU, cOutV);
        if (d->strength > 0.0) {
            const ChromaAxis ax = buildChromaAxis(sw, d->outW, rw, d->shiftX, d);
            const ChromaAxis ay = buildChromaAxis(sh, d->outH, rh, d->shiftY, d);
            const Plane plainU = cOutU, plainV = cOutV;
            nediChroma(cb, cr, cOutU, cOutV, ax, ay, plainU, plainV,
                       float(d->strength), d->reg * d->reg, d->arMargin);
        }
    } else if (d->algo == 4) {
        // Selector: plain base + per-pixel routing between the sim path
        // (axis-aligned hard edges) and the LGF path (diagonal hard edges)
        plainChroma(d, cb, cr, y, gm, sw, sh, cw, ch, cOutU, cOutV);
        if (d->strength > 0.0) {
            LGCRData d4 = *d;
            d4.algo = 2;
            d4.strength = 1.0; // fades still apply; lam applied once in blend
            Plane gU(d->outW, d->outH), gV(d->outW, d->outH);
            Plane w2(d->outW, d->outH), w3(d->outW, d->outH);
            ChromaJob job;
            job.srcU = &cb; job.srcV = &cr; job.srcY = &y; job.gm = &gm;
            job.dstU = &gU; job.dstV = &gV;
            job.srcLumaW = sw; job.srcLumaH = sh;
            job.rw = rw; job.rh = rh; job.shiftX = d->shiftX; job.shiftY = d->shiftY;
            job.d = &d4;
            job.outY = &yOut;
            job.selW2 = &w2;
            job.selW3 = &w3;
            reconstructChroma(job);
            LGFMaps lgf = buildLGFMaps(d, y, cb, cr, cw, ch, rw, rh);
            if (gm.ms.w > 0) // co-edge gate applies to the LGF branch too
                for (int j = 0; j < ch; ++j)
                    for (int i = 0; i < cw; ++i) {
                        lgf.confU.at(i, j) *= gm.ms.at(i, j);
                        lgf.confV.at(i, j) *= gm.ms.at(i, j);
                    }
            const ChromaAxis ax = buildChromaAxis(sw, d->outW, rw, d->shiftX, d);
            const ChromaAxis ay = buildChromaAxis(sh, d->outH, rh, d->shiftY, d);
            blendSelector(cOutU, cOutV, gU, gV, lgf.aU, lgf.bU, lgf.aV, lgf.bV,
                          lgf.confU, lgf.confV, y, ax, ay, w2, w3, cw, ch,
                          float(d->strength));
        }
    } else if (d->algo == 3) {
        // LGF: plain kernel base, then blend with the local linear model
        plainChroma(d, cb, cr, y, gm, sw, sh, cw, ch, cOutU, cOutV);
        if (d->strength > 0.0) {
            LGFMaps lgf = buildLGFMaps(d, y, cb, cr, cw, ch, rw, rh);
            if (gm.ms.w > 0) // co-edge gate applies here too
                for (int j = 0; j < ch; ++j)
                    for (int i = 0; i < cw; ++i) {
                        lgf.confU.at(i, j) *= gm.ms.at(i, j);
                        lgf.confV.at(i, j) *= gm.ms.at(i, j);
                    }
            const ChromaAxis ax = buildChromaAxis(sw, d->outW, rw, d->shiftX, d);
            const ChromaAxis ay = buildChromaAxis(sh, d->outH, rh, d->shiftY, d);
            blendLGF(cOutU, cOutV, lgf.aU, lgf.bU, lgf.aV, lgf.bV,
                     lgf.confU, lgf.confV, y, ax, ay, cw, ch, float(d->strength));
        }
    } else if (d->strength == 0.0) {
        // Pure kernel A/B reference
        plainChroma(d, cb, cr, y, gm, sw, sh, cw, ch, cOutU, cOutV);
    } else {
        // algo 1/2 guided path. Sparse mode: plain kernel everywhere, guided
        // correction only where luma structure makes it worthwhile.
        std::vector<uint8_t> mask;
        if (d->sparse) {
            const int dil = int(std::ceil(d->support * std::max(rw, rh))) + 8;
            mask = buildTrustMask(gm, sw, sh, d->sigma, dil); // source luma res
            plainChroma(d, cb, cr, y, gm, sw, sh, cw, ch, cOutU, cOutV);
        }
        ChromaJob job;
        job.srcU = &cb;
        job.srcV = &cr;
        job.srcY = &y;
        job.outY = &yOut;
        job.gm = &gm;
        job.dstU = &cOutU;
        job.dstV = &cOutV;
        job.srcLumaW = sw;
        job.srcLumaH = sh;
        job.rw = rw;
        job.rh = rh;
        job.shiftX = d->shiftX;
        job.shiftY = d->shiftY;
        job.d = d;
        if (!mask.empty()) {
            job.mask = mask.data();
            job.maskW = sw;
            job.maskH = sh;
            job.plainU = &cOutU;
            job.plainV = &cOutV;
        }
        reconstructChroma(job);
    }
    // Back-projection data consistency: re-downsample the reconstruction and
    // return a fraction of the residual, D_h(C + delta) ~= C_src.
    if (d->bp > 0.0 && d->outW == sw && d->outH == sh && cw * 2 == sw && ch * 2 == sh) {
        backProject(cOutU, cb, float(d->bp));
        backProject(cOutV, cr, float(d->bp));
    }

    for (int p = 0; p < 2; ++p) {
        const Plane &cOut = (p == 0) ? cOutU : cOutV;
        const double outScale = isFloat ? 1.0 : double((1 << fmt->bitsPerSample) - 1);
        const double outOffset = isFloat ? 0.0 : 0.5 * outScale; // zero-centered -> neutral
        if (isFloat)
            floatToPlane<float>(cOut, vsapi->getWritePtr(dst, p + 1), vsapi->getStride(dst, p + 1),
                                d->outW, d->outH, 1.0, 0.0, -1e30f, 1e30f);
        else if (fmt->bytesPerSample == 1)
            floatToPlane<uint8_t>(cOut, vsapi->getWritePtr(dst, p + 1), vsapi->getStride(dst, p + 1),
                                  d->outW, d->outH, outScale, outOffset, 0, 255);
        else
            floatToPlane<uint16_t>(cOut, vsapi->getWritePtr(dst, p + 1), vsapi->getStride(dst, p + 1),
                                   d->outW, d->outH, outScale, outOffset,
                                   0, uint16_t((1 << fmt->bitsPerSample) - 1));
    }

    vsapi->freeFrame(src);
    return dst;
}

// ---------------------------------------------------------------------------
// TRecon: temporal chroma reconstruction (motion-compensated multi-frame)
// ---------------------------------------------------------------------------

static void VS_CC lgcrFree(void *instanceData, VSCore *, const VSAPI *vsapi);

static const VSFrame *VS_CC tReconGetFrame(int n, int activationReason, void *instanceData,
                                           void **, VSFrameContext *frameCtx, VSCore *core,
                                           const VSAPI *vsapi) {
    LGCRData *d = static_cast<LGCRData *>(instanceData);
    const int trad = d->trad;
    if (activationReason == arInitial) {
        const int nLast = d->viIn->numFrames - 1;
        for (int k = -trad; k <= trad; ++k) {
            const int fn = std::clamp(n + k, 0, nLast);
            if (k != 0 && fn == n)
                continue; // clamped to the current frame: no temporal content
            vsapi->requestFrameFilter(fn, d->node, frameCtx);
        }
        return nullptr;
    }
    if (activationReason != arAllFramesReady)
        return nullptr;

    const int nLast = d->viIn->numFrames - 1;
    const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
    const VSVideoFormat *fmt = vsapi->getVideoFrameFormat(src);

    // Per-frame H.273 siting (same rules as Recon; subsampled axes only)
    LGCRData dloc = *d;
    {
        int perr = 0;
        const int64_t cl = vsapi->mapGetInt(vsapi->getFramePropertiesRO(src),
                                            "_ChromaLocation", 0, &perr);
        if (!perr) {
            if (fmt->subSamplingW > 0)
                dloc.shiftX = (cl == 0 || cl == 2 || cl == 4) ? -0.5 : 0.0;
            if (fmt->subSamplingH > 0)
                dloc.shiftY = (cl == 2 || cl == 3) ? -0.5 : (cl == 4 || cl == 5) ? 0.5 : 0.0;
        }
    }
    d = &dloc;
    const int sw = vsapi->getFrameWidth(src, 0);
    const int sh = vsapi->getFrameHeight(src, 0);
    const int cw = vsapi->getFrameWidth(src, 1);
    const int ch = vsapi->getFrameHeight(src, 1);
    const bool isFloat = fmt->sampleType == stFloat;
    const double yScale = isFloat ? 1.0 : 1.0 / double((1 << fmt->bitsPerSample) - 1);
    const double cOffset = isFloat ? 0.0 : -0.5;
    const double rw = double(sw) / cw, rh = double(sh) / ch;

    auto extract = [&](const VSFrame *f, Plane &yp, Plane &cbp, Plane &crp) {
        if (isFloat) {
            planeToFloat<float>(vsapi->getReadPtr(f, 0), vsapi->getStride(f, 0), yp, sw, sh, 1.0, 0.0);
            planeToFloat<float>(vsapi->getReadPtr(f, 1), vsapi->getStride(f, 1), cbp, cw, ch, 1.0, 0.0);
            planeToFloat<float>(vsapi->getReadPtr(f, 2), vsapi->getStride(f, 2), crp, cw, ch, 1.0, 0.0);
        } else if (fmt->bytesPerSample == 1) {
            planeToFloat<uint8_t>(vsapi->getReadPtr(f, 0), vsapi->getStride(f, 0), yp, sw, sh, yScale, 0.0);
            planeToFloat<uint8_t>(vsapi->getReadPtr(f, 1), vsapi->getStride(f, 1), cbp, cw, ch, yScale, cOffset);
            planeToFloat<uint8_t>(vsapi->getReadPtr(f, 2), vsapi->getStride(f, 2), crp, cw, ch, yScale, cOffset);
        } else {
            planeToFloat<uint16_t>(vsapi->getReadPtr(f, 0), vsapi->getStride(f, 0), yp, sw, sh, yScale, 0.0);
            planeToFloat<uint16_t>(vsapi->getReadPtr(f, 1), vsapi->getStride(f, 1), cbp, cw, ch, yScale, cOffset);
            planeToFloat<uint16_t>(vsapi->getReadPtr(f, 2), vsapi->getStride(f, 2), crp, cw, ch, yScale, cOffset);
        }
    };

    Plane y(sw, sh), cb(cw, ch), cr(cw, ch);
    extract(src, y, cb, cr);

    // Neighbor frames: extract + luma ME + footprint luma map
    std::vector<TemporalNbr> nbrs;
    std::vector<std::unique_ptr<Plane>> store; // keep planes alive
    std::vector<std::vector<int16_t>> mvStoreX, mvStoreY;
    std::vector<std::vector<float>> confStore;
    // reserve: nbrs holds POINTERS into these vectors, reallocation would dangle
    nbrs.reserve(2 * trad);
    store.reserve(8 * trad);
    mvStoreX.reserve(2 * trad);
    mvStoreY.reserve(2 * trad);
    confStore.reserve(2 * trad);
    if (d->strength > 0.0) {
        for (int k = -trad; k <= trad; ++k) {
            if (k == 0)
                continue;
            const int fn = std::clamp(n + k, 0, nLast);
            if (fn == n)
                continue; // boundary clamp: a duplicated current frame is not
                          // temporal information (its "gain" was an artifact)
            const VSFrame *nf = vsapi->getFrameFilter(fn, d->node, frameCtx);
            auto ny = std::make_unique<Plane>(sw, sh);
            auto nu = std::make_unique<Plane>(cw, ch);
            auto nv = std::make_unique<Plane>(cw, ch);
            extract(nf, *ny, *nu, *nv);
            vsapi->freeFrame(nf);
            mvStoreX.emplace_back();
            mvStoreY.emplace_back();
            confStore.emplace_back();
            int bw = 0, bh = 0;
            blockMatch(y, *ny, 16, d->tsearch, float(d->tsad),
                       mvStoreX.back(), mvStoreY.back(), confStore.back(), bw, bh);
            auto lc = std::make_unique<Plane>(
                buildLcMap(*ny, cw, ch, rw, rh, d->shiftX, d->shiftY));
            TemporalNbr nb;
            nb.U = nu.get();
            nb.V = nv.get();
            nb.lc = lc.get();
            nb.mvx = &mvStoreX.back();
            nb.mvy = &mvStoreY.back();
            nb.tconf = &confStore.back();
            nb.bw = bw;
            nb.bh = bh;
            nb.block = 16;
            nbrs.push_back(nb);
            store.push_back(std::move(ny));
            store.push_back(std::move(nu));
            store.push_back(std::move(nv));
            store.push_back(std::move(lc));
        }
    }

    VSFrame *dst = vsapi->newVideoFrame(&d->viOut.format, sw, sh, src, core);
    vsapi->mapDeleteKey(vsapi->getFramePropertiesRW(dst), "_ChromaLocation");

    // Y: same size -> verbatim copy
    GuideMaps gm;
    if (d->strength > 0.0) {
        gm = buildGuideMaps(y, y, cw, ch, rw, rh, d->shiftX, d->shiftY, false);
        if (d->ms > 0.0)
            gm.ms = buildMutualGate(gm.lc, cb, cr, d->sigma);
    }
    {
        if (isFloat)
            floatToPlane<float>(y, vsapi->getWritePtr(dst, 0), vsapi->getStride(dst, 0), sw, sh, 1.0, 0.0, -1e30f, 1e30f);
        else if (fmt->bytesPerSample == 1)
            floatToPlane<uint8_t>(y, vsapi->getWritePtr(dst, 0), vsapi->getStride(dst, 0), sw, sh,
                                  double((1 << fmt->bitsPerSample) - 1), 0.0, 0, 255);
        else
            floatToPlane<uint16_t>(y, vsapi->getWritePtr(dst, 0), vsapi->getStride(dst, 0), sw, sh,
                                   double((1 << fmt->bitsPerSample) - 1), 0.0,
                                   0, uint16_t((1 << fmt->bitsPerSample) - 1));
    }

    Plane cOutU(sw, sh), cOutV(sw, sh);
    // Sparse mode (off by default here): plain kernel everywhere, guided
    // correction only near luma structure.
    std::vector<uint8_t> mask;
    if (d->sparse && d->strength > 0.0) {
        const int dil = int(std::ceil(d->support * std::max(rw, rh))) + 8;
        mask = buildTrustMask(gm, sw, sh, d->sigma, dil);
        plainChroma(d, cb, cr, y, gm, sw, sh, cw, ch, cOutU, cOutV);
    }
    {
        ChromaJob job;
        job.srcU = &cb;
        job.srcV = &cr;
        job.srcY = &y;
        job.gm = &gm;
        job.dstU = &cOutU;
        job.dstV = &cOutV;
        job.srcLumaW = sw;
        job.srcLumaH = sh;
        job.rw = rw;
        job.rh = rh;
        job.shiftX = d->shiftX;
        job.shiftY = d->shiftY;
        job.d = d;
        if (!mask.empty()) {
            job.mask = mask.data();
            job.maskW = sw;
            job.maskH = sh;
            job.plainU = &cOutU;
            job.plainV = &cOutV;
        }
        if (!nbrs.empty())
            job.nbrs = &nbrs;
        reconstructChroma(job);
    }

    for (int p = 0; p < 2; ++p) {
        const Plane &cOut = (p == 0) ? cOutU : cOutV;
        const double outScale = isFloat ? 1.0 : double((1 << fmt->bitsPerSample) - 1);
        const double outOffset = isFloat ? 0.0 : 0.5 * outScale;
        if (isFloat)
            floatToPlane<float>(cOut, vsapi->getWritePtr(dst, p + 1), vsapi->getStride(dst, p + 1),
                                sw, sh, 1.0, 0.0, -1e30f, 1e30f);
        else if (fmt->bytesPerSample == 1)
            floatToPlane<uint8_t>(cOut, vsapi->getWritePtr(dst, p + 1), vsapi->getStride(dst, p + 1),
                                  sw, sh, outScale, outOffset, 0, 255);
        else
            floatToPlane<uint16_t>(cOut, vsapi->getWritePtr(dst, p + 1), vsapi->getStride(dst, p + 1),
                                   sw, sh, outScale, outOffset,
                                   0, uint16_t((1 << fmt->bitsPerSample) - 1));
    }

    vsapi->freeFrame(src);
    return dst;
}

static void VS_CC tReconCreate(const VSMap *in, VSMap *out, void *, VSCore *core,
                               const VSAPI *vsapi) {
    auto d = std::make_unique<LGCRData>();
    int err = 0;
    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->viIn = vsapi->getVideoInfo(d->node);
    const VSVideoFormat *fmt = &d->viIn->format;
    if (fmt->colorFamily != cfYUV || (fmt->sampleType == stFloat && fmt->bitsPerSample != 32) ||
        fmt->subSamplingW > 1 || fmt->subSamplingH > 1) {
        vsapi->mapSetError(out, "TRecon: planar YUV 444/422/420, 8-16 bit int or 32 bit float");
        vsapi->freeNode(d->node);
        return;
    }
    if (d->viIn->width == 0 || d->viIn->height == 0 || d->viIn->numFrames == 0) {
        vsapi->mapSetError(out, "TRecon: constant dimensions and frame count required");
        vsapi->freeNode(d->node);
        return;
    }
    d->outW = d->viIn->width;
    d->outH = d->viIn->height;
    d->strength = vsapi->mapGetFloatSaturated(in, "strength", 0, &err); if (err) d->strength = 0.8;
    d->sigma = vsapi->mapGetFloatSaturated(in, "sigma", 0, &err); if (err) d->sigma = 0.01;
    d->sratio = vsapi->mapGetFloatSaturated(in, "sratio", 0, &err); if (err) d->sratio = 0.15;
    d->sdb = vsapi->mapGetFloatSaturated(in, "sdb", 0, &err); if (err) d->sdb = 3.0;
    d->gsigma = vsapi->mapGetFloatSaturated(in, "gsigma", 0, &err); if (err) d->gsigma = 2.5;
    d->stretch = vsapi->mapGetFloatSaturated(in, "stretch", 0, &err); if (err) d->stretch = 1.0;
    d->arMargin = vsapi->mapGetFloatSaturated(in, "ar", 0, &err); if (err) d->arMargin = 0.0;
    d->ms = vsapi->mapGetFloatSaturated(in, "ms", 0, &err); if (err) d->ms = 1.0;
    d->trad = vsapi->mapGetIntSaturated(in, "trad", 0, &err); if (err) d->trad = 1;
    d->tsearch = vsapi->mapGetIntSaturated(in, "tsearch", 0, &err); if (err) d->tsearch = 6;
    d->tsad = vsapi->mapGetFloatSaturated(in, "tsad", 0, &err); if (err) d->tsad = 0.02;
    if (d->strength < 0.0 || d->strength > 1.0 || d->sigma <= 0.0 || d->sratio <= 0.0 ||
        d->sdb <= 0.0 || d->stretch < 0.0 || d->gsigma <= 0.0 || d->ms < 0.0 || d->ms > 1.0 ||
        d->trad < 0 || d->trad > 8 || d->tsearch < 0 || d->tsearch > 64 || d->tsad <= 0.0) {
        vsapi->mapSetError(out, "TRecon: invalid parameter range (need 0<=strength<=1, "
                                "sigma/sratio/sdb/gsigma/tsad>0, stretch>=0, 0<=trad<=8, 0<=tsearch<=64)");
        vsapi->freeNode(d->node);
        return;
    }
    {
        const int64_t ri = vsapi->mapGetIntSaturated(in, "ridge", 0, &err);
        d->ridge = err ? true : (ri != 0);
    }
    // lanczos3 base kernel; sparse off by default (static-region temporal
    // averaging is part of the value)
    d->sparse = false;
    {
        const int64_t sp = vsapi->mapGetIntSaturated(in, "sparse", 0, &err);
        if (!err) d->sparse = (sp != 0);
    }
    d->algo = 2;
    d->taps = 2 * 3;
    d->support = 3.0;
    d->kernel = Kernel::Lanczos;

    d->viOut = *d->viIn;
    if (!vsapi->queryVideoFormat(&d->viOut.format, cfYUV, fmt->sampleType,
                                 fmt->bitsPerSample, 0, 0, core)) {
        vsapi->mapSetError(out, "TRecon: failed to query 4:4:4 output format");
        vsapi->freeNode(d->node);
        return;
    }

    LGCRData *data = d.release();
    VSFilterDependency deps[] = { { data->node, rpGeneral } };
    vsapi->createVideoFilter(out, "TRecon", &data->viOut, tReconGetFrame, lgcrFree,
                             fmParallelRequests, deps, 1, data, core);
}

static void VS_CC lgcrFree(void *instanceData, VSCore *, const VSAPI *vsapi) {
    LGCRData *d = static_cast<LGCRData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

static void VS_CC lgcrCreate(const VSMap *in, VSMap *out, void *, VSCore *core,
                             const VSAPI *vsapi) {
    auto d = std::make_unique<LGCRData>();
    int err = 0;

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->viIn = vsapi->getVideoInfo(d->node);
    const VSVideoFormat *fmt = &d->viIn->format;

    auto fail = [&](const char *msg) {
        vsapi->mapSetError(out, msg);
        vsapi->freeNode(d->node);
    };

    if (fmt->colorFamily != cfYUV || (fmt->sampleType == stFloat && fmt->bitsPerSample != 32)) {
        fail("LGCR: input must be planar YUV (8-16 bit int or 32 bit float)");
        return;
    }
    if (fmt->subSamplingW > 1 || fmt->subSamplingH > 1) {
        fail("LGCR: only 4:4:4 / 4:2:2 / 4:2:0 input is supported");
        return;
    }
    if (d->viIn->width == 0 || d->viIn->height == 0) {
        fail("LGCR: constant input dimensions required");
        return;
    }

    d->outW = vsapi->mapGetIntSaturated(in, "width", 0, &err);
    if (err) d->outW = d->viIn->width;
    d->outH = vsapi->mapGetIntSaturated(in, "height", 0, &err);
    if (err) d->outH = d->viIn->height;
    if (d->outW <= 0 || d->outH <= 0) {
        fail("LGCR: invalid width/height");
        return;
    }

    const char *kname = vsapi->mapGetData(in, "kernel", 0, &err);
    std::string k = err ? "lanczos" : kname;
    if (k == "bilinear") {
        d->kernel = Kernel::Bilinear;
        d->support = 1.0;
    } else if (k == "bicubic") {
        d->kernel = Kernel::Bicubic;
        d->support = 2.0;
        d->kp1 = vsapi->mapGetFloatSaturated(in, "b", 0, &err); if (err) d->kp1 = 0.0;
        d->kp2 = vsapi->mapGetFloatSaturated(in, "c", 0, &err); if (err) d->kp2 = 0.6;
    } else if (k == "lanczos") {
        d->kernel = Kernel::Lanczos;
        d->kp1 = double(vsapi->mapGetIntSaturated(in, "taps", 0, &err)); if (err) d->kp1 = 3.0;
        d->support = d->kp1;
    } else if (k == "spline16") {
        d->kernel = Kernel::Spline16;
        d->support = 2.0;
    } else if (k == "spline36") {
        d->kernel = Kernel::Spline36;
        d->support = 3.0;
    } else if (k == "jinc") {
        d->kernel = Kernel::Jinc;
        d->radial = true;
        d->kp1 = double(vsapi->mapGetIntSaturated(in, "taps", 0, &err)); if (err) d->kp1 = 3.0;
        d->support = d->kp1;
        // Radial profile LUT, 1024 entries per unit distance
        const int n = static_cast<int>(d->support * 1024) + 2;
        d->lut.resize(n);
        for (int i = 0; i < n; ++i)
            d->lut[i] = static_cast<float>(jincWindowed(double(i) / 1024.0, d->support));
        d->lutScale = 1024.0f;
    } else {
        fail("LGCR: kernel must be \"bilinear\", \"bicubic\", \"lanczos\", \"spline16\", \"spline36\" or \"jinc\"");
        return;
    }
    d->taps = 2 * static_cast<int>(std::ceil(d->support));

    d->strength = vsapi->mapGetFloatSaturated(in, "strength", 0, &err); if (err) d->strength = 0.8;
    d->sigma = vsapi->mapGetFloatSaturated(in, "sigma", 0, &err); if (err) d->sigma = 0.01;
    d->sratio = vsapi->mapGetFloatSaturated(in, "sratio", 0, &err); if (err) d->sratio = 0.15;
    {
        const int64_t a = vsapi->mapGetIntSaturated(in, "algo", 0, &err);
        d->algo = err ? 2 : int(a);
    }
    if (d->algo < 1 || d->algo > 5) {
        fail("LGCR: algo must be 1 (v1.2), 2 (v1.3), 3 (LGF), 4 (selector) or 5 (NEDI)");
        return;
    }
    if (d->algo == 5 && (d->outW != d->viIn->width || d->outH != d->viIn->height)) {
        fail("LGCR: algo=5 (NEDI) requires same-size output (no scaling)");
        return;
    }
    d->sdb = vsapi->mapGetFloatSaturated(in, "sdb", 0, &err);
    if (err) d->sdb = (d->algo == 1) ? 1.5 : 3.0;
    d->stretch = vsapi->mapGetFloatSaturated(in, "stretch", 0, &err); if (err) d->stretch = 1.0;
    d->gsigma = vsapi->mapGetFloatSaturated(in, "gsigma", 0, &err); if (err) d->gsigma = 2.5;
    {
        const int64_t ridgeInt = vsapi->mapGetIntSaturated(in, "ridge", 0, &err);
        d->ridge = err ? true : (ridgeInt != 0);
        const int64_t ce = vsapi->mapGetIntSaturated(in, "cedge", 0, &err);
        d->cedge = err ? false : (ce != 0);
    }
    d->ms = vsapi->mapGetFloatSaturated(in, "ms", 0, &err); if (err) d->ms = 1.0;
    d->arMargin = vsapi->mapGetFloatSaturated(in, "ar", 0, &err); if (err) d->arMargin = 0.0;
    d->reg = vsapi->mapGetFloatSaturated(in, "reg", 0, &err); if (err) d->reg = 0.005;

    const char *loc = vsapi->mapGetData(in, "loc", 0, &err);
    if (!err) {
        std::string l = loc;
        if (l == "left") d->shiftX = -0.5;
        else if (l == "center") d->shiftX = 0.0;
        else {
            fail("LGCR: loc must be \"left\" or \"center\"");
            return;
        }
        d->locSet = true;
    } else {
        d->shiftX = -0.5; // MPEG-2 style horizontal siting (may be overridden
    }                     // per-frame by the _ChromaLocation prop)
    {
        const int64_t sp = vsapi->mapGetIntSaturated(in, "sparse", 0, &err);
        d->sparse = err ? true : (sp != 0);
    }
    d->bp = vsapi->mapGetFloatSaturated(in, "bp", 0, &err); if (err) d->bp = 0.0;
    if (fmt->subSamplingW == 0)
        d->shiftX = 0.0;

    if (d->strength < 0.0 || d->strength > 1.0 || d->sigma <= 0.0 || d->sratio <= 0.0 ||
        d->sdb <= 0.0 || d->stretch < 0.0 || d->gsigma <= 0.0 || d->reg <= 0.0 ||
        d->bp < 0.0 || d->bp > 1.0 || d->ms < 0.0 || d->ms > 1.0 ||
        ((d->kernel == Kernel::Lanczos || d->kernel == Kernel::Jinc) && d->kp1 < 1.0)) {
        fail("LGCR: invalid parameter range (need 0<=strength<=1, 0<=bp<=1, 0<=ms<=1, "
             "sigma/sratio/sdb/gsigma/reg>0, stretch>=0, taps>=1)");
        return;
    }

    // Output video info: YUV444, same depth/type
    d->viOut = *d->viIn;
    d->viOut.width = d->outW;
    d->viOut.height = d->outH;
    if (!vsapi->queryVideoFormat(&d->viOut.format, cfYUV, fmt->sampleType,
                                 fmt->bitsPerSample, 0, 0, core)) {
        fail("LGCR: failed to query 4:4:4 output format");
        return;
    }

    LGCRData *data = d.release();
    VSFilterDependency deps[] = { { data->node, rpGeneral } };
    vsapi->createVideoFilter(out, "LGCR", &data->viOut, lgcrGetFrame, lgcrFree,
                             fmParallelRequests, deps, 1, data, core);
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("dev.bsflab.lgcr" LGCR_SUFFIX, "lgcr" LGCR_SUFFIX,
                         "Luma-guided chroma reconstruction with direction-aware kernels",
                         VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction(
        "Recon",
        "clip:vnode;width:int:opt;height:int:opt;kernel:data:opt;taps:int:opt;algo:int:opt;"
        "b:float:opt;c:float:opt;strength:float:opt;sigma:float:opt;sratio:float:opt;"
        "sdb:float:opt;stretch:float:opt;gsigma:float:opt;ridge:int:opt;cedge:int:opt;ar:float:opt;reg:float:opt;loc:data:opt;sparse:int:opt;bp:float:opt;ms:float:opt",
        "clip:vnode", lgcrCreate, nullptr, plugin);
    vspapi->registerFunction(
        "Sharpen",
        "clip:vnode;alpha:float:opt;sigma:float:opt;sratio:float:opt;"
        "gspatial:float:opt;ar:float:opt",
        "clip:vnode", sharpenCreate, nullptr, plugin);
    vspapi->registerFunction(
        "TRecon",
        "clip:vnode;strength:float:opt;sigma:float:opt;sratio:float:opt;sdb:float:opt;"
        "gsigma:float:opt;stretch:float:opt;ar:float:opt;ridge:int:opt;sparse:int:opt;ms:float:opt;"
        "trad:int:opt;tsearch:int:opt;tsad:float:opt",
        "clip:vnode", tReconCreate, nullptr, plugin);
}
