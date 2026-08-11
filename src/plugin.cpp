#include "lgcr.h"

#include <cstdlib>

#ifndef LGCR_VERSION_MAJOR
#define LGCR_VERSION_MAJOR 2
#endif
#ifndef LGCR_VERSION_MINOR
#define LGCR_VERSION_MINOR 1
#endif

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
        GuideMaps gm = buildGuideMaps(y, y, cw, ch, rw, rh, -0.5, 0.0);
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

static bool profilingEnabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("LGCR_PROFILE");
        return value && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

static void writeProfile(VSFrame *frame, const PipelineMetrics &metrics,
                         uint64_t totalNs, const VSAPI *vsapi) {
    VSMap *props = vsapi->getFramePropertiesRW(frame);
    for (size_t i = 0; i < stageCount; ++i) {
        const Stage stage = static_cast<Stage>(i);
        const std::string key = "_LGCR_" + std::string(stageName(stage)) + "_us";
        vsapi->mapSetFloat(props, key.c_str(), metrics.nanoseconds[i] / 1000.0, maReplace);
        const std::string pixelKey = "_LGCR_" + std::string(stageName(stage)) + "_pixels";
        const std::string tapKey = "_LGCR_" + std::string(stageName(stage)) + "_taps";
        vsapi->mapSetInt(props, pixelKey.c_str(), static_cast<int64_t>(metrics.pixels[i]),
                         maReplace);
        vsapi->mapSetInt(props, tapKey.c_str(), static_cast<int64_t>(metrics.taps[i]),
                         maReplace);
    }
    for (size_t i = 0; i < cpuProfileSlotCount; ++i) {
        const auto slot = static_cast<CpuProfileSlot>(i);
        const std::string key = "_LGCR_cpu_" +
            std::string(cpuProfileSlotName(slot)) + "_us";
        vsapi->mapSetFloat(props, key.c_str(),
                           metrics.cpuNanoseconds[i] / 1000.0, maReplace);
    }
    vsapi->mapSetFloat(props, "_LGCR_total_us", totalNs / 1000.0, maReplace);
    vsapi->mapSetInt(props, "_LGCR_output_pixels",
                     static_cast<int64_t>(metrics.outputPixels), maReplace);
    vsapi->mapSetInt(props, "_LGCR_taps_visited",
                     static_cast<int64_t>(metrics.tapsVisited), maReplace);
    const double active = metrics.sparseTotalPixels
        ? double(metrics.sparseActivePixels) / metrics.sparseTotalPixels : 1.0;
    vsapi->mapSetFloat(props, "_LGCR_sparse_active_ratio", active, maReplace);
}

static void applyFrameChromaSiting(LGCRData &d, const VSVideoFormat *fmt,
                                   const VSFrame *frame, const VSAPI *vsapi) {
    if (fmt->subSamplingW == 0)
        d.shiftX = 0.0;
    if (fmt->subSamplingH == 0)
        d.shiftY = 0.0;
    if (d.locSet)
        return;

    int err = 0;
    const int64_t cl = vsapi->mapGetInt(vsapi->getFramePropertiesRO(frame),
                                        "_ChromaLocation", 0, &err);
    if (err)
        return;
    if (fmt->subSamplingW > 0)
        d.shiftX = (cl == 0 || cl == 2 || cl == 4) ? -0.5 : 0.0;
    if (fmt->subSamplingH > 0)
        d.shiftY = (cl == 2 || cl == 3) ? -0.5 : (cl == 4 || cl == 5) ? 0.5 : 0.0;
}

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

    const bool profile = profilingEnabled();
    PipelineMetrics frameMetrics;
    PipelineMetrics *metrics = profile ? &frameMetrics : nullptr;
    const auto frameStart = std::chrono::steady_clock::now();

    const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
    const VSVideoFormat *fmt = vsapi->getVideoFrameFormat(src);

    // H.273 chroma siting from frame props, unless loc= was given explicitly.
    // 0=left 1=center 2=topleft 3=top 4=bottomleft 5=bottom
    // Only applies on the subsampled axis: on 4:4:4 input the prop is
    // meaningless and must not re-introduce a shift (it made 444->444
    // non-identity even at strength=0).
    LGCRData dloc = *d;
    applyFrameChromaSiting(dloc, fmt, src, vsapi);
    d = &dloc;

    const int sw = vsapi->getFrameWidth(src, 0);
    const int sh = vsapi->getFrameHeight(src, 0);
    const int cw = vsapi->getFrameWidth(src, 1);
    const int ch = vsapi->getFrameHeight(src, 1);

    // Extract planes to normalized float (chroma zero-centered)
    const bool isFloat = fmt->sampleType == stFloat;
    const double yScale = isFloat ? 1.0 : 1.0 / double((1 << fmt->bitsPerSample) - 1);
    const double cOffset = isFloat ? 0.0 : -0.5; // int chroma: 0.5 neutral -> 0

    VSCoreInfo coreInfo{};
    vsapi->getCoreInfo(core, &coreInfo);
    FrameWorkspaceLease workspaceLease = d->workspacePool->acquire(coreInfo.numThreads);
    FrameWorkspace &workspace = workspaceLease.get();
    Plane sourceYView, sourceUView, sourceVView;
    Plane *sourceY = &workspace.sourceY;
    Plane *sourceU = &workspace.sourceU;
    Plane *sourceV = &workspace.sourceV;
    {
        ScopedStageTimer timer(metrics, Stage::ConvertInput);
        if (isFloat) {
            sourceYView = Plane::readOnlyView(
                reinterpret_cast<const float *>(vsapi->getReadPtr(src, 0)),
                sw, sh, int(vsapi->getStride(src, 0) / sizeof(float)));
            sourceUView = Plane::readOnlyView(
                reinterpret_cast<const float *>(vsapi->getReadPtr(src, 1)),
                cw, ch, int(vsapi->getStride(src, 1) / sizeof(float)));
            sourceVView = Plane::readOnlyView(
                reinterpret_cast<const float *>(vsapi->getReadPtr(src, 2)),
                cw, ch, int(vsapi->getStride(src, 2) / sizeof(float)));
            sourceY = &sourceYView;
            sourceU = &sourceUView;
            sourceV = &sourceVView;
        } else if (fmt->bytesPerSample == 1) {
            workspace.sourceY.resizeDiscard(sw, sh);
            workspace.sourceU.resizeDiscard(cw, ch);
            workspace.sourceV.resizeDiscard(cw, ch);
            planeToFloat<uint8_t>(vsapi->getReadPtr(src, 0), vsapi->getStride(src, 0), *sourceY, sw, sh, yScale, 0.0);
            planeToFloat<uint8_t>(vsapi->getReadPtr(src, 1), vsapi->getStride(src, 1), *sourceU, cw, ch, yScale, cOffset);
            planeToFloat<uint8_t>(vsapi->getReadPtr(src, 2), vsapi->getStride(src, 2), *sourceV, cw, ch, yScale, cOffset);
        } else {
            workspace.sourceY.resizeDiscard(sw, sh);
            workspace.sourceU.resizeDiscard(cw, ch);
            workspace.sourceV.resizeDiscard(cw, ch);
            planeToFloat<uint16_t>(vsapi->getReadPtr(src, 0), vsapi->getStride(src, 0), *sourceY, sw, sh, yScale, 0.0);
            planeToFloat<uint16_t>(vsapi->getReadPtr(src, 1), vsapi->getStride(src, 1), *sourceU, cw, ch, yScale, cOffset);
            planeToFloat<uint16_t>(vsapi->getReadPtr(src, 2), vsapi->getStride(src, 2), *sourceV, cw, ch, yScale, cOffset);
        }
    }
    if (metrics && !isFloat)
        metrics->addWork(Stage::ConvertInput,
                         uint64_t(sw) * sh + 2 * uint64_t(cw) * ch);
    Plane &y = *sourceY;
    Plane &cb = *sourceU;
    Plane &cr = *sourceV;

    const double rw = double(sw) / cw;
    const double rh = double(sh) / ch;

    // Output frame
    const bool sameSize = d->outW == sw && d->outH == sh;
    const VSFrame *planeSources[3] = { sameSize ? src : nullptr, nullptr, nullptr };
    const int sourcePlanes[3] = { 0, 0, 0 };
    VSFrame *dst = vsapi->newVideoFrame2(
        &d->viOut.format, d->outW, d->outH, planeSources, sourcePlanes, src, core);
    // output is 4:4:4: the chroma siting prop no longer applies
    vsapi->mapDeleteKey(vsapi->getFramePropertiesRW(dst), "_ChromaLocation");

    // Luma: same size -> verbatim copy (a same-size jinc pass would LOW-PASS:
    // jinc(1)=0.18 != 0); scaled -> separable kernel or true 2D radial.
    {
        Plane outputYView;
        Plane *outputY = nullptr;
        {
            ScopedStageTimer timer(metrics, Stage::ResampleLuma);
            if (!sameSize && isFloat) {
                outputYView = Plane::writableView(
                    reinterpret_cast<float *>(vsapi->getWritePtr(dst, 0)),
                    d->outW, d->outH,
                    int(vsapi->getStride(dst, 0) / sizeof(float)));
                outputY = &outputYView;
            } else if (!sameSize) {
                workspace.fullSlot0.resizeDiscard(d->outW, d->outH);
                outputY = &workspace.fullSlot0;
            }
            if (outputY && d->radial) {
                resampleRadial(y, *outputY, *d);
            } else if (outputY) {
                const auto th = cachedWeights(d, sw, d->outW, 0.0);
                const auto tv = cachedWeights(d, sh, d->outH, 0.0);
                workspace.fullSlot1.resizeDiscard(d->outW, sh);
                resampleH(y, workspace.fullSlot1, *th);
                resampleV(workspace.fullSlot1, *outputY, *tv);
            }
        }
        if (metrics && !sameSize)
            metrics->addWork(Stage::ResampleLuma, uint64_t(d->outW) * d->outH);

        if (outputY && !isFloat) {
            ScopedStageTimer timer(metrics, Stage::ConvertOutput);
            if (fmt->bytesPerSample == 1)
                floatToPlane<uint8_t>(*outputY, vsapi->getWritePtr(dst, 0), vsapi->getStride(dst, 0),
                                      d->outW, d->outH, double((1 << fmt->bitsPerSample) - 1), 0.0, 0, 255);
            else
                floatToPlane<uint16_t>(*outputY, vsapi->getWritePtr(dst, 0), vsapi->getStride(dst, 0),
                                       d->outW, d->outH, double((1 << fmt->bitsPerSample) - 1), 0.0,
                                       0, uint16_t((1 << fmt->bitsPerSample) - 1));
            if (metrics)
                metrics->addWork(Stage::ConvertOutput,
                                 uint64_t(d->outW) * d->outH);
        }
    }

    // Guide maps from the SOURCE luma (see recon.cpp for why output-space
    // was tried and rejected); Lc footprint also from the source plane.
    GuideMaps gm;
    SparseWorkset sparseWorkset;
    bool hasSparseWorkset = false;
    std::shared_ptr<const ChromaAxis> frameAxisX, frameAxisY;
    if (d->strength > 0.0) {
        const double trustThreshold = d->algo == 4 ? 0.25 * d->sigma : d->sigma;
        {
            ScopedStageTimer timer(metrics, Stage::BuildGuideMaps);
            gm = buildGuideMaps(y, y, cw, ch, rw, rh, d->shiftX, d->shiftY,
                                metrics, d->sparse ? trustThreshold : -1.0,
                                &workspace.scratch);
        }
        if (metrics)
            metrics->addWork(Stage::BuildGuideMaps,
                             uint64_t(sw) * sh + uint64_t(cw) * ch);
        if (d->sparse) {
            const int dil = int(std::ceil(d->support * std::max(rw, rh))) + 8;
            std::vector<uint8_t> trustMask;
            {
                ScopedStageTimer timer(metrics, Stage::BuildTrustMask);
                trustMask = buildTrustMask(
                    gm, sw, sh, trustThreshold, dil, metrics);
            }
            if (metrics) {
                metrics->addWork(Stage::BuildTrustMask, uint64_t(sw) * sh);
                metrics->sparseTotalPixels += trustMask.size();
                metrics->sparseActivePixels += static_cast<uint64_t>(
                    std::count(trustMask.begin(), trustMask.end(), uint8_t{1}));
            }
            frameAxisX = cachedChromaAxis(d, sw, d->outW, rw, d->shiftX);
            frameAxisY = cachedChromaAxis(d, sh, d->outH, rh, d->shiftY);
            sparseWorkset = buildSparseWorkset(
                std::move(trustMask), sw, sh, d->outW, d->outH,
                *frameAxisX, *frameAxisY, cw, ch);
            hasSparseWorkset = true;
        }
        uint64_t mutualPixels = uint64_t(cw) * ch;
        if (d->ms > 0.0) {
            ScopedStageTimer timer(metrics, Stage::BuildMutualGate);
            if (hasSparseWorkset)
                mutualPixels = sparseWorkset.activeChromaPixels;
            gm.ms = buildMutualGate(gm.lc, cb, cr, d->sigma,
                                    hasSparseWorkset ? sparseWorkset.chromaMask.data() : nullptr,
                                    cw, hasSparseWorkset ? &sparseWorkset : nullptr,
                                    metrics, &workspace.scratch);
        }
        if (metrics && d->ms > 0.0)
            metrics->addWork(Stage::BuildMutualGate, mutualPixels,
                             mutualPixels * 7 * 6);
    }

    // Chroma: guided reconstruction (output 444 grid == output luma grid).
    // Both planes are processed in one pass — guide weights depend only on
    // luma and geometry, so U and V share them.
    const bool needsBackProjection = d->bp > 0.0 && d->outW == sw &&
        d->outH == sh && cw * 2 == sw && ch * 2 == sh;
    Plane outputUView, outputVView;
    Plane *outputU = &workspace.fullSlot0;
    Plane *outputV = &workspace.fullSlot1;
    if (isFloat && !d->bm) {
        outputUView = Plane::writableView(
            reinterpret_cast<float *>(vsapi->getWritePtr(dst, 1)),
            d->outW, d->outH,
            int(vsapi->getStride(dst, 1) / sizeof(float)));
        outputVView = Plane::writableView(
            reinterpret_cast<float *>(vsapi->getWritePtr(dst, 2)),
            d->outW, d->outH,
            int(vsapi->getStride(dst, 2) / sizeof(float)));
        outputU = &outputUView;
        outputV = &outputVView;
    } else {
        workspace.fullSlot0.resizeDiscard(d->outW, d->outH);
        workspace.fullSlot1.resizeDiscard(d->outW, d->outH);
    }
    Plane &cOutU = *outputU;
    Plane &cOutV = *outputV;
    if (d->algo == 4) {
        // Selector: plain base + guided metadata pass. LGF is intentionally
        // delayed until w3 has identified a diagonal-edge ROI.
        {
            ScopedStageTimer timer(metrics, Stage::BuildBaseChroma);
            plainChroma(d, cb, cr, y, gm, sw, sh, cw, ch, cOutU, cOutV, metrics);
        }
        if (d->strength > 0.0) {
            if (!isFloat) {
                // Keep the original single-expression selector for integer
                // output. Materializing the guided pass in a float plane can
                // move an otherwise equivalent value across a 16-bit rounding
                // boundary, which violates Recon's numerical contract.
                LGFMaps lgf;
                {
                    ScopedStageTimer timer(metrics, Stage::BuildLGF);
                    lgf = buildLGFMaps(
                        d, y, cb, cr, cw, ch, rw, rh, metrics,
                        nullptr, 0, &workspace.scratch);
                }
                if (metrics)
                    metrics->addWork(Stage::BuildLGF, uint64_t(cw) * ch);
                if (gm.ms.w > 0)
                    for (int j = 0; j < ch; ++j)
                        for (int i = 0; i < cw; ++i) {
                            const float f = 1.0f - float(d->ms) *
                                (1.0f - gm.ms.at(i, j));
                            lgf.confU.at(i, j) *= f;
                            lgf.confV.at(i, j) *= f;
                        }

                LGCRData integerSelector = *d;
                integerSelector.algo = 2;
                integerSelector.strength = 1.0;
                ChromaJob job;
                job.srcU = &cb; job.srcV = &cr; job.srcY = &y; job.gm = &gm;
                job.dstU = &cOutU; job.dstV = &cOutV;
                job.srcLumaW = sw; job.srcLumaH = sh;
                job.rw = rw; job.rh = rh;
                job.shiftX = d->shiftX; job.shiftY = d->shiftY;
                job.d = &integerSelector;
                job.selectorMaps = &lgf;
                job.selectorStrength = float(d->strength);
                job.plainU = &cOutU;
                job.plainV = &cOutV;
                job.metrics = metrics;
                if (hasSparseWorkset)
                    job.workset = &sparseWorkset;
                {
                    ScopedStageTimer timer(metrics, Stage::ApplyGuidedCorrection);
                    reconstructChroma(job);
                }
                if (metrics)
                    metrics->addWork(Stage::ApplySelector,
                                     uint64_t(d->outW) * d->outH);
            } else {
                LGCRData d4 = *d;
                d4.algo = 2;
                d4.strength = 1.0; // fades still apply; lam applied once in blend
                const bool compressedSelector = hasSparseWorkset &&
                    !sparseWorkset.outputDenseFallback();
                Plane guidedU, guidedV, w2, w3;
                CompressedSelector selector;
                if (compressedSelector)
                    selector.resize(sparseWorkset.activeOutputPixels);
                else {
                    guidedU = Plane(d->outW, d->outH);
                    guidedV = Plane(d->outW, d->outH);
                    w2 = Plane(d->outW, d->outH);
                    w3 = Plane(d->outW, d->outH);
                }
                ChromaJob job;
                job.srcU = &cb; job.srcV = &cr; job.srcY = &y; job.gm = &gm;
                job.dstU = compressedSelector ? &cOutU : &guidedU;
                job.dstV = compressedSelector ? &cOutV : &guidedV;
                job.srcLumaW = sw; job.srcLumaH = sh;
                job.rw = rw; job.rh = rh; job.shiftX = d->shiftX; job.shiftY = d->shiftY;
                job.d = &d4;
                job.selectorStrength = float(d->strength);
                job.selectorMetadata = true;
                if (compressedSelector) {
                    job.compressedSelector = &selector;
                    job.plainU = &cOutU;
                    job.plainV = &cOutV;
                } else {
                    job.selectorW2 = &w2;
                    job.selectorW3 = &w3;
                }
                job.metrics = metrics;
                if (hasSparseWorkset)
                    job.workset = &sparseWorkset;
                {
                    ScopedStageTimer timer(metrics, Stage::ApplyGuidedCorrection);
                    reconstructChroma(job);
                }

                const auto ax = frameAxisX ? frameAxisX
                    : cachedChromaAxis(d, sw, d->outW, rw, d->shiftX);
                const auto ay = frameAxisY ? frameAxisY
                    : cachedChromaAxis(d, sh, d->outH, rh, d->shiftY);
                std::vector<uint8_t> roi(size_t(cw) * ch, 0);
                auto markRoi = [&](int ox, int oy, float selectorW3) {
                    if (selectorW3 < 1e-4f)
                        return;
                    const int cx0 = ax->chromaBilin.i0[ox];
                    const int cx1 = std::min(cx0 + 1, cw - 1);
                    const int cy0 = ay->chromaBilin.i0[oy];
                    const int cy1 = std::min(cy0 + 1, ch - 1);
                    roi[size_t(cy0) * cw + cx0] = 1;
                    roi[size_t(cy0) * cw + cx1] = 1;
                    roi[size_t(cy1) * cw + cx0] = 1;
                    roi[size_t(cy1) * cw + cx1] = 1;
                };
                if (compressedSelector) {
                    for (int oy = 0; oy < d->outH; ++oy) {
                        size_t index = sparseWorkset.outputIndexRowOffsets[oy];
                        for (size_t spanIndex = sparseWorkset.outputRowOffsets[oy];
                             spanIndex < sparseWorkset.outputRowOffsets[oy + 1]; ++spanIndex) {
                            const SparseSpan span = sparseWorkset.outputSpans[spanIndex];
                            for (int ox = span.begin; ox < span.end; ++ox, ++index)
                                markRoi(ox, oy, selector.w3[index]);
                        }
                    }
                } else if (hasSparseWorkset) {
                    for (int oy = 0; oy < d->outH; ++oy)
                        for (size_t spanIndex = sparseWorkset.outputRowOffsets[oy];
                             spanIndex < sparseWorkset.outputRowOffsets[oy + 1]; ++spanIndex) {
                            const SparseSpan span = sparseWorkset.outputSpans[spanIndex];
                            for (int ox = span.begin; ox < span.end; ++ox)
                                markRoi(ox, oy, w3.row(oy)[ox]);
                        }
                } else {
                    for (int oy = 0; oy < d->outH; ++oy)
                        for (int ox = 0; ox < d->outW; ++ox)
                            markRoi(ox, oy, w3.row(oy)[ox]);
                }
                LGFMaps lgf;
                {
                    ScopedStageTimer timer(metrics, Stage::BuildLGF);
                    lgf = buildLGFMaps(d, y, cb, cr, cw, ch, rw, rh, metrics,
                                       roi.data(), cw, &workspace.scratch);
                }
                if (metrics)
                    metrics->addWork(Stage::BuildLGF, uint64_t(std::count(
                        roi.begin(), roi.end(), uint8_t{1})));
                {
                    ScopedCpuTimer timer(metrics, CpuProfileSlot::LGFFinalize);
                    if (gm.ms.w > 0)
                        for (int j = 0; j < ch; ++j)
                            for (int i = 0; i < cw; ++i) {
                                if (roi[size_t(j) * cw + i] == 0)
                                    continue;
                                const float f = 1.0f - float(d->ms) * (1.0f - gm.ms.at(i, j));
                                lgf.confU.at(i, j) *= f;
                                lgf.confV.at(i, j) *= f;
                            }
                }
                {
                    ScopedStageTimer timer(metrics, Stage::ApplySelector);
                    if (compressedSelector)
                        blendSelectorCompressed(
                            cOutU, cOutV, selector, lgf.aU, lgf.bU, lgf.aV, lgf.bV,
                            lgf.confU, lgf.confV, y, *ax, *ay, sparseWorkset,
                            float(d->strength), metrics);
                    else
                        blendSelector(cOutU, cOutV, guidedU, guidedV,
                                      lgf.aU, lgf.bU, lgf.aV, lgf.bV,
                                      lgf.confU, lgf.confV, y, *ax, *ay, w2, w3,
                                      cw, ch, float(d->strength),
                                      hasSparseWorkset ? &sparseWorkset : nullptr, metrics);
                }
                if (metrics)
                    metrics->addWork(Stage::ApplySelector, compressedSelector
                        ? sparseWorkset.activeOutputPixels
                        : uint64_t(d->outW) * d->outH);
            }
        }
    } else if (d->algo == 6) {
        // Constrained detail transfer: plain kernel base + g*a*(Y - P(D(Y))).
        // Low frequency and color reference come from the plain kernel; the
        // affine model only supplies the detail the plain path lost.
        {
            ScopedStageTimer timer(metrics, Stage::BuildBaseChroma);
            plainChroma(d, cb, cr, y, gm, sw, sh, cw, ch, cOutU, cOutV, metrics);
        }
        if (d->strength > 0.0) {
            AffineMaps af;
            {
                ScopedStageTimer timer(metrics, Stage::BuildAffineMaps);
                af = buildAffineMaps(
                    d, y, cb, cr, cw, ch, rw, rh, metrics, &gm,
                    hasSparseWorkset ? &sparseWorkset : nullptr,
                    &workspace.scratch);
            }
            if (metrics)
                metrics->addWork(Stage::BuildAffineMaps, uint64_t(cw) * ch);
            // These planes are dead before the full-resolution detail stage.
            // Releasing them here lets the allocator reuse their storage.
            gm.lc = Plane();
            if (!needsBackProjection) {
                cb = Plane();
                cr = Plane();
            }
            {
                ScopedStageTimer timer(metrics, Stage::BuildDetail);
                buildDetailMap(d, y, cw, ch, rw, rh, af, metrics);
            }
            if (metrics)
                metrics->addWork(Stage::BuildDetail, uint64_t(d->outW) * d->outH);
            if (!d->bm)
                y = Plane();
            const auto ax = frameAxisX ? frameAxisX
                : cachedChromaAxis(d, sw, d->outW, rw, d->shiftX);
            const auto ay = frameAxisY ? frameAxisY
                : cachedChromaAxis(d, sh, d->outH, rh, d->shiftY);
            const uint8_t *mp = nullptr;
            int mw = 0, mh = 0;
            if (hasSparseWorkset) {
                mp = sparseWorkset.mask.data();
                mw = sw;
                mh = sh;
            }
            {
                ScopedStageTimer timer(metrics, Stage::ApplyDetailTransfer);
                detailTransfer(d, cOutU, cOutV, af, gm, *ax, *ay, mp, mw, mh,
                               hasSparseWorkset ? &sparseWorkset : nullptr, metrics);
            }
            if (metrics)
                metrics->addWork(Stage::ApplyDetailTransfer,
                                 uint64_t(d->outW) * d->outH);
        }
    } else if (d->strength == 0.0) {
        // Pure kernel A/B reference
        ScopedStageTimer timer(metrics, Stage::BuildBaseChroma);
        plainChroma(d, cb, cr, y, gm, sw, sh, cw, ch, cOutU, cOutV, metrics);
    } else {
        // algo=2 guided path. Sparse mode: plain kernel everywhere, guided
        // correction only where luma structure makes it worthwhile.
        CompressedChromaHull plainHull;
        const bool reusePlainHull = hasSparseWorkset &&
            !sparseWorkset.outputDenseFallback() && !d->radial &&
            d->kernel != Kernel::Bilinear;
        if (hasSparseWorkset) {
            if (reusePlainHull)
                plainHull.resize(sparseWorkset.activeOutputPixels);
            {
                ScopedStageTimer timer(metrics, Stage::BuildBaseChroma);
                plainChroma(
                    d, cb, cr, y, gm, sw, sh, cw, ch, cOutU, cOutV, metrics,
                    reusePlainHull ? &plainHull : nullptr,
                    reusePlainHull ? &sparseWorkset : nullptr);
            }
        }
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
        job.metrics = metrics;
        if (hasSparseWorkset) {
            job.workset = &sparseWorkset;
            job.plainU = &cOutU;
            job.plainV = &cOutV;
            if (reusePlainHull)
                job.plainHull = &plainHull;
        }
        {
            ScopedStageTimer timer(metrics, Stage::ApplyGuidedCorrection);
            reconstructChroma(job);
        }
    }
    if (d->bm) {
        ScopedStageTimer timer(metrics, Stage::ApplyCollaborativeFilter);
        collaborativeChromaFilter(y, cOutU, cOutV, float(d->sigma), metrics);
    }

    // Back-projection data consistency: re-downsample the reconstruction and
    // return a fraction of the residual, D_h(C + delta) ~= C_src.
    if (needsBackProjection) {
        ScopedStageTimer timer(metrics, Stage::BackProject);
        backProject(cOutU, cb, float(d->bp));
        backProject(cOutV, cr, float(d->bp));
        if (metrics)
            metrics->addWork(Stage::BackProject, 2 * uint64_t(cw) * ch, 8 * uint64_t(cw) * ch);
    }

    if (!isFloat || d->bm) {
        ScopedStageTimer timer(metrics, Stage::ConvertOutput);
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
        if (metrics)
            metrics->addWork(Stage::ConvertOutput,
                             2 * uint64_t(d->outW) * d->outH);
    }

    if (metrics) {
        const auto totalNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - frameStart).count();
        writeProfile(dst, *metrics, static_cast<uint64_t>(totalNs), vsapi);
    }

    vsapi->freeFrame(src);
    return dst;
}

// ---------------------------------------------------------------------------
// TRecon: temporal chroma reconstruction (motion-compensated multi-frame)
// ---------------------------------------------------------------------------

static void VS_CC lgcrFree(void *instanceData, VSCore *, const VSAPI *vsapi);

static std::vector<int> temporalFrameNumbers(int n, int nLast, int trad) {
    std::vector<int> frames;
    frames.reserve(2 * trad);
    for (int k = -trad; k <= trad; ++k) {
        const int fn = std::clamp(n + k, 0, nLast);
        if (fn != n && std::find(frames.begin(), frames.end(), fn) == frames.end())
            frames.push_back(fn);
    }
    return frames;
}

static const VSFrame *VS_CC tReconGetFrame(int n, int activationReason, void *instanceData,
                                           void **, VSFrameContext *frameCtx, VSCore *core,
                                           const VSAPI *vsapi) {
    LGCRData *d = static_cast<LGCRData *>(instanceData);
    const int trad = d->trad;
    if (activationReason == arInitial) {
        const int nLast = d->viIn->numFrames - 1;
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        for (const int fn : temporalFrameNumbers(n, nLast, trad))
            vsapi->requestFrameFilter(fn, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady)
        return nullptr;

    const int nLast = d->viIn->numFrames - 1;
    const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
    const VSVideoFormat *fmt = vsapi->getVideoFrameFormat(src);

    // Per-frame H.273 siting (same rules as Recon; subsampled axes only)
    LGCRData dloc = *d;
    applyFrameChromaSiting(dloc, fmt, src, vsapi);
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
    const std::vector<int> neighborFrames = temporalFrameNumbers(n, nLast, trad);
    nbrs.reserve(neighborFrames.size());
    store.reserve(4 * neighborFrames.size());
    mvStoreX.reserve(neighborFrames.size());
    mvStoreY.reserve(neighborFrames.size());
    confStore.reserve(neighborFrames.size());
    if (d->strength > 0.0) {
        for (const int fn : neighborFrames) {
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
            applyChromaConsistency(cb, cr, *nu, *nv, sw, sh, 16,
                                   mvStoreX.back(), mvStoreY.back(), bw, bh,
                                   confStore.back());
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
        gm = buildGuideMaps(y, y, cw, ch, rw, rh, d->shiftX, d->shiftY);
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

    if (d->bm)
        collaborativeChromaFilter(y, cOutU, cOutV, float(d->sigma));

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
        const int64_t bm = vsapi->mapGetIntSaturated(in, "bm", 0, &err);
        d->bm = err ? false : (bm != 0);
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
        const int64_t taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        const int64_t checkedTaps = err ? 3 : taps;
        if (checkedTaps < 1 || checkedTaps > 64) {
            fail("LGCR: lanczos taps must be in the range 1..64");
            return;
        }
        d->kp1 = double(checkedTaps);
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
        const int64_t taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        const int64_t checkedTaps = err ? 3 : taps;
        if (checkedTaps < 1 || checkedTaps > 64) {
            fail("LGCR: jinc taps must be in the range 1..64");
            return;
        }
        d->kp1 = double(checkedTaps);
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
    if (d->algo != 2 && d->algo != 4 && d->algo != 6) {
        fail("LGCR: algo must be 2 (guided), 4 (selector) or 6 (detail transfer)");
        return;
    }
    d->sdb = vsapi->mapGetFloatSaturated(in, "sdb", 0, &err);
    if (err) d->sdb = 3.0;
    d->stretch = vsapi->mapGetFloatSaturated(in, "stretch", 0, &err); if (err) d->stretch = 1.0;
    d->gsigma = vsapi->mapGetFloatSaturated(in, "gsigma", 0, &err); if (err) d->gsigma = 2.5;
    d->rescue = vsapi->mapGetFloatSaturated(in, "rescue", 0, &err); if (err) d->rescue = 1.0;
    {
        const int64_t ridgeInt = vsapi->mapGetIntSaturated(in, "ridge", 0, &err);
        d->ridge = err ? true : (ridgeInt != 0);
        const int64_t ce = vsapi->mapGetIntSaturated(in, "cedge", 0, &err);
        d->cedge = err ? false : (ce != 0);
    }
    d->ms = vsapi->mapGetFloatSaturated(in, "ms", 0, &err); if (err) d->ms = 1.0;
    d->qgate = vsapi->mapGetFloatSaturated(in, "qgate", 0, &err); if (err) d->qgate = 1.0;
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
        const int64_t bm = vsapi->mapGetIntSaturated(in, "bm", 0, &err);
        d->bm = err ? false : (bm != 0);
    }
    d->bp = vsapi->mapGetFloatSaturated(in, "bp", 0, &err); if (err) d->bp = 0.0;
    if (fmt->subSamplingW == 0)
        d->shiftX = 0.0;
    if (fmt->subSamplingH == 0)
        d->shiftY = 0.0;

    if (d->strength < 0.0 || d->strength > 1.0 || d->sigma <= 0.0 || d->sratio <= 0.0 ||
        d->sdb <= 0.0 || d->stretch < 0.0 || d->gsigma <= 0.0 || d->reg <= 0.0 ||
        d->bp < 0.0 || d->bp > 1.0 || d->ms < 0.0 || d->ms > 1.0 ||
        d->qgate < 0.0 || d->qgate > 1.0 || d->rescue < 0.0 || d->rescue > 1.0) {
        fail("LGCR: invalid parameter range (need 0<=strength/rescue<=1, 0<=bp/ms/qgate<=1, "
             "sigma/sratio/sdb/gsigma/reg>0, stretch>=0)");
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
                             fmParallel, deps, 1, data, core);
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("dev.bsflab.lgcr" LGCR_SUFFIX, "lgcr" LGCR_SUFFIX,
                         "Luma-guided chroma reconstruction with direction-aware kernels",
                         VS_MAKE_VERSION(LGCR_VERSION_MAJOR, LGCR_VERSION_MINOR),
                         VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction(
        "Recon",
        "clip:vnode;width:int:opt;height:int:opt;kernel:data:opt;taps:int:opt;algo:int:opt;"
        "b:float:opt;c:float:opt;strength:float:opt;sigma:float:opt;sratio:float:opt;"
        "sdb:float:opt;stretch:float:opt;gsigma:float:opt;rescue:float:opt;ridge:int:opt;cedge:int:opt;ar:float:opt;reg:float:opt;loc:data:opt;sparse:int:opt;bp:float:opt;ms:float:opt;qgate:float:opt;bm:int:opt",
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
        "trad:int:opt;tsearch:int:opt;tsad:float:opt;bm:int:opt",
        "clip:vnode", tReconCreate, nullptr, plugin);
    registerDownsample(plugin, vspapi);
}
