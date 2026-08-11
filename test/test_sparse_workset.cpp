#include "lgcr.h"

#include <algorithm>
#include <cassert>
#include <cmath>

using namespace lgcr;

namespace {

ChromaAxis makeAxis(int outputs, int source) {
    ChromaAxis axis;
    axis.n = outputs;
    axis.chromaBilin.n = outputs;
    axis.chromaBilin.i0.resize(outputs);
    axis.chromaBilin.f.resize(outputs);
    for (int x = 0; x < outputs; ++x) {
        const float position = (x + 0.5f) * source / outputs - 0.5f;
        const int raw = int(std::floor(position));
        axis.chromaBilin.i0[x] = source == 1 ? 0 : std::clamp(raw, 0, source - 2);
        axis.chromaBilin.f[x] = position - raw;
    }
    return axis;
}

void testMaskAndSpans() {
    const int width = 8, height = 4, chromaWidth = 4, chromaHeight = 2;
    std::vector<uint8_t> mask(size_t(width) * height, 0);
    for (int x = 1; x < 4; ++x)
        mask[size_t(1) * width + x] = 1;
    for (int x = 5; x < 8; ++x)
        mask[size_t(3) * width + x] = 1;
    const ChromaAxis ax = makeAxis(width, chromaWidth);
    const ChromaAxis ay = makeAxis(height, chromaHeight);
    const SparseWorkset workset = buildSparseWorkset(
        mask, width, height, width, height, ax, ay, chromaWidth, chromaHeight);

    assert(workset.activeOutputPixels == 6);
    assert(workset.outputSpans.size() == 2);
    assert(workset.outputSpans[0].begin == 1 && workset.outputSpans[0].end == 4);
    assert(workset.outputSpans[1].begin == 5 && workset.outputSpans[1].end == 8);
    assert(workset.outputRowOffsets[0] == 0 && workset.outputRowOffsets[1] == 0);
    assert(workset.outputRowOffsets[2] == 1 && workset.outputRowOffsets[4] == 2);
    assert(workset.outputIndexRowOffsets[0] == 0 &&
           workset.outputIndexRowOffsets[1] == 0);
    assert(workset.outputIndexRowOffsets[2] == 3 &&
           workset.outputIndexRowOffsets[4] == 6);
    assert(workset.outputIndices.front() == 9 && workset.outputIndices.back() == 31);
    assert(!workset.outputDenseFallback());
    assert(workset.activeChromaPixels == size_t(std::count(
        workset.chromaMask.begin(), workset.chromaMask.end(), uint8_t{1})));

    std::fill(mask.begin(), mask.end(), 1);
    const SparseWorkset dense = buildSparseWorkset(
        mask, width, height, width, height, ax, ay, chromaWidth, chromaHeight);
    assert(dense.outputDenseFallback());
    assert(dense.chromaDenseFallback());

    std::fill(mask.begin(), mask.end(), 0);
    std::fill(mask.begin(), mask.begin() + mask.size() / 2, 1);
    const SparseWorkset half = buildSparseWorkset(
        mask, width, height, width, height, ax, ay, chromaWidth, chromaHeight);
    assert(half.activeOutputPixels * 2 == mask.size());
    assert(half.outputDenseFallback());
}

void testTrustDilation() {
    GuideMaps maps;
    maps.jxx = Plane(9, 7);
    maps.jyy = Plane(9, 7);
    maps.jxx.fill(0.0f);
    maps.jyy.fill(0.0f);
    maps.jxx.at(4, 3) = 1.0f;
    const auto mask = buildTrustMask(maps, 9, 7, 0.1, 2);
    for (int y = 0; y < 7; ++y)
        for (int x = 0; x < 9; ++x)
            assert(bool(mask[size_t(y) * 9 + x]) ==
                   (std::abs(x - 4) <= 2 && std::abs(y - 3) <= 2));
}

void testFusedTrustSeed() {
    Plane y(9, 7);
    y.fill(0.0f);
    for (int py = 0; py < y.h; ++py)
        for (int px = 4; px < y.w; ++px)
            y.at(px, py) = 1.0f;
    const GuideMaps fused = buildGuideMaps(
        y, y, 9, 7, 1.0, 1.0, 0.0, 0.0, nullptr, 0.1);
    GuideMaps fallback = fused;
    fallback.trustSeed.clear();
    const auto fusedMask = buildTrustMask(fused, 9, 7, 0.1, 2);
    const auto fallbackMask = buildTrustMask(fallback, 9, 7, 0.1, 2);
    assert(fusedMask == fallbackMask);
}

void testMutualRoi() {
    constexpr int output = 16, chroma = 8;
    const ChromaAxis ax = makeAxis(output, chroma);
    const ChromaAxis ay = makeAxis(output, chroma);
    std::vector<uint8_t> mask(size_t(output) * output, 0);
    for (int y = 3; y < 13; ++y)
        for (int x = 6; x < 10; ++x)
            mask[size_t(y) * output + x] = 1;
    const SparseWorkset workset = buildSparseWorkset(
        mask, output, output, output, output, ax, ay, chroma, chroma);

    Plane lc(chroma, chroma), u(chroma, chroma), v(chroma, chroma);
    for (int y = 0; y < chroma; ++y)
        for (int x = 0; x < chroma; ++x) {
            const float edge = x >= 4 ? 1.0f : 0.0f;
            lc.at(x, y) = 0.2f + 0.3f * edge + 0.01f * y;
            u.at(x, y) = -0.2f + 0.4f * edge;
            v.at(x, y) = 0.25f - 0.35f * edge;
        }
    const Plane dense = buildMutualGate(lc, u, v, 0.01);
    const Plane sparse = buildMutualGate(
        lc, u, v, 0.01, workset.chromaMask.data(), chroma, &workset);
    for (int y = 0; y < chroma; ++y)
        for (int x = 0; x < chroma; ++x)
            if (workset.chromaMask[size_t(y) * chroma + x])
                assert(std::fabs(dense.at(x, y) - sparse.at(x, y)) <= 1e-6f);
}

void testWorkspaceBudget() {
    auto pool = std::make_shared<FrameWorkspacePool>(1024);
    {
        auto lease = pool->acquire(2);
        lease.get().sourceY.resizeDiscard(8, 8);
        assert(lease.get().retainedBytes() <= 1024);
    }
    {
        auto lease = pool->acquire(2);
        assert(lease.get().retainedBytes() != 0);
        lease.get().sourceY.resizeDiscard(64, 64);
        assert(lease.get().retainedBytes() > 1024);
    }
    {
        auto lease = pool->acquire(2);
        assert(lease.get().retainedBytes() == 0);
    }
}

void testScratchSizeClasses() {
    FrameScratchAllocator scratch;
    {
        Plane first = scratch.acquire(17, 9);
        assert(first.retainedBytes() >= size_t(17 * 9) * sizeof(float));
        assert(scratch.retainedBytes() == 0);
        first.fill(3.0f);
    }
    const size_t retained = scratch.retainedBytes();
    assert(retained != 0 && scratch.largestBytes() == retained);
    {
        Plane reused = scratch.acquire(17, 8);
        assert(scratch.retainedBytes() == 0);
        reused.fill(4.0f);
    }
    assert(scratch.retainedBytes() == retained);
    assert(scratch.releaseLargest());
    assert(scratch.retainedBytes() == 0);

    FrameWorkspace workspace;
    {
        Plane cached = workspace.scratch.acquire(64, 64);
    }
    assert(workspace.retainedBytes() != 0);
    workspace.trimTo(0);
    assert(workspace.retainedBytes() == 0);
}

} // namespace

int main() {
    testMaskAndSpans();
    testTrustDilation();
    testFusedTrustSeed();
    testMutualRoi();
    testWorkspaceBudget();
    testScratchSizeClasses();
}
