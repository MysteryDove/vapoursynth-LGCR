#include "cuda_backend.h"

#include <array>
#include <cassert>

using namespace lgcr;

namespace {

StageStatus copyStage(PipelineContext &context, void *userData) {
    bool *called = static_cast<bool *>(userData);
    *called = true;
    const PlaneView &input = context.inputs[0].view;
    PlaneView &output = context.outputs[0].view;
    const float *source = static_cast<const float *>(input.data);
    float *target = static_cast<float *>(output.data);
    for (int y = 0; y < input.height; ++y)
        for (int x = 0; x < input.width; ++x)
            target[y * output.strideBytes / sizeof(float) + x] =
                source[y * input.strideBytes / sizeof(float) + x];
    return StageStatus::Success;
}

} // namespace

int main() {
    static_assert(stageContracts.size() == stageCount);
    for (size_t i = 0; i < stageCount; ++i)
        assert(static_cast<size_t>(stageContracts[i].stage) == i);

    std::array<float, 8> source{ 0, 1, 2, 3, 4, 5, 6, 7 };
    std::array<float, 8> target{};
    PlaneBuffer input{
        { source.data(), 4, 2, 4 * ptrdiff_t(sizeof(float)),
          ScalarType::Float32, MemoryDomain::Host },
        source.size() * sizeof(float), true, false
    };
    PlaneBuffer output{
        { target.data(), 4, 2, 4 * ptrdiff_t(sizeof(float)),
          ScalarType::Float32, MemoryDomain::Host },
        target.size() * sizeof(float), true, false
    };
    PipelineContext context;
    context.backend = BackendKind::Scalar;
    context.inputs = &input;
    context.outputs = &output;
    context.inputCount = context.outputCount = 1;

    bool called = false;
    assert(dispatchCpuStage(Stage::ResampleLuma, context, copyStage, &called) ==
           StageStatus::Success);
    assert(called && source == target);

    CudaPipeline cuda;
#if !LGCR_ENABLE_CUDA
    assert(!cuda.available());
#endif
    target.fill(0.0f);
    called = false;
    assert(cuda.dispatch(Stage::ResampleLuma, context, nullptr, nullptr,
                         copyStage, &called) == StageStatus::Success);
    assert(called && source == target);
    return 0;
}
