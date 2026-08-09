#include "pipeline.h"

namespace lgcr {

namespace {

size_t scalarBytes(ScalarType type) {
    switch (type) {
    case ScalarType::Float32: return 4;
    case ScalarType::UInt8: return 1;
    case ScalarType::UInt16: return 2;
    }
    return 0;
}

template <class View>
bool validView(const View &view, MemoryDomain expectedDomain) {
    if (!view.data || view.width <= 0 || view.height <= 0 ||
        view.domain != expectedDomain)
        return false;
    const size_t rowBytes = size_t(view.width) * scalarBytes(view.type);
    return view.strideBytes >= static_cast<ptrdiff_t>(rowBytes);
}

} // namespace

bool validPlaneView(const ConstPlaneView &view, MemoryDomain expectedDomain) {
    return validView(view, expectedDomain);
}

bool validPlaneView(const PlaneView &view, MemoryDomain expectedDomain) {
    return validView(view, expectedDomain);
}

StageStatus dispatchCpuStage(Stage stage, PipelineContext &context,
                             CpuStageFunction function, void *userData) {
    if (!function || context.backend == BackendKind::CUDA)
        return StageStatus::InvalidContext;
    const StageContract &contract = stageContract(stage);
    if (contract.support == StageSupport::GPU)
        return StageStatus::Unsupported;
    for (size_t i = 0; i < context.inputCount; ++i) {
        const PlaneView &view = context.inputs[i].view;
        ConstPlaneView constView{ view.data, view.width, view.height,
                                  view.strideBytes, view.type, view.domain };
        if (!validPlaneView(constView, MemoryDomain::Host))
            return StageStatus::InvalidContext;
    }
    for (size_t i = 0; i < context.outputCount; ++i)
        if (!validPlaneView(context.outputs[i].view, MemoryDomain::Host))
            return StageStatus::InvalidContext;
    return function(context, userData);
}

} // namespace lgcr
