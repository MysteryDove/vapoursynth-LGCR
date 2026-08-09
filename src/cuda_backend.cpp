#include "cuda_backend.h"

#include <utility>

#ifndef LGCR_ENABLE_CUDA
#define LGCR_ENABLE_CUDA 0
#endif

#if LGCR_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace lgcr {

namespace {

[[maybe_unused]] size_t bytesPerSample(ScalarType type) {
    switch (type) {
    case ScalarType::Float32: return 4;
    case ScalarType::UInt8: return 1;
    case ScalarType::UInt16: return 2;
    }
    return 0;
}

} // namespace

struct CudaPipeline::Impl {
    bool available = false;
    bool eventTiming = false;
    std::string error;
    std::array<CudaStageFunction, stageCount> handlers{};
    std::array<uint64_t, stageCount> timings{};
#if LGCR_ENABLE_CUDA
    cudaStream_t stream = nullptr;
#else
    void *stream = nullptr;
#endif
};

CudaBuffer::~CudaBuffer() {
    reset();
}

CudaBuffer::CudaBuffer(CudaBuffer &&other) noexcept
    : view_(other.view_), bytes_(other.bytes_) {
    other.view_ = {};
    other.bytes_ = 0;
}

CudaBuffer &CudaBuffer::operator=(CudaBuffer &&other) noexcept {
    if (this != &other) {
        reset();
        view_ = other.view_;
        bytes_ = other.bytes_;
        other.view_ = {};
        other.bytes_ = 0;
    }
    return *this;
}

void CudaBuffer::reset() noexcept {
#if LGCR_ENABLE_CUDA
    if (view_.data)
        cudaFree(view_.data);
#endif
    view_ = {};
    bytes_ = 0;
}

CudaPipeline::CudaPipeline() : impl_(std::make_unique<Impl>()) {
#if LGCR_ENABLE_CUDA
    int devices = 0;
    cudaError_t status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) {
        impl_->error = status == cudaSuccess ? "no CUDA device" : cudaGetErrorString(status);
        return;
    }
    status = cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        impl_->error = cudaGetErrorString(status);
        return;
    }
    impl_->available = true;
#else
    impl_->error = "LGCR was built with LGCR_ENABLE_CUDA=0";
#endif
}

CudaPipeline::~CudaPipeline() {
#if LGCR_ENABLE_CUDA
    if (impl_ && impl_->stream)
        cudaStreamDestroy(impl_->stream);
#endif
}

CudaPipeline::CudaPipeline(CudaPipeline &&) noexcept = default;
CudaPipeline &CudaPipeline::operator=(CudaPipeline &&) noexcept = default;

bool CudaPipeline::available() const noexcept { return impl_->available; }
const std::string &CudaPipeline::lastError() const noexcept { return impl_->error; }
void CudaPipeline::setEventTiming(bool enabled) noexcept { impl_->eventTiming = enabled; }

uint64_t CudaPipeline::lastStageNanoseconds(Stage stage) const noexcept {
    return impl_->timings[static_cast<size_t>(stage)];
}

StageStatus CudaPipeline::allocate(int width, int height, ScalarType type,
                                   CudaBuffer &buffer) {
    if (!available() || width <= 0 || height <= 0)
        return available() ? StageStatus::InvalidContext : StageStatus::Unsupported;
#if LGCR_ENABLE_CUDA
    buffer.reset();
    size_t pitch = 0;
    const size_t rowBytes = size_t(width) * bytesPerSample(type);
    void *pointer = nullptr;
    const cudaError_t status = cudaMallocPitch(&pointer, &pitch, rowBytes, height);
    if (status != cudaSuccess) {
        impl_->error = cudaGetErrorString(status);
        return StageStatus::BackendFailure;
    }
    buffer.view_ = { pointer, width, height, static_cast<ptrdiff_t>(pitch),
                     type, MemoryDomain::CUDADevice };
    buffer.bytes_ = pitch * size_t(height);
    return StageStatus::Success;
#else
    (void)type;
    (void)buffer;
    return StageStatus::Unsupported;
#endif
}

StageStatus CudaPipeline::upload(const ConstPlaneView &host, CudaBuffer &device) {
    if (!validPlaneView(host, MemoryDomain::Host))
        return StageStatus::InvalidContext;
    StageStatus result = allocate(host.width, host.height, host.type, device);
    if (result != StageStatus::Success)
        return result;
#if LGCR_ENABLE_CUDA
    const size_t rowBytes = size_t(host.width) * bytesPerSample(host.type);
    const cudaError_t status = cudaMemcpy2DAsync(
        device.view_.data, size_t(device.view_.strideBytes), host.data,
        size_t(host.strideBytes), rowBytes, host.height,
        cudaMemcpyHostToDevice, impl_->stream);
    if (status != cudaSuccess) {
        impl_->error = cudaGetErrorString(status);
        device.reset();
        return StageStatus::BackendFailure;
    }
    return StageStatus::Success;
#else
    return StageStatus::Unsupported;
#endif
}

StageStatus CudaPipeline::download(const CudaBuffer &device, const PlaneView &host) {
    if (!available() || !device || !validPlaneView(host, MemoryDomain::Host) ||
        device.view_.width != host.width || device.view_.height != host.height ||
        device.view_.type != host.type)
        return available() ? StageStatus::InvalidContext : StageStatus::Unsupported;
#if LGCR_ENABLE_CUDA
    const size_t rowBytes = size_t(host.width) * bytesPerSample(host.type);
    cudaError_t status = cudaMemcpy2DAsync(
        host.data, size_t(host.strideBytes), device.view_.data,
        size_t(device.view_.strideBytes), rowBytes, host.height,
        cudaMemcpyDeviceToHost, impl_->stream);
    if (status == cudaSuccess)
        status = cudaStreamSynchronize(impl_->stream);
    if (status != cudaSuccess) {
        impl_->error = cudaGetErrorString(status);
        return StageStatus::BackendFailure;
    }
    return StageStatus::Success;
#else
    return StageStatus::Unsupported;
#endif
}

StageStatus CudaPipeline::synchronize() {
    if (!available())
        return StageStatus::Unsupported;
#if LGCR_ENABLE_CUDA
    const cudaError_t status = cudaStreamSynchronize(impl_->stream);
    if (status != cudaSuccess) {
        impl_->error = cudaGetErrorString(status);
        return StageStatus::BackendFailure;
    }
    return StageStatus::Success;
#else
    return StageStatus::Unsupported;
#endif
}

void CudaPipeline::registerStage(Stage stage, CudaStageFunction function) noexcept {
    impl_->handlers[static_cast<size_t>(stage)] = function;
}

StageStatus CudaPipeline::dispatch(Stage stage, PipelineContext &context,
                                   CudaStageFunction cudaFunction, void *cudaUserData,
                                   CpuStageFunction cpuFallback, void *cpuUserData) {
#if !LGCR_ENABLE_CUDA
    (void)cudaUserData;
#endif
    CudaStageFunction function = cudaFunction
        ? cudaFunction : impl_->handlers[static_cast<size_t>(stage)];
    if (available() && function) {
#if LGCR_ENABLE_CUDA
        cudaEvent_t begin = nullptr, end = nullptr;
        if (impl_->eventTiming) {
            cudaEventCreate(&begin);
            cudaEventCreate(&end);
            cudaEventRecord(begin, impl_->stream);
        }
        context.backend = BackendKind::CUDA;
        context.backendContext = impl_->stream;
        const StageStatus result = function(context, impl_->stream, cudaUserData);
        if (impl_->eventTiming) {
            cudaEventRecord(end, impl_->stream);
            cudaEventSynchronize(end);
            float milliseconds = 0.0f;
            cudaEventElapsedTime(&milliseconds, begin, end);
            impl_->timings[static_cast<size_t>(stage)] =
                static_cast<uint64_t>(milliseconds * 1.0e6f);
            cudaEventDestroy(begin);
            cudaEventDestroy(end);
        }
        if (result == StageStatus::Success)
            return result;
#endif
    }
    if (!cpuFallback)
        return StageStatus::Unsupported;
#ifdef __AVX2__
    context.backend = BackendKind::AVX2;
#else
    context.backend = BackendKind::Scalar;
#endif
    context.backendContext = nullptr;
    return dispatchCpuStage(stage, context, cpuFallback, cpuUserData);
}

} // namespace lgcr
