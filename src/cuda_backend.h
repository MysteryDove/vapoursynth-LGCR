// Optional CUDA pipeline infrastructure. No VapourSynth types cross this API.
#pragma once

#include "pipeline.h"

#include <array>
#include <memory>
#include <string>

namespace lgcr {

class CudaBuffer {
public:
    CudaBuffer() = default;
    ~CudaBuffer();
    CudaBuffer(CudaBuffer &&other) noexcept;
    CudaBuffer &operator=(CudaBuffer &&other) noexcept;
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    PlaneView view() const noexcept { return view_; }
    size_t bytes() const noexcept { return bytes_; }
    explicit operator bool() const noexcept { return view_.data != nullptr; }
    void reset() noexcept;

private:
    friend class CudaPipeline;
    PlaneView view_{};
    size_t bytes_ = 0;
};

using CudaStageFunction = StageStatus (*)(PipelineContext &context,
                                          void *nativeStream,
                                          void *userData);

class CudaPipeline {
public:
    CudaPipeline();
    ~CudaPipeline();
    CudaPipeline(CudaPipeline &&) noexcept;
    CudaPipeline &operator=(CudaPipeline &&) noexcept;
    CudaPipeline(const CudaPipeline &) = delete;
    CudaPipeline &operator=(const CudaPipeline &) = delete;

    bool available() const noexcept;
    const std::string &lastError() const noexcept;
    void setEventTiming(bool enabled) noexcept;
    uint64_t lastStageNanoseconds(Stage stage) const noexcept;

    StageStatus allocate(int width, int height, ScalarType type,
                         CudaBuffer &buffer);
    StageStatus upload(const ConstPlaneView &host, CudaBuffer &device);
    StageStatus download(const CudaBuffer &device, const PlaneView &host);
    StageStatus synchronize();

    void registerStage(Stage stage, CudaStageFunction function) noexcept;
    StageStatus dispatch(Stage stage, PipelineContext &context,
                         CudaStageFunction cudaFunction, void *cudaUserData,
                         CpuStageFunction cpuFallback = nullptr,
                         void *cpuUserData = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lgcr
