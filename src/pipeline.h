// Backend-neutral pipeline vocabulary. The CPU implementation still owns
// Plane storage; these types keep VapourSynth details out of compute stages
// and form the ABI shared by the optional CUDA executor.
#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace lgcr {

enum class ScalarType : uint8_t { Float32, UInt8, UInt16 };
enum class BoundaryMode : uint8_t { Clamp, InteriorOnly, ScalarFallback };
enum class StageSupport : uint8_t { CPU, GPU, CPUAndGPU };
enum class MemoryDomain : uint8_t { Host, CUDADevice };
enum class BackendKind : uint8_t { Scalar, AVX2, CUDA };

enum class BufferId : uint8_t {
    InputFrame,
    SourceY,
    SourceU,
    SourceV,
    OutputY,
    GuideTensor,
    MutualGate,
    BaseU,
    BaseV,
    GuidedU,
    GuidedV,
    LGFMaps,
    AffineMaps,
    Detail,
    TrustMask,
    OutputU,
    OutputV,
    OutputFrame,
};

using BufferMask = uint64_t;
constexpr BufferMask bufferBit(BufferId id) {
    return BufferMask{1} << static_cast<uint8_t>(id);
}

struct PlaneView {
    void *data = nullptr;
    int width = 0;
    int height = 0;
    ptrdiff_t strideBytes = 0;
    ScalarType type = ScalarType::Float32;
    MemoryDomain domain = MemoryDomain::Host;
};

struct ConstPlaneView {
    const void *data = nullptr;
    int width = 0;
    int height = 0;
    ptrdiff_t strideBytes = 0;
    ScalarType type = ScalarType::Float32;
    MemoryDomain domain = MemoryDomain::Host;
};

struct PlaneBuffer {
    PlaneView view;
    size_t bytes = 0;
    bool frameDependent = true;
    bool allowInPlace = false;
};

struct Geometry {
    int srcWidth = 0;
    int srcHeight = 0;
    int dstWidth = 0;
    int dstHeight = 0;
    double ratioX = 1.0;
    double ratioY = 1.0;
    double shiftX = 0.0;
    double shiftY = 0.0;
};

struct KernelWeights {
    int taps = 0;
    int outputs = 0;
    const float *weights = nullptr;
    const int *starts = nullptr;
    const uint8_t *activity = nullptr;
};

struct GuideMapViews {
    ConstPlaneView jxx, jxy, jyy, lc, ms;
};

struct AffineMapViews {
    ConstPlaneView aU, aV, confidence, detail;
};

enum class Stage : uint8_t {
    ConvertInput,
    ResampleLuma,
    BuildGuideMaps,
    BuildTrustMask,
    BuildMutualGate,
    BuildBaseChroma,
    BuildLGF,
    BuildAffineMaps,
    BuildDetail,
    ApplyGuidedCorrection,
    ApplySelector,
    ApplyDetailTransfer,
    ApplyCollaborativeFilter,
    DownsampleBase,
    DownsampleGuide,
    DownsampleCandidateScore,
    DownsampleOutput,
    BackProject,
    ConvertOutput,
};

struct StageContract {
    Stage stage;
    StageSupport support = StageSupport::CPU;
    BoundaryMode boundary = BoundaryMode::Clamp;
    ScalarType precision = ScalarType::Float32;
    BufferMask inputs = 0;
    BufferMask outputs = 0;
    bool frameDependent = true;
    bool allowInPlace = false;
};

struct PipelineContext {
    Geometry geometry;
    BackendKind backend = BackendKind::Scalar;
    const PlaneBuffer *inputs = nullptr;
    PlaneBuffer *outputs = nullptr;
    size_t inputCount = 0;
    size_t outputCount = 0;
    void *backendContext = nullptr;
};

constexpr size_t stageCount = static_cast<size_t>(Stage::ConvertOutput) + 1;

inline constexpr std::array<StageContract, stageCount> stageContracts = {{
    { Stage::ConvertInput, StageSupport::CPU, BoundaryMode::Clamp,
      ScalarType::Float32, bufferBit(BufferId::InputFrame),
      bufferBit(BufferId::SourceY) | bufferBit(BufferId::SourceU) |
          bufferBit(BufferId::SourceV), true, false },
    { Stage::ResampleLuma, StageSupport::CPU, BoundaryMode::ScalarFallback,
      ScalarType::Float32, bufferBit(BufferId::SourceY),
      bufferBit(BufferId::OutputY), true, false },
    { Stage::BuildGuideMaps, StageSupport::CPU, BoundaryMode::ScalarFallback,
      ScalarType::Float32, bufferBit(BufferId::SourceY),
      bufferBit(BufferId::GuideTensor), true, false },
    { Stage::BuildTrustMask, StageSupport::CPU, BoundaryMode::Clamp,
      ScalarType::UInt8, bufferBit(BufferId::GuideTensor),
      bufferBit(BufferId::TrustMask), true, false },
    { Stage::BuildMutualGate, StageSupport::CPU, BoundaryMode::ScalarFallback,
      ScalarType::Float32, bufferBit(BufferId::GuideTensor) |
          bufferBit(BufferId::SourceU) | bufferBit(BufferId::SourceV),
      bufferBit(BufferId::MutualGate), true, false },
    { Stage::BuildBaseChroma, StageSupport::CPU, BoundaryMode::ScalarFallback,
      ScalarType::Float32, bufferBit(BufferId::SourceU) | bufferBit(BufferId::SourceV),
      bufferBit(BufferId::BaseU) | bufferBit(BufferId::BaseV), true, false },
    { Stage::BuildLGF, StageSupport::CPU, BoundaryMode::Clamp,
      ScalarType::Float32, bufferBit(BufferId::SourceY) |
          bufferBit(BufferId::SourceU) | bufferBit(BufferId::SourceV),
      bufferBit(BufferId::LGFMaps), true, false },
    { Stage::BuildAffineMaps, StageSupport::CPU, BoundaryMode::Clamp,
      ScalarType::Float32, bufferBit(BufferId::SourceY) |
          bufferBit(BufferId::SourceU) | bufferBit(BufferId::SourceV),
      bufferBit(BufferId::AffineMaps), true, false },
    { Stage::BuildDetail, StageSupport::CPU, BoundaryMode::ScalarFallback,
      ScalarType::Float32, bufferBit(BufferId::SourceY) |
          bufferBit(BufferId::AffineMaps), bufferBit(BufferId::Detail), true, false },
    { Stage::ApplyGuidedCorrection, StageSupport::CPU, BoundaryMode::ScalarFallback,
      ScalarType::Float32, bufferBit(BufferId::SourceU) | bufferBit(BufferId::SourceV) |
          bufferBit(BufferId::GuideTensor),
      bufferBit(BufferId::GuidedU) | bufferBit(BufferId::GuidedV), true, false },
    { Stage::ApplySelector, StageSupport::CPU, BoundaryMode::Clamp,
      ScalarType::Float32, bufferBit(BufferId::BaseU) | bufferBit(BufferId::BaseV) |
          bufferBit(BufferId::GuidedU) | bufferBit(BufferId::GuidedV) |
          bufferBit(BufferId::LGFMaps),
      bufferBit(BufferId::OutputU) | bufferBit(BufferId::OutputV), true, true },
    { Stage::ApplyDetailTransfer, StageSupport::CPU, BoundaryMode::Clamp,
      ScalarType::Float32, bufferBit(BufferId::BaseU) | bufferBit(BufferId::BaseV) |
          bufferBit(BufferId::AffineMaps) | bufferBit(BufferId::Detail),
      bufferBit(BufferId::OutputU) | bufferBit(BufferId::OutputV), true, true },
    { Stage::ApplyCollaborativeFilter, StageSupport::CPU, BoundaryMode::Clamp,
      ScalarType::Float32, bufferBit(BufferId::SourceY) |
          bufferBit(BufferId::OutputU) | bufferBit(BufferId::OutputV),
      bufferBit(BufferId::OutputU) | bufferBit(BufferId::OutputV), true, true },
    { Stage::DownsampleBase, StageSupport::CPU, BoundaryMode::Clamp,
      ScalarType::Float32, bufferBit(BufferId::SourceU) | bufferBit(BufferId::SourceV),
      bufferBit(BufferId::BaseU) | bufferBit(BufferId::BaseV), true, false },
    { Stage::DownsampleGuide, StageSupport::CPU, BoundaryMode::Clamp,
      ScalarType::Float32, bufferBit(BufferId::SourceY) |
          bufferBit(BufferId::SourceU) | bufferBit(BufferId::SourceV),
      bufferBit(BufferId::GuidedU) | bufferBit(BufferId::GuidedV), true, false },
    { Stage::DownsampleCandidateScore, StageSupport::CPU, BoundaryMode::Clamp,
      ScalarType::Float32, bufferBit(BufferId::SourceY) |
          bufferBit(BufferId::SourceU) | bufferBit(BufferId::SourceV) |
          bufferBit(BufferId::GuidedU) | bufferBit(BufferId::GuidedV),
      bufferBit(BufferId::OutputU) | bufferBit(BufferId::OutputV), true, false },
    { Stage::DownsampleOutput, StageSupport::CPU, BoundaryMode::Clamp,
      ScalarType::Float32, bufferBit(BufferId::BaseU) | bufferBit(BufferId::BaseV) |
          bufferBit(BufferId::GuidedU) | bufferBit(BufferId::GuidedV),
      bufferBit(BufferId::OutputFrame), true, false },
    { Stage::BackProject, StageSupport::CPU, BoundaryMode::Clamp,
      ScalarType::Float32, bufferBit(BufferId::SourceU) | bufferBit(BufferId::SourceV) |
          bufferBit(BufferId::OutputU) | bufferBit(BufferId::OutputV),
      bufferBit(BufferId::OutputU) | bufferBit(BufferId::OutputV), true, true },
    { Stage::ConvertOutput, StageSupport::CPU, BoundaryMode::Clamp,
      ScalarType::Float32, bufferBit(BufferId::OutputY) |
          bufferBit(BufferId::OutputU) | bufferBit(BufferId::OutputV),
      bufferBit(BufferId::OutputFrame), true, false },
}};

constexpr const StageContract &stageContract(Stage stage) {
    return stageContracts[static_cast<size_t>(stage)];
}

constexpr std::string_view stageName(Stage stage) {
    switch (stage) {
    case Stage::ConvertInput:          return "input_conversion";
    case Stage::ResampleLuma:          return "luma_resample";
    case Stage::BuildGuideMaps:        return "buildGuideMaps";
    case Stage::BuildTrustMask:        return "buildTrustMask";
    case Stage::BuildMutualGate:       return "buildMutualGate";
    case Stage::BuildBaseChroma:       return "plainChroma";
    case Stage::BuildLGF:              return "buildLGF";
    case Stage::BuildAffineMaps:       return "buildAffineMaps";
    case Stage::BuildDetail:           return "buildDetail";
    case Stage::ApplyGuidedCorrection: return "reconstructChroma";
    case Stage::ApplySelector:         return "selectorBlend";
    case Stage::ApplyDetailTransfer:   return "detailTransfer";
    case Stage::ApplyCollaborativeFilter: return "collaborative_chroma";
    case Stage::DownsampleBase:        return "downsample_base";
    case Stage::DownsampleGuide:       return "downsample_guide";
    case Stage::DownsampleCandidateScore: return "downsample_candidate_score";
    case Stage::DownsampleOutput:      return "downsample_output";
    case Stage::BackProject:           return "backProject";
    case Stage::ConvertOutput:         return "output_conversion";
    }
    return "unknown";
}

// CPU-only implementation detail. These slots are deliberately not stages:
// they may change without altering the backend-neutral pipeline contract.
enum class CpuProfileSlot : uint8_t {
    GuideSobel,
    GuideTensor,
    GuideLcMap,
    AffineCandidateBox,
    AffineCandidateBilinear,
    AffineCandidateBicubic,
    AffineMinMaxU,
    AffineMinMaxV,
    AffineWindowMoments,
    AffineFinalizeMedian,
    GuidedActiveRows,
    GuidedMetadata,
    GuidedTapAccumulation,
    GuidedNormalizationSelector,
    TrustSeed,
    TrustDilate,
    MutualGradients,
    MutualGate,
    PlainHorizontal,
    PlainVertical,
    LGFMoments,
    LGFFinalize,
    AffineRolling,
    AffineConsume,
    DetailReconstruct,
    DetailNyquist,
    DetailTransfer,
};

constexpr size_t cpuProfileSlotCount =
    static_cast<size_t>(CpuProfileSlot::DetailTransfer) + 1;

constexpr std::string_view cpuProfileSlotName(CpuProfileSlot slot) {
    switch (slot) {
    case CpuProfileSlot::GuideSobel:                  return "guide_sobel";
    case CpuProfileSlot::GuideTensor:                 return "guide_tensor";
    case CpuProfileSlot::GuideLcMap:                  return "guide_lcMap";
    case CpuProfileSlot::AffineCandidateBox:          return "affine_candidate_box";
    case CpuProfileSlot::AffineCandidateBilinear:     return "affine_candidate_bilinear";
    case CpuProfileSlot::AffineCandidateBicubic:      return "affine_candidate_bicubic";
    case CpuProfileSlot::AffineMinMaxU:               return "affine_minmax_u";
    case CpuProfileSlot::AffineMinMaxV:               return "affine_minmax_v";
    case CpuProfileSlot::AffineWindowMoments:         return "affine_window_moments";
    case CpuProfileSlot::AffineFinalizeMedian:        return "affine_finalize_median";
    case CpuProfileSlot::GuidedActiveRows:            return "guided_active_rows";
    case CpuProfileSlot::GuidedMetadata:              return "guided_metadata";
    case CpuProfileSlot::GuidedTapAccumulation:       return "guided_tap_accumulation";
    case CpuProfileSlot::GuidedNormalizationSelector: return "guided_normalization_selector";
    case CpuProfileSlot::TrustSeed:                   return "trust_seed";
    case CpuProfileSlot::TrustDilate:                 return "trust_dilate";
    case CpuProfileSlot::MutualGradients:             return "mutual_gradients";
    case CpuProfileSlot::MutualGate:                  return "mutual_gate";
    case CpuProfileSlot::PlainHorizontal:             return "plain_horizontal";
    case CpuProfileSlot::PlainVertical:               return "plain_vertical";
    case CpuProfileSlot::LGFMoments:                  return "lgf_moments";
    case CpuProfileSlot::LGFFinalize:                 return "lgf_finalize";
    case CpuProfileSlot::AffineRolling:               return "affine_rolling";
    case CpuProfileSlot::AffineConsume:               return "affine_consume";
    case CpuProfileSlot::DetailReconstruct:           return "detail_reconstruct";
    case CpuProfileSlot::DetailNyquist:               return "detail_nyquist";
    case CpuProfileSlot::DetailTransfer:              return "detail_transfer";
    }
    return "unknown";
}

struct PipelineMetrics {
    std::array<uint64_t, stageCount> nanoseconds{};
    std::array<uint64_t, cpuProfileSlotCount> cpuNanoseconds{};
    std::array<uint64_t, stageCount> pixels{};
    std::array<uint64_t, stageCount> taps{};
    uint64_t outputPixels = 0;
    uint64_t tapsVisited = 0;
    uint64_t sparseActivePixels = 0;
    uint64_t sparseTotalPixels = 0;

    void add(Stage stage, uint64_t ns) {
        nanoseconds[static_cast<size_t>(stage)] += ns;
    }
    void add(CpuProfileSlot slot, uint64_t ns) {
        cpuNanoseconds[static_cast<size_t>(slot)] += ns;
    }
    void addWork(Stage stage, uint64_t pixelCount, uint64_t tapCount = 0) {
        const size_t index = static_cast<size_t>(stage);
        pixels[index] += pixelCount;
        taps[index] += tapCount;
    }
};

class ScopedCpuTimer {
public:
    ScopedCpuTimer(PipelineMetrics *metrics, CpuProfileSlot slot) noexcept
        : metrics_(metrics), slot_(slot) {
        if (metrics_)
            start_ = Clock::now();
    }

    ~ScopedCpuTimer() {
        if (metrics_) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - start_).count();
            metrics_->add(slot_, static_cast<uint64_t>(elapsed));
        }
    }

    ScopedCpuTimer(const ScopedCpuTimer &) = delete;
    ScopedCpuTimer &operator=(const ScopedCpuTimer &) = delete;

private:
    using Clock = std::chrono::steady_clock;
    PipelineMetrics *metrics_ = nullptr;
    CpuProfileSlot slot_;
    Clock::time_point start_{};
};

class ScopedStageTimer {
public:
    ScopedStageTimer(PipelineMetrics *metrics, Stage stage) noexcept
        : metrics_(metrics), stage_(stage) {
        if (metrics_)
            start_ = Clock::now();
    }

    ~ScopedStageTimer() {
        if (metrics_) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - start_).count();
            metrics_->add(stage_, static_cast<uint64_t>(elapsed));
        }
    }

    ScopedStageTimer(const ScopedStageTimer &) = delete;
    ScopedStageTimer &operator=(const ScopedStageTimer &) = delete;

private:
    using Clock = std::chrono::steady_clock;
    PipelineMetrics *metrics_ = nullptr;
    Stage stage_;
    Clock::time_point start_{};
};

enum class StageStatus : uint8_t {
    Success,
    Unsupported,
    InvalidContext,
    BackendFailure,
};

using CpuStageFunction = StageStatus (*)(PipelineContext &, void *userData);

bool validPlaneView(const ConstPlaneView &view, MemoryDomain expectedDomain);
bool validPlaneView(const PlaneView &view, MemoryDomain expectedDomain);
StageStatus dispatchCpuStage(Stage stage, PipelineContext &context,
                             CpuStageFunction function, void *userData);

} // namespace lgcr
