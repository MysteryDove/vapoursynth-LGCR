#include "lgcr.h"

#include <limits>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace lgcr {

namespace {

size_t physicalMemoryQuarter() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status))
        return 0;
    const uint64_t bytes = static_cast<uint64_t>(status.ullTotalPhys);
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || pageSize <= 0)
        return 0;
    const uint64_t bytes = uint64_t(pages) * uint64_t(pageSize);
#endif
    constexpr uint64_t fourGiB = uint64_t{4} << 30;
    return static_cast<size_t>(std::min<uint64_t>(
        std::min(bytes / 4, fourGiB), std::numeric_limits<size_t>::max()));
}

} // namespace

std::shared_ptr<FrameWorkspacePool> makeFrameWorkspacePool() {
    return std::make_shared<FrameWorkspacePool>(physicalMemoryQuarter());
}

Plane FrameScratchAllocator::acquire(int width, int height) {
    const size_t required = size_t(width) * height;
    size_t sizeClass = 1;
    while (sizeClass < required)
        sizeClass <<= 1;
    Plane::Storage storage;
    auto found = idle_.find(sizeClass);
    if (found != idle_.end() && !found->second.empty()) {
        storage = std::move(found->second.back());
        found->second.pop_back();
        retainedBytes_ -= storage.capacity() * sizeof(float);
        if (found->second.empty())
            idle_.erase(found);
    } else {
        storage.reserve(sizeClass);
    }
    return Plane(std::move(storage), width, height, this, &recycleThunk);
}

void FrameScratchAllocator::recycle(Plane::Storage &&storage) {
    const size_t capacity = storage.capacity();
    storage.clear();
    retainedBytes_ += capacity * sizeof(float);
    idle_[capacity].push_back(std::move(storage));
}

bool FrameScratchAllocator::releaseLargest() {
    if (idle_.empty())
        return false;
    auto largest = std::prev(idle_.end());
    Plane::Storage storage = std::move(largest->second.back());
    largest->second.pop_back();
    retainedBytes_ -= storage.capacity() * sizeof(float);
    if (largest->second.empty())
        idle_.erase(largest);
    return true;
}

FrameWorkspacePool::FrameWorkspacePool(size_t idleBudgetBytes)
    : idleBudgetBytes_(idleBudgetBytes) {}

FrameWorkspaceLease FrameWorkspacePool::acquire(int workerCount) {
    std::unique_ptr<FrameWorkspace> workspace;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        workerLimit_ = std::max(1, workerCount);
        while (int(idle_.size()) > workerLimit_) {
            const auto largest = std::max_element(
                idle_.begin(), idle_.end(), [](const auto &left, const auto &right) {
                    return left->retainedBytes() < right->retainedBytes();
                });
            idleBytes_ -= (*largest)->retainedBytes();
            idle_.erase(largest);
        }
        if (!idle_.empty()) {
            workspace = std::move(idle_.back());
            idle_.pop_back();
            idleBytes_ -= workspace->retainedBytes();
        }
    }
    if (!workspace)
        workspace = std::make_unique<FrameWorkspace>();
    return FrameWorkspaceLease(shared_from_this(), std::move(workspace));
}

void FrameWorkspacePool::release(std::unique_ptr<FrameWorkspace> workspace) {
    if (!workspace)
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t workspaceBudget = idleBudgetBytes_ / size_t(workerLimit_);
    workspace->trimTo(workspaceBudget);
    const size_t bytes = workspace->retainedBytes();
    const bool withinBudget = bytes <= idleBudgetBytes_ &&
        idleBytes_ <= idleBudgetBytes_ - bytes;
    if (withinBudget && int(idle_.size()) < workerLimit_) {
        idleBytes_ += bytes;
        idle_.push_back(std::move(workspace));
    }
}

FrameWorkspaceLease::~FrameWorkspaceLease() {
    if (pool_ && workspace_)
        pool_->release(std::move(workspace_));
}

} // namespace lgcr
